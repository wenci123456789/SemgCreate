
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Joint-wise evaluation for calibrated EMGMamba models on NinaPro DB2.

- Supports multiple methods (FT / ATL / CDANR / CDANR++), multiple subjects.
- Loads the *best* checkpoint per method & subject (paths match your training scripts).
- Computes per-joint metrics: NRMSE, Pearson CC, R^2.
- Optionally applies Savitzky-Golay smoothing to predictions before metrics.
- Outputs per-subject Excel files and a combined CSV.

Author: generated to align with your pipeline.
"""

import os
import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

# Project modules
from DataProcess import NinaPro
from Model.EMGMambaAttentionAdapter import EMGMambaAdapter

# Metrics (use the same utilities as in your training for consistency)
try:
    from utils.Methods.methods import pearson_CC, compute_metrics_numpy
except Exception:
    # Fallbacks if the utilities are not available at import-time
    def pearson_CC(y_true, y_pred):
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        y_true = y_true - y_true.mean(axis=0, keepdims=True)
        y_pred = y_pred - y_pred.mean(axis=0, keepdims=True)
        num = (y_true * y_pred).sum(axis=0)
        den = np.sqrt((y_true ** 2).sum(axis=0) * (y_pred ** 2).sum(axis=0) + 1e-12)
        r = num / (den + 1e-12)
        return float(np.nanmean(r))
    def compute_metrics_numpy(y_true, y_pred):
        from skimage import metrics as skimetrics
        from sklearn.metrics import r2_score
        y_true = np.asarray(y_true).reshape(-1, 10)
        y_pred = np.asarray(y_pred).reshape(-1, 10)
        NRMSE = float(skimetrics.normalized_root_mse(y_true, y_pred, normalization="min-max"))
        CC = float(pearson_CC(y_true, y_pred))
        R2 = float(r2_score(y_true.T, y_pred.T, multioutput="variance_weighted"))
        return NRMSE, CC, R2

# Optional smoothing
try:
    from scipy.signal import savgol_filter
except Exception:
    savgol_filter = None


def load_state_flex(ckpt_path: str):
    """Load a checkpoint that may contain 'model_state' or a full state_dict."""
    state = torch.load(ckpt_path, map_location='cpu')
    if isinstance(state, dict) and 'model_state' in state:
        return state['model_state']
    if hasattr(state, 'state_dict'):
        return state.state_dict()
    return state


def get_default_ckpt(method: str, subject: str,
                     ckpt_ft_dir: str, ckpt_atl_dir: str, ckpt_cdanr_dir: str, ckpt_cdanrpp_dir: str) -> str:
    """Resolve default best checkpoint path per method/subject following training scripts."""
    if method.lower() == 'ft':
        return os.path.join(ckpt_ft_dir, f'ft_{subject}', 'ft_best.pth')
    if method.lower() == 'atl':
        return os.path.join(ckpt_atl_dir, f'atl_{subject}', 'atl_best.pth')
    if method.lower() == 'cdanr':
        return os.path.join(ckpt_cdanr_dir, f'cdanr_{subject}', 'cdanr_best.pth')
    if method.lower() in ['cdanrpp', 'cdanr++', 'cdanr_plus']:
        return os.path.join(ckpt_cdanrpp_dir, f'cdanrpp_{subject}', 'cdanrpp_best.pth')
    raise ValueError(f"Unknown method: {method}")


def load_test_loader(data_root: str, subject: str, subframe: int, normalization: str, miu: float, batch_size: int):
    emg_te = os.path.join(data_root, f"{subject}_E2_A1_rms_test.h5")
    glo_te = os.path.join(data_root, f"{subject}_E2_A1_glove_test.h5")
    for p in [emg_te, glo_te]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"[{subject}] Missing file: {p}")
    ds_te = NinaPro.NinaPro(emg_te, glo_te, subframe=subframe, normalization=normalization, mu=miu,
                            dummy_label=0, class_num=1)
    return DataLoader(ds_te, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)


def forward_all(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    preds, targets = [], []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            if isinstance(batch, (list, tuple)):
                x, y = batch[0], batch[1]
            elif isinstance(batch, dict):
                x = batch.get('x', batch.get('emg'))
                y = batch.get('y', batch.get('glove'))
            else:
                raise ValueError("Unsupported batch type.")
            # squeeze last dim if singleton
            if x.dim() == 4 and x.size(-1) == 1:
                x = x.squeeze(-1)
            x = x.to(device)
            y = y.to(device)
            y_hat = model(x)
            preds.append(y_hat.detach().cpu())
            targets.append(y.detach().cpu())
    yh = torch.cat(preds, dim=0).numpy().reshape(-1, 10)
    y  = torch.cat(targets, dim=0).numpy().reshape(-1, 10)
    return yh, y


def jointwise_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Return dict with per-joint NRMSE, CC, R2 and their means/stds."""
    from skimage import metrics as skimetrics
    from sklearn.metrics import r2_score
    n_j = y_true.shape[1]
    nrmses, ccs, r2s = [], [], []
    for j in range(n_j):
        yt = y_true[:, j]
        yp = y_pred[:, j]
        nrmse = float(skimetrics.normalized_root_mse(yt, yp, normalization="min-max"))
        # per-joint Pearson
        yt0, yp0 = yt - yt.mean(), yp - yp.mean()
        cc = float((yt0 * yp0).sum() / (np.sqrt((yt0**2).sum() * (yp0**2).sum()) + 1e-12))
        # sklearn's r2_score for 1D arrays returns a scalar
        from sklearn.metrics import r2_score as _r2
        r2 = float(_r2(yt, yp))
        nrmses.append(nrmse); ccs.append(cc); r2s.append(r2)
    out = {
        'NRMSE_per_joint': nrmses,
        'CC_per_joint': ccs,
        'R2_per_joint': r2s,
        'NRMSE_mean': float(np.mean(nrmses)),
        'NRMSE_std': float(np.std(nrmses, ddof=1)) if len(nrmses) > 1 else 0.0,
        'CC_mean': float(np.mean(ccs)),
        'CC_std': float(np.std(ccs, ddof=1)) if len(ccs) > 1 else 0.0,
        'R2_mean': float(np.mean(r2s)),
        'R2_std': float(np.std(r2s, ddof=1)) if len(r2s) > 1 else 0.0,
    }
    return out


def main():
    ap = argparse.ArgumentParser(description="Joint-wise evaluation for FT/ATL/CDANR/CDANR++ on NinaPro DB2")
    # Data
    ap.add_argument('--data_root', type=str, default='../../../feature/ninapro_db2_trans')
    ap.add_argument('--targets', nargs='+', default=[f"S{i}" for i in range(31, 41)])
    ap.add_argument('--batch_size', type=int, default=64)
    ap.add_argument('--subframe', type=int, default=200)
    ap.add_argument('--normalization', type=str, default='miu')
    ap.add_argument('--miu', type=float, default=2 ** 20)

    # Methods & checkpoints
    ap.add_argument('--methods', nargs='+', default=['ft', 'cdanrpp'],
                    help='Choose from ft, atl, cdanr, cdanrpp')
    ap.add_argument('--ckpt_ft_dir', type=str, default='../result/check/checkpoints_ft')
    ap.add_argument('--ckpt_atl_dir', type=str, default='../result/check/checkpoints_atl')
    ap.add_argument('--ckpt_cdanr_dir', type=str, default='../result/check/checkpoints_cdanr')
    ap.add_argument('--ckpt_cdanrpp_dir', type=str, default='../result/check/checkpoints_cdanrpp')

    # Optional smoothing
    ap.add_argument('--savgol_window', type=int, default=0, help='Set >0 to enable Savitzky-Golay smoothing (odd window)')
    ap.add_argument('--savgol_poly', type=int, default=2)

    # Device & output
    default_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ap.add_argument('--device', default=default_device)
    ap.add_argument('--out_dir', type=str, default='../result/check')

    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device(args.device)

    # Combined table in long-form for easy plotting later
    long_records = []

    for subject in args.targets:
        loader = load_test_loader(args.data_root, subject, args.subframe, args.normalization, args.miu, args.batch_size)

        for method in args.methods:
            try:
                ckpt_path = get_default_ckpt(method, subject,
                                             args.ckpt_ft_dir, args.ckpt_atl_dir, args.ckpt_cdanr_dir, args.ckpt_cdanrpp_dir)
            except ValueError:
                print(f"[Skip] Unknown method {method}")
                continue

            if not os.path.exists(ckpt_path):
                print(f"[{subject}][{method}] checkpoint not found: {ckpt_path} — skipping.")
                continue

            # Build model and load weights
            model = EMGMambaAdapter(input_dim=12, output_dim=10).to(device)
            state = load_state_flex(ckpt_path)
            # strict=False to be robust to head naming
            model.load_state_dict(state, strict=False)

            # Inference
            y_hat, y_true = forward_all(model, loader, device)

            # Optional smoothing
            if args.savgol_window and args.savgol_window > 0 and savgol_filter is not None:
                win = int(args.savgol_window) if args.savgol_window % 2 == 1 else int(args.savgol_window + 1)
                try:
                    y_hat = savgol_filter(y_hat, window_length=win, polyorder=args.savgol_poly, axis=0, mode='interp')
                except Exception as e:
                    print(f"[Warn] savgol_filter failed ({e}), continue without smoothing.")

            # Per-joint metrics
            jw = jointwise_metrics(y_true, y_hat)

            # Save per-subject Excel
            df = pd.DataFrame({
                'Joint': [f'Joint {i+1}' for i in range(10)],
                'NRMSE': jw['NRMSE_per_joint'],
                'CC': jw['CC_per_joint'],
                'R2': jw['R2_per_joint'],
            })
            df.loc[len(df)] = ['MEAN', jw['NRMSE_mean'], jw['CC_mean'], jw['R2_mean']]
            df.loc[len(df)] = ['STD', jw['NRMSE_std'], jw['CC_std'], jw['R2_std']]

            out_xlsx = os.path.join(args.out_dir, f'{subject}_{method}_joint_metrics.xlsx')
            try:
                df.to_excel(out_xlsx, index=False)
            except Exception as e:
                # If openpyxl not available, fall back to CSV
                out_xlsx = os.path.join(args.out_dir, f'{subject}_{method}_joint_metrics.csv')
                df.to_csv(out_xlsx, index=False)

            print(f"[{subject}][{method}] Saved joint metrics to: {out_xlsx}")

            # Append to long-form records
            for j in range(10):
                long_records.append({
                    'Subject': subject,
                    'Method': method,
                    'Joint': j+1,
                    'NRMSE': jw['NRMSE_per_joint'][j],
                    'CC': jw['CC_per_joint'][j],
                    'R2': jw['R2_per_joint'][j],
                })
            # Also add summary rows
            long_records.append({
                'Subject': subject, 'Method': method, 'Joint': 'MEAN',
                'NRMSE': jw['NRMSE_mean'], 'CC': jw['CC_mean'], 'R2': jw['R2_mean']
            })
            long_records.append({
                'Subject': subject, 'Method': method, 'Joint': 'STD',
                'NRMSE': jw['NRMSE_std'], 'CC': jw['CC_std'], 'R2': jw['R2_std']
            })

    if long_records:
        df_long = pd.DataFrame(long_records)
        out_csv = os.path.join(args.out_dir, 'jointwise_summary_long.csv')
        df_long.to_csv(out_csv, index=False)
        print(f"[Summary] Combined long-form CSV saved to: {out_csv}")
    else:
        print("[Summary] No records produced (check checkpoint paths & subjects).")


if __name__ == '__main__':
    main()
