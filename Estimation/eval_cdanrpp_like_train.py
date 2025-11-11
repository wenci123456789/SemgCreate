
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Joint-wise evaluation for calibrated EMGMamba models on NinaPro DB2.

- Supports multiple methods (FT / ATL / CDANR / CDANR++), multiple subjects.
- Loads the *best* checkpoint per method & subject (paths match your training scripts).
- Computes per-joint metrics: NRMSE, Pearson CC, R^2.
- For **CDANR only**, adds:
  - EMA-first checkpoint loading,
  - Optional TTA (Gaussian jitter + averaging),
  - Optional moving-average smoothing,
  - "Train-like" metrics using compute_metrics_numpy (NRMSE/CC/R2).

Outputs per-subject Excel/CSV files and a combined CSV.
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
from utils.Methods.methods import compute_metrics_numpy

# ---------------- utils ----------------

def load_state_flex(ckpt_path: str, prefer_ema: bool = False):
    """Load a checkpoint that may contain EMA or plain 'model_state' / state_dict.
    If prefer_ema=True and EMA weights are present, return them first.
    """
    state = torch.load(ckpt_path, map_location='cpu')
    if isinstance(state, dict):
        if prefer_ema:
            for k in ['ema_state', 'model_ema', 'model_ema_state']:
                if k in state and isinstance(state[k], (dict, torch.nn.Module)):
                    return state[k] if isinstance(state[k], dict) else state[k].state_dict()
        # fallbacks
        if 'model_state' in state and isinstance(state['model_state'], dict):
            return state['model_state']
        if 'state_dict' in state and isinstance(state['state_dict'], dict):
            return state['state_dict']
    # fall back to obj.state_dict if available
    if hasattr(state, 'state_dict'):
        return state.state_dict()
    return state


def get_default_ckpt(method: str, subject: str, model: str) -> str:
    base_dir = '../result/ninapro/Estimation_result'
    noft_dir = '../result/ninapro/checkpoints_pretrain'
    m = method.lower()
    if m == 'no':
        return os.path.join(noft_dir, f'{model}', 'model_best.pth')
    if m == 'ft':
        return os.path.join(base_dir, f'{model}', 'checkpoints_ft', f'ft_{subject}', 'ft_best.pth')
    if m == 'atl':
        return os.path.join(base_dir, f'{model}', 'checkpoints_atl', f'atl_{subject}', 'atl_best.pth')
    if m in ['cdanr', 'cdan']:
        # CDAN-R (not R++) expected path
        return os.path.join(base_dir, f'{model}', 'checkpoints_cdanr', f'cdanr_{subject}', 'cdanr_best.pth')
    if m in ['cdanrpp', 'cdanr++', 'cdanr_plus']:
        return os.path.join(base_dir, f'{model}', 'checkpoints_cdanrpp', f'cdanrpp_{subject}', 'cdanrpp_best.pth')
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


@torch.no_grad()
def forward_all(model: torch.nn.Module, loader: DataLoader, device: torch.device):
    """Plain forward without TTA; returns (y_hat, y_true) as numpy arrays [N,10]."""
    preds, targets = [], []
    model.eval()
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


@torch.no_grad()
def forward_all_cdanr_tta(model: torch.nn.Module, loader: DataLoader, device: torch.device,
                           tta: bool, tta_times: int, tta_noise_std: float, smooth_win: int):
    """CDANR-only inference with optional TTA and moving-average smoothing."""
    preds, targets = [], []
    model.eval()

    def _gaussian_jitter(x, std):
        if std <= 0:
            return x
        return x + torch.randn_like(x) * std

    def _moving_average(arr, win):
        # arr: [N,10]; apply MA separately per joint along time N
        if win <= 1:
            return arr
        out = np.empty_like(arr)
        for j in range(arr.shape[1]):
            x = arr[:, j]
            if win > len(x):
                out[:, j] = x  # too short, skip
                continue
            c = np.convolve(x, np.ones(win)/win, mode='same')
            out[:, j] = c
        return out

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

        if tta:
            yh_sum = 0.0
            for _ in range(tta_times):
                xb = _gaussian_jitter(x, tta_noise_std) if tta_noise_std > 0 else x
                yh = model(xb)
                yh_sum = yh_sum + yh
            y_hat = yh_sum / float(max(1, tta_times))
        else:
            y_hat = model(x)

        preds.append(y_hat.detach().cpu())
        targets.append(y.detach().cpu())

    yh = torch.cat(preds, dim=0).numpy().reshape(-1, 10)
    y  = torch.cat(targets, dim=0).numpy().reshape(-1, 10)

    if smooth_win and smooth_win > 1:
        yh = _moving_average(yh, smooth_win)

    return yh, y


def jointwise_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Per-joint NRMSE (min-max), Pearson CC, R2, with mean/std across joints."""
    from skimage import metrics as skimetrics
    from sklearn.metrics import r2_score as _r2
    n_j = y_true.shape[1]
    nrmses, ccs, r2s = [], [], []
    for j in range(n_j):
        yt = y_true[:, j]
        yp = y_pred[:, j]
        nrmse = float(skimetrics.normalized_root_mse(yt, yp, normalization="min-max"))
        # Pearson
        yt0, yp0 = yt - yt.mean(), yp - yp.mean()
        cc = float((yt0 * yp0).sum() / (np.sqrt((yt0**2).sum() * (yp0**2).sum()) + 1e-12))
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
    ap.add_argument('--model', type=str, default='sEMGMamba') # sEMGMamba、BERT
    ap.add_argument('--data_root', type=str, default='../../../feature/ninapro_db2_trans')
    ap.add_argument('--targets', nargs='+', default=[f"S{i}" for i in range(31, 41)])
    ap.add_argument('--batch_size', type=int, default=32)
    ap.add_argument('--subframe', type=int, default=200)
    ap.add_argument('--normalization', type=str, default='miu')
    ap.add_argument('--miu', type=float, default=2 ** 20)

    # Methods & checkpoints
    ap.add_argument('--methods', nargs='+', default=['no','ft','atl','cdanrpp'],
                    help='Choose from no, ft, atl, cdanr, cdanrpp')

    # CDANR-only options
    ap.add_argument('--cdanr_prefer_ema', action='store_true', help='Load EMA weights if present (CDANR only)')
    ap.add_argument('--cdanr_tta', action='store_true', help='Enable TTA for CDANR (Gaussian jitter + average)')
    ap.add_argument('--cdanr_tta_times', type=int, default=8)
    ap.add_argument('--cdanr_tta_noise_std', type=float, default=0.015)
    ap.add_argument('--cdanr_smooth_win', type=int, default=0, help='Moving-average window (0=off)')
    ap.add_argument('--cdanr_print_train_like', action='store_true',
                    help='Also print training-like metrics via compute_metrics_numpy for CDANR')

    # Device & output
    default_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ap.add_argument('--device', default=default_device)
    ap.add_argument('--out_dir', type=str, default='/mnt/data_nvme/zwc/semg-code/resultFinal/ninapro')

    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device(args.device)

    # Combined table in long-form for easy plotting later
    long_records = []

    for subject in args.targets:
        loader = load_test_loader(args.data_root, subject, args.subframe, args.normalization, args.miu, args.batch_size)

        for method in args.methods:
            try:
                ckpt_path = get_default_ckpt(method, subject, args.model)
            except ValueError:
                print(f"[Skip] Unknown method {method}")
                continue

            if not os.path.exists(ckpt_path):
                print(f"[{subject}][{method}] checkpoint not found: {ckpt_path} — skipping.")
                continue

            # Build model and load weights
            model = EMGMambaAdapter(input_dim=12, output_dim=10).to(device)
            # For CDANR only, prefer EMA if requested
            prefer_ema = args.cdanr_prefer_ema and (method.lower() in ['cdanr','cdan'])
            state = load_state_flex(ckpt_path, prefer_ema=prefer_ema)
            model.load_state_dict(state, strict=False)

            # Inference
            if method.lower() in ['cdanr','cdan']:
                y_hat, y_true = forward_all_cdanr_tta(
                    model, loader, device,
                    tta=args.cdanr_tta,
                    tta_times=args.cdanr_tta_times,
                    tta_noise_std=args.cdanr_tta_noise_std,
                    smooth_win=args.cdanr_smooth_win,
                )
            else:
                y_hat, y_true = forward_all(model, loader, device)

            # Per-joint metrics (always compute & save)
            jw = jointwise_metrics(y_true, y_hat)

            # Save per-subject Excel/CSV
            df = pd.DataFrame({
                'Joint': [f'Joint {i+1}' for i in range(10)],
                'NRMSE': jw['NRMSE_per_joint'],
                'CC': jw['CC_per_joint'],
                'R2': jw['R2_per_joint'],
            })
            df.loc[len(df)] = ['MEAN', jw['NRMSE_mean'], jw['CC_mean'], jw['R2_mean']]
            df.loc[len(df)] = ['STD', jw['NRMSE_std'], jw['CC_std'], jw['R2_std']]

            out_path = os.path.join(args.out_dir, f'{args.model}', f'{subject}_{method}_joint_metrics.xlsx')
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            try:
                df.to_excel(out_path, index=False)
            except Exception:
                out_path = os.path.join(args.out_dir, f'{args.model}', f'{subject}_{method}_joint_metrics.csv')
                df.to_csv(out_path, index=False)
            print(f"[{subject}][{method}] Saved joint metrics to: {out_path}")

            # Optionally print training-like metrics for CDANR
            if method.lower() in ['cdanr','cdan'] and args.cdanr_print_train_like:
                try:
                    NRMSE, CC, R2 = compute_metrics_numpy(y_true, y_hat)
                    print(f"[{subject}][CDANR] train-like metrics — NRMSE={NRMSE:.4f} CC={CC:.4f} R2={R2:.4f} "
                          f"(EMA={prefer_ema}, TTA={int(args.cdanr_tta)}x{args.cdanr_tta_times}@{args.cdanr_tta_noise_std}, "
                          f"SmoothWin={args.cdanr_smooth_win})")
                except Exception as e:
                    print(f"[{subject}][CDANR] train-like metric computation failed: {e}")

            # Append to long-form table
            for j in range(10):
                long_records.append({
                    'Subject': subject, 'Method': method, 'Joint': j+1,
                    'NRMSE': jw['NRMSE_per_joint'][j], 'CC': jw['CC_per_joint'][j], 'R2': jw['R2_per_joint'][j]
                })
            long_records.append({'Subject': subject, 'Method': method, 'Joint': 'MEAN',
                                 'NRMSE': jw['NRMSE_mean'], 'CC': jw['CC_mean'], 'R2': jw['R2_mean']})
            long_records.append({'Subject': subject, 'Method': method, 'Joint': 'STD',
                                 'NRMSE': jw['NRMSE_std'], 'CC': jw['CC_std'], 'R2': jw['R2_std']})

    # Save combined CSV
    long_df = pd.DataFrame(long_records)
    combined_out = os.path.join(args.out_dir, f'{args.model}', 'combined_joint_metrics.csv')
    os.makedirs(os.path.dirname(combined_out), exist_ok=True)
    long_df.to_csv(combined_out, index=False)
    print(f"[ALL] Saved combined metrics to: {combined_out}")


if __name__ == '__main__':
    main()
