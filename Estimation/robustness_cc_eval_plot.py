#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Robustness CC evaluation & bar plot (图2)
----------------------------------------
- Loads best checkpoints for a chosen method across target subjects.
- Evaluates CC on clean test data and under input perturbations:
  (a) additive Gaussian noise with std in --noise_stds
  (b) random channel drop with prob in --drop_probs
- Saves a CSV with per-subject results and a bar chart (mean ± std across subjects).
- Compatible with NinaPro DB2 pipeline (DataProcess.NinaPro, EMGMambaAdapter).
"""
import os, argparse, numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from DataProcess import NinaPro
from Model.EMGMambaAttentionAdapter import EMGMambaAdapter

def get_ckpt_path(method: str, subject: str, ckpt_dir: str) -> str:
    if method.lower() == 'ft':
        return os.path.join(ckpt_dir, f'ft_{subject}', 'ft_best.pth')
    if method.lower() == 'atl':
        return os.path.join(ckpt_dir, f'atl_{subject}', 'atl_best.pth')
    if method.lower() == 'cdanr':
        return os.path.join(ckpt_dir, f'cdanr_{subject}', 'cdanr_best.pth')
    if method.lower() in ['cdanrpp', 'cdanr++', 'cdanr_plus']:
        return os.path.join(ckpt_dir, f'cdanrpp_{subject}', 'cdanrpp_best.pth')
    if method.lower() in ['roformer', 'roformeremg']:
        return os.path.join(ckpt_dir, f'roformer_{subject}', 'roformer_best.pth')
    return os.path.join(ckpt_dir, f'{method}_{subject}', f'{method}_best.pth')

def load_state_flex(p: str):
    state = torch.load(p, map_location='cpu')
    if isinstance(state, dict) and 'model_state' in state:
        return state['model_state']
    if hasattr(state, 'state_dict'):
        return state.state_dict()
    return state

def load_test_loader(data_root: str, subject: str, subframe: int, normalization: str, miu: float, batch_size: int):
    emg_te = os.path.join(data_root, f"{subject}_E2_A1_rms_test.h5")
    glo_te = os.path.join(data_root, f"{subject}_E2_A1_glove_test.h5")
    ds_te = NinaPro.NinaPro(emg_te, glo_te, subframe=subframe, normalization=normalization, mu=miu,
                            dummy_label=0, class_num=1)
    return DataLoader(ds_te, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)

def cc_pearson(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    yt = y_true - y_true.mean(axis=0, keepdims=True)
    yp = y_pred - y_pred.mean(axis=0, keepdims=True)
    num = (yt * yp).sum(axis=0)
    den = np.sqrt((yt**2).sum(axis=0) * (yp**2).sum(axis=0) + 1e-12)
    r = num / (den + 1e-12)
    return float(np.nanmean(r))

def forward_all(model, loader, device, noise_std=0.0, drop_prob=0.0):
    preds, targets = [], []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            if isinstance(batch, (list, tuple)):
                x, y = batch[0], batch[1]
            else:
                x, y = batch['x'], batch['y']
            if x.dim() == 4 and x.size(-1) == 1:
                x = x.squeeze(-1)
            # perturbations
            if noise_std > 0.0:
                x = x + noise_std * torch.randn_like(x)
            if drop_prob > 0.0:
                B = x.size(0)
                if x.dim()==3 and x.size(1)==12:   # [B,C,T]
                    mask = (torch.rand(B, 12, device=x.device) > drop_prob).float().unsqueeze(-1)
                    x = x * mask
                else:                                # [B,T,C]
                    mask = (torch.rand(B, 12, device=x.device) > drop_prob).float().unsqueeze(1)
                    x = x * mask
            x = x.to(device); y = y.to(device)
            y_hat = model(x)
            preds.append(y_hat.detach().cpu())
            targets.append(y.detach().cpu())
    yh = torch.cat(preds, dim=0).numpy().reshape(-1, 10)
    yt = torch.cat(targets, dim=0).numpy().reshape(-1, 10)
    return yh, yt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--method', type=str, default='cdanrpp')
    ap.add_argument('--ckpt_dir', type=str, required=True)
    ap.add_argument('--targets', nargs='+', default=[f"S{i}" for i in range(31, 41)])
    ap.add_argument('--data_root', type=str, default='../../../feature/ninapro_db2_trans')
    ap.add_argument('--subframe', type=int, default=200)
    ap.add_argument('--batch_size', type=int, default=64)
    ap.add_argument('--normalization', type=str, default='miu')
    ap.add_argument('--miu', type=float, default=2**20)
    ap.add_argument('--noise_stds', nargs='+', type=float, default=[0.0, 0.01, 0.02, 0.03])
    ap.add_argument('--drop_probs', nargs='+', type=float, default=[0.0, 0.1, 0.2])
    ap.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--out_dir', type=str, default='/mnt/data/robust_eval')
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device)

    import csv
    csv_path = os.path.join(args.out_dir, f'robust_cc_{args.method}.csv')
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['Subject','Condition','CC'])
        for subject in args.targets:
            ckpt_path = get_ckpt_path(args.method, subject, args.ckpt_dir)
            if not os.path.exists(ckpt_path):
                print(f"[Skip][{subject}] missing ckpt: {ckpt_path}"); continue
            model = EMGMambaAdapter(input_dim=12, output_dim=10).to(device)
            state = load_state_flex(ckpt_path)
            model.load_state_dict(state, strict=False)
            loader = load_test_loader(args.data_root, subject, args.subframe, args.normalization, args.miu, args.batch_size)
            yh, yt = forward_all(model, loader, device, noise_std=0.0, drop_prob=0.0)
            w.writerow([subject, 'clean', f'{cc_pearson(yt, yh):.6f}'])
            for ns in args.noise_stds:
                if ns==0.0: continue
                yh, yt = forward_all(model, loader, device, noise_std=ns, drop_prob=0.0)
                w.writerow([subject, f'noise={ns}', f'{cc_pearson(yt, yh):.6f}'])
            for dp in args.drop_probs:
                if dp==0.0: continue
                yh, yt = forward_all(model, loader, device, noise_std=0.0, drop_prob=dp)
                w.writerow([subject, f'drop={dp}', f'{cc_pearson(yt, yh):.6f}'])

    import pandas as pd, numpy as np
    df = pd.read_csv(csv_path)
    conditions = df['Condition'].unique().tolist()
    conditions.sort(key=lambda x: (x!='clean', x))
    means, stds = [], []
    for cond in conditions:
        vals = df[df['Condition']==cond]['CC'].values.astype(float)
        means.append(vals.mean()); stds.append(vals.std(ddof=1) if len(vals)>1 else 0.0)

    plt.figure(figsize=(8,4.5))
    x = np.arange(len(conditions))
    plt.bar(x, means, yerr=stds, capsize=4)
    plt.xticks(x, conditions, rotation=30, ha='right')
    plt.ylabel('CC (↑)')
    plt.title(f'Robustness under Noise / Channel Drop — {args.method}')
    plt.tight_layout()
    fig_path = os.path.join(args.out_dir, f'robust_cc_bar_{args.method}.png')
    plt.savefig(fig_path, dpi=300)
    print(f"[Done] CSV: {csv_path}")
    print(f"[Done] Figure: {fig_path}")

if __name__ == '__main__':
    main()
