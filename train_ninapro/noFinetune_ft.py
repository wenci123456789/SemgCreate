#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Inference-only script (no fine-tuning).
- Loads a pretrained EMGMambaAdapter
- Runs forward pass on validation/test split
- Computes NRMSE / Pearson CC / R^2
- Robust to datasets that return extra items per batch (e.g., (x, y, meta...))
"""

import os
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader

# === project modules ===
from DataProcess import NinaPro
from Model.EMGMambaAttentionAdapter import EMGMambaAdapter

# Try to import the same Pearson CC function used in pretraining
try:
    from utils.Methods.methods import pearson_CC
except Exception:
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
    """
    Same metrics as pretraining:
      - NRMSE (min-max normalization)
      - Pearson CC
      - R^2 (variance_weighted)
    Expects arrays shaped [N, 10] (will reshape if needed).
    """
    from skimage import metrics as skimetrics
    from sklearn.metrics import r2_score

    y_true = np.asarray(y_true).reshape(-1, 10)
    y_pred = np.asarray(y_pred).reshape(-1, 10)

    NRMSE = float(skimetrics.normalized_root_mse(y_true, y_pred, normalization="min-max"))
    CC = float(pearson_CC(y_true, y_pred))
    R2 = float(r2_score(y_true.T, y_pred.T, multioutput="variance_weighted"))
    return NRMSE, CC, R2

def _squeeze_feat(x: torch.Tensor) -> torch.Tensor:
    # In many pipelines x is [B, T, C, 1]; squeeze the last dim if it's singleton.
    if x.dim() == 4 and x.size(-1) == 1:
        return x.squeeze(-1)  # becomes [B, T, C]
    return x

def run_inference_without_finetuning(args, device, target_id: str):
    # Paths
    emg_te = os.path.join(args.data_root, f"{target_id}_E2_A1_rms_test.h5")
    glo_te = os.path.join(args.data_root, f"{target_id}_E2_A1_glove_test.h5")
    for p in [emg_te, glo_te]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"[{target_id}] Missing file: {p}")

    # Dataset / Loader
    ds_te = NinaPro.NinaPro(
        emg_te, glo_te,
        subframe=args.subframe,
        normalization=args.normalization,
        mu=args.miu,
        dummy_label=0,      # important: avoid IndexError inside dataset
        class_num=1
    )
    ValLoader = DataLoader(ds_te, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=False)

    # Model
    if not os.path.exists(args.pretrained):
        raise FileNotFoundError(f"Pretrained checkpoint not found: {args.pretrained}")
    state = torch.load(args.pretrained, map_location='cpu')
    state = state.get('model_state', state)

    model = EMGMambaAdapter(input_dim=12, output_dim=10).to(device)
    model.load_state_dict(state, strict=False)  # allow missing keys if any
    model.eval()

    preds_cpu, targets_cpu = [], []
    with torch.no_grad():
        for batch in ValLoader:
            # Robustly extract (x, y) from batch that may include extra items
            if isinstance(batch, (list, tuple)):
                if len(batch) < 2:
                    raise ValueError("Dataset batch should return at least (x, y).")
                x, y = batch[0], batch[1]
            else:
                # Some datasets return a dict
                if isinstance(batch, dict):
                    # common keys guess
                    x = batch.get('x', batch.get('emg'))
                    y = batch.get('y', batch.get('glove'))
                    if x is None or y is None:
                        raise ValueError("Dict batch must contain 'x'/'y' (or 'emg'/'glove').")
                else:
                    raise ValueError("Unsupported batch type; expected tuple/list or dict.")

            x = _squeeze_feat(x).to(device)  # [B, T, C]
            y = y.to(device)                  # [B, 1, 10]

            y_hat = model(x)                  # [B, 1, 10]
            preds_cpu.append(y_hat.detach().cpu())
            targets_cpu.append(y.detach().cpu())

    yh_np = torch.cat(preds_cpu, dim=0).numpy().reshape(-1, 10)
    y_np  = torch.cat(targets_cpu, dim=0).numpy().reshape(-1, 10)
    NRMSE, CC, R2 = compute_metrics_numpy(y_np, yh_np)

    print(f"[{target_id}] Validation  NRMSE: {NRMSE:.4f} | CC: {CC:.4f} | R^2: {R2:.4f}")

def main():
    ap = argparse.ArgumentParser(description="Inference without fine-tuning")
    ap.add_argument('--data_root', type=str, default='../../../feature/ninapro_db2_trans')
    ap.add_argument('--pretrained', type=str, default='../result/ninapro/checkpoints_pretrain/sEMGMamba/model_best.pth')
    ap.add_argument('--targets', nargs='+', default=[f"S{i}" for i in range(31, 41)])
    ap.add_argument('--batch_size', type=int, default=64)
    ap.add_argument('--subframe', type=int, default=200)
    ap.add_argument('--normalization', type=str, default='miu')
    ap.add_argument('--miu', type=int, default=2 ** 20)
    # Auto-pick device if not set explicitly
    default_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ap.add_argument('--device', default=default_device)

    args = ap.parse_args()
    device = torch.device(args.device)

    print(f"[*]Cur normalization type is: Mu-normalization with miu={args.miu}")
    print(f"Using device: {device} (CUDA available={torch.cuda.is_available()})")
    print(f"Pretrained: {args.pretrained}")

    for tgt in args.targets:
        print(f"\n====== Inference start: {tgt} ======")
        run_inference_without_finetuning(args, device, tgt)
        print(f"====== Inference done : {tgt} ======\n")

if __name__ == '__main__':
    main()
