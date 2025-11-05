#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, argparse, numpy as np, torch
from torch.utils.data import DataLoader

from DataProcess import NinaPro
from utils.sEMG_models.sEMG_BERT import sEMG_BERT  # ← BERT

try:
    from utils.Methods.methods import pearson_CC
except Exception:
    def pearson_CC(y_true, y_pred):
        y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
        y_true = y_true - y_true.mean(axis=0, keepdims=True)
        y_pred = y_pred - y_pred.mean(axis=0, keepdims=True)
        num = (y_true * y_pred).sum(axis=0)
        den = np.sqrt((y_true ** 2).sum(axis=0) * (y_pred ** 2).sum(axis=0) + 1e-12)
        return float(np.nanmean(num / (den + 1e-12)))

def compute_metrics_numpy(y_true, y_pred):
    from skimage import metrics as skimetrics
    from sklearn.metrics import r2_score
    y_true = np.asarray(y_true).reshape(-1, 10)
    y_pred = np.asarray(y_pred).reshape(-1, 10)
    NRMSE = float(skimetrics.normalized_root_mse(y_true, y_pred, normalization="min-max"))
    CC = float(pearson_CC(y_true, y_pred))
    R2 = float(r2_score(y_true.T, y_pred.T, multioutput="variance_weighted"))
    return NRMSE, CC, R2

def _squeeze_feat(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 4 and x.size(-1) == 1: return x.squeeze(-1)
    return x

def run_inference_without_finetuning(args, device, target_id: str):
    emg_te = os.path.join(args.data_root, f"{target_id}_E2_A1_rms_test.h5")
    glo_te = os.path.join(args.data_root, f"{target_id}_E2_A1_glove_test.h5")
    for p in [emg_te, glo_te]:
        if not os.path.exists(p): raise FileNotFoundError(f"[{target_id}] Missing file: {p}")

    ds_te = NinaPro.NinaPro(emg_te, glo_te, subframe=args.subframe, normalization=args.normalization, mu=args.miu, dummy_label=0, class_num=1)
    ValLoader = DataLoader(ds_te, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=False)

    # 加载预训练模型的权重
    if not os.path.exists(args.pretrained): raise FileNotFoundError(f"Pretrained checkpoint not found: {args.pretrained}")
    state = torch.load(args.pretrained, map_location='cpu'); state = state.get('model_state', state)

    # 获取预训练模型的超参数
    pos = state.get('bert.embedding.position.position_embedding', None)
    if pos is None:
        raise RuntimeError("Checkpoint 缺少 position embedding，无法推断超参。")
    T_ckpt, hidden_ckpt = int(pos.shape[1]), int(pos.shape[2])

    # 从预训练模型获取 token embedding 的权重，计算 C_ckpt 和 vocab_size
    tok_w = state.get('bert.embedding.token.embedding.weight', None)
    if tok_w is not None:
        vocab_size_from_ckpt = tok_w.shape[1]  # 使用预训练模型的 vocab_size
        C_ckpt = int(tok_w.shape[1] // T_ckpt)   # tok_w: [T*hidden, T*C]
    else:
        vocab_size_from_ckpt = 2400  # 默认值
        C_ckpt = 12  # 默认值

    # 构造模型，确保使用预训练模型的一致配置
    model = sEMG_BERT(vocab_size=T_ckpt, hidden=hidden_ckpt,feature_dim=1, n_layers=4, attn_heads=8).to(device)
    # model = sEMG_BERT(vocab_size=200, hidden=128,  n_layers=args.num_layers,
    #                   attn_heads=8, use_se=args.use_se).to(device)
    # # 加载预训练权重
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print("[load_state] missing:", missing)
        print("[load_state] unexpected:", unexpected)
    model.eval()

    preds_cpu, targets_cpu = [], []
    with torch.no_grad():
        for batch in ValLoader:
            if isinstance(batch, (list, tuple)):
                x, y = batch[0], batch[1]
            elif isinstance(batch, dict):
                x = batch.get('x', batch.get('emg')); y = batch.get('y', batch.get('glove'))
                if x is None or y is None: raise ValueError("Dict batch must contain 'x'/'y' (or 'emg'/'glove').")
            else:
                raise ValueError("Unsupported batch type.")

            x = _squeeze_feat(x).to(device).float()
            y = y.to(device).float()
            out = model(x)                 # (output, distr)
            y_hat = out[0] if isinstance(out, (tuple, list)) else out
            preds_cpu.append(y_hat.detach().cpu()); targets_cpu.append(y.detach().cpu())

    yh_np = torch.cat(preds_cpu, dim=0).numpy().reshape(-1, 10)
    y_np  = torch.cat(targets_cpu, dim=0).numpy().reshape(-1, 10)
    NRMSE, CC, R2 = compute_metrics_numpy(y_np, yh_np)
    print(f"[{target_id}] Validation  NRMSE: {NRMSE:.4f} | CC: {CC:.4f} | R^2: {R2:.4f}")

def main():
    ap = argparse.ArgumentParser(description="Inference without finetuning — BERT")
    ap.add_argument('--data_root', type=str, default='../../../../feature/ninapro_db2_trans')
    ap.add_argument('--pretrained', type=str, default='../../result/ninapro/checkpoints_pretrain/BERT/model_best.pth')
    ap.add_argument('--targets', nargs='+', default=[f"S{i}" for i in range(31, 41)])
    ap.add_argument('--batch_size', type=int, default=32)  # 减小 batch_size
    ap.add_argument('--subframe', type=int, default=200)
    ap.add_argument('--normalization', type=str, default='miu')
    ap.add_argument('--miu', type=int, default=2 ** 20)
    ap.add_argument('--device', default=('cuda' if torch.cuda.is_available() else 'cpu'))
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
