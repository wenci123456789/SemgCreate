#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
按被试单独评估预训练好的 sEMG_LSTM 模型 (.pth)

- 对 args.subjects 里面的每一个 subject 单独构建 DataLoader
- 分别计算每个 subject 的 Val Loss / NRMSE / CC / R²
"""

import argparse
import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from DataProcess import NinaPro
from sklearn import metrics as skmetrics
from skimage import metrics

from utils.sEMG_models.sEMG_LSTM import sEMG_LSTM


# ========= 工具函数 =========
def seed_everything(seed=525):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def pearson_CC(x, y):
    x, y = x.flatten(), y.flatten()
    vx, vy = x - x.mean(), y - y.mean()
    cc = np.sum(vx * vy) / (np.sqrt(np.sum(vx ** 2)) * np.sqrt(np.sum(vy ** 2)) + 1e-8)
    return cc


def build_single_subject_loader(sid, args):
    """
    为单个 subject 构建 DataLoader（使用 *_rms_test.h5 / *_glove_test.h5）
    """
    emg_te = os.path.join(args.data_root, f"{sid}_E2_A1_rms_test.h5")
    glove_te = os.path.join(args.data_root, f"{sid}_E2_A1_glove_test.h5")

    if not os.path.exists(emg_te):
        raise FileNotFoundError(f"测试数据不存在: {emg_te}")

    dataset = NinaPro.NinaPro(
        emg_te,
        glove_te,
        subframe=args.subframe,
        normalization=args.normalization,
        mu=args.miu,
        dummy_label=0,
        class_num=1
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )

    return loader


def evaluate(model, loader, device):
    """
    对单个 DataLoader（单一 subject）进行评估
    """
    reg_loss = nn.MSELoss()
    model.eval()
    val_loss = 0.0
    preds, trues = [], []

    with torch.no_grad():
        hidden = None
        for x, y, *_ in loader:
            # x: 预期 [B,200,12,1] 或 [B,12,200,1] 或 [B,200,12]
            x = x.to(device).float()

            # 如果是 [B,200,12]，补成 [B,200,12,1]
            if x.dim() == 3:
                x = x.unsqueeze(-1).contiguous()

            y = y.to(device, non_blocking=True).float()

            # LSTM 前向
            pred, hidden = model(x)
            if isinstance(hidden, (tuple, list)):  # (h_n, c_n)
                hidden = tuple(h.detach() for h in hidden)
            else:
                hidden = hidden.detach()

            # 沿时间维做平均，得到一个回归输出 [B,1,10]
            pred = pred.mean(dim=1, keepdim=True)

            loss = reg_loss(pred, y)
            val_loss += loss.item()

            preds.append(pred.cpu().numpy())
            trues.append(y.cpu().numpy())

    preds = np.concatenate(preds, axis=0)[:, 0, :]   # [N, 10]
    trues = np.concatenate(trues, axis=0)[:, 0, :]   # [N, 10]

    nrmse = metrics.normalized_root_mse(trues, preds)
    cc = pearson_CC(trues, preds)
    r2 = skmetrics.r2_score(trues, preds)
    avg_val = val_loss / len(loader)

    return avg_val, nrmse, cc, r2


def main():
    parser = argparse.ArgumentParser(description="Per-subject test for pretrained sEMG_LSTM model (.pth)")
    parser.add_argument('--subjects', nargs='+', default=[f"S{i}" for i in range(31, 32)],
                        help="要评估的 subject 列表，例如 S1 S2 S3（默认 S1-S30）")
    parser.add_argument('--data_root', type=str,
                        default='../../../../feature/ninapro_db2_trans',
                        help="Ninapro 特征数据根目录")
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--subframe', type=int, default=200)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--normalization', type=str, default='miu')
    parser.add_argument('--miu', type=int, default=2 ** 20)
    parser.add_argument('--ckpt', type=str, default='../../result/ninapro/Estimation_result/LSTM/checkpoints_cdanrpp/cdanrpp_S31/cdanrpp_best.pth',
                        help="预训练权重路径 (.pth)，例如 ../result/ninapro/checkpoints_pretrain/LSTM/model_best.pth")
    args = parser.parse_args()

    seed_everything(525)

    device = args.device

    # ====== 构建模型并加载权重（与预训练脚本保持一致） ======
    model = sEMG_LSTM(vocab_size=200, hidden=128, n_layers=4).to(device)

    print(f"🔄 Loading checkpoint from: {args.ckpt}")
    state = torch.load(args.ckpt, map_location=device)
    if isinstance(state, dict) and 'model_state' in state:
        model.load_state_dict(state['model_state'], strict=True)
    else:
        model.load_state_dict(state, strict=True)

    # ====== 逐个 subject 评估 ======
    print("\n========== PER-SUBJECT TEST (LSTM) ==========")
    all_results = {}

    for sid in args.subjects:
        try:
            print(f"\n----- Subject: {sid} -----")
            loader = build_single_subject_loader(sid, args)
            avg_val, nrmse, cc, r2 = evaluate(model, loader, device)

            all_results[sid] = (avg_val, nrmse, cc, r2)

            print(f"{sid} | Val Loss = {avg_val:.5f} | NRMSE = {nrmse:.4f} | "
                  f"CC = {cc:.4f} | R² = {r2:.4f}")
        except FileNotFoundError as e:
            print(f"[WARN] {e} (skip {sid})")
        except Exception as e:
            print(f"[ERROR] {sid} 评估出错: {e}")

    # ====== 汇总打印 ======
    print("\n========== SUMMARY (LSTM) ==========")
    for sid, (avg_val, nrmse, cc, r2) in all_results.items():
        print(f"{sid}: Val={avg_val:.5f}, NRMSE={nrmse:.4f}, CC={cc:.4f}, R²={r2:.4f}")
    print("====================================")


if __name__ == '__main__':
    main()