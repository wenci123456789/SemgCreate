#!/usr/bin/env python3
"""
测试预训练好的 sEMG_LSTM 模型 (.pth)
- 默认使用 Ninapro DB2 的 *_rms_test.h5 / *_glove_test.h5 做评估
"""

import argparse, os, math, random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset

from DataProcess import NinaPro
from sklearn import metrics as skmetrics
from skimage import metrics

from utils.sEMG_models.sEMG_LSTM import sEMG_LSTM


# ====== 复用的工具函数（和预训练脚本一致） ======
def seed_everything(seed=525):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def pearson_CC(x, y):
    x, y = x.flatten(), y.flatten()
    vx, vy = x - x.mean(), y - y.mean()
    cc = np.sum(vx * vy) / (np.sqrt(np.sum(vx ** 2)) * np.sqrt(np.sum(vy ** 2)) + 1e-8)
    return cc


def build_test_loader(args):
    """
    默认用 *_rms_test.h5 / *_glove_test.h5，跨被试测试
    """
    test_sets = []
    for sid in args.subjects:
        emg_te = os.path.join(args.data_root, f"{sid}_E2_A1_rms_test.h5")
        glove_te = os.path.join(args.data_root, f"{sid}_E2_A1_glove_test.h5")

        if not os.path.exists(emg_te):
            print(f"[WARN] Missing {emg_te}, skip {sid}")
            continue

        test_sets.append(NinaPro.NinaPro(
            emg_te, glove_te, subframe=args.subframe,
            normalization=args.normalization, mu=args.miu,
            dummy_label=0, class_num=1))

    if len(test_sets) == 0:
        raise RuntimeError("没有找到任何测试数据，请检查 data_root 和 subjects 参数")

    TestLoader = DataLoader(
        ConcatDataset(test_sets),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )
    print(f"✅ Loaded {len(test_sets)} subjects for TEST")
    return TestLoader


def evaluate(model, loader, device):
    reg_loss = nn.MSELoss()
    model.eval()
    val_loss = 0.0
    preds, trues = [], []

    with torch.no_grad():
        hidden = None
        for x, y, *_ in loader:
            print("y range:", y.min().item(), y.max().item())
            # x: 预期 [B,200,12,1] 或 [B,12,200,1] 或 [B,200,12]
            x = x.to(device).float()

            # 按照你预训练脚本中的写法稍作兼容处理
            if x.dim() == 3:   # [B,200,12] -> [B,200,12,1]
                x = x.unsqueeze(-1).contiguous()

            y = y.to(device, non_blocking=True).float()

            pred, hidden = model(x)
            if isinstance(hidden, (tuple, list)):  # LSTM: (h_n, c_n)
                hidden = tuple(h.detach() for h in hidden)
            else:
                hidden = hidden.detach()

            # 对时间维求平均，得到一个全局回归输出
            pred = pred.mean(dim=1, keepdim=True)
            print("pred range:", pred.min().item(), pred.max().item())
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
    parser = argparse.ArgumentParser(description="Test pretrained sEMG_LSTM model(.pth)")
    parser.add_argument('--subjects', nargs='+', default=[f"S{i}" for i in range(31, 32)],
                        help="哪些被试参与测试（默认 S1-S30）")
    parser.add_argument('--data_root', type=str,
                        default='../../../../feature/ninapro_db2_trans',
                        help="Ninapro 特征数据根目录")
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--subframe', type=int, default=200)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--normalization', type=str, default='miu')
    parser.add_argument('--miu', type=int, default=2 ** 20)
    parser.add_argument('--ckpt', type=str, default='../../result/ninapro/checkpoints_pretrain/LSTM/model_best.pth',
                        help="预训练权重路径 (.pth)，例如 ../result/ninapro/checkpoints_pretrain/LSTM/model_best.pth")
    args = parser.parse_args()

    seed_everything(525)

    # ====== 构建 DataLoader ======
    TestLoader = build_test_loader(args)

    # ====== 构建模型并加载权重（参数与预训练脚本保持一致） ======
    device = args.device
    model = sEMG_LSTM(vocab_size=200, hidden=128, n_layers=4).to(device)

    print(f"🔄 Loading checkpoint from: {args.ckpt}")
    state = torch.load(args.ckpt, map_location=device)
    if 'model_state' in state:
        model.load_state_dict(state['model_state'], strict=False)
    else:
        # 兼容直接保存 state_dict 的情况
        model.load_state_dict(state, strict=False)

    # ====== 评估 ======
    avg_val, nrmse, cc, r2 = evaluate(model, TestLoader, device)

    print("\n===== TEST RESULTS (LSTM) =====")
    print(f"Val Loss = {avg_val:.5f}")
    print(f"NRMSE    = {nrmse:.4f}")
    print(f"CC       = {cc:.4f}")
    print(f"R²       = {r2:.4f}")
    print("================================")


if __name__ == '__main__':
    main()
