#!/usr/bin/env python3
"""
跨被试预训练 (Multi-s-net)
- 训练集：S1–S30 前 5 次重复（*_rms_train.h5）
- 验证集：S1–S30 第 6 次重复（*_rms_test.h5）
- 模型：EMGMambaAdapter
"""

import argparse, os, math, time, random
import numpy as np
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.tensorboard import SummaryWriter
from DataProcess import NinaPro
from sklearn import metrics as skmetrics
from skimage import metrics

from utils.Methods.methods import str2bool
from utils.sEMG_models.sEMG_BERT import sEMG_BERT


# reproducibility
def seed_everything(seed=525):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True

# Pearson CC
def pearson_CC(x, y):
    x, y = x.flatten(), y.flatten()
    vx, vy = x - x.mean(), y - y.mean()
    cc = np.sum(vx * vy) / (np.sqrt(np.sum(vx ** 2)) * np.sqrt(np.sum(vy ** 2)) + 1e-8)
    return cc

def main():
    parser = argparse.ArgumentParser(description='Pretrain Multi-s-net on S1–S30')
    parser.add_argument('--subjects', nargs='+', default=[f"S{i}" for i in range(1, 31)])
    parser.add_argument('--data_root', type=str, default='../../../feature/ninapro_db2_trans')
    parser.add_argument('--epoch', type=int, default=200)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--subframe', type=int, default=200)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--normalization', type=str, default='miu')
    parser.add_argument('--miu', type=int, default=2 ** 20)
    parser.add_argument('--save_dir', type=str, default='../result/ninapro/checkpoints_pretrain/BERT')
    parser.add_argument('--hidden', type=int, default=128)
    parser.add_argument('--num_layers', type=int, default=4)
    parser.add_argument('--use_se', type=str2bool, default=False)
    args = parser.parse_args()

    seed_everything(525)
    os.makedirs(args.save_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join(args.save_dir, 'tb_logs'))

    # ============ 数据加载 ============
    train_sets, val_sets = [], []
    for sid in args.subjects:
        emg_tr = os.path.join(args.data_root, f"{sid}_E2_A1_rms_train.h5")
        glove_tr = os.path.join(args.data_root, f"{sid}_E2_A1_glove_train.h5")
        emg_te = os.path.join(args.data_root, f"{sid}_E2_A1_rms_test.h5")
        glove_te = os.path.join(args.data_root, f"{sid}_E2_A1_glove_test.h5")

        if not os.path.exists(emg_tr):
            print(f"[WARN] Missing {emg_tr}, skip {sid}")
            continue

        train_sets.append(NinaPro.NinaPro(
            emg_tr, glove_tr, subframe=args.subframe,
            normalization=args.normalization, mu=args.miu,
            dummy_label=0, class_num=1))
        val_sets.append(NinaPro.NinaPro(
            emg_te, glove_te, subframe=args.subframe,
            normalization=args.normalization, mu=args.miu,
            dummy_label=0, class_num=1))

    TrainLoader = DataLoader(ConcatDataset(train_sets), batch_size=args.batch_size, num_workers=0,shuffle=True, drop_last=True)
    ValLoader = DataLoader(ConcatDataset(val_sets), batch_size=args.batch_size, shuffle=False ,num_workers=0 )

    print(f"✅ Loaded {len(train_sets)} subjects for training, {len(val_sets)} for validation")

    # ============ 模型与优化 ============
    device = args.device
    model = sEMG_BERT(vocab_size=args.subframe, hidden=args.hidden, feature_dim=1, n_layers=args.num_layers,
                      attn_heads=8, use_se=args.use_se).to(device)
    reg_loss = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = MultiStepLR(optimizer, milestones=[200], gamma=0.5)

    best_nrmse = math.inf

    # ============ 训练循环 ============
    for epoch in range(1, args.epoch + 1):
        model.train(); train_loss = 0.0
        for x, y, *_ in TrainLoader:
            x = x.squeeze(3).to(device)       # [B,200,12]
            y = y.to(device)                  # [B,1,10]
            optimizer.zero_grad()
            out = model(x.float())  # BERT 的 token 用 Linear，需要 float
            pred = out[0] if isinstance(out, (tuple, list)) else out
            pred = pred.mean(dim=1, keepdim=True)
            loss = reg_loss(pred, y)
            loss.backward(); optimizer.step()
            train_loss += loss.item()
        scheduler.step()

        # ============ 验证 ============
        model.eval(); val_loss = 0.0
        preds, trues = [], []
        with torch.no_grad():
            for x, y, *_ in ValLoader:
                x = x.squeeze(3).to(device)
                y = y.to(device)
                out = model(x.float())  # BERT 的 token 用 Linear，需要 float
                pred = out[0] if isinstance(out, (tuple, list)) else out
                pred = pred.mean(dim=1, keepdim=True)
                val_loss += reg_loss(pred, y).item()
                preds.append(pred.cpu().numpy())
                trues.append(y.cpu().numpy())
        preds = np.concatenate(preds, axis=0)[:, 0, :]
        trues = np.concatenate(trues, axis=0)[:, 0, :]  # [N, 10]
        nrmse = metrics.normalized_root_mse(trues, preds)
        cc = pearson_CC(trues, preds)
        r2 = skmetrics.r2_score(trues, preds)
        avg_train = train_loss / len(TrainLoader)
        avg_val = val_loss / len(ValLoader)

        print(f"[Epoch {epoch:03d}] Train={avg_train:.5f}  Val={avg_val:.5f}  "
              f"NRMSE={nrmse:.4f}  CC={cc:.4f}  R²={r2:.4f}")

        writer.add_scalar('Loss/train_ninapro', avg_train, epoch)
        writer.add_scalar('Loss/val', avg_val, epoch)
        writer.add_scalar('Metrics/NRMSE', nrmse, epoch)
        writer.add_scalar('Metrics/CC', cc, epoch)
        writer.add_scalar('Metrics/R2', r2, epoch)

        # ============ 保存模型 ============
        state = {'epoch': epoch, 'model_state': model.state_dict()}
        torch.save(state, os.path.join(args.save_dir, 'model_latest.pth'))
        if nrmse < best_nrmse:
            best_nrmse = nrmse
            torch.save(state, os.path.join(args.save_dir, 'model_best.pth'))

    print("✅ Pretraining finished. Best NRMSE:", best_nrmse)

if __name__ == '__main__':
    main()
