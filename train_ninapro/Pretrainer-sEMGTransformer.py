#!/usr/bin/env python3
"""
跨被试预训练 (Multi-s-net) + 被试级 K-fold 交叉验证
- 模型：MAFN (Transformer)
"""

import argparse, os, math, random
import numpy as np
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.tensorboard import SummaryWriter
from sklearn import metrics as skmetrics
from skimage import metrics

from DataProcess import NinaPro
from utils.sEMG_models.transformer import MAFN


def seed_everything(seed=525):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def pearson_CC(x, y):
    x, y = x.flatten(), y.flatten()
    vx, vy = x - x.mean(), y - y.mean()
    return np.sum(vx * vy) / (np.sqrt(np.sum(vx ** 2)) *
                              np.sqrt(np.sum(vy ** 2)) + 1e-8)


def main():
    parser = argparse.ArgumentParser(
        description='Pretrain Transformer (MAFN) Multi-s-net with subject-level K-fold')
    parser.add_argument('--subjects', nargs='+',
                        default=["S1", "S4", "S7", "S8", "S11",
                                 "S13", "S18", "S20", "S22", "S24",
                                 "S27", "S31", "S34", "S36", "S39"])
    parser.add_argument('--k_folds', type=int, default=5)
    parser.add_argument('--data_root', type=str,
                        default='../../../feature/ninapro_db2_trans')
    parser.add_argument('--epoch', type=int, default=200)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--subframe', type=int, default=200)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--normalization', type=str, default='miu')
    parser.add_argument('--miu', type=int, default=2 ** 20)
    parser.add_argument('--save_dir', type=str,
                        default='../result/ninapro/checkpoints_fold_pretrain/Transformer')
    args = parser.parse_args()

    subjects = list(args.subjects)
    if len(subjects) < args.k_folds:
        raise ValueError(f"#subjects ({len(subjects)}) < k_folds ({args.k_folds})")
    folds = np.array_split(subjects, args.k_folds)

    print(f"All subjects: {subjects}")
    for i, f in enumerate(folds):
        print(f"Fold {i+1}: val subjects = {list(f)}")

    os.makedirs(args.save_dir, exist_ok=True)

    for fold_idx, val_subjects_np in enumerate(folds):
        val_subjects = list(val_subjects_np)
        train_subjects = [s for s in subjects if s not in val_subjects]

        print("\n" + "=" * 60)
        print(f"⭐ Transformer Fold {fold_idx+1}/{args.k_folds}")
        print(f"Train subjects: {train_subjects}")
        print(f"Val subjects:   {val_subjects}")
        print("=" * 60)

        seed_everything(525 + fold_idx)

        fold_save_dir = os.path.join(args.save_dir, f"fold{fold_idx+1}")
        os.makedirs(fold_save_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=os.path.join(fold_save_dir, 'tb_logs'))

        train_sets, val_sets = [], []
        for sid in train_subjects:
            emg_tr = os.path.join(args.data_root, f"{sid}_E2_A1_rms_train.h5")
            glove_tr = os.path.join(args.data_root, f"{sid}_E2_A1_glove_train.h5")
            if not os.path.exists(emg_tr):
                print(f"[WARN][Fold{fold_idx+1}] Missing {emg_tr}, skip {sid} train")
                continue
            train_sets.append(
                NinaPro.NinaPro(
                    emg_tr, glove_tr, subframe=args.subframe,
                    normalization=args.normalization, mu=args.miu,
                    dummy_label=0, class_num=1
                )
            )
        for sid in val_subjects:
            emg_te = os.path.join(args.data_root, f"{sid}_E2_A1_rms_test.h5")
            glove_te = os.path.join(args.data_root, f"{sid}_E2_A1_glove_test.h5")
            if not os.path.exists(emg_te):
                print(f"[WARN][Fold{fold_idx+1}] Missing {emg_te}, skip {sid} val")
                continue
            val_sets.append(
                NinaPro.NinaPro(
                    emg_te, glove_te, subframe=args.subframe,
                    normalization=args.normalization, mu=args.miu,
                    dummy_label=0, class_num=1
                )
            )

        if len(train_sets) == 0 or len(val_sets) == 0:
            print(f"[ERROR][Fold{fold_idx+1}] No train/val data, skip.")
            continue

        TrainLoader = DataLoader(
            ConcatDataset(train_sets),
            batch_size=args.batch_size,
            shuffle=True, drop_last=True,
            num_workers=12
        )
        ValLoader = DataLoader(
            ConcatDataset(val_sets),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=12
        )

        print(f"[Fold{fold_idx+1}] ✅ Loaded {len(train_sets)} train subjects, "
              f"{len(val_sets)} val subjects")

        device = args.device
        model = MAFN(
            winsize=args.subframe,
            patch_size=1,
            in_c=1,
            num_classes=10,
            depth=4,
            num_heads=4,
            embed_dim=12,
            attn_drop_ratio=0.0,
            drop_ratio=0.3
        ).to(device)
        reg_loss = nn.MSELoss()
        optimizer = optim.AdamW(model.parameters(), lr=args.lr)
        scheduler = MultiStepLR(optimizer, milestones=[100], gamma=0.5)

        best_nrmse = math.inf

        for epoch in range(1, args.epoch + 1):
            model.train(); train_loss = 0.0
            for x, y, *_ in TrainLoader:
                x = x.squeeze(3).to(device)      # [B,200,12]
                y = y.squeeze(1).to(device)      # [B,10]
                optimizer.zero_grad()
                pred = model(x)
                loss = reg_loss(pred, y)
                loss.backward(); optimizer.step()
                train_loss += loss.item()
            scheduler.step()

            model.eval(); val_loss = 0.0
            preds, trues = [], []
            with torch.no_grad():
                for x, y, *_ in ValLoader:
                    x = x.squeeze(3).to(device)
                    y = y.squeeze(1).to(device)
                    pred = model(x)
                    val_loss += reg_loss(pred, y).item()
                    preds.append(pred.cpu().numpy())
                    trues.append(y.cpu().numpy())
            preds = np.concatenate(preds, axis=0)
            trues = np.concatenate(trues, axis=0)
            nrmse = metrics.normalized_root_mse(trues, preds)
            cc = pearson_CC(trues, preds)
            r2 = skmetrics.r2_score(trues, preds)
            avg_train = train_loss / len(TrainLoader)
            avg_val = val_loss / len(ValLoader)

            print(f"[Transformer Fold {fold_idx+1} | Epoch {epoch:03d}] "
                  f"Train={avg_train:.5f}  Val={avg_val:.5f}  "
                  f"NRMSE={nrmse:.4f}  CC={cc:.4f}  R²={r2:.4f}")

            writer.add_scalar('Loss/train', avg_train, epoch)
            writer.add_scalar('Loss/val', avg_val, epoch)
            writer.add_scalar('Metrics/NRMSE', nrmse, epoch)
            writer.add_scalar('Metrics/CC', cc, epoch)
            writer.add_scalar('Metrics/R2', r2, epoch)

            state = {
                'epoch': epoch,
                'model_state': model.state_dict(),
                'fold': fold_idx + 1,
                'train_subjects': train_subjects,
                'val_subjects': val_subjects
            }
            torch.save(state, os.path.join(fold_save_dir, 'model_latest.pth'))
            if nrmse < best_nrmse:
                best_nrmse = nrmse
                torch.save(state, os.path.join(fold_save_dir, 'model_best.pth'))

        print(f"[Transformer Fold{fold_idx+1}] ✅ Finished. Best NRMSE: {best_nrmse:.4f}")

    print("🎉 Transformer K-fold pretraining finished.")


if __name__ == '__main__':
    main()
