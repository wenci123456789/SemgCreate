#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv, os, math, argparse, numpy as np
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from DataProcess import NinaPro
from utils.Methods.methods import compute_metrics_numpy
from utils.sEMG_models.sEMG_RoFormer import RoFormerEMG  # RoFormer 主干

def run_ft_for_one_target(args, device, target_id: str):
    # -------- 文件路径 --------
    emg_tr = os.path.join(args.data_root, f"{target_id}_E2_A1_rms_train.h5")
    glo_tr = os.path.join(args.data_root, f"{target_id}_E2_A1_glove_train.h5")
    emg_te = os.path.join(args.data_root, f"{target_id}_E2_A1_rms_test.h5")
    glo_te = os.path.join(args.data_root, f"{target_id}_E2_A1_glove_test.h5")
    for p in [emg_tr, glo_tr, emg_te, glo_te]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"[{target_id}] Missing file: {p}")

    # -------- 数据加载 --------
    ds_tr = NinaPro.NinaPro(emg_tr, glo_tr, subframe=args.subframe, normalization=args.normalization,
                            mu=args.miu, dummy_label=0, class_num=1)
    ds_te = NinaPro.NinaPro(emg_te, glo_te, subframe=args.subframe, normalization=args.normalization,
                            mu=args.miu, dummy_label=0, class_num=1)
    TrainLoader = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True, drop_last=True,
                             num_workers=0, pin_memory=False)
    ValLoader   = DataLoader(ds_te, batch_size=args.batch_size, shuffle=False,
                             num_workers=0, pin_memory=False)

    # -------- 加载预训练权重 --------
    if not os.path.exists(args.pretrained):
        raise FileNotFoundError(f"Pretrained ckpt not found: {args.pretrained}")
    ckpt  = torch.load(args.pretrained, map_location='cpu')
    state = ckpt.get('model_state', ckpt)

    # -------- 构造 RoFormer（与预训练同构）--------
    model = RoFormerEMG(
        input_dim=12, output_dim=10,
        d_model=120, num_layers=2, num_heads=5,
        use_mu_law=False
    ).to(device)

    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print("[load_state] missing:", missing)
        print("[load_state] unexpected:", unexpected)

    # -------- 只训练输出层（常见名字：regressor / fc / output）--------
    for n, p in model.named_parameters():
        p.requires_grad = any(k in n for k in ['regressor', 'fc', 'output'])
    reg_loss  = nn.MSELoss()
    optimizer = optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=args.lr, weight_decay=1e-4)

    # -------- 日志与保存 --------
    save_dir_one = os.path.join(args.save_dir, f"ft_{target_id}")
    os.makedirs(save_dir_one, exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join(save_dir_one, "tb"))
    csv_path = os.path.join(save_dir_one, "history.csv")
    need_header = not os.path.exists(csv_path)
    csv_f = open(csv_path, "a", newline=""); csv_w = csv.writer(csv_f)
    if need_header:
        csv_w.writerow(["epoch","train_mse","val_mse","nrmse","cc","r2","select_metric","score","is_best"])

    best_score = math.inf if args.select_metric in ['mse','nrmse'] else -math.inf

    # -------- 训练过程 --------
    for epoch in range(1, args.epochs + 1):
        model.train(); train_loss = 0.0; nb = 0
        for x, y, *_ in TrainLoader:  # ← 只取前两个，其余忽略
            x = x.squeeze(3).to(device).float()          # [B,T,C]
            y = y.to(device).float().view(x.size(0), -1) # [B,10]

            out = model(x)                                # 可能是 [B,T,10]
            y_hat = out[0] if isinstance(out, (tuple, list)) else out
            if y_hat.dim() == 3:                          # [B,T,10] -> [B,10]
                y_hat = y_hat.mean(dim=1)

            loss = reg_loss(y_hat, y)
            optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
            train_loss += float(loss.item()); nb += 1

        avg_train = train_loss / max(1, nb)

        # -------- 验证过程 --------
        model.eval(); val_loss = 0.0; nb = 0
        preds_cpu, targets_cpu = [], []
        with torch.no_grad():
            for x, y, *_ in ValLoader:  # ← 同样只取前两个
                x = x.squeeze(3).to(device).float()
                y = y.to(device).float().view(x.size(0), -1)  # [B,10]

                out = model(x)
                y_hat = out[0] if isinstance(out, (tuple, list)) else out
                if y_hat.dim() == 3:
                    y_hat = y_hat.mean(dim=1)  # [B,10]

                val_loss += float(reg_loss(y_hat, y).item()); nb += 1
                preds_cpu.append(y_hat.detach().cpu()); targets_cpu.append(y.detach().cpu())

        avg_val = val_loss / max(1, nb)

        # -------- 评估指标 --------
        try:
            yh_np = torch.cat(preds_cpu, dim=0).numpy().reshape(-1, 10)
            y_np  = torch.cat(targets_cpu, dim=0).numpy().reshape(-1, 10)
            NRMSE, CC, R2 = compute_metrics_numpy(y_np, yh_np)
        except Exception as e:
            print(f"[Warn] metric computation failed: {e}")
            NRMSE = CC = R2 = float('nan')

        print(f"Epoch [{epoch}/{args.epochs}] - "
              f"Train MSE: {avg_train:.6f}, Val MSE: {avg_val:.6f}, "
              f"NRMSE: {NRMSE:.6f}, CC: {CC:.6f}, R2: {R2:.6f}")

        writer.add_scalar("loss/train_mse", avg_train, epoch)
        writer.add_scalar("loss/val_mse",   avg_val,   epoch)
        writer.add_scalar("metrics/NRMSE",  NRMSE,     epoch)
        writer.add_scalar("metrics/CC",     CC,        epoch)
        writer.add_scalar("metrics/R2",     R2,        epoch)
        writer.flush()

        # -------- 保存权重 --------
        if args.select_metric == 'mse':
            cur_score, is_better = avg_val, (avg_val < best_score)
        elif args.select_metric == 'nrmse':
            cur_score, is_better = NRMSE, (NRMSE < best_score)
        elif args.select_metric == 'cc':
            cur_score, is_better = CC, (CC > best_score)
        elif args.select_metric == 'r2':
            cur_score, is_better = R2, (R2 > best_score)
        else:
            cur_score, is_better = avg_val, (avg_val < best_score)

        torch.save({'epoch': epoch, 'model_state': model.state_dict()},
                   os.path.join(save_dir_one, 'ft_latest.pth'))

        if is_better and not (isinstance(cur_score, float) and (math.isnan(cur_score) or math.isinf(cur_score))):
            best_score = cur_score
            torch.save({'epoch': epoch, 'model_state': model.state_dict()},
                       os.path.join(save_dir_one, 'ft_best.pth'))

        csv_w.writerow([epoch, avg_train, avg_val, NRMSE, CC, R2,
                        args.select_metric, cur_score, int(is_better)])
        csv_f.flush()
        torch.cuda.empty_cache()

    writer.close(); csv_f.close()

def main():
    ap = argparse.ArgumentParser(description="Fine-tuning (RoFormer backbone)")
    # 数据与路径
    ap.add_argument('--data_root', type=str, default='../../../../feature/ninapro_db2_trans')
    ap.add_argument('--pretrained', type=str, default='../../result/ninapro/checkpoints_pretrain/Roformer/model_best.pth')
    ap.add_argument('--target_subject', type=str, default=None)
    ap.add_argument('--targets', nargs='+', default=[f"S{i}" for i in range(31, 41)])
    ap.add_argument('--save_dir', type=str, default='../../result/ninapro/Estimation_result/Roformer/checkpoints_ft')

    # 训练超参
    ap.add_argument('--epochs', type=int, default=50)
    ap.add_argument('--batch_size', type=int, default=32)
    ap.add_argument('--lr', type=float, default=1e-3)

    # 数据形状/归一化
    ap.add_argument('--subframe', type=int, default=200)
    ap.add_argument('--normalization', type=str, default='miu')
    ap.add_argument('--miu', type=int, default=2 ** 20)

    # 设备 & 指标
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--eval_metrics', action='store_true', default=True)
    ap.add_argument('--select_metric', type=str, default='mse', choices=['mse', 'nrmse', 'cc', 'r2'])
    args = ap.parse_args()

    targets = args.targets if args.targets else [args.target_subject]
    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device(args.device)

    print(f"Pretrained: {args.pretrained}")
    print(f"Targets: {targets}")
    print(f"μ-law miu={args.miu}, subframe={args.subframe}")
    print(f"Select metric: {args.select_metric}")

    for tgt in targets:
        print(f"\n====== FT-RoFormer start: {tgt} ======")
        run_ft_for_one_target(args, device, tgt)
        print(f"====== FT-RoFormer done : {tgt} ======\n")

if __name__ == '__main__':
    main()
