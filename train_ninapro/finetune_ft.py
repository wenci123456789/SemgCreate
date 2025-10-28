#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import csv
import os, math, argparse, numpy as np
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader
from DataProcess import NinaPro
from Model.EMGMambaAttentionAdapter import EMGMambaAdapter
from utils.Methods.methods import compute_metrics_numpy  # 预训练脚本里用的实现
from torch.utils.tensorboard import SummaryWriter


# 运行一个个体的 FT
def run_ft_for_one_target(args, device, target_id: str):
    # 1) 数据路径（默认 E2_A1；如你的特征是 E1_A1，请把 E2 改成 E1）
    emg_tr = os.path.join(args.data_root, f"{target_id}_E2_A1_rms_train.h5")
    glo_tr = os.path.join(args.data_root, f"{target_id}_E2_A1_glove_train.h5")
    emg_te = os.path.join(args.data_root, f"{target_id}_E2_A1_rms_test.h5")
    glo_te = os.path.join(args.data_root, f"{target_id}_E2_A1_glove_test.h5")

    for p in [emg_tr, glo_tr, emg_te, glo_te]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"[{target_id}] Missing file: {p}")

    # 2) DataLoader
    ds_tr = NinaPro.NinaPro(emg_tr, glo_tr, subframe=args.subframe,
                            normalization=args.normalization, mu=args.miu,
                            dummy_label=0, class_num=1)
    ds_te = NinaPro.NinaPro(emg_te, glo_te, subframe=args.subframe,
                            normalization=args.normalization, mu=args.miu,
                            dummy_label=0, class_num=1)

    TrainLoader = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True, drop_last=True,
                             num_workers=0, pin_memory=False)
    ValLoader   = DataLoader(ds_te, batch_size=args.batch_size, shuffle=False,
                             num_workers=0, pin_memory=False)

    # 3) 模型：加载预训练参数；只训练 adapter + output_proj
    if not os.path.exists(args.pretrained):
        raise FileNotFoundError(f"Pretrained ckpt not found: {args.pretrained}")
    state = torch.load(args.pretrained, map_location='cpu')
    state = state.get('model_state', state)

    model = EMGMambaAdapter(input_dim=12, output_dim=10).to(device)
    model.load_state_dict(state, strict=False)
    #冻结参数
    for n, p in model.named_parameters():
        p.requires_grad = ('adapter' in n) or ('output_proj' in n)

    reg_loss  = nn.MSELoss()
    optimizer = optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)

    # 4) 保存目录
    save_dir_one = os.path.join(args.save_dir, f"ft_{target_id}")
    os.makedirs(save_dir_one, exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join(save_dir_one, "tb"))
    csv_path = os.path.join(save_dir_one, "history.csv")
    need_header = not os.path.exists(csv_path)
    csv_f = open(csv_path, "a", newline="")
    csv_w = csv.writer(csv_f)
    if need_header:
        csv_w.writerow(["epoch","train_mse","val_mse","nrmse","cc","r2","select_metric","score","is_best"])
    # 5) 训练循环
    # best_score：按指标方向选择；mse/nrmse 越小越好；cc/r2 越大越好
    best_score = math.inf if args.select_metric in ['mse', 'nrmse'] else -math.inf

    for epoch in range(1, args.epochs + 1):
        # ======= Train =======
        model.train()
        train_loss = 0.0
        nb = 0
        for batch in TrainLoader:
            # 兼容返回 (x,y,*_) 的形式
            x, y = batch[0], batch[1]
            x = x.squeeze(3).to(device)    # [B,200,12]
            y = y.to(device)               # [B,1,10]

            y_hat = model(x)               # [B,1,10]
            loss = reg_loss(y_hat, y)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            train_loss += float(loss.item())
            nb += 1
        avg_train = train_loss / max(1, nb)

        # ======= Val =======
        model.eval()
        val_loss = 0.0
        nb = 0
        preds_cpu, targets_cpu = [], []
        with torch.no_grad():
            for batch in ValLoader:
                x, y = batch[0], batch[1]
                x = x.squeeze(3).to(device)
                y = y.to(device)

                y_hat = model(x)
                val_loss += float(reg_loss(y_hat, y).item())
                nb += 1

                preds_cpu.append(y_hat.detach().cpu())
                targets_cpu.append(y.detach().cpu())

        avg_val = val_loss / max(1, nb)

        # 计算评估指标（参考预训练）
        if args.eval_metrics:
            yh_np = torch.cat(preds_cpu, dim=0).numpy().reshape(-1, 10)
            y_np  = torch.cat(targets_cpu, dim=0).numpy().reshape(-1, 10)
            try:
                NRMSE, CC, R2 = compute_metrics_numpy(y_np, yh_np)
            except Exception as e:
                print(f"[Warn] metric computation failed: {e}")
                NRMSE, CC, R2 = float('nan'), float('nan'), float('nan')
            writer.add_scalar("loss/train_mse", avg_train, epoch)
            writer.add_scalar("loss/val_mse", avg_val, epoch)
            writer.add_scalar("metrics/NRMSE", NRMSE, epoch)
            writer.add_scalar("metrics/CC", CC, epoch)
            writer.add_scalar("metrics/R2", R2, epoch)

            writer.flush()
        else:
            NRMSE, CC, R2 = float('nan'), float('nan'), float('nan')

        # 打印
        print(f"[FT {target_id}] Epoch {epoch:03d}  "
              f"Train(MSE)={avg_train:.6f}  Val(MSE)={avg_val:.6f}  "
              f"NRMSE={NRMSE:.4f}  CC={CC:.4f}  R2={R2:.4f}")

        # 通用评分逻辑 + 保存
        if args.select_metric == 'mse':
            cur_score = avg_val; is_better = cur_score < best_score
        elif args.select_metric == 'nrmse':
            cur_score = NRMSE;   is_better = cur_score < best_score
        elif args.select_metric == 'cc':
            cur_score = CC;      is_better = cur_score > best_score
        elif args.select_metric == 'r2':
            cur_score = R2;      is_better = cur_score > best_score
        else:
            cur_score = avg_val; is_better = cur_score < best_score  # 兜底

        # 始终保存 latest
        torch.save({'epoch': epoch, 'model_state': model.state_dict()},
                   os.path.join(save_dir_one, 'ft_latest.pth'))

        # 保存 best
        if is_better and not (isinstance(cur_score, float) and (math.isnan(cur_score) or math.isinf(cur_score))):
            best_score = cur_score
            torch.save({'epoch': epoch, 'model_state': model.state_dict()},
                       os.path.join(save_dir_one, 'ft_best.pth'))
        csv_w.writerow([epoch, avg_train, avg_val, NRMSE, CC, R2, args.select_metric, cur_score, int(is_better)])
        csv_f.flush()

        print(f"[FT {target_id}] Epoch {epoch:03d}  Train(MSE)={avg_train:.6f}  Val(MSE)={avg_val:.6f}  "
              f"NRMSE={NRMSE:.4f}  CC={CC:.4f}  R2={R2:.4f}  ")


        torch.cuda.empty_cache()

    writer.close()
    csv_f.close()

def main():
    ap = argparse.ArgumentParser(description="Fine-tuning (FT) with multi-target support & rich metrics")
    ap.add_argument('--data_root', type=str, default='../../../feature/ninapro_db2_trans')
    # 让 --pretrained 有默认值；也可在命令行覆盖
    ap.add_argument('--pretrained', type=str, default='../result/ninapro/checkpoints_pretrain/sEMGMamba/model_best.pth')
    # 向后兼容的单目标
    ap.add_argument('--target_subject', type=str, default=None)
    # 多目标：S31 S32 ... S40
    ap.add_argument('--targets', nargs='+', default=[f"S{i}" for i in range(31, 41)])
    ap.add_argument('--save_dir', type=str, default='../result/ninapro/Estimation_result/sEMGMamba/checkpoints_ft')
    ap.add_argument('--epochs', type=int, default=50)
    ap.add_argument('--batch_size', type=int, default=64)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--subframe', type=int, default=200)
    ap.add_argument('--normalization', type=str, default='miu')
    ap.add_argument('--miu', type=int, default=2 ** 20)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')

    # 新增：评估指标与选择 best 的策略
    ap.add_argument('--eval_metrics', action='store_true', default=True,
                    help='在验证阶段计算并打印 NRMSE/CC/R2')
    ap.add_argument('--select_metric', type=str, default='mse',
                    choices=['mse', 'nrmse', 'cc', 'r2'],
                    help='用哪个指标选择 best（mse/nrmse 越小越好；cc/r2 越大越好）')

    args = ap.parse_args()

    # 兼容：--targets 优先；否则退回到 --target_subject
    if args.targets is not None and len(args.targets) > 0:
        targets = args.targets
    elif args.target_subject is not None:
        targets = [args.target_subject]
    else:
        raise ValueError("请通过 --targets S31 S32 ... 或 --target_subject S31 指定受试者。")

    os.makedirs(args.save_dir, exist_ok=True)
    device = args.device

    print(f"Pretrained: {args.pretrained}")
    print(f"Targets: {targets}")
    print(f"μ-law miu={args.miu}, subframe={args.subframe}")
    print(f"Select metric: {args.select_metric}")

    for tgt in targets:
        print(f"\n====== FT start: {tgt} ======")
        run_ft_for_one_target(args, device, tgt)
        print(f"====== FT done : {tgt} ======\n")

if __name__ == '__main__':
    main()
