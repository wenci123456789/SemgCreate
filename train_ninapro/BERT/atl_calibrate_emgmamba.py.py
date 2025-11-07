#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ATL calibrate - BERT backbone (calibration-style)
- Follow finetune_ft.py: infer T/hidden from checkpoint, freeze backbone, train only 'fc'
"""
import os, math, argparse, csv
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
from torch.utils.tensorboard import SummaryWriter

from DataProcess import NinaPro
from utils.Methods.methods import compute_metrics_numpy
from utils.sEMG_models.sEMG_BERT import sEMG_BERT  # ← 改成 BERT

# ---------------- Domain Discriminator ----------------
class DomainDiscriminator(nn.Module):
    """ATL：对齐倒数第二层特征（这里取 BERT 的 fc 的输入张量）"""
    def __init__(self, in_dim, hidden=256, p=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(inplace=True), nn.Dropout(p),
            nn.Linear(hidden, 1)
        )
    def forward(self, f):
        return self.net(f).squeeze(-1)

# ---------------- Hook 包装器：抓取 head 输入特征 ----------------
class HookedModel(nn.Module):
    def __init__(self, backbone: nn.Module, feature_module_name: str = "fc"):
        super().__init__()
        self.backbone = backbone
        self._feat = None
        m = dict(self.backbone.named_modules())
        assert feature_module_name in m, f"找不到模块 {feature_module_name}"
        m[feature_module_name].register_forward_hook(self._hook)

    @torch.no_grad()
    def _hook(self, module, fin, fout):
        x = fin[0]            # BERT 的 fc 输入是 [B, vocab*hidden]
        self._feat = x

    def forward_with_features(self, x: torch.Tensor):
        out = self.backbone(x)            # sEMG_BERT 返回 (output, distr)
        if isinstance(out, (tuple, list)): y = out[0]
        else: y = out
        return y, self._feat

# ---------------- 其他小工具 ----------------
def load_state_flex(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location='cpu')
    if isinstance(ckpt, dict) and 'model_state' in ckpt:
        return ckpt['model_state']
    if hasattr(ckpt, 'state_dict'):
        return ckpt.state_dict()
    return ckpt

def build_loader(root, sid, subframe, normalization, mu, batch_size, shuffle, drop_last):
    e = os.path.join(root, f"{sid}_E2_A1_rms_train.h5")
    g = os.path.join(root, f"{sid}_E2_A1_glove_train.h5")
    e_te = os.path.join(root, f"{sid}_E2_A1_rms_test.h5")
    g_te = os.path.join(root, f"{sid}_E2_A1_glove_test.h5")
    ds_tr = NinaPro.NinaPro(e, g, subframe=subframe, normalization=normalization, mu=mu,
                            dummy_label=0, class_num=1)
    ds_te = NinaPro.NinaPro(e_te, g_te, subframe=subframe, normalization=normalization, mu=mu,
                            dummy_label=0, class_num=1)
    L_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last,
                      num_workers=0, pin_memory=False)
    L_te = DataLoader(ds_te, batch_size=batch_size, shuffle=False,
                      num_workers=0, pin_memory=False)
    return L_tr, L_te

def run_atl_for_target(args, device, target_subject: str, source_subjects: list):
    # ===== 数据 =====
    T_loader, Te_loader = build_loader(args.data_root, target_subject,
                                       args.subframe, args.normalization, args.miu,
                                       args.batch_size, shuffle=True, drop_last=True)

    # 去除与 target 相同的源域
    src_list = [s for s in source_subjects if s != target_subject]
    if len(src_list) == 0:
        raise ValueError("source_subjects 为空或仅包含 target。请提供至少一个不同的源被试。")

    S_sets = []
    for i, sid in enumerate(src_list):
        e = os.path.join(args.data_root, f"{sid}_E2_A1_rms_train.h5")
        g = os.path.join(args.data_root, f"{sid}_E2_A1_glove_train.h5")
        S_sets.append(NinaPro.NinaPro(e, g, subframe=args.subframe,
                                      normalization=args.normalization, mu=args.miu,
                                      dummy_label=i, class_num=len(src_list)))
    S_loader = DataLoader(ConcatDataset(S_sets), batch_size=args.batch_size, shuffle=True, drop_last=True,
                          num_workers=0, pin_memory=False)

    # ===== 模型（Teacher: 冻结；Student: 拷贝）=====
    ms_net = sEMG_BERT(vocab_size=args.subframe, hidden=128, feature_dim=1, n_layers=4, attn_heads=8).to(device)  # 12 通道 → 10 关节（模型里 fc 做 10）
    ms_net.load_state_dict(load_state_flex(args.pretrained), strict=False)
    for p in ms_net.parameters(): p.requires_grad = False

    nt_net = sEMG_BERT(vocab_size=args.subframe, hidden=128, feature_dim=1, n_layers=4, attn_heads=8).to(device)
    nt_net.load_state_dict(ms_net.state_dict(), strict=False)

    ms = HookedModel(ms_net, feature_module_name='fc').to(device)
    nt = HookedModel(nt_net, feature_module_name='fc').to(device)

    # ===== 域判别器（输入维度=fc 输入维度）=====
    with torch.no_grad():
        xb, yb, *_ = next(iter(T_loader))
        xb = xb.squeeze(3).to(device).float()       # [B,200,12] float
        _y, f = nt.forward_with_features(xb)
        feat_dim = f.shape[-1]
    D = DomainDiscriminator(in_dim=feat_dim).to(device)

    # ===== 优化器/损失 =====
    reg_loss = nn.MSELoss()
    bce_logits = nn.BCEWithLogitsLoss()
    opt_t = optim.AdamW(nt_net.parameters(), lr=args.lr_t, weight_decay=1e-4)
    opt_d = optim.AdamW(D.parameters(),     lr=args.lr_d, weight_decay=1e-4)

    # ===== 日志/保存 =====
    save_dir_one = os.path.join(args.save_dir, f"atl_{target_subject}")
    os.makedirs(save_dir_one, exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join(save_dir_one, "tb"))

    csv_path = os.path.join(save_dir_one, "history.csv")
    need_header = not os.path.exists(csv_path)
    csv_f = open(csv_path, "a", newline="")
    csv_w = csv.writer(csv_f)
    if need_header:
        csv_w.writerow(["epoch","train_reg","train_D","val_mse","nrmse","cc","r2","select_metric","score","is_best"])

    best_score = math.inf if args.select_metric in ['mse','nrmse'] else -math.inf

    # ===== 训练 =====
    for epoch in range(1, args.epochs + 1):
        nt.train(); D.train(); ms.eval()
        it_s = iter(S_loader)
        total_reg, total_d = 0.0, 0.0

        for batch_t in T_loader:
            x_t = batch_t[0].squeeze(3).to(device).float()
            y_t = batch_t[1].to(device).float()

            try:
                batch_s = next(it_s)
            except StopIteration:
                it_s = iter(S_loader); batch_s = next(it_s)
            x_s = batch_s[0].squeeze(3).to(device).float()

            # 1) 训练 D
            with torch.no_grad():
                _ , f_s = ms.forward_with_features(x_s)
                _ , f_t = nt.forward_with_features(x_t)
                f_s = f_s.detach(); f_t = f_t.detach()
            logit_s, logit_t = D(f_s), D(f_t)
            loss_d = bce_logits(logit_s, torch.ones_like(logit_s)) + \
                     bce_logits(logit_t, torch.zeros_like(logit_t))
            opt_d.zero_grad(); loss_d.backward(); opt_d.step()

            # 2) 训练 nt（任务 + 混淆判别器）
            y_hat, f_t = nt.forward_with_features(x_t)
            loss_map = bce_logits(D(f_t), torch.ones_like(logit_t))
            loss_reg = reg_loss(y_hat, y_t)
            loss_tot = loss_reg + args.lambda_adv * loss_map
            opt_t.zero_grad(); loss_tot.backward(); opt_t.step()

            total_reg += loss_reg.item(); total_d += loss_d.item()

        # ===== 验证 =====
        nt.eval()
        val_mse, n_batch = 0.0, 0
        preds_cpu, targets_cpu = [], []
        with torch.no_grad():
            for batch in Te_loader:
                x = batch[0].squeeze(3).to(device).float()
                y = batch[1].to(device).float()
                y_hat, _ = nt.forward_with_features(x)
                val_mse += reg_loss(y_hat, y).item(); n_batch += 1
                preds_cpu.append(y_hat.detach().cpu()); targets_cpu.append(y.detach().cpu())
        val_mse /= max(1, n_batch)

        try:
            yh_np = torch.cat(preds_cpu, dim=0).numpy().reshape(-1, 10)
            y_np  = torch.cat(targets_cpu, dim=0).numpy().reshape(-1, 10)
            NRMSE, CC, R2 = compute_metrics_numpy(y_np, yh_np)
        except Exception as e:
            print(f"[Warn] metric computation failed: {e}")
            NRMSE, CC, R2 = float('nan'), float('nan'), float('nan')

        writer.add_scalar("loss/val_mse", val_mse, epoch)
        writer.add_scalar("metrics/NRMSE", NRMSE, epoch)
        writer.add_scalar("metrics/CC", CC, epoch)
        writer.add_scalar("metrics/R2", R2, epoch)
        writer.add_scalar("loss/train_reg", total_reg/len(T_loader), epoch)
        writer.add_scalar("loss/train_D", total_d/len(T_loader), epoch)
        writer.flush()

        print(f"[ATL {target_subject}] Epoch {epoch:03d}  "
              f"train_reg={total_reg/len(T_loader):.6f}  train_D={total_d/len(T_loader):.6f}  "
              f"Val(MSE)={val_mse:.6f}  NRMSE={NRMSE:.4f}  CC={CC:.4f}  R2={R2:.4f}")

        # 选择 best & 保存
        if args.select_metric == 'mse':
            cur_score = val_mse; is_better = cur_score < best_score
        elif args.select_metric == 'nrmse':
            cur_score = NRMSE;   is_better = cur_score < best_score
        elif args.select_metric == 'cc':
            cur_score = CC;      is_better = cur_score > best_score
        elif args.select_metric == 'r2':
            cur_score = R2;      is_better = cur_score > best_score
        else:
            cur_score = val_mse; is_better = cur_score < best_score

        torch.save({'epoch': epoch, 'model_state': nt_net.state_dict()},
                   os.path.join(save_dir_one, 'atl_latest.pth'))
        if is_better and not (isinstance(cur_score, float) and (math.isnan(cur_score) or math.isinf(cur_score))):
            best_score = cur_score
            torch.save({'epoch': epoch, 'model_state': nt_net.state_dict()},
                       os.path.join(save_dir_one, 'atl_best.pth'))

        writer.flush()
        torch.cuda.empty_cache()

    writer.close(); csv_f.close()

def main():
    ap = argparse.ArgumentParser(description="ATL calibrate — BERT backbone")
    ap.add_argument('--data_root', type=str, default='../../../../feature/ninapro_db2_trans')
    ap.add_argument('--pretrained', type=str, default='../../result/ninapro/checkpoints_pretrain/BERT/model_best.pth')
    ap.add_argument('--save_dir', type=str, default='../../result/ninapro/Estimation_result/BERT/checkpoints_atl')
    ap.add_argument('--epochs', type=int, default=50)
    ap.add_argument('--batch_size', type=int, default=64)
    ap.add_argument('--lr_t', type=float, default=1e-3)
    ap.add_argument('--lr_d', type=float, default=1e-3)
    ap.add_argument('--lambda_adv', type=float, default=1.0)
    ap.add_argument('--subframe', type=int, default=200)
    ap.add_argument('--normalization', type=str, default='miu')
    ap.add_argument('--miu', type=int, default=2 ** 20)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--targets', nargs='+', default=[f"S{i}" for i in range(31, 41)])
    ap.add_argument('--source_subjects', nargs='+', default=[f"S{i}" for i in range(1, 31)])
    ap.add_argument('--eval_metrics', action='store_true', default=True)
    ap.add_argument('--select_metric', type=str, default='mse', choices=['mse','nrmse','cc','r2'])
    args = ap.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    device = args.device

    targets = args.targets if args.targets else [args.target_subject]
    print(f"Pretrained: {args.pretrained}")
    print(f"Targets: {targets}")
    print(f"Sources: {args.source_subjects}")
    print(f"Save dir: {args.save_dir}")
    print(f"Select metric: {args.select_metric}")

    for tgt in targets:
        print(f"\n====== ATL start: {tgt} ======")
        run_atl_for_target(args, device, tgt, args.source_subjects)
        print(f"====== ATL done : {tgt} ======\n")

if __name__ == "__main__":
    main()
