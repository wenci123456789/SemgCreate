# cdanr_bert.py
# 说明：
# - 主干：sEMG_BERT（与预训练保持同构）
# - 校准：CDANR（外积条件判别器 + GRL + 对抗 warm-up + 可选 R1）
# - 无 CorrBoost / 无 TTA
# - best.pth 依据“十个关节逐关节 Pearson CC 的平均值”保存

import os, math, argparse, random
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
from torch.utils.tensorboard import SummaryWriter

# ====== 项目内依赖（按你的工程结构） ======
from DataProcess import NinaPro
from utils.sEMG_models.sEMG_BERT import sEMG_BERT
from utils.Methods.methods import compute_metrics_numpy, pearson_CC

# ------------------ 实用函数 ------------------
def set_seed(seed: int = 525):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def l2n(z: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return z / (z.norm(dim=-1, keepdim=True) + eps)

def load_state_flex(path: str):
    obj = torch.load(path, map_location='cpu')
    if isinstance(obj, dict) and 'model_state' in obj:
        return obj['model_state']
    if hasattr(obj, 'state_dict'):
        return obj.state_dict()
    return obj

# ------------------ GRL ------------------
class GRL(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd: float):
        ctx.lambd = lambd
        return x.view_as(x)
    @staticmethod
    def backward(ctx, g):
        return -ctx.lambd * g, None

def grl(x, lambd: float):
    return GRL.apply(x, lambd)

# ------------------ 轻增广（可选） ------------------
def aug_gaussian_jitter(x: torch.Tensor, std: float) -> torch.Tensor:
    if std <= 0: return x
    return x + torch.randn_like(x) * std

def aug_channel_drop(x: torch.Tensor, p: float = 0.1) -> torch.Tensor:
    if p <= 0: return x
    B, T, C = x.shape
    mask = (torch.rand(B, C, device=x.device) > p).float()
    return x * mask.unsqueeze(1)

# ------------------ 数据构建 ------------------
def build_loader(root: str, sid: str, subframe: int, normalization: str, mu: float,
                 batch_size: int, shuffle: bool, drop_last: bool):
    e_tr = os.path.join(root, f"{sid}_E2_A1_rms_train.h5")
    g_tr = os.path.join(root, f"{sid}_E2_A1_glove_train.h5")
    e_te = os.path.join(root, f"{sid}_E2_A1_rms_test.h5")
    g_te = os.path.join(root, f"{sid}_E2_A1_glove_test.h5")
    ds_tr = NinaPro.NinaPro(e_tr, g_tr, subframe=subframe, normalization=normalization, mu=mu,
                            dummy_label=0, class_num=1)
    ds_te = NinaPro.NinaPro(e_te, g_te, subframe=subframe, normalization=normalization, mu=mu,
                            dummy_label=0, class_num=1)
    L_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last,
                      num_workers=0, pin_memory=False)
    L_te = DataLoader(ds_te, batch_size=batch_size, shuffle=False,
                      num_workers=0, pin_memory=False)
    return L_tr, L_te

def build_source_loader(root: str, subjects: List[str], subframe: int, normalization: str, mu: float, batch_size: int) -> DataLoader:
    S_sets = []
    for i, sid in enumerate(subjects):
        e = os.path.join(root, f"{sid}_E2_A1_rms_train.h5")
        g = os.path.join(root, f"{sid}_E2_A1_glove_train.h5")
        S_sets.append(
            NinaPro.NinaPro(
                e, g, subframe=subframe, normalization=normalization, mu=mu,
                dummy_label=i, class_num=len(subjects),
            )
        )
    return DataLoader(ConcatDataset(S_sets), batch_size=batch_size, shuffle=True, drop_last=True,
                      num_workers=0, pin_memory=False)

# ------------------ HookedModel：抓“头前特征” ------------------
class HookedModel(nn.Module):
    """
    对 BERT：默认抓 'fc' 的输入作为“头前特征 f”，并通过 head 产生 y_hat。
    如果你的 sEMG_BERT 最后一层线性层名不是 'fc'，用 --head_name 指定。
    """
    def __init__(self, model: nn.Module, head_name: str = 'fc'):
        super().__init__()
        self.model = model
        self._feat: Optional[torch.Tensor] = None
        self.head_name = head_name

        named = dict(self.model.named_modules())
        if head_name not in named:
            raise RuntimeError(f"找不到输出头模块: {head_name}")
        self.head = named[head_name]
        named[head_name].register_forward_hook(self._hook)

    def _hook(self, mod, fin, fout):
        # 线性层的输入： [B, D]
        self._feat = fin[0]

    def forward(self, x):
        out = self.model(x)
        return out

    def forward_with_features(self, x):
        out = self.model(x)
        if isinstance(out, (tuple, list)):
            y = out[0]
        else:
            y = out
        return y, self._feat

    # 若需要单独把 f 走 head，可按需改（此处默认直接用模型自己的 head）

# ------------------ 判别器（CDAN 外积条件） ------------------
class CDANRDiscriminator(nn.Module):
    def __init__(self, f_dim: int, y_dim: int, hidden: int = 512, p: float = 0.2):
        super().__init__()
        in_dim = f_dim * y_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(inplace=True), nn.Dropout(p),
            nn.Linear(hidden, hidden // 2), nn.ReLU(inplace=True), nn.Dropout(p),
            nn.Linear(hidden // 2, 1)
        )
    def forward(self, f, y_code):
        if f.dim() == 3 and f.size(1) == 1: f = f.squeeze(1)
        if y_code.dim() == 3 and y_code.size(1) == 1: y_code = y_code.squeeze(1)
        f = l2n(f); y_code = l2n(y_code)
        outer = torch.bmm(f.unsqueeze(2), y_code.unsqueeze(1))  # [B,d,c]
        return self.net(outer.view(f.size(0), -1)).squeeze(-1)

# ------------------ R1 正则（判别器输入梯度） ------------------
def r1_penalty(d_out: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
    grads = torch.autograd.grad(outputs=d_out.sum(), inputs=inputs, create_graph=True, retain_graph=True, only_inputs=True)[0]
    return (grads.pow(2).sum(dim=list(range(1, grads.dim())))).mean()

# ------------------ 训练主逻辑 ------------------
def run_cdanr_bert_for_target(args, device, target_subject: str, source_subjects: List[str]):
    # 数据
    T_loader, Te_loader = build_loader(args.data_root, target_subject, args.subframe, args.normalization, args.miu,
                                       args.batch_size, shuffle=True, drop_last=True)
    S_loader = build_source_loader(args.data_root, source_subjects, args.subframe, args.normalization, args.miu, args.batch_size)

    # Teacher / Student
    # 注意：sEMG_BERT 的构造参数需与你预训练一致（下面仅举例，请按你的模型定义改）
    ms_net = sEMG_BERT(vocab_size=args.subframe, hidden=128, feature_dim=1, n_layers=4, attn_heads=8).to(device)
    ms_net.load_state_dict(load_state_flex(args.pretrained), strict=False)
    for p in ms_net.parameters(): p.requires_grad = False

    nt_net = sEMG_BERT(vocab_size=args.subframe, hidden=128, feature_dim=1, n_layers=4, attn_heads=8).to(device)
    nt_net.load_state_dict(ms_net.state_dict(), strict=False)

    # 冻结骨干，仅训练输出头/adapter（若 BERT 有 adapter，可按名称筛选）
    for p in nt_net.parameters(): p.requires_grad = False
    for name, p in nt_net.named_parameters():
        if ('adapter' in name) or (args.head_name in name) or ('output' in name) or ('fc' in name):
            p.requires_grad = True

    ms = HookedModel(ms_net, head_name=args.head_name).to(device)
    nt = HookedModel(nt_net, head_name=args.head_name).to(device)

    # 探测特征维度
    with torch.no_grad():
        xb, *_ = next(iter(T_loader))
        xb = xb.squeeze(3).to(device).float()
        y_try, f0 = nt.forward_with_features(xb)
        if f0.dim() == 3 and f0.size(1) == 1: f0 = f0.squeeze(1)
        feat_dim = f0.shape[-1]
    y_dim = 10

    # 判别器 & 优化器
    D = CDANRDiscriminator(f_dim=feat_dim, y_dim=y_dim, hidden=args.d_hidden, p=0.2).to(device)
    reg_loss = nn.MSELoss()
    bce_logits = nn.BCEWithLogitsLoss()

    train_params = [p for p in nt_net.parameters() if p.requires_grad]
    opt_g = optim.AdamW(train_params, lr=args.lr, weight_decay=1e-4)
    opt_d = optim.AdamW(D.parameters(), lr=args.lr_d, weight_decay=1e-4)

    # 学习率调度（与原代码风格一致）
    def build_warmup_cosine_scheduler(optimizer, total_epochs: int, warmup_epochs: int, base_lr: float, final_lr_ratio: float = 0.1):
        import math as _m
        min_lr = base_lr * final_lr_ratio
        def set_lr(lr):
            for g in optimizer.param_groups: g['lr'] = lr
        def step(epoch_idx: int):
            if warmup_epochs > 0 and epoch_idx <= warmup_epochs:
                lr = base_lr * (epoch_idx / float(max(1, warmup_epochs)))
            else:
                t = (epoch_idx - max(1, warmup_epochs)) / max(1, total_epochs - max(1, warmup_epochs))
                t = min(max(t, 0.0), 1.0)
                lr = min_lr + 0.5 * (base_lr - min_lr) * (1.0 + _m.cos(_m.pi * t))
            set_lr(lr); return lr
        return step

    sched_g = build_warmup_cosine_scheduler(opt_g, args.epochs, args.warmup_epochs, args.lr, 0.1) if args.sched=='cosine' else None
    sched_d = build_warmup_cosine_scheduler(opt_d, args.epochs, args.warmup_epochs, args.lr_d, 0.1) if args.sched=='cosine' else None

    # EMA（可选）
    class EMA:
        def __init__(self, params, decay=0.999):
            self.params = [p for p in params if p.requires_grad]
            self.shadow = {id(p): p.data.clone() for p in self.params}
            self.back = {}; self.decay = decay
        @torch.no_grad()
        def update(self):
            for p in self.params:
                self.shadow[id(p)].mul_(self.decay).add_(p.data, alpha=1.0-self.decay)
        @torch.no_grad()
        def apply(self):
            self.back.clear()
            for p in self.params:
                pid = id(p); self.back[pid] = p.data.clone(); p.data.copy_(self.shadow[pid])
        @torch.no_grad()
        def restore(self):
            for p in self.params:
                pid = id(p); p.data.copy_(self.back[pid])
            self.back.clear()

    ema = EMA(train_params, decay=args.ema_decay) if args.use_ema else None

    # 日志/保存
    os.makedirs(args.save_dir, exist_ok=True)
    save_dir_one = os.path.join(args.save_dir, f"cdanr_bert_{target_subject}")
    os.makedirs(save_dir_one, exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join(save_dir_one, "tb")) if args.tensorboard else None

    best_score = math.inf if args.select_metric in ['mse','nrmse'] else -math.inf

    # 源域循环器
    def cycle(loader):
        while True:
            for b in loader: yield b
    S_iter = cycle(S_loader)

    for epoch in range(1, args.epochs + 1):
        cur_lr_g = sched_g(epoch) if sched_g is not None else opt_g.param_groups[0]['lr']
        cur_lr_d = sched_d(epoch) if sched_d is not None else opt_d.param_groups[0]['lr']

        nt.train(); D.train()
        total_reg, total_adv, total_d, total_r1 = 0.0, 0.0, 0.0, 0.0
        lam_adv = args.lambda_adv * min(1.0, epoch / float(max(1, args.adv_warmup_epochs)))

        for (xb_t, yb_t, *_) in T_loader:
            # 取源域 batch
            try:
                xb_s, yb_s, *_ = next(S_iter)
            except StopIteration:
                S_iter = cycle(S_loader); xb_s, yb_s, *_ = next(S_iter)

            # to device & 形状
            xb_t = xb_t.squeeze(3).to(device).float()     # [B,T,C]
            yb_t = yb_t.to(device).float().view(yb_t.size(0), -1)  # [B,10]
            xb_s = xb_s.squeeze(3).to(device).float()
            yb_s = yb_s.to(device).float().view(yb_s.size(0), -1)

            # 训练增广（目标域）
            if args.train_noise_std > 0 or args.train_drop_ch > 0:
                xb_t = aug_gaussian_jitter(xb_t, args.train_noise_std)
                xb_t = aug_channel_drop(xb_t, args.train_drop_ch)

            # -------- D step --------
            for p in D.parameters(): p.requires_grad = True
            ms.zero_grad(set_to_none=True); nt.zero_grad(set_to_none=True); D.zero_grad(set_to_none=True)

            # 源特征
            _, f_s = ms.forward_with_features(xb_s)
            if f_s.dim() == 3 and f_s.size(1) == 1: f_s = f_s.squeeze(1)
            # 目标特征 + 预测（用于条件）
            _, f_t = nt.forward_with_features(xb_t)
            if f_t.dim() == 3 and f_t.size(1) == 1: f_t = f_t.squeeze(1)
            with torch.no_grad():
                y_hat_t_cur, _ = nt.forward_with_features(xb_t)
                if isinstance(y_hat_t_cur, (tuple, list)): y_hat_t_cur = y_hat_t_cur[0]
                if y_hat_t_cur.dim() == 3 and y_hat_t_cur.size(1) == 1: y_hat_t_cur = y_hat_t_cur.squeeze(1)

            # R1：需要输入梯度时才 requires_grad
            f_s_rg = f_s.detach().requires_grad_(args.r1_gamma > 0)
            f_t_rg = f_t.detach().requires_grad_(args.r1_gamma > 0)

            logit_s = D(f_s_rg, yb_s)                 # 源：用真值 y_s 作为条件
            logit_t = D(f_t_rg, y_hat_t_cur.detach()) # 目标：用预测 ŷ_t 作为条件

            bce_logits = nn.BCEWithLogitsLoss()
            loss_d = bce_logits(logit_s, torch.ones_like(logit_s)) + \
                     bce_logits(logit_t, torch.zeros_like(logit_t))

            if args.r1_gamma > 0:
                gp_s = r1_penalty(logit_s, f_s_rg)
                gp_t = r1_penalty(logit_t, f_t_rg)
                loss_d_total = loss_d + 0.5 * args.r1_gamma * (gp_s + gp_t)
            else:
                loss_d_total = loss_d

            loss_d_total.backward(); opt_d.step()
            total_d += float(loss_d.item())
            if args.r1_gamma > 0:
                total_r1 += float((0.5 * args.r1_gamma * (gp_s + gp_t)).item())

            # -------- G step（MSE + 对抗）--------
            for p in D.parameters(): p.requires_grad = False

            y_hat_t, f_t = nt.forward_with_features(xb_t)
            if f_t.dim() == 3 and f_t.size(1) == 1: f_t = f_t.squeeze(1)
            if isinstance(y_hat_t, (tuple, list)): y_hat_t = y_hat_t[0]
            if y_hat_t.dim() == 3 and y_hat_t.size(1) == 1: y_hat_t = y_hat_t.squeeze(1)

            L_mse = reg_loss(y_hat_t, yb_t)

            # GRL：外积条件的对抗项
            logit_s_g = D(grl(f_s.detach(), lam_adv), yb_s.detach())
            logit_t_g = D(grl(f_t, lam_adv), y_hat_t.detach())
            L_adv = bce_logits(logit_s_g, torch.ones_like(logit_s_g)) + \
                    bce_logits(logit_t_g, torch.zeros_like(logit_t_g))

            L = L_mse + L_adv
            opt_g.zero_grad(set_to_none=True)
            L.backward()
            nn.utils.clip_grad_norm_(train_params, max_norm=args.max_grad_norm)
            opt_g.step()
            if ema is not None: ema.update()

            total_reg += float(L_mse.item())
            total_adv += float(L_adv.item())

        # -------- 验证 --------
        nt.eval()
        if ema is not None: ema.apply()
        with torch.no_grad():
            val_mse = 0.0
            preds_cpu, targets_cpu = [], []
            for xb, yb, *_ in Te_loader:
                xb = xb.squeeze(3).to(device).float()
                yb = yb.to(device).float().view(yb.size(0), -1)
                y_hat, f = nt.forward_with_features(xb)
                if isinstance(y_hat, (tuple, list)): y_hat = y_hat[0]
                if y_hat.dim() == 3 and y_hat.size(1) == 1: y_hat = y_hat.squeeze(1)
                val_mse += reg_loss(y_hat, yb).item()
                preds_cpu.append(y_hat.detach().cpu()); targets_cpu.append(yb.detach().cpu())
            val_mse /= len(Te_loader)

            try:
                yh_np = torch.cat(preds_cpu, dim=0).numpy().reshape(-1, 10)
                y_np  = torch.cat(targets_cpu, dim=0).numpy().reshape(-1, 10)
                NRMSE, _CC_unused, R2 = compute_metrics_numpy(y_np, yh_np)
                # === 与评估一致：十个关节逐个 Pearson CC，取均值 ===
                cc_list = [pearson_CC(y_np[:, i], yh_np[:, i]) for i in range(10)]
                CC = float(np.mean(cc_list))
            except Exception as e:
                print(f"[Warn] metric computation failed: {e}")
                NRMSE, CC, R2 = float('nan'), float('nan'), float('nan')
        if ema is not None: ema.restore()

        if writer is not None:
            writer.add_scalar("opt/lr_g", cur_lr_g, epoch)
            writer.add_scalar("opt/lr_d", cur_lr_d, epoch)
            writer.add_scalar("loss/train_mse", total_reg/len(T_loader), epoch)
            writer.add_scalar("loss/train_adv", total_adv/len(T_loader), epoch)
            writer.add_scalar("loss/train_D", total_d/len(T_loader), epoch)
            if args.r1_gamma > 0:
                writer.add_scalar("loss/train_R1", total_r1/len(T_loader), epoch)
            writer.add_scalar("loss/val_mse", val_mse, epoch)
            writer.add_scalar("metrics/NRMSE", NRMSE, epoch)
            writer.add_scalar("metrics/CC_mean_per_joint", CC, epoch)
            writer.add_scalar("metrics/R2", R2, epoch)
            writer.flush()

        print(f"[CDANR-BERT {target_subject}] Epoch {epoch:03d}  "
              f"LRg={cur_lr_g:.2e} LRd={cur_lr_d:.2e}  "
              f"train_mse={total_reg/len(T_loader):.6f}  train_adv={total_adv/len(T_loader):.6f}  "
              f"train_D={total_d/len(T_loader):.6f}  "
              f"{'train_R1='+format(total_r1/len(T_loader),'.6f') if args.r1_gamma>0 else ''}  "
              f"Val(MSE)={val_mse:.6f}  NRMSE={NRMSE:.4f}  CC(mean@10)={CC:.4f}  R2={R2:.4f}  "
              f"lam_adv={lam_adv:.3f}")

        # 保存 latest
        torch.save({'epoch': epoch, 'model_state': nt_net.state_dict()},
                   os.path.join(save_dir_one, 'cdanr_latest.pth'))

        # 保存 best（按 select_metric；CC 为“十关节平均 CC”）
        if args.select_metric == 'mse':
            cur_score = val_mse; is_better = cur_score < best_score
        elif args.select_metric == 'nrmse':
            cur_score = NRMSE; is_better = cur_score < best_score
        elif args.select_metric == 'cc':
            cur_score = CC; is_better = cur_score > best_score
        elif args.select_metric == 'r2':
            cur_score = R2; is_better = cur_score > best_score
        else:
            cur_score = val_mse; is_better = cur_score < best_score

        if is_better and not (isinstance(cur_score, float) and (math.isnan(cur_score) or math.isinf(cur_score))):
            best_score = cur_score
            if ema is not None:
                ema.apply()
            try:
                torch.save({'epoch': epoch, 'model_state': nt_net.state_dict(), 'saved_with_ema': bool(ema is not None)},
                           os.path.join(save_dir_one, 'cdanr_best.pth'))
            finally:
                if ema is not None:
                    ema.restore()

    if writer is not None:
        writer.close()

# ------------------ 入口 ------------------
def main():
    ap = argparse.ArgumentParser(description="CDANR calibration on sEMG-BERT (NinaPro)")

    # data
    ap.add_argument('--data_root', type=str, default='../../../../feature/ninapro_db2_trans')
    ap.add_argument('--pretrained', type=str, default='../../result/ninapro/checkpoints_pretrain/BERT/model_best.pth')
    ap.add_argument('--targets', nargs='+', default=[f"S{i}" for i in range(31, 41)])
    ap.add_argument('--source_subjects', nargs='+', default=[f"S{i}" for i in range(1, 31)])
    ap.add_argument('--subframe', type=int, default=200)
    ap.add_argument('--normalization', type=str, default='miu')
    ap.add_argument('--miu', type=float, default=2 ** 20)


    ap.add_argument('--head_name',   type=str, default='fc')  # 抓取“头前特征”的层名

    # train
    ap.add_argument('--epochs', type=int, default=50 )
    ap.add_argument('--batch_size', type=int, default=64)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--lr_d', type=float, default=1e-3)
    ap.add_argument('--max_grad_norm', type=float, default=5.0)
    ap.add_argument('--sched', type=str, default='cosine', choices=['none','cosine'])
    ap.add_argument('--warmup_epochs', type=int, default=3)
    ap.add_argument('--tensorboard', action='store_true')
    ap.add_argument('--save_dir', type=str, default='../result/ninapro/Estimation_result/BERT/checkpoints_cdanr')

    # adversarial
    ap.add_argument('--lambda_adv', type=float, default=0.5)
    ap.add_argument('--adv_warmup_epochs', type=int, default=5)

    # EMA
    ap.add_argument('--use_ema', action='store_true')
    ap.add_argument('--ema_decay', type=float, default=0.999)

    # discriminator
    ap.add_argument('--d_hidden', type=int, default=512)
    ap.add_argument('--r1_gamma', type=float, default=1.0)  # 设 0 即关闭 R1 做消融

    # train-time aug
    ap.add_argument('--train_noise_std', type=float, default=0.01)
    ap.add_argument('--train_drop_ch', type=float, default=0.1)

    # selection metric（默认按 CC；你也可传 --select_metric mse|nrmse|r2）
    ap.add_argument('--select_metric', type=str, default='cc', choices=['mse','nrmse','cc','r2'])

    args = ap.parse_args()
    set_seed(525)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"Pretrained: {args.pretrained}")
    print(f"Targets: {args.targets}")
    print(f"Sources: {args.source_subjects}")
    print(f"Save dir: {args.save_dir}")
    print(f"Select metric: {args.select_metric}")

    for tgt in args.targets:
        print(f"\n====== CDANR-BERT start: {tgt} ======")
        run_cdanr_bert_for_target(args, device, tgt, args.source_subjects)
        print(f"====== CDANR-BERT done : {tgt} ======\n")

if __name__ == '__main__':
    main()
