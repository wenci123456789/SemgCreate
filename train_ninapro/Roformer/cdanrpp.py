# cdanrpp_roformer.py
# 仅针对 RoFormer 的 CDANR 校准脚本（修复 [B,T,10] vs [B,10] 维度问题）
# - Backbone: RoFormerEMG(12->10, d_model=120, num_layers=2, num_heads=5, use_mu_law=False)
# - 判别器：CDANR 外积条件 + GRL + 对抗 warm-up + 可选 R1
# - 输出：按时间维均值池化到 [B,10] 后计算 MSE 与作为条件输入判别器
# - 指标：NRMSE / mean(CC per joint) / R2；best 以 mean-CC 选择
# - 无 CorrBoost、无 TTA

import os, math, argparse, random
from typing import Optional, List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
from torch.utils.tensorboard import SummaryWriter

# ===== 项目内依赖（按你的工程结构） =====
from DataProcess import NinaPro
from utils.sEMG_models.sEMG_RoFormer import RoFormerEMG
from utils.Methods.methods import compute_metrics_numpy, pearson_CC

# ================== 工具 ==================
def set_seed(seed: int = 525):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def l2n(z: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return z / (z.norm(dim=-1, keepdim=True) + eps)

def load_state_flex(path: str):
    obj = torch.load(path, map_location='cpu')
    if isinstance(obj, dict) and 'model_state' in obj:
        return obj['model_state']
    if hasattr(obj, 'state_dict'):
        return obj.state_dict()
    return obj

# ================== GRL ==================
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

# =============== 训练端轻增广（可选） ===============
def aug_gaussian_jitter(x: torch.Tensor, std: float) -> torch.Tensor:
    if std <= 0: return x
    return x + torch.randn_like(x) * std

def aug_channel_drop(x: torch.Tensor, p: float = 0.1) -> torch.Tensor:
    if p <= 0: return x
    B, T, C = x.shape
    mask = (torch.rand(B, C, device=x.device) > p).float()
    return x * mask.unsqueeze(1)

# ================== 数据构建 ==================
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

def build_source_loader(root: str, subjects: List[str], subframe: int, normalization: str, mu: float, batch_size: int):
    S_sets = []
    for i, sid in enumerate(subjects):
        e = os.path.join(root, f"{sid}_E2_A1_rms_train.h5")
        g = os.path.join(root, f"{sid}_E2_A1_glove_train.h5")
        S_sets.append(
            NinaPro.NinaPro(e, g, subframe=subframe, normalization=normalization, mu=mu,
                            dummy_label=i, class_num=len(subjects))
        )
    return DataLoader(ConcatDataset(S_sets), batch_size=batch_size, shuffle=True, drop_last=True,
                      num_workers=0, pin_memory=False)

# =============== RoFormer 专用 Hook（优先 regressor） ===============
class HookedModel(nn.Module):
    """
    RoFormer 专用：
    - 优先查找 'regressor'（若不存在则回退为模型里最后一个 nn.Linear）
    - 钩取线性层的 **输入**（头前特征）
    """
    def __init__(self, model: nn.Module, head_name: str = 'regressor'):
        super().__init__()
        self.model = model
        self._feat: Optional[torch.Tensor] = None

        named = dict(self.model.named_modules())
        if head_name in named:
            target = named[head_name]
            self.head_name = head_name
        else:
            last_linear, last_name = None, None
            for n, m in self.model.named_modules():
                if isinstance(m, nn.Linear):
                    last_linear, last_name = m, n
            if last_linear is None:
                raise RuntimeError("RoFormer 主干中未找到任何 nn.Linear 作为输出头。")
            target = last_linear
            self.head_name = last_name
            print(f"[HookedModel-RoFormer] 未找到 '{head_name}'，使用最后一个 Linear: '{last_name}'")

        def _hook(mod, fin, fout):
            self._feat = fin[0]  # 线性层输入 [B, D] 或 [B,T,D]
        target.register_forward_hook(_hook)

    def forward(self, x):  # RoFormerEMG 直接返回 y_hat（可能是 [B,T,10]）
        return self.model(x)

    def forward_with_features(self, x):
        out = self.model(x)
        y = out[0] if isinstance(out, (tuple, list)) else out
        return y, self._feat

# ================== 判别器（CDAN 外积条件，自适配形状） ==================
class CDANRDiscriminator(nn.Module):
    def __init__(self, f_dim: int, y_dim: int, hidden: int = 512, p: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(f_dim * y_dim, hidden), nn.ReLU(inplace=True), nn.Dropout(p),
            nn.Linear(hidden, hidden // 2), nn.ReLU(inplace=True), nn.Dropout(p),
            nn.Linear(hidden // 2, 1)
        )

    @staticmethod
    def _to_BD(x: torch.Tensor) -> torch.Tensor:
        """
        将任意形状规整到 [B, D]:
        - [B,D] -> [B,D]
        - [B,T,D] -> 对 T 做均值池化 -> [B,D]
        - 其它 -> [B, -1]
        """
        if x.dim() == 2:
            return x
        if x.dim() == 3:
            return x.mean(dim=1)
        return x.view(x.size(0), -1)

    def forward(self, f: torch.Tensor, y_code: torch.Tensor) -> torch.Tensor:
        f = self._to_BD(f)
        y_code = self._to_BD(y_code)
        f = f / (f.norm(dim=-1, keepdim=True) + 1e-6)
        y_code = y_code / (y_code.norm(dim=-1, keepdim=True) + 1e-6)
        outer = torch.bmm(f.unsqueeze(2), y_code.unsqueeze(1))  # [B,D,1]x[B,1,C]->[B,D,C]
        return self.net(outer.view(outer.size(0), -1)).squeeze(-1)

# ================== R1（可选，--r1_gamma 控制） ==================
def r1_penalty(d_out: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
    grads = torch.autograd.grad(outputs=d_out.sum(), inputs=inputs, create_graph=True, retain_graph=True, only_inputs=True)[0]
    return (grads.pow(2).sum(dim=list(range(1, grads.dim())))).mean()

# ================== 主训练流程（单目标被试） ==================
def run_cdanr_roformer_for_target(args, device, target_subject: str, source_subjects: List[str]):
    # 数据
    T_loader, Te_loader = build_loader(args.data_root, target_subject, args.subframe, args.normalization, args.miu,
                                       args.batch_size, shuffle=True, drop_last=True)
    S_loader = build_source_loader(args.data_root, source_subjects, args.subframe, args.normalization, args.miu, args.batch_size)

    # Teacher（冻结）/ Student（待校准）：严格同构于预训练
    ms_net = RoFormerEMG(input_dim=12, output_dim=10, d_model=120, num_layers=2, num_heads=5, use_mu_law=False).to(device)
    ms_net.load_state_dict(load_state_flex(args.pretrained), strict=False)
    for p in ms_net.parameters(): p.requires_grad = False

    nt_net = RoFormerEMG(input_dim=12, output_dim=10, d_model=120, num_layers=2, num_heads=5, use_mu_law=False).to(device)
    nt_net.load_state_dict(ms_net.state_dict(), strict=False)

    # 只训练输出相关参数（RoFormer 没 adapter 就训练 regressor/最后线性层/输出头）
    for n, p in nt_net.named_parameters():
        p.requires_grad = False
    for n, p in nt_net.named_parameters():
        if ('adapter' in n) or ('output' in n) or ('regressor' in n) or ('fc' in n):
            p.requires_grad = True

    ms = HookedModel(ms_net, head_name=args.head_name).to(device)
    nt = HookedModel(nt_net, head_name=args.head_name).to(device)

    # 探测特征维度（用 Hook 抓到的特征形状自适配）
    with torch.no_grad():
        xb, *_ = next(iter(T_loader))
        xb = xb.squeeze(3).to(device).float()
        y_try, f0 = nt.forward_with_features(xb)
        # 判别器内部会自适配形状，这里仅用于估算 feat_dim
        if f0.dim() == 3:
            feat_dim = f0.size(-1)
        elif f0.dim() == 2:
            feat_dim = f0.size(-1)
        else:
            feat_dim = int(np.prod(f0.shape[1:]))
    y_dim = 10

    # 判别器与优化器
    D = CDANRDiscriminator(f_dim=feat_dim, y_dim=y_dim, hidden=args.d_hidden, p=0.2).to(device)
    reg_loss = nn.MSELoss()
    bce_logits = nn.BCEWithLogitsLoss()

    train_params = [p for p in nt_net.parameters() if p.requires_grad]
    opt_g = optim.AdamW(train_params, lr=args.lr, weight_decay=1e-4)
    opt_d = optim.AdamW(D.parameters(), lr=args.lr_d, weight_decay=1e-4)

    # 学习率调度（余弦 + warmup）
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
    save_dir_one = os.path.join(args.save_dir, f"cdanrpp_{target_subject}")
    os.makedirs(save_dir_one, exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join(save_dir_one, "tb")) if args.tensorboard else None

    # best 选择逻辑（默认按 CC；CC 定义为“十关节均值”）
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
            # 源 batch
            try:
                xb_s, yb_s, *_ = next(S_iter)
            except StopIteration:
                S_iter = cycle(S_loader); xb_s, yb_s, *_ = next(S_iter)

            xb_t = xb_t.squeeze(3).to(device).float()
            yb_t = yb_t.to(device).float().view(yb_t.size(0), -1)  # [B,10]
            xb_s = xb_s.squeeze(3).to(device).float()
            yb_s = yb_s.to(device).float().view(yb_s.size(0), -1)  # [B,10]

            # 目标域增广（可选）
            if args.train_noise_std > 0 or args.train_drop_ch > 0:
                xb_t = aug_gaussian_jitter(xb_t, args.train_noise_std)
                xb_t = aug_channel_drop(xb_t, args.train_drop_ch)

            # ---------- D step ----------
            for p in D.parameters(): p.requires_grad = True
            ms.zero_grad(set_to_none=True); nt.zero_grad(set_to_none=True); D.zero_grad(set_to_none=True)

            # 源特征
            _, f_s = ms.forward_with_features(xb_s)

            # 目标特征 + 预测（条件：池化到 [B,10]）
            y_hat_t_cur, f_t = nt.forward_with_features(xb_t)
            if isinstance(y_hat_t_cur, (tuple, list)): y_hat_t_cur = y_hat_t_cur[0]
            if y_hat_t_cur.dim() == 3:  # [B,T,10] -> [B,10]
                y_hat_t_cur = y_hat_t_cur.mean(dim=1)

            f_s_rg = f_s.detach().requires_grad_(args.r1_gamma > 0)
            f_t_rg = f_t.detach().requires_grad_(args.r1_gamma > 0)

            logit_s = D(f_s_rg, yb_s)                  # 源：用真值 y_s
            logit_t = D(f_t_rg, y_hat_t_cur.detach())  # 目标：用池化后的 ŷ_t

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

            # ---------- G step（MSE + 对抗） ----------
            for p in D.parameters(): p.requires_grad = False

            y_hat_t, f_t = nt.forward_with_features(xb_t)
            if isinstance(y_hat_t, (tuple, list)): y_hat_t = y_hat_t[0]
            if y_hat_t.dim() == 3:  # [B,T,10] -> [B,10]
                y_hat_t = y_hat_t.mean(dim=1)

            L_mse = reg_loss(y_hat_t, yb_t)

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

        # ---------- 验证 ----------
        nt.eval()
        if ema is not None: ema.apply()
        with torch.no_grad():
            val_mse = 0.0
            preds_cpu, targets_cpu = [], []
            for xb, yb, *_ in Te_loader:
                xb = xb.squeeze(3).to(device).float()
                yb = yb.to(device).float().view(yb.size(0), -1)  # [B,10]
                y_hat, f = nt.forward_with_features(xb)
                if isinstance(y_hat, (tuple, list)): y_hat = y_hat[0]
                if y_hat.dim() == 3:  # [B,T,10] -> [B,10]
                    y_hat = y_hat.mean(dim=1)
                val_mse += reg_loss(y_hat, yb).item()
                preds_cpu.append(y_hat.detach().cpu()); targets_cpu.append(yb.detach().cpu())
            val_mse /= len(Te_loader)

            try:
                yh_np = torch.cat(preds_cpu, dim=0).numpy().reshape(-1, 10)
                y_np  = torch.cat(targets_cpu, dim=0).numpy().reshape(-1, 10)
                NRMSE, _CC_unused, R2 = compute_metrics_numpy(y_np, yh_np)
                # 十关节逐个 Pearson CC，取均值（与评估一致）
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

        print(f"[CDANR-RoFormer {target_subject}] Epoch {epoch:03d}  "
              f"LRg={cur_lr_g:.2e} LRd={cur_lr_d:.2e}  "
              f"train_mse={total_reg/len(T_loader):.6f}  train_adv={total_adv/len(T_loader):.6f}  "
              f"train_D={total_d/len(T_loader):.6f}  "
              f"{'train_R1='+format(total_r1/len(T_loader),'.6f') if args.r1_gamma>0 else ''}  "
              f"Val(MSE)={val_mse:.6f}  NRMSE={NRMSE:.4f}  CC(mean@10)={CC:.4f}  R2={R2:.4f}  "
              f"lam_adv={lam_adv:.3f}")

        # 保存 latest
        # torch.save({'epoch': epoch, 'model_state': nt_net.state_dict()},
        #            os.path.join(save_dir_one, 'cdanrpp_latest.pth'))

        # 保存 best（按 --select_metric；CC 为十关节均值）
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
            if ema is not None: ema.apply()
            # try:
            #     torch.save({'epoch': epoch, 'model_state': nt_net.state_dict(), 'saved_with_ema': bool(ema is not None)},
            #                os.path.join(save_dir_one, 'cdanrpp_best.pth'))
            # finally:
            #     if ema is not None: ema.restore()

    if writer is not None: writer.close()

# ================== 入口 ==================
def main():
    ap = argparse.ArgumentParser(description="CDANR calibration for RoFormer (NinaPro)")

    # data
    ap.add_argument('--data_root', type=str, default='../../../../feature/ninapro_db2_trans')
    ap.add_argument('--pretrained', type=str, default='../../result/ninapro/checkpoints_pretrain/Roformer/model_best.pth')
    ap.add_argument('--targets', nargs='+', default=[f"S{i}" for i in range(31, 41)])
    ap.add_argument('--source_subjects', nargs='+', default=[f"S{i}" for i in range(1, 31)])
    ap.add_argument('--subframe', type=int, default=200)
    ap.add_argument('--normalization', type=str, default='miu')
    ap.add_argument('--miu', type=float, default=2 ** 20)

    # head（RoFormer 常用名 regressor；不存在时自动回退）
    ap.add_argument('--head_name', type=str, default='regressor')

    # train
    ap.add_argument('--epochs', type=int, default=50 )
    ap.add_argument('--batch_size', type=int, default=64)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--lr_d', type=float, default=1e-3)
    ap.add_argument('--max_grad_norm', type=float, default=5.0)
    ap.add_argument('--sched', type=str, default='cosine', choices=['none','cosine'])
    ap.add_argument('--warmup_epochs', type=int, default=3)
    ap.add_argument('--tensorboard', action='store_true')
    ap.add_argument('--save_dir', type=str, default='../../result/ninapro/Estimation_result/Roformer/checkpoints_cdanrpp')

    # adversarial
    ap.add_argument('--lambda_adv', type=float, default=0.5)
    ap.add_argument('--adv_warmup_epochs', type=int, default=5)

    # EMA
    ap.add_argument('--use_ema', action='store_true')
    ap.add_argument('--ema_decay', type=float, default=0.999)

    # discriminator
    ap.add_argument('--d_hidden', type=int, default=512)
    ap.add_argument('--r1_gamma', type=float, default=1.0)  # 设 0 做“无 R1”消融

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
        print(f"\n====== CDANR-RoFormer start: {tgt} ======")
        run_cdanr_roformer_for_target(args, device, tgt, args.source_subjects)
        print(f"====== CDANR-RoFormer done : {tgt} ======\n")

if __name__ == '__main__':
    main()
