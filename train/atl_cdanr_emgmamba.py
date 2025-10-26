#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
COAST - Conditional Optimal-Transport Adapter (with stronger training options)

在你原脚本基础上增强：
  - 可选对齐：none / KcMMD / OT（Sinkhorn）
  - 可选相关性损失（1-Pearson）
  - 可选 EMA / AMP / 预热+余弦调度 / 梯度裁剪
  - 可指定 Hook 层名（默认 output_proj）
"""

import os, argparse, random
from typing import List, Tuple, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
from torch.utils.tensorboard import SummaryWriter

# === Your project imports ===
from DataProcess import NinaPro
from Model.EMGMambaAttentionAdapter import EMGMambaAdapter
from utils.Methods.methods import compute_metrics_numpy


# ---------------------------------------------------------
# Utility
# ---------------------------------------------------------
def set_seed(seed: int = 2025):
    import numpy as np
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def l2n(z: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """L2 normalize last-dim to stabilize alignment losses."""
    return z / (z.norm(dim=-1, keepdim=True) + eps)


def corr_loss(y_hat: torch.Tensor, y: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    1 - Pearson correlation（对每个输出通道计算相关性，再取均值）
    适配 [B,1,C] 或 [B,C]
    """
    if y_hat.dim() == 3 and y_hat.size(1) == 1:
        y_hat = y_hat.squeeze(1)
    if y.dim() == 3 and y.size(1) == 1:
        y = y.squeeze(1)
    y_hat = y_hat - y_hat.mean(dim=0, keepdim=True)
    y = y - y.mean(dim=0, keepdim=True)
    num = (y_hat * y).sum(dim=0)
    den = (y_hat.pow(2).sum(dim=0).sqrt() * y.pow(2).sum(dim=0).sqrt() + eps)
    corr = num / den  # [C]
    return 1.0 - corr.mean()


class EMA:
    """简洁的参数 EMA，仅在验证时切换，训练期间持续更新 shadow。"""
    def __init__(self, params, decay=0.999):
        self.decay = decay
        self.params = [p for p in params if p.requires_grad]
        self.shadow = {id(p): p.data.clone() for p in self.params}
        self.back = {}

    @torch.no_grad()
    def update(self):
        for p in self.params:
            sid = id(p)
            self.shadow[sid].mul_(self.decay).add_(p.data, alpha=1.0 - self.decay)

    @torch.no_grad()
    def apply(self):
        self.back.clear()
        for p in self.params:
            sid = id(p)
            self.back[sid] = p.data.clone()
            p.data.copy_(self.shadow[sid])

    @torch.no_grad()
    def restore(self):
        for p in self.params:
            sid = id(p)
            p.data.copy_(self.back[sid])
        self.back.clear()


def build_warmup_cosine_scheduler(optimizer, total_epochs: int, warmup_epochs: int, base_lr: float, final_lr_ratio: float = 0.1):
    """线性预热 + 余弦退火（按 epoch 调用 step(epoch) 返回当前 lr）"""
    import math as _m
    min_lr = base_lr * final_lr_ratio

    def set_lr(lr):
        for g in optimizer.param_groups:
            g['lr'] = lr

    def step(epoch_idx: int):
        if warmup_epochs > 0 and epoch_idx <= warmup_epochs:
            lr = base_lr * (epoch_idx / float(max(1, warmup_epochs)))
        else:
            t = (epoch_idx - max(1, warmup_epochs)) / max(1, total_epochs - max(1, warmup_epochs))
            t = min(max(t, 0.0), 1.0)
            lr = min_lr + 0.5 * (base_lr - min_lr) * (1.0 + _m.cos(_m.pi * t))
        set_lr(lr); return lr
    return step


# ---------------------------------------------------------
# Adapter: bottleneck residual MLP
# ---------------------------------------------------------
class BottleneckAdapter(nn.Module):
    def __init__(self, d: int, r: int = 64, p: float = 0.1):
        super().__init__()
        self.down = nn.Linear(d, r, bias=False)
        self.act = nn.ReLU(inplace=True)
        self.up = nn.Linear(r, d, bias=False)
        self.drop = nn.Dropout(p)

    def forward(self, f: torch.Tensor) -> torch.Tensor:
        return f + self.drop(self.up(self.act(self.down(f))))


# ---------------------------------------------------------
# Sinkhorn OT (entropic)
# ---------------------------------------------------------
@torch.no_grad()
def _sinkhorn_scaling(K: torch.Tensor, iters: int = 50):
    # K: [m,n]
    m, n = K.shape
    u = torch.full((m, 1), 1.0 / float(m), device=K.device)
    v = torch.full((n, 1), 1.0 / float(n), device=K.device)
    for _ in range(iters):
        u = 1.0 / (K @ v + 1e-8)
        v = 1.0 / (K.t() @ u + 1e-8)
    return u, v


def sinkhorn_cost(x: torch.Tensor, y: torch.Tensor, eps: float = 0.05, iters: int = 50) -> torch.Tensor:
    """Approximate Sinkhorn distance between two sets: x:[m,d], y:[n,d] -> scalar"""
    if x.dim() > 2: x = x.view(x.size(0), -1)
    if y.dim() > 2: y = y.view(y.size(0), -1)
    C = torch.cdist(x, y, p=2) ** 2  # [m,n]
    K = torch.exp(-C / eps)          # [m,n]
    with torch.no_grad():
        u, v = _sinkhorn_scaling(K, iters)
    P = (u.squeeze(-1).unsqueeze(1) * K) * v.squeeze(-1).unsqueeze(0)
    return torch.sum(P * C)


# ---------------------------------------------------------
# KcMMD (label-conditioned MMD) —— 更稳的对齐项
# ---------------------------------------------------------
def _rbf(x: torch.Tensor, y: torch.Tensor, sigma: float) -> torch.Tensor:
    if x.dim() > 2: x = x.view(x.size(0), -1)
    if y.dim() > 2: y = y.view(y.size(0), -1)
    C = torch.cdist(x, y, p=2) ** 2
    return torch.exp(-C / (2.0 * sigma * sigma))

def kcmmd_loss(f_s, f_t, y_s, y_t, sigma_f=1.0, sigma_y=0.5):
    f_s = l2n(f_s); f_t = l2n(f_t)
    y_s = y_s.view(y_s.size(0), -1); y_t = y_t.view(y_t.size(0), -1)
    k_ss = _rbf(f_s, f_s, sigma_f); k_tt = _rbf(f_t, f_t, sigma_f); k_st = _rbf(f_s, f_t, sigma_f)
    l_ss = _rbf(y_s, y_s, sigma_y); l_tt = _rbf(y_t, y_t, sigma_y); l_st = _rbf(y_s, y_t, sigma_y)
    return (k_ss*l_ss).mean() + (k_tt*l_tt).mean() - 2.0*(k_st*l_st).mean()


# ---------------------------------------------------------
# Label prototypes (KMeans in label space)
# ---------------------------------------------------------
class LabelPrototypes:
    def __init__(self, k: int = 32, iters: int = 20):
        self.k = k
        self.iters = iters
        self.centroids: Optional[torch.Tensor] = None  # [k,C]

    @torch.no_grad()
    def fit(self, Y: torch.Tensor):
        device = Y.device
        N, _ = Y.shape
        idx = torch.randperm(N, device=device)[: self.k]
        centroids = Y[idx].clone()
        for _ in range(self.iters):
            dists = torch.cdist(Y, centroids, p=2)  # [N,k]
            labels = dists.argmin(dim=1)
            for j in range(self.k):
                mask = (labels == j)
                if mask.any():
                    centroids[j] = Y[mask].mean(dim=0)
        self.centroids = centroids

    def assign(self, Y: torch.Tensor) -> torch.Tensor:
        assert self.centroids is not None, "fit() first"
        dists = torch.cdist(Y, self.centroids.to(Y.device), p=2)  # [N,k]
        return dists.argmin(dim=1)


# ---------------------------------------------------------
# Hook to capture features before output head, and reuse the head
# ---------------------------------------------------------
class HookedModelCOAST(nn.Module):
    def __init__(self, model: nn.Module, candidate_head_names=("output_proj",)):
        super().__init__()
        self.model = model
        self._feat: Optional[torch.Tensor] = None
        self.head: Optional[nn.Module] = None
        self.head_name: Optional[str] = None

        # find head module
        for name in candidate_head_names:
            if hasattr(model, name):
                self.head = getattr(model, name)
                self.head_name = name
                break
        if self.head is None:
            last_linear = None
            last_name = None
            for n, m in model.named_modules():
                if isinstance(m, nn.Linear):
                    last_linear = m; last_name = n
            if last_linear is None:
                raise RuntimeError("Could not locate output head; please pass the correct name.")
            self.head = last_linear
            self.head_name = last_name

        # register hook to capture features at head input
        target_module = dict(self.model.named_modules())[self.head_name]

        def _fwd_hook(mod, inp, out):
            # store incoming feature vector of the head (no grad through backbone)
            self._feat = inp[0].detach()

        target_module.register_forward_hook(_fwd_hook)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    @torch.no_grad()
    def forward_with_features(self, x: torch.Tensor):
        y = self.model(x)
        f = self._feat  # [B, d] 或 [B,1,d]
        return y, f

    def apply_head(self, f: torch.Tensor) -> torch.Tensor:
        return self.head(f)


# ---------------------------------------------------------
# Data
# ---------------------------------------------------------
def load_state_flex(ckpt_path: str):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if isinstance(ckpt, dict) and "model_state" in ckpt:
        return ckpt["model_state"]
    if hasattr(ckpt, "state_dict"):
        return ckpt.state_dict()
    return ckpt


def build_loader(root: str, sid: str, subframe: int, normalization: str, mu: float,
                 batch_size: int, shuffle: bool, drop_last: bool) -> Tuple[DataLoader, DataLoader]:
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


def build_source_loader(args) -> DataLoader:
    S_sets = []
    for i, sid in enumerate(args.source_subjects):
        e = os.path.join(args.data_root, f"{sid}_E2_A1_rms_train.h5")
        g = os.path.join(args.data_root, f"{sid}_E2_A1_glove_train.h5")
        S_sets.append(
            NinaPro.NinaPro(
                e, g,
                subframe=args.subframe,
                normalization=args.normalization,
                mu=args.miu,
                dummy_label=i,
                class_num=len(args.source_subjects),
            )
        )
    return DataLoader(
        ConcatDataset(S_sets),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,
        pin_memory=False,
    )


def collect_source_labels(S_loader: DataLoader, max_samples: int = 20000, device: str = "cpu") -> torch.Tensor:
    Y_list = []; seen = 0
    for xb, yb, *_ in S_loader:
        y = yb.to(device).float().view(yb.size(0), -1)  # [B,C]
        Y_list.append(y); seen += y.size(0)
        if seen >= max_samples: break
    return torch.cat(Y_list, dim=0)


def smooth_loss(seq_preds: Optional[torch.Tensor]) -> torch.Tensor:
    """If per-frame predictions available [B,T,C], penalize first-order differences."""
    if seq_preds is None:
        return torch.tensor(0.0, device="cuda" if torch.cuda.is_available() else "cpu")
    dt = seq_preds[:, 1:, :] - seq_preds[:, :-1, :]
    return dt.pow(2).mean()


# ---------------------------------------------------------
# Training — COAST
# ---------------------------------------------------------
def run_coast_for_target(args, device: torch.device, target_subject: str, source_subjects: List[str]):
    # ===== Data =====
    T_loader, Te_loader = build_loader(
        args.data_root, target_subject, args.subframe, args.normalization, args.miu,
        args.batch_size, shuffle=True, drop_last=True
    )
    S_loader = build_source_loader(args)

    # ===== Models: freeze ms_net, clone to nt_net =====
    ms_net = EMGMambaAdapter(input_dim=12, output_dim=10)
    ms_net.load_state_dict(load_state_flex(args.pretrained), strict=False)
    for p in ms_net.parameters(): p.requires_grad = False

    nt_net = EMGMambaAdapter(input_dim=12, output_dim=10)
    nt_net.load_state_dict(ms_net.state_dict(), strict=False)

    # 容量对齐：只训练 nt_net 内部 adapter 与 output_proj
    for p in nt_net.parameters(): p.requires_grad = False
    for name, p in nt_net.named_parameters():
        if ('adapter' in name) or ('output_proj' in name):
            p.requires_grad = True

    # 构建 Hook（支持自定义 head 名）
    ms = HookedModelCOAST(ms_net, candidate_head_names=(args.head_name, 'output_proj')).to(device)
    nt = HookedModelCOAST(nt_net, candidate_head_names=(args.head_name, 'output_proj')).to(device)

    # 探测 head 输入维度
    with torch.no_grad():
        xb, *_ = next(iter(T_loader))
        xb = xb.squeeze(3).to(device)
        _, f0 = nt.forward_with_features(xb)  # [B,d] 或 [B,1,d]
        if f0.dim() == 3 and f0.size(1) == 1: f0 = f0.squeeze(1)
        feat_dim = f0.shape[-1]

    # 额外 Adapter（我们的方法）
    adapter = BottleneckAdapter(d=feat_dim, r=max(8, feat_dim // 16), p=0.1).to(device)

    # ===== Label prototypes from sources =====（OT 需要；KcMMD 不强依赖）
    proto = LabelPrototypes(k=args.k_proto, iters=args.kmeans_iters)
    Y_src = collect_source_labels(S_loader, max_samples=args.max_proto_samples, device=device)
    proto.fit(Y_src)

    # ===== Losses & Optimizer / Scheduler / EMA / AMP =====
    reg_loss = nn.MSELoss()
    train_params = list(adapter.parameters()) + [p for p in nt_net.parameters() if p.requires_grad]
    opt = optim.AdamW(train_params, lr=args.lr, weight_decay=1e-4)

    sched = build_warmup_cosine_scheduler(opt, args.epochs, args.warmup_epochs, args.lr, final_lr_ratio=0.1) \
            if args.sched == 'cosine' else None
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)
    ema = EMA(train_params, decay=args.ema_decay) if args.use_ema else None

    writer = SummaryWriter(log_dir=os.path.join(args.save_dir, f"tb_{target_subject}")) if args.tensorboard else None

    best_score = float("inf")
    os.makedirs(args.save_dir, exist_ok=True)
    save_path = os.path.join(args.save_dir, f"COAST_{target_subject}.pt")

    # Helper: cycle source loader
    def cycle(loader):
        while True:
            for batch in loader:
                yield batch
    S_iter = cycle(S_loader)

    for epoch in range(1, args.epochs + 1):
        if sched is not None:
            cur_lr = sched(epoch)
        else:
            cur_lr = opt.param_groups[0]['lr']

        nt.train(); adapter.train()
        total_reg, total_align, total_sm = 0.0, 0.0, 0.0

        # 对齐权重热身：前 N 个 epoch 线性从 0 → lambda_ot
        lam_align = args.lambda_ot * (min(1.0, epoch / float(max(1, args.ot_warmup_epochs))))

        for xb_t, yb_t, *_ in T_loader:
            xb_t = xb_t.squeeze(3).to(device)
            yb_t = yb_t.to(device).float().view(yb_t.size(0), -1)  # [B,10]

            # Source batch for对齐
            xb_s, yb_s, *_ = next(S_iter)
            xb_s = xb_s.squeeze(3).to(device)
            yb_s = yb_s.to(device).float().view(yb_s.size(0), -1)

            with torch.cuda.amp.autocast(enabled=args.amp):
                # Forward: target
                _, f_t = nt.forward_with_features(xb_t)
                if f_t.dim() == 3 and f_t.size(1) == 1: f_t = f_t.squeeze(1)
                f_t_adapt = adapter(f_t)
                y_hat_t = nt.apply_head(f_t_adapt)
                if y_hat_t.dim() == 3 and y_hat_t.size(1) == 1: y_hat_t = y_hat_t.squeeze(1)

                # Forward: source (no adapter)
                with torch.no_grad():
                    _, f_s = ms.forward_with_features(xb_s)
                    if f_s.dim() == 3 and f_s.size(1) == 1: f_s = f_s.squeeze(1)

                # 监督回归（MSE + 可选 Corr）
                L_reg = reg_loss(y_hat_t, yb_t)
                if args.lambda_corr > 0:
                    L_reg = L_reg + args.lambda_corr * corr_loss(y_hat_t, yb_t)

                # 对齐项
                if args.align == 'ot':
                    y_assign_t = proto.assign(y_hat_t.detach() if args.use_pseudo else yb_t)
                    y_assign_s = proto.assign(yb_s)
                    L_align = 0.0
                    shared_ids = set(y_assign_s.unique().tolist()) & set(y_assign_t.unique().tolist())
                    for k in shared_ids:
                        ms_mask = (y_assign_s == k); mt_mask = (y_assign_t == k)
                        if ms_mask.any() and mt_mask.any():
                            fs_k = l2n(f_s[ms_mask]); ft_k = l2n(f_t_adapt[mt_mask])
                            L_align = L_align + sinkhorn_cost(fs_k, ft_k, eps=args.ot_eps, iters=args.ot_iters)
                elif args.align == 'kcmmd':
                    y_t_for_align = (y_hat_t.detach() if args.use_pseudo else yb_t)
                    L_align = kcmmd_loss(f_s, f_t_adapt, yb_s, y_t_for_align,
                                         sigma_f=args.kcmmd_sigma_f, sigma_y=args.kcmmd_sigma_y)
                else:
                    L_align = 0.0

                # 可选时序平滑（这里只是占位，保持与你原逻辑一致）
                L_sm = smooth_loss(None)

                L = L_reg + lam_align * (L_align if isinstance(L_align, torch.Tensor) else torch.tensor(L_align, device=device)) \
                          + args.lambda_sm * L_sm

            opt.zero_grad(set_to_none=True)
            scaler.scale(L).backward()
            if args.max_grad_norm > 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(train_params, max_norm=args.max_grad_norm)
            scaler.step(opt)
            scaler.update()
            if ema is not None: ema.update()

            total_reg += float(L_reg.detach().item())
            total_align += float(L_align.detach().item() if isinstance(L_align, torch.Tensor) else L_align)
            total_sm += float(L_sm.detach().item() if isinstance(L_sm, torch.Tensor) else L_sm)

        # ===== Validation on target test split（使用 EMA 权重评估） =====
        nt.eval(); adapter.eval()
        if ema is not None: ema.apply()
        with torch.no_grad():
            val_mse = 0.0
            preds_cpu, targets_cpu = [], []
            for xb, yb, *_ in Te_loader:
                xb = xb.squeeze(3).to(device)
                yb = yb.to(device).float().view(yb.size(0), -1)
                _, f = nt.forward_with_features(xb)
                if f.dim() == 3 and f.size(1) == 1: f = f.squeeze(1)
                y_hat = nt.apply_head(adapter(f))
                if y_hat.dim() == 3 and y_hat.size(1) == 1: y_hat = y_hat.squeeze(1)
                val_mse += nn.MSELoss()(y_hat, yb).item()
                preds_cpu.append(y_hat.detach().cpu())
                targets_cpu.append(yb.detach().cpu())
            val_mse /= len(Te_loader)
            try:
                yh_np = torch.cat(preds_cpu, dim=0).numpy().reshape(-1, 10)
                y_np  = torch.cat(targets_cpu, dim=0).numpy().reshape(-1, 10)
                NRMSE, CC, R2 = compute_metrics_numpy(y_np, yh_np)
            except Exception as e:
                print(f"[Warn] metric computation failed: {e}")
                NRMSE, CC, R2 = float('nan'), float('nan'), float('nan')
        if ema is not None: ema.restore()

        if writer is not None:
            writer.add_scalar("opt/lr", cur_lr, epoch)
            writer.add_scalar("loss/train_reg", total_reg/len(T_loader), epoch)
            writer.add_scalar("loss/train_align", total_align/len(T_loader), epoch)
            writer.add_scalar("loss/train_smooth", total_sm/len(T_loader), epoch)
            writer.add_scalar("loss/val_mse", val_mse, epoch)
            writer.add_scalar("metrics/NRMSE", NRMSE, epoch)
            writer.add_scalar("metrics/CC", CC, epoch)
            writer.add_scalar("metrics/R2", R2, epoch)
            writer.flush()

        print(
            f"[COAST {target_subject}] Epoch {epoch:03d}  "
            f"LR={cur_lr:.2e}  train_reg={total_reg/len(T_loader):.6f}  "
            f"train_align={total_align/len(T_loader):.6f}  train_sm={total_sm/len(T_loader):.6f}  "
            f"Val(MSE)={val_mse:.6f}  NRMSE={NRMSE:.4f}  CC={CC:.4f}  R2={R2:.4f}  "
            f"lam_align={lam_align:.3f}  align={args.align}"
        )

        # Save best (by Val MSE)
        if val_mse < best_score:
            best_score = val_mse
            torch.save(
                {
                    "adapter_state": adapter.state_dict(),
                    "nt_state": nt_net.state_dict(),
                    "nt_head_name": nt.head_name,
                    "feat_dim": feat_dim,
                    "args": vars(args),
                },
                save_path,
            )

    if writer is not None: writer.close()
    print(f"[COAST {target_subject}] best Val(MSE)={best_score:.6f}; saved to {save_path}")


# ---------------------------------------------------------
# CLI
# ---------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="COAST: Conditional Adapter for cross-subject sEMG regression (enhanced)")
    parser.add_argument('--data_root', type=str, default='../../../feature/ninapro_db2_trans')
    parser.add_argument('--pretrained', type=str, default='../result/checkpoints_pretrain/model_best.pth')
    parser.add_argument('--save_dir', type=str, default='../result/check/checkpoints_atl')
    parser.add_argument('--targets', nargs='+', default=[f"S{i}" for i in range(31, 41)])
    parser.add_argument('--source_subjects', nargs='+', default=[f"S{i}" for i in range(1, 31)])

    # Data settings
    parser.add_argument('--subframe', type=int, default=200)
    parser.add_argument('--normalization', type=str, default='miu')
    parser.add_argument('--miu', type=float, default=2 ** 20)

    # Training
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--tensorboard', action='store_true')

    # Head/Hook
    parser.add_argument('--head-name', type=str, default='output_proj',
                        help='module name to hook as the output head input')

    # Alignment options
    parser.add_argument('--align', type=str, default='none', choices=['none','kcmmd','ot'],
                        help='choose alignment regularizer')
    parser.add_argument('--kcmmd-sigma-f', type=float, default=1.0)
    parser.add_argument('--kcmmd-sigma-y', type=float, default=0.5)

    # Label prototypes for OT
    parser.add_argument('--k-proto', type=int, default=16)
    parser.add_argument('--kmeans-iters', type=int, default=25)
    parser.add_argument('--max-proto-samples', type=int, default=20000)
    parser.add_argument('--use-pseudo', action='store_true',
                        help='use y_hat (pseudo) for target prototype assignment (for OT/KcMMD)')

    # Loss weights
    parser.add_argument('--lambda-ot', type=float, default=0.0, help='alignment weight (used for both OT/KcMMD)')
    parser.add_argument('--ot-eps', type=float, default=0.07)
    parser.add_argument('--ot-iters', type=int, default=50)
    parser.add_argument('--ot-warmup-epochs', type=int, default=5)
    parser.add_argument('--lambda-sm', type=float, default=0.0)
    parser.add_argument('--lambda-corr', type=float, default=0.2, help='(1-Pearson) correlation loss weight')

    # Stability & speed
    parser.add_argument('--sched', type=str, default='cosine', choices=['none','cosine'])
    parser.add_argument('--warmup-epochs', type=int, default=3)
    parser.add_argument('--amp', action='store_true')
    parser.add_argument('--max-grad-norm', type=float, default=5.0)
    parser.add_argument('--use-ema', action='store_true')
    parser.add_argument('--ema-decay', type=float, default=0.999)

    args = parser.parse_args()

    set_seed(2025)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"Pretrained: {args.pretrained}")
    print(f"Targets: {args.targets}")
    print(f"Sources: {args.source_subjects}")
    print(f"Save dir: {args.save_dir}")
    print(f"Align: {args.align}, lambda_align={args.lambda_ot}, head={args.head_name}")

    for tgt in args.targets:
        print(f"\n====== COAST start: {tgt} ======")
        run_coast_for_target(args, device, tgt, args.source_subjects)
        print(f"====== COAST done : {tgt} ======\n")


if __name__ == '__main__':
    main()
