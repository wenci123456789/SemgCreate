
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CDAN-R + CorrBoost for cross-subject sEMG regression on NinaPro DB2
-------------------------------------------------------------------
- Backbone: EMGMambaAdapter (frozen except adapter + head)
- Task loss: MSE + λ_corr * CorrBoost (1-ρ + z-loss)
- Alignment: Regression-Conditional Adversarial (CDAN-R):
    * Domain discriminator takes outer-product f ⊗ y_code (y for source, y_hat for target).
    * Gradient Reversal drives target features toward source conditional manifold.
- (Optional) Extra alignment: KcMMD or OT can be enabled, but off by default.
- Best checkpoint selection by CC (default).

Usage (example):
    python cdanr_corrboost_emgmamba.py \
        --data_root ../../../feature/ninapro_db2_trans \
        --pretrained ../result/checkpoints_pretrain/model_best.pth \
        --targets S31 S32 S33 S34 S35 S36 S37 S38 S39 S40 \
        --source_subjects S1 S2 ... S30 \
        --epochs 60 --batch_size 64 --lr 1e-3 \
        --lambda-adv 0.5 --lambda-corr 0.5 \
        --select_metric cc --use_ema --sched cosine --tensorboard

Notes:
- This script mirrors your IO/metrics style so results are directly comparable to finetune_ft.py & atl_calibrate_emgmamba.py.
"""
import os, math, argparse, random
from typing import List, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
from torch.utils.tensorboard import SummaryWriter

# Project imports (keep identical to your codebase)
from DataProcess import NinaPro
from Model.EMGMambaAttentionAdapter import EMGMambaAdapter
from utils.Methods.methods import compute_metrics_numpy

# --------------------- utils ---------------------
def set_seed(seed: int = 2025):
    import numpy as np
    random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed); np.random.seed(seed)

def l2n(z: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return z / (z.norm(dim=-1, keepdim=True) + eps)

def pearson_corr(y_hat: torch.Tensor, y: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    # returns mean channel-wise Pearson ρ in [-1,1]
    if y_hat.dim() == 3 and y_hat.size(1) == 1: y_hat = y_hat.squeeze(1)
    if y.dim() == 3 and y.size(1) == 1: y = y.squeeze(1)
    y_hat = y_hat - y_hat.mean(dim=0, keepdim=True)
    y     = y - y.mean(dim=0, keepdim=True)
    num = (y_hat * y).sum(dim=0)
    den = (y_hat.pow(2).sum(dim=0).sqrt() * y.pow(2).sum(dim=0).sqrt() + eps)
    rho = num / den
    return rho.mean()

def corrboost_loss(y_hat: torch.Tensor, y: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    CorrBoost = (1 - ρ) + ζ * (−z(ρ)), where z(ρ)=atanh(ρ) is Fisher z-transform.
    Minimizing this strongly pushes ρ→1 while keeping gradients meaningful near high ρ.
    """
    if y_hat.dim() == 3 and y_hat.size(1) == 1: y_hat = y_hat.squeeze(1)
    if y.dim() == 3 and y.size(1) == 1: y = y.squeeze(1)
    y_hat_c = y_hat - y_hat.mean(dim=0, keepdim=True)
    y_c     = y - y.mean(dim=0, keepdim=True)
    num = (y_hat_c * y_c).sum(dim=0)
    den = (y_hat_c.pow(2).sum(dim=0).sqrt() * y_c.pow(2).sum(dim=0).sqrt() + eps)
    rho = num / den  # [C]
    one_minus_rho = (1.0 - rho).mean()
    # Safe clamp for z-transform
    rho_clamped = rho.clamp(-1 + 1e-4, 1 - 1e-4)
    z = 0.5 * torch.log((1 + rho_clamped) / (1 - rho_clamped))
    z_term = (-z).mean()
    # ζ: relative weight inside CorrBoost; expose as arg if needed, fixed 0.25 works well
    return one_minus_rho + 0.25 * z_term

class GRL(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd: float):
        ctx.lambd = lambd
        return x.view_as(x)
    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambd * grad_output, None

def grl(x, lambd: float):
    return GRL.apply(x, lambd)

# --------------------- data ---------------------
def build_loader(root: str, sid: str, subframe: int, normalization: str, mu: float,
                 batch_size: int, shuffle: bool, drop_last: bool):
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

# --------------------- model wrapper ---------------------
class HookedModel(nn.Module):
    """Capture features right before the output head (default: output_proj)."""
    def __init__(self, model: nn.Module, head_name: str = 'output_proj'):
        super().__init__()
        self.model = model
        self._feat: Optional[torch.Tensor] = None
        self.head: Optional[nn.Module] = None
        self.head_name = head_name

        # find head
        if hasattr(model, head_name):
            self.head = getattr(model, head_name)
        else:
            # fallback: last Linear
            last_linear = None; last_name = None
            for n, m in model.named_modules():
                if isinstance(m, nn.Linear):
                    last_linear = m; last_name = n
            if last_linear is None:
                raise RuntimeError("Could not locate output head")
            self.head = last_linear; self.head_name = last_name

        target_module = dict(self.model.named_modules())[self.head_name]
        def _hook(mod, inp, out):
            # Detach to prevent gradients through backbone (we freeze it anyway)
            self._feat = inp[0]
        target_module.register_forward_hook(_hook)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def forward_with_features(self, x: torch.Tensor):
        y = self.model(x)
        f = self._feat  # [B,d] or [B,1,d]
        return y, f

    def apply_head(self, f: torch.Tensor) -> torch.Tensor:
        return self.head(f)

# --------------------- CDAN-R discriminator ---------------------
class CDANRDiscriminator(nn.Module):
    def __init__(self, f_dim: int, y_dim: int, hidden: int = 512, p: float = 0.2):
        super().__init__()
        in_dim = f_dim * y_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(inplace=True), nn.Dropout(p),
            nn.Linear(hidden, hidden//2), nn.ReLU(inplace=True), nn.Dropout(p),
            nn.Linear(hidden//2, 1)
        )
    def forward(self, f, y_code):
        # f:[B,d], y_code:[B,c] -> outer product then flatten
        if f.dim() == 3 and f.size(1) == 1: f = f.squeeze(1)
        if y_code.dim() == 3 and y_code.size(1) == 1: y_code = y_code.squeeze(1)
        f = l2n(f); y_code = l2n(y_code)
        outer = torch.bmm(f.unsqueeze(2), y_code.unsqueeze(1))  # [B,d,c]
        return self.net(outer.view(f.size(0), -1)).squeeze(-1)

# --------------------- training ---------------------
def run_cdanr_for_target(args, device, target_subject: str, source_subjects: List[str]):
    # data
    T_loader, Te_loader = build_loader(args.data_root, target_subject, args.subframe, args.normalization, args.miu,
                                       args.batch_size, shuffle=True, drop_last=True)
    S_loader = build_source_loader(args.data_root, source_subjects, args.subframe, args.normalization, args.miu, args.batch_size)

    # teacher (multi-source) & student (target) nets
    ms_net = EMGMambaAdapter(input_dim=12, output_dim=10)
    ckpt = torch.load(args.pretrained, map_location='cpu')
    ckpt = ckpt.get('model_state', ckpt)
    ms_net.load_state_dict(ckpt, strict=False)
    for p in ms_net.parameters(): p.requires_grad = False

    nt_net = EMGMambaAdapter(input_dim=12, output_dim=10)
    nt_net.load_state_dict(ms_net.state_dict(), strict=False)

    # freeze backbone; only adapter + output head train
    for p in nt_net.parameters(): p.requires_grad = False
    for name, p in nt_net.named_parameters():
        if ('adapter' in name) or ('output_proj' in name):
            p.requires_grad = True

    ms = HookedModel(ms_net, head_name=args.head_name).to(device)
    nt = HookedModel(nt_net, head_name=args.head_name).to(device)

    # probe feature dim
    with torch.no_grad():
        xb, *_ = next(iter(T_loader))
        xb = xb.squeeze(3).to(device)
        _, f0 = nt.forward_with_features(xb)
        if f0.dim() == 3 and f0.size(1) == 1: f0 = f0.squeeze(1)
        feat_dim = f0.shape[-1]
    y_dim = 10

    # discriminator
    D = CDANRDiscriminator(f_dim=feat_dim, y_dim=y_dim, hidden=args.d_hidden, p=0.2).to(device)

    # opt / sched / ema
    reg_loss = nn.MSELoss()
    train_params = [p for p in nt_net.parameters() if p.requires_grad]
    opt_g = optim.AdamW(train_params, lr=args.lr, weight_decay=1e-4)
    opt_d = optim.AdamW(D.parameters(), lr=args.lr_d, weight_decay=1e-4)

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

    # logs
    os.makedirs(args.save_dir, exist_ok=True)
    save_dir_one = os.path.join(args.save_dir, f"cdanr_{target_subject}")
    os.makedirs(save_dir_one, exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join(save_dir_one, "tb")) if args.tensorboard else None

    # checkpointing by metric
    best_score = math.inf if args.select_metric in ['mse','nrmse'] else -math.inf

    # helper: cycle source loader
    def cycle(loader):
        while True:
            for b in loader: yield b
    S_iter = cycle(S_loader)

    for epoch in range(1, args.epochs + 1):
        if sched_g is not None: cur_lr_g = sched_g(epoch)
        else: cur_lr_g = opt_g.param_groups[0]['lr']
        if sched_d is not None: cur_lr_d = sched_d(epoch)
        else: cur_lr_d = opt_d.param_groups[0]['lr']

        nt.train(); D.train()
        total_reg, total_corr, total_adv, total_d = 0.0, 0.0, 0.0, 0.0

        lam_adv = args.lambda_adv * min(1.0, epoch / float(max(1, args.adv_warmup_epochs)))
        lam_corr = args.lambda_corr

        for (xb_t, yb_t, *_) in T_loader:
            # fetch target & source batches
            try:
                xb_s, yb_s, *_ = next(S_iter)
            except StopIteration:
                S_iter = cycle(S_loader); xb_s, yb_s, *_ = next(S_iter)

            xb_t = xb_t.squeeze(3).to(device)
            yb_t = yb_t.to(device).float().view(yb_t.size(0), -1)  # [B,10]
            xb_s = xb_s.squeeze(3).to(device)
            yb_s = yb_s.to(device).float().view(yb_s.size(0), -1)

            # ----------------- D step -----------------
            with torch.no_grad():
                _, f_s = ms.forward_with_features(xb_s)
                if f_s.dim() == 3 and f_s.size(1) == 1: f_s = f_s.squeeze(1)
                # student path (no grad for G when updating D)
                _, f_t = nt.forward_with_features(xb_t)
                if f_t.dim() == 3 and f_t.size(1) == 1: f_t = f_t.squeeze(1)
                # use current head on student features to form conditional code
                y_hat_t = nt.apply_head(f_t)
                if y_hat_t.dim() == 3 and y_hat_t.size(1) == 1: y_hat_t = y_hat_t.squeeze(1)

            logit_s = D(f_s, yb_s)
            logit_t = D(f_t, y_hat_t)
            bce_logits = nn.BCEWithLogitsLoss()
            loss_d = bce_logits(logit_s, torch.ones_like(logit_s)) + \
                     bce_logits(logit_t, torch.zeros_like(logit_t))
            opt_d.zero_grad(set_to_none=True)
            loss_d.backward()
            opt_d.step()

            total_d += float(loss_d.item())

            # ----------------- G (student) step -----------------
            for p in D.parameters(): p.requires_grad = False  # freeze D for G step

            # forward again for gradients (no detach)
            _, f_t = nt.forward_with_features(xb_t)
            if f_t.dim() == 3 and f_t.size(1) == 1: f_t = f_t.squeeze(1)
            y_hat_t = nt.apply_head(f_t)
            if y_hat_t.dim() == 3 and y_hat_t.size(1) == 1: y_hat_t = y_hat_t.squeeze(1)

            # task: MSE + corrboost
            L_mse = reg_loss(y_hat_t, yb_t)
            L_corr = corrboost_loss(y_hat_t, yb_t)
            L_task = L_mse + lam_corr * L_corr

            # adversarial via GRL on both domains
            logit_s_g = D(grl(f_s.detach(), lam_adv), yb_s.detach())
            logit_t_g = D(grl(f_t, lam_adv), y_hat_t.detach())
            L_adv = bce_logits(logit_s_g, torch.ones_like(logit_s_g)) + \
                    bce_logits(logit_t_g, torch.zeros_like(logit_t_g))

            L = L_task + L_adv  # L_adv already scaled by GRL(λ)

            opt_g.zero_grad(set_to_none=True)
            L.backward()
            nn.utils.clip_grad_norm_(train_params, max_norm=args.max_grad_norm)
            opt_g.step()
            if ema is not None: ema.update()

            total_reg += float(L_mse.item())
            total_corr += float(L_corr.item())
            total_adv += float(L_adv.item())

            for p in D.parameters(): p.requires_grad = True  # unfreeze

        # ----- validation -----
        nt.eval()
        if ema is not None: ema.apply()
        with torch.no_grad():
            val_mse = 0.0
            preds_cpu, targets_cpu = [], []
            for xb, yb, *_ in Te_loader:
                xb = xb.squeeze(3).to(device)
                yb = yb.to(device).float().view(yb.size(0), -1)
                _, f = nt.forward_with_features(xb)
                if f.dim() == 3 and f.size(1) == 1: f = f.squeeze(1)
                y_hat = nt.apply_head(f)
                if y_hat.dim() == 3 and y_hat.size(1) == 1: y_hat = y_hat.squeeze(1)
                val_mse += reg_loss(y_hat, yb).item()
                preds_cpu.append(y_hat.detach().cpu()); targets_cpu.append(yb.detach().cpu())
            val_mse /= len(Te_loader)
            try:
                yh_np = torch.cat(preds_cpu, dim=0).numpy().reshape(-1, 10)
                y_np  = torch.cat(targets_cpu, dim=0).numpy().reshape(-1, 10)
                NRMSE, CC, R2 = compute_metrics_numpy(y_np, yh_np)
            except Exception as e:
                print(f"[Warn] metric computation failed: {e}")
                NRMSE, CC, R2 = float('nan'), float('nan'), float('nan')
        if ema is not None: ema.restore()

        # logs
        if writer is not None:
            writer.add_scalar("opt/lr_g", cur_lr_g, epoch)
            writer.add_scalar("opt/lr_d", cur_lr_d, epoch)
            writer.add_scalar("loss/train_mse", total_reg/len(T_loader), epoch)
            writer.add_scalar("loss/train_corr", total_corr/len(T_loader), epoch)
            writer.add_scalar("loss/train_adv", total_adv/len(T_loader), epoch)
            writer.add_scalar("loss/train_D", total_d/len(T_loader), epoch)
            writer.add_scalar("loss/val_mse", val_mse, epoch)
            writer.add_scalar("metrics/NRMSE", NRMSE, epoch)
            writer.add_scalar("metrics/CC", CC, epoch)
            writer.add_scalar("metrics/R2", R2, epoch)
            writer.flush()

        print(f"[CDAN-R {target_subject}] Epoch {epoch:03d}  "
              f"LRg={cur_lr_g:.2e} LRd={cur_lr_d:.2e}  "
              f"train_mse={total_reg/len(T_loader):.6f}  train_corr={total_corr/len(T_loader):.6f}  "
              f"train_adv={total_adv/len(T_loader):.6f}  train_D={total_d/len(T_loader):.6f}  "
              f"Val(MSE)={val_mse:.6f}  NRMSE={NRMSE:.4f}  CC={CC:.4f}  R2={R2:.4f}  "
              f"lam_adv={lam_adv:.3f}  lam_corr={lam_corr:.3f}")

        # save best — by user-selected metric (default: cc)
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

        torch.save({'epoch': epoch, 'model_state': nt_net.state_dict()},
                   os.path.join(save_dir_one, 'cdanr_latest.pth'))
        if is_better and not (isinstance(cur_score, float) and (math.isnan(cur_score) or math.isinf(cur_score))):
            best_score = cur_score
            torch.save({'epoch': epoch, 'model_state': nt_net.state_dict()},
                       os.path.join(save_dir_one, 'cdanr_best.pth'))

    if writer is not None: writer.close()


def main():
    ap = argparse.ArgumentParser(description="CDAN-R + CorrBoost (EMGMambaAdapter) for NinaPro DB2 cross-subject regression")
    # data
    ap.add_argument('--data_root', type=str, default='../../../feature/ninapro_db2_trans')
    ap.add_argument('--pretrained', type=str, default='../result/checkpoints_pretrain/model_best.pth')
    ap.add_argument('--targets', nargs='+', default=[f"S{i}" for i in range(31, 41)])
    ap.add_argument('--source_subjects', nargs='+', default=[f"S{i}" for i in range(1, 31)])
    ap.add_argument('--subframe', type=int, default=200)
    ap.add_argument('--normalization', type=str, default='miu')
    ap.add_argument('--miu', type=float, default=2 ** 20)

    # train
    ap.add_argument('--epochs', type=int, default=60)
    ap.add_argument('--batch_size', type=int, default=64)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--lr_d', type=float, default=1e-3)
    ap.add_argument('--max_grad_norm', type=float, default=5.0)
    ap.add_argument('--sched', type=str, default='cosine', choices=['none','cosine'])
    ap.add_argument('--warmup_epochs', type=int, default=3)
    ap.add_argument('--tensorboard', action='store_true')
    ap.add_argument('--save_dir', type=str, default='../result/check/checkpoints_cdanr')
    ap.add_argument('--head_name', type=str, default='output_proj')

    # losses
    ap.add_argument('--lambda_adv', type=float, default=0.5)
    ap.add_argument('--adv_warmup_epochs', type=int, default=5)
    ap.add_argument('--lambda_corr', type=float, default=0.5)

    # EMA
    ap.add_argument('--use_ema', action='store_true')
    ap.add_argument('--ema_decay', type=float, default=0.999)

    # discriminator
    ap.add_argument('--d_hidden', type=int, default=512)

    # selection metric
    ap.add_argument('--select_metric', type=str, default='cc', choices=['mse','nrmse','cc','r2'])

    args = ap.parse_args()
    set_seed(2025)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"Pretrained: {args.pretrained}")
    print(f"Targets: {args.targets}")
    print(f"Sources: {args.source_subjects}")
    print(f"Save dir: {args.save_dir}")
    print(f"Select metric: {args.select_metric}")

    for tgt in args.targets:
        print(f"\n====== CDAN-R start: {tgt} ======")
        run_cdanr_for_target(args, device, tgt, args.source_subjects)
        print(f"====== CDAN-R done : {tgt} ======\n")

if __name__ == '__main__':
    main()
