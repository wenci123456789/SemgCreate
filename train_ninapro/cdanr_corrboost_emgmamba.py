
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CDAN-R++ (EMGMambaAdapter) for NinaPro DB2
------------------------------------------
On top of CDAN-R + CorrBoost, this version adds:
1) CorrBoost warmup: λ_corr linearly ramps up to target value (improves stability & CC).
2) R1 gradient penalty on the domain discriminator (stabilizes D, reduces overfitting).
3) Mild train_ninapro-time augmentation (Gaussian jitter + channel drop) to improve generalization.
4) Test-Time Augmentation (TTA): average predictions over N noisy passes to boost CC.
5) CC-based checkpointing (unchanged).

Usage (example):
    python cdanr_corrboost_emgmamba_plus.py \
        --data_root ../../../feature/ninapro_db2_trans \
        --pretrained ../result/checkpoints_pretrain/model_best.pth \
        --targets S31 S32 S33 S34 S35 S36 S37 S38 S39 S40 \
        --source_subjects S1 S2 ... S30 \
        --epochs 60 --batch_size 64 --lr 1e-3 \
        --lambda-adv 0.5 --lambda-corr 0.6 \
        --corr-warmup-epochs 10 \
        --r1-gamma 1.0 \
        --tta --tta-times 8 --tta-noise-std 0.015 \
        --use_ema --sched cosine --warmup_epochs 3 \
        --select_metric cc --tensorboard

Notes:
- This is drop-in compatible with your project structure (DataProcess, EMGMambaAttentionAdapter, compute_metrics_numpy).
- Only adapter + output head are trainable, like your FT/ATL/COAST baselines.
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
    if y_hat.dim() == 3 and y_hat.size(1) == 1: y_hat = y_hat.squeeze(1)
    if y.dim() == 3 and y.size(1) == 1: y = y.squeeze(1)
    y_hat = y_hat - y_hat.mean(dim=0, keepdim=True)
    y     = y - y.mean(dim=0, keepdim=True)
    num = (y_hat * y).sum(dim=0)
    den = (y_hat.pow(2).sum(dim=0).sqrt() * y.pow(2).sum(dim=0).sqrt() + eps)
    rho = num / den
    return rho.mean()

def corrboost_loss(y_hat: torch.Tensor, y: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """CorrBoost = (1 - ρ) + 0.25*(-atanh(ρ))"""
    if y_hat.dim() == 3 and y_hat.size(1) == 1: y_hat = y_hat.squeeze(1)
    if y.dim() == 3 and y.size(1) == 1: y = y.squeeze(1)
    y_hat_c = y_hat - y_hat.mean(dim=0, keepdim=True)
    y_c     = y - y.mean(dim=0, keepdim=True)
    num = (y_hat_c * y_c).sum(dim=0)
    den = (y_hat_c.pow(2).sum(dim=0).sqrt() * y_c.pow(2).sum(dim=0).sqrt() + eps)
    rho = num / den  # [C]
    one_minus_rho = (1.0 - rho).mean()
    rho_clamped = rho.clamp(-1 + 1e-4, 1 - 1e-4)
    z = 0.5 * torch.log((1 + rho_clamped) / (1 - rho_clamped))
    z_term = (-z).mean()
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

# --------------------- light augmentations ---------------------
def aug_gaussian_jitter(x: torch.Tensor, std: float) -> torch.Tensor:
    if std <= 0: return x
    return x + torch.randn_like(x) * std

def aug_channel_drop(x: torch.Tensor, p: float = 0.1) -> torch.Tensor:
    # x: [B,T,C], randomly drops channels to zero (same mask across time)
    if p <= 0: return x
    B,T,C = x.shape
    mask = (torch.rand(B, C, device=x.device) > p).float()  # 1 keep, 0 drop
    return x * mask.unsqueeze(1)

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
            self._feat = inp[0]  # keep grad for adapters
        target_module.register_forward_hook(_hook)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def forward_with_features(self, x: torch.Tensor):
        y = self.model(x)
        f = self._feat  # [B,d] or [B,1,d]
        return y, f

    def apply_head(self, f: torch.Tensor) -> torch.Tensor:
        return self.head(f)

# --------------------- CDAN-R discriminator with R1 ---------------------
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
        if f.dim() == 3 and f.size(1) == 1: f = f.squeeze(1)
        if y_code.dim() == 3 and y_code.size(1) == 1: y_code = y_code.squeeze(1)
        f = l2n(f); y_code = l2n(y_code)
        outer = torch.bmm(f.unsqueeze(2), y_code.unsqueeze(1))  # [B,d,c]
        return self.net(outer.view(f.size(0), -1)).squeeze(-1)

def r1_penalty(d_out: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
    grads = torch.autograd.grad(outputs=d_out.sum(), inputs=inputs, create_graph=True, retain_graph=True, only_inputs=True)[0]
    return (grads.pow(2).sum(dim=list(range(1, grads.dim())))).mean()

# --------------------- training ---------------------
def run_cdanrpp_for_target(args, device, target_subject: str, source_subjects: List[str]):
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

    # freeze backbone; only adapter + output head train_ninapro
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
    save_dir_one = os.path.join(args.save_dir, f"cdanrpp_{target_subject}")
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
        total_reg, total_corr, total_adv, total_d, total_r1 = 0.0, 0.0, 0.0, 0.0, 0.0

        lam_adv = args.lambda_adv * min(1.0, epoch / float(max(1, args.adv_warmup_epochs)))
        # Corr warmup: linearly ramp from 0 to target
        lam_corr = args.lambda_corr * min(1.0, epoch / float(max(1, args.corr_warmup_epochs)))

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

            # light augmentation on target batch
            if args.train_noise_std > 0 or args.train_drop_ch > 0:
                xb_t = aug_gaussian_jitter(xb_t, args.train_noise_std)
                xb_t = aug_channel_drop(xb_t, args.train_drop_ch)

            # ----------------- D step with R1 -----------------
            for p in D.parameters(): p.requires_grad = True
            # source path
            ms.zero_grad(set_to_none=True); nt.zero_grad(set_to_none=True); D.zero_grad(set_to_none=True)
            _, f_s = ms.forward_with_features(xb_s)
            if f_s.dim() == 3 and f_s.size(1) == 1: f_s = f_s.squeeze(1)
            f_s.requires_grad_(True)
            logit_s = D(f_s, yb_s)
            # target path
            _, f_t = nt.forward_with_features(xb_t)
            if f_t.dim() == 3 and f_t.size(1) == 1: f_t = f_t.squeeze(1)
            with torch.no_grad():
                y_hat_t_cur = nt.apply_head(f_t.detach() if f_t.grad_fn is not None else f_t)
                if y_hat_t_cur.dim() == 3 and y_hat_t_cur.size(1) == 1: y_hat_t_cur = y_hat_t_cur.squeeze(1)
            f_t_detached = f_t.detach().requires_grad_(True)
            logit_t = D(f_t_detached, y_hat_t_cur.detach())
            bce_logits = nn.BCEWithLogitsLoss()
            loss_d = bce_logits(logit_s, torch.ones_like(logit_s)) + bce_logits(logit_t, torch.zeros_like(logit_t))

            # R1 gradient penalty
            gp_s = r1_penalty(logit_s, f_s)
            gp_t = r1_penalty(logit_t, f_t_detached)
            loss_d_total = loss_d + 0.5 * args.r1_gamma * (gp_s + gp_t)

            loss_d_total.backward()
            opt_d.step()
            total_d += float(loss_d.item()); total_r1 += float((0.5 * args.r1_gamma * (gp_s + gp_t)).item())

            # ----------------- G (student) step -----------------
            for p in D.parameters(): p.requires_grad = False  # freeze D for G step
            _, f_t = nt.forward_with_features(xb_t)
            if f_t.dim() == 3 and f_t.size(1) == 1: f_t = f_t.squeeze(1)
            y_hat_t = nt.apply_head(f_t)
            if y_hat_t.dim() == 3 and y_hat_t.size(1) == 1: y_hat_t = y_hat_t.squeeze(1)

            # task: MSE + CorrBoost (with warmup)
            L_mse = reg_loss(y_hat_t, yb_t)
            L_corr = corrboost_loss(y_hat_t, yb_t)
            L_task = L_mse + lam_corr * L_corr

            # adversarial via GRL on both domains (source frozen)
            logit_s_g = D(grl(f_s.detach(), lam_adv), yb_s.detach())
            logit_t_g = D(grl(f_t, lam_adv), y_hat_t.detach())
            L_adv = bce_logits(logit_s_g, torch.ones_like(logit_s_g)) + \
                    bce_logits(logit_t_g, torch.zeros_like(logit_t_g))

            L = L_task + L_adv  # L_adv scaled by GRL

            opt_g.zero_grad(set_to_none=True)
            L.backward()
            nn.utils.clip_grad_norm_(train_params, max_norm=args.max_grad_norm)
            opt_g.step()
            if ema is not None: ema.update()

            total_reg += float(L_mse.item())
            total_corr += float(L_corr.item())
            total_adv += float(L_adv.item())

        # ----- validation -----
        nt.eval()
        if ema is not None: ema.apply()
        with torch.no_grad():
            val_mse = 0.0
            preds_cpu, targets_cpu = [], []
            for xb, yb, *_ in Te_loader:
                xb = xb.squeeze(3).to(device)
                yb = yb.to(device).float().view(yb.size(0), -1)
                # TTA
                if args.tta:
                    yh_sum = 0.0
                    for _ in range(args.tta_times):
                        xb_aug = aug_gaussian_jitter(xb, args.tta_noise_std) if args.tta_noise_std > 0 else xb
                        _, f = nt.forward_with_features(xb_aug)
                        if f.dim() == 3 and f.size(1) == 1: f = f.squeeze(1)
                        yh = nt.apply_head(f)
                        if yh.dim() == 3 and yh.size(1) == 1: yh = yh.squeeze(1)
                        yh_sum = yh_sum + yh
                    y_hat = yh_sum / float(args.tta_times)
                else:
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
            writer.add_scalar("loss/train_R1", total_r1/len(T_loader), epoch)
            writer.add_scalar("loss/val_mse", val_mse, epoch)
            writer.add_scalar("metrics/NRMSE", NRMSE, epoch)
            writer.add_scalar("metrics/CC", CC, epoch)
            writer.add_scalar("metrics/R2", R2, epoch)
            writer.flush()

        print(f"[CDAN-R++ {target_subject}] Epoch {epoch:03d}  "
              f"LRg={cur_lr_g:.2e} LRd={cur_lr_d:.2e}  "
              f"train_mse={total_reg/len(T_loader):.6f}  train_corr={total_corr/len(T_loader):.6f}  "
              f"train_adv={total_adv/len(T_loader):.6f}  train_D={total_d/len(T_loader):.6f}  train_R1={total_r1/len(T_loader):.6f}  "
              f"Val(MSE)={val_mse:.6f}  NRMSE={NRMSE:.4f}  CC={CC:.4f}  R2={R2:.4f}  "
              f"lam_adv={lam_adv:.3f}  lam_corr={lam_corr:.3f}  TTA={int(args.tta)}x{args.tta_times}@{args.tta_noise_std}")

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
                   os.path.join(save_dir_one, 'cdanrpp_latest.pth'))
        if is_better and not (isinstance(cur_score, float) and (math.isnan(cur_score) or math.isinf(cur_score))):
            best_score = cur_score
            torch.save({'epoch': epoch, 'model_state': nt_net.state_dict()},
                       os.path.join(save_dir_one, 'cdanrpp_best.pth'))

    if writer is not None: writer.close()


def main():
    ap = argparse.ArgumentParser(description="CDAN-R++ (EMGMambaAdapter) for NinaPro DB2 cross-subject regression")
    # data
    ap.add_argument('--data_root', type=str, default='../../../feature/ninapro_db2_trans')
    ap.add_argument('--pretrained', type=str, default='../result/ninapro/checkpoints_pretrain/sEMGMamba/model_best.pth')
    ap.add_argument('--targets', nargs='+', default=[f"S{i}" for i in range(31, 41)])
    ap.add_argument('--source_subjects', nargs='+', default=[f"S{i}" for i in range(1, 31)])
    ap.add_argument('--subframe', type=int, default=200)
    ap.add_argument('--normalization', type=str, default='miu')
    ap.add_argument('--miu', type=float, default=2 ** 20)

    # train_ninapro
    ap.add_argument('--epochs', type=int, default=60)
    ap.add_argument('--batch_size', type=int, default=64)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--lr_d', type=float, default=1e-3)
    ap.add_argument('--max_grad_norm', type=float, default=5.0)
    ap.add_argument('--sched', type=str, default='cosine', choices=['none','cosine'])
    ap.add_argument('--warmup_epochs', type=int, default=3)
    ap.add_argument('--tensorboard', action='store_true')
    ap.add_argument('--save_dir', type=str, default='../result/ninapro/Estimation_result/sEMGMamba/checkpoints_cdanrpp')
    ap.add_argument('--head_name', type=str, default='output_proj')

    # losses
    ap.add_argument('--lambda_adv', type=float, default=0.5)
    ap.add_argument('--adv_warmup_epochs', type=int, default=5)
    ap.add_argument('--lambda_corr', type=float, default=0.6)
    ap.add_argument('--corr_warmup_epochs', type=int, default=10)

    # EMA
    ap.add_argument('--use_ema', action='store_true')
    ap.add_argument('--ema_decay', type=float, default=0.999)

    # discriminator
    ap.add_argument('--d_hidden', type=int, default=512)
    ap.add_argument('--r1_gamma', type=float, default=1.0)

    # TTA & train_ninapro-time aug
    ap.add_argument('--tta', action='store_true')
    ap.add_argument('--tta_times', type=int, default=8)
    ap.add_argument('--tta_noise_std', type=float, default=0.015)
    ap.add_argument('--train_noise_std', type=float, default=0.01)
    ap.add_argument('--train_drop_ch', type=float, default=0.1)

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
        print(f"\n====== CDAN-R++ start: {tgt} ======")
        run_cdanrpp_for_target(args, device, tgt, args.source_subjects)
        print(f"====== CDAN-R++ done : {tgt} ======\n")

if __name__ == '__main__':
    main()
