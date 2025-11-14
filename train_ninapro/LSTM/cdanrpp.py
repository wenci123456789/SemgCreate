# cdanrpp_lstm_fc_plus_last.py
# LSTM 的 CDANR 校准脚本：将“只训 fc”升级为“fc + 最后一层 LSTM”
# - 仅修改：训练范围与模型构造（不再硬编码超参；自动解析最后一层 LSTM）
# - 其它（对抗/GRL/R1/EMA/保存/日志）与原版保持一致

import os, math, argparse, random, re
from typing import Optional, List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
from torch.utils.tensorboard import SummaryWriter

# ===== 项目内依赖（按你的工程结构） =====
from DataProcess import NinaPro
from utils.sEMG_models.sEMG_LSTM import sEMG_LSTM
from utils.Methods.methods import compute_metrics_numpy, pearson_CC

# ================== 工具 ==================
def set_seed(seed: int = 525):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

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

# =============== LSTM 专用 Hook（优先 fc） ===============
class HookedModelLSTM(nn.Module):
    """
    - 优先查找 'fc'（若不存在则回退为模型里最后一个 nn.Linear）
    - 钩取线性层的 **输入**（头前特征），形状通常为 [B,D] 或 [B,T,D]
    - LSTM forward 常返回 (pred, hidden)，其中 pred ≈ [B,1,10]
    """
    def __init__(self, model: nn.Module, head_name: str = 'fc'):
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
                raise RuntimeError("LSTM 主干中未找到任何 nn.Linear 作为输出头。")
            target = last_linear
            self.head_name = last_name
            print(f"[HookedModel-LSTM] 未找到 '{head_name}'，使用最后一个 Linear: '{last_name}'")

        def _hook(mod, fin, fout):
            self._feat = fin[0]  # 线性层输入
        target.register_forward_hook(_hook)

    def forward_with_features(self, x):
        out = self.model(x, None)  # 不在 batch 间传递隐状态
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

# ================== 帮助函数：解析最后一层 LSTM index ==================
def find_last_lstm_layer_index(model: nn.Module) -> int:
    """
    在命名参数中查找 'lstm.(weight_ih|weight_hh|bias_ih|bias_hh)_l{idx}' 的最大 idx。
    找不到则返回 -1。
    """
    pat = re.compile(r"lstm\.(?:weight_ih|weight_hh|bias_ih|bias_hh)_l(\d+)")
    last_idx = -1
    for n, _ in model.named_parameters():
        m = pat.search(n)
        if m:
            idx = int(m.group(1))
            if idx > last_idx:
                last_idx = idx
    return last_idx

def set_train_scope_fc_plus_last_lstm(model: nn.Module):
    """
    先全冻；打开所有 'fc' 参数；再自动打开最后一层 LSTM 的四组参数。
    """
    for _, p in model.named_parameters():
        p.requires_grad = False

    for n, p in model.named_parameters():
        if 'fc' in n:
            p.requires_grad = True

    last_idx = find_last_lstm_layer_index(model)
    hit = 0
    if last_idx >= 0:
        keys = [
            f"lstm.weight_ih_l{last_idx}",
            f"lstm.weight_hh_l{last_idx}",
            f"lstm.bias_ih_l{last_idx}",
            f"lstm.bias_hh_l{last_idx}",
        ]
        for n, p in model.named_parameters():
            if any(k in n for k in keys):
                p.requires_grad = True
                hit += 1
        print(f"[tune_scope] fc + last LSTM layer (l{last_idx}) — trainable params in last layer: {hit}")
    else:
        print("[tune_scope] only fc (未解析到 LSTM 层)")

# ================== 主训练流程（单目标被试） ==================
def run_cdanr_lstm_for_target(args, device, target_subject: str, source_subjects: List[str]):
    # 数据
    T_loader, Te_loader = build_loader(args.data_root, target_subject, args.subframe, args.normalization, args.miu,
                                       args.batch_size, shuffle=True, drop_last=True)
    src_list = [s for s in source_subjects if s != target_subject]
    if len(src_list) == 0:
        raise ValueError("source_subjects 需至少包含一个不同于 target 的被试。")
    S_loader = build_source_loader(args.data_root, src_list, args.subframe, args.normalization, args.miu, args.batch_size)

    # Teacher（冻结）/ Student（待校准）：与预训练同构（不再硬编码）
    ms_net = sEMG_LSTM(vocab_size=200, hidden=128, n_layers=4).to(device)
    ms_net.load_state_dict(load_state_flex(args.pretrained), strict=False)
    for p in ms_net.parameters(): p.requires_grad = False

    nt_net = sEMG_LSTM(vocab_size=200, hidden=128, n_layers=4).to(device)
    nt_net.load_state_dict(ms_net.state_dict(), strict=False)

    # 训练范围：fc + 最后一层 LSTM
    set_train_scope_fc_plus_last_lstm(nt_net)

    ms = HookedModelLSTM(ms_net, head_name=args.head_name).to(device)
    nt = HookedModelLSTM(nt_net, head_name=args.head_name).to(device)

    # 探测特征维度（用 Hook 抓到的特征形状自适配）
    with torch.no_grad():
        xb, *_ = next(iter(T_loader))
        xb = xb.squeeze(3).to(device).float()  # [B,T,12]
        y_try, f0 = nt.forward_with_features(xb)
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
                pid=id(p); self.back[pid]=p.data.clone(); p.data.copy_(self.shadow[pid])
        @torch.no_grad()
        def restore(self):
            for p in self.params:
                pid=id(p); p.data.copy_(self.back[pid])
            self.back.clear()
    ema = EMA(train_params, decay=args.ema_decay) if args.use_ema else None

    # 日志/保存
    os.makedirs(args.save_dir, exist_ok=True)
    save_dir_one = os.path.join(args.save_dir, f"cdanrpp_{target_subject}")
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
            # 源 batch
            try:
                xb_s, yb_s, *_ = next(S_iter)
            except StopIteration:
                S_iter = cycle(S_loader); xb_s, yb_s, *_ = next(S_iter)

            xb_t = xb_t.squeeze(3).to(device).float()  # [B,T,12]
            yb_t = yb_t.to(device).float().view(yb_t.size(0), -1)  # [B,10]
            xb_s = xb_s.squeeze(3).to(device).float()
            yb_s = yb_s.to(device).float().view(yb_s.size(0), -1)  # [B,10]

            # ---------- D step ----------
            for p in D.parameters(): p.requires_grad = True
            ms.zero_grad(set_to_none=True); nt.zero_grad(set_to_none=True); D.zero_grad(set_to_none=True)

            # 源特征
            _, f_s = ms.forward_with_features(xb_s)

            # 目标特征 + 预测
            y_hat_t_cur, f_t = nt.forward_with_features(xb_t)
            if isinstance(y_hat_t_cur, (tuple, list)): y_hat_t_cur = y_hat_t_cur[0]
            if y_hat_t_cur.dim() == 3:  # [B,1,10] -> [B,10]
                y_hat_t_cur = y_hat_t_cur.view(y_hat_t_cur.size(0), -1)

            f_s_rg = f_s.detach().requires_grad_(args.r1_gamma > 0)
            f_t_rg = f_t.detach().requires_grad_(args.r1_gamma > 0)

            logit_s = D(f_s_rg, yb_s)                  # 源：用真值 y_s
            logit_t = D(f_t_rg, y_hat_t_cur.detach())  # 目标：用 ŷ_t

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
            if y_hat_t.dim() == 3:  # [B,1,10] -> [B,10]
                y_hat_t = y_hat_t.view(y_hat_t.size(0), -1)

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
                if y_hat.dim() == 3:  # [B,1,10] -> [B,10]
                    y_hat = y_hat.view(y_hat.size(0), -1)
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

        print(f"[CDANR-LSTM {target_subject}] Epoch {epoch:03d}  "
              f"LRg={cur_lr_g:.2e} LRd={cur_lr_d:.2e}  "
              f"train_mse={total_reg/len(T_loader):.6f}  train_adv={total_adv/len(T_loader):.6f}  "
              f"train_D={total_d/len(T_loader):.6f}  "
              f"{'train_R1='+format(total_r1/len(T_loader),'.6f') if args.r1_gamma>0 else ''}  "
              f"Val(MSE)={val_mse:.6f}  NRMSE={NRMSE:.4f}  CC(mean@10)={CC:.4f}  R2={R2:.4f}  "
              f"lam_adv={lam_adv:.3f}")

        # 保存 latest
        torch.save({'epoch': epoch, 'model_state': nt_net.state_dict()},
                   os.path.join(save_dir_one, 'cdanrpp_latest.pth'))

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
            try:
                torch.save({'epoch': epoch, 'model_state': nt_net.state_dict(), 'saved_with_ema': bool(ema is not None)},
                           os.path.join(save_dir_one, 'cdanrpp_best.pth'))
            finally:
                if ema is not None: ema.restore()

    if writer is not None: writer.close()

# ================== 入口 ==================
def main():
    ap = argparse.ArgumentParser(description="CDANR calibration for LSTM (NinaPro) — fc + last LSTM")
    # data
    ap.add_argument('--data_root', type=str, default='../../../../feature/ninapro_db2_trans')
    ap.add_argument('--pretrained', type=str, default='../../result/ninapro/checkpoints_pretrain/LSTM/model_best.pth')
    ap.add_argument('--targets', nargs='+', default=[f"S{i}" for i in range(31, 41)])
    ap.add_argument('--source_subjects', nargs='+', default=[f"S{i}" for i in range(1, 31)])
    ap.add_argument('--subframe', type=int, default=200)
    ap.add_argument('--normalization', type=str, default='miu')
    ap.add_argument('--miu', type=float, default=2 ** 20)

    # LSTM 结构（与预训练一致）
    ap.add_argument('--hidden', type=int, default=128)
    ap.add_argument('--n_layers', type=int, default=4)
    ap.add_argument('--dropout', type=float, default=0.8)

    # head（LSTM 常用名 fc；不存在时自动回退）
    ap.add_argument('--head_name', type=str, default='fc')

    # train
    ap.add_argument('--epochs', type=int, default=50 )
    ap.add_argument('--batch_size', type=int, default=64)
    ap.add_argument('--lr', type=float, default=3e-4)     # ⚠️ 解冻到 LSTM 一层，默认稍小更稳
    ap.add_argument('--lr_d', type=float, default=1e-3)
    ap.add_argument('--max_grad_norm', type=float, default=5.0)
    ap.add_argument('--sched', type=str, default='cosine', choices=['none','cosine'])
    ap.add_argument('--warmup_epochs', type=int, default=3)
    ap.add_argument('--tensorboard', action='store_true')
    ap.add_argument('--save_dir', type=str, default='../../result/ninapro/Estimation_result/LSTM/checkpoints_cdanrpp')

    # adversarial
    ap.add_argument('--lambda_adv', type=float, default=0.5)
    ap.add_argument('--adv_warmup_epochs', type=int, default=5)

    # EMA
    ap.add_argument('--use_ema', action='store_true')
    ap.add_argument('--ema_decay', type=float, default=0.999)

    # discriminator
    ap.add_argument('--d_hidden', type=int, default=512)
    ap.add_argument('--r1_gamma', type=float, default=1.0)  # 设 0 做“无 R1”消融

    # train-time aug（默认关闭）
    ap.add_argument('--train_noise_std', type=float, default=0.0)
    ap.add_argument('--train_drop_ch', type=float, default=0.0)

    # selection metric（默认按 CC；也可传 mse|nrmse|r2）
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
        print(f"\n====== CDANR-LSTM (fc+last) start: {tgt} ======")
        run_cdanr_lstm_for_target(args, device, tgt, args.source_subjects)
        print(f"====== CDANR-LSTM (fc+last) done : {tgt} ======\n")

if __name__ == '__main__':
    main()
