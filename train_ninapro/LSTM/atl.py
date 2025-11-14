# atl_lstm.py
# 仅用于 LSTM 的 ATL 校准脚本（对齐原 RoFormer-ATL 流程）
# - Backbone: sEMG_LSTM(vocab_size=subframe, input_channels=12, hidden, n_layers, dropout)
# - ATL: 判别器只看“头前特征”（hook 到 fc 的输入），将目标特征混淆为“源”
# - LSTM 输出 [B,1,10]；训练/评估时展平为 [B,10]
# - 指标：NRMSE / CC / R2；best 默认按 CC（十关节均值）保存

import os, math, argparse, csv, random
from typing import Optional, List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
from torch.utils.tensorboard import SummaryWriter

# ===== 项目内依赖 =====
from DataProcess import NinaPro
from utils.Methods.methods import compute_metrics_numpy, pearson_CC
from utils.sEMG_models.sEMG_LSTM import sEMG_LSTM

# ---------------- 工具函数 ----------------
def set_seed(seed: int = 525):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def load_state_flex(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location='cpu')
    if isinstance(ckpt, dict) and 'model_state' in ckpt:
        return ckpt['model_state']
    if hasattr(ckpt, 'state_dict'):
        return ckpt.state_dict()
    return ckpt

def build_loader(root, sid, subframe, normalization, mu, batch_size, shuffle, drop_last):
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

# ---------------- 域判别器（ATL；仅看特征） ----------------
class DomainDiscriminator(nn.Module):
    """
    只对齐“头前特征”（不做外积条件）。
    兼容 [B,D] / [B,T,D]：若有时间维，做均值池化到 [B,D] 再判别。
    """
    def __init__(self, in_dim: int, hidden: int = 256, p: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(inplace=True), nn.Dropout(p),
            nn.Linear(hidden, 1)
        )

    @staticmethod
    def _to_BD(x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            return x
        if x.dim() == 3:
            return x.mean(dim=1)  # [B,T,D] -> [B,D]
        return x.view(x.size(0), -1)

    def forward(self, f: torch.Tensor) -> torch.Tensor:
        f = self._to_BD(f)
        return self.net(f).squeeze(-1)

# ---------------- Hook：抓“输出头”输入（头前特征） ----------------
class HookedModelLSTM(nn.Module):
    """
    LSTM 专用 Hook：
    - 优先找 'fc'；找不到就回退为模型里最后一个 nn.Linear
    - 钩的是该线性层的“输入张量”（一般 [B,D] 或 [B,T,D]）
    - LSTM forward 通常返回 (pred, hidden)，pred 形如 [B,1,10]
    """
    def __init__(self, backbone: nn.Module, head_name: str = "fc"):
        super().__init__()
        self.backbone = backbone
        self._feat: Optional[torch.Tensor] = None

        named = dict(self.backbone.named_modules())
        if head_name in named:
            target = named[head_name]; self.head_name = head_name
        else:
            last_linear, last_name = None, None
            for n, m in self.backbone.named_modules():
                if isinstance(m, nn.Linear):
                    last_linear, last_name = m, n
            if last_linear is None:
                raise RuntimeError("LSTM 主干中未找到任何 nn.Linear 作为输出头。")
            target = last_linear; self.head_name = last_name
            print(f"[HookedModel-LSTM] 未找到 '{head_name}'，使用最后一个 Linear: '{last_name}'")

        def _hook(mod, fin, fout):
            self._feat = fin[0]  # 线性层的输入
        target.register_forward_hook(_hook)

    def forward_with_features(self, x: torch.Tensor):
        # LSTM forward 可能需要 hidden；这里不跨 batch 传递，置 None
        out = self.backbone(x, None)
        # 兼容 (pred, hidden) 或仅 pred
        if isinstance(out, (tuple, list)):
            y = out[0]
        else:
            y = out
        return y, self._feat

# ---------------- 主训练（单目标被试） ----------------
def run_atl_lstm_for_target(args, device, target_subject: str, source_subjects: List[str]):
    # 目标域/测试加载
    T_loader, Te_loader = build_loader(args.data_root, target_subject,
                                       args.subframe, args.normalization, args.miu,
                                       args.batch_size, shuffle=True, drop_last=True)
    # 源域（排除目标）
    src_list = [s for s in source_subjects if s != target_subject]
    if len(src_list) == 0:
        raise ValueError("source_subjects 为空或仅包含 target。请提供至少一个不同的源被试。")
    S_loader = build_source_loader(args.data_root, src_list, args.subframe, args.normalization, args.miu, args.batch_size)

    # Teacher（冻结）/ Student（待校准）：与预训练同构
    ms_net = sEMG_LSTM(vocab_size=200, hidden=128, n_layers=4).to(device)
    ms_net.load_state_dict(load_state_flex(args.pretrained), strict=False)
    for p in ms_net.parameters(): p.requires_grad = False

    nt_net = sEMG_LSTM(vocab_size=200, hidden=128, n_layers=4).to(device)
    nt_net.load_state_dict(ms_net.state_dict(), strict=False)

    # 只训练输出层（回归头）
    for n, p in nt_net.named_parameters():
        p.requires_grad = ('fc' in n)

    ms = HookedModelLSTM(ms_net, head_name=args.head_name).to(device)
    nt = HookedModelLSTM(nt_net, head_name=args.head_name).to(device)

    # 判别器输入维度（用一次正向探测）
    with torch.no_grad():
        xb, *_ = next(iter(T_loader))
        xb = xb.squeeze(3).to(device).float()   # [B,T,12]
        y_try, f = nt.forward_with_features(xb)
        feat_dim = f.size(-1) if f.dim() >= 2 else int(np.prod(f.shape[1:]))
    D = DomainDiscriminator(in_dim=feat_dim, hidden=args.d_hidden, p=0.2).to(device)

    # 损失 & 优化器
    reg_loss = nn.MSELoss()
    bce_logits = nn.BCEWithLogitsLoss()
    opt_t = optim.AdamW([p for p in nt_net.parameters() if p.requires_grad], lr=args.lr_t, weight_decay=1e-4)
    opt_d = optim.AdamW(D.parameters(), lr=args.lr_d, weight_decay=1e-4)

    # 日志与保存
    save_dir_one = os.path.join(args.save_dir, f"atl_{target_subject}")
    os.makedirs(save_dir_one, exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join(save_dir_one, "tb")) if args.tensorboard else None

    csv_path = os.path.join(save_dir_one, "history.csv")
    need_header = not os.path.exists(csv_path)
    csv_f = open(csv_path, "a", newline="")
    csv_w = csv.writer(csv_f)
    if need_header:
        csv_w.writerow(["epoch","train_reg","train_D","val_mse","nrmse","cc","r2","select_metric","score","is_best"])

    best_score = -math.inf if args.select_metric == 'cc' else math.inf

    # 训练循环
    it_s = iter(S_loader)
    for epoch in range(1, args.epochs + 1):
        nt.train(); D.train(); ms_net.eval()
        total_reg, total_d = 0.0, 0.0

        for batch_t in T_loader:
            x_t = batch_t[0].squeeze(3).to(device).float()            # [B,T,12]
            y_t = batch_t[1].to(device).float().view(x_t.size(0), -1) # [B,10]

            try:
                batch_s = next(it_s)
            except StopIteration:
                it_s = iter(S_loader); batch_s = next(it_s)
            x_s = batch_s[0].squeeze(3).to(device).float()

            # --------- D step：源（teacher 特征） vs 目标（student 特征）---------
            with torch.no_grad():
                _, f_s = ms.forward_with_features(x_s)  # teacher: 源
                _, f_t = nt.forward_with_features(x_t)  # student: 目标
                f_s = f_s.detach(); f_t = f_t.detach()
            logit_s, logit_t = D(f_s), D(f_t)
            loss_d = bce_logits(logit_s, torch.ones_like(logit_s)) + \
                     bce_logits(logit_t, torch.zeros_like(logit_t))
            opt_d.zero_grad(); loss_d.backward(); opt_d.step()

            # --------- G step：任务 + 混淆 D（把目标特征判成源=1）---------
            y_hat, f_t = nt.forward_with_features(x_t)     # y_hat: [B,1,10]
            if isinstance(y_hat, (tuple, list)): y_hat = y_hat[0]
            if y_hat.dim() == 3:
                y_hat = y_hat.view(y_hat.size(0), -1)      # [B,10]

            loss_map = bce_logits(D(f_t), torch.ones_like(logit_t))   # 目标特征标 1
            loss_reg = reg_loss(y_hat, y_t)
            loss_tot = loss_reg + args.lambda_adv * loss_map

            opt_t.zero_grad(); loss_tot.backward()
            if args.max_grad_norm > 0:
                nn.utils.clip_grad_norm_([p for p in nt_net.parameters() if p.requires_grad],
                                         max_norm=args.max_grad_norm)
            opt_t.step()

            total_reg += loss_reg.item(); total_d += loss_d.item()

        # --------- 验证 ---------
        nt.eval()
        val_mse, n_batch = 0.0, 0
        preds_cpu, targets_cpu = [], []
        with torch.no_grad():
            for batch in Te_loader:
                x = batch[0].squeeze(3).to(device).float()
                y = batch[1].to(device).float().view(x.size(0), -1)  # [B,10]
                y_hat, _ = nt.forward_with_features(x)
                if isinstance(y_hat, (tuple, list)): y_hat = y_hat[0]
                if y_hat.dim() == 3:
                    y_hat = y_hat.view(y_hat.size(0), -1)             # [B,10]
                val_mse += reg_loss(y_hat, y).item(); n_batch += 1
                preds_cpu.append(y_hat.detach().cpu()); targets_cpu.append(y.detach().cpu())
        val_mse /= max(1, n_batch)

        try:
            yh_np = torch.cat(preds_cpu, dim=0).numpy().reshape(-1, 10)
            y_np  = torch.cat(targets_cpu, dim=0).numpy().reshape(-1, 10)
            # 与原口径一致：NRMSE, CC(整体), R2
            NRMSE, CC_all, R2 = compute_metrics_numpy(y_np, yh_np)
            # 额外：十关节逐个 Pearson CC 求均值（用于保存 best 时的 'cc' 指标）
            cc_list = [pearson_CC(y_np[:, i], yh_np[:, i]) for i in range(10)]
            CC_mean10 = float(np.mean(cc_list))
            CC_for_select = CC_mean10 if args.select_metric == 'cc' else CC_all
        except Exception as e:
            print(f("[Warn] metric computation failed: {e}"))
            NRMSE, CC_all, R2 = float('nan'), float('nan'), float('nan')
            CC_mean10 = CC_for_select = float('nan')

        # 日志
        if writer is not None:
            writer.add_scalar("loss/train_reg", total_reg/len(T_loader), epoch)
            writer.add_scalar("loss/train_D",   total_d/len(T_loader), epoch)
            writer.add_scalar("loss/val_mse",   val_mse, epoch)
            writer.add_scalar("metrics/NRMSE",  NRMSE, epoch)
            writer.add_scalar("metrics/CC_all", CC_all, epoch)
            writer.add_scalar("metrics/CC_mean10", CC_mean10, epoch)
            writer.add_scalar("metrics/R2",     R2, epoch)
            writer.flush()

        print(f"[ATL-LSTM {target_subject}] Epoch {epoch:03d}  "
              f"train_reg={total_reg/len(T_loader):.6f}  train_D={total_d/len(T_loader):.6f}  "
              f"Val(MSE)={val_mse:.6f}  NRMSE={NRMSE:.4f}  CC_all={CC_all:.4f}  CC(mean@10)={CC_mean10:.4f}  R2={R2:.4f}")

        # 保存 latest
        torch.save({'epoch': epoch, 'model_state': nt_net.state_dict()},
                   os.path.join(save_dir_one, 'atl_latest.pth'))

        # 选择 best & 保存
        if args.select_metric == 'mse':
            cur_score = val_mse; is_better = cur_score < best_score
        elif args.select_metric == 'nrmse':
            cur_score = NRMSE;   is_better = cur_score < best_score
        elif args.select_metric == 'cc':
            cur_score = CC_for_select; is_better = cur_score > best_score
        elif args.select_metric == 'r2':
            cur_score = R2;      is_better = cur_score > best_score
        else:
            cur_score = val_mse; is_better = cur_score < best_score

        if is_better and not (isinstance(cur_score, float) and (math.isnan(cur_score) or math.isinf(cur_score))):
            best_score = cur_score
            torch.save({'epoch': epoch, 'model_state': nt_net.state_dict()},
                       os.path.join(save_dir_one, 'atl_best.pth'))

        csv_w.writerow([epoch, total_reg/len(T_loader), total_d/len(T_loader),
                        val_mse, NRMSE, CC_all if args.select_metric!='cc' else CC_mean10, R2,
                        args.select_metric, cur_score, int(is_better)])
        csv_f.flush()
        torch.cuda.empty_cache()

    if writer is not None: writer.close()
    csv_f.close()

# ---------------- 入口 ----------------
def main():
    ap = argparse.ArgumentParser(description="ATL calibrate — LSTM backbone (NinaPro)")
    # data
    ap.add_argument('--data_root', type=str, default='../../../../feature/ninapro_db2_trans')
    ap.add_argument('--pretrained', type=str, default='../../result/ninapro/checkpoints_pretrain/LSTM/model_best.pth')
    ap.add_argument('--save_dir', type=str, default='../../result/ninapro/Estimation_result/LSTM/checkpoints_atl')
    ap.add_argument('--targets', nargs='+', default=[f"S{i}" for i in range(31, 41)])
    ap.add_argument('--source_subjects', nargs='+', default=[f"S{i}" for i in range(1, 31)])
    ap.add_argument('--subframe', type=int, default=200)
    ap.add_argument('--normalization', type=str, default='miu')
    ap.add_argument('--miu', type=int, default=2 ** 20)
    ap.add_argument('--d_hidden', type=int, default=256)

    # LSTM 结构（与预训练一致）
    ap.add_argument('--hidden', type=int, default=128)
    ap.add_argument('--n_layers', type=int, default=4)
    ap.add_argument('--dropout', type=float, default=0.8)

    # train
    ap.add_argument('--epochs', type=int, default=50)
    ap.add_argument('--batch_size', type=int, default=64)
    ap.add_argument('--lr_t', type=float, default=1e-3)
    ap.add_argument('--lr_d', type=float, default=1e-3)
    ap.add_argument('--lambda_adv', type=float, default=1.0)
    ap.add_argument('--max_grad_norm', type=float, default=5.0)
    ap.add_argument('--tensorboard', action='store_true')
    ap.add_argument('--head_name', type=str, default='fc')

    # selection metric（默认按 CC 的“十关节均值”）
    ap.add_argument('--select_metric', type=str, default='cc', choices=['mse','nrmse','cc','r2'])

    args = ap.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"Pretrained(LSTM): {args.pretrained}")
    print(f"Targets: {args.targets}")
    print(f"Sources: {args.source_subjects}")
    print(f"Save dir: {args.save_dir}")
    print(f"Select metric: {args.select_metric}")

    set_seed(525)
    for tgt in args.targets:
        print(f"\n====== ATL-LSTM start: {tgt} ======")
        run_atl_lstm_for_target(args, device, tgt, args.source_subjects)
        print(f"====== ATL-LSTM done : {tgt} ======\n")

if __name__ == "__main__":
    main()
