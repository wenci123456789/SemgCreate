# atl_cdanr_emgmamba.py
import argparse, os, math
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
from DataProcess import NinaPro                         # 数据接口（μ-law、窗口/标签一致）:contentReference[oaicite:4]{index=4}
from Model.EMGMambaAttentionAdapter import EMGMambaAdapter  # 模型结构（hook到 output_proj 上游）:contentReference[oaicite:5]{index=5}

# ========== 判别器 ==========
class DomainDiscriminator(nn.Module):
    """标准 ATL：仅对齐倒数第二层特征 f"""
    def __init__(self, in_dim, hidden=256, p=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(True), nn.Dropout(p),
            nn.Linear(hidden, 1)
        )
    def forward(self, z):
        return self.net(z).squeeze(-1)

class JointDomainDiscriminator(nn.Module):
    """CDAN‑R：对齐 (f, ŷ) 的联合表示（外积展平）"""
    def __init__(self, in_f, out_y=10, hidden=512, p=0.2):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_f*out_y, hidden), nn.ReLU(True), nn.Dropout(p),
            nn.Linear(hidden, 1)
        )
        self.out_y = out_y
        self.in_f = in_f
    def forward(self, f, yhat):
        # f:[B,D], yhat:[B,10] -> 外积 [B,D,10] -> [B,D*10]
        outer = torch.bmm(f.unsqueeze(2), yhat.detach().unsqueeze(1)).view(f.size(0), -1)
        return self.fc(outer).squeeze(-1)

# ========== Hook 包装器：抓取 output_proj 的输入作为“倒数第二层特征” ==========
class HookedModel(nn.Module):
    def __init__(self, backbone: nn.Module, feature_module_name="output_proj"):
        super().__init__()
        self.backbone = backbone
        self._feat = None
        modules = dict(self.backbone.named_modules())
        assert feature_module_name in modules, f"未找到模块 {feature_module_name}"
        modules[feature_module_name].register_forward_hook(self._hook)
    def _hook(self, module, fin, fout):
        x = fin[0]             # 取 output_proj 的输入
        if x.dim() == 3:       # [B,T,D] -> 池化到 [B,D]
            x = x.mean(dim=1)
        self._feat = x
    def forward_with_features(self, x):
        y = self.backbone(x)   # 正常前向；hook 把特征存到 self._feat
        return y, self._feat

# ========== 可选：U‑ATL 的不确定性头（在 New‑t‑net 侧基于特征 f 估计 logvar） ==========
class UncertaintyHead(nn.Module):
    def __init__(self, in_dim, out_y=10):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_y)
    def forward(self, f):
        return self.fc(f)   # log-variance

def load_state_flex(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location='cpu')
    if isinstance(ckpt, dict) and 'model_state' in ckpt:
        return ckpt['model_state']
    if hasattr(ckpt, 'state_dict'):
        return ckpt.state_dict()
    return ckpt

def make_paths(root, sid):
    return (os.path.join(root, f"{sid}_E2_A1_rms_train.h5"),
            os.path.join(root, f"{sid}_E2_A1_glove_train.h5"),
            os.path.join(root, f"{sid}_E2_A1_rms_test.h5"),
            os.path.join(root, f"{sid}_E2_A1_glove_test.h5"))

def main():
    ap = argparse.ArgumentParser()
    # 数据
    ap.add_argument('--data_root', default='../../semg-wenci/Data/featureset')
    ap.add_argument('--source_subjects', nargs='+', required=True)
    ap.add_argument('--target_subject', required=True)
    ap.add_argument('--subframe', type=int, default=200)
    ap.add_argument('--batch_size', type=int, default=64)
    ap.add_argument('--normalization', default='miu')
    ap.add_argument('--miu', type=int, default=220)  # 论文建议 μ=220:contentReference[oaicite:6]{index=6}
    # 预训练
    ap.add_argument('--pretrained', required=True, help='跨被试 Multi-s-net checkpoint')
    # 训练
    ap.add_argument('--epochs', type=int, default=50)      # 论文：50:contentReference[oaicite:7]{index=7}
    ap.add_argument('--lr_t', type=float, default=1e-3)    # 论文：1e-3:contentReference[oaicite:8]{index=8}
    ap.add_argument('--lr_d', type=float, default=1e-3)
    ap.add_argument('--lambda_adv', type=float, default=1.0)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--save_dir', default='./checkpoints_atl_plus')
    # 选项
    ap.add_argument('--cond_align', action='store_true', help='启用 CDAN‑R 条件式对齐')
    ap.add_argument('--u_atl', action='store_true', help='启用 U‑ATL（异方差 + 不确定性调权）')
    args = ap.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    device = args.device

    # ========== 1) 加载 目标域（t）与 源域（s）数据 ==========
    emg_t_tr, g_t_tr, emg_t_te, g_t_te = make_paths(args.data_root, args.target_subject)
    tgt_train = NinaPro.NinaPro(emg_t_tr, g_t_tr, subframe=args.subframe,
                                normalization=args.normalization, mu=args.miu,
                                dummy_label=0, class_num=1)                      #:contentReference[oaicite:9]{index=9}
    tgt_test  = NinaPro.NinaPro(emg_t_te, g_t_te, subframe=args.subframe,
                                normalization=args.normalization, mu=args.miu,
                                dummy_label=0, class_num=1)
    T_loader  = DataLoader(tgt_train, batch_size=args.batch_size, shuffle=True,  drop_last=True)
    Te_loader = DataLoader(tgt_test,  batch_size=args.batch_size, shuffle=False)

    S_sets = []
    for i, sid in enumerate(args.source_subjects):
        e_tr, g_tr, _, _ = make_paths(args.data_root, sid)
        S_sets.append(NinaPro.NinaPro(e_tr, g_tr, subframe=args.subframe,
                                      normalization=args.normalization, mu=args.miu,
                                      dummy_label=i, class_num=len(args.source_subjects)))  #:contentReference[oaicite:10]{index=10}
    S_loader = DataLoader(ConcatDataset(S_sets), batch_size=args.batch_size, shuffle=True, drop_last=True)

    # ========== 2) Multi-s-net（冻结） & New‑t‑net（可训练） ==========
    ms_net = EMGMambaAdapter(input_dim=12, output_dim=10)
    ms_net.load_state_dict(load_state_flex(args.pretrained), strict=False)
    for p in ms_net.parameters(): p.requires_grad = False

    nt_net = EMGMambaAdapter(input_dim=12, output_dim=10)
    nt_net.load_state_dict(ms_net.state_dict(), strict=False)

    ms = HookedModel(ms_net).to(device)
    nt = HookedModel(nt_net).to(device)

    # 预跑一次确定特征维度
    with torch.no_grad():
        xb, yb, *_ = next(iter(T_loader))
        _ = nt.forward_with_features(xb.squeeze(3).to(device))
        feat_dim = nt._feat.shape[-1]

    # U‑ATL：不确定性头（只在 New‑t‑net 侧使用）
    u_head = UncertaintyHead(feat_dim, out_y=10).to(device) if args.u_atl else None

    # 判别器
    if args.cond_align:
        D = JointDomainDiscriminator(in_f=feat_dim, out_y=10).to(device)  # CDAN‑R
    else:
        D = DomainDiscriminator(in_dim=feat_dim).to(device)               # 标准 ATL

    # ========== 3) 优化器与损失 ==========
    bce_logits = nn.BCEWithLogitsLoss(reduction='none')  # 手动加权
    mse = nn.MSELoss(reduction='mean')

    opt_t = optim.AdamW(list(nt.parameters()) + ([p for p in u_head.parameters()] if u_head else []), lr=args.lr_t)
    opt_d = optim.AdamW(D.parameters(), lr=args.lr_d)

    def nll_gauss(y_hat, y_true, logvar):
        # y_hat,y_true:[B,10]; logvar:[B,10]
        logvar = torch.clamp(logvar, -5, 5)
        inv_var = torch.exp(-logvar)
        return 0.5 * ((y_true - y_hat)**2 * inv_var + logvar).mean(), inv_var.mean(dim=1)  # (scalar, [B])

    best = math.inf
    for epoch in range(1, args.epochs + 1):
        nt.train(); D.train(); ms.eval()
        if u_head: u_head.train()
        it_s = iter(S_loader)
        total_reg, total_d = 0.0, 0.0

        for batch_t in T_loader:
            # ----- 取一批目标域 -----
            x_t = batch_t[0].squeeze(3).to(device)    # [B,200,12]:contentReference[oaicite:11]{index=11}
            y_t = batch_t[1].to(device).squeeze(1)    # [B,10]

            # ----- 对齐一批源域 -----
            try:
                batch_s = next(it_s)
            except StopIteration:
                it_s = iter(S_loader); batch_s = next(it_s)
            x_s = batch_s[0].squeeze(3).to(device)

            # ===== 3.1 训练判别器 D =====
            with torch.no_grad():
                y_s_pred, f_s = ms.forward_with_features(x_s)     # Multi-s-net 特征
                y_t_pred, f_t = nt.forward_with_features(x_t)     # New‑t‑net 特征
                y_s_pred = y_s_pred.squeeze(1)                    # [B,10]
                y_t_pred = y_t_pred.squeeze(1)                    # [B,10]

            if args.cond_align:
                logit_s = D(f_s, y_s_pred); logit_t = D(f_t, y_t_pred)
            else:
                logit_s = D(f_s); logit_t = D(f_t)
            loss_d = bce_logits(logit_s, torch.ones_like(logit_s)) + \
                     bce_logits(logit_t, torch.zeros_like(logit_t))
            loss_d = loss_d.mean()

            opt_d.zero_grad(); loss_d.backward(); opt_d.step()

            # ===== 3.2 训练 New‑t‑net（任务 + 对齐）=====
            y_t_pred, f_t = nt.forward_with_features(x_t)         # 不要 detach
            y_t_pred = y_t_pred.squeeze(1)                        # [B,10]

            if args.u_atl:
                logvar_t = u_head(f_t)                            # [B,10]
                loss_reg, w_batch = nll_gauss(y_t_pred, y_t, logvar_t)  # NLL + 权重
            else:
                loss_reg = mse(y_t_pred, y_t)
                w_batch = torch.ones(x_t.size(0), device=device)  # 无权重

            if args.cond_align:
                logit_t = D(f_t, y_t_pred)
            else:
                logit_t = D(f_t)

            map_loss_each = bce_logits(logit_t, torch.ones_like(logit_t))  # [B]
            # U‑ATL：用 σ^{-2} 作为样本权重；标准 ATL：权重=1
            loss_map = (map_loss_each * w_batch).mean()

            loss_tot = loss_reg + args.lambda_adv * loss_map
            opt_t.zero_grad(); loss_tot.backward(); opt_t.step()

            total_reg += float(loss_reg.item())
            total_d   += float(loss_d.item())

        # ===== 验证（只看 MSE 以便与已有脚本/论文对齐）=====
        nt.eval();
        if u_head: u_head.eval()
        val, n = 0.0, 0
        with torch.no_grad():
            for batch in Te_loader:
                x = batch[0].squeeze(3).to(device)
                y = batch[1].to(device).squeeze(1)
                yhat = nt.forward_with_features(x)[0].squeeze(1)
                val += mse(yhat, y).item(); n += 1
        val /= n

        print(f"Epoch {epoch:03d}  train_reg={total_reg/len(T_loader):.4f}  "
              f"train_D={total_d/len(T_loader):.4f}  val_mse={val:.4f}")

        torch.save({'epoch': epoch,
                    'model_state': nt_net.state_dict(),
                    'u_head_state': (u_head.state_dict() if u_head else None),
                    'flags': {'cond_align': args.cond_align, 'u_atl': args.u_atl}},
                   os.path.join(args.save_dir, 'atl_plus_latest.pth'))
        if val < best:
            best = val
            torch.save({'epoch': epoch,
                        'model_state': nt_net.state_dict(),
                        'u_head_state': (u_head.state_dict() if u_head else None),
                        'flags': {'cond_align': args.cond_align, 'u_atl': args.u_atl}},
                       os.path.join(args.save_dir, 'atl_plus_best.pth'))

if __name__ == "__main__":
    main()
