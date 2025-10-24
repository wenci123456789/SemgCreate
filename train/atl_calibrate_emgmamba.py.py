# atl_calibrate_emgmamba.py
import argparse, os, math, itertools
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
from DataProcess import NinaPro
from Model.EMGMambaAttentionAdapter import EMGMambaAdapter  # 你的模型  :contentReference[oaicite:6]{index=6}

# ============ 1) 域判别器（MLP） ============
class DomainDiscriminator(nn.Module):
    def __init__(self, in_dim, hidden=256, p=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(inplace=True), nn.Dropout(p),
            nn.Linear(hidden, 1)  # 二分类 logit
        )
    def forward(self, f):
        return self.net(f).squeeze(-1)

# ============ 2) 用 hook 拿到“倒数第二层特征”（output_proj 的输入） ============
class HookedModel(nn.Module):
    def __init__(self, backbone: nn.Module, feature_module_name: str = "output_proj"):
        super().__init__()
        self.backbone = backbone
        self._feat = None
        m = dict(self.backbone.named_modules())
        assert feature_module_name in m, f"找不到模块 {feature_module_name}"
        m[feature_module_name].register_forward_hook(self._hook)

    def _hook(self, module, fin, fout):
        # fin[0] = output_proj 的输入，形状通常是 [B, 1, hidden]
        x = fin[0]
        if x.dim() == 3:
            x = x.mean(dim=1)  # [B, hidden]
        self._feat = x  # 不要 detach，mapping loss 要反传到 New-t-net

    def forward_with_features(self, x):
        y = self.backbone(x)  # 正常前向，hook 会拿到 _feat
        return y, self._feat

# ============ 3) 兼容多种 checkpoint 保存方式 ============
def load_state_flex(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location='cpu')
    # 可能是 {'model_state': state_dict}；也可能是直接 state_dict；也可能是整模型
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
                            dummy_label=0, class_num=1)  # 形状等同于你 Trainer 中的用法  :contentReference[oaicite:7]{index=7}
    ds_te = NinaPro.NinaPro(e_te, g_te, subframe=subframe, normalization=normalization, mu=mu,
                            dummy_label=0, class_num=1)
    L_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)
    L_te = DataLoader(ds_te, batch_size=batch_size, shuffle=False)
    return L_tr, L_te

def main():
    ap = argparse.ArgumentParser()
    # 数据 & 目录
    ap.add_argument('--data_root', default='../../semg-wenci/Data/featureset')
    ap.add_argument('--source_subjects', nargs='+', required=True, help='源域：除目标外的若干被试')
    ap.add_argument('--target_subject', required=True, help='目标域：新被试')
    ap.add_argument('--subframe', type=int, default=200)
    ap.add_argument('--batch_size', type=int, default=64)
    ap.add_argument('--normalization', default='miu')
    ap.add_argument('--miu', type=int, default=2 ** 20)  # 论文设置：μ=220  :contentReference[oaicite:8]{index=8}
    # 预训练权重
    ap.add_argument('--pretrained', required=True, help='跨被试 Multi-s-net 的 checkpoint 路径')
    # 优化 & 超参（论文对齐）
    ap.add_argument('--epochs', type=int, default=50)        # 论文：50  :contentReference[oaicite:9]{index=9}
    ap.add_argument('--lr_t', type=float, default=1e-3)      # 论文：1e-3  :contentReference[oaicite:10]{index=10}
    ap.add_argument('--lr_d', type=float, default=1e-3)
    ap.add_argument('--lambda_adv', type=float, default=1.0)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--save_dir', default='./checkpoints_atl')
    args = ap.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    # ===== 3.1 数据加载器 =====
    # 目标域
    T_loader, Te_loader = build_loader(args.data_root, args.target_subject,
                                       args.subframe, args.normalization, args.miu,
                                       args.batch_size, shuffle=True, drop_last=True)

    # 源域（可拼接多个被试）
    S_sets = []
    for i, sid in enumerate(args.source_subjects):
        e = os.path.join(args.data_root, f"{sid}_E2_A1_rms_train.h5")
        g = os.path.join(args.data_root, f"{sid}_E2_A1_glove_train.h5")
        S_sets.append(NinaPro.NinaPro(e, g, subframe=args.subframe,
                                      normalization=args.normalization, mu=args.miu,
                                      dummy_label=i, class_num=len(args.source_subjects)))  # :contentReference[oaicite:11]{index=11}
    S_loader = DataLoader(ConcatDataset(S_sets), batch_size=args.batch_size, shuffle=True, drop_last=True)

    # ===== 3.2 构建 Multi-s-net（冻结） & New-t-net（可训练） =====
    ms_net = EMGMambaAdapter(input_dim=12, output_dim=10)
    ms_net.load_state_dict(load_state_flex(args.pretrained), strict=False)
    for p in ms_net.parameters():
        p.requires_grad = False

    nt_net = EMGMambaAdapter(input_dim=12, output_dim=10)
    nt_net.load_state_dict(ms_net.state_dict(), strict=False)

    ms = HookedModel(ms_net).to(args.device)
    nt = HookedModel(nt_net).to(args.device)

    # 探测 DD 输入维度
    with torch.no_grad():
        xb, yb, *_ = next(iter(T_loader))
        _ = nt.forward_with_features(xb.squeeze(3).to(args.device))  # [B,200,12]  :contentReference[oaicite:12]{index=12}
        feat_dim = nt._feat.shape[-1]
    D = DomainDiscriminator(in_dim=feat_dim).to(args.device)

    # ===== 3.3 优化器与损失 =====
    reg_loss = nn.MSELoss()
    bce_logits = nn.BCEWithLogitsLoss()
    opt_t = optim.AdamW(nt.parameters(), lr=args.lr_t)
    opt_d = optim.AdamW(D.parameters(),  lr=args.lr_d)

    # ===== 3.4 ATL 训练 =====
    best = math.inf
    for epoch in range(1, args.epochs + 1):
        nt.train(); D.train(); ms.eval()
        it_s = iter(S_loader)
        total_reg, total_d = 0.0, 0.0

        for batch_t in T_loader:
            x_t = batch_t[0].squeeze(3).to(args.device)  # [B,200,12]
            y_t = batch_t[1].to(args.device)             # [B,1,10]

            try:
                batch_s = next(it_s)
            except StopIteration:
                it_s = iter(S_loader)
                batch_s = next(it_s)
            x_s = batch_s[0].squeeze(3).to(args.device)

            # 1) 更新 D：区分源/目标
            with torch.no_grad():
                _ = ms.forward_with_features(x_s); f_s = ms._feat.detach()
                _ = nt.forward_with_features(x_t); f_t = nt._feat.detach()
            logit_s, logit_t = D(f_s), D(f_t)
            loss_d = bce_logits(logit_s, torch.ones_like(logit_s)) + \
                     bce_logits(logit_t, torch.zeros_like(logit_t))
            opt_d.zero_grad(); loss_d.backward(); opt_d.step()

            # 2) 更新 New-t-net：任务回归 + 让 D 误判为“源”（mapping）
            pred_t, f_t = nt.forward_with_features(x_t)
            loss_map = bce_logits(D(f_t), torch.ones_like(logit_t))
            loss_reg = reg_loss(pred_t, y_t)
            loss_tot = loss_reg + args.lambda_adv * loss_map
            opt_t.zero_grad(); loss_tot.backward(); opt_t.step()

            total_reg += loss_reg.item(); total_d += loss_d.item()

        # 简单验证（MSE）
        nt.eval()
        val = 0.0; n = 0
        with torch.no_grad():
            for batch in Te_loader:
                x = batch[0].squeeze(3).to(args.device)
                y = batch[1].to(args.device)
                yhat, _ = nt.forward_with_features(x)
                val += reg_loss(yhat, y).item(); n += 1
        val /= n

        print(f"Epoch {epoch:03d}  train_reg={total_reg/len(T_loader):.4f}  "
              f"train_D={total_d/len(T_loader):.4f}  val_mse={val:.4f}")

        torch.save({'epoch': epoch, 'model_state': nt_net.state_dict()},
                   os.path.join(args.save_dir, 'atl_latest.pth'))
        if val < best:
            best = val
            torch.save({'epoch': epoch, 'model_state': nt_net.state_dict()},
                       os.path.join(args.save_dir, 'atl_best.pth'))

if __name__ == "__main__":
    main()
