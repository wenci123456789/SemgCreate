# finetune_ft.py
"""
普通微调（FT）：冻结骨干，仅训练 Adapter + 输出层
- 载入跨被试预训练权重（Multi-s-net）
- 在目标被试的训练集上微调 Adapter & output_proj
- 在目标被试的测试集上评估并保存 best/latest
"""
import argparse, os, math, random, numpy as np
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
from DataProcess import NinaPro
from Model.EMGMambaAttentionAdapter import EMGMambaAdapter  # 模型结构:contentReference[oaicite:2]{index=2}

def seed_everything(seed=525):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

def freeze_to_ft(model):
    """只训练 Adapter + output_proj；其余全部冻结:contentReference[oaicite:3]{index=3}"""
    for name, p in model.named_parameters():
        p.requires_grad = ('adapter' in name) or ('output_proj' in name)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"[FT] Trainable params: {trainable}/{total} ({100*trainable/total:.2f}%)")
    return model

def load_state_flex(ckpt_path):
    """兼容 {'model_state': sd} / 直接 state_dict / 整模型对象"""
    ckpt = torch.load(ckpt_path, map_location='cpu')
    if isinstance(ckpt, dict) and 'model_state' in ckpt:
        return ckpt['model_state']
    if hasattr(ckpt, 'state_dict'):
        return ckpt.state_dict()
    return ckpt

def make_loader(data_root, sid, subframe, normalization, miu, batch, shuffle, drop_last):
    e_tr = os.path.join(data_root, f"{sid}_E2_A1_rms_train.h5")
    g_tr = os.path.join(data_root, f"{sid}_E2_A1_glove_train.h5")
    e_te = os.path.join(data_root, f"{sid}_E2_A1_rms_test.h5")
    g_te = os.path.join(data_root, f"{sid}_E2_A1_glove_test.h5")
    ds_tr = NinaPro.NinaPro(e_tr, g_tr, subframe=subframe, normalization=normalization, mu=miu,
                            dummy_label=0, class_num=1)   # 形状/归一化与您现有代码一致:contentReference[oaicite:4]{index=4}
    ds_te = NinaPro.NinaPro(e_te, g_te, subframe=subframe, normalization=normalization, mu=miu,
                            dummy_label=0, class_num=1)
    L_tr = DataLoader(ds_tr, batch_size=batch, shuffle=shuffle, drop_last=drop_last)
    L_te = DataLoader(ds_te, batch_size=batch, shuffle=False)
    return L_tr, L_te

def main():
    ap = argparse.ArgumentParser()
    # 数据
    ap.add_argument('--data_root', default='../../semg-wenci/Data/featureset')
    ap.add_argument('--target_subject', required=True, help='目标被试 ID，如 S21')
    ap.add_argument('--subframe', type=int, default=200)
    ap.add_argument('--batch_size', type=int, default=64)
    ap.add_argument('--normalization', default='miu')
    ap.add_argument('--miu', type=int, default=220)            # 建议与论文一致 μ=220:contentReference[oaicite:5]{index=5}
    # 预训练
    ap.add_argument('--pretrained', required=True, help='跨被试预训练权重路径 model_best.pth')
    # 训练
    ap.add_argument('--epochs', type=int, default=50)           # 与常见校准轮次一致:contentReference[oaicite:6]{index=6}
    ap.add_argument('--lr', type=float, default=1e-3)           # FT 常用 1e-3:contentReference[oaicite:7]{index=7}
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--save_dir', default='./checkpoints_ft')
    args = ap.parse_args()

    seed_everything(525)
    os.makedirs(args.save_dir, exist_ok=True)

    # 数据加载（目标被试）
    TrainL, TestL = make_loader(args.data_root, args.target_subject, args.subframe,
                                args.normalization, args.miu, args.batch_size, True, True)

    # 模型与预训练权重
    model = EMGMambaAdapter(input_dim=12, output_dim=10).to(args.device)
    state = load_state_flex(args.pretrained)
    model.load_state_dict(state, strict=False)                  # 你的 Trainer 保存方式已兼容:contentReference[oaicite:8]{index=8}
    freeze_to_ft(model)                                         # 冻结骨干，只训 Adapter+Head:contentReference[oaicite:9]{index=9}

    # 优化器与损失
    reg_loss = nn.MSELoss()
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

    # 训练循环
    best = math.inf
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for batch in TrainL:
            x = batch[0].squeeze(3).to(args.device)             # [B,200,12]:contentReference[oaicite:10]{index=10}
            y = batch[1].to(args.device)                        # [B,1,10]
            optimizer.zero_grad()
            pred = model(x)                                     # 前向：输出 [B,1,10]:contentReference[oaicite:11]{index=11}
            loss = reg_loss(pred, y)
            loss.backward()
            optimizer.step()
            total += loss.item()
        avg_train = total / len(TrainL)

        # 验证
        model.eval()
        val = 0.0
        with torch.no_grad():
            for batch in TestL:
                x = batch[0].squeeze(3).to(args.device)
                y = batch[1].to(args.device)
                val += reg_loss(model(x), y).item()
        avg_val = val / len(TestL)

        print(f"Epoch {epoch:03d}  train={avg_train:.4f}  val={avg_val:.4f}")

        # 保存
        torch.save({'epoch': epoch, 'model_state': model.state_dict()},
                   os.path.join(args.save_dir, 'ft_latest.pth'))
        if avg_val < best:
            best = avg_val
            torch.save({'epoch': epoch, 'model_state': model.state_dict()},
                       os.path.join(args.save_dir, 'ft_best.pth'))

if __name__ == '__main__':
    main()
