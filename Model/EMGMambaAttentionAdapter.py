import torch
import torch.nn as nn
from mamba_ssm import Mamba

# Adapter 模块
class Adapter(nn.Module):
    def __init__(self, hidden_dim, reduction=4):
        super().__init__()
        self.adapter = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // reduction, hidden_dim)
        )
    def forward(self, x):
        return self.adapter(x)

# CBAM 模块
class CBAM(nn.Module):
    def __init__(self, channels, reduction=16, kernel_size=7):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False)
        )
        self.sigmoid_channel = nn.Sigmoid()
        self.conv_spatial = nn.Conv1d(2,1,kernel_size=kernel_size,padding=kernel_size//2,bias=False)
        self.sigmoid_spatial = nn.Sigmoid()

    def forward(self, x):
        # x: [B,C,T]
        B,C,T = x.shape
        # Channel
        avg_out = self.mlp(self.avg_pool(x).view(B,C))
        max_out = self.mlp(self.max_pool(x).view(B,C))
        channel_att = self.sigmoid_channel(avg_out + max_out).view(B,C,1)
        x = x * channel_att
        # Spatial
        avg_out = torch.mean(x,dim=1,keepdim=True)
        max_out,_ = torch.max(x,dim=1,keepdim=True)
        spatial_cat = torch.cat([avg_out,max_out],dim=1)  # [B,2,T]
        spatial_att = self.conv_spatial(spatial_cat)      # [B,1,T]
        spatial_att = self.sigmoid_spatial(spatial_att)
        x = x * spatial_att
        return x

# 主干模型
class EMGMambaAdapter(nn.Module):
    def __init__(self, input_dim=12, hidden_dim=256, depth=5, output_dim=10,
                 d_state=16, d_conv=4, expand=2, dropout=0.1, kernel_sizes=(3,5,7)):
        super().__init__()
        self.name = "EMGMambaAdapter"
        # 多尺度卷积
        self.multi_conv = nn.ModuleList([
            nn.Conv1d(input_dim, hidden_dim, k, padding=k//2) for k in kernel_sizes
        ])
        self.dropout = nn.Dropout(dropout)

        # Mamba + CBAM + Adapter
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                "mamba": Mamba(d_model=hidden_dim, d_state=d_state, d_conv=d_conv, expand=expand),
                "attn": CBAM(hidden_dim),
                "adapter": Adapter(hidden_dim, reduction=4),
                "norm": nn.LayerNorm(hidden_dim)
            }) for _ in range(depth)
        ])

        self.output_proj = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # 多尺度卷积
        x = x.transpose(1,2)  # [B,C,T]
        conv_outs = [conv(x) for conv in self.multi_conv]
        x = sum(conv_outs)/len(conv_outs)
        x = self.dropout(x)
        x = x.transpose(1,2)  # [B,T,hidden]

        for layer in self.layers:
            residual = x
            x = x + layer["mamba"](x)
            x = layer["norm"](x)
            x = layer["attn"](x.transpose(1,2)).transpose(1,2)
            x = x + layer["adapter"](x)  # ✅ Adapter
            x = x + residual  # 残差

        x = x.mean(dim=1, keepdim=True)
        x = self.output_proj(x)
        return x


# model = EMGMambaAdapter(input_dim=12, output_dim=10).to(device)
#
# # 冻结主干，只训练 Adapter 和输出层
# for name, param in model.named_parameters():
#     if "adapter" not in name and "output_proj" not in name:
#         param.requires_grad = False
#
# # optimizer 只更新 Adapter & output_proj
# optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("使用设备:", device)

    # 模拟输入
    batch_size = 8
    seq_len = 200
    input_dim = 12
    output_dim = 10
    x = torch.randn(batch_size, seq_len, input_dim).to(device)

    # 初始化模型
    model = EMGMambaAdapter(input_dim=input_dim, output_dim=output_dim).to(device)

    # 前向推理
    y = model(x)

    print("输入形状:", x.shape)
    print("输出形状:", y.shape)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"总参数量: {total_params/1e6:.3f} M")

    # 冻结主干，只训练 Adapter
    for name, param in model.named_parameters():
        if "adapter" not in name and "output_proj" not in name:
            param.requires_grad = False

    # 输出可训练参数量
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"可训练参数量 (Adapter + 输出层): {trainable_params/1e6:.3f} M")