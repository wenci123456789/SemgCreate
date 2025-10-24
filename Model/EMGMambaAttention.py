import torch
import torch.nn as nn
from mamba_ssm import Mamba


# ============ CBAM 模块 ============ #
class CBAM(nn.Module):
    def __init__(self, channels, reduction=16, kernel_size=7):
        super().__init__()
        # Channel Attention
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False)
        )
        self.sigmoid_channel = nn.Sigmoid()

        # Spatial Attention
        self.conv_spatial = nn.Conv1d(
            in_channels=2,
            out_channels=1,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            bias=False
        )
        self.sigmoid_spatial = nn.Sigmoid()

    def forward(self, x):
        # x: [B, C, T]
        B, C, T = x.shape

        # --- Channel Attention ---
        avg_out = self.mlp(self.avg_pool(x).view(B, C))
        max_out = self.mlp(self.max_pool(x).view(B, C))
        channel_att = self.sigmoid_channel(avg_out + max_out).view(B, C, 1)
        x = x * channel_att  # 广播到 [B,C,T]

        # --- Spatial Attention ---
        avg_out = torch.mean(x, dim=1, keepdim=True)  # [B,1,T]
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # [B,1,T]
        spatial_cat = torch.cat([avg_out, max_out], dim=1)  # [B,2,T]
        spatial_att = self.conv_spatial(spatial_cat)        # [B,1,T]
        spatial_att = self.sigmoid_spatial(spatial_att)     # [B,1,T]
        x = x * spatial_att                                 # 广播到 [B,C,T]

        return x


# ============ 改进主干模型 ============ #
class EMGMambaRegressor(nn.Module):
    """
    多尺度卷积 + Mamba + CBAM 注意力模型
    输入: [B, T, C] 输出: [B, 1, output_dim]
    """
    def __init__(self,
                 input_dim=12,
                 hidden_dim=256,
                 depth=5,
                 output_dim=10,
                 d_state=16,
                 d_conv=4,
                 expand=2,
                 dropout=0.1,
                 kernel_sizes=(3, 5, 7)):
        super().__init__()
        self.name = "mamba_multiscale_cbam"

        # 多尺度卷积
        self.multi_conv = nn.ModuleList([
            nn.Conv1d(input_dim, hidden_dim, kernel_size=k, padding=k // 2)
            for k in kernel_sizes
        ])

        # Mamba + CBAM 层堆叠
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                "mamba": Mamba(d_model=hidden_dim, d_state=d_state, d_conv=d_conv, expand=expand),
                "attn": CBAM(hidden_dim),
                "norm": nn.LayerNorm(hidden_dim)
            }) for _ in range(depth)
        ])

        self.dropout = nn.Dropout(dropout)
        self.output_proj = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # 多尺度卷积提取
        x = x.transpose(1, 2)  # [B, C, T]
        conv_outs = [conv(x) for conv in self.multi_conv]
        x = sum(conv_outs) / len(conv_outs)  # [B, hidden_dim, T]
        x = self.dropout(x)
        x = x.transpose(1, 2)  # [B, T, hidden_dim]

        # Mamba + CBAM 注意力堆叠
        for layer in self.layers:
            residual = x
            x = x + layer["mamba"](x)
            x = layer["norm"](x)
            # CBAM 作用于 [B, C, T]
            x = layer["attn"](x.transpose(1, 2)).transpose(1, 2)
            x = x + residual  # 保持稳定残差

        # 汇聚 + 输出
        x = x.mean(dim=1, keepdim=True)
        x = self.output_proj(x)
        return x


# ============ 测试运行 ============ #
if __name__ == "__main__":
    batch_size, seq_len, input_dim, output_dim = 8, 200, 12, 10
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = EMGMambaRegressor(input_dim=input_dim, output_dim=output_dim).to(device)
    x = torch.randn(batch_size, seq_len, input_dim).to(device)
    y = model(x)

    print("输入形状:", x.shape)
    print("输出形状:", y.shape)
    print("参数量:", sum(p.numel() for p in model.parameters()) / 1e6, "M")
