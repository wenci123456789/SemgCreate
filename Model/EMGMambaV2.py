import torch
import torch.nn as nn
from mamba_ssm import Mamba


class EMGMambaRegressor(nn.Module):
    """
    多尺度卷积 + Mamba 连续估计模型
    输入: [B, T, C]   输出: [B, 1, output_dim]
    """
    def __init__(self,
                 input_dim=12,      # sEMG 通道数
                 hidden_dim=256,    # 隐藏层维度
                 depth=5,           # Mamba 层数
                 output_dim=10,     # 输出维度
                 d_state=16,
                 d_conv=4,
                 expand=2,
                 dropout=0.1,
                 kernel_sizes=(3, 5, 7)):  # 多尺度卷积核
        super().__init__()
        self.name = "mamba_multiscale"

        # ==== 多尺度卷积特征提取 ====
        self.multi_conv = nn.ModuleList([
            nn.Conv1d(in_channels=input_dim,
                      out_channels=hidden_dim,
                      kernel_size=k,
                      padding=k // 2)
            for k in kernel_sizes
        ])

        self.dropout = nn.Dropout(dropout)

        # ==== Mamba 层堆叠 ====
        self.mamba_layers = nn.ModuleList([
            Mamba(
                d_model=hidden_dim,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand
            ) for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        """
        x: [B, T, C]
        输出: [B, 1, output_dim]
        """
        # --- 多尺度卷积 ---
        x = x.transpose(1, 2)  # [B, C, T]
        conv_outs = []
        for conv in self.multi_conv:
            conv_outs.append(conv(x))
        x = sum(conv_outs) / len(conv_outs)  # 平均融合多尺度特征
        x = x.transpose(1, 2)  # [B, T, hidden_dim]
        x = self.dropout(x)

        # --- Mamba 编码 ---
        for layer in self.mamba_layers:
            x = x + layer(x)
            x = self.norm(x)

        # --- 全局时序汇聚 + 输出 ---
        x = x.mean(dim=1, keepdim=True)  # [B, 1, hidden_dim]
        x = self.output_proj(x)
        return x  # [B, 1, output_dim]


if __name__ == "__main__":
    batch_size = 32
    seq_len = 200
    input_dim = 12
    output_dim = 10

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("使用设备:", device)

    x = torch.randn(batch_size, seq_len, input_dim).to(device)
    model = EMGMambaRegressor(input_dim=input_dim, output_dim=output_dim).to(device)

    y = model(x)
    print("输入形状:", x.shape)
    print("输出形状:", y.shape)
