
import torch
import torch.nn as nn
from mamba_ssm import Mamba


class EMGMambaRegressor(nn.Module):
    """
    Mamba 连续估计模型 —— 输入 [B, T, C]，输出 [B, 1, output_dim]
    """
    def __init__(self,
                 input_dim=12,      # sEMG 通道数
                 hidden_dim=256,    # 隐藏层维度
                 depth=5,           # Mamba 层数
                 output_dim=10,     # 输出维度
                 d_state=16,
                 d_conv=4,
                 expand=2,
                 dropout=0.1):
        super().__init__()
        self.name = "mamba"
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=hidden_dim, kernel_size=3, padding=1)
        self.dropout = nn.Dropout(dropout)
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
        x = x.transpose(1, 2)  # [B, C, T]
        x = self.conv1(x)
        x = x.transpose(1, 2)
        x = self.dropout(x)
        for layer in self.mamba_layers:
            x = x + layer(x)
            x = self.norm(x)
        x = x.mean(dim=1, keepdim=True)  # [B, 1, hidden_dim]
        x = self.output_proj(x)
        return x       # [B, 1, output_dim]


if __name__ == "__main__":
    batch_size = 32
    seq_len = 200
    input_dim = 12
    output_dim = 10

    # 选择设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("使用设备:", device)

    # 模拟输入
    x = torch.randn(batch_size, seq_len, input_dim).to(device)

    # 初始化模型并放到相同设备
    model = EMGMambaRegressor(input_dim=input_dim, output_dim=output_dim).to(device)

    # 前向推理
    y = model(x)
    print("输入形状:", x.shape)
    print("输出形状:", y.shape)
    print("输出示例:", y)

