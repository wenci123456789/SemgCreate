# 定义DepthwiseSeparableConv类，继承自nn.Module
import torch
from torch import nn


class DepthwiseSeparableConv(nn.Module):
    # 类的初始化方法
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        # 调用父类的初始化方法
        super(DepthwiseSeparableConv, self).__init__()

        # 深度卷积层，使用与输入通道数相同的组数，使每个输入通道独立卷积
        self.depthwise = nn.Sequential(nn.Conv2d(in_channels, in_channels, kernel_size,
                                                 stride, padding, groups=in_channels),
                                       nn.BatchNorm2d(in_channels),
                                       # 激活函数层，使用LeakyReLU
                                       nn.LeakyReLU(0.1, inplace=True)
                                       )
        # 逐点卷积层，使用1x1卷积核进行卷积，以改变通道数
        self.pointwise = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1),
                                       nn.BatchNorm2d(out_channels),
                                       # 激活函数层，使用LeakyReLU
                                       nn.LeakyReLU(0.1, inplace=True)
                                       )

    # 定义前向传播方法
    def forward(self, x):
        # 输入x通过深度卷积层
        x = self.depthwise(x)
        # 经过深度卷积层处理后的x通过逐点卷积层
        x = self.pointwise(x)
        # 返回最终的输出
        return x


if __name__ == "__main__":
    # 定义输入张量，大小为[1, 3, 224, 224]，模拟一个batch大小为1，3通道的224x224的图像
    input_tensor = torch.randn(16, 65, 12, 200)
    # 实例化DepthwiseSeparableConv，输入通道数为3，输出通道数为64
    model = DepthwiseSeparableConv(in_channels=65, out_channels=256)
    # 将输入张量通过模型进行前向传播
    output_tensor = model(input_tensor)
    # 打印输出张量的形状，期望为[1, 64, 224, 224]
    print(f"Output tensor shape: {output_tensor.shape}")