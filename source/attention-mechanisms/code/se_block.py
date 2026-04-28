import torch
import torch.nn as nn

class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation 模块

    三步核心操作：
    1. Squeeze: 全局平均池化, 将 C×H×W 压缩为 C×1×1
    2. Excitation: 两个全连接层学习通道权重 (瓶颈结构)
    3. Scale: 权重乘回原始特征图

    参数量: 2 * C * (C/r) = 2C²/r
    输入: (B, C, H, W)
    输出: (B, C, H, W)
    """
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()

        # Squeeze: 全局平均池化, 每个通道压缩为一个标量
        # 输入 (B, C, H, W) → 输出 (B, C, 1, 1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # Excitation: 瓶颈结构, 压缩比 r=16 减少参数量
        # C → C/r → C, 参数: C*(C/r) + (C/r)*C = 2C²/r
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),  # 降维
            nn.ReLU(inplace=True),                                   # 非线性
            nn.Linear(channels // reduction, channels, bias=False),  # 升维
            nn.Sigmoid()                                             # 输出 (0,1) 权重
        )

    def forward(self, x):
        b, c, _, _ = x.size()

        # Squeeze: (B, C, H, W) → (B, C, 1, 1) → (B, C)
        y = self.avg_pool(x).view(b, c)

        # Excitation: (B, C) → (B, C/r) → (B, C) → (B, C, 1, 1)
        y = self.fc(y).view(b, c, 1, 1)

        # Scale: 逐通道乘法, 广播到 H×W
        # 每个通道乘以对应的标量权重
        return x * y.expand_as(x)

if __name__ == "__main__":
    x = torch.randn(2, 64, 32, 32)
    se = SEBlock(64, reduction=16)
    y = se(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    print(f"SEBlock parameters: {sum(p.numel() for p in se.parameters())}  (2*64²/16 = 512)")
