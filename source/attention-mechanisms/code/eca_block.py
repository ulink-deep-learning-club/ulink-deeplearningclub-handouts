import math
import torch
import torch.nn as nn


class ECABlock(nn.Module):
    """Efficient Channel Attention模块"""
    def __init__(self, channels, gamma=2, b=1):
        super(ECABlock, self).__init__()
        # 自适应确定卷积核大小
        t = int(abs((math.log(channels, 2) + b) / gamma))
        k = t if t % 2 else t + 1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=k//2, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).squeeze(-1).transpose(-1, -2)  # (b, 1, c)
        y = self.conv(y).transpose(-1, -2).unsqueeze(-1)    # (b, c, 1, 1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)


if __name__ == "__main__":
    # 测试ECABlock
    x = torch.randn(2, 64, 32, 32)
    eca = ECABlock(64)
    y = eca(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    print(f"ECABlock parameters: {sum(p.numel() for p in eca.parameters())}")