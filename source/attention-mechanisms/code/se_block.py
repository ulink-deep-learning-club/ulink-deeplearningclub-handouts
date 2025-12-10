import torch
import torch.nn as nn
import torch.nn.functional as F


class SEBlock(nn.Module):
    """Squeeze-and-Excitation模块"""
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        
        # 两个全连接层构成瓶颈结构
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        b, c, _, _ = x.size()
        
        # Squeeze: 全局平均池化
        y = self.avg_pool(x).view(b, c)
        
        # Excitation: 通过全连接层
        y = self.fc(y).view(b, c, 1, 1)
        
        # Scale: 特征重标定
        return x * y.expand_as(x)


if __name__ == "__main__":
    # 测试SEBlock
    x = torch.randn(2, 64, 32, 32)
    se = SEBlock(64, reduction=16)
    y = se(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    print(f"SEBlock parameters: {sum(p.numel() for p in se.parameters())}")