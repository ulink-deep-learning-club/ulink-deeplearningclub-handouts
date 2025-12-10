import torch
import torch.nn as nn


class ConvSpatialAttention(nn.Module):
    """卷积生成的空间注意力"""
    def __init__(self, in_channels, reduction=16):
        super(ConvSpatialAttention, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1),
            nn.BatchNorm2d(in_channels // reduction),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, 1, 3, padding=1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        return self.conv(x)


if __name__ == "__main__":
    # 测试ConvSpatialAttention
    x = torch.randn(2, 64, 32, 32)
    csa = ConvSpatialAttention(64, reduction=16)
    y = csa(x)
    print(f"Input shape: {x.shape}")
    print(f"Attention shape: {y.shape}")
    print(f"ConvSpatialAttention parameters: {sum(p.numel() for p in csa.parameters())}")