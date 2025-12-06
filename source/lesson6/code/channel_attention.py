import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    """CBAM的通道注意力模块"""
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        # 共享的MLP
        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
        )
        
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # 平均池化和最大池化
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        
        # 结合两种池化结果
        out = avg_out + max_out
        return self.sigmoid(out)


if __name__ == "__main__":
    # 测试ChannelAttention
    x = torch.randn(2, 64, 32, 32)
    ca = ChannelAttention(64, reduction=16)
    y = ca(x)
    print(f"Input shape: {x.shape}")
    print(f"Attention shape: {y.shape}")
    print(f"ChannelAttention parameters: {sum(p.numel() for p in ca.parameters())}")