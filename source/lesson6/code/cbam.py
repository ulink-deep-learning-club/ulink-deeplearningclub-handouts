import torch
import torch.nn as nn
from .channel_attention import ChannelAttention
from .spatial_attention import SpatialAttention


class CBAM(nn.Module):
    """完整的CBAM模块"""
    def __init__(self, in_channels, reduction=16, kernel_size=7):
        super(CBAM, self).__init__()
        
        self.channel_attention = ChannelAttention(in_channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)
        
    def forward(self, x):
        # 先应用通道注意力
        x = x * self.channel_attention(x)
        
        # 再应用空间注意力
        x = x * self.spatial_attention(x)
        
        return x


if __name__ == "__main__":
    # 测试CBAM
    x = torch.randn(2, 64, 32, 32)
    cbam = CBAM(64, reduction=16, kernel_size=7)
    y = cbam(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    print(f"CBAM parameters: {sum(p.numel() for p in cbam.parameters())}")