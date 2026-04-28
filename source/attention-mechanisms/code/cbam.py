import torch
import torch.nn as nn
from .channel_attention import ChannelAttention
from .spatial_attention import SpatialAttention


class CBAM(nn.Module):
    """
    完整的 CBAM 模块

    先通道注意力 → 再空间注意力, 串联组合:
    1. 通道注意力: 判断"什么特征重要" (C×1×1 权重)
    2. 空间注意力: 判断"哪里重要" (1×H×W 权重)
    
    输入: (B, C, H, W)
    输出: (B, C, H, W)
    """
    def __init__(self, in_channels, reduction=16, kernel_size=7):
        super(CBAM, self).__init__()

        self.channel_attention = ChannelAttention(in_channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        # Step 1: 通道注意力 → 重新校准通道重要性
        # x * channel_attention(x): 每个通道乘以其重要性权重
        x = x * self.channel_attention(x)

        # Step 2: 空间注意力 → 聚焦重要空间区域
        # x * spatial_attention(x): 每个空间位置乘以其重要性权重
        x = x * self.spatial_attention(x)

        return x

if __name__ == "__main__":
    x = torch.randn(2, 64, 32, 32)
    cbam = CBAM(64, reduction=16)
    y = cbam(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    print(f"CBAM params: {sum(p.numel() for p in cbam.parameters())}")
