import torch
import torch.nn as nn


class SpatialAttention(nn.Module):
    """CBAM的空间注意力模块"""
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # 沿通道维度进行平均池化和最大池化
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        
        # 拼接并卷积
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        
        return self.sigmoid(x)


if __name__ == "__main__":
    # 测试SpatialAttention
    x = torch.randn(2, 64, 32, 32)
    sa = SpatialAttention(kernel_size=7)
    y = sa(x)
    print(f"Input shape: {x.shape}")
    print(f"Attention shape: {y.shape}")
    print(f"SpatialAttention parameters: {sum(p.numel() for p in sa.parameters())}")