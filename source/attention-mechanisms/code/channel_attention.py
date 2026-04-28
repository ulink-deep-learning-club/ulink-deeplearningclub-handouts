import torch
import torch.nn as nn

class ChannelAttention(nn.Module):
    """
    CBAM 的通道注意力模块

    与 SE-Net 的区别: 同时使用平均池化和最大池化, 然后相加融合
    - 平均池化: 捕获全局统计信息
    - 最大池化: 捕获最显著响应
    - 两者互补, 提供更丰富的通道描述

    使用 1×1 Conv2d 替代 Linear, 保持 4D 张量格式 (兼容性更好)
    参数量: 2 * C * (C/r) = 2C²/r
    输入: (B, C, H, W)
    输出: (B, C, 1, 1)
    """
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # 共享的 MLP, 用 1×1 Conv2d 实现 (等价于 Linear, 但保持 4D)
        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),  # 降维
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)   # 升维
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 两种池化分别通过共享 MLP, 然后逐元素相加
        avg_out = self.mlp(self.avg_pool(x))  # (B, C, 1, 1)
        max_out = self.mlp(self.max_pool(x))  # (B, C, 1, 1)
        out = avg_out + max_out                # 融合
        return self.sigmoid(out)

if __name__ == "__main__":
    x = torch.randn(2, 64, 32, 32)
    ca = ChannelAttention(64, reduction=16)
    y = ca(x)
    print(f"Input shape: {x.shape}")
    print(f"Attention shape: {y.shape}")
    print(f"ChannelAttention params: {sum(p.numel() for p in ca.parameters())}")
