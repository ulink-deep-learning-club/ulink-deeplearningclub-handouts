import torch
import torch.nn as nn

class SpatialAttention(nn.Module):
    """
    CBAM 风格的空间注意力模块

    沿通道维度聚合信息, 生成空间注意力图 (1×H×W):
    1. 通道平均池化 + 通道最大池化 → 2×H×W
    2. 7×7 卷积压缩为 1×H×W
    3. Sigmoid 激活

    参数量: 2 * 7 * 7 = 98 (k=7 时)
    输入: (B, C, H, W)
    输出: (B, 1, H, W)  — 空间注意力图
    """
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        # 输入 2 通道 (avg + max), 输出 1 通道 (注意力图)
        # 参数量: 2 * 1 * k * k = 2k²
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 沿通道维度 (dim=1) 聚合
        # avg_out: (B, 1, H, W), 每个位置的平均激活强度
        avg_out = torch.mean(x, dim=1, keepdim=True)
        # max_out: (B, 1, H, W), 每个位置的最强响应
        max_out, _ = torch.max(x, dim=1, keepdim=True)

        # 拼接: (B, 2, H, W)
        x = torch.cat([avg_out, max_out], dim=1)
        # 卷积压缩 + Sigmoid: (B, 1, H, W)
        x = self.conv(x)
        return self.sigmoid(x)

if __name__ == "__main__":
    x = torch.randn(2, 64, 32, 32)
    sa = SpatialAttention(kernel_size=7)
    y = sa(x)
    print(f"Input shape: {x.shape}")
    print(f"Attention shape: {y.shape}")
    print(f"SpatialAttention params: {sum(p.numel() for p in sa.parameters())}  (2*7²=98)")
