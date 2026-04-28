import torch
import torch.nn as nn
from .se_block import SEBlock


class SEBasicBlock(nn.Module):
    """
    带有 SE 模块的 ResNet 基础块

    SE 模块插入在第二个卷积之后、残差连接之前:
    Conv1 → BN → ReLU → Conv2 → BN → SE → + → ReLU
                                     ↑
                             残差连接 ─┘

    输入: (B, C, H, W)
    输出: (B, planes*expansion, H/stride, W/stride)
    """
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, reduction=16):
        super(SEBasicBlock, self).__init__()

        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        # SE 模块: 在第二个卷积后, 残差连接前
        self.se = SEBlock(planes, reduction)

        self.downsample = downsample
        self.stride = stride
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # 注意力: 在特征融合前重新校准通道重要性
        out = self.se(out)

        # 残差连接
        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

if __name__ == "__main__":
    x = torch.randn(2, 64, 32, 32)
    block = SEBasicBlock(64, 64)
    y = block(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    print(f"Block params: {sum(p.numel() for p in block.parameters())}")
