import torch
import torch.nn as nn
from .se_block import SEBlock


class SEBasicBlock(nn.Module):
    """带有SE模块的ResNet基础块"""
    expansion = 1
    
    def __init__(self, inplanes, planes, stride=1, downsample=None, reduction=16):
        super(SEBasicBlock, self).__init__()
        
        # 第一个卷积层
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        
        # 第二个卷积层
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        
        # SE模块
        self.se = SEBlock(planes, reduction)
        
        # 下采样和激活函数
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
        
        # 应用SE模块
        out = self.se(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
            
        out += identity
        out = self.relu(out)
        
        return out


if __name__ == "__main__":
    # 测试SEBasicBlock
    x = torch.randn(2, 64, 32, 32)
    block = SEBasicBlock(64, 64)
    y = block(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    print(f"Block parameters: {sum(p.numel() for p in block.parameters())}")