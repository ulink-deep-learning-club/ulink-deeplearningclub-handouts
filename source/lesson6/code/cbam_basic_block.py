import torch
import torch.nn as nn
from .cbam import CBAM


class CBAMBasicBlock(nn.Module):
    """带有CBAM模块的ResNet基础块"""
    expansion = 1
    
    def __init__(self, inplanes, planes, stride=1, downsample=None, reduction=16):
        super(CBAMBasicBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        
        # CBAM模块
        self.cbam = CBAM(planes, reduction)
        
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
        
        # 应用CBAM模块
        out = self.cbam(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
            
        out += identity
        out = self.relu(out)
        
        return out


if __name__ == "__main__":
    # 测试CBAMBasicBlock
    x = torch.randn(2, 64, 32, 32)
    block = CBAMBasicBlock(64, 64)
    y = block(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    print(f"Block parameters: {sum(p.numel() for p in block.parameters())}")