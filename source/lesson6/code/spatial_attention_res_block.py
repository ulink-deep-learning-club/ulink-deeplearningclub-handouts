import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialAttentionResBlock(nn.Module):
    """带有空间注意力的残差块"""
    def __init__(self, in_channels, out_channels, stride=1, downsample=None, attention_type='conv'):
        super(SpatialAttentionResBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        
        # 空间注意力模块
        if attention_type == 'conv':
            self.spatial_attention = nn.Sequential(
                nn.Conv2d(out_channels, 1, kernel_size=7, padding=3, bias=False),
                nn.Sigmoid()
            )
        elif attention_type == 'pool':
            self.spatial_attention = nn.Sequential(
                nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
                nn.Sigmoid()
            )
        else:
            raise ValueError(f"Unknown attention_type: {attention_type}")
        
        self.attention_type = attention_type
        
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        # 应用空间注意力
        if self.attention_type == 'conv':
            attn = self.spatial_attention(out)
        else:  # pool
            avg_out = torch.mean(out, dim=1, keepdim=True)
            max_out, _ = torch.max(out, dim=1, keepdim=True)
            concat = torch.cat([avg_out, max_out], dim=1)
            attn = self.spatial_attention(concat)
        
        out = out * attn
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = F.relu(out)
        
        return out


if __name__ == "__main__":
    # 测试SpatialAttentionResBlock
    x = torch.randn(2, 64, 32, 32)
    
    # 测试conv类型
    block1 = SpatialAttentionResBlock(64, 128, stride=2, attention_type='conv')
    y1 = block1(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape (conv attention): {y1.shape}")
    print(f"Block parameters: {sum(p.numel() for p in block1.parameters())}")
    
    # 测试pool类型
    block2 = SpatialAttentionResBlock(64, 128, stride=2, attention_type='pool')
    y2 = block2(x)
    print(f"Output shape (pool attention): {y2.shape}")
    print(f"Block parameters: {sum(p.numel() for p in block2.parameters())}")