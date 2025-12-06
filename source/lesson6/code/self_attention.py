import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    """自注意力模块"""
    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()
        
        self.query_conv = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.key_conv = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, 1)
        
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, x):
        batch_size, C, width, height = x.size()
        
        # 生成query, key, value
        proj_query = self.query_conv(x).view(batch_size, -1, width * height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(batch_size, -1, width * height)
        proj_value = self.value_conv(x).view(batch_size, -1, width * height)
        
        # 计算注意力图
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        
        # 应用注意力
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, width, height)
        
        # 残差连接
        out = self.gamma * out + x
        
        return out


if __name__ == "__main__":
    # 测试SelfAttention
    x = torch.randn(2, 64, 32, 32)
    sa = SelfAttention(64)
    y = sa(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    print(f"SelfAttention parameters: {sum(p.numel() for p in sa.parameters())}")