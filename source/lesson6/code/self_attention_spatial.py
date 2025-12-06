import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttentionSpatial(nn.Module):
    """自注意力空间注意力模块"""
    def __init__(self, in_channels):
        super(SelfAttentionSpatial, self).__init__()
        
        self.query = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.key = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.value = nn.Conv2d(in_channels, in_channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        
    def forward(self, x):
        batch_size, C, H, W = x.size()
        
        # 生成query, key, value
        q = self.query(x).view(batch_size, -1, H * W).permute(0, 2, 1)  # (B, N, C')
        k = self.key(x).view(batch_size, -1, H * W)                     # (B, C', N)
        v = self.value(x).view(batch_size, -1, H * W)                   # (B, C, N)
        
        # 计算注意力图
        attn = torch.bmm(q, k)  # (B, N, N)
        attn = F.softmax(attn, dim=-1)
        
        # 应用注意力
        out = torch.bmm(v, attn.permute(0, 2, 1))  # (B, C, N)
        out = out.view(batch_size, C, H, W)
        
        # 残差连接
        out = self.gamma * out + x
        
        return out


if __name__ == "__main__":
    # 测试SelfAttentionSpatial
    x = torch.randn(2, 64, 32, 32)
    sas = SelfAttentionSpatial(64)
    y = sas(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    print(f"SelfAttentionSpatial parameters: {sum(p.numel() for p in sas.parameters())}")