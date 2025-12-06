import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):
    """多头注意力模块"""
    def __init__(self, in_channels, num_heads=8):
        super(MultiHeadAttention, self).__init__()
        
        self.num_heads = num_heads
        self.head_dim = in_channels // num_heads
        
        assert self.head_dim * num_heads == in_channels, "in_channels必须能被num_heads整除"
        
        # 为每个头创建独立的线性变换
        self.qkv = nn.Conv2d(in_channels, in_channels * 3, 1)
        self.proj = nn.Conv2d(in_channels, in_channels, 1)
        
    def forward(self, x):
        batch_size, C, H, W = x.size()
        
        # 生成Q, K, V
        qkv = self.qkv(x).reshape(batch_size, 3, self.num_heads, self.head_dim, H * W)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
        
        # 缩放点积注意力
        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn = attn.softmax(dim=-1)
        
        # 应用注意力
        out = (attn @ v).transpose(1, 2).reshape(batch_size, C, H, W)
        out = self.proj(out)
        
        return out


if __name__ == "__main__":
    # 测试MultiHeadAttention
    x = torch.randn(2, 64, 32, 32)
    mha = MultiHeadAttention(64, num_heads=8)
    y = mha(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    print(f"MultiHeadAttention parameters: {sum(p.numel() for p in mha.parameters())}")