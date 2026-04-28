import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):
    """
    多头注意力模块

    将 Q, K, V 投影到 h 个不同的子空间, 独立计算注意力后拼接
    MultiHead(Q,K,V) = Concat(head_1, ..., head_h)·W_O
    其中 head_i = Attention(Q·W_Q_i, K·W_K_i, V·W_V_i)

    使用 1×1 Conv2d 实现投影, 适合 CNN 特征图
    输入: (B, C, H, W)
    输出: (B, C, H, W)
    """
    def __init__(self, in_channels, num_heads=8):
        super(MultiHeadAttention, self).__init__()

        self.num_heads = num_heads
        self.head_dim = in_channels // num_heads

        assert self.head_dim * num_heads == in_channels, "in_channels 必须能被 num_heads 整除"

        # QKV 联合投影: (B, C, H, W) → (B, 3C, H, W)
        self.qkv = nn.Conv2d(in_channels, in_channels * 3, 1)
        # 输出投影: 拼接后映射回 C 维
        self.proj = nn.Conv2d(in_channels, in_channels, 1)

    def forward(self, x):
        batch_size, C, H, W = x.size()
        N = H * W

        # QKV 投影并分头
        # (B, 3C, H, W) → (B, 3, h, d, N) 其中 C = h*d, N = H*W
        qkv = self.qkv(x).reshape(batch_size, 3, self.num_heads, self.head_dim, N)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]  # 各 (B, h, d, N)

        # 转置为 (B, h, N, d) — 标准注意力需要 N 维在最后第二维
        q, k, v = q.transpose(-2, -1), k.transpose(-2, -1), v.transpose(-2, -1)

        # 缩放点积注意力: Attention = softmax(QK^T/√d)
        # q @ k^T: (B, h, N, d) @ (B, h, d, N) → (B, h, N, N)
        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn = attn.softmax(dim=-1)  # (B, h, N, N)

        # 加权聚合: (B, h, N, N) @ (B, h, N, d) → (B, h, N, d)
        out = (attn @ v).transpose(1, 2).reshape(batch_size, C, H, W)
        out = self.proj(out)  # 输出投影

        return out

if __name__ == "__main__":
    x = torch.randn(2, 64, 32, 32)
    mha = MultiHeadAttention(64, num_heads=8)
    y = mha(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    print(f"MultiHeadAttention params: {sum(p.numel() for p in mha.parameters())}")
