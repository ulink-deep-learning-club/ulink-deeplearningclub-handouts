import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    """
    自注意力模块 (Non-local 风格)
    
    核心: 计算所有空间位置两两之间的相似度, 用相似度加权聚合特征
    实现 QKV 形式: Attention(Q,K,V) = softmax(QK^T/√d)V

    输入: (B, C, H, W)
    输出: (B, C, H, W) — 带残差连接
    """
    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()

        # 1×1 卷积生成 Q, K, V (降维 Q,K 到 C/8 减少计算量)
        self.query_conv = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.key_conv = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, 1)

        # 可学习的缩放因子, 初始为 0 (先学习局部特征, 再逐步引入全局信息)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, C, H, W = x.size()
        N = H * W  # 空间位置数

        # Q: (B, C/8, N) → (B, N, C/8) — 每个位置一个查询向量
        proj_query = self.query_conv(x).view(batch_size, -1, N).permute(0, 2, 1)
        # K: (B, C/8, N) — 每个位置一个键向量
        proj_key = self.key_conv(x).view(batch_size, -1, N)
        # V: (B, C, N) — 每个位置一个值向量
        proj_value = self.value_conv(x).view(batch_size, -1, N)

        # 注意力矩阵: (B, N, N)
        # energy[i,j] = query_i · key_j^T = 位置 i 与位置 j 的相似度
        energy = torch.bmm(proj_query, proj_key)
        # softmax 将每行归一化为概率: attention[i,:] 是位置 i 对所有位置的注意力分布
        attention = self.softmax(energy)

        # 加权聚合: (B, C, N) — 每个位置的输出是 V 的加权和
        # out[:,:,j] = sum_k V[:,:,k] * attention[j,k]
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, H, W)

        # 残差连接: gamma 控制全局信息的注入强度
        # gamma=0 时退化为恒等映射, 训练中逐步学习全局依赖
        out = self.gamma * out + x

        return out

if __name__ == "__main__":
    x = torch.randn(2, 64, 32, 32)
    sa = SelfAttention(64)
    y = sa(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    print(f"SelfAttention params: {sum(p.numel() for p in sa.parameters())}")
