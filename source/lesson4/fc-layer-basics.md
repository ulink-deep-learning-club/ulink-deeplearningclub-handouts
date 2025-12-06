# 全连接神经网络

## 架构设计

全连接神经网络（FCN）是最基础的神经网络架构，其中每一层的每个神经元都与下一层的所有神经元相连。

```python
import torch.nn as nn

class FullyConnectedNet(nn.Module):
    def __init__(self, input_size=784, hidden_sizes=[128, 64], num_classes=10):
        super(FullyConnectedNet, self).__init__()
        
        # 创建全连接层序列
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            prev_size = hidden_size
            
        layers.append(nn.Linear(prev_size, num_classes))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        x = x.view(x.size(0), -1)  # 展平输入
        return self.network(x)
```

## 参数分析

对于MNIST图像（28×28=784像素），一个简单的三层全连接网络参数数量为：

```{math}
\text{参数总数} = 784 \times 128 + 128 \times 64 + 64 \times 10 + (128 + 64 + 10) \approx 109,386
```

其中偏置项参数相对较少，主要参数来自权重矩阵。

## 局限性

全连接网络在处理图像数据时存在明显局限性：

1. **参数爆炸**：输入维度高导致参数数量巨大
2. **空间信息丢失**：展平操作破坏了图像的二维结构
3. **平移不变性差**：同一特征在不同位置需要重新学习
4. **计算效率低**：大量参数导致计算和内存开销大
