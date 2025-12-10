# 全连接神经网络

## 全连接层的基本原理

全连接层（Fully Connected Layer），也称为线性层或密集层，是神经网络中最基本的构建块。其核心思想是每个输入节点都与每个输出节点相连接。

```{tikz} 全连接层结构示意图

\begin{tikzpicture}[scale=0.8]
    % 输入层
    \foreach \i in {1,2,3,4,5}
        \node[circle, draw=blue!50, fill=blue!20, minimum size=0.6cm] (in\i) at (0,\i) {};
    
    % 输出层
    \foreach \i in {1,2,3}
        \node[circle, draw=red!50, fill=red!20, minimum size=0.6cm] (out\i) at (4,\i+1) {};
    
    % 全连接
    \foreach \i in {1,2,3,4,5}
        \foreach \j in {1,2,3}
            \draw[->, gray!50] (in\i) -- (out\j);
    
    \node at (-1, -0.5) {输入层 (5个神经元)};
    \node at (5, -0.5) {输出层 (3个神经元)};
    \node at (2, -1.5) {每个输入都连接到每个输出};
\end{tikzpicture}
```

```{note}
**全连接层的数学表达**

对于输入向量 $\mathbf{x} \in \mathbb{R}^n$ 和输出向量 $\mathbf{y} \in \mathbb{R}^m$，全连接层的变换为：

$$
\mathbf{y} = \mathbf{W}\mathbf{x} + \mathbf{b}
$$

其中 $\mathbf{W} \in \mathbb{R}^{m \times n}$ 是权重矩阵，$\mathbf{b} \in \mathbb{R}^m$ 是偏置向量。
```

## 参数数量分析

对于从 $n$ 维到 $m$ 维的全连接层，参数总数为：

```{math}
\text{Parameters} = m \times n + m = m(n + 1)
```

以MNIST为例，如果将28×28=784像素的图像直接输入到全连接层：

- 第一层：假设有256个神经元，参数数量为 $256 \times 784 + 256 = 200,960$
- 第二层：从256到128个神经元，参数数量为 $128 \times 256 + 128 = 32,896$
- 输出层：从128到10个类别，参数数量为 $10 \times 128 + 10 = 1,290$

**总参数数量：** 235,146个参数

## PyTorch实现

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class FullyConnectedNet(nn.Module):
    def __init__(self, input_size=784, hidden_size=256, num_classes=10):
        super(FullyConnectedNet, self).__init__()
        # 将28x28图像展平为784维向量
        self.flatten = nn.Flatten()
        
        # 全连接层堆叠
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, num_classes)
        
        # Dropout层用于防止过拟合
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        # 展平输入
        x = self.flatten(x)
        
        # 第一层：784 -> 256
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        # 第二层：256 -> 128
        x = self.fc2(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        # 输出层：128 -> 10
        x = self.fc3(x)
        
        return x

# 模型实例化
model = FullyConnectedNet()
print(f"模型总参数数量: {sum(p.numel() for p in model.parameters()):,}")
```

## 全连接层的优缺点

```{list-table} 全连接层优缺点对比
:header-rows: 1
:widths: 50 50

* - **优点**
  - **缺点**
* - 实现简单直观
  - 忽略空间结构信息
* - 理论成熟完善
  - 参数数量巨大
* - 易于理解和调试
  - 容易过拟合
* - 计算效率高（小模型）
  - 对平移不具备鲁棒性
* - 适用于非结构化数据
  - 需要大量训练数据
```

## 局限性

全连接网络在处理图像数据时存在明显局限性：

1. **参数爆炸**：输入维度高导致参数数量巨大
2. **空间信息丢失**：展平操作破坏了图像的二维结构
3. **平移不变性差**：同一特征在不同位置需要重新学习
4. **计算效率低**：大量参数导致计算和内存开销大

```{admonition} 为什么需要卷积神经网络？
:class: warning

全连接网络的这些局限性正是卷积神经网络（CNN）被提出的原因。CNN通过局部连接、参数共享和池化操作，有效地解决了上述问题，成为图像处理任务的首选架构。
```
