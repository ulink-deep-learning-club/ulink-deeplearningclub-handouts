# 卷积神经网络

## 卷积操作的基本原理

卷积操作是CNN的核心，它通过滑动窗口的方式在输入图像上应用（通过点积）滤波器（卷积核）来提取特征。这种思想最早由LeCun等人在1989年提出，并在后续的LeNet-5工作中得到完善。

```{figure} ../../_static/images/conv-process.png
:width: 40%
:align: center

卷积操作示意图
```

```{note}
**卷积操作的数学定义**

$$
(f * g)[m,n] = \sum_{i=-\infty}^{\infty}\sum_{j=-\infty}^{\infty} f[i,j] \cdot g[m-i, n-j]
$$

在离散图像处理中，卷积操作可以表示为：

$$
Y[i,j] = \sum_{u=0}^{k-1}\sum_{v=0}^{k-1} X[i+u, j+v] \cdot K[u,v] + b
$$

其中：
- $X$：输入特征图
- $K$：卷积核（滤波器）
- $b$：偏置项
- $k$：卷积核大小
```

简单来说，卷积操作通过将卷积核在输入特征图上滑动，计算局部区域的加权和，即该区域与卷积核的相似程度，来生成输出特征图。

## 特征图尺寸计算

对于输入尺寸为 $W \times H$，卷积核大小为 $K \times K$，步长为 $S$，填充为 $P$ 的情况，输出特征图的尺寸为：

```{math}
W_{out} = \left\lfloor \frac{W - K + 2P}{S} \right\rfloor + 1
```

```{math}
H_{out} = \left\lfloor \frac{H - K + 2P}{S} \right\rfloor + 1
```

```{admonition} MNIST实例计算
:class: example

对于28×28的MNIST图像，使用3×3卷积核，步长S=1，填充P=1：
- 输出宽度：$W_{out} = \lfloor \frac{28 - 3 + 2 \times 1}{1} \rfloor + 1 = 28$
- 输出高度：$H_{out} = \lfloor \frac{28 - 3 + 2 \times 1}{1} \rfloor + 1 = 28$
- 结论：输出特征图尺寸保持28×28不变

如果使用2×2池化，步长S=2，无填充P=0：
- 输出尺寸：$W_{out} = \lfloor \frac{28 - 2 + 0}{2} \rfloor + 1 = 14$
- 结论：池化将特征图尺寸减半
```

## 参数共享机制

卷积层的一个重要特性是参数共享。同一个卷积核在图像的不同位置重复使用，这大大减少了参数数量。

```{figure} ../../_static/images/conv-param-share.png
:width: 80%
:align: center

参数共享机制示意图
```

对于 $C_{\text{out}}$ 个输出通道，每个通道使用独立的卷积核：

```{math}
\text{Parameters} = C_{\text{out}} \times (K \times K \times C_{in} + 1)
```

其中 $C_{in}$ 是输入通道数。

以MNIST为例（单通道输入，32个3×3卷积核）：

```{math}
\text{Parameters} = 32 \times (3 \times 3 \times 1 + 1) = 32 \times 10 = 320
```

相比全连接层的数万参数，卷积层的参数数量显著减少了99%以上。

## 卷积层的优势

```{admonition} 卷积层的核心优势
:class: note

1. **局部连接**：每个神经元只连接输入图像的局部区域
2. **权值共享**：同一卷积核在整个图像上滑动，大幅减少参数
3. **平移不变性**：相同特征在不同位置被同一卷积核检测
4. **层次特征提取**：浅层提取边缘等低级特征，深层提取语义等高级特征
```

## 池化层

池化层用于降采样，减少特征图的空间尺寸：

- **最大池化（Max Pooling）**：取局部区域的最大值
- **平均池化（Average Pooling）**：取局部区域的平均值

池化操作不提供可学习参数，但能有效减少计算量和过拟合风险。

## PyTorch实现CNN

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        
        # 第一个卷积块：1 -> 32通道
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        # 第二个卷积块：32 -> 64通道
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        # 池化层
        self.pool = nn.MaxPool2d(2, 2)
        
        # 全连接层
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)
        
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        # 第一个卷积块：28x28 -> 14x14
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool(x)
        
        # 第二个卷积块：14x14 -> 7x7
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool(x)
        
        # 展平
        x = x.view(x.size(0), -1)
        
        # 全连接层
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

# 模型实例化
model = SimpleCNN()
print(f"CNN模型总参数数量: {sum(p.numel() for p in model.parameters()):,}")
```

## CNN的优缺点

```{list-table} 卷积神经网络优缺点对比
:header-rows: 1
:widths: 50 50

* - **优点**
  - **缺点**
* - 保留空间结构信息
  - 实现相对复杂
* - 参数数量少
  - 超参数选择敏感
* - 平移不变性
  - 计算复杂度高
* - 局部连接
  - 需要更多内存
* - 分层特征提取
  - 训练时间较长
```

```{admonition} 从全连接到卷积的演进
:class: warning

卷积神经网络的设计直接针对全连接网络的局限性：
1. **参数爆炸** → 参数共享大幅减少参数
2. **空间信息丢失** → 局部连接保留空间结构
3. **平移不变性差** → 平移不变性提高泛化能力
4. **计算效率低** → 分层特征提取提高计算效率

这种演进体现了深度学习架构设计中的“归纳偏置”思想：通过引入适合任务结构的先验知识，让模型更高效地学习。
