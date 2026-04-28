(fc-layer-basics)=
# 全连接神经网络

本章我们将实现最简单的神经网络架构——全连接层（Fully Connected Layer）。在深入之前，建议回顾{ref}`activation-functions`中讨论的**线性回归→多层感知机**的演变过程。全连接层源于感知机（Perceptron）{cite}`rosenblatt1958perceptron` 的思想，是多层感知机的现代实现。

## 全连接层的基本原理

全连接层（Fully Connected Layer），也称为线性层（Linear）或密集层（Dense），是神经网络中最基本的构建块。其核心思想是每个输入节点都与每个输出节点相连接。

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

这正是{ref}`activation-functions`中提到的**线性变换** $Wx + b$ 的矩阵形式，其中：
- $\mathbf{W} \in \mathbb{R}^{m \times n}$ 是权重矩阵（所有连接）
- $\mathbf{b} \in \mathbb{R}^m$ 是偏置向量

正如我们在{ref}`activation-functions`中看到的，如果没有非线性激活函数，多层全连接等价于单层。这就是为什么代码中需要使用 `F.relu()`。
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

## 参数爆炸：全连接的核心问题

235,146个参数看似不多，但考虑以下问题：

| 问题 | 说明 |
|------|------|
| **空间信息丢失** | 784维向量完全打乱了28×28的二维结构，相邻像素在向量中可能相距很远 |
| **参数效率低** | 每个像素都连接到每个神经元，无论距离多远，没有"局部性"概念 |
| **过拟合风险** | 参数量大，而MNIST是相对简单的10类分类任务 |

**直觉理解**：全连接网络把图像当作"表格"，而非"图片"。它不知道像素A和像素B是相邻的——这种空间关系的感知必须从零学习。

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
        # 展平：28x28 图像 → 784 维向量
        # 这就是"空间信息丢失"的发生点！二维结构变一维
        x = self.flatten(x)
        
        # 第一层：784 -> 256
        x = self.fc1(x)      # 线性变换 y = Wx + b
        x = F.relu(x)        # 非线性激活（见 activation-functions）
        x = self.dropout(x)  # 随机丢弃50%神经元，防止过拟合
        
        # 第二层：256 -> 128
        x = self.fc2(x)
        x = F.relu(x)        # 再次激活，增加表达能力
        x = self.dropout(x)
        
        # 输出层：128 -> 10（10个数字类别）
        # 注意：最后一层不加激活，CrossEntropyLoss内部会应用Softmax
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

## 核心问题：参数效率低下

全连接网络有**235,146个参数**，但大部分都是"浪费"的。

### 什么是参数效率？

$$
\text{参数效率} = \frac{\text{模型表达能力}}{\text{参数量}}
$$

| 指标 | 全连接网络 | 评价 |
|------|-----------|------|
| 参数量 | 235,146 | 过多 |
| MNIST准确率 | ~98% | 尚可 |
| **参数效率** | **低** | 用20万参数"记住"数据，而非学会规律 |

**问题本质**：全连接缺乏{ref}`inductive-bias`——它没有利用"图像是二维的"、"相邻像素相关"这些先验知识。从贝叶斯角度看，全连接相当于**无信息先验**，必须从零开始学习所有可能；而CNN利用**有信息先验**，直接关注有用的假设空间。

## 总结与展望

全连接网络虽然实现简单，但存在**致命缺陷**：

| 问题 | 影响 | CNN解决方案 |
|------|------|-------------|
| **参数爆炸** | 20万+参数，容易过拟合 | 局部感受野，只看相邻像素 |
| **空间信息丢失** | 不知道像素位置关系 | 保留2D结构，卷积核滑动 |
| **平移不变性差** | 同一特征需重复学习 | 权值共享，一个卷积核全图复用 |
| **归纳偏置弱** | 参数效率极低 | 内置局部性和平移不变性假设 |

### 关键对比预告

下一节我们将学习{doc}`cnn-basics`，通过两个核心设计将参数量从**20万**减少到**几百**：

- **局部感受野**：3×3卷积核只看相邻9个像素，而非全部784个
- **权值共享**：同一个卷积核滑过整张图像，参数重复使用

```{admonition} 实验预告
:class: tip

在{doc}`exp-cmp`中，我们将用实验数据对比：
- 参数量：全连接（235K）vs CNN（~60K）
- 准确率：两者都能达到98%+
- 训练速度：CNN收敛更快

这证明了**好的架构设计比单纯增加参数更重要**。
```

---

## 参考文献

```{bibliography}
:filter: docname in docnames
```
