# Squeeze-and-Excitation Networks (SE-Net)

## 架构概述

SE-Net由Jie Hu等人于2017年提出，核心思想是通过学习通道间的依赖关系，自适应地重新校准通道特征响应。

```{figure} ../../_static/images/se-block.png
:width: 80%
:align: center

SE模块结构示意图
```

## SE模块的三个阶段

### 1. Squeeze操作

通过全局平均池化将每个通道的二维特征压缩为一个标量：

```{math}
z_c = \frac{1}{H \times W} \sum_{i=1}^H \sum_{j=1}^W x_c(i,j)
```

其中 $x_c$ 是第 $c$ 个通道的特征图，$H \times W$ 是空间维度。

### 2. Excitation操作

通过两个全连接层学习通道间的非线性关系：

```{math}
s = \sigma(W_2 \delta(W_1 z))
```

其中：

- $W_1 \in \mathbb{R}^{C/r \times C}$：降维层，减少参数和计算量
- $W_2 \in \mathbb{R}^{C \times C/r}$：升维层，恢复原始维度
- $\delta$：ReLU激活函数
- $\sigma$：Sigmoid激活函数，输出[0,1]的注意力权重
- $r$：压缩比（通常为16）

### 3. Scale操作

将学习到的注意力权重与原始特征图逐通道相乘：

```{math}
\tilde{x}_c = s_c \cdot x_c
```

## PyTorch实现

```{literalinclude} code/se_block.py
:language: python
:linenos:
:caption: SE模块实现
```

## 数学分析

### 参数数量

对于输入通道数 $C$，压缩比 $r$，SE模块的参数数量为：

```{math}
\text{Params} = \frac{C}{r} \times C + C \times \frac{C}{r} = \frac{2C^2}{r}
```

当 $C=256, r=16$ 时，参数数量为 $2 \times 256^2 / 16 = 8,192$，相对于卷积层的参数可以忽略不计。

### 计算复杂度

SE模块的计算开销主要来自全连接层：

```{math}
\text{FLOPs} \approx \frac{2C^2}{r} + C \times H \times W
```

其中 $H \times W$ 是特征图尺寸。通常SE模块增加的计算开销小于1%。

## SE-ResNet集成

```{figure} ../../_static/images/se-resnet.png
:width: 60%
:align: center

SE模块在ResNet中的集成方式
```

```{literalinclude} code/se_basic_block.py
:language: python
:linenos:
:caption: SE-ResNet基础块实现
```
