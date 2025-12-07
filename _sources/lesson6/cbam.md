# Convolutional Block Attention Module (CBAM)

## 架构概述

CBAM由Sanghyun Woo等人于2018年提出，结合了通道注意力和空间注意力，形成串行的注意力机制。

```{figure} ../../_static/images/cbam-block.png
:width: 80%
:align: center

CBAM模块结构示意图
```

### 通道注意力模块

CBAM的通道注意力模块与SE-Net类似，但使用了最大池化和平均池化的组合：

```{math}
M_c(F) = \sigma(\text{MLP}(\text{AvgPool}(F)) + \text{MLP}(\text{MaxPool}(F)))
```

```{literalinclude} code/channel_attention.py
:language: python
:linenos:
:caption: CBAM通道注意力模块实现
```

### 空间注意力模块

空间注意力模块通过通道维度的池化和卷积生成空间注意力图：

```{math}
M_s(F) = \sigma(f^{7×7}([\text{AvgPool}(F); \text{MaxPool}(F)]))
```

其中 $f^{7×7}$ 表示7×7卷积，$[·;·]$ 表示通道拼接。

```{literalinclude} code/spatial_attention.py
:language: python
:linenos:
:caption: CBAM空间注意力模块实现
```

### 完整CBAM模块

```{literalinclude} code/cbam.py
:language: python
:linenos:
:caption: 完整CBAM模块实现
```

### CBAM-ResNet集成

```{figure} ../../_static/images/cbam-resnet.png
:width: 80%
:align: center

CBAM模块在ResNet中的集成方式
```

```{literalinclude} code/cbam_basic_block.py
:language: python
:linenos:
:caption: CBAM-ResNet基础块实现
```
