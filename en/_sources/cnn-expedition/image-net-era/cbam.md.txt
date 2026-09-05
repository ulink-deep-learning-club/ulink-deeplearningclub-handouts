(cbam)=
# CBAM：哪里重要 + 什么重要

{doc}`se-net` 让 CNN 学会了"看什么"（通道注意力）。但只看"什么"不够——**同一个特征在不同空间位置的意义完全不同**。

一张猫的图片中，"耳朵"特征出现在画面中央和出现在背景角落，含义天差地别。登山者不会在整个山壁上均匀用力——他们先看"抓什么"（裂缝还是岩棱），再看"抓哪里"（这个裂缝的中间段还是末端）。

CBAM（Convolutional Block Attention Module）{cite}`woo2018cbam` 把这两个问题合并成一个答案：**先选频道，再调区域**。

---

## 空间注意力

空间注意力（Spatial Attention）的目标是生成一个**空间注意力图**——与输入特征图同等宽高的二维权重图，重要的区域更亮，不重要的更暗。

```{admonition} 空间注意力的直觉
:class: tip

通道注意力是"给每个频道调音量"，空间注意力则是 **"给屏幕的每个区域调亮度"**。
```

输入特征图 $F \in \mathbb{R}^{C \times H \times W}$，空间注意力模块生成注意力图 $M_s \in \mathbb{R}^{1 \times H \times W}$：

```{math}
\tilde{F} = F \odot M_s
```

**关键问题**：如何从 $C$ 个通道生成 $1$ 个通道的空间注意力图？答案是**沿通道维度聚合信息**。

最常用的方法来自 CBAM {cite}`woo2018cbam`：

1. **通道池化**：对每个空间位置，分别计算该位置在所有通道上的平均值和最大值
2. **拼接**：把两个聚合图拼成 $2 \times H \times W$
3. **卷积**：用 $7 \times 7$ 卷积将 $2$ 个通道压缩回 $1$ 个通道
4. **激活**：Sigmoid 映射到 $(0,1)$

```{math}
M_s = \sigma(f^{7\times7}([\text{AvgPool}^c(F); \text{MaxPool}^c(F)]))
```

```{admonition} 为什么同时用平均池化和最大池化？
:class: note

平均池化捕获"该位置的整体激活强度"，最大池化捕获"该位置的最强响应"。两者互补。
```

```{figure} ../../../_static/images/cbam-channel-spatial-module.png
:width: 600px
:align: center

通道注意力（选频道）与空间注意力（调区域）的输入输出对比
```

## CBAM：通道 + 空间，两个都要

{ref}`se-net` 解决了"什么特征重要"（通道维度）。空间注意力解决了"哪里重要"（空间维度）。那么问题来了：**能不能两个都要？**

CBAM（Convolutional Block Attention Module）{cite}`woo2018cbam` 就是答案——它把通道注意力和空间注意力串联起来，形成一个更强大的注意力模块。

## 核心思想：先"选频道"再"调区域"

CBAM 的工作流程很直观：先判断哪些通道重要（通道注意力），再判断这些重要通道中的哪些空间位置最关键（空间注意力）。

```{admonition} CBAM的直觉
:class: tip

想象你在看一张照片找猫：
1. **通道注意力**：首先，你决定看"颜色"这个维度（而不是"纹理"或"亮度"），因为猫的颜色信息最有用。
2. **空间注意力**：然后，你在"颜色"维度下，聚焦到照片中特定的区域——猫所在的位置。

这就是 CBAM 的两步走：先"选对频道"，再"看对位置"。
```

```{figure} ../../../_static/images/cbam-block.png
:width: 600px
:align: center

CBAM 模块结构：通道注意力 → 空间注意力串联
```

```{mermaid}
graph LR
    A[输入特征 F] --> B[通道注意力<br/>Mc]
    B --> C[中间特征 F']
    C --> D[空间注意力<br/>Ms]
    D --> E[输出特征 F'']
    
    B1[Mc = σMLPAvgPoolF + MLPMaxPoolF] -.-> B
    D1[Ms = σf⁷×⁷AvgPoolF' ; MaxPoolF'] -.-> D
```

## 数学形式

CBAM 的完整计算过程为：

```{math}
F' = M_c(F) \otimes F
```

```{math}
F'' = M_s(F') \otimes F'
```

其中 $\otimes$ 表示逐元素乘法。

### 通道注意力部分

CBAM 的通道注意力与 SE-Net 类似，但增加了一个改进：**同时使用平均池化和最大池化**：

```{math}
M_c(F) = \sigma(\text{MLP}(\text{AvgPool}(F)) + \text{MLP}(\text{MaxPool}(F)))
```

```{literalinclude} code/channel_attention.py
:language: python
:linenos:
:caption: CBAM通道注意力模块实现
```

### 空间注意力部分

与上一节描述的一致，使用通道池化 + $7 \times 7$ 卷积生成空间注意力图。

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

```{literalinclude} code/cbam_basic_block.py
:language: python
:linenos:
:caption: CBAM-ResNet基础块实现
```

```{figure} ../../../_static/images/cbam-resnet.png
:width: 600px
:align: center

CBAM-ResNet：在残差块中插入CBAM模块
```

## SE-Net vs CBAM

| 对比项 | SE-Net | CBAM |
| ---------- | ---------- | ---------- |
| 注意力维度 | 仅通道 | 通道 + 空间 |
| 通道池化方式 | 仅平均池化 | 平均 + 最大池化 |
| 参数量增加 | $2C^2/r$ | $2C^2/r + k^2$ |
| 计算开销 | ~1% | ~2% |
| 适用场景 | 分类任务 | 检测/分割等需要定位的任务 |

CBAM 的实验表明 {cite}`woo2018cbam`：在 ImageNet 上，CBAM-ResNet-50 达到 **78.49%** Top-1 准确率，比基线 ResNet-50（76.15%）提升 **+2.34%**，比 SE-ResNet-50（77.62%）额外提升 **+0.87%**。

```{admonition} 为什么要先通道再空间？
:class: note

CBAM 的作者实验发现**先通道再空间**的效果优于先空间再通道或并行。直觉上：先通过通道注意力突出重要的语义特征，再通过空间注意力定位这些特征在图像中的具体位置——这个顺序更符合人的认知习惯："先知道看什么，再知道看哪里"。
```

### 下一步

通道注意力、空间注意力——你学会了让网络"看重点"。

但回头看这八年：AlexNet 首登、Inception 多尺度、ResNet 补给站、SE-Net/CBAM 聚焦——每一件装备都在回答"怎么爬得更高"。山一直是同一座：ImageNet 上的准确率。

**现在，山变了。** 新的问题不是"怎么爬得更高"，而是"怎么用最少的资源爬上去"。你的手机只有几瓦功耗，你的无人机只有一颗嵌入式芯片。预算几乎为零——还能登顶吗？

{doc}`mobilenet`——走。一个人，一把冰镐。

---

```{only} not latex

~~~{rubric} 参考文献
:heading-level: 2
~~~
```

~~~{bibliography}
:filter: docname in docnames
~~~
