(cbam)=
# 通道+空间注意力：CBAM

{ref}`se-net` 解决了"什么特征重要"（通道维度），{doc}`spatial-attn` 解决了"哪里重要"（空间维度）。那么问题来了：**能不能两个都要？**

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

与 {doc}`spatial-attn` 中描述的一致，使用通道池化 + $7 \times 7$ 卷积生成空间注意力图。

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

## SE-Net vs CBAM

| 对比项 | SE-Net | CBAM |
|--------|--------|------|
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

理解了 SE-Net、空间注意力和 CBAM 后，你可能想知道：**实际项目中应该选哪个？** {doc}`comparison` 将通过实验数据帮你做出选择。

---

## 参考文献

```{bibliography}
:filter: docname in docnames
```
