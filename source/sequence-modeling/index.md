(sequence-modeling)=
# 序列建模：从 RNN 到 Transformer 到 Mamba

```{toctree}
:maxdepth: 2
:hidden:

rnn-basics
lstm
from-rnn-to-attention
transformer
mamba-intro
the-end
```

## 摘要

到目前为止，我们一直在处理图像 ~~（因为涉及到的数学知识不是太深）~~ ——{doc}`../neural-network-basics/cnn-basics` 中的卷积核在二维像素格上滑动，{doc}`../attention-mechanisms/index` 中的注意力模块在通道和空间维度上重加权。这些架构共享一个隐含假设：**输入是静态的、定长的、没有先后顺序的**。

但现实中的很多数据不是这样。一段文字、一首歌、一段视频——它们都有**时间顺序**。单词的排列决定语义，音节的先后决定旋律，帧的连续决定动作。

本章要回答的核心问题是：**如何让神经网络理解"顺序"？**

从受到大脑启发的循环神经网络（RNN），到用全局注意力彻底替代循环的 Transformer，再到回归 RNN 效率哲学的新架构 Mamba——我们将追溯一条横跨 35 年的思想演化之路。

```{admonition} 学习目标
:class: important

完成本章后，你将能够：
1. **理解RNN的原理和局限**：解释循环结构如何模拟序列处理，以及为什么长程依赖和梯度消失让它失效
2. **掌握LSTM的门控机制**：理解细胞状态和隐状态的分离，以及遗忘门、输入门、输出门如何管理长期记忆
3. **掌握因果注意力的直觉来源**：理解"让每个隐状态与所有前序状态建立联系"如何自然导出注意力机制
4. **理解Transformer的核心设计**：掌握自注意力（Q/K/V）、多头机制、位置编码、FFN的设计哲学，以及O(n²)的来源
5. **认识Mamba的思想回归**：理解状态空间模型为什么是"现代版RNN"，以及选择性机制如何解决长程依赖
```

## 本章概览

| 章节 | 内容 | 与前面章节的联系 |
| -------- | ---------- | ----------------- |
| {doc}`rnn-basics` | RNN原理、BPTT、梯度消失 | {ref}`gradient-vanishing` 在序列上的具体体现 |
| {doc}`lstm` | LSTM门控机制、细胞状态、梯度高速公路 | {ref}`res-net` 的残差思想在序列上的应用 |
| {doc}`from-rnn-to-attention` | 因果注意力的直觉起源 | {ref}`inductive-bias` 如何从"时序因果"走向"全局关联" |
| {doc}`transformer` | 自注意力、多头、FFN、O(n²)问题 | 与 {ref}`attention-mechanisms` 的两种注意力对比 |
| {doc}`mamba-intro` | 状态空间模型与RNN思想的回归 | 选择性机制——{ref}`lstm` 门控思想的延续 |
| {doc}`the-end` | 总结与对比 | RNN/LSTM vs Transformer vs Mamba 的系统对比 |

## 学习路径

本章讲述了一个"出发—迷失—回归"的思想故事：

```{mermaid}
graph LR
    A[RNN<br/>模仿大脑<br/>串行处理] -->|问题| B[梯度消失<br/>长程依赖]
    B -->|缓解| C[LSTM<br/>门控机制<br/>梯度高速公路]
    C -->|根本矛盾<br/>仍在| D[因果注意力<br/>连接所有前序]
    D -->|极致化| E[Transformer<br/>纯注意力<br/>O n²]
    E -->|反思| F[Mamba<br/>回归RNN效率<br/>选择性SSM]
```

**核心认知**：这不是简单的技术堆叠，而是一个思想在螺旋上升——从 RNN 出发，经历注意力的彻底革命，最终在更高的层次上回归 RNN 的效率哲学。

## 本章定位

前面章节我们聚焦于**空间结构**的建模：

- CNN 用局部感受野和权值共享捕捉二维图像的局部模式（{doc}`../neural-network-basics/cnn-basics`）
- 注意力机制（SE-Net、CBAM）在通道和空间维度的特征上动态重加权（{doc}`../attention-mechanisms/index`）

本章转向**时序结构**的建模：

- 如何设计架构捕捉序列中的先后依赖？
- 如何在长序列中保持信息不衰减？
- 如何在效率和表达能力之间取得平衡？

**学习路径**：理解RNN的直觉 → 识别其数学缺陷 → LSTM的门控缓解 → 追踪注意力的诞生 → 理解Transformer的成功与代价 → 认识Mamba的回归

## 前置要求

```{admonition} 学习本章前，请确保你已经掌握
:class: caution

1. **梯度消失的数学根源**：{ref}`gradient-vanishing` 和 {ref}`gradient-vanishing-math` 中的 Jacobian 连乘分析
2. **归纳偏置的概念**：{ref}`inductive-bias` 中讨论的架构先验假设
3. **计算图的直觉**：{ref}`computational-graph` 中数据流动和梯度回流的基本概念
4. **CNN注意力与自注意力的区别**：{doc}`../attention-mechanisms/introduction` 中通道/空间注意力的基本概念（本章的自注意力是一种不同的"注意"）
```

```{admonition} 还没掌握？
:class: tip

如果梯度消失和 BPTT 的概念还不够清晰，建议先回顾：
- {ref}`gradient-vanishing-math`：Jacobian 连乘如何导致梯度指数级衰减
- {ref}`inductive-bias`：好的架构设计如何把先验内置到模型中
```
