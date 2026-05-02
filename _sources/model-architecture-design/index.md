(model-architecture-design)=
# CNN 架构改造：设计卷积网络的心法

## 摘要

还记得 {doc}`../neural-network-basics/index` 中我们学习过的经典架构吗？LeNet 的卷积-池化-全连接、Inception 的多尺度并行、ResNet 的跳跃连接——每个设计都解决了一个具体问题。但有一个问题一直藏在水面下：**这些设计思想之间有什么联系？我自己的模型该怎么改？**

{doc}`../attention-mechanisms/index` 展示了通道注意力如何让网络"学会看重点"，但注意力只是改造武器库中的一种。真正的问题是：**面对一个表现不佳的模型，你该从哪下手？加注意力？改连接？还是换卷积方式？**

本章要回答的正是这个核心问题。我们不介绍更多"新技术"，而是提炼**设计 CNN 的心法**——从感受野、信息流动、注意力、计算效率四个维度，建立一套系统化的架构改造方法论。

```{admonition} 本章范围
:class: note

本章聚焦于 **CNN（卷积神经网络）** 的架构设计。虽然部分原则（如跳跃连接、注意力）具有跨架构的通用性，但感受野、空洞卷积、DW 卷积等策略是 CNN 特有的。序列处理（RNN、Transformer）和生成模型（扩散模型、GAN）的架构设计不在本章讨论范围内——它们将在后续章节中涉及。
```

```{admonition} 学习目标
:class: important

完成本章后，你将能够：
1. **用信息论视角诊断瓶颈**：区分"容量不够"和"信息传不过去"
2. **操控感受野**：根据目标大小选择合适的多尺度策略
3. **操控深度与连接**：用跳跃连接和特征融合保证信息流动
4. **操控注意力**：知道什么时候需要通道注意力、空间注意力或长程依赖
5. **操控效率**：在效果与速度之间做出合理权衡
6. **建立设计直觉**：面对任何模型，能自主判断需要改什么、怎么改
```

## 本章概览

| 章节 | 内容 | 核心问题 |
|------|------|----------|
| {doc}`part1-intro` | 改造的本质与信息论诊断工具 | 瓶颈在哪——容量还是信息流？ |
| {doc}`part2-receptive-field` | 操控感受野：多尺度、空洞卷积 | 网络能看到多大范围？ |
| {doc}`part3-depth-connection` | 操控深度与连接：跳跃连接、特征融合 | 梯度能传回去吗？ |
| {doc}`part4-attention` | 操控注意力与长程依赖 | 网络知道看哪吗？远近能关联上吗？ |
| {doc}`part5-efficiency` | 操控效率：DW 卷积与 Bottleneck | 如何在效果与速度间权衡？ |
| {doc}`part6-diagnosis` | 诊断与心法：反直觉案例与设计决策 | 我的模型出了什么问题？ |
| {doc}`the-end` | 总结 | 知识体系梳理与下一步方向 |

## 学习路径

本章将前面的具体技术**升维为方法论**：

```{mermaid}
graph LR
    A[Neural Network Basics<br/>了解架构组件] --> B[Attention Mechanisms<br/>掌握一种改造技术]
    B --> C[模型架构改造<br/>建立设计心法]
    C --> D[迁移学习、分割<br/>应用心法解决具体问题]
```

**核心认知**：好的架构设计不是"堆砌新技术"，而是理解每个设计选择解决了**哪个维度的什么问题**。

## 本章定位

前面章节我们积累了丰富的"零件知识"——卷积、池化、跳跃连接、注意力模块。本章将这些零件组合成**一套系统化的设计方法论**。

你不是在学"ResNet 是什么"，而是在学"为什么 ResNet 这样设计"。
你不是在学"SE-Net 怎么用"，而是在学"什么时候该加注意力、加哪种"。

| 前置章节 | 本章如何延伸 |
|---------|------------|
| {doc}`../neural-network-basics/cnn-basics` | 感受野的操控策略 |
| {doc}`../neural-network-basics/res-net` | 跳跃连接的通用原理（不是只有 ResNet）|
| {doc}`../neural-network-basics/inception` | 多尺度设计的多种实现 |
| {doc}`../attention-mechanisms/se-net` | 注意力作为一种信息路由机制 |
| {doc}`../attention-mechanisms/cbam` | 空间注意力与通道注意力的组合逻辑 |

## 前置要求

```{admonition} 学习本章前，请确保你已经掌握
:class: caution

本章假设你已熟悉以下内容：

1. **CNN 基础知识**：卷积、池化、感受野（{doc}`../neural-network-basics/cnn-basics`）
2. **跳跃连接**：ResNet 的残差连接原理（{doc}`../neural-network-basics/res-net`）
3. **注意力机制**：SE-Net 与 CBAM（{doc}`../attention-mechanisms/index`）
4. **PyTorch 实践**：能搭建和训练模型（{doc}`../pytorch-practice/index`）
```

```{admonition} 还没掌握？
:class: tip

建议先完成 {doc}`../neural-network-basics/index` 和 {doc}`../attention-mechanisms/index` 的学习。本章假设你已熟悉这些架构组件。
```

## 目录

```{toctree}
:maxdepth: 2

part1-intro
part2-receptive-field
part3-depth-connection
part4-attention
part5-efficiency
part6-diagnosis
the-end
```
