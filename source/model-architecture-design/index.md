(model-architecture-design)=
# CNN 架构改造：设计卷积网络的心法

```{only} html
LeNet 的卷积-池化-全连接、Inception 的多尺度并行、ResNet 的跳跃连接——每个设计都解决了一个具体问题。**这些设计思想之间有什么联系？我自己的模型该怎么改？** 本章提炼设计 CNN 的心法，从感受野、信息流、注意力、效率四个维度，建立系统化的架构改造方法论。

~~~{rubric} 本章范围
:heading-level: 2
~~~

聚焦于 **CNN（卷积神经网络）** 的架构设计。序列处理和生成模型的架构设计不在讨论范围内——详见 {ref}`sequence-modeling`。

~~~{rubric} 本章概览
:heading-level: 2
~~~

| 章节 | 内容 | 核心问题 |
| ---------- | ---------- | ---------- |
| {doc}`part1-intro` | 改造的本质与信息论诊断工具 | 瓶颈在哪——容量还是信息流？ |
| {doc}`part2-receptive-field` | 操控感受野：多尺度、空洞卷积 | 网络能看到多大范围？ |
| {doc}`part3-depth-connection` | 操控深度与连接：跳跃连接、特征融合 | 梯度能传回去吗？ |
| {doc}`part4-attention` | 操控注意力与长程依赖 | 网络知道看哪吗？远近能关联上吗？ |
| {doc}`part5-efficiency` | 操控效率：参数、计算、控制流、硬件四维优化 | 同样的准确率，为什么一个模型比另一个慢10倍？ |
| {doc}`part6-diagnosis` | 诊断与心法：反直觉案例与设计决策 | 我的模型出了什么问题？ |
| {doc}`the-end` | 总结 | 知识体系梳理与下一步方向 |

~~~{rubric} 学习路径
:heading-level: 2
~~~

本章将前面的具体技术**升维为方法论**：

~~~{mermaid}
graph LR
    A[Neural Network Basics<br/>了解架构组件] --> B[Attention Mechanisms<br/>掌握一种改造技术]
    B --> C[模型架构改造<br/>建立设计心法]
    C --> D[迁移学习、分割<br/>应用心法解决具体问题]
~~~

**核心认知**：好的架构设计不是"堆砌新技术"，而是理解每个设计选择解决了**哪个维度的什么问题**。

~~~{rubric} 前置知识
:heading-level: 2
~~~

| 前置章节 | 本章如何延伸 |
| ---------- | ---------- |
| {doc}`../neural-network-basics/cnn-basics` | 感受野的操控策略 |
| {doc}`../neural-network-basics/res-net` | 跳跃连接的通用原理 |
| {doc}`../neural-network-basics/inception` | 多尺度设计的多种实现 |
| {doc}`../attention-mechanisms/se-net` | 注意力作为一种信息路由机制 |
| {doc}`../attention-mechanisms/cbam` | 空间注意力与通道注意力的组合逻辑 |
```

```{toctree}
:maxdepth: 2
:hidden:

part1-intro
part2-receptive-field
part3-depth-connection
part4-attention
part5-efficiency
part6-diagnosis
the-end
```
