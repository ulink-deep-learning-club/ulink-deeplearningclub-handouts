(model-architecture-design)=
# 架构心法：从会用装备到会设计装备

```{only} html
冰镐、绳索、补给站、放大镜——爬完两座山，你的装备包已经满了。LeNet 的卷积-池化-全连接、Inception 的多尺度、ResNet 的跳跃连接——每个设计都解决了一个具体问题。

**但下次面对一座全新的山，你该怎么选装备？** 架构心法就是把经验升维为原则：感受野、深度连接、注意力、效率——四个维度，四个旋钮。

本章以**登山者笔记**收尾——不是干巴巴的总结，而是回望三十年攀登史，把四件装备、演化脉络、规模规律缝成一张可以带走的地图。

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
| {doc}`the-end` | 🏁 登山者笔记 | 回望三十年——四件装备 + 演化史 + 下一座山 |

~~~{rubric} 学习路径
:heading-level: 2
~~~

本章将前面的具体技术**升维为方法论**：

~~~{mermaid}
graph LR
    A[⛰️ 下篇：未登峰<br/>八年八件装备] --> B[🔬 架构心法<br/>四维度提炼]
    B --> C[🏁 登山者笔记<br/>三十年回望]

    style A fill:#FF6F00,color:#fff
    style C fill:#1565C0,color:#fff
~~~

**核心认知**：好的架构设计不是"堆砌新技术"，而是理解每个设计选择解决了**哪个维度的什么问题**。

~~~{rubric} 前置知识
:heading-level: 2
~~~

| 前置章节 | 本章如何延伸 |
| ---------- | ---------- |
| {doc}`../practice-peak/cnn-basics` | 感受野的操控策略 |
| {doc}`../image-net-era/res-net` | 跳跃连接的通用原理 |
| {doc}`../image-net-era/inception` | 多尺度设计的多种实现 |
| {doc}`../image-net-era/se-net` | 注意力作为一种信息路由机制 |
| {doc}`../image-net-era/cbam` | 空间注意力与通道注意力的组合逻辑 |
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
