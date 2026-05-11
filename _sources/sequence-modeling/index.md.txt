(sequence-modeling)=
# 序列建模：从 RNN 到 Transformer 到 Mamba

```{only} html
CNN 处理静态图像，但文字、音乐、视频都有**时间顺序**。**如何让神经网络理解"顺序"？** 从 RNN 到 Transformer 再到 Mamba——35 年的思想演化，一章讲完。

~~~{rubric} 本章概览
:heading-level: 2
~~~

| 章节 | 内容 | 与前面章节的联系 |
| -------- | ---------- | ----------------- |
| {doc}`introduction` | 从空间到时间：序列建模的问题引入 | 为什么顺序很重要 |
| {doc}`rnn-basics` | RNN原理、BPTT、梯度消失 | {ref}`gradient-vanishing` 在序列上的具体体现 |
| {doc}`lstm` | LSTM门控机制、细胞状态、梯度高速公路 | {ref}`res-net` 的残差思想在序列上的应用 |
| {doc}`from-rnn-to-attention` | 因果注意力的直觉起源 | {ref}`inductive-bias` 如何从"时序因果"走向"全局关联" |
| {doc}`transformer` | 自注意力、多头、FFN、O(n²)问题 | 与 {ref}`attention-mechanisms` 的两种注意力对比 |
| {doc}`mamba-intro` | 状态空间模型与RNN思想的回归 | 选择性机制——{ref}`lstm` 门控思想的延续 |
| {doc}`the-end` | 总结与对比 | RNN/LSTM vs Transformer vs Mamba 的系统对比 |

~~~{rubric} 学习路径
:heading-level: 2
~~~

本章讲述了一个"出发—迷失—回归"的思想故事：

~~~{mermaid}
graph LR
    A[RNN<br/>模仿大脑<br/>串行处理] -->|问题| B[梯度消失<br/>长程依赖]
    B -->|缓解| C[LSTM<br/>门控机制<br/>梯度高速公路]
    C -->|根本矛盾<br/>仍在| D[因果注意力<br/>连接所有前序]
    D -->|极致化| E[Transformer<br/>纯注意力<br/>O n²]
    E -->|反思| F[Mamba<br/>回归RNN效率<br/>选择性SSM]
~~~

**核心认知**：这不是简单的技术堆叠，而是一个思想在螺旋上升——从 RNN 出发，经历注意力的彻底革命，最终在更高的层次上回归 RNN 的效率哲学。

~~~{rubric} 前置知识
:heading-level: 2
~~~

- **梯度消失**：{ref}`gradient-vanishing` 和 Jacobian 连乘分析
- **归纳偏置**：{ref}`inductive-bias` 中的架构先验假设
- **计算图**：{ref}`computational-graph` 中数据流动和梯度回流的概念
- **CNN注意力与自注意力的区别**：{doc}`../attention-mechanisms/introduction` 中通道/空间注意力的基本概念
```

```{toctree}
:maxdepth: 2
:hidden:

introduction
rnn-basics
lstm
from-rnn-to-attention
transformer
mamba-intro
the-end
```
