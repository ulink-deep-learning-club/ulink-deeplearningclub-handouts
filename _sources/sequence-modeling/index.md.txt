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

~~~{mermaid}
graph LR
    A[RNN<br/>理解循环] --> B[LSTM<br/>理解门控]
    B --> C[注意力起源<br/>理解连接]
    C --> D[Transformer<br/>理解并行]
    D --> E[Mamba<br/>理解回归]
~~~

**核心认知**：这 35 年的故事不是让你记住所有架构细节，而是让你理解每个新设计**为什么要抛弃旧方案**——RNN 为什么需要 LSTM，LSTM 为什么需要注意力，Transformer 为什么又被 Mamba 挑战。理解了"为什么变"，就理解了"是什么"。

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
