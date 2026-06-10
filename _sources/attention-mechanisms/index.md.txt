(attention-mechanisms)=

# CNN 中的注意力机制

```{only} html
CNN 对所有特征一视同仁——无论背景还是关键目标。**如何让网络动态地知道"哪些特征更重要"？** 注意力机制就是答案：通过可学习的"放大镜"，关注重要特征、抑制无关特征。

~~~{rubric} 本章概览
:heading-level: 2
~~~

| 章节 | 内容 | 与前面章节的联系 |
| ---------- | ---------- | ---------- |
| {doc}`introduction` | 为什么 CNN 需要注意力？ | {doc}`../cnn-ablation-study/experiment-design` 的问题驱动 |
| {doc}`se-net` | 通道注意力（SE-Net） | 压缩→激励→缩放的直觉 |
| {doc}`spatial-attn` | 空间注意力 | 关注"哪里重要" |
| {doc}`cbam` | 通道+空间注意力（CBAM） | 两种注意力的组合策略 |
| {doc}`comparison` | 选择与应用 | 不同注意力的适用场景和性能数据 |
| {doc}`practice` | 实践指南 | 超参数调优、常见陷阱 |

~~~{rubric} 学习路径
:heading-level: 2
~~~

本章是在 CNN 基础上的**能力增强**：

~~~{mermaid}
graph LR
    A[CNN基础<br/>平等对待特征] --> B[发现问题<br/>不同特征重要性不同]
    B --> C[通道注意力<br/>SE-Net]
    C --> D[空间注意力<br/>关注哪里]
    D --> E[CBAM<br/>两者结合]
    E --> F[选择与应用<br/>实践指南]
~~~

**核心认知**：注意力不是替代 CNN，而是让 CNN "学会看重点"的可学习组件。

~~~{rubric} 前置知识
:heading-level: 2
~~~

| 前置章节 | 本章应用 |
| ---------- | ---------- |
| {doc}`../cnn-ablation-study/experiment-design` | 各组件贡献分析 → 注意力能进一步提升 |
| {doc}`../neural-network-basics/cnn-basics` | 卷积特征提取 → 通道/空间的重要性差异 |
| {doc}`../pytorch-practice/neural-network-module` | nn.Module 实现注意力模块 |
```

```{toctree}
:maxdepth: 2
:hidden:

introduction
se-net
spatial-attn
cbam
comparison
practice
the-end
```
