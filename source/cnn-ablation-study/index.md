(cnn-ablation-study)=

# CNN 消融研究：理解卷积神经网络各组件的作用

```{only} html
每个组件"是什么"我们都知道，但**每个组件对最终效果的真正贡献是多少？** 移除卷积层、换掉激活函数——本章通过控制变量法，用实验数据回答这些问题。

~~~{rubric} 本章概览
:heading-level: 2
~~~

| 章节 | 内容 | 与前面章节的联系 |
| ---------- | ---------- | ---------- |
| {doc}`introduction` | 什么是消融研究？为什么要做？ | 科学方法论：控制变量法 |
| {doc}`experiment-design` | 基线模型、实验方案、结果分析 | {doc}`../pytorch-practice/using-framework` 的实验管理 |
| {doc}`implementation` | 完整的 PyTorch 代码实现 | {doc}`../pytorch-practice/train-workflow` 的实践 |
| {doc}`the-end` | 总结与学习路径 | 知识体系梳理 |

~~~{rubric} 学习路径
:heading-level: 2
~~~

本章是从"知道怎么做"到"知道为什么这样做"的**思维跃迁**：

~~~{mermaid}
graph LR
    A[理论学习<br/>CNN组件] --> B[基线模型<br/>跑通完整代码]
    B --> C[消融实验1<br/>移除卷积层]
    C --> D[消融实验2<br/>更换激活函数]
    D --> E[消融实验3<br/>移除Dropout]
    E --> F[数据分析<br/>组件重要性排序]
    F --> G[设计原则<br/>优化新架构]
~~~

**核心认知**：消融研究不是"破坏"，而是"理解"——通过系统地做减法，看清每个组件的真实贡献。

**关键原则**：每次只改一个变量，确保结果可归因。

~~~{rubric} 前置知识
:heading-level: 2
~~~

本章假设你已掌握完整训练流程和 CNN 各组件原理，建议完成 {doc}`../pytorch-practice/using-framework` 和 {doc}`../transfer-learning/index` 后再学习。

| 前置章节 | 本章应用 |
| ---------- | ---------- |
| {doc}`../neural-network-basics/cnn-basics` | 理解卷积层、池化层的作用，作为消融对象 |
| {doc}`../neural-network-basics/neural-training-basics` | 应用批归一化、Dropout 等正则化技术 |
| {doc}`../pytorch-practice/neural-network-module` | 搭建实验用的 CNN 模型 |
| {doc}`../pytorch-practice/train-workflow` | 实现训练循环，记录实验数据 |
| {doc}`../pytorch-practice/using-framework` | 使用框架管理实验、配置和模型 |
| {doc}`../transfer-learning/index` | 理解微调背后的"组件重要性"思想 |

**预计耗时**：2-4 周（需要训练多个模型进行对比实验）
```

```{toctree}
:maxdepth: 2
:hidden:

introduction
experiment-design
implementation
the-end
```
