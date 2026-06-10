(math-fundamentals)=
# 计算图、反向传播与梯度下降：深度学习核心数学基础

```{only} html
深度学习的核心问题——**让机器从数据中学习规律**——本章将揭示答案背后的数学机制。

~~~{rubric} 本章概览
:heading-level: 2
~~~

| 章节 | 内容 | 核心收获 |
| ---------- | ---------- | ---------- |
| {doc}`introduction` | 问题引入与整体概览 | 为什么需要这些数学工具？ |
| {doc}`computational-graph` | 计算图原理 | 复杂计算的可视化表达 |
| {doc}`task-formulations` | 任务类型与优化目标 | 分类 vs 回归的数学表述 |
| {doc}`activation-functions` | 激活函数详解 | 非线性引入与决策边界 |
| {doc}`loss-functions` | 损失函数设计 | 如何量化预测误差 |
| {doc}`back-propagation` | 反向传播算法 | 高效计算梯度的链式法则 |
| {doc}`gradient-descent` | 梯度下降优化 | 参数更新的迭代策略 |
| {doc}`the-end` | 总结与展望 | 从理论到实践的过渡 |

~~~{rubric} 学习路径
:heading-level: 2
~~~

本章是整个系列的**理论基础层**：

~~~{mermaid}
graph LR
    A[建立直觉<br/>几何直观] --> B[理解原理<br/>数学机制]
    B --> C[动手实践<br/>代码实现]
    C --> D[后续章节<br/>架构与应用]
~~~

**核心认知**：我们不涉及复杂的网络架构，而是聚焦于让神经网络"工作"起来的核心数学机制——每个概念都有对应的代码实现。

~~~{rubric} 前置知识
:heading-level: 2
~~~

本章是深度学习系列的**理论基础**，不需要深度学习前置知识，但需要：

- **基础 Python 编程**：熟悉基本语法和函数
- **高中数学**：函数、导数概念
- **NumPy 入门**：建议了解基本数组操作
```

```{toctree}
:maxdepth: 2
:hidden:

introduction
computational-graph
task-formulations
activation-functions
loss-functions
back-propagation
gradient-descent
the-end
```
