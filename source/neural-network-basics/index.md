# 神经网络基础

## 学习目标

本章我们将把前一章学习的理论付诸实践，以MNIST手写数字识别为案例，掌握：

- **架构设计**：理解全连接网络与CNN的结构差异和设计思想
- **PyTorch实现**：从零搭建神经网络，完成训练、评估全流程
- **性能对比**：通过实验直观感受CNN的归纳偏置优势
- **缩放定律**：理解模型规模、数据量与性能的关系

```{admonition} 前置知识
:class: important

本章假设你已掌握{doc}`../math-fundamentals/index`中的核心概念：
- {ref}`computational-graph` 和 {ref}`back-propagation`
- {ref}`activation-functions` 的非线性作用
- {ref}`loss-functions` 和 {ref}`gradient-descent`
```

## 本章概览

| 章节 | 内容 | 关键点 |
|------|------|--------|
| {doc}`introduction` | MNIST案例与全连接/CNN对比预览 | 为什么需要CNN？ |
| {doc}`fc-layer-basics` | 全连接层原理与PyTorch实现 | 参数量爆炸问题 |
| {doc}`cnn-basics` | 卷积操作与参数共享机制 | 局部感受野、权值共享 |
| {doc}`le-net` | LeNet-5架构逐层解析 | 经典CNN设计模式 |
| {doc}`neural-training-basics` | 完整训练流程与监控 | 实践中的调试技巧 |
| {doc}`exp-cmp` | 全连接 vs CNN实验对比 | 数据说话 |
| {doc}`scaling-law` | 模型缩放定律理论 | 大模型时代的指导原则 |

## 本章定位

前一章我们学习了深度学习的**数学原理**：
- 计算图如何描述计算过程
- 激活函数如何引入非线性
- 反向传播如何高效计算梯度
- 梯度下降如何优化参数

本章我们进入**实践阶段**：
- 用PyTorch实现这些机制
- 理解不同架构的设计思想
- 通过实验验证理论
- 掌握训练调试技巧

**学习路径**：理论 → 实现 → 实验 → 洞察

```{admonition} 目录
:class: note
~~~{toctree}
:maxdepth: 2

introduction
fc-layer-basics
cnn-basics
le-net
neural-training-basics
exp-cmp
scaling-law
the-end
~~~
```

与 Kimi K2-0905, Kimi K2.5 合作
