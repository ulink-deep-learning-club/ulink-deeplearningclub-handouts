(math-fundamentals)=
# 计算图、反向传播与梯度下降：深度学习核心数学基础

## 摘要

本章用几何直观和物理类比，帮你建立深度学习核心机制的直觉理解。我们会用MNIST手写数字识别作为例子，从概念到代码，一步步拆解这些技术如何让神经网络"学会"知识。

```{admonition} 前置知识
:class: note

本章是深度学习系列的**理论基础**，不需要深度学习前置知识。
但需要掌握：
- 基础Python编程
- 高中数学（函数、导数概念）
```

## 学习目标

完成本章学习后，你将能够：

- **理解计算图**：将复杂数学计算可视化为数据流图
- **掌握激活函数**：解释非线性变换如何划分决策边界
- **应用损失函数**：选择合适的指标衡量模型预测质量
- **解释反向传播**：理解误差如何在网络中反向传递并分配责任
- **运用梯度下降**：利用梯度信息优化模型参数

## 章节概览

| 章节 | 内容 | 核心收获 |
|------|------|----------|
| {doc}`introduction` | 问题引入与整体概览 | 为什么需要这些数学工具？ |
| {doc}`computational-graph` | 计算图原理 | 复杂计算的可视化表达 |
| {doc}`task-formulations` | 任务类型与优化目标 | 分类 vs 回归的数学表述 |
| {doc}`activation-functions` | 激活函数详解 | 非线性引入与决策边界 |
| {doc}`loss-functions` | 损失函数设计 | 如何量化预测误差 |
| {doc}`back-propagation` | 反向传播算法 | 高效计算梯度的链式法则 |
| {doc}`gradient-descent` | 梯度下降优化 | 参数更新的迭代策略 |
| {doc}`the-end` | 总结与展望 | 从理论到实践的过渡 |

## 本章定位

本章是整个系列的**理论基础层**。我们不涉及复杂的网络架构，而是聚焦于让神经网络"工作"起来的核心数学机制：

- **计算图** → 描述"计算如何进行"
- **激活函数** → 解决"如何表达复杂模式"
- **损失函数** → 定义"什么是好的预测"
- **反向传播** → 实现"如何高效求导"
- **梯度下降** → 回答"如何找到最优解"

**学习路径**：建立直觉 → 理解原理 → 动手实践 → 为后续章节铺垫

下一章 {doc}`../neural-network-basics/index` 将基于这些理论，用PyTorch搭建和训练实际的神经网络。

```{admonition} 目录
:class: note
~~~{toctree}
:maxdepth: 2

introduction
computational-graph
task-formulations
activation-functions
loss-functions
back-propagation
gradient-descent
the-end
~~~
```
