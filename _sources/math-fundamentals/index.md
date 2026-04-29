(math-fundamentals)=
# 计算图、反向传播与梯度下降：深度学习核心数学基础

## 摘要

深度学习的核心问题是什么？**让机器从数据中学习规律**。

但如何做到呢？本章将揭示答案背后的数学机制：
- **计算图**——将复杂计算可视化为数据流图，让数学变得直观
- **激活函数**——引入非线性，让网络能够拟合任意复杂函数
- **损失函数**——定义"什么是好的预测"，量化模型与目标的差距
- **反向传播**——高效计算梯度的链式法则，让误差能够反向传递
- **梯度下降**——沿着梯度方向优化参数，找到损失曲面的最低点

我们将用 MNIST 手写数字识别作为贯穿例子，从概念到代码，一步步拆解这些技术如何让神经网络"学会"知识。

```{admonition} 学习目标
:class: important

完成本章后，你将能够：
1. **理解计算图**：将复杂数学计算可视化为数据流图
2. **掌握激活函数**：解释非线性变换如何划分决策边界
3. **应用损失函数**：选择合适的指标衡量模型预测质量
4. **解释反向传播**：理解误差如何在网络中反向传递并分配责任
5. **运用梯度下降**：利用梯度信息优化模型参数
```

## 本章概览

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

## 学习路径

本章是整个系列的**理论基础层**：

```{mermaid}
graph LR
    A[建立直觉<br/>几何直观] --> B[理解原理<br/>数学机制]
    B --> C[动手实践<br/>代码实现]
    C --> D[后续章节<br/>架构与应用]
```

**核心认知**：我们不涉及复杂的网络架构，而是聚焦于让神经网络"工作"起来的核心数学机制——每个概念都有对应的代码实现。

## 本章定位

本章聚焦于让神经网络"工作"起来的五大核心机制：

- **计算图** → 描述"计算如何进行"
- **激活函数** → 解决"如何表达复杂模式"
- **损失函数** → 定义"什么是好的预测"
- **反向传播** → 实现"如何高效求导"
- **梯度下降** → 回答"如何找到最优解"

**学习路径**：建立直觉 → 理解原理 → 动手实践 → 为后续章节铺垫

## 前置要求

```{admonition} 学习本章前，请确保你已经掌握
:class: caution

本章是深度学习系列的**理论基础**，不需要深度学习前置知识，但需要：

1. **基础 Python 编程**：熟悉基本语法和函数
2. **高中数学基础**：函数、导数概念
3. **NumPy 入门**：建议了解基本数组操作
```

```{admonition} 环境准备
:class: tip

如果你还没有配置 Python 环境，可以参考 {doc}`../appendix/environment-setup/index` 中的安装指南。
```

## 与后续章节的联系

本章为整个系列奠定基础：

| 本章概念 | 后续章节应用 |
|---------|-------------|
| {ref}`computational-graph` | {doc}`../neural-network-basics/fc-layer-basics` 中的网络架构 |
| {ref}`back-propagation` | {doc}`../neural-network-basics/neural-training-basics` 中的训练流程 |
| {ref}`gradient-descent` | {doc}`../pytorch-practice/optimiser` 中的优化器 |
| {ref}`activation-functions` | {doc}`../neural-network-basics/cnn-basics` 中的卷积网络 |

下一章 {doc}`../neural-network-basics/index` 将基于这些理论，用 PyTorch 搭建和训练实际的神经网络。

## 目录

```{toctree}
:maxdepth: 2

introduction
computational-graph
task-formulations
activation-functions
loss-functions
back-propagation
gradient-descent
the-end
```
