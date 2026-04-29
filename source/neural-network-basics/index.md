(neural-network-basics)=
# 神经网络基础：从理论到架构

## 摘要

还记得 {doc}`../math-fundamentals/index` 中那些抽象的数学原理吗？
- **计算图**描述了数据如何流动和变换
- **反向传播**把误差从输出层反向传递到每一层
- **梯度下降**在损失曲面上寻找最优参数
- **激活函数**引入非线性，让网络能拟合复杂函数

**但你可能还有一个疑问：这些理论怎么变成实际的网络架构？**

本章就是答案。我们将从 MNIST 手写数字识别出发，深入理解两种核心架构——**全连接网络**和**卷积神经网络（CNN）**，让你真正从"知道原理"进化到"知道如何设计"。

```{admonition} 学习目标
:class: important

完成本章后，你将能够：
1. 理解全连接网络与 CNN 的结构差异和设计思想
2. 用 PyTorch 从零搭建神经网络，完成训练、评估全流程
3. 通过实验直观感受 CNN 的归纳偏置（{ref}`inductive-bias`）优势
4. 理解模型规模、数据量与性能的缩放定律（Scaling Law）
```

## 本章概览

| 章节 | 内容 | 与前面章节的联系 |
|------|------|-----------------|
| {doc}`introduction` | MNIST 案例与全连接/CNN 对比预览 | 为什么需要 CNN？ |
| {doc}`fc-layer-basics` | 全连接层原理与 PyTorch 实现 | {ref}`computational-graph` 的架构实现 |
| {doc}`cnn-basics` | 卷积操作与参数共享机制 | {ref}`inductive-bias` 的具体体现 |
| {doc}`le-net` | LeNet-5 架构逐层解析 | 经典 CNN 设计模式 |
| {doc}`neural-training-basics` | 完整训练流程与监控 | {ref}`back-propagation` 的实践应用 |
| {doc}`exp-cmp` | 全连接 vs CNN 实验对比 | 数据说话：参数量与准确率 |
| {doc}`scaling-law` | 模型缩放定律理论 | {ref}`gradient-descent` 与效率优化 |

## 学习路径

本章是前一章理论的**架构延伸**：

```{mermaid}
graph LR
    A[数学基础<br/>理解原理] --> B[神经网络<br/>设计架构]
    B --> C[实验对比<br/>验证理论]
    C --> D[缩放定律<br/>洞察规律]
```

**核心认知**：神经网络架构不是凭空设计，而是数学原理的**结构表达**——每一层都有对应的计算图和梯度流动。

## 本章定位

前一章我们学习了深度学习的**数学原理**：
- 计算图如何描述计算过程
- 激活函数如何引入非线性
- 反向传播如何高效计算梯度
- 梯度下降如何优化参数

本章我们进入**架构设计阶段**：
- 用 PyTorch 实现这些机制
- 理解不同架构的设计思想
- 通过实验验证理论
- 掌握训练调试技巧

**学习路径**：理论 → 实现 → 实验 → 洞察

## 前置要求

```{admonition} 学习本章前，请确保你已经掌握
:class: caution

1. **数学基础**：{doc}`../math-fundamentals/index` 中的计算图、反向传播、梯度下降
2. **Python 基础**：熟悉 NumPy 数组操作
```

```{admonition} 还没掌握？
:class: tip

如果前一章的内容已经有些模糊，建议先快速回顾：
- {ref}`computational-graph`：数据流动的直觉
- {ref}`back-propagation`：梯度如何回传
- {ref}`gradient-descent`：参数如何更新
```

## 目录

```{toctree}
:maxdepth: 2

introduction
fc-layer-basics
cnn-basics
le-net
neural-training-basics
exp-cmp
scaling-law
the-end
```
