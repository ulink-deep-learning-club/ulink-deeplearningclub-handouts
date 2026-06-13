(neural-network-basics)=
# 神经网络基础：从理论到架构

```{only} html
还记得 {doc}`../math-fundamentals/index` 中那些抽象的数学原理吗？**这些理论怎么变成实际的网络架构？** 本章从 MNIST 出发，深入理解全连接网络和 CNN，从"知道原理"进化到"知道如何设计"。

~~~{rubric} 本章概览
:heading-level: 2
~~~

| 章节 | 内容 | 与前面章节的联系 |
| -------- | ---------- | ----------------- |
| {doc}`introduction` | MNIST 案例与全连接/CNN 对比预览 | 为什么需要 CNN？ |
| {doc}`fc-layer-basics` | 全连接层原理与 PyTorch 实现 | {ref}`computational-graph` 的架构实现 |
| {doc}`cnn-basics` | 卷积操作、池化层与参数共享机制 | {ref}`inductive-bias` 的具体体现 |
| {doc}`dataset-preparation` | 数据加载、transform、DataLoader、train/val/test 拆分 | 从文件到GPU的完整流水线 |
| {doc}`le-net` | LeNet-5 架构逐层解析 | 经典 CNN 设计模式 |
| {doc}`inception` | Inception多尺度感受野探索 | {ref}`receptive-field`的实际应用 |
| {doc}`res-net` | ResNet残差连接与深层网络 | {ref}`gradient-vanishing`的解决方案 |
| {doc}`neural-training-basics` | 完整训练流程与监控 | {ref}`back-propagation` 的实践应用 |
| {doc}`exp-cmp` | 全连接 vs CNN 实验对比 | 数据说话：参数量与准确率 |
| {doc}`scaling-law` | 模型缩放定律理论 | {ref}`gradient-descent` 与效率优化 |

~~~{rubric} 学习路径
:heading-level: 2
~~~

本章是前一章理论的**架构延伸**：

~~~{mermaid}
graph LR
    A[数学基础<br/>理解原理] --> B[神经网络<br/>设计架构]
    B --> C[实验对比<br/>验证理论]
    C --> D[缩放定律<br/>洞察规律]
~~~

**核心认知**：神经网络架构不是凭空设计，而是数学原理的**结构表达**——每一层都有对应的计算图和梯度流动。

~~~{rubric} 前置知识
:heading-level: 2
~~~

- **数学基础**：{doc}`../math-fundamentals/index` 中的计算图、反向传播、梯度下降
- **Python 基础**：熟悉 NumPy 数组操作
```

```{toctree}
:maxdepth: 2
:hidden:

introduction
fc-layer-basics
cnn-basics
dataset-preparation
le-net
inception
res-net
neural-training-basics
exp-cmp
scaling-law
the-end
```
