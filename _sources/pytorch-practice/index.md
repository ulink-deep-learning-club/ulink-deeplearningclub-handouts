(pytorch-practice)=
# PyTorch 实践：把理论变成代码

## 摘要

还记得 {doc}`../math-fundamentals/index` 中那些抽象的公式吗？
- **计算图**描述了数据如何流动
- **反向传播**把误差分摊给每个参数
- **梯度下降**在损失曲面上寻找最优解

还记得 {doc}`../neural-network-basics/index` 中那些架构分析吗？
- 全连接网络有 **20万参数**
- CNN 用权值共享减少到 **6万**
- LeNet-5 是如何端到端训练的

**但你可能还有一个疑问：这些理论怎么变成可运行的代码？**

本章就是答案。我们将用 PyTorch 把前面的理论全部实现出来——从张量操作到完整训练流程，让你真正**从知道到做到**。

~~~{admonition} 学习目标
:class: important

完成本章后，你将能够：
1. 用 PyTorch 实现 {doc}`../math-fundamentals/index` 中的计算图和反向传播
2. 搭建 {doc}`../neural-network-basics/index` 中的全连接网络和 CNN
3. 理解 PyTorch API 与理论概念的对应关系
4. 训练并调试一个完整的 MNIST 分类器
5. 掌握深度学习开发的工程实践技巧
6. 使用社团的训练框架进行高效的实验管理
~~~

## 本章概览

| 章节 | 内容 | 与前面章节的联系 |
|------|------|-----------------|
| {doc}`introduction` | PyTorch 的设计哲学与学习路径 | 为什么要用框架？ |
| {doc}`from-numpy-to-pytorch` | 从 NumPy 平滑过渡到 PyTorch | {ref}`computational-graph` 的实现 |
| {doc}`tensor-ops` | 张量操作详解 | 数据在计算图中如何流动 |
| {doc}`neural-network-module` | 用 `nn.Module` 搭建网络 | {doc}`../neural-network-basics/fc-layer-basics` 和 {doc}`../neural-network-basics/cnn-basics` 的代码实现 |
| {doc}`auto-grad` | 自动微分机制 | {ref}`back-propagation` 的 PyTorch 实现 |
| {doc}`optimiser` | 优化器与参数更新 | {ref}`gradient-descent` 的多种变体 |
| {doc}`train-workflow` | 完整训练流程 | {doc}`../neural-network-basics/neural-training-basics` 的代码化 |
| {doc}`debug-and-visualise` | 调试与可视化技巧 | 训练中的常见问题诊断 |
| {doc}`best-practices` | 工程最佳实践 | {doc}`../neural-network-basics/scaling-law` 中的效率优化 |
| {doc}`using-framework` | 使用训练框架 | {doc}`best-practices` 的工程化落地 |

## 学习路径

本章是前两章的**实践延伸**：

~~~{mermaid}
graph LR
    A[数学基础<br/>理解原理] --> B[PyTorch 实践<br/>代码实现]
    C[神经网络<br/>知道建什么] --> B
    B --> D[你能独立搭建<br/>和训练神经网络]
~~~

**核心认知**：PyTorch 不是新技术，而是前两章理论的**代码表达**——每个 API 都有对应的数学概念。

## 前置要求

~~~{admonition} 学习本章前，请确保你已经掌握
:class: caution

1. **数学基础**：{doc}`../math-fundamentals/index` 中的计算图、反向传播、梯度下降
2. **神经网络概念**：{doc}`../neural-network-basics/index` 中的全连接、CNN、训练流程
3. **Python 基础**：熟悉 NumPy 数组操作
~~~

~~~{admonition} 还没掌握？
:class: tip

如果前两章的内容已经有些模糊，建议先快速回顾：
- {ref}`computational-graph`：数据流动的直觉
- {ref}`back-propagation`：梯度如何回传
- {doc}`../neural-network-basics/le-net`：一个完整网络长什么样
~~~

## 环境配置

```{admonition} 安装指南
:class: tip

**最低配置**（CPU 即可学习）：
~~~bash
pip install torch torchvision
~~~

**推荐配置**（如果有 GPU）：
~~~bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
~~~

**验证安装**：
~~~python
import torch
print(torch.__version__)
print(torch.cuda.is_available())  # 有 GPU 会显示 True
~~~
```

## 目录

```{toctree}
:maxdepth: 2

introduction
from-numpy-to-pytorch
tensor-ops
neural-network-module
auto-grad
optimiser
train-workflow
debug-and-visualise
best-practices
using-framework
the-end
```

---

**开始之前的一句话**：PyTorch 看起来很复杂，但核心就三样东西——**张量**（存数据）、**自动微分**（算梯度）、**优化器**（更新参数）。掌握了这三样，你就掌握了深度学习开发的全部工具。
