(pytorch-practice)=
# PyTorch 实践：把理论变成代码

```{only} html
还记得 {doc}`../math-fundamentals/index` 的理论和 {doc}`../neural-network-basics/index` 的架构吗？**这些理论怎么变成可运行的代码？** 本章用 PyTorch 全部实现出来——从张量操作到完整训练流程，真正从知道到做到。

~~~{rubric} 本章概览
:heading-level: 2
~~~

| 章节 | 内容 | 与前面章节的联系 |
| ------ | ------ | ----------------- |
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

~~~{rubric} 学习路径
:heading-level: 2
~~~

本章是前两章的**实践延伸**：

~~~{mermaid}
graph LR
    A[数学基础<br/>理解原理] --> B[PyTorch 实践<br/>代码实现]
    C[神经网络<br/>知道建什么] --> B
    B --> D[你能独立搭建<br/>和训练神经网络]
~~~

**核心认知**：PyTorch 不是新技术，而是前两章理论的**代码表达**——每个 API 都有对应的数学概念。

~~~{rubric} 前置知识
:heading-level: 2
~~~

- **数学基础**：{doc}`../math-fundamentals/index` 中的计算图、反向传播、梯度下降
- **神经网络概念**：{doc}`../neural-network-basics/index` 中的全连接、CNN、训练流程
- **Python 基础**：熟悉 NumPy 数组操作

~~~{rubric} 环境配置
:heading-level: 2
~~~

~~~bash
pip install torch torchvision
~~~

如有 GPU：
~~~bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
~~~

验证：
~~~python
import torch
print(torch.__version__)
print(torch.cuda.is_available())
~~~
```

```{toctree}
:maxdepth: 2
:hidden:

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
