(pytorch-introduction)=
# 引言：从理论到代码

还记得 {doc}`../neural-network-basics/le-net` 中那 61,706 个参数吗？我们分析了每一层的维度、计算了参数量、讨论了为什么 CNN 比全连接更高效。

**但有一个问题：这些数字是怎么存到电脑里的？那些矩阵乘法是怎么算的？反向传播到底怎么实现？**

本章将回答这些问题。我们会用 PyTorch 把 {doc}`../neural-network-basics/index` 中的理论全部实现出来，让你真正理解**从数学公式到可运行代码**的转化过程。

## 为什么要用框架？

### 想象你要手写一个 LeNet

如果没有框架，你需要自己实现：

1. **张量存储**：如何存 32 个 5×5 卷积核？用 Python 列表？效率太低
2. **卷积运算**：嵌套循环实现滑动窗口？代码复杂还容易错
3. **反向传播**：手动推导每一层的梯度？LeNet 有 8 层，每层梯度公式都不一样
4. **GPU 加速**：想让计算快 10 倍？你得学 CUDA 编程

~~~{admonition} 手动实现的痛苦
:class: caution

{doc}`../neural-network-basics/fc-layer-basics` 中那个 3 层全连接网络，手写反向传播就需要：
- 推导每一层的梯度公式
- 小心矩阵维度匹配
- 调试数值稳定性问题
- 花了半天，最后发现是某个下标写错了

这还只是 3 层！想象一下 ResNet-152 的 152 层...
~~~

### 框架做了什么？

PyTorch 把上面这些痛苦都解决了：

| 问题 | 手写实现 | PyTorch 方案 |
|------|---------|-------------|
| 数据存储 | Python 列表/NumPy 数组 | `torch.Tensor`：统一的数据结构 |
| 卷积运算 | 嵌套循环 | `nn.Conv2d`：一行代码 |
| 反向传播 | 手动推导梯度 | `.backward()`：自动计算 |
| GPU 加速 | 写 CUDA 代码 | `.to('cuda')`：一键转移 |

**核心洞察**：PyTorch 不是新技术，而是 {doc}`../math-fundamentals/index` 中理论的**工程封装**——每个 API 都对应一个数学概念。

## PyTorch 与前面章节的对应

让我们建立一个"理论→代码"的映射表：

| {doc}`../math-fundamentals/index` 理论 | PyTorch 实现 | {doc}`../neural-network-basics/index` 应用 |
|--------------------------------------|-------------|-------------------------------------------|
| {ref}`computational-graph` | `torch.Tensor` + 运算 | 数据如何在网络中流动 |
| {ref}`back-propagation` | `.backward()` | 梯度如何回传更新参数 |
| {ref}`gradient-descent` | `optim.SGD/Adam` | 参数如何一步步优化 |
| {ref}`activation-functions` | `nn.ReLU/Sigmoid` | 引入非线性 |
| {ref}`loss-functions` | `nn.CrossEntropyLoss` | 衡量预测好坏 |

**学习策略**：每学一个 PyTorch API，问自己"这对应哪个理论概念？"

## PyTorch 的设计哲学

### 动态计算图：调试友好

PyTorch 采用**动态计算图**（define-by-run）：每次前向传播时实时构建计算图。

~~~{admonition} 动态 vs 静态
:class: note

**静态计算图**（TensorFlow 1.x）：
- 先定义完整的计算图
- 然后在一个独立的 session 中运行
- **调试痛苦**：出错时不知道是哪一行

**动态计算图**（PyTorch）：
- 代码按顺序执行
- 计算图在运行时构建
- **调试友好**：可以用 `print()` 随时查看中间结果
~~~

这对学习很重要——你可以随时停下来检查张量的形状和内容，就像调试普通 Python 代码一样。

### Pythonic：符合直觉

PyTorch 的 API 设计遵循 Python 的习惯：

~~~python
# NumPy 风格的操作
import torch
x = torch.tensor([1.0, 2.0, 3.0])
y = x * 2 + 1  # 就像普通的 Python 运算

# 查看形状、设备、是否需要梯度
print(x.shape, x.device, x.requires_grad)
~~~

**没有魔法**：你看到的就是实际发生的。没有隐藏的图构建过程，没有复杂的 session 管理。

## 本章学习路线图

我们将按照"数据→模型→训练"的自然顺序学习：

~~~{mermaid}
flowchart LR
    A[张量<br/>存数据] --> B[nn.Module<br/>建模型]
    B --> C[自动微分<br/>算梯度]
    C --> D[优化器<br/>更新参数]
    D --> E[训练循环<br/>完整流程]
~~~

与 {doc}`../neural-network-basics/index` 的对应：

1. **{doc}`from-numpy-to-pytorch`**：理解 `torch.Tensor` 如何对应 {ref}`computational-graph` 中的节点
2. **{doc}`tensor-ops`**：数据如何在网络中流动（reshape、transpose 对应维度变换）
3. **{doc}`neural-network-module`**：用 `nn.Module` 实现 {doc}`../neural-network-basics/fc-layer-basics` 和 {doc}`../neural-network-basics/cnn-basics` 中的架构
4. **{doc}`auto-grad`**：`.backward()` 就是 {ref}`back-propagation` 的自动化
5. **{doc}`optimiser`**：`optimizer.step()` 实现 {ref}`gradient-descent` 的各种变体
6. **{doc}`train-workflow`**：把 {doc}`../neural-network-basics/neural-training-basics` 中的流程代码化

## 核心认知：API 即理论

记住这个公式：

$$
\text{PyTorch API} = \text{数学概念} + \text{工程优化}
$$

**例子**：`nn.Conv2d` 的背后
- **数学**：卷积运算 $Y[i,j] = \sum_{u,v} X[i+u, j+v] \cdot K[u,v]$
- **工程**：CuDNN 优化的 GPU 实现，比手写快 100 倍

你的任务是理解**数学概念**，工程优化 PyTorch 已经帮你做了。

## 开始之前

~~~{admonition} 本章的学习心法
:class: tip

1. **带着理论学代码**：每看到一个 API，问"这对应哪个数学概念？"
2. **动手实验**：修改参数看结果变化，比看书更有效
3. **善用帮助**：`help(torch.nn.Conv2d)` 会告诉你数学公式和参数说明
4. **连接前后**：随时回顾 {doc}`../math-fundamentals/index` 和 {doc}`../neural-network-basics/index` 的对应内容
~~~

**一句话总结**：本章不是学新东西，而是用 PyTorch **重新表达**你已经懂的理论。

准备好了吗？让我们从张量开始——这是 PyTorch 世界的"原子"。
