(back-propagation)=
# 反向传播算法

## 反向传播的本质：信用分配

神经网络训练时，我们根据最终损失来调整参数。核心问题是：**损失是由很多参数共同造成的，每个参数该"背多少锅"？**

反向传播（Backpropagation）就是解决这个"**信用分配问题**"的高效算法。

### 类比：团队项目的责任分摊

想象一个团队项目失败了（损失很大），需要找出每个人的责任：

- **前向传播**：项目执行过程，每个人完成自己的任务
- **反向传播**：从失败结果倒推，计算每个人对失败的责任（梯度）
- **链式法则**：如果A的工作影响了B，B的责任要按贡献度传递给A

```{tikz} 反向传播：梯度从输出流回输入
\begin{tikzpicture}[scale=0.9]
    % 前向传播箭头（蓝色）
    \draw[->, blue, very thick] (0, 3.2) -- (3, 3.2);
    \draw[->, blue, very thick] (3, 3.2) -- (6, 3.2);
    \draw[->, blue, very thick] (6, 3.2) -- (9, 3.2);
    \node[blue] at (4.5, 3.7) {前向传播：计算预测};
    
    % 反向传播箭头（红色）
    \draw[->, red, very thick] (9, 1.8) -- (6, 1.8);
    \draw[->, red, very thick] (6, 1.8) -- (3, 1.8);
    \draw[->, red, very thick] (3, 1.8) -- (0, 1.8);
    \node[red] at (4.5, 1.3) {反向传播：回传梯度};
    
    % 节点
    \node[circle, draw, fill=blue!20, minimum size=0.8cm] at (0, 2.5) {$x$};
    \node[circle, draw, fill=blue!20, minimum size=0.8cm] at (3, 2.5) {$h$};
    \node[circle, draw, fill=blue!20, minimum size=0.8cm] at (6, 2.5) {$z$};
    \node[circle, draw, fill=red!20, minimum size=0.8cm] at (9, 2.5) {$L$};
    
    % 梯度标注
    \node[red, font=\small] at (9, 0.7) {$\frac{\partial L}{\partial L}=1$};
    \node[red, font=\small] at (6, 0.7) {$\frac{\partial L}{\partial z}$};
    \node[red, font=\small] at (3, 0.7) {$\frac{\partial L}{\partial h}$};
    \node[red, font=\small] at (0, 0.7) {$\frac{\partial L}{\partial x}$};
\end{tikzpicture}
```

## 链式法则：梯度的传递机制

{ref}`back-propagation`的核心是**链式法则**（Chain Rule），它在{ref}`computational-graph`中让梯度高效传递。对于复合函数 $z = f(g(x))$：

$$\frac{\partial z}{\partial x} = \frac{\partial z}{\partial g} \cdot \frac{\partial g}{\partial x}$$

**直觉**：总的责任 = 当前环节的责任 × 上一环节的责任

### 示例：简单计算图

考虑 $f(x, y) = (x + y) \times y$，设 $x=2, y=3$：

**前向传播**：

- $a = x + y = 2 + 3 = 5$
- $f = a \times y = 5 \times 3 = 15$

**反向传播**（假设损失对 $f$ 的梯度为1）：

- $\frac{\partial f}{\partial f} = 1$
- $\frac{\partial f}{\partial a} = y = 3$（乘法规则）
- $\frac{\partial f}{\partial y} = a = 5$（乘法规则）
- $\frac{\partial f}{\partial x} = \frac{\partial f}{\partial a} \cdot \frac{\partial a}{\partial x} = 3 \times 1 = 3$（链式法则）
- $\frac{\partial f}{\partial y}$ 有两条路径：$= 5 + 3 = 8$

## PyTorch中的自动微分

PyTorch通过`autograd`自动完成反向传播：

```python
import torch

# 创建张量，启用梯度跟踪
x = torch.tensor(2.0, requires_grad=True)
y = torch.tensor(3.0, requires_grad=True)

# 前向传播
a = x + y      # a = 5
f = a * y      # f = 15

# 反向传播
f.backward()

print(f"∂f/∂x = {x.grad}")  # 输出: 3.0
print(f"∂f/∂y = {y.grad}")  # 输出: 8.0
```

**关键机制**：

- `requires_grad=True`：告诉PyTorch需要跟踪这个张量的{ref}`computational-graph`
- PyTorch自动构建**动态计算图**（参见{ref}`computational-graph`）
- `.backward()`：自动执行{ref}`back-propagation`计算所有叶节点的梯度

## 神经网络中的反向传播

对于多层神经网络，反向传播从输出层逐层回传：

```{tikz} 神经网络反向传播示意
\begin{tikzpicture}[scale=0.8]
    % 输入层
    \foreach \i in {1,2,3}
        \node[circle, draw=blue!50, fill=blue!20, minimum size=0.6cm] (in\i) at (0,\i) {};
    \node at (-1.2, 2) {输入};
    
    % 隐藏层
    \foreach \i in {1,2,3,4}
        \node[circle, draw=orange!50, fill=orange!20, minimum size=0.6cm] (hid\i) at (3,\i-0.5) {};
    \node at (3, 4.5) {隐藏层};
    
    % 输出层
    \foreach \i in {1,2}
        \node[circle, draw=green!50, fill=green!20, minimum size=0.6cm] (out\i) at (6,\i+0.5) {};
    \node at (7.2, 2) {输出};
    
    % 前向连接（蓝色）
    \foreach \i in {1,2,3}
        \foreach \j in {1,2,3,4}
            \draw[->, blue!30] (in\i) -- (hid\j);
    
    \foreach \i in {1,2,3,4}
        \foreach \j in {1,2}
            \draw[->, blue!30] (hid\i) -- (out\j);
    
    % 反向梯度流（红色箭头）
    \draw[->, red, thick] (5, 0.8) -- (4.5, 0.4);
    \draw[->, red, thick] (5, 3.2) -- (4.5, 3.4);
    \node[red, font=\small] at (7, 3.5) {损失};
    \node[red, font=\small] at (7, 3) {$\nabla L$};
\end{tikzpicture}
```

**流程**：

1. **前向传播**：数据从输入层 → 隐藏层 → 输出层，计算预测和损失
2. **反向传播**：梯度从输出层 ← 隐藏层 ← 输入层，计算每个参数的梯度
3. **参数更新**：使用梯度下降更新权重

## 常见操作的梯度

| 操作 | 前向 | 反向梯度 |
|------|------|----------|
| 加法 $z = x + y$ | $z = x + y$ | $\frac{\partial L}{\partial x} = \frac{\partial L}{\partial z}$, $\frac{\partial L}{\partial y} = \frac{\partial L}{\partial z}$ |
| 乘法 $z = x \times y$ | $z = x \times y$ | $\frac{\partial L}{\partial x} = \frac{\partial L}{\partial z} \cdot y$, $\frac{\partial L}{\partial y} = \frac{\partial L}{\partial z} \cdot x$ |
| ReLU $z = \max(0, x)$ | $z = \max(0, x)$ | $\frac{\partial L}{\partial x} = \frac{\partial L}{\partial z}$ (if $x>0$), else 0 |
| Sigmoid $z = \sigma(x)$ | $z = \sigma(x)$ | $\frac{\partial L}{\partial x} = \frac{\partial L}{\partial z} \cdot z(1-z)$ |

## 反向传播的优势

### 1. 计算效率

{ref}`back-propagation`的时间复杂度是 $O(n)$，其中 $n$ 是{ref}`computational-graph`中边的数量。相比之下，数值差分需要 $O(n \times p)$，其中 $p$ 是参数数量。

**关键**：通过复用中间计算结果，避免重复求导。

### 2. 模块化设计

每个操作只需要定义：

- 前向：如何计算输出
- 反向：如何计算梯度

这使得添加新操作变得简单。

## 常见问题

### 梯度消失（Vanishing Gradient）

**现象**：深层网络中，靠近输入层的梯度变得非常小，参数几乎不更新。

**原因**：{ref}`activation-functions`中的Sigmoid/Tanh在饱和区梯度接近0，{ref}`back-propagation`多层连乘后梯度指数级衰减。

**解决**：使用ReLU激活函数、批归一化、残差连接。

### 梯度爆炸（Exploding Gradient）

**现象**：梯度变得非常大，导致参数更新剧烈，训练不稳定。

**原因**：权重初始化不当或网络结构导致梯度连乘后指数级增长。

**解决**：梯度裁剪、权重正则化、更好的初始化。

## 总结

反向传播是深度学习的核心算法：

1. **信用分配**：将最终损失"分摊"给每个参数
2. **链式法则**：梯度从输出层逐层回传
3. **高效计算**：复用中间结果，时间与网络规模线性相关

反向传播通过{ref}`computational-graph`高效计算梯度。理解反向传播后，我们将探讨{ref}`gradient-descent`——如何利用这些梯度来优化模型参数。
