# 损失函数

## 基本概念

损失函数（Loss Function）是机器学习中用于衡量模型预测值与真实值之间差异的函数。它是优化算法的目标函数，梯度下降等优化算法通过最小化损失函数来调整模型参数。

```{admonition} 损失函数的作用
:class: note

- **量化误差**：将预测误差转化为可计算的数值
- **指导优化**：提供梯度方向，指导参数更新
- **评估模型**：作为模型性能的评估指标
- **正则化**：通过添加正则项控制模型复杂度
```

## 直观理解

### 损失函数的几何意义

损失函数可以看作是一个“误差曲面”，模型参数对应于曲面上的点，损失值对应于该点的高度。优化过程就是寻找曲面的最低点（全局最小值）。

```{mermaid}
graph TD
    A[输入数据] --> B[模型预测]
    B --> C["计算损失 L(y, ŷ)"]
    C --> D{损失是否可接受？}
    D -->|是| E[训练完成]
    D -->|否| F[计算梯度 ∇L]
    F --> G[更新参数]
    G --> B
```

### 不同损失函数的形状

不同的损失函数对应不同的误差曲面形状：
- **MSE**：平滑的二次曲面，有唯一最小值
- **MAE**：V形曲面，在零点不可导
- **交叉熵**：复杂的非线性曲面，但梯度计算稳定

### 示例：回归问题的损失函数比较

假设真实值 $y=0$，预测值 $\hat{y}$ 在区间 $[-3, 3]$ 内变化，我们可以比较不同损失函数的值：

```{code-block} python
:caption: 回归问题中不同损失函数的对比
:linenos:

import numpy as np
import matplotlib.pyplot as plt

y = 0
y_hat = np.linspace(-3, 3, 100)
mse = (y_hat - y)**2
mae = np.abs(y_hat - y)
huber = np.where(np.abs(y_hat - y) <= 1, 0.5*(y_hat - y)**2, np.abs(y_hat - y) - 0.5)

plt.figure(figsize=(10, 6))
plt.plot(y_hat, mse, label='MSE')
plt.plot(y_hat, mae, label='MAE')
plt.plot(y_hat, huber, label='Huber (δ=1)')
plt.xlabel('预测值 $\hat{y}$')
plt.ylabel('损失值')
plt.title('不同损失函数对比')
plt.legend()
plt.grid(True)
plt.show()
```

该图展示了不同损失函数如何惩罚预测误差。MSE对大误差惩罚更重，MAE线性惩罚，Huber在误差较小时类似MSE，较大时类似MAE。

**可视化对比**：

```{tikz}
\begin{tikzpicture}[scale=0.8]
    \draw[->] (-3,0) -- (3,0) node[right] {$\hat{y}$};
    \draw[->] (0,-0.5) -- (0,4) node[above] {损失值};
    
    \draw[domain=-3:3,smooth,variable=\x,blue,thick] plot ({\x},{\x*\x});
    \node[blue] at (2.5, 3.5) {MSE};
    
    \draw[domain=-3:3,smooth,variable=\x,red,thick] plot ({\x},{abs(\x)});
    \node[red] at (-2.5, 2.5) {MAE};
    
    \draw[domain=-3:-1,smooth,variable=\x,green,thick] plot ({\x},{abs(\x) - 0.5});
    \draw[domain=-1:1,smooth,variable=\x,green,thick] plot ({\x},{0.5*\x*\x});
    \draw[domain=1:3,smooth,variable=\x,green,thick] plot ({\x},{abs(\x) - 0.5});
    \node[green] at (1.5, 1.5) {Huber};
    
    \draw[dashed] (0,0) -- (0,0);
    \node at (-0.3,-0.4) {$y=0$};
    
    \node[gray, font=\small] at (1.5, 0.5) {MSE惩罚更重};
    \node[gray, font=\small] at (-1.5, 0.5) {MAE线性};
\end{tikzpicture}
```

该图展示了不同损失函数如何惩罚预测误差。MSE对大误差惩罚更重，MAE线性惩罚，Huber在误差较小时类似MSE，较大时类似MAE。

## 常见损失函数

### 1. 均方误差（Mean Squared Error, MSE）

均方误差是回归问题中最常用的损失函数，计算预测值与真实值之差的平方的平均值：

```{math}
\text{MSE} = \frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)^2
```

其中 $y_i$ 是真实值，$\hat{y}_i$ 是预测值，$n$ 是样本数量。

**特点**：
- 对异常值敏感（因为平方项放大了大误差）
- 可导，便于梯度计算
- 假设误差服从高斯分布

**PyTorch实现**：

```{code-block} python
:caption: MSE损失函数的PyTorch实现
:linenos:

import torch.nn as nn

mse_loss = nn.MSELoss()
output = model(input)
loss = mse_loss(output, target)
```

**可视化**：

```{tikz}
\begin{tikzpicture}[scale=0.8]
    \draw[->] (-3,0) -- (3,0) node[right] {$\hat{y}$};
    \draw[->] (0,-0.5) -- (0,5) node[above] {$\text{MSE}(y, \hat{y})$};
    
    \draw[domain=-3:3,smooth,variable=\x,blue,thick] plot ({\x},{\x*\x});
    \draw[dashed] (0,0) -- (0,0);
    \node at (-0.3,-0.4) {$y=0$};
    
    \node[blue, font=\small] at (2.5, 4) {二次曲面};
    \node[red, font=\small] at (1.5, 0.5) {误差越大，惩罚呈平方增长};
    
    \draw[<-, red, thick] (1, 0.5) -- (1.5, 1.5) node[right, red] {$(\hat{y}-y)^2$};
    \draw[<-, red, thick] (-1, 0.5) -- (-1.5, 1.5) node[left, red] {对称};
\end{tikzpicture}
```

**直观理解**：MSE损失函数是一个抛物面形状，在真实值处取得最小值0。误差越大，惩罚呈平方级增长，这意味着MSE对离群点非常敏感——一个很大的误差可能会主导整个损失值。

### 2. 平均绝对误差（Mean Absolute Error, MAE）

平均绝对误差计算预测值与真实值之差的绝对值的平均值：

```{math}
\text{MAE} = \frac{1}{n} \sum_{i=1}^n |y_i - \hat{y}_i|
```

**特点**：
- 对异常值不敏感（比MSE更鲁棒）
- 在零点不可导（但实际应用中可通过次梯度处理）
- 假设误差服从拉普拉斯分布

**PyTorch实现**：

```{code-block} python
:caption: MAE损失函数的PyTorch实现
:linenos:

mae_loss = nn.L1Loss()
loss = mae_loss(output, target)
```

**可视化**：

```{tikz}
\begin{tikzpicture}[scale=0.8]
    \draw[->] (-3,0) -- (3,0) node[right] {$\hat{y}$};
    \draw[->] (0,-0.5) -- (0,3.5) node[above] {$\text{MAE}(y, \hat{y})$};
    
    \draw[domain=-3:0,smooth,variable=\x,red,thick] plot ({\x},{abs(\x)});
    \draw[domain=0:3,smooth,variable=\x,red,thick] plot ({\x},{abs(\x)});
    \draw[dashed] (0,0) -- (0,0);
    \node at (-0.3,-0.4) {$y=0$};
    
    \node[red, font=\small] at (-2.5, 2.5) {V形};
    \node[blue, font=\small] at (1.5, 0.5) {线性惩罚};
    
    \draw[<-, blue, thick] (1, 0.5) -- (1.5, 1.0) node[right, blue] {$|\hat{y}-y|$};
    \draw[<-, blue, thick] (-1, 0.5) -- (-1.5, 1.0) node[left, blue] {对称};
    
    \node[orange, font=\small] at (0.5, 2.5) {不可导点};
    \filldraw[orange] (0,0) circle (2pt);
\end{tikzpicture}
```

**直观理解**：MAE损失函数是一个V形曲线，在真实值处取得最小值0。与MSE不同，MAE对误差的惩罚是线性的，这意味着它对异常值更加鲁棒——一个很大的误差不会主导整个损失值。

### 3. 交叉熵损失（Cross-Entropy Loss）

**前置知识：什么是One-Hot编码？**

One-Hot编码（独热编码）是将分类标签转换为向量表示的一种方法。在这种表示中：
- 向量的长度等于类别的数量 $C$
- 对应真实类别的位置为1，其他位置都为0

例如，对于一个3分类问题（类别0、1、2）：
- 真实类别为0的样本：$y = [1, 0, 0]$
- 真实类别为1的样本：$y = [0, 1, 0]$
- 真实类别为2的样本：$y = [0, 0, 1]$

这种表示方法的优点是：
- 消除了类别之间的数值大小关系（类别1并不比类别0"大"）
- 明确表示每个样本所属的类别
- 与概率分布的表示方式兼容

交叉熵损失是分类问题中最常用的损失函数，特别适用于多分类问题：

```{math}
\text{CE} = -\frac{1}{n} \sum_{i=1}^n \sum_{c=1}^C y_{i,c} \log(\hat{y}_{i,c})
```

其中 $C$ 是类别数，$y_{i,c}$ 是样本 $i$ 属于类别 $c$ 的真实概率（通常为 one-hot 编码），$\hat{y}_{i,c}$ 是模型预测的概率。

对于二分类问题，交叉熵损失简化为：

```{math}
\text{BCE} = -\frac{1}{n} \sum_{i=1}^n \left[ y_i \log(\hat{y}_i) + (1-y_i) \log(1-\hat{y}_i) \right]
```

**特点**：
- 与 softmax 激活函数配合使用效果最佳
- 梯度计算稳定，适合深度网络
- 对错误分类的惩罚较大

**PyTorch实现**：

```{code-block} python
:caption: 交叉熵损失函数的PyTorch实现
:linenos:

# 多分类交叉熵损失（包含softmax）
ce_loss = nn.CrossEntropyLoss()
loss = ce_loss(output, target)

# 二分类交叉熵损失
bce_loss = nn.BCELoss()
loss = bce_loss(output, target)
```

**可视化**：

```{tikz}
\begin{tikzpicture}[scale=0.8]
    \draw[->] (0,0) -- (5,0) node[right] {$\hat{y}$ (预测概率)};
    \draw[->] (0,-0.5) -- (0,4) node[above] {$-\log(\hat{y})$};
    
    \draw[domain=0.01:4.99,smooth,variable=\x,blue,thick] plot ({\x},{ -ln(\x/5) });
    \node[blue] at (4, 3.5) {正确分类时损失};
    
    \draw[domain=0.01:4.99,smooth,variable=\x,red,thick] plot ({\x},{ -ln(1 - \x/5) });
    \node[red] at (1, 3.5) {错误分类时损失};
    
    \draw[dashed] (5,0) -- (5,4);
    \node at (-0.3,-0.4) {$y=1$ (真实标签)};
    \node at (4.7,-0.4) {$\hat{y}=1$};
    
    \node[green, font=\small] at (4, 0.5) {预测越准确，损失越低};
    \node[green, font=\small] at (0.5, 3) {预测错误，损失急剧增加};
    
    \filldraw[blue] (5,0) circle (2pt);
    \filldraw[red] (0,0) circle (2pt);
\end{tikzpicture}
```

**直观理解**：交叉熵损失衡量的是预测概率分布与真实分布之间的差异。当预测接近真实标签时，损失接近0；当预测完全错误时，损失趋向无穷大。这种特性使得模型在训练过程中会努力提高正确类别的预测概率。

### 4. 负对数似然损失（Negative Log-Likelihood Loss, NLL）

负对数似然损失通常与 log-softmax 结合使用：

```{math}
\text{NLL} = -\frac{1}{n} \sum_{i=1}^n \log(\hat{y}_{i, y_i})
```

其中 $\hat{y}_{i, y_i}$ 是模型对真实类别 $y_i$ 的预测概率。

**PyTorch实现**：

```{code-block} python
:caption: NLL损失函数的PyTorch实现
:linenos:

nll_loss = nn.NLLLoss()
# 输入需要先经过 log-softmax
log_probs = nn.LogSoftmax(dim=1)(output)
loss = nll_loss(log_probs, target)
```

### 5. Huber损失（Smooth L1损失）

Huber损失结合了MSE和MAE的优点，在误差较小时使用平方项，误差较大时使用线性项：

```{math}
L_\delta(a) = \begin{cases}
\frac{1}{2}a^2 & \text{for } |a| \le \delta \\
\delta(|a| - \frac{1}{2}\delta) & \text{otherwise}
\end{cases}
```

其中 $a = y - \hat{y}$，$\delta$ 是超参数。

**特点**：
- 对异常值比MSE更鲁棒
- 处处可导
- 常用于回归问题，特别是目标检测

**PyTorch实现**：

```{code-block} python
:caption: Huber损失函数的PyTorch实现
:linenos:

huber_loss = nn.SmoothL1Loss(beta=1.0)  # beta对应δ
loss = huber_loss(output, target)
```

**可视化**：

```{tikz}
\begin{tikzpicture}[scale=0.8]
    \draw[->] (-3,0) -- (3,0) node[right] {$\hat{y}$};
    \draw[->] (0,-0.5) -- (0,3.5) node[above] {$\text{Huber}(y, \hat{y})$};
    
    \draw[domain=-3:-1,smooth,variable=\x,blue,thick] plot ({\x},{abs(\x) - 0.5});
    \draw[domain=-1:1,smooth,variable=\x,blue,thick] plot ({\x},{0.5*\x*\x});
    \draw[domain=1:3,smooth,variable=\x,blue,thick] plot ({\x},{abs(\x) - 0.5});
    
    \draw[dashed] (-1,0.5) -- (-1,0);
    \draw[dashed] (1,0.5) -- (1,0);
    \node at (-0.3,-0.4) {$y=0$};
    
    \node[blue, font=\small] at (0, 0.3) {MSE区域};
    \node[blue, font=\small] at (-2.5, 2) {MAE区域};
    \node[blue, font=\small] at (2.5, 2) {MAE区域};
    
    \draw[<-, red, thick] (-2, 1.5) -- (-2.5, 2.2) node[left, red] {$\delta=1$};
    \draw[<-, red, thick] (2, 1.5) -- (2.5, 2.2) node[right, red] {$\delta=1$};
    
    \filldraw[red] (-1,0.5) circle (2pt);
    \filldraw[red] (1,0.5) circle (2pt);
\end{tikzpicture}
```

**直观理解**：Huber损失函数结合了MSE和MAE的优点——在误差较小时（|a| ≤ δ），它像MSE一样使用平方项，提供平滑的梯度；在误差较大时，它像MAE一样使用线性项，避免对异常值过度惩罚。

### 6. KL散度（Kullback-Leibler Divergence）

KL散度用于衡量两个概率分布之间的差异：

```{math}
D_{KL}(P \| Q) = \sum_{i} P(i) \log \frac{P(i)}{Q(i)}
```

**应用**：
- 变分自编码器（VAE）
- 知识蒸馏
- 强化学习

**PyTorch实现**：

```{code-block} python
:caption: KL散度的PyTorch实现
:linenos:

kl_loss = nn.KLDivLoss(reduction='batchmean')
loss = kl_loss(log_input, target)
```

## 损失函数的选择原则

```{list-table} 损失函数选择指南
:header-rows: 1
:widths: 30 35 35

* - **问题类型**
  - **推荐损失函数**
  - **注意事项**
* - 回归问题
  - MSE、MAE、Huber
  - MSE对异常值敏感，MAE在零点不可导
* - 二分类问题
  - 二元交叉熵（BCE）
  - 输出需经过sigmoid激活
* - 多分类问题
  - 交叉熵（CE）
  - 输出需经过softmax激活
* - 多标签分类
  - 二元交叉熵（BCE）
  - 每个类别独立计算损失
* - 概率分布匹配
  - KL散度、JS散度
  - 确保输入为概率分布
```

## 损失函数的数学性质

### 凸性

凸损失函数保证梯度下降能找到全局最优解。常见凸损失函数包括：
- 均方误差（MSE）
- 逻辑损失（Logistic Loss）
- Huber损失（当 $\delta > 0$ 时）

### 可导性

损失函数需要在参数空间上可导（或次可导），以便使用梯度下降优化：
- MSE、交叉熵处处可导
- MAE在零点不可导，但可使用次梯度
- ReLU等激活函数引入的非光滑点可通过次梯度处理

### 利普希茨连续性

利普希茨连续的损失函数具有有界梯度，有利于优化稳定性：
- 交叉熵损失是利普希茨连续的
- MSE在有限域上是利普希茨连续的

## 正则化损失

**为什么需要正则化？**

在深度学习中，模型通常拥有大量参数，有时甚至超过了训练样本的数量。这种过高的模型容量使得模型能够"记住"训练数据的噪声和细节，而不是学习到真正的通用规律，这种现象称为**过拟合（Overfitting）**。

正则化是防止过拟合的核心技术之一，它通过在损失函数中添加额外约束来限制模型的复杂度。

```{admonition} 过拟合的典型表现
:class: warning

- 训练集损失很低，但验证集损失很高
- 模型在训练样本上表现完美，但泛化能力差
- 权重值过大，导致对输入的微小变化过于敏感
```

正则化的核心思想是：**在拟合训练数据的能力和模型复杂度之间找到平衡**。

### 正则化的直观理解

想象我们要拟合一组数据点：

- **无正则化**：模型可能为了完美拟合每一个点而形成复杂的曲线
- **适度正则化**：模型学习到数据的一般趋势，忽略噪声
- **过度正则化**：模型过于简单，无法捕捉数据的基本模式

**正则化强度的可视化**：

```{tikz}
\begin{tikzpicture}[scale=0.8]
    % 坐标轴
    \draw[->] (-0.5,0) -- (5,0) node[right] {模型复杂度};
    \draw[->] (0,-0.5) -- (0,4) node[above] {损失值};
    
    % 数据拟合损失曲线（递减）
    \draw[domain=0.2:4.5,smooth,variable=\x,blue,thick] plot ({\x},{4 / (\x + 0.5)});
    \node[blue] at (4.5, 0.8) {数据拟合损失};
    
    % 正则化损失曲线（递增）
    \draw[domain=0.2:4.5,smooth,variable=\x,red,thick] plot ({\x},{\x * 0.4});
    \node[red] at (4.5, 1.8) {正则化损失};
    
    % 总损失曲线（U形）
    \draw[domain=0.2:4.5,smooth,variable=\x,green,thick] plot ({\x},{4 / (\x + 0.5) + \x * 0.4 - 0.5});
    \node[green] at (2.5, 3.5) {总损失};
    
    % 标注最优位置
    \draw[dashed] (1.8, 2.2) -- (1.8, 0);
    \node at (1.8, -0.4) {最优复杂度};
    
    \filldraw[green] (1.8, 2.2) circle (3pt);
    
    % 区域标注
    \node[gray, font=\small] at (0.8, 3.2) {欠拟合};
    \node[gray, font=\small] at (3.8, 3.2) {过拟合};
    \node[blue, font=\small] at (0.5, 1) {简单模型};
    \node[blue, font=\small] at (4.2, 1) {复杂模型};
\end{tikzpicture}
```

这个图展示了正则化如何平衡模型复杂度与数据拟合：
- **蓝色曲线**：数据拟合损失随复杂度增加而降低
- **红色曲线**：正则化损失随复杂度增加而增加
- **绿色曲线**：总损失在两者之间取得最优平衡点

### L1正则化（Lasso）

**数学定义**：

L1正则化在损失函数中添加权重的绝对值之和：

$$L_{\text{total}} = L_{\text{data}} + \lambda \sum_{i=1}^{n} |w_i|$$

其中 $\lambda$ 是正则化强度超参数，控制正则化的力度。

**L1正则化的独特性质：稀疏性**

L1正则化倾向于产生稀疏权重向量，即大多数权重为0，只有少数权重非零。这是因为L1正则化的等值面是菱形（多边形），在与误差曲面的交点处更容易落在坐标轴上。

**几何解释**：

```{tikz}
\begin{tikzpicture}[scale=0.8]
    % 绘制等高线
    \draw[->] (-2.5,0) -- (2.5,0) node[right] {$w_1$};
    \draw[->] (0,-2.5) -- (0,2.5) node[above] {$w_2$};
    
    % 椭圆（数据损失等高线）
    \draw[blue, thick] (0,0) ellipse (2 and 1.2);
    \draw[blue, thick] (0,0) ellipse (1.2 and 0.7);
    \node[blue] at (2.2, 1.5) {数据损失};
    
    % 菱形（L1正则化等高线）
    \draw[red, thick] (0,1.5) -- (1.5,0) -- (0,-1.5) -- (-1.5,0) -- cycle;
    \draw[red, thick] (0,1) -- (1,0) -- (0,-1) -- (-1,0) -- cycle;
    \node[red] at (-2.2, 1.5) {L1约束};
    
    % 最优点（稀疏解）
    \filldraw[green] (1,0) circle (3pt);
    \node[green] at (1.3, 0.3) {最优解};
    
    % 标注
    \node[gray, font=\small] at (0.5, -0.5) {稀疏解};
    \node[gray, font=\small] at (-0.5, 0.5) {$w_2 = 0$};
\end{tikzpicture}
```

图中展示了L1正则化如何产生稀疏解：最优解通常位于坐标轴上（某些权重为0）。

**L1正则化的效果**：
- **特征选择**：自动识别重要特征，非重要特征的权重趋近于0
- **可解释性**：最终模型只包含少数关键特征
- **降维**：有效减少模型参数数量

**适用场景**：
- 特征数量远大于样本数量
- 需要特征选择或降维
- 追求模型可解释性

### L2正则化（Ridge）

**数学定义**：

L2正则化在损失函数中添加权重的平方和：

$$L_{\text{total}} = L_{\text{data}} + \lambda \sum_{i=1}^{n} w_i^2$$

**L2正则化的特点：权重衰减**

L2正则化不会产生稀疏权重，但会使得权重普遍较小。它通过惩罚大权重来防止模型过度依赖任何一个特征。

**几何解释**：

```{tikz}
\begin{tikzpicture}[scale=0.8]
    % 绘制等高线
    \draw[->] (-2.5,0) -- (2.5,0) node[right] {$w_1$};
    \draw[->] (0,-2.5) -- (0,2.5) node[above] {$w_2$};
    
    % 椭圆（数据损失等高线）
    \draw[blue, thick] (0,0) ellipse (2 and 1.2);
    \draw[blue, thick] (0,0) ellipse (1.2 and 0.7);
    \node[blue] at (2.2, 1.5) {数据损失};
    
    % 圆形（L2正则化等高线）
    \draw[red, thick] (0,0) circle (1.5);
    \draw[red, thick] (0,0) circle (1);
    \node[red] at (-2.2, 1.5) {L2约束};
    
    % 最优点（收缩解）
    \filldraw[green] (0.8, 0.5) circle (3pt);
    \node[green] at (1.1, 0.8) {最优解};
    
    % 与无正则化的对比
    \draw[dashed, gray] (1.5, 0.9) circle (3pt);
    \node[gray] at (1.8, 1.2) {无正则化};
    
    % 标注
    \node[gray, font=\small] at (0.3, 1.8) {权重收缩};
    \draw[<->, gray, thick] (0,0) -- (0.8, 0.5);
\end{tikzpicture}
```

图中展示了L2正则化如何收缩权重：最优解位于原点与无正则化解的连线上，但更靠近原点。

**L2正则化的效果**：
- **权重衰减**：所有权重都变小，但不为零
- **提高泛化**：防止模型过度依赖少数特征
- **数值稳定**：避免权重过大导致的数值问题

**适用场景**：
- 特征数量与样本数量相当
- 需要提高模型泛化能力
- 特征之间存在多重共线性

### L1 vs L2 正则化对比

| 特性 | L1正则化 | L2正则化 |
|------|---------|---------|
| 几何形状 | 菱形（多边形） | 圆形（椭球） |
| 稀疏性 | 产生稀疏解 | 权重普遍较小但不稀疏 |
| 特征选择 | 自动进行特征选择 | 不进行特征选择 |
| 计算复杂度 | 需要特殊的优化算法 | 梯度下降即可优化 |
| 对异常值 | 较敏感 | 较鲁棒 |
| 典型应用 | 特征选择、压缩感知 | 防止过拟合、权重衰减 |

### Elastic Net

Elastic Net结合了L1和L2正则化的优点：

$$L_{\text{total}} = L_{\text{data}} + \lambda_1 \sum_{i} |w_i| + \lambda_2 \sum_{i} w_i^2$$

**为什么需要Elastic Net？**

Elastic Net解决了L1正则化的两个主要问题：
1. 当特征数量大于样本数量时，L1可能只选择一个特征
2. L1对特征的缩放敏感

Elastic Net通过同时使用L1和L2正则化：
- 保持L1的稀疏性（特征选择能力）
- 通过L2稳定L1的优化过程
- 在高度相关的特征组中同时选择多个特征

### PyTorch中的正则化实现

**方法1：手动添加正则化项**

```{code-block} python
:caption: 手动实现L1和L2正则化
:linenos:

import torch

def l1_regularization(model, l1_lambda):
    """计算L1正则化损失"""
    l1_reg = torch.tensor(0., requires_grad=True)
    for param in model.parameters():
        l1_reg = l1_reg + torch.norm(param, p=1)
    return l1_lambda * l1_reg

def l2_regularization(model, l2_lambda):
    """计算L2正则化损失"""
    l2_reg = torch.tensor(0., requires_grad=True)
    for param in model.parameters():
        l2_reg = l2_reg + torch.norm(param, p=2)
    return l2_lambda * l2_reg

# 使用示例
model = MyModel()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for batch in dataloader:
    inputs, targets = batch
    outputs = model(inputs)
    data_loss = criterion(outputs, targets)
    
    # 添加L1正则化
    l1_loss = l1_regularization(model, l1_lambda=0.001)
    
    # 添加L2正则化
    l2_loss = l2_regularization(model, l2_lambda=0.001)
    
    # 总损失
    total_loss = data_loss + l1_loss + l2_loss
    
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
```

**方法2：使用PyTorch内置参数**

PyTorch的优化器内置了weight_decay参数（对应L2正则化）：

```{code-block} python
:caption: 使用优化器的weight_decay参数
:linenos:

# L2正则化（weight_decay）
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001,
    weight_decay=0.01  # L2正则化强度
)

# 注意：Adam优化器中的weight_decay与标准L2正则化略有不同
# 它使用L2正则化的变体（权重衰减），在某些情况下效果更好
```

**方法3：Elastic Net组合**

```{code-block} python
:caption: Elastic Net正则化实现
:linenos:

class ElasticNetLoss(nn.Module):
    def __init__(self, l1_lambda=0.001, l2_lambda=0.001):
        super().__init__()
        self.l1_lambda = l1_lambda
        self.l2_lambda = l2_lambda
    
    def forward(self, data_loss, model):
        l1_reg = torch.tensor(0., requires_grad=True)
        l2_reg = torch.tensor(0., requires_grad=True)
        
        for param in model.parameters():
            l1_reg = l1_reg + torch.norm(param, p=1)
            l2_reg = l2_reg + torch.norm(param, p=2)
        
        elastic_loss = self.l1_lambda * l1_reg + self.l2_lambda * l2_reg
        return data_loss + elastic_loss

# 使用示例
criterion = ElasticNetLoss(l1_lambda=0.001, l2_lambda=0.001)

for batch in dataloader:
    inputs, targets = batch
    outputs = model(inputs)
    data_loss = criterion(outputs, targets)
    total_loss = criterion(data_loss, model)
    
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
```

### 正则化参数的选择

**超参数 $\lambda$ 的选择**：

$\lambda$ 是正则化强度的超参数，需要仔细调优：
- $\lambda$ 太小：正则化效果弱，可能过拟合
- $\lambda$ 太大：正则化效果过强，可能欠拟合

**常用的选择方法**：

1. **网格搜索（Grid Search）**
   - 在预定义的参数范围内搜索最优 $\lambda$
   - 适用于参数空间较小的情况

2. **随机搜索（Random Search）**
   - 在参数空间中随机采样
   - 通常比网格搜索更高效

3. **验证集选择**
   - 在验证集上评估不同 $\lambda$ 的效果
   - 选择使验证损失最小的 $\lambda$

4. **贝叶斯优化**
   - 使用贝叶斯方法高效搜索超参数
   - 适用于昂贵的评估场景

**$\lambda$ 的常见取值范围**：

| 数据规模 | 典型 $\lambda$ 范围 |
|---------|------------------|
| 小数据集 | 0.01 ~ 0.1 |
| 中等数据集 | 0.001 ~ 0.01 |
| 大数据集 | 0.0001 ~ 0.001 |

### Dropout：另一种正则化方法

除了L1/L2正则化，Dropout也是一种非常有效的正则化技术：

```{code-block} python
:caption: Dropout正则化实现
:linenos:

class DropoutModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_rate=0.5):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout_rate)  # Dropout层
        self.layer2 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = self.dropout(x)  # 训练时随机丢弃神经元
        x = self.layer2(x)
        return x

# 使用Dropout的注意事项
model = DropoutModel(784, 256, 10, dropout_rate=0.5)

# 训练模式：启用Dropout
model.train()

# 评估模式：禁用Dropout
model.eval()
# 或者使用 torch.inference_mode()
```

**Dropout的工作原理**：
- 训练时：随机丢弃一定比例的神经元
- 预测时：使用所有神经元，但权重按比例缩放

**Dropout的效果**：
- 防止神经元之间过度共适应
- 类似于训练多个不同的模型并集成
- 提供隐式的模型集成效果

### 正则化策略的选择

根据不同场景选择合适的正则化策略：

| 场景 | 推荐策略 | 说明 |
|------|---------|------|
| 特征数量 >> 样本数量 | L1/Elastic Net | 特征选择，降低过拟合风险 |
| 高度相关特征 | Elastic Net | 同时选择相关特征组 |
| 需要特征可解释性 | L1 | 产生稀疏模型 |
| 防止权重过大 | L2 | 权重衰减 |
| 大型深度学习模型 | Dropout | 深度网络的标准正则化 |
| 多重正则化 | L1 + L2 + Dropout | 综合使用多种技术 |

## 损失函数的梯度计算

### MSE梯度
```{math}
\frac{\partial \text{MSE}}{\partial \hat{y}_i} = \frac{2}{n} (\hat{y}_i - y_i)
```

### 交叉熵梯度
对于softmax输出后的交叉熵损失，梯度具有简洁形式：
```{math}
\frac{\partial \text{CE}}{\partial z_i} = \hat{y}_i - y_i
```
其中 $z_i$ 是softmax层的输入。

### MAE次梯度
```{math}
\frac{\partial \text{MAE}}{\partial \hat{y}_i} = \begin{cases}
1 & \text{if } \hat{y}_i > y_i \\
-1 & \text{if } \hat{y}_i < y_i \\
[-1, 1] & \text{if } \hat{y}_i = y_i
\end{cases}
```

## 实践建议

### 1. 损失函数缩放

```{code-block} python
:caption: 多任务学习中的损失加权
:linenos:

# 多任务学习中的损失加权
total_loss = alpha * loss1 + beta * loss2 + gamma * loss3
```

### 2. 自定义损失函数

```{code-block} python
:caption: 自定义损失函数的实现
:linenos:

class CustomLoss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, output, target):
        # 自定义损失计算
        loss = torch.mean(torch.abs(output - target) ** 1.5)
        return loss
```

### 3. 损失函数监控

```{code-block} python
:caption: 损失函数监控的实现
:linenos:

def monitor_losses(loss_dict, epoch):
    """监控多个损失分量"""
    print(f"Epoch {epoch}:")
    for name, value in loss_dict.items():
        print(f"  {name}: {value:.4f}")
    
    # 可视化
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    for name, values in loss_history.items():
        plt.plot(values, label=name)
    plt.legend()
    plt.show()
```

## 总结

损失函数是深度学习的核心组件，它连接了模型预测和参数优化。选择合适的损失函数需要考虑问题类型、数据特性和模型结构。理解不同损失函数的数学性质和梯度行为，有助于设计更有效的训练策略和解决实际问题。

```{admonition} 关键要点
:class: tip

1. **回归问题**：优先考虑MSE，对异常值敏感时使用MAE或Huber损失
2. **分类问题**：交叉熵损失是标准选择，配合适当的激活函数
3. **正则化**：通过L1/L2正则化防止过拟合，提高泛化能力
4. **多任务学习**：合理加权不同任务的损失函数
5. **自定义损失**：根据特定问题设计专用损失函数
```
