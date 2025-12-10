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

```python
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
```python
import torch.nn as nn

mse_loss = nn.MSELoss()
output = model(input)
loss = mse_loss(output, target)
```

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
```python
mae_loss = nn.L1Loss()
loss = mae_loss(output, target)
```

### 3. 交叉熵损失（Cross-Entropy Loss）

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
```python
# 多分类交叉熵损失（包含softmax）
ce_loss = nn.CrossEntropyLoss()
loss = ce_loss(output, target)

# 二分类交叉熵损失
bce_loss = nn.BCELoss()
loss = bce_loss(output, target)
```

### 4. 负对数似然损失（Negative Log-Likelihood Loss, NLL）

负对数似然损失通常与 log-softmax 结合使用：

```{math}
\text{NLL} = -\frac{1}{n} \sum_{i=1}^n \log(\hat{y}_{i, y_i})
```

其中 $\hat{y}_{i, y_i}$ 是模型对真实类别 $y_i$ 的预测概率。

**PyTorch实现**：
```python
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
```python
huber_loss = nn.SmoothL1Loss(beta=1.0)  # beta对应δ
loss = huber_loss(output, target)
```

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
```python
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

为了防止过拟合，常在损失函数中添加正则化项：

### L1正则化（Lasso）
```{math}
L_{\text{total}} = L_{\text{data}} + \lambda \sum_{i} |w_i|
```

**效果**：产生稀疏权重，可用于特征选择。

### L2正则化（Ridge）
```{math}
L_{\text{total}} = L_{\text{data}} + \lambda \sum_{i} w_i^2
```

**效果**：限制权重大小，提高泛化能力。

### Elastic Net
```{math}
L_{\text{total}} = L_{\text{data}} + \lambda_1 \sum_{i} |w_i| + \lambda_2 \sum_{i} w_i^2
```

结合L1和L2正则化的优点。

**PyTorch实现**：
```python
# 手动添加L2正则化
l2_lambda = 0.01
l2_reg = torch.tensor(0.)
for param in model.parameters():
    l2_reg += torch.norm(param)

loss = criterion(output, target) + l2_lambda * l2_reg
```

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
```python
# 多任务学习中的损失加权
total_loss = alpha * loss1 + beta * loss2 + gamma * loss3
```

### 2. 自定义损失函数
```python
class CustomLoss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, output, target):
        # 自定义损失计算
        loss = torch.mean(torch.abs(output - target) ** 1.5)
        return loss
```

### 3. 损失函数监控
```python
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
