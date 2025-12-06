# 反向传播算法

## 链式法则回顾

反向传播算法的核心是链式法则。对于复合函数 $z = f(g(x))$，其导数为：

```{math}
\frac{dz}{dx} = \frac{dz}{dg} \cdot \frac{dg}{dx}
```

在多元函数的情况下，对于 $z = f(x_1, x_2, \dots, x_n)$，偏导数为：

```{math}
\frac{\partial z}{\partial x_i} = \sum_{j} \frac{\partial z}{\partial y_j} \cdot \frac{\partial y_j}{\partial x_i}
```

## 反向传播的基本思想

反向传播算法通过以下步骤计算梯度：

1. **前向传播**：计算图中每个节点的值
2. **反向传播**：从输出节点开始，反向计算每个节点的梯度
3. **梯度累积**：利用链式法则累积梯度

## 算法实现

```python
import torch

# 简单的计算图示例
x = torch.tensor(2.0, requires_grad=True)
y = torch.tensor(3.0, requires_grad=True)

# 前向传播
z = x * y + torch.sin(x)

# 反向传播
z.backward()

print(f"x.grad = {x.grad}")  # ∂z/∂x
print(f"y.grad = {y.grad}")  # ∂z/∂y
```
