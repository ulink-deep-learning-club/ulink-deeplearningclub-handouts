# 梯度下降优化

## 基本概念

梯度下降是一种迭代优化算法，用于寻找函数的最小值。基本更新规则为：

```{math}
\theta_{t+1} = \theta_t - \eta \cdot \nabla_\theta J(\theta)
```

其中：
- $\theta$ 是参数向量
- $\eta$ 是学习率
- $\nabla_\theta J(\theta)$ 是损失函数关于参数的梯度

## 梯度下降变体

```{list-table} 梯度下降算法比较
:header-rows: 1
:widths: 30 35 35

* - **算法**
  - **优点**
  - **缺点**
* - 批量梯度下降
  - 稳定收敛，理论保证
  - 计算开销大，内存要求高
* - 随机梯度下降
  - 计算高效，可在线学习
  - 收敛不稳定，噪声大
* - 小批量梯度下降
  - 平衡计算效率和稳定性
  - 需要调整批量大小
```

## 学习率调度

```python
import torch.optim as optim

# 创建优化器
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 学习率调度
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

for epoch in range(100):
    # 训练步骤...
    scheduler.step()
```
