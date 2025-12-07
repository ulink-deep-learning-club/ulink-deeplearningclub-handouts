# 梯度下降优化

## 基本概念

梯度下降是一种迭代优化算法，用于寻找函数的最小值。基本更新规则为：

```{math}
\theta_{t+1} = \theta_t - \eta \cdot \nabla_\theta J(\theta)
```

其中：
- $\theta$ 是参数向量
- $\eta$ 是学习率（learning rate）
- $\nabla_\theta J(\theta)$ 是损失函数关于参数的梯度
- $t$ 是迭代次数

## 梯度下降的数学原理

### 泰勒展开视角

梯度下降可以从泰勒展开的角度理解。在点 $\theta_t$ 处，损失函数的一阶泰勒展开为：

```{math}
J(\theta_{t+1}) \approx J(\theta_t) + \nabla J(\theta_t)^\top (\theta_{t+1} - \theta_t)
```

为了使 $J(\theta_{t+1}) < J(\theta_t)$，我们选择：

```{math}
\theta_{t+1} = \theta_t - \eta \nabla J(\theta_t)
```

这样保证在梯度方向上前进一小步，损失函数会减小。

### 收敛性分析

对于凸函数，梯度下降的收敛速度是 $O(1/t)$。对于强凸函数，收敛速度可以达到 $O(e^{-t})$。

## 梯度下降变体

### 1. 批量梯度下降（Batch Gradient Descent）

使用整个训练集计算梯度：

```{math}
\theta_{t+1} = \theta_t - \eta \cdot \frac{1}{N} \sum_{i=1}^N \nabla_\theta J(\theta; x_i, y_i)
```

**特点**：
- 每次迭代计算整个数据集的梯度
- 收敛稳定，但计算开销大
- 可能陷入局部最小值

### 2. 随机梯度下降（Stochastic Gradient Descent, SGD）

每次使用一个样本计算梯度：

```{math}
\theta_{t+1} = \theta_t - \eta \cdot \nabla_\theta J(\theta; x_i, y_i)
```

**特点**：
- 计算高效，适合在线学习
- 收敛不稳定，有噪声
- 可能跳出局部最小值

### 3. 小批量梯度下降（Mini-batch Gradient Descent）

使用一小批样本计算梯度：

```{math}
\theta_{t+1} = \theta_t - \eta \cdot \frac{1}{m} \sum_{i=1}^m \nabla_\theta J(\theta; x_i, y_i)
```

其中 $m$ 是批量大小。

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

## 高级优化算法

### 1. 动量法（Momentum）

引入动量项加速收敛：

```{math}
\begin{aligned}
v_{t+1} &= \beta v_t + \nabla J(\theta_t) \\
\theta_{t+1} &= \theta_t - \eta v_{t+1}
\end{aligned}
```

其中 $\beta$ 是动量系数（通常取0.9）。

### 2. AdaGrad

自适应调整学习率：

```{math}
\begin{aligned}
G_t &= G_{t-1} + \nabla J(\theta_t)^2 \\
\theta_{t+1} &= \theta_t - \frac{\eta}{\sqrt{G_t + \epsilon}} \nabla J(\theta_t)
\end{aligned}
```

### 3. RMSProp

改进的AdaGrad，使用指数移动平均：

```{math}
\begin{aligned}
E[g^2]_t &= \beta E[g^2]_{t-1} + (1-\beta) \nabla J(\theta_t)^2 \\
\theta_{t+1} &= \theta_t - \frac{\eta}{\sqrt{E[g^2]_t + \epsilon}} \nabla J(\theta_t)
\end{aligned}
```

### 4. Adam（Adaptive Moment Estimation）

结合动量和自适应学习率：

```{math}
\begin{aligned}
m_t &= \beta_1 m_{t-1} + (1-\beta_1) \nabla J(\theta_t) \\
v_t &= \beta_2 v_{t-1} + (1-\beta_2) \nabla J(\theta_t)^2 \\
\hat{m}_t &= \frac{m_t}{1-\beta_1^t} \\
\hat{v}_t &= \frac{v_t}{1-\beta_2^t} \\
\theta_{t+1} &= \theta_t - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t
\end{aligned}
```

## 学习率调度

### 1. 固定学习率

最简单的策略，但需要手动调整。

### 2. 步长衰减（Step Decay）

```python
import torch.optim as optim

# 创建优化器
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 每30个epoch将学习率乘以0.1
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

for epoch in range(100):
    # 训练步骤...
    scheduler.step()
```

### 3. 指数衰减

```python
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
```

### 4. 余弦退火

```python
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
```

### 5. 循环学习率

```python
scheduler = optim.lr_scheduler.CyclicLR(
    optimizer, 
    base_lr=0.001, 
    max_lr=0.01,
    step_size_up=2000
)
```

## 梯度下降的实现

### 基本实现

```python
import numpy as np

def gradient_descent(f, grad_f, theta0, lr=0.01, max_iter=1000, tol=1e-6):
    """
    梯度下降算法实现
    
    参数：
    f: 目标函数
    grad_f: 梯度函数
    theta0: 初始参数
    lr: 学习率
    max_iter: 最大迭代次数
    tol: 收敛容忍度
    
    返回：
    theta: 最优参数
    history: 历史损失值
    """
    theta = theta0.copy()
    history = [f(theta)]
    
    for i in range(max_iter):
        # 计算梯度
        grad = grad_f(theta)
        
        # 更新参数
        theta = theta - lr * grad
        
        # 记录损失
        loss = f(theta)
        history.append(loss)
        
        # 检查收敛
        if i > 0 and abs(history[-1] - history[-2]) < tol:
            print(f"在第 {i} 次迭代收敛")
            break
    
    return theta, history
```

### PyTorch实现

```python
import torch
import torch.optim as optim
import torch.nn as nn

# 定义模型
model = nn.Linear(10, 1)

# 定义损失函数
criterion = nn.MSELoss()

# 定义优化器
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# 训练循环
def train_epoch(model, dataloader, criterion, optimizer):
    model.train()
    total_loss = 0
    
    for batch_idx, (data, target) in enumerate(dataloader):
        # 前向传播
        output = model(data)
        loss = criterion(output, target)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        
        # 梯度裁剪（防止梯度爆炸）
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # 参数更新
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)
```

## 超参数调优

### 学习率选择

```python
def find_learning_rate(model, train_loader, criterion, lr_range=(1e-5, 1)):
    """寻找合适的学习率"""
    losses = []
    learning_rates = []
    
    for lr in np.logspace(np.log10(lr_range[0]), np.log10(lr_range[1]), num=20):
        optimizer = optim.SGD(model.parameters(), lr=lr)
        
        # 训练一个epoch
        loss = train_epoch(model, train_loader, criterion, optimizer)
        
        losses.append(loss)
        learning_rates.append(lr)
        
        print(f"LR: {lr:.2e}, Loss: {loss:.4f}")
    
    return learning_rates, losses
```

### 批量大小选择

批量大小影响：
1. **内存使用**：批量越大，内存需求越高
2. **收敛速度**：小批量通常收敛更快
3. **泛化能力**：小批量通常泛化更好

```python
def find_batch_size(dataset, model, available_memory=8e9):
    """根据可用内存确定批量大小"""
    # 估计一个样本的内存使用
    sample_size = dataset[0][0].element_size() * dataset[0][0].nelement()
    
    # 考虑模型参数和梯度
    model_params = sum(p.numel() for p in model.parameters())
    model_memory = model_params * 4 * 3  # 参数、梯度、优化器状态
    
    # 计算最大批量大小
    max_batch_size = int((available_memory - model_memory) / sample_size)
    
    # 选择2的幂次
    batch_size = 1
    while batch_size * 2 <= max_batch_size:
        batch_size *= 2
    
    return min(batch_size, 256)  # 不超过256
```

## 梯度下降的挑战与解决方案

### 1. 局部最小值

**问题**：梯度下降可能陷入局部最小值。

**解决方案**：
- 使用随机梯度下降
- 添加动量
- 多次随机初始化
- 使用模拟退火

### 2. 鞍点问题

**问题**：在高维空间中，鞍点比局部最小值更常见。

**解决方案**：
- 使用二阶优化方法（牛顿法）
- 使用自适应学习率算法（Adam）
- 添加噪声

### 3. 学习率选择

**问题**：学习率过大导致震荡，过小导致收敛慢。

**解决方案**：
- 学习率调度
- 自适应学习率算法
- 学习率预热

### 4. 梯度噪声

**问题**：随机梯度下降的梯度估计有噪声。

**解决方案**：
- 增加批量大小
- 使用梯度平均
- 使用动量

## 实践建议

### 1. 初始化策略

```python
# 好的初始化方法
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

model.apply(init_weights)
```

### 2. 监控训练过程

```python
def monitor_training(model, train_loader, val_loader, epochs=100):
    """监控训练过程"""
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    for epoch in range(epochs):
        # 训练
        train_loss, train_acc = train_epoch(model, train_loader)
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # 验证
        val_loss, val_acc = evaluate(model, val_loader)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        # 打印进度
        print(f"Epoch {epoch+1}/{epochs}: "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
    
    return train_losses, val_losses, train_accs, val_accs
```

### 3. 早停策略

```python
def early_stopping(val_losses, patience=10):
    """早停策略"""
    if len(val_losses) < patience + 1:
        return False
    
    # 检查最近patience个epoch是否没有改善
    best_loss = min(val_losses)
    recent_losses = val_losses[-patience:]
    
    if min(recent_losses) > best_loss:
        return True
    
    return False
```

## 总结

梯度下降是深度学习的核心优化算法。理解不同变体的特点、学习率调度策略以及常见问题的解决方案，对于有效训练深度学习模型至关重要。在实践中，通常从小批量梯度下降开始，然后根据具体情况选择更高级的优化算法如Adam，并配合适当的学习率调度策略。
