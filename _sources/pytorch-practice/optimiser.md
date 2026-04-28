(pytorch-optimiser)=
# 优化器：用梯度更新参数

{doc}`./auto-grad`中我们学会了自动计算梯度。现在的问题是：**有了梯度，如何更新参数？**

在 {doc}`../math-fundamentals/gradient-descent`中，我们详细学习了梯度下降的原理和各种改进算法（SGD、Momentum、Adam）。**本节聚焦 PyTorch 实现**——如何把理论变成可运行的代码。

## 从理论到代码

| {doc}`../math-fundamentals/gradient-descent` 理论 | PyTorch 实现 | 核心作用 |
|--------------------------------------------------|-------------|---------|
| 基础梯度下降 | `optim.SGD(lr=0.01)` | 沿负梯度方向更新 |
| 动量（Momentum） | `optim.SGD(momentum=0.9)` | 累积速度，减少震荡 |
| Adam | `optim.Adam(lr=0.001)` | 自适应学习率，自动调整 |
| 学习率调度 | `lr_scheduler.StepLR` | 训练后期减小步长 |

## SGD：基础梯度下降

### 数学回顾

{ref}`gradient-descent`的核心公式：

$$
\theta_{t+1} = \theta_t - \eta \cdot \nabla_\theta J(\theta)
$$

**含义**：新参数 = 当前参数 - 学习率 × 梯度

**PyTorch 实现**：

```python
import torch
import torch.nn as nn
import torch.optim as optim

# 定义模型
model = nn.Linear(10, 1)

# 创建优化器
optimizer = optim.SGD(
    model.parameters(),    # 要优化的参数
    lr=0.01               # 学习率 η
)

# 训练循环
for epoch in range(100):
    optimizer.zero_grad()      # 1. 清零梯度
    output = model(input_data) # 2. 前向传播
    loss = criterion(output, target)
    loss.backward()            # 3. 反向传播
    optimizer.step()           # 4. 更新参数（执行 θ = θ - lr * grad）
```

## Momentum：加速收敛

### 理论回顾

{ref}`gradient-descent`中的 Momentum 引入"速度"概念：

$$
v_{t+1} = \gamma v_t + \nabla_\theta J(\theta_t) \\
\theta_{t+1} = \theta_t - \eta \cdot v_{t+1}
$$

**核心思想**：新速度 = 保留的原有速度 + 新梯度，用速度代替直接梯度更新参数。

**效果**：
- 同方向梯度累积加速（下坡更快）
- 反方向梯度相互抵消（减少震荡）

### PyTorch 实现

```python
# SGD with Momentum
optimizer = optim.SGD(
    model.parameters(),
    lr=0.01,
    momentum=0.9,      # 动量系数 γ，通常 0.9
    nesterov=False     # 是否使用 Nesterov 加速梯度
)
```

**对比实验**：

```python
# 创建两个相同模型
model_vanilla = SimpleNet()
model_momentum = SimpleNet()
model_momentum.load_state_dict(model_vanilla.state_dict())

# 不同优化器
opt_vanilla = optim.SGD(model_vanilla.parameters(), lr=0.01, momentum=0.0)
opt_momentum = optim.SGD(model_momentum.parameters(), lr=0.01, momentum=0.9)

# 训练后比较：momentum 通常收敛更快，损失曲线更平滑
```

## Adam：自适应学习率

### 理论回顾

{ref}`gradient-descent`中的 Adam 结合了两种技术：
1. **一阶矩估计**（$m_t$）：梯度的移动平均，类似 Momentum
2. **二阶矩估计**（$v_t$）：梯度平方的移动平均，用于自适应学习率

更新公式：

$$
\theta_{t+1} = \theta_t - \eta \cdot \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}
$$

**核心洞察**：分母 $\sqrt{\hat{v}_t}$ 自动调整每个参数的步长：
- 历史梯度大的参数 → 步长自动变小
- 历史梯度小的参数 → 步长自动变大

### PyTorch 实现

```python
# Adam 优化器（最常用的默认选择）
optimizer = optim.Adam(
    model.parameters(),
    lr=0.001,           # 默认学习率
    betas=(0.9, 0.999), # β1 和 β2
    eps=1e-8,           # 数值稳定性
    weight_decay=0.0    # L2 正则化
)
```

**为什么选 Adam？**

| 优势 | 说明 |
|------|------|
| 自适应 | 每个参数自动调整学习率，无需手动调参 |
| 动量 | 内置一阶矩估计，加速收敛 |
| 稳定 | 偏差修正让初期训练更稳定 |
| 易用 | 对超参数不敏感，默认配置通常工作良好 |

**经验法则**：
- 不知道用什么？→ 先用 Adam(lr=1e-3)
- 图像分类追求精度？→ 最后用 SGD + Momentum 微调
- 稀疏数据（NLP）？→ Adam 的自适应特性更适合

## 优化器选择指南

| 优化器 | 优点 | 缺点 | 推荐使用场景 |
|--------|------|------|-------------|
| **SGD** | 泛化性能好 | 收敛慢，需调学习率 | 图像分类最终调优 |
| **SGD+Momentum** | 收敛快，减少震荡 | 仍需调学习率 | 通用选择 |
| **Adam** | 自适应，易使用 | 泛化可能略差 | 默认首选，NLP |
| **AdamW** | 正确的权重衰减 | 计算量稍大 | 需要正则化时 |

## 学习率调度

### 为什么需要调整学习率？

{ref}`gradient-descent`中的分析：
- **前期**：大步快走，快速接近最优
- **后期**：小步微调，精确收敛

### 常用调度策略

```python
# 1. StepLR：每 step_size 个 epoch，学习率乘以 gamma
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
# 例如：0.1 → 0.01（30 epoch 后）→ 0.001（60 epoch 后）

# 2. ExponentialLR：指数衰减
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

# 3. CosineAnnealingLR：余弦退火
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

# 4. ReduceLROnPlateau：根据验证损失调整
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, 
    mode='min',        # 监控最小值
    factor=0.1,        # 学习率乘以 0.1
    patience=10,       # 10 个 epoch 不改善才调整
    verbose=True
)
```

### 训练循环中使用

```python
# 创建优化器和调度器
optimizer = optim.SGD(model.parameters(), lr=0.1)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

for epoch in range(num_epochs):
    # 训练
    train_epoch(...)
    
    # 更新学习率
    if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
        scheduler.step(val_loss)  # 需要传入指标
    else:
        scheduler.step()          # 其他调度器不需要参数
    
    print(f"Epoch {epoch}, LR: {optimizer.param_groups[0]['lr']:.6f}")
```

## 完整示例：CIFAR-10 训练

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models

# 加载预训练 ResNet
model = models.resnet18(pretrained=True)

# 冻结 backbone（可选）
for param in model.layer1.parameters():
    param.requires_grad = False

# 修改分类头
model.fc = nn.Linear(model.fc.in_features, 10)

# 分组设置学习率（迁移学习常用）
optimizer = optim.SGD([
    {'params': model.layer1.parameters(), 'lr': 0.0},       # 冻结
    {'params': model.layer2.parameters(), 'lr': 1e-4},      # 微调
    {'params': model.layer3.parameters(), 'lr': 1e-4},
    {'params': model.layer4.parameters(), 'lr': 1e-3},
    {'params': model.fc.parameters(), 'lr': 1e-2}           # 新层，大学习率
], momentum=0.9, weight_decay=5e-4)

# 学习率调度
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

# 损失函数
criterion = nn.CrossEntropyLoss()

# 训练循环
for epoch in range(90):
    model.train()
    for images, labels in train_loader:
        images, labels = images.cuda(), labels.cuda()
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    
    scheduler.step()
    
    current_lr = optimizer.param_groups[0]['lr']
    print(f"Epoch [{epoch}/90], LR: {current_lr:.6f}")
```

## 学习率调参建议

1. **初始学习率**：
   - SGD：0.1 或 0.01
   - Adam：0.001（默认）或 0.0001

2. **调试技巧**：
   - 损失不下降？→ 学习率太大，尝试除以 10
   - 损失震荡？→ 学习率太大，或 batch size 太小
   - 收敛太慢？→ 学习率太小，或尝试 Momentum

## 总结

### 核心方法回顾

| 方法 | 作用 | 何时调用 |
|------|------|---------|
| `optimizer.zero_grad()` | 清零梯度 | 每次反向传播前 |
| `loss.backward()` | 计算梯度 | 前向传播后 |
| `optimizer.step()` | 更新参数 | 反向传播后 |
| `scheduler.step()` | 更新学习率 | 每个 epoch 后 |

### 下一步

掌握了优化器后，下一节 {doc}`./train-workflow` 我们将把所有组件整合起来——构建一个完整的训练流程。

**从"更新参数"到"完整训练流程"，让我们把知识变成可运行的系统！**
