# 优化器

## 优化器基础

```python
import torch.optim as optim

model = SimpleNet(784, 128, 10)

# 创建优化器
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# 训练循环中的使用
for epoch in range(num_epochs):
    for batch_idx, (data, target) in enumerate(train_loader):
        # 清零梯度
        optimizer.zero_grad()
        
        # 前向传播
        output = model(data)
        loss = criterion(output, target)
        
        # 反向传播
        loss.backward()
        
        # 更新参数
        optimizer.step()
```

## 常用优化器

```python
# SGD with Momentum
optimizer_sgd = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# Adam
optimizer_adam = optim.Adam(model.parameters(), lr=0.001, 
                            betas=(0.9, 0.999), eps=1e-8)

# RMSprop
optimizer_rms = optim.RMSprop(model.parameters(), lr=0.01, alpha=0.99)

# Adagrad
optimizer_ada = optim.Adagrad(model.parameters(), lr=0.01)

# 不同参数组不同学习率
optimizer = optim.SGD([
    {'params': model.fc1.parameters(), 'lr': 0.01},
    {'params': model.fc2.parameters(), 'lr': 0.001}
], momentum=0.9)
```

## 学习率调度

```python
# 创建优化器
optimizer = optim.SGD(model.parameters(), lr=0.1)

# 各种调度器
scheduler_step = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
scheduler_exp = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
scheduler_plateau = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.1, patience=10
)

# 使用调度器
for epoch in range(num_epochs):
    # 训练...
    train_loss = train_epoch(model, train_loader, optimizer, criterion)
    
    # 更新学习率
    scheduler_step.step()  # 按步数更新
    # scheduler_plateau.step(train_loss)  # 根据指标更新
    
    print(f"Epoch {epoch}, LR: {optimizer.param_groups[0]['lr']}")
```
