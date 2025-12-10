# 最佳实践

## 代码组织

```python
# project/
# ├── data/
# │   ├── __init__.py
# │   └── datasets.py
# ├── models/
# │   ├── __init__.py
# │   └── mnist_net.py
# ├── utils/
# │   ├── __init__.py
# │   └── metrics.py
# ├── config.py
# ├── train.py
# └── test.py

# config.py：统一配置
class Config:
    batch_size = 64
    learning_rate = 0.001
    num_epochs = 10
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
# train.py：主训练脚本
def main(config):
    # 使用配置
    model = MNISTNet().to(config.device)
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    # ...
```

## 性能优化

```python
# 1. 使用DataLoader的多进程
train_loader = DataLoader(dataset, batch_size=64, 
                          shuffle=True, num_workers=4, 
                          pin_memory=True)

# 2. 混合精度训练
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()
for data, target in train_loader:
    optimizer.zero_grad()
    
    with autocast():
        output = model(data)
        loss = criterion(output, target)
    
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

# 3. 梯度累积
accumulation_steps = 4
for i, (data, target) in enumerate(train_loader):
    output = model(data)
    loss = criterion(output, target) / accumulation_steps
    loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

## 常见错误与解决方案

```{admonition} 常见问题与解决
:class: warning

1. **GPU内存不足**：减小批量大小，使用梯度累积，混合精度训练
2. **梯度爆炸/消失**：梯度裁剪，合适的权重初始化，批归一化
3. **过拟合**：增加Dropout，数据增强，L2正则化，早停法
4. **训练不稳定**：合适的学习率，学习率预热，梯度裁剪
5. **验证集性能差**：检查数据泄露，确保训练/验证集分布一致
```
