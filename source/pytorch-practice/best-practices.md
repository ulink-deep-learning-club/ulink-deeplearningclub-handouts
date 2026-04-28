(best-practices)=
# 工程最佳实践

{doc}`debug-and-visualise`教会我们如何诊断和修复训练中的问题。但要成为专业的深度学习开发者，还需要掌握工程实践——**如何组织代码、优化性能、保证实验可复现**。

这些实践来自于工业界和学术界多年积累的经验，参考了{doc}`scaling-law`中的效率优化思想。

---

## 项目结构组织

一个良好组织的项目结构能让代码更易维护、更易协作。

### 推荐目录结构

```
my_project/
├── configs/              # 配置文件
│   ├── base.yaml
│   ├── experiment1.yaml
│   └── experiment2.yaml
├── data/                 # 数据处理（不包含原始数据）
│   ├── __init__.py
│   ├── datasets.py       # 自定义Dataset
│   └── transforms.py     # 数据增强
├── models/               # 模型定义
│   ├── __init__.py
│   ├── layers.py         # 自定义层
│   └── mnist_net.py      # 具体模型
├── utils/                # 工具函数
│   ├── __init__.py
│   ├── metrics.py        # 评估指标
│   ├── logger.py         # 日志记录
│   └── visualization.py  # 可视化工具
├── checkpoints/          # 模型检查点
├── logs/                 # 训练日志
├── notebooks/            # Jupyter笔记本（探索性分析）
├── tests/                # 单元测试
├── train.py              # 训练入口
├── eval.py               # 评估入口
├── config.py             # 配置管理
└── requirements.txt      # 依赖列表
```

### 为什么这样组织？

| 目录 | 作用 | 与{doc}`../neural-network-basics/scaling-law`的联系 |
|------|------|---------------------------|
| `configs/` | 集中管理超参数 | 方便进行{doc}`../neural-network-basics/scaling-law`中的系统实验 |
| `models/` | 模块化网络结构 | 支持快速迭代不同架构 |
| `utils/` | 复用工具代码 | 减少重复劳动，提高效率 |
| `checkpoints/` | 保存训练状态 | 支持断点续训和模型选择 |

---

## 配置管理

### 集中式配置

不要硬编码参数，使用配置文件：

```python
# config.py：集中管理所有配置
from dataclasses import dataclass
from typing import Optional

@dataclass
class Config:
    """训练配置"""
    # 数据配置
    data_dir: str = "./data"
    batch_size: int = 64
    num_workers: int = 4
    
    # 模型配置
    model_name: str = "MNISTNet"
    hidden_dim: int = 128
    dropout_rate: float = 0.5
    
    # 训练配置
    num_epochs: int = 10
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    
    # 优化器配置（对应{ref}`gradient-descent`中的算法）
    optimizer: str = "Adam"  # "SGD", "Adam", "AdamW"
    momentum: float = 0.9    # 仅用于SGD
    
    # 学习率调度（对应{ref}`learning-rate-scheduling`）
    scheduler: str = "StepLR"  # "StepLR", "CosineAnnealingLR"
    step_size: int = 5
    gamma: float = 0.1
    
    # 系统配置
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42  # 保证可复现
    
    # 日志配置
    log_dir: str = "./logs"
    save_dir: str = "./checkpoints"
    log_interval: int = 100  # 每100个batch记录一次

# 使用示例
config = Config(batch_size=128, learning_rate=0.0005)
print(f"使用设备: {config.device}")
print(f"学习率: {config.learning_rate}")
```

**好处**：
1. **可复现**：所有参数在一个地方，方便记录和分享
2. **易修改**：改配置不用改代码
3. **支持实验**：可以快速创建不同的配置进行对照实验

### 使用YAML配置文件

对于复杂项目，使用YAML管理配置：

```yaml
# configs/experiment.yaml
data:
  batch_size: 64
  num_workers: 4
  augmentation:
    random_crop: true
    horizontal_flip: true

model:
  name: "ResNet18"
  pretrained: true
  num_classes: 10

training:
  epochs: 100
  optimizer:
    name: "AdamW"
    lr: 0.001
    weight_decay: 0.01
  scheduler:
    name: "CosineAnnealingLR"
    T_max: 100
```

```python
# 加载YAML配置
import yaml

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

# 使用
config = load_config('configs/experiment.yaml')
print(config['training']['optimizer']['name'])  # "AdamW"
```

---

## 可复现性保证

{doc}`../neural-network-basics/scaling-law`中的实验需要严格的可复现性。如何做到？

### 固定随机种子

```python
def set_seed(seed=42):
    """
    固定所有随机种子，保证实验可复现
    
    对应 scaling law 中控制变量的思想
    """
    import random
    import numpy as np
    import torch
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 如果使用多GPU
    
    # 让cuDNN确定性运行
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    print(f"🎲 随机种子已设置为: {seed}")

# 在训练开始时调用
set_seed(42)
```

**注意**：完全确定性可能会降低训练速度（`benchmark=False`），仅在需要严格对比时使用。

### 实验记录

```python
import json
import os
from datetime import datetime

class ExperimentLogger:
    """实验记录器：保存所有实验信息"""
    
    def __init__(self, config, experiment_name=None):
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_name = experiment_name or f"exp_{self.timestamp}"
        
        # 创建实验目录
        self.exp_dir = os.path.join("experiments", self.experiment_name)
        os.makedirs(self.exp_dir, exist_ok=True)
        os.makedirs(os.path.join(self.exp_dir, "checkpoints"), exist_ok=True)
        
        # 保存配置
        self.config_path = os.path.join(self.exp_dir, "config.json")
        with open(self.config_path, 'w') as f:
            json.dump(vars(config), f, indent=2)
        
        # 保存代码版本（如果使用git）
        self._save_git_info()
        
        print(f"📁 实验目录: {self.exp_dir}")
    
    def _save_git_info(self):
        """记录git版本信息"""
        import subprocess
        try:
            git_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode().strip()
            git_branch = subprocess.check_output(['git', 'rev-parse', '--abbrev-ref', 'HEAD']).decode().strip()
            
            git_info = {
                'hash': git_hash,
                'branch': git_branch,
                'timestamp': self.timestamp
            }
            
            with open(os.path.join(self.exp_dir, "git_info.json"), 'w') as f:
                json.dump(git_info, f, indent=2)
        except:
            print("⚠️ 无法获取git信息")
    
    def save_metrics(self, metrics):
        """保存训练指标"""
        metrics_path = os.path.join(self.exp_dir, "metrics.json")
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
    
    def get_checkpoint_path(self, epoch):
        """获取检查点保存路径"""
        return os.path.join(self.exp_dir, "checkpoints", f"epoch_{epoch}.pt")
```

---

## 性能优化

{doc}`../neural-network-basics/scaling-law` 告诉我们，计算效率直接影响实验迭代速度。

### 数据加载优化

```python
# 优化前（慢）
train_loader = DataLoader(dataset, batch_size=64, shuffle=True)

# 优化后（快）
train_loader = DataLoader(
    dataset, 
    batch_size=64,
    shuffle=True,
    num_workers=4,        # 多进程加载数据，建议设为CPU核心数
    pin_memory=True,      # 将数据固定在页锁定内存，加速CPU到GPU传输
    persistent_workers=True,  # 保持worker进程存活，避免每次epoch重新创建
    prefetch_factor=2     # 每个worker预取2个batch
)
```

**性能对比**：

| 配置 | 训练时间（每epoch） | 说明 |
|------|-------------------|------|
| num_workers=0 | 120s | 主进程加载，成为瓶颈 |
| num_workers=4, pin_memory=False | 80s | 并行加载但传输慢 |
| num_workers=4, pin_memory=True | 45s | 最优配置 |

### 混合精度训练

现代 GPU（V100及以上）支持 Tensor Cores，使用混合精度可以加速2-3倍：

```python
from torch.cuda.amp import autocast, GradScaler

# 初始化梯度缩放器（处理梯度下溢）
scaler = GradScaler()

for data, target in train_loader:
    data, target = data.to(device), target.to(device)
    optimizer.zero_grad()
    
    # 使用autocast自动选择精度（FP16/FP32）
    with autocast():
        output = model(data)
        loss = criterion(output, target)
    
    # 缩放梯度避免下溢，然后反向传播
    scaler.scale(loss).backward()
    
    # 梯度裁剪（如果需要）
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    
    # 更新参数
    scaler.step(optimizer)
    scaler.update()
```

**原理**：
- FP16（半精度）：占内存少、计算快，但范围小容易溢出
- FP32（单精度）：范围大、精度高，但计算慢
- **混合精度**：前向/反向用FP16，参数更新用FP32，兼顾速度和稳定性

### 梯度累积

当GPU内存不足以支持大批量时，使用梯度累积：

```python
accumulation_steps = 4  # 累积4个batch再更新

for batch_idx, (data, target) in enumerate(train_loader):
    data, target = data.to(device), target.to(device)
    
    # 前向传播
    output = model(data)
    
    # 损失除以累积步数（保证梯度大小一致）
    loss = criterion(output, target) / accumulation_steps
    loss.backward()
    
    # 每accumulation_steps步更新一次参数
    if (batch_idx + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
        
        print(f"更新参数: batch {batch_idx}")

# 处理最后不足一个累积周期的梯度
if (batch_idx + 1) % accumulation_steps != 0:
    optimizer.step()
    optimizer.zero_grad()
```

**效果**：batch_size=16，accumulation_steps=4，等效batch_size=64

### 编译模型（PyTorch 2.0+）

```python
# PyTorch 2.0引入torch.compile，可以显著加速
model = MyModel()

# 编译模型（第一次运行会稍慢，后续更快）
compiled_model = torch.compile(model)

# 正常训练
for data, target in train_loader:
    output = compiled_model(data)
    # ...
```

**加速效果**：
- 训练：通常提速10-50%
- 推理：通常提速20-100%

---

## 模型保存与加载的最佳实践

### 保存检查点

```python
def save_checkpoint(model, optimizer, scheduler, epoch, best_acc, path):
    """
    保存完整训练状态，支持断点续训
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'best_acc': best_acc,
    }
    torch.save(checkpoint, path)
    print(f"✅ 检查点已保存: {path}")

def load_checkpoint(path, model, optimizer=None, scheduler=None):
    """
    加载训练状态，恢复训练
    """
    checkpoint = torch.load(path)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    epoch = checkpoint.get('epoch', 0)
    best_acc = checkpoint.get('best_acc', 0.0)
    
    print(f"✅ 检查点已加载，从epoch {epoch + 1}继续训练")
    return epoch, best_acc
```

### 保存最佳模型

```python
# 训练循环中的早停和模型保存
best_acc = 0.0
patience = 10  # 早停耐心值
epochs_no_improve = 0

for epoch in range(num_epochs):
    train_loss, train_acc = train_epoch(...)
    val_loss, val_acc = validate(...)
    
    # 保存最佳模型
    if val_acc > best_acc:
        best_acc = val_acc
        epochs_no_improve = 0
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'best_acc': best_acc,
            'config': config,
        }, 'best_model.pt')
        
        print(f"💾 保存最佳模型，验证准确率: {best_acc:.2f}%")
    else:
        epochs_no_improve += 1
        print(f"⏳ 验证准确率未提升，耐心值: {epochs_no_improve}/{patience}")
    
    # 早停检查（对应{ref}`early-stopping`）
    if epochs_no_improve >= patience:
        print(f"🛑 早停触发！连续{patience}个epoch未提升")
        break
```

---

## 下一步

恭喜你完成了 PyTorch 实践章节的全部内容！回顾本章学到的：

| 技能 | 对应理论 |
|------|----------|
| 张量操作 | {ref}`computational-graph`的代码实现 |
| 自动微分 | {ref}`back-propagation`的PyTorch机制 |
| 优化器 | {ref}`gradient-descent`的各种变体 |
| 调试技巧 | {doc}`debug-and-visualise`中的诊断方法 |
| 工程实践 | {doc}`scaling-law`中的效率优化 |

你已经具备了从理论到实践的完整能力。下一步建议：

1. **实战项目**：尝试实现一个完整的图像分类项目
2. **深入研究**：回到{doc}`../neural-network-basics/index`学习更复杂的架构
3. **前沿探索**：阅读论文，复现SOTA模型

{doc}`the-end` 中，我们整理了更多学习资源，帮助你在深度学习之路上继续前进。
