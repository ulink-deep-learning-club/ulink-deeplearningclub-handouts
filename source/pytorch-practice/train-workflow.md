(pytorch-train-workflow)=
# 完整训练流程

{doc}`./optimiser`中我们掌握了如何更新参数。现在的问题是：**如何把所有组件整合成一个完整的训练系统？**

在 {doc}`../neural-network-basics/neural-training-basics`中，我们学习了训练的核心概念——Epoch、Batch、过拟合与欠拟合、正则化技巧等。但这些知识零散，需要串联成可运行的代码。

**本节就是答案**：我们将构建一个完整的 MNIST 训练系统，涵盖数据加载、模型训练、验证评估、模型保存等全部环节。

## 训练流程概览

```{mermaid}
flowchart LR
    A[数据准备] --> B[模型定义]
    B --> C[训练循环]
    C --> D[验证评估]
    D --> E[模型保存]
    E --> F{继续训练?}
    F -->|是| C
    F -->|否| G[结束]
```

**核心流程**（对应 {doc}`../neural-network-basics/neural-training-basics`）：
1. **数据准备**：加载数据，划分训练/验证集，创建 DataLoader
2. **模型定义**：搭建网络架构
3. **训练循环**：前向传播 → 计算损失 → 反向传播 → 更新参数
4. **验证评估**：评估模型性能，检测过拟合
5. **模型保存**：保存最佳模型

## 数据准备

### 数据集与 DataLoader

{doc}`../neural-network-basics/neural-training-basics`中提到：**Batch Size** 是每次参数更新使用的样本数。PyTorch 用 `DataLoader` 实现这个机制。

```python
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# 数据预处理
# transforms.Compose 将多个预处理步骤串联
# MNIST 数据原始范围 [0, 255]，需要归一化到 [0, 1]
transform = transforms.Compose([
    transforms.ToTensor(),           # PIL Image → Tensor，自动除以 255
    transforms.Normalize((0.1307,), (0.3081,))  # 标准化：(x - mean) / std
])

# 加载 MNIST 数据集
# train=True 表示训练集，train=False 表示测试集
train_dataset = torchvision.datasets.MNIST(
    root='./data',      # 数据存储路径
    train=True,         # 训练集
    download=True,      # 自动下载
    transform=transform # 应用预处理
)

test_dataset = torchvision.datasets.MNIST(
    root='./data',
    train=False,        # 测试集
    download=True,
    transform=transform
)

# 创建 DataLoader
# batch_size：每批样本数（对应 neural-training-basics 中的 Batch Size）
# shuffle=True：每 epoch 打乱数据顺序，防止模型记住顺序
# num_workers：多进程加载数据，加速训练
train_loader = DataLoader(
    train_dataset, 
    batch_size=64,      # MNIST 有 60,000 样本，每 epoch 约 938 个 batch
    shuffle=True,       # 打乱顺序
    num_workers=2       # 2 个子进程加载数据
)

test_loader = DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=False,      # 测试集不需要打乱
    num_workers=2
)

print(f"训练集大小: {len(train_dataset)}")  # 60,000
print(f"测试集大小: {len(test_dataset)}")   # 10,000
print(f"每 epoch 迭代次数: {len(train_loader)}")  # 938
```

### 数据增强（可选）

{doc}`../neural-network-basics/neural-training-basics`中的**数据增强**可以增加训练样本多样性：

```python
# 训练时增强，测试时不增强
train_transform = transforms.Compose([
    transforms.RandomRotation(10),      # 随机旋转 ±10 度
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # 随机平移
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
```

## 模型定义

### CNN 架构实现

对应 {doc}`../neural-network-basics/cnn-basics`中的卷积网络设计：

```python
import torch.nn as nn
import torch.nn.functional as F

class MNISTNet(nn.Module):
    """
    MNIST 分类网络
    输入: [batch, 1, 28, 28]
    输出: [batch, 10] (10 个数字类别的 logits)
    """
    def __init__(self):
        super(MNISTNet, self).__init__()
        
        # C1: 卷积层，1→32 通道，3×3 卷积核，输出 28×28
        # 参数量: 32×1×3×3 + 32 = 320
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        
        # C2: 卷积层，32→64 通道，3×3 卷积核，输出 28×28
        # 参数量: 64×32×3×3 + 64 = 18,496
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        
        # 池化层: 2×2，输出 14×14 → 7×7
        self.pool = nn.MaxPool2d(2)
        
        # 全连接层1: 64×7×7=3136 → 128
        # 参数量: 3136×128 + 128 = 401,536
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        
        # 全连接层2: 128 → 10（类别数）
        # 参数量: 128×10 + 10 = 1,290
        self.fc2 = nn.Linear(128, 10)
        
        # Dropout: 防止过拟合（见 neural-training-basics）
        self.dropout = nn.Dropout(0.25)  # 25% 的 dropout
        
    def forward(self, x):
        # 卷积块1: [batch, 1, 28, 28] → [batch, 32, 28, 28] → [batch, 32, 14, 14]
        x = self.conv1(x)
        x = F.relu(x)           # 激活函数
        x = self.pool(x)        # 降采样
        
        # 卷积块2: [batch, 32, 14, 14] → [batch, 64, 14, 14] → [batch, 64, 7, 7]
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool(x)
        
        # 展平: [batch, 64, 7, 7] → [batch, 3136]
        x = x.view(x.size(0), -1)
        
        # 全连接 + Dropout
        x = self.dropout(x)
        x = self.fc1(x)
        x = F.relu(x)
        
        x = self.dropout(x)
        x = self.fc2(x)  # 输出 logits（未归一化）
        
        return x
```

**总参数量**：320 + 18,496 + 401,536 + 1,290 = **421,642**

## 训练函数

### 单 epoch 训练

对应 {doc}`../neural-network-basics/neural-training-basics`中的训练循环：

```python
def train_epoch(model, device, train_loader, optimizer, criterion, epoch):
    """
    训练一个 epoch
    
    Args:
        model: 神经网络模型
        device: 计算设备（CPU/GPU）
        train_loader: 训练数据加载器
        optimizer: 优化器
        criterion: 损失函数
        epoch: 当前 epoch 数
    
    Returns:
        avg_loss: 平均损失
        accuracy: 准确率
    """
    model.train()  # 设置为训练模式（启用 Dropout、BatchNorm 等）
    
    train_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        # 1. 数据转移到设备
        data, target = data.to(device), target.to(device)
        
        # 2. 清零梯度（关键！见 auto-grad）
        optimizer.zero_grad()
        
        # 3. 前向传播
        output = model(data)  # [batch, 10]
        
        # 4. 计算损失
        # CrossEntropyLoss 内部已经包含 Softmax
        loss = criterion(output, target)
        
        # 5. 反向传播
        loss.backward()
        
        # 6. 更新参数
        optimizer.step()
        
        # 7. 统计信息
        train_loss += loss.item()
        _, predicted = output.max(1)  # 取最大值的索引作为预测类别
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
        
        # 8. 进度显示
        if batch_idx % 100 == 0:
            print(f'Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\t'
                  f'Loss: {loss.item():.6f}')
    
    # 计算平均指标
    avg_loss = train_loss / len(train_loader)
    accuracy = 100. * correct / total
    
    return avg_loss, accuracy
```

**训练模式的重要性**：
- `model.train()`：启用 Dropout（随机丢弃神经元）、BatchNorm 使用 batch 统计量
- `model.eval()`：关闭 Dropout，BatchNorm 使用全局统计量

## 验证函数

### 评估模型性能

```python
def validate(model, device, val_loader, criterion):
    """
    验证模型性能
    
    Args:
        model: 神经网络模型
        device: 计算设备
        val_loader: 验证数据加载器
        criterion: 损失函数
    
    Returns:
        avg_loss: 平均验证损失
        accuracy: 验证准确率
    """
    model.eval()  # 设置为评估模式
    
    val_loss = 0.0
    correct = 0
    total = 0
    
    # 禁用梯度计算（节省内存，加速推理）
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            
            # 前向传播
            output = model(data)
            loss = criterion(output, target)
            
            # 统计
            val_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
    
    avg_loss = val_loss / len(val_loader)
    accuracy = 100. * correct / total
    
    return avg_loss, accuracy
```

## 主训练循环

### 完整训练流程

```python
import torch
import torch.optim as optim
import matplotlib.pyplot as plt

def main():
    # ========== 1. 设备设置 ==========
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # ========== 2. 数据准备 ==========
    # ...（前面的数据加载代码）...
    
    # ========== 3. 模型、损失、优化器 ==========
    model = MNISTNet().to(device)
    
    # 损失函数：交叉熵（见 neural-training-basics）
    criterion = nn.CrossEntropyLoss()
    
    # 优化器：Adam（常用选择）
    # weight_decay 对应 neural-training-basics 中的 L2 正则化
    optimizer = optim.Adam(
        model.parameters(), 
        lr=0.001,
        weight_decay=1e-4  # L2 正则化系数 λ
    )
    
    # 学习率调度器：每 5 个 epoch 学习率乘以 0.1
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    
    # ========== 4. 训练历史记录 ==========
    # 用于检测过拟合（见 neural-training-basics）
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }
    
    best_acc = 0.0
    
    # ========== 5. 训练循环 ==========
    num_epochs = 10
    
    for epoch in range(1, num_epochs + 1):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch}/{num_epochs}")
        print(f"{'='*50}")
        
        # 训练
        train_loss, train_acc = train_epoch(
            model, device, train_loader, optimizer, criterion, epoch
        )
        
        # 验证
        val_loss, val_acc = validate(model, device, test_loader, criterion)
        
        # 更新学习率
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # 记录历史
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # 打印结果
        print(f'\n训练集 - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%')
        print(f'验证集 - Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%')
        print(f'学习率: {current_lr:.6f}')
        
        # 保存最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_acc': best_acc,
            }, 'best_model.pth')
            print(f'保存最佳模型，准确率: {best_acc:.2f}%')
    
    return history
```

## 可视化训练过程

### 绘制损失和准确率曲线

对应 {doc}`../neural-network-basics/neural-training-basics`中的过拟合检测：

```python
def plot_history(history):
    """
    可视化训练历史
    用于检测过拟合：训练损失↓但验证损失↑时，说明过拟合
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # 损失曲线
    axes[0].plot(history['train_loss'], label='Train Loss')
    axes[0].plot(history['val_loss'], label='Val Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Loss Curve')
    axes[0].legend()
    axes[0].grid(True)
    
    # 准确率曲线
    axes[1].plot(history['train_acc'], label='Train Acc')
    axes[1].plot(history['val_acc'], label='Val Acc')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Accuracy Curve')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()

# 使用
# history = main()
# plot_history(history)
```

**曲线解读**：
- **理想情况**：两条曲线同步下降/上升，差距小
- **过拟合迹象**：训练准确率持续上升，验证准确率停滞或下降
- **欠拟合迹象**：两条曲线都停滞在较低水平

## 模型加载与推理

### 加载保存的模型

```python
def load_and_predict(model_path, image):
    """
    加载模型并进行预测
    
    Args:
        model_path: 模型文件路径
        image: 输入图像 tensor，形状 [1, 1, 28, 28]
    
    Returns:
        prediction: 预测的类别（0-9）
        probability: 预测概率
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 创建模型
    model = MNISTNet().to(device)
    
    # 加载权重
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 推理
    model.eval()
    with torch.no_grad():
        image = image.to(device)
        output = model(image)
        
        # Softmax 获取概率
        probabilities = F.softmax(output, dim=1)
        
        # 预测结果
        prediction = output.argmax(dim=1).item()
        confidence = probabilities.max().item()
    
    return prediction, confidence

# 使用示例
# image = test_dataset[0][0].unsqueeze(0)  # 取第一张测试图片，加 batch 维度
# pred, conf = load_and_predict('best_model.pth', image)
# print(f"预测结果: {pred}, 置信度: {conf:.2%}")
```

## 总结

### 训练流程回顾

| 步骤 | 关键代码 | 对应理论 |
|------|---------|---------|
| 数据加载 | `DataLoader` | {doc}`../neural-network-basics/neural-training-basics` 中的 Batch |
| 模型定义 | `nn.Module` | {doc}`../neural-network-basics/cnn-basics` |
| 损失函数 | `CrossEntropyLoss` | {doc}`../math-fundamentals/loss-functions` |
| 优化器 | `optim.Adam` | {doc}`../math-fundamentals/gradient-descent` |
| 反向传播 | `loss.backward()` | {doc}`../math-fundamentals/back-propagation` |
| 正则化 | `weight_decay`, `Dropout` | {doc}`../neural-network-basics/neural-training-basics` |
| 验证评估 | `model.eval()` | 检测过拟合 |
| 学习率调度 | `lr_scheduler` | 训练后期微调 |

### 完整训练脚本模板

```python
# main.py - 可运行的完整训练脚本
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

def main():
    # 配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 64
    epochs = 10
    lr = 0.001
    
    # 数据
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=True, download=True, transform=transform),
        batch_size=batch_size, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=False, transform=transform),
        batch_size=batch_size
    )
    
    # 模型
    model = MNISTNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    
    # 训练
    for epoch in range(1, epochs + 1):
        train_epoch(model, device, train_loader, optimizer, criterion, epoch)
        validate(model, device, test_loader, criterion)
    
    # 保存
    torch.save(model.state_dict(), 'mnist_model.pth')

if __name__ == '__main__':
    main()
```

### 下一步

掌握了完整训练流程后，下一节 {doc}`./debug-and-visualise` 我们将学习如何调试训练过程中的常见问题，以及如何使用 TensorBoard 等工具可视化训练过程，让"黑盒"训练变得透明可控。

**从"能训练"到"会调试"，让我们成为真正的深度学习工程师！**
