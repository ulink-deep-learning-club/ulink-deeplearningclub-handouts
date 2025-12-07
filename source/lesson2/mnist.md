# MNIST实例：完整实现

本章将通过MNIST手写数字识别任务，完整展示计算图、反向传播和梯度下降在实际深度学习项目中的应用。

## 任务概述

MNIST（Modified National Institute of Standards and Technology）是一个经典的手写数字识别数据集，包含60,000个训练样本和10,000个测试样本。每个样本是28×28像素的灰度图像，对应0-9的数字标签。

**任务目标**：构建一个神经网络，能够准确识别手写数字。

## 数据准备

### 1. 数据集加载与预处理

```python
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

# 数据转换管道
transform = transforms.Compose([
    transforms.ToTensor(),           # 将PIL图像转换为Tensor
    transforms.Normalize((0.1307,), (0.3081,))  # MNIST的均值和标准差
])

# 加载训练集和测试集
train_dataset = torchvision.datasets.MNIST(
    root='./data', 
    train=True, 
    download=True, 
    transform=transform
)

test_dataset = torchvision.datasets.MNIST(
    root='./data', 
    train=False, 
    download=True, 
    transform=transform
)

# 创建数据加载器
batch_size = 64
train_loader = torch.utils.data.DataLoader(
    train_dataset, 
    batch_size=batch_size, 
    shuffle=True,
    num_workers=2
)

test_loader = torch.utils.data.DataLoader(
    test_dataset, 
    batch_size=batch_size, 
    shuffle=False,
    num_workers=2
)

print(f"训练集大小: {len(train_dataset)}")
print(f"测试集大小: {len(test_dataset)}")
print(f"批量大小: {batch_size}")
print(f"训练批次数: {len(train_loader)}")
print(f"测试批次数: {len(test_loader)}")
```

### 2. 数据可视化

```python
def visualize_dataset(dataset, num_samples=10):
    """可视化数据集样本"""
    fig, axes = plt.subplots(1, num_samples, figsize=(15, 3))
    
    for i in range(num_samples):
        image, label = dataset[i]
        
        # 转换为numpy数组并调整维度
        image_np = image.squeeze().numpy()
        
        axes[i].imshow(image_np, cmap='gray')
        axes[i].set_title(f'Label: {label}')
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()

# 可视化训练集样本
visualize_dataset(train_dataset)
```

### 3. 数据增强（可选）

```python
# 数据增强可以提高模型泛化能力
train_transform_augmented = transforms.Compose([
    transforms.RandomRotation(10),      # 随机旋转±10度
    transforms.RandomAffine(0, translate=(0.1, 0.1)),  # 随机平移
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
```

## 模型定义

### 1. 简单全连接神经网络

```python
import torch.nn as nn
import torch.nn.functional as F

class SimpleNN(nn.Module):
    """简单的全连接神经网络"""
    def __init__(self, input_size=784, hidden_size=128, num_classes=10):
        super(SimpleNN, self).__init__()
        
        # 网络层定义
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, num_classes)
        
        # 批量归一化层（可选，可以加速训练）
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size // 2)
        
        # Dropout层（防止过拟合）
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        # 展平输入图像 (batch_size, 1, 28, 28) -> (batch_size, 784)
        x = x.view(-1, 28*28)
        
        # 第一层：全连接 + 批量归一化 + ReLU激活 + Dropout
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        # 第二层：全连接 + 批量归一化 + ReLU激活 + Dropout
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        # 输出层：全连接（无激活函数，使用CrossEntropyLoss时会自动应用softmax）
        x = self.fc3(x)
        
        return x
    
    def get_num_parameters(self):
        """计算模型参数数量"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total_params, trainable_params

# 创建模型实例
model = SimpleNN()
total_params, trainable_params = model.get_num_parameters()
print(f"模型总参数: {total_params:,}")
print(f"可训练参数: {trainable_params:,}")
```

### 2. 模型计算图分析

```python
def analyze_computational_graph(model, input_size=(1, 1, 28, 28)):
    """分析模型的计算图"""
    # 创建虚拟输入
    dummy_input = torch.randn(input_size)
    
    # 前向传播构建计算图
    output = model(dummy_input)
    
    print("计算图分析:")
    print(f"输入形状: {dummy_input.shape}")
    print(f"输出形状: {output.shape}")
    
    # 计算FLOPs（浮点运算次数）
    # 注意：这是一个简化的估计
    flops = 0
    for layer in [model.fc1, model.fc2, model.fc3]:
        if isinstance(layer, nn.Linear):
            # 全连接层的FLOPs = 2 * input_size * output_size
            flops += 2 * layer.in_features * layer.out_features
    
    print(f"估计FLOPs: {flops:,}")
    
    return output

# 分析计算图
analyze_computational_graph(model)
```

## 训练循环

### 1. 训练函数

```python
import torch.optim as optim
from tqdm import tqdm

def train_epoch(model, device, train_loader, optimizer, criterion, epoch):
    """训练一个epoch"""
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    
    # 使用tqdm显示进度条
    pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
    
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        
        # 前向传播
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        
        # 反向传播
        loss.backward()
        
        # 梯度裁剪（防止梯度爆炸）
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # 参数更新
        optimizer.step()
        
        # 统计信息
        train_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
        
        # 更新进度条
        accuracy = 100. * correct / total
        pbar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'Acc': f'{accuracy:.2f}%'
        })
    
    avg_loss = train_loss / len(train_loader)
    accuracy = 100. * correct / total
    
    return avg_loss, accuracy
```

### 2. 验证函数

```python
def validate(model, device, test_loader, criterion):
    """在测试集上验证模型"""
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():  # 禁用梯度计算，节省内存
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            
            # 前向传播
            output = model(data)
            loss = criterion(output, target)
            
            # 统计信息
            test_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
    
    avg_loss = test_loss / len(test_loader)
    accuracy = 100. * correct / total
    
    return avg_loss, accuracy
```

### 3. 完整训练流程

```python
def train_model(model, train_loader, test_loader, num_epochs=10, lr=0.001):
    """完整的模型训练流程"""
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    model = model.to(device)
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    
    # 记录训练历史
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'learning_rates': []
    }
    
    # 训练循环
    for epoch in range(1, num_epochs + 1):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch}/{num_epochs}")
        print(f"学习率: {optimizer.param_groups[0]['lr']:.6f}")
        
        # 训练
        train_loss, train_acc = train_epoch(
            model, device, train_loader, optimizer, criterion, epoch
        )
        
        # 验证
        val_loss, val_acc = validate(model, device, test_loader, criterion)
        
        # 更新学习率
        scheduler.step()
        
        # 记录历史
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['learning_rates'].append(optimizer.param_groups[0]['lr'])
        
        # 打印结果
        print(f"训练结果: Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")
        print(f"验证结果: Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")
    
    return model, history
```

## 模型评估

### 1. 训练结果可视化

```python
def plot_training_history(history):
    """可视化训练历史"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # 损失曲线
    axes[0].plot(history['train_loss'], label='训练损失')
    axes[0].plot(history['val_loss'], label='验证损失')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('损失曲线')
    axes[0].legend()
    axes[0].grid(True)
    
    # 准确率曲线
    axes[1].plot(history['train_acc'], label='训练准确率')
    axes[1].plot(history['val_acc'], label='验证准确率')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('准确率曲线')
    axes[1].legend()
    axes[1].grid(True)
    
    # 学习率曲线
    axes[2].plot(history['learning_rates'])
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Learning Rate')
    axes[2].set_title('学习率变化')
    axes[2].grid(True)
    
    plt.tight_layout()
    plt.show()
```

### 2. 混淆矩阵

```python
from sklearn.metrics import confusion_matrix
import seaborn as sns

def plot_confusion_matrix(model, test_loader, device):
    """绘制混淆矩阵"""
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, preds = output.max(1)
            
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    
    # 计算混淆矩阵
    cm = confusion_matrix(all_targets, all_preds)
    
    # 可视化
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    plt.title('混淆矩阵')
    plt.show()
    
    # 计算每个类别的准确率
    class_acc = cm.diagonal() / cm.sum(axis=1)
    for i, acc in enumerate(class_acc):
        print(f"类别 {i}: {acc:.2%}")
    
    return cm
```

### 3. 错误分析

```python
def analyze_errors(model, test_loader, device, num_errors=10):
    """分析分类错误"""
    model.eval()
    errors = []
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, preds = output.max(1)
            
            # 找出预测错误的样本
            incorrect_mask = preds != target
            incorrect_data = data[incorrect_mask]
            incorrect_preds = preds[incorrect_mask]
            incorrect_targets = target[incorrect_mask]
            
            for i in range(min(len(incorrect_data), num_errors - len(errors))):
                errors.append({
                    'image': incorrect_data[i].cpu(),
                    'predicted': incorrect_preds[i].item(),
                    'target': incorrect_targets[i].item()
                })
            
            if len(errors) >= num_errors:
                break
    
    # 可视化错误样本
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.flatten()
    
    for i, error in enumerate(errors[:10]):
        image = error['image'].squeeze().numpy()
        axes[i].imshow(image, cmap='gray')
        axes[i].set_title(f'预测: {error["predicted"]}, 真实: {error["target"]}')
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    return errors
```

## 完整训练示例

```python
# 主程序
if __name__ == "__main__":
    # 1. 数据准备
    print("步骤1: 数据准备")
    # [数据加载代码...]
    
    # 2. 创建模型
    print("\n步骤2: 创建模型")
    model = SimpleNN()
    total_params, trainable_params = model.get_num_parameters()
    print(f"模型总参数: {total_params:,}")
    print(f"可训练参数: {trainable_params:,}")
    
    # 3. 训练模型
    print("\n步骤3: 训练模型")
    trained_model, history = train_model(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        num_epochs=10,
        lr=0.001
    )
    
    # 4. 评估模型
    print("\n步骤4: 评估模型")
    
    # 可视化训练历史
    plot_training_history(history)
    
    # 计算最终准确率
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    final_val_loss, final_val_acc = validate(
        trained_model, device, test_loader, nn.CrossEntropyLoss()
    )
    print(f"最终验证准确率: {final_val_acc:.2f}%")
    
    # 绘制混淆矩阵
    cm = plot_confusion_matrix(trained_model, test_loader, device)
    
    # 分析错误
    errors = analyze_errors(trained_model, test_loader, device)
    
    # 5. 保存模型
    print("\n步骤5: 保存模型")
    torch.save({
        'model_state_dict': trained_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'history': history,
        'final_val_acc': final_val_acc
    }, 'mnist_model.pth')
    
    print("训练完成!")
```

## 计算图、反向传播和梯度下降的体现

在这个MNIST示例中，我们可以清楚地看到三个核心概念的体现：

### 1. 计算图的体现
- PyTorch自动构建动态计算图
- 每个`nn.Module`层对应计算图中的一个子图
- 前向传播时构建计算图，反向传播时使用

### 2. 反向传播的体现
- `loss.backward()`自动执行反向传播
- 每个操作节点都有对应的梯度计算规则
- 梯度从输出层反向传播到输入层

### 3. 梯度下降的体现
- 优化器（如Adam）执行梯度下降
- 学习率调度器调整学习率
- 批量大小影响梯度估计的准确性

## 性能优化建议

### 1. 模型架构优化
- 尝试卷积神经网络（CNN）以获得更好的性能
- 增加网络深度和宽度
- 使用残差连接

### 2. 训练技巧
- 使用学习率预热
- 尝试不同的优化器（SGD with momentum, RMSProp等）
- 使用标签平滑技术

### 3. 正则化技术
- 增加Dropout率
- 使用权重衰减（L2正则化）
- 尝试数据增强

### 4. 超参数调优
- 使用网格搜索或随机搜索
- 尝试贝叶斯优化
- 使用自动机器学习（AutoML）工具

## 总结

通过这个完整的MNIST实例，我们展示了如何将计算图、反向传播和梯度下降这三个核心概念应用于实际的深度学习项目中。从数据准备、模型定义、训练循环到模型评估，每个步骤都体现了这些核心概念的重要性。

这个实例不仅提供了一个可运行的代码框架，还展示了深度学习项目的最佳实践，包括：
- 模块化的代码结构
- 完整的训练监控和可视化
- 详细的错误分析
- 模型保存和加载

读者可以通过修改这个实例来探索不同的网络架构、优化算法和训练技巧，从而深入理解深度学习的核心原理。
