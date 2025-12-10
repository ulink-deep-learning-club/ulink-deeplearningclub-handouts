# 完整训练流程

## 数据准备

```python
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# 数据转换
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # MNIST的均值和标准差
])

# 加载数据集
train_dataset = torchvision.datasets.MNIST(
    root='./data', train=True, download=True, transform=transform
)
test_dataset = torchvision.datasets.MNIST(
    root='./data', train=False, download=True, transform=transform
)

# 创建数据加载器
train_loader = DataLoader(train_dataset, batch_size=64, 
                          shuffle=True, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=64, 
                         shuffle=False, num_workers=2)
```

## 模型定义

```python
import torch.nn as nn
import torch.nn.functional as F

class MNISTNet(nn.Module):
    def __init__(self):
        super(MNISTNet, self).__init__()
        
        # 卷积层
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        
        # 全连接层
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        
        # 池化和Dropout
        self.pool = nn.MaxPool2d(2)
        self.dropout = nn.Dropout(0.25)
        
    def forward(self, x):
        # 卷积块1
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool(x)
        
        # 卷积块2
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool(x)
        
        # 展平
        x = x.view(-1, 64 * 7 * 7)
        
        # 全连接层
        x = self.dropout(x)
        x = self.fc1(x)
        x = F.relu(x)
        
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x
```

## 训练函数

```python
def train_epoch(model, device, train_loader, optimizer, criterion, epoch):
    """训练一个epoch"""
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        # 清零梯度
        optimizer.zero_grad()
        
        # 前向传播
        output = model(data)
        loss = criterion(output, target)
        
        # 反向传播
        loss.backward()
        
        # 更新参数
        optimizer.step()
        
        # 统计信息
        train_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
        
        # 进度显示
        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
    
    # 计算平均损失和准确率
    avg_loss = train_loss / len(train_loader)
    accuracy = 100. * correct / total
    
    return avg_loss, accuracy
```

## 测试函数

```python
def test(model, device, test_loader, criterion):
    """测试模型"""
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            
            output = model(data)
            loss = criterion(output, target)
            
            test_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
    
    avg_loss = test_loss / len(test_loader)
    accuracy = 100. * correct / total
    
    print(f'\nTest set: Average loss: {avg_loss:.4f}, '
          f'Accuracy: {correct}/{total} ({accuracy:.2f}%)\n')
    
    return avg_loss, accuracy
```

## 主训练循环

```python
def main():
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 创建模型
    model = MNISTNet().to(device)
    
    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    
    # 训练历史记录
    history = {
        'train_loss': [], 'train_acc': [],
        'test_loss': [], 'test_acc': []
    }
    
    # 训练循环
    num_epochs = 10
    for epoch in range(1, num_epochs + 1):
        # 训练
        train_loss, train_acc = train_epoch(
            model, device, train_loader, optimizer, criterion, epoch
        )
        
        # 测试
        test_loss, test_acc = test(model, device, test_loader, criterion)
        
        # 更新学习率
        scheduler.step()
        
        # 记录历史
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)
        
        # 保存最佳模型
        if test_acc >= max(history['test_acc']):
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'test_acc': test_acc,
            }, 'best_model.pth')
    
    return history
```
