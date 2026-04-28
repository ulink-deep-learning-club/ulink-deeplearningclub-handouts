import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def train_model(model, train_loader, test_loader, num_epochs=20):
    """
    训练模型并返回结果
    
    训练流程基于 {doc}`../pytorch-practice/train-workflow` 中的最佳实践
    """
    # 设备选择：优先使用GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # 损失函数：交叉熵损失，适合多分类任务
    # 参见 {doc}`../math-fundamentals/loss-functions` 中交叉熵详解
    criterion = nn.CrossEntropyLoss()
    
    # 优化器：Adam，自适应学习率
    # 参见 {doc}`../pytorch-practice/optimiser` 中优化器选择
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 记录训练过程
    train_losses, test_losses = [], []
    train_accs, test_accs = [], []
    
    for epoch in range(num_epochs):
        # ========== 训练阶段 ==========
        model.train()  # 设置为训练模式（启用Dropout等）
        train_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, targets in train_loader:
            # 数据移至GPU（如果有）
            inputs, targets = inputs.to(device), targets.to(device)
            
            # 梯度清零
            # 参见 {doc}`../pytorch-practice/auto-grad` 中梯度累积机制
            optimizer.zero_grad()
            
            # 前向传播
            outputs = model(inputs)
            
            # 计算损失
            loss = criterion(outputs, targets)
            
            # 反向传播
            loss.backward()
            
            # 参数更新
            optimizer.step()
            
            # 统计
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        
        train_losses.append(train_loss / len(train_loader))
        train_accs.append(100. * correct / total)
        
        # ========== 测试阶段 ==========
        model.eval()  # 设置为评估模式（禁用Dropout）
        test_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():  # 禁用梯度计算，节省内存
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        test_losses.append(test_loss / len(test_loader))
        test_accs.append(100. * correct / total)
        
        # 打印进度
        print(f'Epoch {epoch+1:2d} | '
              f'Train Loss: {train_losses[-1]:.4f} | '
              f'Train Acc: {train_accs[-1]:.2f}% | '
              f'Test Loss: {test_losses[-1]:.4f} | '
              f'Test Acc: {test_accs[-1]:.2f}%')
    
    return {
        'train_losses': train_losses,
        'test_losses': test_losses,
        'train_accs': train_accs,
        'test_accs': test_accs
    }

# ========== 数据加载 ==========
# CIFAR-10数据集：10类32×32彩色图像
# 数据预处理：归一化到[-1, 1]范围
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_dataset = datasets.CIFAR10(root='./data', train=True,
                                 download=True, transform=transform)
test_dataset = datasets.CIFAR10(root='./data', train=False,
                                download=True, transform=transform)

# DataLoader：批量加载数据
# batch_size=64是常用选择，平衡内存和训练速度
# 参见 {doc}`../pytorch-practice/train-workflow` 中数据加载
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# 训练基线模型
print("训练基线模型...")
print("=" * 60)
results = train_model(model, train_loader, test_loader)

# 使用建议：
# 1. 记录基线模型的最终准确率
# 2. 修改模型（如移除Dropout），重新训练
# 3. 对比两个结果，分析Dropout的作用
