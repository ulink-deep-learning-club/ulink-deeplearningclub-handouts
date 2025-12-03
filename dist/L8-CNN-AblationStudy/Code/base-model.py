import torch
import torch.nn as nn
import torch.nn.functional as F

class BaselineCNN(nn.Module):
    """基线CNN模型"""
    def __init__(self, num_classes=10):
        super(BaselineCNN, self).__init__()
        
        # 卷积层
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        
        # 池化层
        self.pool = nn.MaxPool2d(2, 2)
        
        # 全连接层
        self.fc1 = nn.Linear(64 * 8 * 8, 512)  # 经过两次池化后尺寸：32→16→8
        self.fc2 = nn.Linear(512, num_classes)
        
        # Dropout
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        # 卷积层1 + ReLU + 池化
        x = self.pool(F.relu(self.conv1(x)))
        
        # 卷积层2 + ReLU + 池化
        x = self.pool(F.relu(self.conv2(x)))
        
        # 展平
        x = x.view(-1, 64 * 8 * 8)
        
        # 全连接层1 + ReLU + Dropout
        x = self.dropout(F.relu(self.fc1(x)))
        
        # 输出层
        x = self.fc2(x)
        return x

# 模型实例化
model = BaselineCNN()
print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")