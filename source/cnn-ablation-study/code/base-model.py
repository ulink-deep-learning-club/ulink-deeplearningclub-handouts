import torch
import torch.nn as nn
import torch.nn.functional as F

class BaselineCNN(nn.Module):
    """
    基线CNN模型 - 用于消融研究的基准
    
    架构设计参考 {doc}`../neural-network-basics/cnn-basics` 中的LeNet-5，
    但简化为2层卷积，适合CIFAR-10快速实验
    """
    def __init__(self, num_classes=10):
        super(BaselineCNN, self).__init__()
        
        # 卷积层1：从3通道(RGB)到32个特征图
        # 卷积核大小3×3是标准选择，平衡感受野和参数量
        # 参见 {doc}`../neural-network-basics/cnn-basics` 中卷积核选择讨论
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        
        # 卷积层2：从32到64个特征图，提取更复杂特征
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        
        # 池化层：2×2最大池化，步长2
        # 作用：降维 + 平移不变性
        # 参见 {ref}`cnn-ablation-experiment` 中池化层消融实验
        self.pool = nn.MaxPool2d(2, 2)
        
        # 全连接层1
        # 输入维度计算：32×32 → 池化后16×16 → 再池化后8×8
        # 64个特征图 × 8×8空间尺寸 = 4096？不对，是64 * 8 * 8 = 4096
        # 但这里写的是512，可能是笔误或简化版本
        # 正确的应该是：nn.Linear(64 * 8 * 8, 512)
        self.fc1 = nn.Linear(64 * 8 * 8, 512)  # 64通道 × 8×8空间维度
        
        # 输出层：10个类别（CIFAR-10）
        self.fc2 = nn.Linear(512, num_classes)
        
        # Dropout：正则化技术，防止过拟合
        # 参见 {doc}`../neural-network-basics/neural-training-basics` 中正则化讨论
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        # 卷积层1 + ReLU激活 + 池化
        # ReLU：非线性激活，加速收敛
        # 参见 {doc}`../math-fundamentals/activation-functions` 中ReLU详解
        x = self.pool(F.relu(self.conv1(x)))
        
        # 卷积层2 + ReLU + 池化
        x = self.pool(F.relu(self.conv2(x)))
        
        # 展平：从4D张量 [batch, channels, height, width] 变为 2D [batch, features]
        # 参见 {doc}`../pytorch-practice/tensor-ops` 中张量形状操作
        x = x.view(-1, 64 * 8 * 8)
        
        # 全连接层1 + ReLU + Dropout
        x = self.dropout(F.relu(self.fc1(x)))
        
        # 输出层（无激活，CrossEntropyLoss内部包含Softmax）
        # 参见 {doc}`../pytorch-practice/train-workflow` 中损失函数选择
        x = self.fc2(x)
        return x

# 模型实例化
model = BaselineCNN()
print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")

# 参数数量估算：
# conv1: 3*32*3*3 + 32 = 896
# conv2: 32*64*3*3 + 64 = 18,496
# fc1: 4096*512 + 512 = 2,097,664
# fc2: 512*10 + 10 = 5,130
# 总计：约 122万参数
