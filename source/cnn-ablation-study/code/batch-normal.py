import torch.nn as nn
import torch.nn.functional as F

class CNNWithBN(nn.Module):
    """
    带批归一化(Batch Normalization)的CNN
    
    批归一化：在每个batch内标准化特征分布
    参见 {doc}`../neural-network-basics/neural-training-basics` 中归一化讨论
    """
    def __init__(self):
        super(CNNWithBN, self).__init__()
        
        # 卷积层 + 批归一化
        # 批归一化放在卷积后、激活前是标准做法
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)  # 对32个通道分别归一化
        
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)  # 对64个通道分别归一化
        
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 10)
        
    def forward(self, x):
        # 卷积 → 批归一化 → ReLU → 池化
        # 批归一化稳定了每层的输入分布，允许使用更大学习率
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        
        x = x.view(-1, 64 * 8 * 8)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 消融实验对比：
# 模型A（无BN）：基线模型
# 模型B（有BN）：本模型
# 
# 预期结果（参见 {ref}`cnn-ablation-experiment`）：
# - 有BN的模型收敛更快（约50%的训练时间）
# - 有BN的模型允许使用更大学习率
# - 准确率可能略有提升或持平
# 
# 实验方法：
# 1. 训练两个模型各20个epoch
# 2. 记录每轮的训练/测试准确率
# 3. 绘制对比曲线，观察BN的影响
