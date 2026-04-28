import torch.nn as nn
import torch.nn.functional as F

class CNNWithDropout(nn.Module):
    """
    带Dropout正则化的CNN
    
    Dropout：训练时随机"关闭"部分神经元，防止过拟合
    参见 {doc}`../neural-network-basics/neural-training-basics` 中正则化讨论
    """
    def __init__(self, dropout_rate=0.5):
        """
        Args:
            dropout_rate: 神经元被丢弃的概率（0.5表示50%）
                         常见选择：0.3-0.5
        """
        super(CNNWithDropout, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 10)
        
        # Dropout层：只在全连接层使用是常见做法
        # 卷积层通常不需要Dropout（参数量少，不易过拟合）
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 8 * 8)
        
        # 只在训练时应用Dropout（model.train()模式）
        # 测试时自动关闭（model.eval()模式）
        x = self.dropout(F.relu(self.fc1(x)))
        
        # 输出层不加Dropout
        x = self.fc2(x)
        return x

# 消融实验对比（参见 {ref}`cnn-ablation-experiment`）：
# 
# | Dropout率 | 训练准确率 | 测试准确率 | 过拟合差距 |
# |-----------|------------|------------|------------|
# | 0.0（无）  | 95%        | 78%        | 17%        |
# | 0.5（有）  | 89%        | 78%        | 11%        |
# 
# 关键观察：
# - 无Dropout：训练准确率高但测试低 = 过拟合
# - 有Dropout：训练准确率降低但测试不变 = 更好的泛化
# 
# 建议实验：
# 1. 训练 dropout_rate=0.0 的模型
# 2. 训练 dropout_rate=0.5 的模型
# 3. 对比两者的 train/test 准确率差距
# 4. 体会"正则化"的实际意义
