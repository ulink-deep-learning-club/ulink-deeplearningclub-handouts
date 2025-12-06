# LeNet-5架构详解

## 网络结构

LeNet-5的经典架构包含以下层：

```{mermaid}
flowchart LR
    Input["输入\n(1×28×28)"]
    C1["卷积 C1\n(6×24×24)"]
    S2["池化 S2\n(6×12×12)"]
    C3["卷积 C3\n(16×8×8)"]
    S4["池化 S4\n(16×4×4)"]
    F5["全连接 F5\n(120)"]
    F6["全连接 F6\n(84)"]
    Out["输出\n(10)"]

    Input --> C1 --> S2 --> C3 --> S4 --> F5 --> F6 --> Out
```

## PyTorch实现

```python
import torch.nn as nn
import torch.nn.functional as F

class LeNet5(nn.Module):
    def __init__(self, num_classes=10):
        super(LeNet5, self).__init__()
        
        # 卷积层
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1)
        
        # 全连接层
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)
        
    def forward(self, x):
        # 卷积层 + 激活函数 + 池化
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        
        # 展平
        x = x.view(-1, 16 * 5 * 5)
        
        # 全连接层
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x
```

## 参数计算

LeNet-5的参数数量计算：

1. **C1层**: $5×5×1×6 + 6 = 156$ 参数
2. **C3层**: $5×5×6×16 + 16 = 2,416$ 参数  
3. **全连接层**: $(400×120) + (120×84) + (84×10) + (120+84+10) = 48,000 + 10,080 + 840 + 214 = 59,134$ 参数
4. **总计**: 约 61,706 参数

相比全连接网络的109,386参数，LeNet-5减少了约44%的参数，但性能更好。
