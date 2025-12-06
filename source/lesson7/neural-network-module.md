# 神经网络模块

## nn.Module基础

`nn.Module`是所有神经网络模块的基类：

```python
import torch.nn as nn

class SimpleNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNet, self).__init__()
        
        # 定义网络层
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        # 定义前向传播
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x

# 创建网络实例
model = SimpleNet(input_size=784, hidden_size=128, output_size=10)
print(model)
```

## 常用层类型

```python
# 全连接层
fc = nn.Linear(in_features=784, out_features=256)

# 卷积层
conv2d = nn.Conv2d(in_channels=3, out_channels=64, 
                    kernel_size=3, stride=1, padding=1)

# 池化层
maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
avgpool = nn.AvgPool2d(kernel_size=2, stride=2)

# 归一化层
batchnorm = nn.BatchNorm2d(num_features=64)
layernorm = nn.LayerNorm(normalized_shape=[64, 28, 28])

# 正则化层
dropout = nn.Dropout(p=0.5)
```

## 序列容器

```python
# nn.Sequential：顺序容器
model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(128, 10),
    nn.Softmax(dim=1)
)

# nn.ModuleList：模块列表
class DynamicNet(nn.Module):
    def __init__(self, layer_sizes):
        super(DynamicNet, self).__init__()
        
        self.layers = nn.ModuleList()
        for i in range(len(layer_sizes) - 1):
            self.layers.append(
                nn.Linear(layer_sizes[i], layer_sizes[i+1])
            )
            
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

# nn.ModuleDict：模块字典
class MultiHeadNet(nn.Module):
    def __init__(self):
        super(MultiHeadNet, self).__init__()
        
        self.heads = nn.ModuleDict({
            'classification': nn.Linear(128, 10),
            'regression': nn.Linear(128, 1),
            'embedding': nn.Linear(128, 64)
        })
```

## 参数管理

```python
model = SimpleNet(784, 128, 10)

# 访问参数
for name, param in model.named_parameters():
    print(f"{name}: {param.shape}")

# 获取特定参数
fc1_weight = model.fc1.weight
fc1_bias = model.fc1.bias

# 参数初始化
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)

model.apply(init_weights)

# 冻结参数
for param in model.fc1.parameters():
    param.requires_grad = False
```
