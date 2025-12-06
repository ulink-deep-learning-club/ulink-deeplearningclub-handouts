# 实验与比较

## 训练设置

```python
import torch.optim as optim
from torch.utils.data import DataLoader

# 超参数配置
config = {
    'batch_size': 64,
    'learning_rate': 0.001,
    'num_epochs': 10,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

# 创建模型
fcn_model = FullyConnectedNet()
cnn_model = LeNet5()

# 优化器
fcn_optimizer = optim.Adam(fcn_model.parameters(), lr=config['learning_rate'])
cnn_optimizer = optim.Adam(cnn_model.parameters(), lr=config['learning_rate'])

# 损失函数
criterion = nn.CrossEntropyLoss()
```

## 性能比较

```{list-table} 模型性能比较
:header-rows: 1
:widths: 20 20 20 20 20

* - **模型**
  - **参数数量**
  - **训练准确率**
  - **测试准确率**
  - **训练时间**
* - 全连接网络
  - 109,386
  - 98.5%
  - 97.8%
  - 45分钟
* - LeNet-5
  - 61,706
  - 99.2%
  - 98.9%
  - 30分钟
```

## 结果分析

从实验结果可以看出：

1. **CNN参数更少但性能更好**：得益于参数共享和局部连接
2. **训练效率更高**：CNN训练时间更短
3. **泛化能力更强**：CNN在测试集上表现更好，过拟合程度更低
