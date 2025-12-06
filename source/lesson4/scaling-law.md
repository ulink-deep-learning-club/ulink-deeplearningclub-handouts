# 神经网络缩放定律

## 计算复杂度分析

神经网络的性能通常随模型规模增大而提升，但存在边际递减效应：

```{math}
\text{性能} \propto \log(\text{参数量}) \quad \text{或} \quad \text{性能} \propto (\text{计算量})^\alpha
```

## 现代发展

从LeNet到现代CNN架构的发展历程：

1. **AlexNet (2012)**：更深网络，ReLU激活，Dropout
2. **VGG (2014)**：小卷积核堆叠，统一架构
3. **GoogLeNet (2014)**：Inception模块，多尺度特征
4. **ResNet (2015)**：残差连接，训练极深网络
5. **EfficientNet (2019)**：复合缩放，平衡深度、宽度、分辨率
