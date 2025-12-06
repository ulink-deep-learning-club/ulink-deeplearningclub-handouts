# 卷积神经网络

## 卷积操作原理

卷积操作通过滑动窗口（卷积核）在输入图像上提取局部特征：

```{figure} ../../_static/images/conv-process.png
:width: 80%
:align: center

卷积操作示意图
```

数学上，二维卷积定义为：

```{math}
(I * K)[i,j] = \sum_{m} \sum_{n} I[i+m, j+n] \cdot K[m, n]
```

## 卷积层的优势

```{admonition} 卷积层的核心优势
:class: note

1. **局部连接**：每个神经元只连接输入图像的局部区域
2. **权值共享**：同一卷积核在整个图像上滑动，大幅减少参数
3. **平移不变性**：相同特征在不同位置被同一卷积核检测
4. **层次特征提取**：浅层提取边缘等低级特征，深层提取语义等高级特征
```

## 参数共享

```{figure} ../../_static/images/conv-param-share.png
:width: 80%
:align: center

卷积核参数共享示意图
```

参数共享使得CNN能够以少量参数处理高维输入，这是其成功的关键因素之一。
