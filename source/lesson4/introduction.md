# 引言

## MNIST数据集简介

MNIST（Modified National Institute of Standards and Technology）数据集是机器学习领域最经典的数据集之一，由Yann LeCun等人创建。该数据集包含：

- 训练集：60,000张手写数字图像
- 测试集：10,000张手写数字图像
- 图像尺寸：28×28像素，灰度图像
- 类别：0-9共10个数字类别

```{figure} ../../_static/images/mnist.png
:width: 80%
:align: center

MNIST图像数据集
```

MNIST数据集之所以成为 “Hello World” 级别的基准测试，是因为：

1. **规模适中**：足够大以展示机器学习的效果，又足够小以便快速实验
2. **预处理完善**：图像已经过标准化处理，可直接用于训练
3. **评估标准明确**：分类准确率的计算简单直观
4. **历史意义**：见证了从传统机器学习到深度学习的发展历程

## LeNet的历史意义

LeNet由Yann LeCun在1989年提出[^lecun1989backpropagation]，是最早的卷积神经网络之一。其历史意义在于：

- **开创性**：首次将卷积操作引入神经网络
- **实用性**：成功应用于银行支票的手写数字识别
- **理论基础**：奠定了现代CNN架构的基础
- **持久影响**：其设计思想至今仍在使用

```{admonition} LeNet的关键创新
:class: note

- **局部感受野**：通过卷积核捕捉局部特征
- **权值共享**：大幅减少参数数量
- **下采样**：通过池化层减少空间维度
- **端到端训练**：直接从原始像素学习特征表示
```

[^lecun1989backpropagation]: Y. LeCun, B. Boser, J. S. Denker, D. Henderson, R. E. Howard, W. Hubbard, and L. D. Jackel, "Backpropagation applied to handwritten zip code recognition," *Neural Computation*, vol. 1, no. 4, pp. 541–551, Winter 1989.
