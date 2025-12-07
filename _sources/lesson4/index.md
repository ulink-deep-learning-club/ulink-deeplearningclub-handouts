# MNIST数字识别：从全连接网络到卷积神经网络

**作者**: Anson, 深度学习社 (与 Kimi K2 0905 合作)  

**日期**: 2025-12-05

## 摘要

本文深入探讨了MNIST手写数字识别任务中的两种主要神经网络架构：经典全连接神经网络（Fully Connected Networks）和卷积神经网络（Convolutional Neural Networks, CNN）。文章首先系统介绍了神经网络训练的基础概念和术语，包括损失函数、优化算法、过拟合与正则化技术等核心内容。通过详细分析LeNet架构，我们展示了如何将卷积层用于特征提取，以及如何将全连接层用于分类。文章包含完整的数学推导、PyTorch实现代码，并深入对比了两种架构的优缺点。此外，我们还探讨了神经网络中的缩放定律，为理解现代深度学习的发展提供了理论基础。

```{admonition} 目录
:class: note
```{toctree}
:maxdepth: 2

introduction
neutral-training-basics
fc-layer-basics
cnn-basics
le-net
exp-cmp
scaling-law
the-end
```
