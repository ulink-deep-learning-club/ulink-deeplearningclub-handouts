# PyTorch基础教程：从NumPy到深度学习

```{admonition} 文档信息
:class: note

**作者**: Anson, 深度学习社（与 DeepSeek V3.2 合作）  
**最后更新**: 2025-12-15  
```

## 摘要

本文为具有Python背景的初学者提供全面的PyTorch教程。我们从NumPy的基础知识出发，逐步引入PyTorch的核心概念，包括张量操作、自动微分、神经网络模块和优化算法。通过对比NumPy和PyTorch的相似性与差异，我们帮助读者平滑过渡到深度学习框架。

文章包含大量实际代码示例，涵盖数据加载、模型定义、训练循环和模型评估等完整流程。我们还介绍了调试技巧、可视化工具和最佳实践，帮助读者避免常见错误并提高开发效率。最后，通过一个完整的MNIST手写数字分类示例，我们将所有概念整合到实际应用中。

```{admonition} 学习目标
:class: important

完成本教程后，您将能够：
1. 理解PyTorch的核心概念和设计哲学
2. 熟练使用PyTorch张量进行各种操作
3. 掌握自动微分和梯度计算机制
4. 构建和训练自定义神经网络模型
5. 应用最佳实践进行模型调试和优化
6. 实现完整的深度学习项目工作流
```

## 目录结构

```{admonition} 快速导航
:class: note
~~~{toctree}
:maxdepth: 2

introduction
from-numpy-to-pytorch
tensor-ops
auto-grad
optimiser
neural-network-module
best-practices
debug-and-visualise
train-workflow
the-end
~~~
```

## 前置要求

```{admonition} 必备知识
:class: caution

在开始学习本教程前，请确保您具备以下基础：
1. **Python编程**：熟悉Python语法和基本数据结构
2. **NumPy基础**：了解数组操作和数学函数
3. **线性代数**：理解矩阵运算和向量空间
4. **微积分**：掌握导数和偏导数的概念
5. **基础深度学习概念**：了解神经网络的基本原理
```

## 环境配置

```{admonition} 安装指南
:class: tip

推荐使用以下环境配置：
1. **Python 3.8+**：最新稳定版本
2. **PyTorch 2.0+**：`pip install torch torchvision`
3. **Jupyter Notebook**：用于交互式学习
4. **CUDA 11.8+**：如需GPU加速（可选）
5. **uv / Conda**: **可选的**虚拟环境管理工具

详细安装说明请参考[PyTorch官方文档](https://pytorch.org/get-started/locally/)。
```
