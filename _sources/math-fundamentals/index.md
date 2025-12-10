# 计算图、反向传播与梯度下降：深度学习核心数学基础

**作者**: Anson, 深度学习社 (与 Kimi K2 0905 合作)  
**日期**: 2025-12-05

## 摘要

本文带你深入理解深度学习的三大数学基础：计算图、反向传播和梯度下降。我们会用MNIST手写数字识别作为例子，从概念到代码，一步步拆解这些技术如何让神经网络“学会”知识。

你将学到：
- **计算图**：如何把复杂的数学计算变成直观的流程图
- **反向传播**：神经网络如何通过链式法则快速计算梯度
- **梯度下降**：如何利用梯度信息调整参数，让模型越变越准

文中不仅有清晰的数学推导和图示，还有可以直接运行的PyTorch代码。无论你是数学爱好者还是编程新手，都能通过这些内容打下坚实的深度学习基础。

```{admonition} 目录
:class: note
   ~~~{toctree}
   :maxdepth: 2

   introduction
   computational-graph
   back-propagation
   gradient-decent
   loss-functions
   activation-functions
   the-end
   ~~~
