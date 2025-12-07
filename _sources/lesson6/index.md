# CNN中的注意力机制：从 SE-Net 到 CBAM

**作者**: Anson, 深度学习社（与 DeepSeek V3.2 合作）

**日期**: 2025-12-05

## 摘要

本文全面探讨了卷积神经网络（CNN）中的注意力机制，从基础的Squeeze-and-Excitation Networks（SE-Net）到更复杂的Convolutional Block Attention Module（CBAM）。文章首先介绍了注意力机制的生物学灵感和数学基础，然后详细分析了SE-Net如何通过通道注意力重新校准特征图的重要性。接着，我们深入探讨了CBAM如何结合通道注意力和空间注意力，实现更全面的特征优化。通过详细的数学推导、线性代数分析和信息论视角，我们解释了这些注意力机制的工作原理。文章包含完整的PyTorch实现代码、性能比较和实际应用指南，帮助读者深入理解注意力机制在CNN中的重要作用。

```{admonition} 目录
:class: note
```{toctree}
:maxdepth: 2

introduction
se-net
spatial-attn
cbam
channel-spatial-attm-cmp
experiments
attn-math
practical-guide
extensions-variations
the-end
```
