(attention-mechanisms)=
# CNN中的注意力机制

## 学习目标

- **理解为什么需要注意力**：解释CNN平等对待所有特征的局限性
- **掌握通道注意力**：理解SE-Net的Squeeze-Excitation机制及其实现
- **掌握空间注意力**：理解空间注意力的作用和实现方式
- **理解CBAM**：掌握通道与空间注意力的组合策略
- **学会选择**：根据任务特点选择适合的注意力模块

```{admonition} 前置知识
:class: important

本章假设你已掌握以下内容：

1. **CNN基础知识**：卷积层、池化层、全连接层的作用（{doc}`../neural-network-basics/cnn-basics`）
2. **CNN各组件的贡献**：通过消融研究理解每个组件的重要性（{doc}`../cnn-ablation-study/index`）
3. **PyTorch基础**：能独立搭建和训练模型（{doc}`../pytorch-practice/neural-network-module`）
```

## 本章概览

| 章节 | 内容 | 核心收获 |
|------|------|----------|
| {doc}`introduction` | 为什么CNN需要注意力？ | 问题驱动：平等处理特征的局限 |
| {doc}`se-net` | 通道注意力（SE-Net） | 压缩→激励→缩放的直觉 |
| {doc}`spatial-attn` | 空间注意力 | 关注"哪里重要" |
| {doc}`cbam` | 通道+空间注意力（CBAM） | 两种注意力的组合 |
| {doc}`comparison` | 选择与应用 | 不同注意力的适用场景和性能数据 |
| {doc}`practice` | 实践指南 | 超参数调优、常见陷阱 |

## 本章定位

前面章节我们学习了 CNN 的工作原理和各组件的贡献——卷积核提取特征、池化降维、激活函数引入非线性、批归一化加速训练。但所有这些组件都有一个共同点：**它们对所有输入特征一视同仁**。{doc}`../cnn-ablation-study/experiment-design` 的消融实验告诉我们，不同通道和空间位置的重要性其实差异很大。本章要回答的核心问题是：**如何让网络动态地知道"哪些特征更重要"？** {doc}`../neural-network-basics/cnn-basics` 中讨论的 {ref}`inductive-bias` 告诉我们，好的架构设计能把先验知识内置到网络中。注意力机制正是这种思想的延续——通过增加"动态权重"组件，让网络学会关注重要特征、抑制无关特征。它不替换现有的卷积或池化，而是在它们之上叠加一层"可学习的放大镜"。

```{mermaid}
graph LR
    A["CNN局限<br/>平等对待所有特征"] --> B["通道注意力<br/>SE-Net"]
    B --> C["空间注意力<br/>关注哪里"]
    C --> D["CBAM<br/>两者结合"]
    D --> E["选择与应用<br/>实践指南"]
```

| 前置章节 | 本章应用 |
|---------|---------|
| {doc}`../cnn-ablation-study/experiment-design` | 各组件贡献分析→注意力能进一步提升 |
| {doc}`../neural-network-basics/cnn-basics` | 卷积特征提取→通道/空间的重要性差异 |
| {doc}`../pytorch-practice/neural-network-module` | nn.Module实现注意力模块 |

## 目录

```{toctree}
:maxdepth: 2

introduction
se-net
spatial-attn
cbam
comparison
practice
the-end
```
