(attention-mechanisms)=
# CNN 中的注意力机制

## 摘要

还记得 {doc}`../cnn-ablation-study/experiment-design` 中的消融实验吗？我们发现不同通道和空间位置的特征重要性差异很大。但 CNN 有一个根本局限：**它对所有输入特征一视同仁**。

卷积核滑过图像时，无论当前区域是背景还是关键目标，都使用相同的权重计算。这就像阅读时每个字都用同样的精力，不分重点——显然效率低下。

本章要回答的核心问题是：**如何让网络动态地知道"哪些特征更重要"？**

{doc}`../neural-network-basics/cnn-basics` 中讨论的 {ref}`inductive-bias` 告诉我们，好的架构设计能把先验知识内置到网络中。注意力机制正是这种思想的延续——通过增加"动态权重"组件，让网络学会**关注重要特征、抑制无关特征**。它不替换现有的卷积或池化，而是在它们之上叠加一层"可学习的放大镜"。

```{admonition} 学习目标
:class: important

完成本章后，你将能够：
1. **理解为什么需要注意力**：解释 CNN 平等对待所有特征的局限性
2. **掌握通道注意力**：理解 SE-Net 的 Squeeze-Excitation 机制及其实现
3. **掌握空间注意力**：理解空间注意力的作用"哪里重要"和实现方式
4. **理解 CBAM**：掌握通道与空间注意力的组合策略
5. **学会选择**：根据任务特点选择适合的注意力模块
```

## 本章概览

| 章节 | 内容 | 与前面章节的联系 |
|------|------|-----------------|
| {doc}`introduction` | 为什么 CNN 需要注意力？ | {doc}`../cnn-ablation-study/experiment-design` 的问题驱动 |
| {doc}`se-net` | 通道注意力（SE-Net） | 压缩→激励→缩放的直觉 |
| {doc}`spatial-attn` | 空间注意力 | 关注"哪里重要" |
| {doc}`cbam` | 通道+空间注意力（CBAM） | 两种注意力的组合策略 |
| {doc}`comparison` | 选择与应用 | 不同注意力的适用场景和性能数据 |
| {doc}`practice` | 实践指南 | 超参数调优、常见陷阱 |

## 学习路径

本章是在 CNN 基础上的**能力增强**：

```{mermaid}
graph LR
    A[CNN基础<br/>平等对待特征] --> B[发现问题<br/>不同特征重要性不同]
    B --> C[通道注意力<br/>SE-Net]
    C --> D[空间注意力<br/>关注哪里]
    D --> E[CBAM<br/>两者结合]
    E --> F[选择与应用<br/>实践指南]
```

**核心认知**：注意力不是替代 CNN，而是让 CNN "学会看重点"的可学习组件。

## 本章定位

前面章节我们学习了 CNN 的工作原理和各组件的贡献——卷积核提取特征、池化降维、激活函数引入非线性、批归一化加速训练。但所有这些组件都有一个共同点：**它们对所有输入特征一视同仁**。

注意力机制的解决方案：
- **通道注意力**：让网络学会"哪些特征通道更重要"（SE-Net）
- **空间注意力**：让网络学会"图像的哪些位置更重要"
- **组合注意力**：同时关注"什么"和"哪里"（CBAM）

**学习路径**：理解局限 → 掌握机制 → 动手实现 → 学会选择

| 前置章节 | 本章应用 |
|---------|---------|
| {doc}`../cnn-ablation-study/experiment-design` | 各组件贡献分析 → 注意力能进一步提升 |
| {doc}`../neural-network-basics/cnn-basics` | 卷积特征提取 → 通道/空间的重要性差异 |
| {doc}`../pytorch-practice/neural-network-module` | nn.Module 实现注意力模块 |

## 前置要求

```{admonition} 学习本章前，请确保你已经掌握
:class: caution

本章假设你已掌握以下内容：

1. **CNN 基础知识**：卷积层、池化层、全连接层的作用（{doc}`../neural-network-basics/cnn-basics`）
2. **CNN 各组件的贡献**：通过消融研究理解每个组件的重要性（{doc}`../cnn-ablation-study/index`）
3. **PyTorch 基础**：能独立搭建和训练模型（{doc}`../pytorch-practice/neural-network-module`）
```

```{admonition} 还没掌握？
:class: tip

建议先完成 {doc}`../cnn-ablation-study/index` 的学习，理解为什么不同特征的重要性会有差异。
```

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
