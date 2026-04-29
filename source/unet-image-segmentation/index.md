(unet-image-segmentation)=
# U-Net：图像分割的革命

## 摘要

还记得 {doc}`../neural-network-basics/cnn-basics` 中的 CNN 分类吗？输入一张图，输出一个标签——"这是一只猫"。但现实中很多任务需要的不只是"这是什么"，而是 **"这些像素属于什么"**：哪里是肿瘤、哪里是道路、哪里是细胞核。

分类问题是"看整体"，分割问题是"看每个点"。这看似只是"输出变多了"（从 1 个标签到几十万个标签），但实际上是一个**本质不同的问题**——你既要知道一个大象的整体形状（高层的语义理解），又要精确到象鼻子尖端的每个像素（低层的空间精度）。

CNN 的编码器擅长前者（下采样→语义），但牺牲了后者。U-Net 的答案很巧妙：**去做一个对称的解码器，再用"抄近道"（跳跃连接）把丢失的空间信息直接传送回来**。

```{admonition} 学习目标
:class: important

完成本章后，你将能够：
1. **理解分割任务**：图像分割是什么、为什么比分类更难、和 CNN 的关联
2. **掌握 U-Net 架构**：U 形结构、编码器-解码器、跳跃连接的直觉和原理
3. **理解跳跃连接的价值**：为什么"抄近道"能同时保留位置和语义
4. **能够实现 U-Net**：从核心组件到完整模型
5. **了解损失函数**：Dice 损失为什么比交叉熵更适合分割任务
```

## 本章概览

| 章节 | 内容 | 与前面章节的联系 |
|------|------|-----------------|
| {doc}`introduction` | 分割问题与 U-Net 的核心思想 | "在哪"和"是什么"必须同时解决 |
| {doc}`u-net` | U 形架构：编码器、解码器、跳跃连接 | {doc}`../neural-network-basics/le-net` 的对称扩展 |
| {doc}`loss-func` | Dice 损失与交叉熵损失 | {ref}`loss-functions` 的进阶应用 |
| {doc}`core-impl` | 从零实现 U-Net | {doc}`../pytorch-practice/neural-network-module` 的实践 |
| {doc}`practice` | 数据增强与训练技巧 | {doc}`../neural-network-basics/neural-training-basics` 的迁移 |
| {doc}`the-end` | 应用、变体、局限与展望 | U-Net 为什么十年不过时 |

## 学习路径

本章是从"分类"到"分割"的**范式跃迁**：

```{mermaid}
graph LR
    A[CNN分类<br/>这是什么？] --> B[图像分割<br/>每个像素是什么？]
    B --> C[编码器<br/>提取语义]
    C --> D[解码器<br/>恢复空间]
    D --> E[跳跃连接<br/>保留细节]
```

**核心认知**：分割不是"输出更多的分类"，而是需要同时解决"是什么"和"在哪"两个互补问题。

## 本章定位

{doc}`../neural-network-basics/cnn-basics` 中我们学会了 CNN 如何用卷积、池化、全连接做**分类**。但 CNN 的编码器通过下采样获取语义信息时，丢失了精确的空间位置信息。

U-Net 的解决方案：
- **编码器**：像 CNN 一样下采样，提取高级语义
- **解码器**：对称地上采样，恢复空间分辨率
- **跳跃连接**：把编码器的低层特征直接传到解码器，保留细节

**学习路径**：理解分割问题 → 掌握 U-Net 架构 → 动手实现 → 训练技巧

| 前置章节 | 本章应用 |
|---------|---------|
| {doc}`../neural-network-basics/cnn-basics` | 卷积/池化 → 编码器的构建块 |
| {doc}`../neural-network-basics/le-net` | 编码器-分类器 → 编码器-解码器 |
| {doc}`../attention-mechanisms/index` | 注意力 → Attention U-Net |

## 前置要求

```{admonition} 学习本章前，请确保你已经掌握
:class: caution

本章假设你已掌握以下内容：

1. **CNN 核心机制**：卷积、池化、感受野（{doc}`../neural-network-basics/cnn-basics`）
2. **LeNet-5 架构**：典型的编码器结构（{doc}`../neural-network-basics/le-net`）
3. **参数效率思维**：为什么局部连接比全连接好（{doc}`../neural-network-basics/exp-cmp`）
4. **PyTorch 基础**：能独立搭建和训练模型（{doc}`../pytorch-practice/neural-network-module`）
```

```{admonition} 还没掌握？
:class: tip

如果注意力机制（{doc}`../attention-mechanisms/index`）还没读过也不影响，但读过会更有感觉。
```

## 目录

```{toctree}
:maxdepth: 2

introduction
u-net
loss-func
core-impl
practice
the-end
```
