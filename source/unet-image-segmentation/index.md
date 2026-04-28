(unet)=
# U-Net：图像分割的革命

## 学习目标

- **理解分割任务**：图像分割是什么、为什么比分类更难、和 CNN 的关联
- **掌握 U-Net 架构**：U 形结构、编码器-解码器、跳跃连接的直觉和原理
- **理解跳跃连接的价值**：为什么"抄近道"能同时保留位置和语义
- **能够实现 U-Net**：从核心组件到完整模型
- **了解损失函数**：Dice 损失为什么比交叉熵更适合分割任务

```{admonition} 前置知识
:class: important

本章假设你已掌握以下内容：

1. **CNN 核心机制**：卷积、池化、感受野（{doc}`../neural-network-basics/cnn-basics`）
2. **LeNet-5 架构**：典型的编码器结构（{doc}`../neural-network-basics/le-net`）
3. **参数效率思维**：为什么局部连接比全连接好（{doc}`../neural-network-basics/exp-cmp`）
4. **注意力机制**：如果没读过也不影响，但读过会更有感觉（{doc}`../attention-mechanisms/index`）
```

## 本章概览

| 章节 | 内容 | 核心收获 |
|------|------|----------|
| {doc}`introduction` | 分割问题与 U-Net 的核心思想 | "在哪"和"是什么"必须同时解决 |
| {doc}`u-net` | U 形架构：编码器、解码器、跳跃连接 | 对称结构如何同时捕捉细节和语义 |
| {doc}`loss-func` | Dice 损失与交叉熵损失 | 为什么"重叠度"比"逐像素准确率"更重要 |
| {doc}`core-impl` | 从零实现 U-Net | 核心组件 → 完整模型 |
| {doc}`practice` | 数据增强与训练技巧 | 小数据也能训练好分割模型的关键 |
| {doc}`the-end` | 应用、变体、局限与展望 | U-Net 为什么十年不过时 |

## 本章定位

{doc}`../neural-network-basics/cnn-basics` 中我们学会了 CNN 如何用卷积、池化、全连接做**分类**——输入一张图，输出一个标签。但现实中很多任务需要的不只是"这是什么"，而是 **"这些像素属于什么"**：哪里是肿瘤、哪里是道路、哪里是细胞核。

分类问题是"看整体"，分割问题是"看每个点"。这看似只是"输出变多了"（从 1 个标签到几十万个标签），但实际上是一个**本质不同的问题**——你既要知道一个大象的整体形状（高层的语义理解），又要精确到象鼻子尖端的每个像素（低层的空间精度）。CNN 的编码器擅长前者（下采样→语义），但牺牲了后者。U-Net 的答案很巧妙：**去做一个对称的解码器，再用"抄近道"（跳跃连接）把丢失的空间信息直接传送回来**。

```{mermaid}
graph LR
    A["分类<br/>这是什么？"] --> B["分割<br/>每个像素是什么？"]
    B --> C["编码器<br/>提取语义"]
    C --> D["解码器<br/>恢复空间"]
    D --> E["跳跃连接<br/>保留细节"]
```

| 前置章节 | 本章应用 |
|---------|---------|
| {doc}`../neural-network-basics/cnn-basics` | 卷积/池化→编码器的构建块 |
| {doc}`../neural-network-basics/le-net` | 编码器-分类器→编码器-解码器 |
| {doc}`../attention-mechanisms/index` | 注意力→Attention U-Net |

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
