(cnn-ablation-study)=
# CNN消融研究：理解卷积神经网络各组件的作用

## 学习目标

- **理解消融研究的思想**：解释什么是控制变量法，为什么一次只能改一个组件
- **掌握实验设计**：独立设计消融实验，包括基线模型、消融方案、评估指标
- **量化组件贡献**：通过实验数据判断哪些组件是"必需"，哪些是"可选"
- **应用设计原则**：根据消融结果优化神经网络架构，避免盲目堆叠组件
- **培养科学思维**：将"提出假设→设计实验→收集数据→分析结论"的方法应用到其他领域

```{admonition} 前置知识
:class: important

本章假设你已掌握 {doc}`../pytorch-practice/train-workflow` 中的完整训练流程，以及 {doc}`../neural-network-basics/cnn-basics` 中 CNN 各组件的原理。建议在完成 {doc}`../pytorch-practice/using-framework`（掌握社团框架的基本使用）和 {doc}`../transfer-learning/index` 后再进行学习。

**预计耗时**：2-4周（需要训练多个模型进行对比实验）
```

## 本章概览

| 章节 | 内容 | 核心收获 |
|------|------|----------|
| {doc}`introduction` | 什么是消融研究？为什么要做？ | 科学方法论：控制变量法 |
| {doc}`experiment-design` | 基线模型、实验方案、结果分析 | 实验设计能力 |
| {doc}`implementation` | 完整的 PyTorch 代码实现 | 动手实践能力 |
| {doc}`the-end` | 总结与学习路径 | 知识体系梳理 |

## 本章定位

前面章节我们学习了 CNN 的各个组件（卷积、池化、激活函数等）以及迁移学习的实践应用。本章我们进入**科学实验阶段**——通过控制变量法，理解每个组件的真正价值。

**学习路径**：理解思想 → 设计实验 → 动手实现 → 分析结论

```{mermaid}
graph LR
    A[理论学习<br/>CNN组件] --> B[基线模型<br/>跑通完整代码]
    B --> C[消融实验1<br/>移除卷积层]
    C --> D[消融实验2<br/>更换激活函数]
    D --> E[消融实验3<br/>移除Dropout]
    E --> F[数据分析<br/>组件重要性排序]
    F --> G[设计原则<br/>优化新架构]
```

**关键原则**：每次只改一个变量，确保结果可归因。

| 前置章节 | 本章应用 |
|---------|---------|
| {doc}`../neural-network-basics/cnn-basics` | 理解卷积层、池化层的作用，作为消融对象 |
| {doc}`../neural-network-basics/neural-training-basics` | 应用批归一化、Dropout等正则化技术 |
| {doc}`../pytorch-practice/neural-network-module` | 搭建实验用的 CNN 模型 |
| {doc}`../pytorch-practice/train-workflow` | 实现训练循环，记录实验数据 |
| {doc}`../pytorch-practice/using-framework` | 使用框架管理实验、配置和模型 |
| {doc}`../transfer-learning/index` | 理解微调背后的"组件重要性"思想 |

## 目录

```{toctree}
:maxdepth: 2

introduction
experiment-design
implementation
the-end

```

```{admonition} 提示
:class: note
本章的所有实验数据都是示例，你的实际结果可能不同——这正是科学研究的魅力所在！
```
