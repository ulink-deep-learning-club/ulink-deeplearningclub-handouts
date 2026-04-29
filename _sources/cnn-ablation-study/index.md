(cnn-ablation-study)=
# CNN 消融研究：理解卷积神经网络各组件的作用

## 摘要

还记得 {doc}`../neural-network-basics/cnn-basics` 中学习的卷积、池化、激活函数吗？还记得 {doc}`../neural-network-basics/le-net` 中完整的 LeNet-5 架构吗？

我们已经知道每个组件"是什么"和"怎么做"，但还有一个更深层的问题：**每个组件对最终效果的真正贡献是多少？** 如果去掉某个组件，模型还能工作吗？如果把 ReLU 换成 Sigmoid，准确率会下降多少？

这就是**消融研究（Ablation Study）**要回答的问题——通过系统地移除或修改组件，用实验数据说话，理解每个部分的真正价值。

本章将带你进入**科学实验阶段**，通过控制变量法，掌握"提出假设→设计实验→收集数据→分析结论"的完整方法论。

```{admonition} 学习目标
:class: important

完成本章后，你将能够：
1. **理解消融研究的思想**：解释什么是控制变量法，为什么一次只能改一个组件
2. **掌握实验设计**：独立设计消融实验，包括基线模型、消融方案、评估指标
3. **量化组件贡献**：通过实验数据判断哪些组件是"必需"，哪些是"可选"
4. **应用设计原则**：根据消融结果优化神经网络架构，避免盲目堆叠组件
5. **培养科学思维**：将"提出假设→设计实验→收集数据→分析结论"的方法应用到其他领域
```

## 本章概览

| 章节 | 内容 | 与前面章节的联系 |
|------|------|-----------------|
| {doc}`introduction` | 什么是消融研究？为什么要做？ | 科学方法论：控制变量法 |
| {doc}`experiment-design` | 基线模型、实验方案、结果分析 | {doc}`../pytorch-practice/using-framework` 的实验管理 |
| {doc}`implementation` | 完整的 PyTorch 代码实现 | {doc}`../pytorch-practice/train-workflow` 的实践 |
| {doc}`the-end` | 总结与学习路径 | 知识体系梳理 |

## 学习路径

本章是从"知道怎么做"到"知道为什么这样做"的**思维跃迁**：

```{mermaid}
graph LR
    A[理论学习<br/>CNN组件] --> B[基线模型<br/>跑通完整代码]
    B --> C[消融实验1<br/>移除卷积层]
    C --> D[消融实验2<br/>更换激活函数]
    D --> E[消融实验3<br/>移除Dropout]
    E --> F[数据分析<br/>组件重要性排序]
    F --> G[设计原则<br/>优化新架构]
```

**核心认知**：消融研究不是"破坏"，而是"理解"——通过系统地做减法，看清每个组件的真实贡献。

**关键原则**：每次只改一个变量，确保结果可归因。

## 本章定位

前面章节我们学习了 CNN 的各个组件（卷积、池化、激活函数等）以及迁移学习的实践应用。本章我们进入**科学实验阶段**——通过控制变量法，理解每个组件的真正价值。

本章涉及的关键问题：
- 卷积层真的必要吗？去掉会怎样？
- 激活函数选 ReLU 还是 Sigmoid？为什么？
- Dropout 能提升多少泛化能力？
- 批归一化对训练速度的影响有多大？

**学习路径**：理解思想 → 设计实验 → 动手实现 → 分析结论

| 前置章节 | 本章应用 |
|---------|---------|
| {doc}`../neural-network-basics/cnn-basics` | 理解卷积层、池化层的作用，作为消融对象 |
| {doc}`../neural-network-basics/neural-training-basics` | 应用批归一化、Dropout 等正则化技术 |
| {doc}`../pytorch-practice/neural-network-module` | 搭建实验用的 CNN 模型 |
| {doc}`../pytorch-practice/train-workflow` | 实现训练循环，记录实验数据 |
| {doc}`../pytorch-practice/using-framework` | 使用框架管理实验、配置和模型 |
| {doc}`../transfer-learning/index` | 理解微调背后的"组件重要性"思想 |

## 前置要求

```{admonition} 学习本章前，请确保你已经掌握
:class: caution

本章假设你已掌握 {doc}`../pytorch-practice/train-workflow` 中的完整训练流程，以及 {doc}`../neural-network-basics/cnn-basics` 中 CNN 各组件的原理。建议在完成 {doc}`../pytorch-practice/using-framework`（掌握社团框架的基本使用）和 {doc}`../transfer-learning/index` 后再进行学习。

**预计耗时**：2-4 周（需要训练多个模型进行对比实验）
```

```{admonition} 提示
:class: note

本章的所有实验数据都是示例，你的实际结果可能不同——这正是科学研究的魅力所在！
```

## 目录

```{toctree}
:maxdepth: 2

introduction
experiment-design
implementation
the-end
```
