(transfer-learning)=
# 迁移学习与微调：站在巨人的肩膀上

## 摘要

还记得 {doc}`../pytorch-practice/train-workflow` 中那个 MNIST 分类器吗？我们用 60,000 张图片训练，达到了不错的准确率。但现实世界中的很多任务**没有这么多数据**——医学影像可能只有几百张，工业缺陷图片可能只有几十张。

{doc}`../neural-network-basics/neural-training-basics` 告诉我们，小数据集上从头训练深度网络极易过拟合。但如果有一种方法，能让你用**100 张图片**就达到别人用**10,000 张图片**训练的效果呢？

这就是**迁移学习（Transfer Learning）**——站在巨人的肩膀上，借用别人已经学到的知识来解决你的问题。

迁移学习是深度学习中最重要的技术范式之一，通过复用预训练模型的知识，有效解决数据稀缺、计算资源受限等实际问题。

```{admonition} 学习目标
:class: important

完成本章后，你将能够：
1. **理解迁移学习的核心思想**：解释为什么需要迁移学习，掌握领域与任务的基本定义
2. **掌握分类体系**：区分归纳式、直推式、无监督迁移学习，理解四种迁移方法的技术路线
3. **熟练运用核心技术**：正确选择并实施特征提取与微调策略
4. **避免实践陷阱**：识别并解决灾难性遗忘、过拟合等常见问题
5. **完成实际项目**：根据任务特点选择合适的预训练模型和迁移策略
```

## 本章概览

| 章节 | 内容 | 与前面章节的联系 |
|------|------|-----------------|
| {doc}`part1-intro` | 迁移学习的核心动机与形式化定义 | 为什么要迁移？ |
| {doc}`part2-taxonomy` | 从情境和方法两个维度的分类体系 | 有哪些迁移方式？ |
| {doc}`part3-model-transfer` | 特征提取、微调、分层学习率等核心技术 | {doc}`../pytorch-practice/optimiser` 的高级应用 |
| {doc}`part4-practical-guide` | 实操指南与常见问题解决方案 | 工程实践技巧 |
| {doc}`the-end` | 总结与展望 | 知识体系梳理 |

## 学习路径

本章是 {doc}`../pytorch-practice/index` 中技能的**高级应用**：

```{mermaid}
graph LR
    A[预训练模型<br/>已学知识] --> B[迁移学习<br/>复用知识]
    B --> C[特征提取<br/>冻结骨干]
    B --> D[微调<br/>适配任务]
    C --> E[小数据<br/>大效果]
    D --> E
```

**核心认知**：迁移学习不是新技术，而是前面所有知识的**综合运用**——你仍然在使用相同的 PyTorch API，只是站在了巨人的肩膀上。

## 本章定位

本章系统介绍迁移学习的基础与实践：阐述迁移学习的核心动机与领域/任务的形式化定义；从迁移情境和方法两个维度分类；深入讲解特征提取、微调、分层学习率等核心技术；最后提供实操指南与常见问题解决方案。

**学习路径**：理解思想 → 掌握分类 → 动手实践 → 解决实际问题

| 前置章节 | 本章应用 |
|---------|---------|
| {doc}`../pytorch-practice/neural-network-module` | 加载预训练模型、替换分类头 |
| {doc}`../pytorch-practice/optimiser` | 分层学习率、参数冻结 |
| {doc}`../pytorch-practice/train-workflow` | 完整训练流程、早停机制 |
| {doc}`../neural-network-basics/neural-training-basics` | 过拟合诊断、正则化策略 |
| {doc}`../neural-network-basics/cnn-basics` | {ref}`inductive-bias` 与预训练权重的关系 |
| {doc}`../model-architecture-design/index` | 微调中的架构改造——何时解冻、何时加模块 |

## 前置要求

```{admonition} 学习本章前，请确保你已经掌握
:class: caution

本讲义假设你已经熟悉神经网络训练的基本流程，包括：

1. **训练流程**：理解数据加载、前向传播、损失计算、反向传播、参数更新（{doc}`../pytorch-practice/train-workflow`）
2. **模型构建**：能够使用 PyTorch 定义神经网络，理解 `nn.Module`、层、激活函数（{doc}`../pytorch-practice/neural-network-module`）
3. **优化器**：熟悉 SGD、Adam 等优化器的使用（{doc}`../pytorch-practice/optimiser`）
4. **过拟合与正则化**：了解过拟合现象，会使用 Dropout、早停等策略（{ref}`regularization`）
5. **实际训练经验**：至少完成过 1-2 个完整的训练项目
```

```{admonition} 还没掌握？
:class: tip

如果你还没有这些基础，建议先学习 {doc}`../pytorch-practice/index` 章节。
```

## 目录

```{toctree}
:maxdepth: 2

part1-intro
part2-taxonomy
part3-model-transfer
part4-practical-guide
the-end
```
