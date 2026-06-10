(transfer-learning)=
# 迁移学习与微调：站在巨人的肩膀上

```{only} html
如果你的数据集只有 **100 张图片**，能达到别人用 **10,000 张图片**训练出的效果吗？**迁移学习（Transfer Learning）** 就是这个答案——站在巨人的肩膀上，借用别人已经学到的知识来解决你的问题。

~~~{rubric} 本章概览
:heading-level: 2
~~~

| 章节 | 内容 | 与前面章节的联系 |
| ------ | --------- | --------- |
| {doc}`part1-intro` | 迁移学习的核心动机与形式化定义 | 为什么要迁移？ |
| {doc}`part2-taxonomy` | 从情境和方法两个维度的分类体系 | 有哪些迁移方式？ |
| {doc}`part3-model-transfer` | 特征提取、微调、分层学习率等核心技术 | {doc}`../pytorch-practice/optimiser` 的高级应用 |
| {doc}`part4-practical-guide` | 实操指南与常见问题解决方案 | 工程实践技巧 |
| {doc}`the-end` | 总结与展望 | 知识体系梳理 |

~~~{rubric} 学习路径
:heading-level: 2
~~~

本章是 {doc}`../pytorch-practice/index` 中技能的**高级应用**：

~~~{mermaid}
graph LR
    A[预训练模型<br/>已学知识] --> B[迁移学习<br/>复用知识]
    B --> C[特征提取<br/>冻结骨干]
    B --> D[微调<br/>适配任务]
    C --> E[小数据<br/>大效果]
    D --> E
~~~

**核心认知**：迁移学习不是新技术，而是前面所有知识的**综合运用**——你仍然在使用相同的 PyTorch API，只是站在了巨人的肩膀上。

~~~{rubric} 前置知识
:heading-level: 2
~~~

| 前置章节 | 本章应用 |
| ------ | --------- |
| {doc}`../pytorch-practice/neural-network-module` | 加载预训练模型、替换分类头 |
| {doc}`../pytorch-practice/optimiser` | 分层学习率、参数冻结 |
| {doc}`../pytorch-practice/train-workflow` | 完整训练流程、早停机制 |
| {doc}`../neural-network-basics/neural-training-basics` | 过拟合诊断、正则化策略 |
| {doc}`../neural-network-basics/cnn-basics` | {ref}`inductive-bias` 与预训练权重的关系 |
| {doc}`../model-architecture-design/index` | 微调中的架构改造——何时解冻、何时加模块 |
```

```{toctree}
:maxdepth: 2
:hidden:

part1-intro
part2-taxonomy
part3-model-transfer
part4-practical-guide
the-end
```
