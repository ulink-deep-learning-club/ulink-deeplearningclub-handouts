(arch-end)=
# 总结与展望

恭喜！你已经完成了 CNN 架构改造与优化章节。

从 {doc}`part1-intro` 的信息论诊断工具，到 {doc}`part2-receptive-field` 的控制感受野，{doc}`part3-depth-connection` 的信息流动，{doc}`part4-attention` 的信息路由，{doc}`part5-efficiency` 的效率权衡，直至 {doc}`part6-diagnosis` 的心法——你已不再只是"会用架构"，而是真正理解了**如何设计架构**。

## 核心概念映射

| 维度 | 核心问题 | 关键策略 | 信息论解释 |
|------|---------|---------|-----------|
| 感受野 | 看多大范围？ | 多尺度、空洞卷积、FPN | 多频段采样，最大化 $I(X;Y)$ |
| 深度与连接 | 信息怎么传？ | 跳跃连接、密集连接、特征融合 | 防止信息瓶颈，保证梯度通路 |
| 注意力 | 重点看哪？ | 通道注意力、空间注意力、长程依赖 | 最大化任务相关互信息 |
| 效率 | 怎么更快？ | DW 卷积、Bottleneck | 利用信息冗余降本 |

## 与前面章节的联系

本章是前面所有知识的**升维整合**：

| 前置知识 | 本章如何升维 |
|---------|------------|
| {doc}`../neural-network-basics/cnn-basics` 的卷积机制 | 感受野不只是被动增长，可以主动操控 |
| {doc}`../neural-network-basics/res-net` 的残差连接 | 跳跃连接不是特例，是通用信息通路设计原则 |
| {doc}`../neural-network-basics/inception` 的多尺度 | 多尺度不是一种架构，是一类设计心法 |
| {doc}`../attention-mechanisms/index` 的注意力 | 注意力不只是模块，是信息路由机制 |

**核心认知**：好的架构设计不是"记住了哪些技术"，而是掌握了**四个维度的trade-off**——知道什么时候牺牲哪个维度来换取另一个维度的提升。

## 关键数字速查

### 改造策略对比

| 策略 | 效果提升 | 效率影响 | 实施难度 |
|------|---------|---------|---------|
| 多尺度并行（Inception） | +3-5% | 中等增加 | 中等 |
| 空洞卷积 | 感受野扩大不增参 | 几乎零影响 | 低 |
| FPN | +5-8%（检测） | 中等增加 | 高 |
| 跳跃连接 | 使深层可能（质变） | 零影响 | 低 |
| SE 注意力 | +1-2% | 增加 <3% 参数量 | 低 |
| DW 卷积 | -1% 精度 | 减少 5-10× 计算量 | 低 |

### 帕累托边界上的里程碑

| 模型 | 年份 | 核心贡献 | 推动边界的维度 |
|------|------|---------|--------------|
| Inception | 2014 | 多尺度 + 1×1 降维 | 感受野 + 效率 |
| ResNet | 2015 | 跳跃连接 | 深度（信息流动） |
| MobileNet | 2017 | DW 卷积 | 效率 |
| SE-Net | 2018 | 通道注意力 | 注意力 |
| EfficientNet | 2019 | 复合缩放 | 系统性多维度 |
| ConvNeXt | 2022 | 现代化 CNN | 多个维度精细优化 |

## 下一节方向

掌握了架构设计的心法后，你可以探索：

1. **{doc}`../transfer-learning/index`**：如何在有限数据下应用这些心法——站在预训练模型的肩膀上改造
2. **{doc}`../unet-image-segmentation/index`**：分割任务中的多尺度与跳跃连接实战
3. **{doc}`../cnn-ablation-study/index`**：用科学方法验证你的改造是否有效

## 推荐资源

### 必读论文

- He et al., "Deep Residual Learning for Image Recognition", CVPR 2016 — ResNet 原始论文{cite}`he2016deep`
- Szegedy et al., "Going Deeper with Convolutions", CVPR 2015 — Inception 原始论文{cite}`szegedy2015going`
- Howard et al., "MobileNets: Efficient Convolutional Neural Networks", 2017{cite}`howard2017mobilenets`
- Lin et al., "Feature Pyramid Networks for Object Detection", CVPR 2017

### 拓展阅读

- Tishby et al., "The Information Bottleneck Method", 2000 — 信息瓶颈理论入门{cite}`tishby2000information`
- Tan et al., "EfficientNet: Rethinking Model Scaling", ICML 2019 — 系统化的多维度缩放
- Liu et al., "A ConvNet for the 2020s", CVPR 2022 — ConvNeXt，展示如何用设计心法现代化 CNN

---

## 最后的话

这一章的核心哲学可以总结为一句话：

> **架构设计不是技术堆砌，而是理解信息如何流动，然后优化它。**

感受野决定入口宽度，连接决定通道质量，注意力决定路由智慧，效率决定实用价值。

这四个维度不是孤立的——伟大的架构（如 ResNet、EfficientNet）总是在多个维度上同时推动帕累托边界。而你的任务，就是用这些心法，设计出你自己的"伟大架构"。

**下一步**：{doc}`../transfer-learning/index`——站在巨人的肩膀上改造。

---

## 参考文献

```{bibliography}
:filter: docname in docnames
```
