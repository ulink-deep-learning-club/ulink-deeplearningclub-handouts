(image-net-era)=
# 下篇：真正的未登峰

```{only} html
{doc}`../practice-peak/index` 中我们爬完了练习峰——MNIST 教会了 CNN 的基础：赤手空拳走不远，把山的结构刻进装备里。

**但 MNIST 只是练习。** 现实世界的山——ImageNet——140 万张彩色照片，1,000 个类别，比练习峰大 1,000 倍。2012 年 AlexNet 首登成功，此后八年，每一次"爬不上去"都逼出了新的设计。本章追溯从多尺度工具到高原反应补给站，从学会聚焦到预算有限时精打细算的完整演化。


~~~{rubric} 本章概览
:heading-level: 2
~~~

| # | 章节 | 叙事节点 | 核心问题 |
|---|------|---------|---------|
| ⑧ | {doc}`introduction` | 一座新的山 | 练习峰的技巧，在真山上还管用吗？ |
| ⑨ | {doc}`alexnet` | 🟠 首登 ImageNet！ | ReLU + Dropout + GPU——首登的四个关键 |
| ⑩ | {doc}`inception` | 多尺度工具 | 一种冰镐打不了所有岩壁 |
| ⑪ | {doc}`res-net` | 高原反应 | 越深越缺氧——跳跃连接就是补给站 |
| ⑫ | {doc}`se-net` | 学会聚焦 | 不是所有特征都值得抓——通道注意力 |
| ⑬ | {doc}`cbam` | 空间+通道 | 哪里重要 + 什么重要——完整的"看重点" |
| ⑭ | {doc}`mobilenet` | 极致省钱 | 深度可分离卷积——一个人一把冰镐 |
| ⑮ | {doc}`efficientnet` | 精打细算 | 深度/宽度/分辨率——三个旋钮一起拧 |
| 🏁 | {doc}`the-end` | 八年回顾 | 回望来路——指向架构心法 |

~~~{rubric} 学习路径
:heading-level: 2
~~~

~~~{mermaid}
graph LR
    A[一座新的山] --> B[AlexNet<br/>首登]
    B --> C[Inception<br/>多尺度]
    C --> D[ResNet<br/>高原反应]
    D --> E[注意力<br/>学会聚焦]
    E --> F[MobileNet<br/>省钱]
    F --> G[EfficientNet<br/>花钱]
    G --> H[八年回顾]
~~~

**核心认知**：登山上篇教会了你**怎么爬**。登山下篇教会你**在真正的山上怎么活**——AlexNet 的首登、ResNet 的补给站、注意力的聚焦、MobileNet/EfficientNet 的预算学。

~~~{rubric} 前置知识
:heading-level: 2
~~~

| 前置章节 | 本章如何延伸 |
|---------|------------|
| {doc}`../practice-peak/index` | MNIST 练习峰的完整攀登经验 |
| {doc}`../practice-peak/cnn-basics` | 卷积、池化、感受野——所有装备的基础操作 |
| {doc}`../practice-peak/le-net` | LeNet——AlexNet 的直接祖先 |
| {doc}`../ablation-study/index` | 消融研究——知道每个零件贡献多少，才能系统性地设计 |
```

```{toctree}
:maxdepth: 2
:hidden:

introduction
alexnet
inception
res-net
se-net
cbam
mobilenet
efficientnet
the-end
```
