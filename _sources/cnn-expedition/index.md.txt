(cnn-expedition)=
# 翻越两座山：CNN 三十年攀登史

```{only} html
{doc}`../../math-fundamentals/index` 给了你理论。但知道公式，和设计出真正能工作的网络，是两回事。

**深度学习面前有一座山：完美的泛化。** 我们永远到不了山顶，但每一次攀登都让我们离它更近。本章是一场翻越——两座山，三十年。{doc}`登山上篇 <practice-peak/index>` 爬练习峰（MNIST），{doc}`登山下篇 <image-net-era/index>` 换真正的未登峰（ImageNet）。

~~~{rubric} 本章概览
:heading-level: 2
~~~

| # | 章节 | 叙事节点 | 核心问题 |
|---|------|---------|---------|
| | {doc}`introduction` | 出发宣言 | 为什么要攀登？ |
| | | | |
| 🏔️ | {doc}`practice-peak/index` | 上篇：练习峰 MNIST | 赤手空拳 → 发现装备 → 首登 |
| 🔧 | {doc}`ablation-study/index` | 拆解装备 | 每个零件到底贡献了多少？ |
| ⛰️ | {doc}`image-net-era/index` | 下篇：真正的未登峰 | AlexNet首登 → 高原反应 → 聚焦 → 预算学 |
| 🔬 | {doc}`model-architecture-design/index` | 架构心法 → 🏁 登山者笔记 | 感受野/深度连接/注意力/效率——四维度 + 三十年回望 |
| | | | |
| 🏁 | {doc}`the-end` | 下一座山 | 从设计到落地：模型部署 |

~~~{rubric} 学习路径
:heading-level: 2
~~~

~~~{mermaid}
graph LR
    A[🏔️ 上篇：练习峰<br/>MNIST] --> B[🔧 消融研究<br/>拆装备]
    B --> C[⛰️ 下篇：未登峰<br/>ImageNet]
    C --> D[🔬 架构心法<br/>四维度]
    D --> E[🏁 登山者笔记]

    style A fill:#4CAF50,color:#fff
    style C fill:#FF6F00,color:#fff
    style E fill:#1565C0,color:#fff
~~~

**核心认知**：每一个架构都不是凭空设计的。每一个突破都是对"上一代为什么摔倒"的深刻回应。

~~~{rubric} 前置知识
:heading-level: 2
~~~

| 前置章节 | 本章如何延伸 |
|---------|------------|
| {doc}`../../math-fundamentals/index` | 计算图 → PyTorch 自动求导；梯度下降 → optimizer.step() |
| {ref}`inductive-bias` | 为什么 CNN 比全连接更适合图像——把山的结构刻进装备里 |
```

```{toctree}
:maxdepth: 2
:hidden:

introduction
practice-peak/index
ablation-study/index
image-net-era/index
model-architecture-design/index
the-end
```
