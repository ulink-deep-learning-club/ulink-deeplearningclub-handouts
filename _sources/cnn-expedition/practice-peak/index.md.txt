(cnn-fundamentals)=
# 上篇：练习峰

```{only} html
{doc}`../../math-fundamentals/index` 拆解了核心机制——计算图、反向传播、梯度下降。这一章讲具体怎么用这些理论开始登山。


~~~{rubric} 本章概览
:heading-level: 2
~~~

| # | 章节 | 叙事节点 | 核心问题 |
|---|------|---------|---------|
| ① | {doc}`introduction` | 出发宣言 | 为什么要攀登？ |
| ② | {doc}`fc-layer-basics` | 赤手空拳 | 为什么不看山体结构就走不远？ |
| ③ | {doc}`cnn-basics` | 发现装备 | 局部感受野 + 权值共享 = 600 倍效率差 |
| ④ | {doc}`dataset-preparation` | 打包行囊 | 数据怎么从硬盘飞到 GPU？ |
| ⑤ | {doc}`neural-training-basics` | 登山纪律 | 装备好也会摔——监控损失曲线，控制学习率 |
| ⑥ | {doc}`le-net` | 🟢 练习峰首登 | LeNet：第一个成功的 CNN 架构 |
| ⑦ | {doc}`exp-cmp` | 登山日志 | 赤手 vs 装备——数据验证 |
| 🏁 | {doc}`the-end` | 登顶回顾 | 练习峰教会了我们什么？指向消融和登山下篇 |

~~~{rubric} 学习路径
:heading-level: 2
~~~

登山上篇是数学基础的**实战延伸**：

~~~{mermaid}
graph LR
    A[赤手空拳<br/>全连接] --> B[发现装备<br/>CNN]
    B --> C[打包行囊<br/>数据流水线]
    C --> D[登山纪律<br/>训练法则]
    D --> E[🟢 首登<br/>LeNet]
    E --> F[登山日志<br/>实验验证]
    F --> G[🏁 登顶回顾]
~~~

**核心认知**：赤手空拳爬不远。把山的结构刻进装备里——这就是 {ref}`inductive-bias`。

~~~{rubric} 前置知识
:heading-level: 2
~~~

| 前置章节 | 本章如何延伸 |
|---------|------------|
| {doc}`../../math-fundamentals/index` | 计算图 → PyTorch 自动求导；梯度下降 → optimizer.step() |
| {ref}`activation-functions` | ReLU 为什么替代 Sigmoid？实践中见分晓 |
```

```{toctree}
:maxdepth: 2
:hidden:

introduction
fc-layer-basics
cnn-basics
dataset-preparation
neural-training-basics
le-net
exp-cmp
the-end
```
