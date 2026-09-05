(cnn-fundamentals-end)=
# 练习峰，登顶

从 {doc}`introduction` 的出发宣言，到 {doc}`fc-layer-basics` 的赤手空拳，{doc}`cnn-basics` 的发现装备，{doc}`dataset-preparation` 的打包行囊，{doc}`neural-training-basics` 的登山纪律，{doc}`le-net` 的首登，{doc}`exp-cmp` 的登山日志——你爬完了第一座山。

```{admonition} 登山上篇教会了你什么
:class: important

**不是"CNN 有几层"，而是"为什么 CNN 比全连接更适合图像"。**

- 全连接的 20 万参数不是"算力不够"——是**压根不看山体结构**（{ref}`inductive-bias`）
- CNN 用 320 个参数做到全连接 20 万参数的事——**把山的形状刻进装备里**
- LeNet 证明：练习峰，可以爬
- 登山日志用数据说话：赤手空拳 vs 冰镐绳索——山不会听理论，山只看数据
```

---

## 下一步：两件事

### 第一件：拆开看看

你登顶了。但登山者不满足于"知道什么有用"——他们想知道：**每件装备到底贡献了多少？**

把卷积层拆掉会跌多少？换掉激活函数呢？去掉 Dropout 呢？

{doc}`../ablation-study/index`——**拆。一件一件拆。** 这是科学：控制变量，数据说话。从"我会用"到"我理解为什么"——这是一次思维跃迁。

### 第二件：换一座山

MNIST 是 28×28 的灰度手写数字。现实世界的山比这大 1,000 倍。

ImageNet——140 万张彩色照片，1,000 个类别。这座山在 2012 年之前从未有人登顶。

你带着练习峰上积累的每一滴经验，去面对一座真正配得上"未登峰"之名的山。

{doc}`../image-net-era/introduction`——**走。真正的攀登，现在开始。**

---

```{mermaid}
graph LR
    A["🏔️ 练习峰<br/>MNIST<br/>学会攀登"] --> B["🔧 拆装备<br/>消融研究<br/>每件贡献多少？"]
    B --> C["⛰️ 真正的未登峰<br/>ImageNet<br/>征服极限"]

    style A fill:#4CAF50,color:#fff
    style C fill:#FF6F00,color:#fff
```
