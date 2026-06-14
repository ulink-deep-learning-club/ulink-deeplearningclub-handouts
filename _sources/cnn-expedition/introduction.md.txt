(cnn-expedition-intro)=
# 引言：因为山在那里

1924 年，登山家乔治·马洛里被问到："为什么要登珠峰？" 他答：**"因为山在那里。"**

深度学习面前也有一座山。一座没有山顶的山——**完美的泛化**。无论投入多少算力、数据、层数，我们永远在接近，永远到不了。

那为什么还要攀登？因为每一个"爬不上去"的困境，都逼出了一个新的设计。**攀登本身，就是理解的过程。**

```{admonition} 这一章不是百科全书
:class: important

这一章讲的不是"CNN有哪些层"或"ResNet有几条连接"。

这一章讲的是：一群人如何从**赤手空拳**出发，一步步发现更好的工具、更聪明的路线、更可靠的生存法则——直到站在离山顶最近的地方，回望来路，然后把旗帜递给下一个人。

**每一个架构都不是凭空设计的。每一个突破都是对前一代"为什么摔倒"的深刻回应。**
```

```{admonition} 已经熟悉 CNN？这里有快速路线
:class: tip

这一章非常长——占全书约 40%。如果你已经了解卷积、池化、LeNet 这些基础，不需要从头爬练习峰。

**推荐路线**：
1. 快速扫一遍 {doc}`登山上篇的引言 <practice-peak/introduction>`，重点看 {ref}`inductive-bias`——它回答了"为什么 CNN 比全连接更适合图像"，是整个上篇的核心洞察
2. 翻翻 {doc}`消融研究 <ablation-study/index>`——拆开装备、验证每个组件的贡献，这套方法论在任何实验中都有用
3. 主力时间花在 {doc}`登山下篇 <image-net-era/index>`——AlexNet、Inception、ResNet、MobileNet、EfficientNet，每一个都是对前代"为什么摔倒"的回应
4. 收尾读 {doc}`架构心法 <model-architecture-design/index>`——把具体技术升维为设计原则

**可以跳过**：{doc}`全连接：赤手空拳 <practice-peak/fc-layer-basics>`、{doc}`卷积：发现装备 <practice-peak/cnn-basics>`、{doc}`数据准备：打包行囊 <practice-peak/dataset-preparation>`（除非你想复习 MNIST 数据管线的细节）。
```

---

## 两座山

| 🏔️ {doc}`登山上篇 <practice-peak/index>` | ⛰️ {doc}`登山下篇 <image-net-era/index>` |
|------|------|
| MNIST：练习峰 | ImageNet：真正的未登峰 |
| 赤手空拳 → 发现装备 → 首登 | AlexNet首登 → 多尺度 → 高原反应 → 聚焦 → 预算学 |
| **学会攀登** | **征服极限** |

**上篇**，从赤手空拳出发，一步步发现冰镐和绳索，学会打包行囊、遵守纪律，完成第一次登顶。**每一章都是对前一章"为什么爬不上去"的回应。**

这背后的核心原理是 {ref}`inductive-bias`——把山的结构刻进装备里。全连接不看山体，20 万参数走不远；CNN 顺应山体，几百个参数就能爬更高。

两座山之间，{doc}`消融研究 <ablation-study/index>` 拆开装备——不是破坏，是理解。把卷积层拆掉会跌多少？换掉激活函数呢？每个零件到底贡献了多少？

**下篇**，换一座真正的山。ImageNet 比 MNIST 大 1,000 倍。AlexNet 在 2012 年首登成功——错误率从 28% 砸到 15%。此后八年：多尺度工具、高原反应补给站、学会聚焦的注意力、预算有限时的精打细算。练习峰教会了你**怎么爬**，登山下篇教会你**在真正的山上怎么活**。

---

## 攀登之后：把经验炼成心法

爬完两座山，你的装备包里塞满了具体技术——卷积、跳跃连接、通道注意力、深度可分离卷积。但下次面对一座全新的山，你该怎么选装备？

{doc}`架构心法 <model-architecture-design/index>` 把经验升维为原则：感受野、深度连接、注意力、效率——四个维度，四个旋钮。不是"加哪个模块"，而是"面对新任务时怎么判断该加什么"。它以**登山者笔记**收尾——回望三十年的路，提炼出可以带走的智慧。

---

## 攀登路线

```{mermaid}
graph LR
    A[🏔️ 上篇：练习峰<br/>MNIST] --> B[🔧 消融研究<br/>拆装备]
    B --> C[⛰️ 下篇：未登峰<br/>ImageNet]
    C --> D[🔬 架构心法<br/>四维度]
    D --> E[🏁 登山者笔记]

    style A fill:#4CAF50,color:#fff
    style C fill:#FF6F00,color:#fff
    style E fill:#1565C0,color:#fff
```

---

## 出发

{doc}`../../math-fundamentals/index` 给了你理论——计算图、反向传播、梯度下降。登山上篇是这些理论的**实战延伸**：

- {ref}`computational-graph` → PyTorch 自动求导
- {ref}`activation-functions` → ReLU 和 Softmax 的实际表现
- {ref}`back-propagation` → `loss.backward()` 的魔法
- {ref}`gradient-descent` → `optimizer.step()` 的参数更新

系紧鞋带。

{doc}`practice-peak/index`——走。
