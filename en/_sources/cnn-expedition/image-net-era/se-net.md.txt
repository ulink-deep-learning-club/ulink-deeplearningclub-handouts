(se-net)=
# SE-Net：学会聚焦

{doc}`res-net` 建好了补给站，解决了高原反应。但还有一个更深的问题：

**CNN 对所有特征一视同仁。** 无论这个通道编码的是"狗耳朵"还是"背景噪声"，卷积核给的权重一样。就像登山者盯着每一块岩石——包括那些根本不值得抓的碎石。

登山老手不会这样。他们知道什么时候聚焦裂缝，什么时候忽略光滑的岩面。**不是所有特征都值得抓。**

SE-Net（Squeeze-and-Excitation Networks）{cite}`hu2018squeeze` 给 CNN 装上了"可学习的放大镜"——让网络自己判断"哪些通道重要"，然后动态分配权重。

它的核心洞察只有三行代码：**压缩 → 激励 → 缩放**。

想想你在管理一个团队：你有很多个员工（通道），每个员工都有不同的专长。但你的资源有限，需要决定谁更重要。

```{admonition} SE-Net的三步直觉
:class: tip

1. **压缩（Squeeze）**：你让每个员工交一份"工作总结"——用一句话概括他这周做了什么。对应到网络：把每个通道的整个特征图压缩成一个数字（全局平均池化）。
2. **激励（Excite）**：你看了所有总结，判断谁的工作更重要——给每个员工分配一个"重要性分数"。对应到网络：用两个全连接层学习通道间的依赖关系，输出每个通道的权重。
3. **缩放（Scale）**：按重要性分配资源——重要员工获得更多支持，不重要员工减少资源。对应到网络：把权重乘回原始特征图，重要通道被增强，不重要通道被抑制。
```

**关键洞察**：SE-Net 不改变网络的结构（通道数不变），而是改变"流经每个通道的信息量"——就像调节音量旋钮，而不是换音箱。

```{figure} ../../../_static/images/se-block.png
:width: 400px
:align: center

SE-Net 模块结构：Squeeze → Excite → Scale
```

## 三步详解

### 1. Squeeze：压缩

输入是一个特征图 $X \in \mathbb{R}^{C \times H \times W}$（$C$ 个通道，每个 $H \times W$）。Squeeze 操作把每个通道压缩成单个数字：

```{math}
z_c = \frac{1}{H \times W} \sum_{i=1}^H \sum_{j=1}^W x_c(i,j)
```

这其实就是全局平均池化（Global Average Pooling）。为什么用平均？因为我们想获取每个通道的"全局统计信息"——不只看局部响应，而是看整个特征图的平均激活强度。

```{note}
**直觉**：如果一个通道的平均激活值很高，说明该通道编码的特征在整张图片中都很"活跃"，可能很重要。反之，平均激活值低的通道可能对当前任务贡献不大。
```

### 2. Excitation：学习权重

有了每个通道的"工作总结"$z \in \mathbb{R}^C$，接下来要学习通道间的依赖关系：

```{math}
s = \sigma(W_2 \cdot \delta(W_1 \cdot z))
```

其中：

- $W_1 \in \mathbb{R}^{C/r \times C}$：降维层，先把 $C$ 维压缩到 $C/r$ 维
- $\delta$：ReLU 激活函数
- $W_2 \in \mathbb{R}^{C \times C/r}$：升维层，恢复回 $C$ 维
- $\sigma$：Sigmoid 激活函数，输出 $(0,1)$ 之间的权重

**为什么用瓶颈结构（先降维再升维）？**

```{admonition} 瓶颈结构的设计动机
:class: note

1. **减少参数量**：直接学习 $C \times C$ 的变换需要 $C^2$ 个参数。用瓶颈结构只需要 $2C^2/r$ 个参数，当 $r=16$ 时减少了 8 倍。
2. **引入非线性**：降维→ReLU→升维的结构比单层线性变换有更强的表达能力。
3. **学习通道间关系**：瓶颈迫使信息通过一个低维"瓶颈"，这迫使网络学到通道间的紧凑表示。

压缩比 $r$ 控制瓶颈的宽度：$r$ 越大参数越少，但表达能力也越弱。通常取 $r=16$。
```

### 3. Scale：加权

把学习到的权重 $s_c \in (0,1)$ 乘回原始特征图的对应通道：

```{math}
\tilde{x}_c = s_c \cdot x_c
```

这就是一个逐通道的缩放操作。权重接近 1 的通道被保留甚至增强，权重接近 0 的通道被抑制。

## SE-ResNet 集成

SE 模块通常插入到残差块中、残差连接之前：

```{literalinclude} code/se_basic_block.py
:language: python
:linenos:
:caption: SE-ResNet基础块实现
```

```{figure} ../../../_static/images/se-resnet.png
:width: 400px
:align: center

SE-ResNet：在残差块中插入SE模块
```


## 参数量与计算开销

SE 模块的参数量为：

```{math}
\text{Params} = \frac{C}{r} \times C + C \times \frac{C}{r} = \frac{2C^2}{r}
```

当 $C=256, r=16$ 时，参数数量为 $2 \times 256^2 / 16 = 8,192$——相对于 ResNet-50 的 2500 万参数可以忽略不计。计算开销增加也小于 1%。

```{admonition} SE-Net的关键贡献
:class: important

- 仅增加约 **1%** 计算量，提升 **1-2%** 准确率
- 即插即用：可集成到任何CNN架构中（ResNet、MobileNet、Inception等）
- 在 ImageNet 上，SE-ResNet-50 达到 **77.62%** Top-1 准确率，比基线提升 **+1.47%**
```

## PyTorch 实现

```{literalinclude} code/se_block.py
:language: python
:linenos:
:caption: SE模块的PyTorch实现
```

**代码要点**：

- `AdaptiveAvgPool2d((1, 1))` 实现 Squeeze：把任意尺寸的特征图压缩到 $1 \times 1$
- `Linear` 层的输入输出维度由压缩比 $r$ 控制
- Sigmoid 保证输出在 $(0,1)$ 范围内
- `x * self.scale` 实现逐通道加权

## 本章小结

- SE-Net 通过 **Squeeze → Excitation → Scale** 三步实现通道注意力
- 核心是让网络学习"每个通道的重要性权重"，然后据此重新校准特征
- 结构简单、计算开销小、即插即用

通道注意力解决了"什么特征重要"。但还有一个问题：**同一个特征，出现在画面的不同位置，意义完全不同。** "耳朵"在画面中央是猫，在角落是噪声。

{doc}`cbam`——走。不只选频道，还要调区域。

---

```{only} not latex

~~~{rubric} 参考文献
:heading-level: 2
~~~
```

~~~{bibliography}
:filter: docname in docnames
~~~
