(efficientnet)=
# EfficientNet：每一分钱，花在刀刃上

{doc}`mobilenet` 教会了我们一件事：**极致省钱，也能爬。** 深度可分离卷积把参数量砍到 1/9，一个人一把冰镐就能出发。

但大多数攀登者的处境不在两个极端——既不是 Google 的无限预算，也不是嵌入式芯片的极限抠门。你有一笔预算：几个 GPU，几天的训练时间，几百 MB 的模型限制。**怎么把这笔钱花在刀刃上？**

**关键问题**：深度、宽度、分辨率——三个旋钮，应该按什么比例一起拧？

## 历史背景：三个旋钮，各自为政

在 EfficientNet 之前，人们已经知道扩大模型可以提升性能：

- **更深**（ResNet-152 → ResNet-1000）：更多层 = 更强的表达能力
- **更宽**（MobileNet 的 $\alpha$ 参数）：更多通道 = 更丰富的特征
- **更大分辨率**（224×224 → 380×380 → 600×600）：更精细的输入 = 更细节的信息

但问题是——**大家都是凭感觉调的。** 有人加深度，有人加宽度，有人加大分辨率。没有一个系统性的原则告诉你在给定预算下，三者怎么组合最优。

2019 年，Mingxing Tan 和 Google 团队提出了 **EfficientNet** {cite}`tan2019efficientnet`。核心洞察只有一个，但它统一了三年的零散经验：

**深度、宽度、分辨率不是独立的。它们应该按一个复合比例同时缩放。**

## 复合缩放：为什么一起拧才对？

```{admonition} 直觉
:class: tip

想象你要建一个更大的登山营地：

- **更宽** = 多雇人（更细的分工：有人专门探路，有人专门做饭）
- **更深** = 建更多补给站（更深的补给线：大本营 → C1 → C2 → C3）
- **更大分辨率** = 更精细的地图（能看到更小的冰裂缝）

如果你只加人不建补给站——人挤在低海拔，到了高海拔没人能送物资。
如果你只建补给站不加人——空有路线，人手不够。
如果你只换精细地图不加人也不建站——看到了冰裂缝但没人能处理。

**三者必须按比例一起加。**
```

数学上，EfficientNet 用了一个复合系数 $\phi$：

$$
\begin{aligned}
d &= \alpha^\phi \quad &\text{（深度：补给站数量）} \\
w &= \beta^\phi \quad &\text{（宽度：每站人数）} \\
r &= \gamma^\phi \quad &\text{（分辨率：地图精细度）}
\end{aligned}
$$

其中 $\alpha \cdot \beta^2 \cdot \gamma^2 \approx 2$——意味着每增大 $\phi$ 一个单位，总计算量翻倍。而 $\alpha, \beta, \gamma$ 的具体值是通过在**小模型上的网格搜索**找到的：$\alpha=1.2, \beta=1.1, \gamma=1.15$。

```{admonition} 为什么比例是搜索出来的，不是推导出来的？
:class: note

公平地说，$\alpha=1.2, \beta=1.1, \gamma=1.15$ 这三个数字没有任何理论必然性。它们是在一个 8 卡 GPU 上对小基线模型做网格搜索找到的最优组合。

但这不是 EfficientNet 的"弱点"——这恰恰是它的务实之处。就像登山者不会从理论推导出"氧气瓶应该带 3.2 升而不是 3.1 升"——你在低海拔试了不同组合，选最好的，然后在大山上按比例放大。
```

## 架构家族：从 B0 到 B7

基于复合缩放，EfficientNet 推出了一个模型家族。基线模型 EfficientNet-B0 是通过 **NAS（神经架构搜索）** 找到的，然后通过增大 $\phi$ 得到 B1 到 B7：

| 模型 | $\phi$ | 参数 | Top-1 准确率 | 相比 ResNet-50 |
|------|--------|------|-------------|---------------|
| EfficientNet-B0 | 0 | 5.3M | 77.1% | 4.5× 更小，同准确率 |
| EfficientNet-B1 | 1 | 7.8M | 79.1% | — |
| EfficientNet-B3 | 3 | 12M | 81.6% | — |
| EfficientNet-B5 | 5 | 30M | 83.6% | — |
| EfficientNet-B7 | 7 | 66M | **84.3%** | 当时 SOTA，比 ResNet-152 还高 |

```{admonition} 登山者的选择
:class: tip

| 你的预算 | 建议模型 | 比喻 |
|---------|---------|------|
| 一台手机 | B0 (5.3M) | 一个人、一天、基本装备 |
| 一台 GPU 服务器 | B3 (12M) | 小团队、一周 |
| 多卡集群 | B5 (30M) | 中型探险队 |
| 土豪 | B7 (66M) | 直升飞机空投补给 |
```

## MobileNet vs EfficientNet：省钱的两条路

| | MobileNet | EfficientNet |
|------|-----------|-------------|
| 回答的问题 | 几乎没钱，怎么爬？ | 有一笔预算，怎么花？ |
| 核心操作 | **拆开**卷积（深度可分离） | **按比例缩放**三个维度 |
| 哲学 | 极致瘦身——先砍到骨头，再往上加 | 系统优化——不瘦身，而是聪明地胖 |
| 就像登山 | 一个人、一把冰镐 | 有限团队、精细分工、每分钱花在刀刃上 |

```{admonition} 二者并不矛盾
:class: note

事实上，EfficientNet 的基线架构本身就用到了深度可分离卷积——MobileNet 的"拆开"思想。EfficientNet 的创新不在于"怎么省"，而在于 **"省完之后，多出来的预算怎么分配"**。

MobileNet 回答的是"怎么省到极致"。EfficientNet 回答的是"省出了预算之后，往哪个方向花最值"。
```

## 代码实践

```python
import torch
import torch.nn as nn
import math


class MBConvBlock(nn.Module):
    """Mobile Inverted Bottleneck Conv Block（EfficientNet 的基本构建块）
    
    这是 MobileNetV2 的倒置残差块 + SE 注意力（{doc}`se-net`）：
    1. 1×1 扩展（升维——先给更多"呼吸空间"）
    2. Depthwise 3×3（空间混合）
    3. SE 注意力（自适应通道加权）
    4. 1×1 压缩（降维——回到原来的通道数）
    5. 残差连接（如果 shape 匹配）
    
    为什么叫"倒置"？因为传统 bottleneck 是先压缩再扩展，这里反过来：先扩展再压缩。
    """
    def __init__(self, in_channels, out_channels, kernel_size, 
                 stride, expand_ratio, se_ratio=0.25):
        super().__init__()
        
        # 是否使用残差连接
        self.use_residual = (stride == 1 and in_channels == out_channels)
        
        # 扩展后的通道数
        expanded_channels = in_channels * expand_ratio
        
        # 1. 1×1 扩展（如果有扩展）
        self.expand_conv = nn.Sequential(
            nn.Conv2d(in_channels, expanded_channels, 1, bias=False),
            nn.BatchNorm2d(expanded_channels),
            nn.SiLU()  # Swish 激活——比 ReLU 更平滑，EfficientNet 的默认选择
        ) if expand_ratio != 1 else nn.Identity()
        
        # 2. Depthwise 3×3 卷积——{ref}`mobilenet` 的核心思想
        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(expanded_channels, expanded_channels, kernel_size,
                     stride, padding=kernel_size//2, 
                     groups=expanded_channels, bias=False),
            nn.BatchNorm2d(expanded_channels),
            nn.SiLU()
        )
        
        # 3. SE 注意力模块——{doc}`se-net`
        se_channels = max(1, int(in_channels * se_ratio))
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(expanded_channels, se_channels, 1),
            nn.SiLU(),
            nn.Conv2d(se_channels, expanded_channels, 1),
            nn.Sigmoid()
        )
        
        # 4. 1×1 压缩
        self.project_conv = nn.Sequential(
            nn.Conv2d(expanded_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
    
    def forward(self, x):
        residual = x
        x = self.expand_conv(x)
        x = self.depthwise_conv(x)
        x = x * self.se(x)  # SE 注意力重标定
        x = self.project_conv(x)
        
        if self.use_residual:
            x = x + residual  # 残差连接——{ref}`res-net` 的补给站
        return x


def efficientnet_b0(num_classes=1000):
    """EfficientNet-B0 的简化实现
    
    架构原理：
    - 基于 MBConv 块的倒置残差结构
    - 复合缩放：如果要做 B1-B7，只需按比例调整深度/宽度/分辨率
    
    参数量：约 5.3M
    Top-1 准确率：77.1%（ImageNet）
    """
    return nn.Sequential(
        # Stem: 标准卷积，快速降采样
        nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(32),
        nn.SiLU(),
        
        # Stage 1: 112×112
        MBConvBlock(32, 16, 3, stride=1, expand_ratio=1),
        
        # Stage 2: 112→56
        MBConvBlock(16, 24, 3, stride=2, expand_ratio=6),
        MBConvBlock(24, 24, 3, stride=1, expand_ratio=6),
        
        # Stage 3: 56→28
        MBConvBlock(24, 40, 5, stride=2, expand_ratio=6),
        MBConvBlock(40, 40, 5, stride=1, expand_ratio=6),
        
        # Stage 4: 28→14
        MBConvBlock(40, 80, 3, stride=2, expand_ratio=6),
        MBConvBlock(80, 80, 3, stride=1, expand_ratio=6),
        MBConvBlock(80, 80, 3, stride=1, expand_ratio=6),
        
        # Stage 5: 14×14
        MBConvBlock(80, 112, 5, stride=1, expand_ratio=6),
        MBConvBlock(112, 112, 5, stride=1, expand_ratio=6),
        MBConvBlock(112, 112, 5, stride=1, expand_ratio=6),
        
        # Stage 6: 14→7
        MBConvBlock(112, 192, 5, stride=2, expand_ratio=6),
        MBConvBlock(192, 192, 5, stride=1, expand_ratio=6),
        MBConvBlock(192, 192, 5, stride=1, expand_ratio=6),
        MBConvBlock(192, 192, 5, stride=1, expand_ratio=6),
        
        # Stage 7: 7×7
        MBConvBlock(192, 320, 3, stride=1, expand_ratio=6),
        
        # Head
        nn.Conv2d(320, 1280, 1, bias=False),
        nn.BatchNorm2d(1280),
        nn.SiLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Dropout(0.2),
        nn.Linear(1280, num_classes),
    )


# 参数量验证
model = efficientnet_b0()
params = sum(p.numel() for p in model.parameters())
print(f"EfficientNet-B0: {params:,} 参数")
```

---

## 总结

| 维度 | 内容 |
|------|------|
| **核心洞察** | 深度、宽度、分辨率不是独立的——复合缩放，按比例一起拧 |
| **哲学** | 在给定计算预算下，将每一分钱系统性地分配到三个维度 |
| **三系数** | $\alpha=1.2$（深度）、$\beta=1.1$（宽度）、$\gamma=1.15$（分辨率）——小模型网格搜索得出 |
| **历史意义** | 统一了 ResNet（深度派）、MobileNet（宽度派）、高分辨率方法的经验 |

**登山者笔记**：MobileNet 告诉你一个人怎么爬。EfficientNet 告诉你有团队怎么爬。但底层道理是一样的——**不是"更多就一定更好"，而是"更多要加在对的地方"。**

---

## 下一步

两座山都翻越了。从 1989 年 LeNet 的练习峰首登，到 2019 年 EfficientNet 的复合缩放——三十年架构演化，一言以蔽之：

**好的设计，是把山的形状刻进装备里。**

但登山者不会在山顶停留。八年的路走完了——回头看，每一件装备都在回答同一个问题。

{doc}`the-end`——坐下来，写登山笔记。

---

[^tan2019]: Mingxing Tan, Quoc V. Le, "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks", ICML 2019.

```{only} not latex

~~~{rubric} 参考文献
:heading-level: 2
~~~
```

~~~{bibliography}
:filter: docname in docnames
~~~
