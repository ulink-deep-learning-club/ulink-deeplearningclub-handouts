(alexnet)=
# AlexNet：真正的未登峰，首登！

2012 年，Alex Krizhevsky、Ilya Sutskever 和 Geoffrey Hinton 提交了一个叫 AlexNet 的模型 {cite}`krizhevsky2012imagenet`。它的架构并不神秘——本质上就是一个**放大的 LeNet**：

```
INPUT → CONV → POOL → CONV → POOL → CONV → CONV → CONV → POOL → FC → FC → FC → OUTPUT
```

和 LeNet 的区别是什么呢？更深（8 层 vs 5 层）、更宽（最多 384 个通道 vs 16）、更大（60M 参数 vs 61K）。但真正让所有人震惊的不是架构本身，而是**它把错误率从 28.2% 砸到了 15.3%——几乎砍半。**

那一刻，整个计算机视觉领域听到了炮声。深度学习不是玩具。它能爬真山。

```{admonition} 为什么叫"炮火"？
:class: note

2012 年之前，ImageNet 竞赛的进步是每年 1-2% 的缓慢爬升，主要由传统 CV 方法的渐进改进驱动。AlexNet 一举将错误率降低了 **13 个百分点**——相当于过去十年的进步总和被一个模型超越了。

这不是改进，是**革命**。
```

## 首登的秘密：不只是更大的 LeNet

把 LeNet 放大就能赢？没那么简单。AlexNet 之所以能首登，靠的是四个关键设计——每一个都在回答"更大的山上会遇到什么新问题"：

| 设计 | 解决的问题 | 就像登山 |
|------|-----------|---------|
| **ReLU 激活** | Sigmoid/Tanh 在深层网络中梯度消失严重 | 高海拔需要高效呼吸——ReLU 不"憋气" |
| **Dropout** | 60M 参数在 140 万张图上极易过拟合 | 每次攀登随机"关掉"一部分神经元——逼它们学会独立 |
| **GPU 并行** | 一张 GPU 装不下 60M 参数的训练 | 两个人抬装备——模型切成两半，两张 GPU 各算各的 |
| **数据增强** | 140 万张图不够 60M 参数学 | 同一块岩壁从不同角度看——随机裁剪、翻转、颜色抖动 |

```{note}
**LeNet 用的是 Tanh，AlexNet 用的是 ReLU。** 这不是随意的替换。{ref}`activation-functions` 中我们讨论过：Tanh 的饱和区让梯度趋近于零，网络越深问题越严重。ImageNet 需要 8 层——Tanh 根本喘不过气。

ReLU 简单粗暴：$f(x) = \max(0, x)$。正区间梯度恒为 1，不会衰减。就像登山者在高海拔换上更高效的呼吸方式。
```

## 架构详解

### 金字塔视角：数据流的维度变化

```{tikz} AlexNet 金字塔架构：从图像到1,000类
\begin{tikzpicture}[scale=0.75]
    % 输入层 - 最宽
    \fill[green!20] (-5.5, 10) rectangle (5.5, 11);
    \draw[thick] (-5.5, 10) rectangle (5.5, 11);
    \node at (0, 10.5) {\small 输入：224×224×3};
    \node[right] at (7, 10.5) {\scriptsize 彩色图像};
    
    % CONV1
    \fill[blue!20] (-4.8, 8.5) rectangle (4.8, 9.5);
    \draw[thick] (-4.8, 8.5) rectangle (4.8, 9.5);
    \node at (0, 9) {\small CONV1：55×55×96};
    \node[right] at (7, 9) {\scriptsize 11×11, s=4, ReLU};
    \draw[->, thick] (0, 10) -- (0, 9.5);
    
    % POOL1
    \fill[cyan!20] (-4.2, 7) rectangle (4.2, 8);
    \draw[thick] (-4.2, 7) rectangle (4.2, 8);
    \node at (0, 7.5) {\small POOL1：27×27×96};
    \node[right] at (7, 7.5) {\scriptsize 3×3, s=2};
    \draw[->, thick] (0, 8.5) -- (0, 8);
    
    % CONV2
    \fill[blue!20] (-3.8, 5.5) rectangle (3.8, 6.5);
    \draw[thick] (-3.8, 5.5) rectangle (3.8, 6.5);
    \node at (0, 6) {\small CONV2：27×27×256};
    \node[right] at (7, 6) {\scriptsize 5×5, p=2, ReLU};
    \draw[->, thick] (0, 7) -- (0, 6.5);
    
    % POOL2
    \fill[cyan!20] (-3.4, 4) rectangle (3.4, 5);
    \draw[thick] (-3.4, 4) rectangle (3.4, 5);
    \node at (0, 4.5) {\small POOL2：13×13×256};
    \node[right] at (7, 4.5) {\scriptsize 3×3, s=2};
    \draw[->, thick] (0, 5.5) -- (0, 5);
    
    % CONV3-5（三层连续3×3卷积）
    \fill[blue!20] (-3.2, 2.5) rectangle (3.2, 3.5);
    \draw[thick] (-3.2, 2.5) rectangle (3.2, 3.5);
    \node at (0, 3) {\small CONV3-5：13×13, 384→256};
    \node[right] at (7, 3) {\scriptsize 3×3×3, p=1, ReLU};
    \draw[->, thick] (0, 4) -- (0, 3.5);
    
    % POOL3
    \fill[cyan!20] (-2.8, 1) rectangle (2.8, 2);
    \draw[thick] (-2.8, 1) rectangle (2.8, 2);
    \node at (0, 1.5) {\small POOL3：6×6×256};
    \node[right] at (7, 1.5) {\scriptsize 3×3, s=2};
    \draw[->, thick] (0, 2.5) -- (0, 2);
    
    % 全连接层
    \fill[red!20] (-1.6, -0.5) rectangle (1.6, 0.5);
    \draw[thick] (-1.6, -0.5) rectangle (1.6, 0.5);
    \node at (0, 0) {\small FC};
    \node[right] at (7, 0) {\scriptsize 4096→4096→1000};
    \draw[->, thick] (0, 1) -- (0, 0.5);
    
    % 标注维度变化
    \draw[<->, gray, thick] (-7, 10) -- (-7, 11);
    \node[left, gray, font=\scriptsize] at (-7, 10.5) {空间维度};
    
    \draw[<->, red, thick] (9, -0.5) -- (9, 11);
    \node[right, red, font=\scriptsize, align=center] at (9, 5.25) {空间↓\\语义↑};
\end{tikzpicture}
```

**金字塔的核心洞察**：

- **空间维度递减**：224×224 → 6×6（缩小约1,400倍——比LeNet的64倍剧烈得多）
- **特征通道递增**：3 → 96 → 256 → 384 → 256（语义丰富度先升后调）
- **参数量分布**：卷积层约3.7M（占6%），全连接层约57M（占94%）——与LeNet的97%全连接占比一脉相承
- **比LeNet更深更宽**：8层 vs 5层，最多384通道 vs 16通道，对应更大的山需要更大的容量

```{admonition} 和 LeNet 的对比
:class: tip

| 对比维度 | LeNet-5 | AlexNet |
|---------|---------|---------|
| 输入 | 32×32 灰度 | 224×224 彩色 |
| 层数 | 5 | 8 |
| 参数 | 61K | 60M (**×1,000**) |
| 激活函数 | Tanh | **ReLU** |
| 正则化 | 无 | **Dropout** |
| 训练硬件 | CPU | **2× GPU** |
| 首层卷积核 | 5×5 | **11×11**（更大的山需要更广的视野） |
```

## ReLU：高海拔的呼吸方式

AlexNet 是第一个大规模使用 ReLU 的深度网络。为什么这很重要？

{doc}`../practice-peak/cnn-basics` 中的 LeNet 用的是 Tanh。Tanh 的问题是：输入稍微大一点，输出就饱和在 ±1，梯度趋近于零。8 层 Tanh，梯度连乘 8 次——前面的层几乎收不到信号。

ReLU 换了一种呼吸方式：

$$
\text{ReLU}(x) = \max(0, x)
$$

**ReLU 的直觉**：在正区间，梯度永远是 1——不会衰减。在负区间，梯度是 0——神经元"关掉"。这种"要么全开要么全关"的方式，在深层网络中意外地高效。

## 首登之后：暴露的新问题

AlexNet 证明了 ImageNet 可以爬。但成功的攀登者会立刻看到新的障碍：

1. **11×11 的卷积核太大**——一次看太多，浪费参数且容易过拟合（{doc}`inception` 和 VGG 会回答：用更小的核堆叠）
2. **只有一条前向路径**——如果某一层"学废了"，整条路就断了（{doc}`res-net` 会回答：加跳跃连接）
3. **对所有特征一视同仁**——无论这个通道是"狗耳朵"还是"背景噪声"，权重一样（{doc}`se-net` 会回答：动态加权）

**首登永远不是终点。它只是告诉你：这座山更高，新问题更多。**

---

## 代码实践

```python
import torch
import torch.nn as nn

class AlexNet(nn.Module):
    """AlexNet 的 PyTorch 实现
    
    架构：5层卷积 + 3层全连接
    关键设计：ReLU（高海拔呼吸）、Dropout（防止过拟合）
    
    参数量：约 60M（LeNet 的 1,000 倍）
    """
    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        
        # 特征提取部分（5层卷积）
        self.features = nn.Sequential(
            # CONV1: 11×11 大核——ImageNet 需要更大的初始感受野
            # 输出: (96, 55, 55)
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),  # {ref}`activation-functions`——高海拔呼吸方式
            nn.MaxPool2d(kernel_size=3, stride=2),  # 降采样
            
            # CONV2: 5×5 核——缩小感受野，提取更精细特征
            # 输出: (256, 27, 27)
            nn.Conv2d(96, 256, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            # CONV3-CONV5: 连续3×3卷积——小核堆叠 = 感受野扩大 + 参数更少
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        
        # 分类器部分（3层全连接）
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),  # 每次随机"关掉"一半神经元——逼它们独立学习
            nn.Linear(256 * 6 * 6, 4096),  # 6×6 来自输入 224×224 经过5层卷积+3次池化
            nn.ReLU(inplace=True),
            
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            
            nn.Linear(4096, num_classes),  # ImageNet: 1,000 类
        )
    
    def forward(self, x):
        """
        Args:
            x: 输入图像 (batch_size, 3, 224, 224)
        Returns:
            logits: (batch_size, num_classes)
        """
        x = self.features(x)       # 卷积特征提取
        x = torch.flatten(x, 1)    # 展平 → 全连接层输入
        x = self.classifier(x)     # 分类
        return x
```

```python
# 参数量计算
model = AlexNet(num_classes=1000)
total_params = sum(p.numel() for p in model.parameters())
print(f"AlexNet 总参数量: {total_params:,}")  # 约 60,965,000
print(f"LeNet-5 总参数量: 61,706")
print(f"倍数: {total_params / 61706:.0f}×")  # 约 988×
```

---

## 总结

| 维度 | 内容 |
|------|------|
| **历史意义** | 2012年 ImageNet 首登——错误率从 28.2% → 15.3%，深度学习革命的第一声炮响 |
| **核心创新** | ReLU（高海拔呼吸）+ Dropout（独立训练）+ GPU 并行 + 数据增强 |
| **参数量** | 60M——LeNet 的 1,000 倍。但爬的是一座大 1,000 倍的山 |
| **暴露的问题** | 大卷积核浪费参数、无跳跃连接、对所有特征一视同仁 |

---

首登成功的那一刻，欢呼声还没落地，真正的攀登者已经在看下一段岩壁了。

## 下一步

AlexNet 证明了 ImageNet 能爬。但它用 11×11 的大卷积核一次性看太多——浪费参数，且只能看一个尺度。

**山比想象的更复杂。不同的岩壁需要不同的工具。**

{doc}`inception`——走。

---

```{only} not latex

~~~{rubric} 参考文献
:heading-level: 2
~~~
```

~~~{bibliography}
:filter: docname in docnames
~~~
