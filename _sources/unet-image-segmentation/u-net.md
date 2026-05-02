(unet-arch)=
# U 形架构详解

## 直觉：U-Net 的整体结构

{ref}`unet-introduction` 我们提出了核心问题：**如何把缩小的特征图恢复回原始分辨率，同时不丢失空间细节？**

U-Net 的答案是一个对称的 U 形：

```{figure} ../../_static/images/u-net-architecture.png
:width: 640px
:align: center

U-Net 的 U 形对称架构（原论文 {cite}`ronneberger2015u`）
```

### 架构速览

从图中可以看出核心设计：左侧编码器逐步下采样（空间缩小、通道增加），右侧解码器逐步上采样（空间恢复、通道减少），底部是信息瓶颈，灰色箭头是**跳跃连接**。

几个关键数字：
- **输入 572×572×1** → **输出 388×388×2**（有效填充导致边界丢失 92 像素）
- **通道数变化**：1 → 64 → 128 → 256 → 512 → 1024 → 512 → ... → 64 → 2
- **每层跳跃连接拼接**：编码器特征（保留精确位置）与解码器特征（携带语义信息）在通道维度拼接

### 三部分的角色

| 部分 | 做什么 | 效果 |
|------|--------|------|
| **编码器（左半）** | 卷积+池化，逐步下采样 | 空间变小，通道变多，语义变强 |
| **瓶颈（底部）** | 最深的卷积，特征最抽象 | 感受野最大，知道"整体是什么" |
| **解码器（右半）** | 转置卷积+跳跃拼接，逐步上采样 | 空间变大，恢复细节 |

## 编码器路径：提取语义

编码器和 {doc}`../neural-network-basics/le-net` 前半部分几乎一样——重复的卷积块 + 最大池化。

```{admonition} 编码器的直觉
:class: note

每一步都在做"退一步看全局"：
- 3×3 卷积看局部模式（边缘、纹理）
- 2×2 池化取最显著特征，空间减半
- 通道数加倍，给更多"记忆空间"存储提炼出的信息

就像写摘要：先逐段阅读（卷积），再浓缩成要点（池化）。反复几次后，你就从具体的文字得到了文章主旨。
```

### 编码器中的感受野

{ref}`receptive-field` 中我们学了感受野的递推公式。U-Net 中每个编码器块的感受野：

| 层 | 累积感受野 | 看到什么 |
|----|-----------|----------|
| 第 1 层 | 3×3 | 边缘、角点、纹理 |
| 第 2 层 | 5×5 | 简单的形状组合 |
| 第 3 层 | 9×9 | 局部结构模式 |
| 第 4 层 | 17×17 | 较大的语义部件 |
| 底部 | 33×33 | 完整的语义对象 |

感受野逐层扩大，意味着一路向下，每个神经元看到的输入区域越来越大，学到的特征从"具体边缘"变成了"抽象语义"。

### 特征图尺寸变化公式

原始 U-Net 用有效填充，每步卷积后尺寸变化为：

```{math}
H_{\text{out}} = H_{\text{in}} - k + 1
```

其中 $k=3$ 是卷积核大小，所以每次卷积后宽高各减 2。池化后尺寸减半。

## 解码器路径：恢复空间

解码器是编码器的"镜像"——把缩小的特征图一步步放大回原始分辨率。核心操作有两个：

### 1. 上采样（Upsampling）

U-Net 使用**转置卷积（Transposed Convolution）** 进行上采样。直觉上，它和卷积做的事相反：

```{tikz} 转置卷积：2×2 → 4×4
\begin{tikzpicture}[scale=0.7]
  % 输入 2x2
  \draw[fill=blue!20, thick] (0,0) rectangle (1,1);
  \draw[fill=blue!20, thick] (1,0) rectangle (2,1);
  \draw[fill=blue!20, thick] (0,1) rectangle (1,2);
  \draw[fill=blue!20, thick] (1,1) rectangle (2,2);
  \node at (0.5,0.5) {$x_1$};
  \node at (1.5,0.5) {$x_2$};
  \node at (0.5,1.5) {$x_3$};
  \node at (1.5,1.5) {$x_4$};
  \node at (1, -0.6) {输入 2×2};

  % 箭头
  \draw[->, thick] (2.5, 1) -- (3.5, 1);
  \node at (3, 1.8) {插入零};

  % 中间 4x4（插零）
  \draw[fill=blue!5, thick] (4,0) rectangle (5,1);
  \draw[fill=blue!5, thick] (5,0) rectangle (6,1);
  \draw[fill=blue!5, thick] (6,0) rectangle (7,1);
  \draw[fill=blue!5, thick] (4,1) rectangle (5,2);
  \draw[fill=blue!5, thick] (5,1) rectangle (6,2);
  \draw[fill=blue!5, thick] (6,1) rectangle (7,2);
  \draw[fill=blue!5, thick] (4,2) rectangle (5,3);
  \draw[fill=blue!5, thick] (5,2) rectangle (6,3);
  \draw[fill=blue!5, thick] (6,2) rectangle (7,3);
  \node at (4.5,0.5) {$x_1$};
  \node at (5.5,1.5) {$x_2$};
  \node at (4.5,2.5) {$x_3$};
  \node at (5.5,2.5) {$x_4$};
  \node[font=\tiny, gray] at (4.5,1.5) {0};
  \node[font=\tiny, gray] at (5.5,0.5) {0};
  \node[font=\tiny, gray] at (6.5,0.5) {0};
  \node[font=\tiny, gray] at (5.5,1.5) {0};
  \node at (5.5, -0.6) {插零 4×4};

  % 箭头
  \draw[->, thick] (7.5, 1.5) -- (8.5, 1.5);
  \node at (8, 2.3) {3×3 卷积};

  % 输出 4x4
  \draw[fill=green!20, thick] (9,0) rectangle (10,1);
  \draw[fill=green!20, thick] (10,0) rectangle (11,1);
  \draw[fill=green!20, thick] (11,0) rectangle (12,1);
  \draw[fill=green!20, thick] (9,1) rectangle (10,2);
  \draw[fill=green!20, thick] (10,1) rectangle (11,2);
  \draw[fill=green!20, thick] (11,1) rectangle (12,2);
  \draw[fill=green!20, thick] (9,2) rectangle (10,3);
  \draw[fill=green!20, thick] (10,2) rectangle (11,3);
  \draw[fill=green!20, thick] (11,2) rectangle (12,3);
  \node at (10.5, -0.6) {输出 4×4};
\end{tikzpicture}
```

```{admonition} 转置卷积的直觉
:class: note

普通卷积：3×3 窗口滑过输入 → 输出更小
转置卷积：每个输入点"膨胀"成 2×2 区域 → 输出更大
```

转置卷积通过在每个输入元素之间插入零值，然后做标准卷积来实现尺寸加倍。它的参数也是**可学习的**，不像双线性插值是固定的。

(skip-connection)=
### 2. 跳跃连接（Skip Connection）

这是 U-Net 的灵魂。上采样后的特征图与**编码器对应层的特征图在通道维度上拼接**（concatenate）。

```{math}
F_{\text{concat}} = \text{concat}(F_{\text{encoder}}, F_{\text{decoder}}) \in \mathbb{R}^{H \times W \times (C_{\text{enc}} + C_{\text{dec}})}
```

**为什么拼接而不是相加？**

FCN 用相加，U-Net 用拼接。拼接**保留了编码器特征的全部信息**（不压缩），让解码器可以"看到"原始细节。

```{admonition} 跳跃连接的三个作用
:class: tip

1. **保留空间信息**：编码器浅层知道"边缘在第 35 行"，解码器需要这个信息
2. **改善梯度流动**：梯度可以"抄近道"直接传到浅层，缓解梯度消失
3. **多尺度特征融合**：不同感受野的特征同时被利用
```

### 跳跃连接的梯度分析

跳跃连接对训练的最大贡献是**解决了梯度消失问题**。{ref}`gradient-vanishing` 中我们详细讨论过梯度消失的成因——反向传播时梯度经过多层连乘会指数级衰减，导致浅层无法学习。

要理解跳跃连接为什么能缓解这个问题，需要先理解 {ref}`jacobian-matrix` 的概念（详见 {doc}`../math-fundamentals/back-propagation`）。核心洞察是：深层网络的梯度计算等于**多层 Jacobian 矩阵的连乘**。

#### 没有跳跃连接时：Jacobian 连乘

链式法则告诉我们，损失 $L$ 对第 $l$ 层权重 $W_l$ 的梯度相当于**每一层 Jacobian 的连乘**：

```{math}
\frac{\partial L}{\partial W_l} = \frac{\partial L}{\partial \hat{y}} \cdot
\underbrace{\frac{\partial h_{L}}{\partial h_{L-1}} \cdot
\frac{\partial h_{L-1}}{\partial h_{L-2}} \cdot \dots \cdot
\frac{\partial h_{l+1}}{\partial h_{l}}}_{L-l \text{ 个 Jacobian 相乘}}
```

每乘一个 Jacobian 都可能缩小或放大梯度。如果每层的"放大倍数"都小于 1，$L-l$ 个 Jacobian 连乘后梯度就会指数级衰减——梯度消失。

#### 有了跳跃连接：梯度抄近道

跳跃连接在反向传播中创建了一条**直达路径**，绕过了中间的深度：

```{math}
\frac{\partial L}{\partial W_l} = \underbrace{\text{主路径（$L-l$ 个 Jacobian 连乘）}}_{\text{可能消失}} + \underbrace{\text{跳跃路径（仅 1-2 个 Jacobian）}}_{\text{几乎不衰减}}
```

第二条路径的 Jacobian 链极短——编码器第 2 层的梯度可以通过跳跃连接直接传到解码器第 2 层，中间只经过 1-2 个变换，而不是从底部绕上来的 8-10 个。

```{admonition} 数值对比：64 倍的差距
:class: note

假设每层 Jacobian 的放大倍数 = 0.8：

原始路径（穿过 20 层）：$0.8^{20} \approx 0.01$（衰减了 99%）

跳跃路径（只穿过 2 层）：$0.8^{2} \approx 0.64$（只衰减了 36%）

差距：$0.01$ vs $0.64$，整整 **64 倍**。有了跳跃连接，浅层收到的梯度信号强了几十倍。
```

这就是跳跃连接被称为"梯度高速公路"的原因——它为梯度提供了**绕过深度、直达浅层**的捷径。这个思路与 ResNet {cite}`he2016deep` 的残差连接一脉相承：都是通过短路连接为梯度创造高速公路。{doc}`../cnn-ablation-study/experiment-design` 中的消融实验也可以验证：去掉跳跃连接后，深层网络的浅层几乎学不到东西。

## 输出层设计

输出层用 1×1 卷积把通道数映射到类别数：

```{math}
\text{输出} = \text{Conv2D}(C_{\text{in}}, C_{\text{out}}, \text{kernel\_size}=1)
```

对于二分类任务（如细胞 vs 背景），$C_{\text{out}} = 2$，后接 softmax。每个像素的 2 个值表示"属于背景的概率"和"属于细胞的概率"。

## 为什么 U 形架构这么成功？

回顾 {ref}`unet-introduction` 中讨论的核心矛盾，现在看 U-Net 如何一一解决：

| 矛盾 | 编码器的做法 | 解码器的做法 | 跳跃连接的作用 |
|------|-------------|-------------|---------------|
| 需要语义理解 | 下采样扩大感受野 | - | 深层特征传到解码器 |
| 需要空间精度 | 浅层保留边缘信息 | - | 浅层特征直接"抄近道" |
| 需要端到端训练 | - | 可学习的上采样 | 梯度直达浅层 |

三个组件缺一不可。没有解码器，输出分辨率不够；没有跳跃连接，恢复的细节不够；没有编码器，没有语义理解。

## 嵌入核心代码：核心组件

下面先看 U-Net 的四个核心组件，完整的 U-Net 组装会在 {doc}`core-impl` 中展示。

### 双卷积块（核心构建单元）

```python
class DoubleConv(nn.Module):
    """U-Net 中的基本构建单元：两个 3×3 卷积
    
    参数量计算（输入 C_in，输出 C_out）：
    第一个卷积: C_in × 3 × 3 × C_out
    第二个卷积: C_out × 3 × 3 × C_out
    总计: 9 × (C_in × C_out + C_out²)
    例如 64→64: 9×(64×64+64²) = 73,728
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.conv(x)
```

**为什么两个 3×3 卷积堆叠？**

一个 3×3 卷积感受野是 3×3。堆叠两个 3×3，感受野变成 5×5（{ref}`receptive-field` 的递推公式）。但参数量比一个 5×5 少——$2 \times 9 \times C^2$ vs $25 \times C^2$，减少了 28%。如果堆叠三个 3×3，感受野达到 7×7，参数比一个 7×7 减少 45%。这个设计思路最早来自 VGG 网络 {cite}`simonyan2014very`——用小卷积核的堆叠代替大卷积核，在保证感受野的同时减少参数。

每个卷积后接一个 BatchNorm {cite}`ioffe2015batch`，作用是稳定训练：把激活值拉回到零均值单位方差，防止深层网络的激活值剧烈漂移。

### 下采样模块

```python
class DownSample(nn.Module):
    """编码器下采样：MaxPool → DoubleConv

    输入: (B, C_in,  H,   W )
    输出: (B, C_out, H/2, W/2)
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )
    
    def forward(self, x):
        return self.maxpool_conv(x)
```

### 上采样模块（带跳跃连接）

```python
class UpSample(nn.Module):
    """解码器上采样：转置卷积 → 跳跃拼接 → DoubleConv
    
    输入 x1: (B, C_dec,  H,   W )
    输入 x2: (B, C_enc, 2H, 2W )
    输出:   (B, C_out, 2H, 2W )
    
    转置卷积把 C_dec 减半、尺寸加倍；
    拼接编码器特征后，DoubleConv 缩回 C_out
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(
            in_channels, in_channels // 2, kernel_size=2, stride=2
        )
        self.conv = DoubleConv(in_channels, out_channels)
    
    def forward(self, x1, x2):
        x1 = self.up(x1)
        # 原始 U-Net 用有效填充，尺寸可能不匹配，需要填充对齐
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # 跳跃连接：通道拼接
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)
```

### 输出层

```python
class OutConv(nn.Module):
    """输出层：1×1 卷积
    
    输入: (B, C_in,  H, W)
    输出: (B, n_classes, H, W)
    
    1×1 卷积不改变空间尺寸，只改变通道数
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    
    def forward(self, x):
        return self.conv(x)
```

这些组件的完整组装见 {doc}`core-impl`。下一节 {doc}`loss-func` 我们先来谈谈一个问题：分割任务的损失函数为什么和分类任务不一样？

---

## 参考文献

```{bibliography}
:filter: docname in docnames
```
