(arch-efficiency)=
# 操控效率：如何在效果与速度间权衡

前面的章节都在讲"如何让模型更好"。但现实是：你的模型需要在具体的硬件上跑。如果训练一个 epoch 要三天，推理一张图要一分钟，再好的效果也无法落地。

效率优化的核心矛盾：**如何在不显著牺牲效果的前提下，大幅降低计算量？**

## 帕累托边界：效果的物理上限

在讨论效率优化之前，需要先理解一个基本概念：**没有免费的午餐**。

对于给定的计算预算，存在一个**帕累托边界（Pareto Frontier）**——在这个边界上，任何效果的提升都必须以效率为代价，任何效率的提升都必须以效果为代价。

```{tikz}
\begin{tikzpicture}[
    scale=0.9,
    font=\small,
    >={Stealth}
]

% 坐标轴（X轴反转：左=高效率）
\draw[->, thick] (0,0) -- (11,0) node[right] {计算量};
\draw[->, thick] (0,0) -- (0,7.5) node[above] {准确率};

% -------- 帕累托边界（对数形状，左上到右下）--------
\draw[very thick, blue!80!black, domain=0.5:10, smooth, samples=80]
    plot (\x, {1.5 + 2.2*ln(\x + 0.3)});

% 边界标签（放在曲线右侧末端）
\node[blue!80!black, right] at (10, {1.5 + 2.2*ln(10.3)}) {帕累托边界};

% -------- 绿色三角形（最佳折中，左上区域）--------
\fill[green!70!black] (2, {1.5 + 2.2*ln(2.3)})
    ++(-0.2,-0.18) -- ++(0.4,0) -- ++(-0.2,0.35) -- cycle;
\node[green!70!black, above left] at (1.8, {1.5 + 2.2*ln(2.3) + 0.3})
    {最佳折中};

% -------- 边界上的模型 --------
% 轻量模型（左下）
\fill[green!60!black] (1, {1.5 + 2.2*ln(1.3)}) circle (0.12);
\node[left] at (0.8, {1.5 + 2.2*ln(1.3)}) {轻量模型};

% 中等模型（中间）
\fill[orange!80!black] (4, {1.5 + 2.2*ln(4.3)}) circle (0.12);
\node[below right] at (4.2, {1.5 + 2.2*ln(4.3)}) {中等模型};

% 大模型（右上）
\fill[red!70!black] (8.5, {1.5 + 2.2*ln(8.8)}) circle (0.12);
\node[above left] at (8.3, {1.5 + 2.2*ln(8.8)}) {大模型};

% -------- 次优点（边界下方）--------
\fill[gray] (5, 3.5) circle (0.1);
\node[gray, below] at (5, 3.3) {次优模型};

% -------- 理想点（边界上方，左上）--------
\fill[green!40!black] (1.5, 6.5) circle (0.15);
\node[green!70!black, right] at (1.8, 6.8) {理想：又快又好};
\node[green!60!black, right] at (1.8, 6.3) {（当前不可达）};

% -------- 边界上移（创新）--------
\draw[dashed, purple, domain=0.5:10, smooth, samples=80]
    plot (\x, {2.3 + 2.2*ln(\x + 0.3)});

\node[purple, right] at (10, {2.3 + 2.2*ln(10.3)}) {架构创新};
\draw[->, purple, thick] (10.2, {1.5 + 2.2*ln(10.3) + 0.15})
    -- (10.2, {2.3 + 2.2*ln(10.3) - 0.15});

\end{tikzpicture}
```

**帕累托最优**的意思是：如果一个模型在帕累托边界上，你没法让它在不牺牲效果的前提下变得更快，也没法在不牺牲效率的前提下让它更好。

而**架构改造的意义**就是**推动帕累托边界向外移动**——不是在同一曲线上往返，而是在相同计算量下达到更好效果，或在相同效果下减少更多计算量。

这解释了为什么 Inception、ResNet、MobileNet 被认为是里程碑级别的贡献：它们不只是在已有曲线上滑动，而是**重新定义了什么算"最优"**。

## 效率问题从哪来

普通卷积的参数量和计算量：

一个 3×3 卷积（输入 $C_{in}$ 通道，输出 $C_{out}$ 通道，空间 $H \times W$）：

- 参数量：$3 \times 3 \times C_{in} \times C_{out}$
- 计算量：$3 \times 3 \times C_{in} \times C_{out} \times H \times W$

**关键观察**：空间和通道混在一起计算。能不能分拆？

## 策略一：深度可分离卷积

深度可分离卷积（Depthwise Separable Convolution，MobileNet 核心{cite}`howard2017mobilenets`）的洞察：**把空间和通道分开处理**。

```{tikz}
\begin{tikzpicture}[scale=0.65]
    \tikzset{op/.style={draw, minimum width=2.5cm, minimum height=0.8cm, align=center, fill=blue!10}}
    \tikzset{result/.style={draw, minimum width=1.8cm, minimum height=0.8cm, align=center, fill=green!10}}
    
    % 普通卷积
    \node at (-3, 6) {\textbf{普通卷积}};
    \node[op] (normal_in) at (-3, 4) {输入 $H \times W \times C_{in}$};
    \node[op] (normal_conv) at (-3, 2) {$K \times K \times C_{in} \times C_{out}$};
    \node[result] (normal_out) at (-3, 0) {输出 $H \times W \times C_{out}$};
    
    \draw[->] (normal_in) -- (normal_conv);
    \draw[->] (normal_conv) -- (normal_out);
    
    \node[left=0.3cm] at (normal_conv.west) {\small 空间和通道混在一起};
    
    % 分离卷积
    \node at (6, 6) {\textbf{深度可分离卷积}};
    \node[op] (dw_in) at (6, 4) {输入 $H \times W \times C_{in}$};
    \node[op] (dw_conv) at (6, 2) {Depthwise: $K \times K \times C_{in}$};
    \node[op] (pw_conv) at (6, 0) {Pointwise: $1 \times 1 \times C_{in} \times C_{out}$};
    \node[result] (dw_out) at (6, -2) {输出 $H \times W \times C_{out}$};
    
    \draw[->] (dw_in) -- (dw_conv);
    \draw[->] (dw_conv) -- (pw_conv);
    \draw[->] (pw_conv) -- (dw_out);
    
    \node[right=0.3cm] at (dw_conv.east) {\small 仅空间, 每通道独立};
    \node[right=0.3cm] at (pw_conv.east) {\small 仅通道, 1×1融合};
\end{tikzpicture}
```

**两步拆分**：
1. **Depthwise**：$K \times K$ 卷积，但每个通道独立处理（不跨通道）——参数量 $K^2 \cdot C_{in}$
2. **Pointwise**：$1 \times 1$ 卷积，只在通道间融合——参数量 $C_{in} \cdot C_{out}$

普通卷积参数量：$K^2 \cdot C_{in} \cdot C_{out}$
分离卷积参数量：$K^2 \cdot C_{in} + C_{in} \cdot C_{out}$

减少倍数 $= \dfrac{K^2 \cdot C_{in} \cdot C_{out}}{K^2 \cdot C_{in} + C_{in} \cdot C_{out}} = \dfrac{K^2 \cdot C_{out}}{K^2 + C_{out}}$

当 $C_{out} = 256$, $K=3$ 时，$\dfrac{9 \times 256}{9 + 256} \approx 8.7$，约减少 **8.7 倍**。

## 策略二：Bottleneck

Bottleneck 的核心思想来自 {doc}`../neural-network-basics/inception` 的 1×1 降维——**先压缩再膨胀**：

```{mermaid}
flowchart LR
  A["`输入
  $$H \times W \times C_{in}$$`"]
  B["`1×1 卷积
  $$C_{in} \times C_{out}$$`"]
  C["`3×3 卷积
  $$C_{out} \times C_{out}$$`"]
  D["`输出
  $$H \times W \times C_{out}$$`"]

  A --> B --> C --> D
```

这种"沙漏结构"将最昂贵的 3×3 卷积放在低维空间执行：

| 方式 | 参数分布 | 主要计算在哪 |
|------|---------|------------|
| 直接 3×3 | 256×3×3×256 | 高维空间计算 |
| Bottleneck | 256×1×1×64 + 64×3×3×64 + 64×1×1×256 | 3×3 在低维执行 |

**为什么 Bottleneck 有效**：1×1 卷积发现，256 个通道中的有效信息可以用更少的维度（64 维）表示——信息的**低维结构**保证了压缩不会丢失太多信息。

#### Bottleneck 的变体：Pre-activation vs Post-activation

原始的 ResNet Bottleneck 使用 **Post-activation**（卷积后激活）：

```{mermaid}
flowchart LR
    Input[输入] --> C1[Conv 1×1]
    C1 --> BN1[BN]
    BN1 --> ReLU1[ReLU]
    ReLU1 --> C3[Conv 3×3]
    C3 --> BN2[BN]
    BN2 --> ReLU2[ReLU]
    ReLU2 --> C1b[Conv 1×1]
    C1b --> BN3[BN]
    BN3 --> Add[相加]
    Input -.->|跳跃连接| Add
    Add --> ReLU3[ReLU]
    ReLU3 --> Output[输出]
```

改进的 **Pre-activation**（激活在卷积前）{cite}`he2016identity`：

```{mermaid}
flowchart LR
    Input[输入] --> BN1[BN]
    BN1 --> ReLU1[ReLU]
    ReLU1 --> C1[Conv 1×1]
    C1 --> BN2[BN]
    BN2 --> ReLU2[ReLU]
    ReLU2 --> C3[Conv 3×3]
    C3 --> BN3[BN]
    BN3 --> ReLU3[ReLU]
    ReLU3 --> C1b[Conv 1×1]
    C1b --> Add[相加]
    Input -.->|跳跃连接| Add
    Add --> Output[输出]
```

**Pre-activation 的优势**：
1. **梯度更畅通**：跳跃连接直接传递的是归一化后的特征，没有激活函数的"阻断"
2. **正则化效果更好**：每个卷积层前面都有 BN，训练更稳定
3. **理论上更优**：恒等映射 $y = x$ 不需要经过非线性变换

**实际效果**：在极深网络（1000+层）上，Pre-activation 优势明显；普通深度（50-200层）两者差距不大。

#### MobileNetV2 的倒残差结构（Inverted Residual）

MobileNetV2 结合了 DW 卷积和 Bottleneck，创造了**倒残差结构**：

```{tikz} MobileNetV2 倒残差结构对比
\begin{tikzpicture}[
    font=\small,
    >={Stealth[length=3mm]},
    conv/.style={
        draw,
        rounded corners=4pt,
        minimum height=1cm,
        align=center
    },
    arrow/.style={->, thick}
]

% ================= 左：ResNet =================
\node[font=\bfseries] at (-6,8) {标准残差块（ResNet）};

\node (rin) at (-6,7) {输入：高维};

\node[conv, fill=blue!15, minimum width=4.5cm] (r1) at (-6,5.8)
{1×1 卷积（降维）};

\node[conv, fill=blue!25, minimum width=2.6cm] (r2) at (-6,4.4)
{3×3 卷积};

\node[conv, fill=blue!15, minimum width=4.5cm] (r3) at (-6,3)
{1×1 卷积（升维）};

\node (rout) at (-6,1.8) {输出：高维};

% 主路径
\draw[arrow] (rin) -- (r1);
\draw[arrow] (r1) -- (r2);
\draw[arrow] (r2) -- (r3);
\draw[arrow] (r3) -- (rout);

% 真正 residual 路径
\draw[arrow, blue!60!black]
(rin.east) -- ++(2.8,0) 
             -- ++(0,-5.2) 
             -- (rout.east);

\node at (-6,0.6) {高 → 低 → 高（沙漏结构）};

% ================= 右：MobileNetV2 =================
\node[font=\bfseries] at (6,8) {倒残差块（MobileNetV2）};

\node (iin) at (6,7) {输入：低维};

\node[conv, fill=green!15, minimum width=2.6cm] (i1) at (6,5.8)
{1×1 卷积（升维）};

\node[conv, fill=orange!25, minimum width=4.5cm] (i2) at (6,4.4)
{DW 3×3 卷积};

\node[conv, fill=green!15, minimum width=2.6cm] (i3) at (6,3)
{1×1 卷积（降维）};

\node (iout) at (6,1.8) {输出：低维};

% 主路径
\draw[arrow] (iin) -- (i1);
\draw[arrow] (i1) -- (i2);
\draw[arrow] (i2) -- (i3);
\draw[arrow] (i3) -- (iout);

% residual
\draw[arrow, green!60!black]
(iin.west) -- ++(-2.8,0)
            -- ++(0,-5.2)
            -- (iout.west);

\node at (6,0.6) {低 → 高 → 低（漏斗结构）};

\end{tikzpicture}
```

**关键区别**：

| 特性 | 标准残差块 (ResNet) | 倒残差块 (MobileNetV2) |
|------|-------------------|----------------------|
| 形状 | 高 → 低 → 高（沙漏） | 低 → 高 → 低（漏斗） |
| 核心卷积 | 普通 3×3 | DW 3×3 |
| 升维/降维 | 先降后升 | 先升后降 |
| 跳跃连接 | 高维连接 | **低维连接**（关键！） |
| 激活位置 | 升维后激活 | 只在中间高维处激活 |

**为什么倒残差更高效？**

1. **DW卷积在高维执行更高效**：DW卷积的计算量与通道数成正比，但标准卷积与通道数的平方成正比
   - 在高维空间（192通道）执行 DW 卷积：$3 \times 3 \times 192 = 1,728$ 次乘法
   - 如果用标准卷积：$3 \times 3 \times 32 \times 192 = 55,296$ 次乘法
   - DW 节省 **32×**！

2. **低维跳跃连接节省内存**：跳跃连接不需要存储高维特征图

3. **Linear Bottleneck（线性瓶颈）**：最后的 1×1 降维后**不接 ReLU**
   - 原因：ReLU 在低维空间会破坏信息（一旦变负就永远为0）
   - 保持线性，保留更多信息

**设计心法**：倒残差是" DW 卷积的 Bottleneck 版本"——先升维创造"计算空间"，用 DW 卷积高效处理，再降维回到紧凑表示。

```{admonition} 设计心法
:class: important

Bottleneck、DW 卷积、倒残差本质上是一回事——**利用信息冗余来降本**，只是实现方式不同：

- **Bottleneck**：利用通道间的冗余（256→64→256）
- **DW 卷积**：利用空间和通道的可分离性（先空间再通道）
- **倒残差**：结合两者，在高维空间用 DW 卷积，两端保持低维

理解了这个本质，你就能灵活组合这些策略，设计出适合你自己任务的模块。
```

## 两种策略的对比与选择

| 策略 | 减少幅度 | 适用场景 | 代价 |
|------|---------|----------|------|
| 深度可分离 | ~5-10× | 需要极度轻量 | 可能损失少量精度 |
| Bottleneck | ~3-4× | 平衡效果与效率 | 需要合适通道设计 |
| 两者结合（MobileNetV2） | ~8-15× | 移动端部署 | 结构略复杂 |

## 失败案例：效率优化的陷阱

### 案例一：Bottleneck 压缩率过大

**场景**：设计图像分类网络，你将 Bottleneck 的压缩比设为 256→8→256（32 倍压缩）。

**结果**：
- 参数量确实大幅减少（约 15×）
- 但准确率从 75% 暴跌到 45%

**为什么失败**：
- 8 维的 Bottleneck 太小，无法承载 1000 类 ImageNet 的分类信息
- 信息瓶颈过于狭窄，有用的特征在压缩过程中被"挤掉"
- 从互信息角度：$I(X; Z)$ 太小，$Z$ 无法保留足够区分 1000 类的信息

**教训**：**Bottleneck 的维度有下限**。对于 C 类分类任务，Bottleneck 维度至少应该是 C 的 2-4 倍。1000 类任务，Bottleneck 不应小于 256 维。

### 案例二：DW 卷积用于通道数少的层

**场景**：网络第一层（3 通道输入，16 通道输出），你用了 DW 卷积+1×1 pointwise，而不是标准 3×3 卷积。

**结果**：
- 计算量反而增加了！
- 准确率下降 2%

**为什么失败**：
- DW 卷积的优势在于"空间和通道分离"
- 但当输入通道只有 3 时，DW 卷积的"逐通道卷积"几乎等价于标准卷积（反正每个输入通道独立处理）
- 但 DW 卷积后还要加 1×1 pointwise 来融合通道，多了一步计算
- 标准卷积在 3→16 时本来就是 3×3×3×16，没有冗余可省

**教训**：**DW 卷积只适用于通道数 $\geq 32$ 的情况**。浅层（通道数少）用标准卷积更高效。

### 案例三：盲目追求轻量导致无法收敛

**场景**：为了部署到嵌入式设备，你将 MobileNetV2 的宽度乘子（width multiplier）设为 0.1（所有通道数变为原来的 10%）。

**结果**：
- 模型大小只有 0.5MB，满足部署要求
- 但训练 loss 根本不下降，模型无法学习

**为什么失败**：
- 宽度乘子 0.1 意味着 Bottleneck 维度从 192 降到 19
- 19 维的特征空间无法表达 CIFAR-10 甚至 MNIST 的特征
- 模型容量被压缩到连简单任务都无法完成

**教训**：**效率优化有极限**。宽度乘子通常不应小于 0.25，否则模型会"饿死"。

### 案例四：倒残差用错激活函数位置

**场景**：实现倒残差块时，你在最后的 1×1 降维后加了 ReLU 激活。

**结果**：
- 准确率比标准 MobileNetV2 低 3-5%
- 特征可视化显示大量神经元"死亡"（恒为 0）

**为什么失败**：
- 倒残差的最后降维到低维空间（如 24 通道）
- ReLU 在低维空间是"信息杀手"——一旦输出为负，信息永久丢失
- MobileNetV2 论文明确建议在最后的 Bottleneck 用线性激活（Linear Bottleneck）

**教训**：**倒残差的最后必须是线性激活**。这是 MobileNetV2 的关键设计，不能省略。

### 决策树：效率优化策略选择

```{mermaid}
flowchart TD
    START[目标平台?] --> SERVER{服务器/桌面 GPU}
    SERVER -->|计算充裕| STD[标准 ResNet<br/>追求最佳效果]
    
    SERVER -->|否| EDGE{边缘设备<br/>Jetson/树莓派}
    EDGE -->|精度要求高| BOT[Bottleneck + 标准卷积]
    EDGE -->|极度受限| MBV2[MobileNetV2<br/>DW + 倒残差]
    
    EDGE -->|否| MOBILE{移动端 APP}
    MOBILE -->|有 NPU| EFF[MobileNetV2<br/>或 EfficientNet]
    MOBILE -->|纯 CPU| SMALL[宽度乘子 0.5-0.75<br/>的 MobileNet]
```

**核心原则**：
- 效率优化是**权衡**，不是"免费午餐"
- 先确定 baseline 能正常工作，再逐步优化
- 每次优化后都要验证准确率，别只盯着模型大小

## 信息论视角：效率优化为什么可能

效率优化能成功的根本原因来自信息论：**自然图像的特征具有稀疏性和低维结构**。

通道之间不是独立的——256 个通道中，很多通道捕捉的是高度相关的信息。1×1 卷积能发现这种相关性，将 256 维压缩到 64 维，保留的信息量基本相同。

同样，空间和通道的**可分离性**意味着：空间模式（边缘、纹理）和通道模式（哪种特征）的交互不是任意组合的——把它们分开处理，信息损失很小。

用{ref}`mutual-information`的语言说：
- Bottleneck 做了 $I(X;Z) \approx I(X;Y)$ 的信息压缩——压缩后信息几乎不变
- DW 卷积利用了 $I_{\text{space}, \text{channel}} \approx I_{\text{space}} + I_{\text{channel}}$ 的近似分离性

这就是为什么我们能"白嫖"计算量——压缩掉的是冗余，保留的是信息。

## 下一步

四个维度全部讲完。现在你有了完整的改造武器库。但知道武器和会用武器是两码事——下一节{doc}`part6-diagnosis`我们将学习**如何诊断具体问题并选择合适的武器**。

```{admonition} 效率优化的更多方向
:class: tip

本章聚焦于**架构级**的改造（改变层的设计）。除此之外，效率优化还有两大方向值得了解：

1. **剪枝（Pruning）**：训练后删除不重要的连接或通道——已有足够信息，不需要全部保留
2. **量化（Quantization）**：用更低精度（如 FP32 → INT8）存储和计算——信息精度有冗余

这两个方向与架构设计是互补的：先用 DW 卷积/Bottleneck 设计高效架构，再通过剪枝和量化进一步压缩。感兴趣可参阅 `torch.quantization` 和剪枝相关文献。
```

---

## 参考文献

```{bibliography}
:filter: docname in docnames
```
