(arch-receptive-field)=
# 操控感受野：网络应该看多大

{ref}`receptive-field` 告诉我们：卷积层越深，感受野越大。但 {doc}`../neural-network-basics/inception` 揭示了一个根本问题——**每一层只有一个固定大小的感受野**。

这就是操控感受野要解决的核心矛盾：**不同的目标需要不同大小的感受野，但传统 CNN 只有一个**。

## 什么时候需要改造感受野

先确认你是不是真的遇到了感受野问题：

| 症状 | 可能原因 | 是否感受野问题 |
|------|----------|--------------|
| 大物体（占据半张图）识别差 | 感受野只能看到局部 | **是** |
| 同时有大物体和小物体的图效果差 | 单一感受野顾此失彼 | **是** |
| 所有物体识别都差 | 可能不是感受野问题 | 先检查训练 |
| 小物体识别差、大物体尚可 | 大感受野丢失细节 | **是** |

```{admonition} 关键判断
:class: tip

如果加大卷积核尺寸（比如 3→7）能提升效果，那大概率是感受野问题。
```

## 策略一：多尺度并行

这是 {doc}`../neural-network-basics/inception` 的核心思想，但我们从**设计心法**而非架构本身来理解它。

**本质**：不说"用多大"，而说"都试试"。同时运行多个感受野的卷积，让网络自己选。

```{tikz} Inception 的多尺度并行
\begin{tikzpicture}[
    scale=1,
    font=\small,
    >={Stealth},
    block/.style={
        draw,
        rounded corners=4pt,
        minimum width=2.8cm,
        minimum height=1.1cm,
        align=center
    },
    conv3/.style={block, fill=blue!15},
    conv5/.style={block, fill=orange!25},
    conv1/.style={block, fill=green!25},
    io/.style={block}
]

% ======================================================
% 左：传统卷积
% ======================================================

\node[font=\bfseries] at (-5,6) {传统卷积};

\node[io] (lin) at (-5,4.8) {输入};
\node[conv3] (lconv) at (-5,3.2) {$3\times3$ 卷积};
\node[io] (lout) at (-5,1.6) {输出};

\draw[->] (lin) -- (lconv);
\draw[->] (lconv) -- (lout);

\node[gray] at (-5,0.6) {固定感受野};


% ======================================================
% 右：Inception
% ======================================================

\node[font=\bfseries] at (4,6) {多尺度并行（Inception）};

\node[io] (rin) at (4,4.8) {输入};

\node[conv1] (r1) at (0.5,3.2) {$1\times1$};
\node[conv3] (r3) at (4,3.2) {$3\times3$};
\node[conv5] (r5) at (7.5,3.2) {$5\times5$};

\node[draw, rounded corners=4pt, minimum width=3cm, minimum height=1.1cm] 
(concat) at (4,1.6) {Concat};

\node[io] (rout) at (4,0.2) {输出};

% 连接
\draw[->] (rin) -- (r1);
\draw[->] (rin) -- (r3);
\draw[->] (rin) -- (r5);

\draw[->] (r1) -- (concat);
\draw[->] (r3) -- (concat);
\draw[->] (r5) -- (concat);

\draw[->] (concat) -- (rout);

\node[gray] at (4,-0.8) {多尺度特征融合};

\end{tikzpicture}
```

**适用场景**：你有多个大小悬殊的目标类别（比如自然图像分类）。

**代价与缓解**：参数和计算量增加（多个分支同时跑）。{doc}`../neural-network-basics/inception` 中的关键创新是用 **1×1 卷积先降维**——在昂贵的 3×3 和 5×5 卷积之前，先用 1×1 把通道数降下来：

**心法**：这是一种"信息保险"——不确定感受野应该多大时，把不同大小的都保留下来。1×1 降维则是"用计算换信息"的精打细算。

## 策略二：空洞卷积

{ref}`arch-intro` 中我们简单介绍了空洞卷积——它在不增加参数的情况下扩大感受野。现在从**设计心法**来理解它。

**本质**：用稀疏采样替代密集采样。3×3 卷积 + 空洞率 2 = 5×5 的感受野，但参数还是 9 个。{doc}`../neural-network-basics/inception` 中我们已用 TikZ 图展示了普通卷积与空洞卷积的直观对比，这里不再重复。

**设计原则**：

| 空洞率 | 3×3 卷积的感受野 | 相当于普通 | 参数量 |
|--------|----------------|-----------|--------|
| 1 | 3×3 | 3×3 | 9 |
| 2 | 5×5 | 5×5 | 9 |
| 3 | 7×7 | 7×7 | 9 |
| 4 | 9×9 | 9×9 | 9 |

**心法**：空洞卷积是"用稀疏换取广度"。你牺牲了每个卷积核的采样密度，换来不花钱的大感受野。

**代价：网格效应（Gridding Effect）**

稀疏采样不是免费的午餐。当空洞率过大时，卷积核只采样输入的"离散点"，会丢失局部连续性信息，产生棋盘格状的伪影：

```{tikz} 空洞卷积的网格效应对比
\begin{tikzpicture}[
    font=\small,
    sample/.style={circle, fill=blue!70, inner sep=2.5pt},
    blind/.style={rectangle, fill=red!25, minimum size=0.8cm, draw=none},
    gridline/.style={gray!40}
]

% =====================================================
% 左：普通卷积
% =====================================================
\node[font=\bfseries] at (0,4.2) {普通 $3\times3$ 卷积};

\begin{scope}[shift={(0,1)}]
    \draw[gridline] (-1,-1) grid (1,1);

    \foreach \x in {-1,0,1}
        \foreach \y in {-1,0,1}
            \node[sample] at (\x,\y) {};
\end{scope}

\node[align=center] at (0,-3)
{密集采样\\9个采样点};


% =====================================================
% 中：空洞卷积
% =====================================================
\node[font=\bfseries] at (5,4.2) {空洞率 $r=2$};

\begin{scope}[shift={(5,1)}]
    \draw[gridline] (-2,-2) grid (2,2);

    \foreach \x in {-2,0,2}
        \foreach \y in {-2,0,2}
            \node[sample] at (\x,\y) {};
\end{scope}

\node[align=center] at (5,-3)
{稀疏采样\\$5\times5$ 感受野，仅9个采样点};


% =====================================================
% 右：网格效应
% =====================================================
\node[font=\bfseries] at (10,4.2) {网格效应};

\begin{scope}[shift={(10,1)}]
    \draw[gridline] (-2,-2) grid (2,2);

    % 盲区（非采样点）
    \foreach \x in {-2,-1,0,1,2}
        \foreach \y in {-2,-1,0,1,2}
        {
            \pgfmathparse{mod(\x,2)==0 && mod(\y,2)==0 ? 0 : 1}
            \ifnum\pgfmathresult=1
                \node[blind] at (\x,\y) {};
            \fi
        }

    % 采样点
    \foreach \x in {-2,0,2}
        \foreach \y in {-2,0,2}
            \node[sample] at (\x,\y) {};
\end{scope}

\node[align=center] at (10,-3)
{采样盲区（红色）\\导致棋盘状伪影};

\end{tikzpicture}
```

**缓解策略**：
- 不要把空洞率设得太大（通常 ≤ 16）
- 不同空洞率的卷积并行（ASPP），互相弥补采样盲区

**适用场景**：需要大感受野但参数量、计算量受限（分割任务、轻量模型）。

**典型应用——ASPP（Atrous Spatial Pyramid Pooling）**

DeepLab 系列的核心模块，同时并行 4 个不同空洞率的卷积（空洞率=6, 12, 18, 24）+ 1 个全局平均池化：

```{tikz} ASPP 结构：多空洞率并行
\begin{tikzpicture}[
    font=\small,
    >={Stealth[length=2mm]},
    node distance=1.4cm and 1.6cm,
    block/.style={
        draw,
        rounded corners=4pt,
        minimum height=0.9cm,
        minimum width=2.2cm,
        align=center
    },
    conv/.style={block, fill=blue!12},
    pool/.style={block, fill=green!12},
    fusion/.style={block, fill=orange!18, minimum width=6.5cm},
    arrow/.style={->, thick}
]

% ================= 输入 =================
\node[block, fill=gray!10] (input) {输入特征图};

% ================= 分支层 =================
\node[conv, below left=of input, xshift=-2.2cm] (r6)
{r=6\\$5\times5$};

\node[conv, below=of input, xshift=-2.6cm] (r12)
{r=12\\$11\times11$};

\node[conv, below=of input, xshift=2.6cm] (r18)
{r=18\\$17\times17$};

\node[conv, below right=of input, xshift=2.2cm] (r24)
{r=24\\$23\times23$};

% ================= GAP =================
\node[pool, below=1.6cm of input] (gap)
{全局平均池化};

% ================= 融合 =================
\node[fusion, below=2.2cm of gap] (concat)
{拼接 (Concat) + $1\times1$ 卷积};

% ================= 连线 =================
\foreach \branch in {r6,r12,r18,r24}
    \draw[arrow] (input.south) -- (\branch.north);

\draw[arrow] (input.south) -- (gap.north);

\foreach \branch in {r6,r12,r18,r24}
    \draw[arrow] (\branch.south) -- (concat.north);

\draw[arrow] (gap.south) -- (concat.north);

\end{tikzpicture}
```

ASPP 与 Inception 的区别：
- Inception：多个卷积核尺寸（1×1, 3×3, 5×5）
- ASPP：相同卷积核，不同空洞率（3×3 with rate=6,12,18,24）

前者增加参数，后者不增加参数——都是多尺度，但代价不同。

## 策略三：特征金字塔（FPN）——不同深度、不同感受野

Inception 和空洞卷积解决的是**同一层的多尺度**问题。但 CNN 有一个天然的多尺度结构——**不同深度的层自然拥有不同感受野**。

浅层特征：分辨率高、感受野小 → 适合检测小物体
深层特征：分辨率低、感受野大 → 适合检测大物体

特征金字塔网络（FPN, Feature Pyramid Network）的洞察是：不给每一层单独分配目标大小，而是**在不同深度的特征之间建立信息通道，让它们对话**。

**心法**：FPN/PAN 的本质是**在不同感受野的特征之间建立信息通道**。信息不仅在层之间流动，还在不同感受野之间流动。

### FPN 的结构：自上而下 + 横向连接

FPN 并不是简单的"把不同层的特征拼起来"，它有一个精妙的结构设计：

```{tikz} FPN 结构：自上而下路径与横向连接
\begin{tikzpicture}[
    scale=1,
    font=\small,
    >={Stealth},
    feat/.style={
        draw,
        minimum width=2.4cm,
        minimum height=0.9cm,
        align=center
    },
    enc/.style={feat, fill=blue!12},
    fpn/.style={feat, fill=green!12},
    Nout/.style={feat, fill=orange!15}
]

% ======================
% 标题
% ======================
\node[font=\bfseries] at (-4,7) {自下而上（Encoder）};
\node[font=\bfseries] at (5.5,7) {FPN（自上而下 + 横向连接）};

% ======================
% Encoder
% ======================
\node[enc] (c1) at (-4,5.5) {$C_1$};
\node[enc] (c2) at (-4,4) {$C_2$};
\node[enc] (c3) at (-4,2.5) {$C_3$};
\node[enc] (c4) at (-4,1.0) {$C_4$};

\draw[->] (c1) -- (c2);
\draw[->] (c2) -- (c3);
\draw[->] (c3) -- (c4);

% ======================
% FPN Pyramid
% ======================
\node[fpn] (p4) at (4,1.0) {$P_4$};
\node[fpn] (p3) at (4,2.5) {$P_3$};
\node[fpn] (p2) at (4,4) {$P_2$};
\node[fpn] (p1) at (4,5.5) {$P_1$};

% top-down
\draw[->] (p4) -- (p3);
\draw[->] (p3) -- (p2);
\draw[->] (p2) -- (p1);

% lateral connections
\draw[->] (c4.east) -- (p4.west);
\draw[->] (c3.east) -- (p3.west);
\draw[->] (c2.east) -- (p2.west);
\draw[->] (c1.east) -- (p1.west);

% outputs
\node[Nout] (o1) at (7,5.5) {输出};
\node[Nout] (o2) at (7,4) {输出};
\node[Nout] (o3) at (7,2.5) {输出};
\node[Nout] (o4) at (7,1.0) {输出};

\draw[->] (p1) -- (o1);
\draw[->] (p2) -- (o2);
\draw[->] (p3) -- (o3);
\draw[->] (p4) -- (o4);

\end{tikzpicture}
```

**为什么需要这种结构？**

单纯使用深层特征的问题：分辨率太低，小物体可能已经"消失"了。单纯使用浅层特征的问题：感受野太小，看不清全局上下文。

FPN 的解决方案：
1. **横向连接**：保留浅层的高分辨率细节
2. **自上而下路径**：把深层的强语义信息传递回浅层  
3. **逐层融合**：每个输出层都同时具有"高分辨率 + 强语义"

**融合机制的具体操作**

横向连接和自上而下路径相遇时，不是简单相加，而是有一个精心设计的流程：

```{tikz} FPN融合机制流程
\begin{tikzpicture}[
    font=\small,
    >={Stealth[length=2mm]},
    block/.style={
        draw,
        rounded corners=4pt,
        minimum width=3.6cm,
        minimum height=1.1cm,
        align=center
    },
    deep/.style={block, fill=blue!8},
    shallow/.style={block, fill=green!8},
    neutral/.style={block, fill=gray!10},
    arrow/.style={->, thick}
]

% ================= 顶层 =================
\node[deep] (deep) at (-4,6)
{深层特征 $P_{l+1}$\\小尺寸，强语义};

\node[shallow] (shallow) at (4,6)
{浅层特征 $C_l$\\大尺寸，高分辨率};

% ================= 中间处理 =================
\node[neutral] (upsample) at (-4,4)
{上采样 $\times2$};

\node[neutral] (conv1x1) at (4,4)
{$1\times1$ 卷积\\通道调整};

% ================= 对齐 =================
\node[deep] (pup) at (-4,2)
{$P_{l+1}^{up}$\\尺寸对齐};

\node[shallow] (clprime) at (4,2)
{$C_l'$\\通道匹配};

% ================= 融合 =================
\node[neutral] (add) at (0,0)
{逐元素相加};

\node[neutral] (pl) at (0,-2)
{$P_l$\\高分辨率 + 强语义};

% 可选卷积
\node[neutral] (smooth) at (5,-2)
{可选 $3\times3$ 卷积\\去混叠};

% ================= 连线 =================
\draw[arrow] (deep) -- (upsample);
\draw[arrow] (upsample) -- (pup);

\draw[arrow] (shallow) -- (conv1x1);
\draw[arrow] (conv1x1) -- (clprime);

\draw[arrow] (pup.south) |- (add.west);
\draw[arrow] (clprime.south) |- (add.east);

\draw[arrow] (add) -- (pl);
\draw[arrow] (pl) -- (smooth);

\end{tikzpicture}
```

**关键设计选择**：

| 操作 | 选项 | FPN的选择 | 原因 |
|------|------|----------|------|
| 上采样 | 最近邻/双线性/转置卷积 | 最近邻或双线性 | 简单高效，后续有3×3优化 |
| 通道调整 | 1×1卷积/直接截断 | 1×1卷积 | 可学习地选择重要通道 |
| 融合方式 | 相加/拼接 | **相加** | 保持通道数，计算效率高 |
| 后处理 | 3×3卷积/无 | 可选3×3 | 消除上采样的混叠效应 |

为什么用**相加**而不是拼接？
- 相加：通道数不变，计算效率高，适合"融合语义"
- 拼接：通道数翻倍，信息保留更完整但计算量大

FPN 选择相加，是因为深层的强语义已经足够指导浅层检测——目标是**融合**而非**堆叠**信息。

```{admonition} 与 Inception 的本质区别
:class: tip

| 策略 | 多尺度来源 | 信息如何融合 |
|------|-----------|-------------|
| Inception | 同一层的多个卷积核 | 拼接（concat）|
| FPN | 不同深度的自然层次 | 自上而下 + 横向连接 |

Inception 是"横向并行"，FPN 是"纵向融合"。两者可以结合使用（如 PANet）。
```

## 三种策略的对比与选择

| 策略 | 感受野 | 计算代价 | 适用任务 |
|------|--------|----------|----------|
| 多尺度并行（Inception） | 同一层多个固定大小 | 中等（多分支） | 分类——目标大小不确定 |
| 空洞卷积 | 不增参扩大感受野 | 低 | 分割——需要大感受野但计算受限 |
| FPN/PAN | 不同层不同感受野+融合 | 中高（上采样+融合） | 检测/分割——多尺度目标 |

```{admonition} 设计心法
:class: important

**操控感受野的本质不是"选一个大小"，而是"让网络同时看到多个尺度"**。

三种策略的区别仅在于**实现方式**：Inception 在同一层内并行多个卷积核、空洞卷积用稀疏采样换广度、FPN 利用网络深度天然的多尺度结构。
```

## 失败案例：什么情况下不该用

### 案例一：MNIST 上用 Inception

**场景**：MNIST 手写数字识别（$28\times28$ 灰度图），你在第一层就加了 Inception 模块（1×1、3×3、5×5、MaxPool 并行）。

**结果**：
- 训练时间翻倍
- 准确率从 99.2% 降到 99.0%（或没有提升）

**为什么失败**：
- MNIST 数字只占图像中心一小块，大小相对固定
- 5×5 卷积核几乎覆盖了整个数字，没有"多尺度"的必要
- 增加的分支引入了更多参数，但没有带来信息增益

**教训**：**感受野策略的前提是"确实存在多尺度"**。如果任务本身尺度单一（如对齐良好的人脸、固定大小的字符），多尺度就是浪费。

### 案例二：分割任务中空洞率过大

**场景**：医学图像分割（细胞分割），你用空洞率 16 的 3×3 卷积来扩大感受野。

**结果**：
- 大细胞分割还可以
- **小细胞大量丢失**——在特征图上彻底"消失"

**为什么失败**：
- 空洞率 16 意味着卷积核实际采样点间距为 16 像素
- 对于直径只有 20-30 像素的细胞，3×3 空洞卷积只能看到 3 个采样点
- 信息严重欠采样，小目标被"跳过"

**教训**：**空洞卷积的上限是目标最小尺寸**。如果目标只有 20 像素，空洞率不应超过 5-7。

### 案例三：分类任务盲目用 FPN

**场景**：ImageNet 分类，你把 ResNet 改造成了 FPN 结构——多尺度输出、自上而下连接。

**结果**：
- 参数量和计算量显著增加
- Top-5 准确率没有提升，甚至略有下降

**为什么失败**：
- FPN 的核心价值是**多尺度目标的定位和分类**（检测/分割）
- 分类任务只需要一个全局判断，不需要"每层都输出"
- 额外的上采样和融合引入了计算负担，但没有对应的收益

**教训**：**FPN 是为目标检测/分割设计的，不是为纯分类**。分类任务用 FPN 是"杀鸡用牛刀"。

### 决策树：什么时候用什么

```
任务类型?
├── 检测/分割（需要定位）
│   ├── 多尺度目标明显? → FPN/PAN
│   ├── 计算资源紧张? → 空洞卷积
│   └── 固定尺度目标? → 普通卷积即可
│
└── 分类（只需要全局判断）
    ├── 尺度变化大? → Inception/多尺度池化
    └── 尺度固定? → 标准卷积，别折腾
```

**核心原则**：感受野改造是为了解决**特定问题**，不是为了"看起来厉害"。如果 baseline 没有明显问题，不要引入复杂度。

## 信息论视角：多尺度 = 多频段采样

从信息论来看，感受野策略都是**多频段采样**{cite}`tishby2000information`：

- 自然界图像的{ref}`mutual-information`分布在不同的空间频率上
- 小感受野捕捉高频（细节），大感受野捕捉低频（全局结构）
- 单一感受野 = 信息窄带采样，丢失其他频段的信息
- 多尺度策略 = 宽带采样，**最大化 $I(X;Y)$**

{ref}`arch-intro` 中讨论的信息漏斗问题，感受野的操控策略本质上是**在空间维度上拓宽漏斗的入口**——不止截取一个频段，而是同时截取所有频段。

## 下一步

操控感受野解决了"看什么"的问题。但光看还不够——信息必须在网络中有效地流动，梯度必须能传回来。这就是下一节{doc}`part3-depth-connection`要解决的问题。

---

## 参考文献

```{bibliography}
:filter: docname in docnames
```
