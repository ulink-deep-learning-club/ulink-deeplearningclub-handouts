(rnn-basics)=
# 循环神经网络：让网络拥有"记忆"

{doc}`../neural-network-basics/fc-layer-basics` 中的全连接网络和 {doc}`../neural-network-basics/cnn-basics` 中的 CNN 有一个共同特征：**给定一个输入，产生一个输出，然后"忘记"这个输入**。每次前向传播是独立的，网络不记得上一秒看到了什么。

这显然不符合我们处理序列的方式。读这句话时，你的大脑记住了前文——"这"指代什么，"显然"修饰什么——这些理解都依赖于**记忆**。

**关键问题**：如何让神经网络在处理序列时拥有记忆？

## 历史背景

1986年，David Rumelhart 和 Geoffrey Hinton 等人发表了反向传播论文 {cite}`rumelhart1986learning`，为训练多层网络奠定了基础。几乎同时，Michael Jordan 提出了带有循环连接的网络架构 {cite}`jordan1986serial`——让网络将上一时刻的输出作为当前输入的一部分，从而拥有对过去的"记忆"。

1990年，Jeffrey Elman 在此基础上提出了**简单循环网络（Simple Recurrent Network）** {cite}`elman1990finding`，其核心结构——一个隐状态 $\mathbf{h}_t$ 在每一步基于 $\mathbf{h}_{t-1}$ 和 $\mathbf{x}_t$ 更新——成为此后三十年所有循环架构的基石。

```{note}
**为什么叫"循环"？** 因为网络在时间上的每一步使用**相同的权重** $\mathbf{W}_h$ 和 $\mathbf{W}_x$。这类似于 {ref}`cnn-basics` 中卷积核的权值共享——同一个规则应用到所有位置，只是 CNN 共享空间位置，RNN 共享时间位置。
```

## 从大脑到RNN：记忆的直觉

### 你如何理解一句话？

想象你逐字阅读："今天天气很好所以我决定去公园散步"。

- 读到"今天天气很好"时，你建立了一个上下文
- 读到"所以"时，你知道接下来是结论
- 读到"我决定去"时，你预期一个动作
- 读到"公园散步"时，你把它和"天气好"建立了因果联系

每一步，你的大脑都在**更新一个内部状态**——一个不断演化的、包含前文信息的"摘要"。这个摘要不是完整记忆（你不会逐字背诵前文），而是对理解当前词最关键的**压缩表示**。

### 生物学的启发：大脑不"反复读取"历史

RNN 的灵感直接来自大脑中一个更根本的事实——**大脑不是每个瞬间把所有历史"重新读一遍"**。

想想你是怎么理解一段音乐的。听到第三个音符时，你并没有重新播放前两个音符。你对前两个音符的感知已经"融入"了当前的神经活动——大脑利用一系列**持续变化的物理状态**来自然携带历史信息：

- **神经元膜电位**：一个神经元被激活后，它的膜电位不会瞬间归零，而是逐渐衰减。这个衰减过程本身就在"记住"最近发生过什么。
- **突触短期可塑性**：高频使用一个突触后，它的传递效率会在短时间内改变（增强或抑制）——这是突触层面上的"工作记忆"。
- **钙离子浓度**：当神经元放电时，钙离子流入细胞内，浓度缓慢下降。这个化学梯度编码了过去几百毫秒内的活动强度。
- **神经递质残留**：突触间隙中的神经递质分子不会瞬间清除，它们的残余浓度携带着"刚才这里发生了信号传递"的信息。

这些都不是"显式的记忆存储"——大脑没有一个单独的"历史缓冲区"。历史信息是**物理地、连续地**嵌入在神经元和突触的状态变化中的。每一个瞬间的大脑状态都已经是所有过去输入的**压缩结果**。

```{note}
**RNN 的本质**：$\mathbf{h}_t = \tanh(\mathbf{W}_h \mathbf{h}_{t-1} + \mathbf{W}_x \mathbf{x}_t)$ 就是对这个过程的数学抽象。$\mathbf{h}_t$ 对应神经元群的膜电位模式——它随时间连续演化，每一步吸收新的输入（$\mathbf{W}_x \mathbf{x}_t$），同时自然地携带上一步的状态（$\mathbf{W}_h \mathbf{h}_{t-1}$）。信息通过时间上连续的状态变化向前传递，而不是反复从静态记忆中读取。

这就是"循环"二字的生物学含义：**不是重新读取，而是持续继承**。
```

### RNN的循环结构

RNN做的正是这件事 {cite}`elman1990finding`。它维护一个**隐状态（hidden state）** $\mathbf{h}_t$，在处理每一步输入 $\mathbf{x}_t$ 时：

```{tikz} RNN循环结构：同一模块在时间上展开
\begin{tikzpicture}[
    >=Stealth, % 使用更现代的箭头
    node distance=1.2cm and 1cm, % 控制节点间的默认间距
    every node/.style={font=\small},
    % 定义 RNN 模块样式
    rnn/.style={
        rectangle, 
        rounded corners=6pt,
        draw=blue!60, 
        fill=blue!8,
        minimum width=2.8cm,
        minimum height=1.5cm,
        align=center
    },
    % 定义隐藏状态样式
    state/.style={
        circle,
        draw=teal!60,
        fill=teal!10,
        minimum size=1cm
    },
    % 定义输入样式
    input/.style={
        circle,
        draw=orange!70!black,
        fill=orange!15,
        minimum size=0.9cm
    },
    % 统一的箭头样式
    arrow/.style={thick, ->},
    recarrow/.style={thick, ->, blue!70}
]

% =====================================================
% 上方：折叠视图 (居中于 x=0)
% =====================================================

% 主循环模块
\node[rnn] (folded) at (0,0) {
    $\mathbf{h}_t = \tanh(\mathbf{W}_h \mathbf{h}_{t-1} + \mathbf{W}_x \mathbf{x}_t)$
    \\[6pt] % 增加一点行间距
    \scriptsize 循环模块
};

% 输入和输出箭头
\draw[arrow] ($(folded.west)+(-1.5,0)$) node[left] {$\mathbf{x}_t$} -- (folded.west);
\draw[arrow] (folded.east) -- ($(folded.east)+(1.5,0)$) node[right] {$\mathbf{h}_t$};

% 顶部循环箭头
\draw[recarrow] (folded.north east) 
    to[out=60, in=120, looseness=2] 
    node[above] {\scriptsize 循环} 
    (folded.north west);

% 折叠视图标签
\node[below=0.6cm of folded] (label1) {\textbf{折叠视图}};


% =====================================================
% 下方：展开视图 (以 x=0 为中心向两边展开)
% =====================================================

% 为了完美居中，我们将中间的 cell2 放在 x=0 的正下方
\node[rnn, below=2.5cm of label1] (cell2) {RNN};

% 向左构建节点
\node[state, left=of cell2] (h1) {$\mathbf{h}_1$};
\node[rnn, left=of h1] (cell1) {RNN};
\node[state, left=of cell1] (h0) {$\mathbf{h}_0$};

% 向右构建节点
\node[state, right=of cell2] (h2) {$\mathbf{h}_2$};
\node[rnn, right=of h2] (cell3) {RNN};
\node[state, right=of cell3] (h3) {$\mathbf{h}_3$};

% 下方输入节点
\node[input, below=1cm of cell1] (x1) {$\mathbf{x}_1$};
\node[input, below=1cm of cell2] (x2) {$\mathbf{x}_2$};
\node[input, below=1cm of cell3] (x3) {$\mathbf{x}_3$};

% 绘制横向隐藏状态流动的箭头
\draw[arrow] (h0) -- (cell1);
\draw[arrow] (cell1) -- (h1);
\draw[arrow] (h1) -- (cell2);
\draw[arrow] (cell2) -- (h2);
\draw[arrow] (h2) -- (cell3);
\draw[arrow] (cell3) -- (h3);

% 绘制自底向上的输入箭头
\draw[arrow] (x1) -- (cell1);
\draw[arrow] (x2) -- (cell2);
\draw[arrow] (x3) -- (cell3);

% 展开视图标签 (对齐中间的 x2 节点)
\node[below=0.8cm of x2] {\textbf{展开视图（时间维度）}};

\end{tikzpicture}
```

两行公式定义了RNN的全部行为：

$$
\mathbf{h}_t = \tanh(\mathbf{W}_h \mathbf{h}_{t-1} + \mathbf{W}_x \mathbf{x}_t + \mathbf{b})
$$

$$
\mathbf{y}_t = \mathbf{W}_y \mathbf{h}_t + \mathbf{b}_y
$$

其中：
- $\mathbf{h}_t \in \mathbb{R}^d$：时刻 $t$ 的隐状态（网络的"记忆"）
- $\mathbf{x}_t \in \mathbb{R}^m$：时刻 $t$ 的输入
- $\mathbf{W}_h \in \mathbb{R}^{d \times d}$：**循环权重**——控制"如何混合旧记忆和新输入"
- $\mathbf{W}_x \in \mathbb{R}^{d \times m}$：输入投影——将输入映射到隐空间
- $\mathbf{W}_y \in \mathbb{R}^{k \times d}$：输出投影——从隐状态产生预测

```{note}
**直觉**：$\mathbf{W}_h \mathbf{h}_{t-1}$ 是"提炼旧记忆"，$\mathbf{W}_x \mathbf{x}_t$ 是"吸收新信息"，$\tanh$ 将两者混合并压缩到 $(-1, 1)$，防止数值发散。

注意这里用的是 $\tanh$ 而不是 ReLU。这是因为循环连接会让信号反复通过激活函数——ReLU 的正区间可能让激活值在循环中不断放大直至爆炸，而 $\tanh$ 天然有界，提供了一种"自我保护"机制（{ref}`activation-functions` 中讨论过各激活函数的特性）。
```

(rnn-training)=
## RNN的训练：穿过时间的反向传播（BPTT）

RNN如何学习？答案就在 {ref}`back-propagation` 和 {ref}`computational-graph` 中。

把RNN在时间上展开：时刻1的处理产生 $\mathbf{h}_1$，传给时刻2的使用；时刻2产生 $\mathbf{h}_2$，传给时刻3……这恰好形成了一个**计算图**——每个时间步是一个节点，$\mathbf{h}_{t-1} \to \mathbf{h}_t$ 是连接节点的边。

反向传播在这个展开的图上运行，称为 **BPTT（Backpropagation Through Time）** {cite}`werbos1990backpropagation`：

```{tikz} BPTT：梯度沿时间反向流动
\begin{tikzpicture}[
    >=Stealth,
    node distance=2.2cm,
    every node/.style={font=\small},
    
    state/.style={
        circle,
        draw=teal!60,
        fill=teal!8,
        minimum size=0.9cm
    },
    input/.style={
        circle,
        draw=orange!70!black,
        fill=orange!12,
        minimum size=0.75cm
    },
    loss/.style={
        circle,
        draw=purple!60,
        fill=purple!10,
        minimum size=0.9cm
    },
    forward/.style={->, line width=1pt, blue!70},
    backward/.style={->, line width=1pt, red!70, dashed}
]

% ===============================
% Hidden states
% ===============================
\node[state] (h0) {$\mathbf{h}_0$};
\node[state, right=of h0] (h1) {$\mathbf{h}_1$};
\node[state, right=of h1] (h2) {$\mathbf{h}_2$};
\node[state, right=of h2] (h3) {$\mathbf{h}_3$};
\node[state, right=of h3] (h4) {$\mathbf{h}_4$};

% ===== Forward（整体上移）=====
\foreach \a/\b/\lab in {
    h0/h1/{\mathbf{W}_h},
    h1/h2/{\mathbf{W}_h},
    h2/h3/{\mathbf{W}_h},
    h3/h4/{\mathbf{W}_h}
}
{
    \draw[forward]
    ($(\a.north)+(0,0.15)$)
    -- node[above=4pt] {$\lab$}
    ($(\b.north)+(0,0.15)$);
}

% ===== Backward（整体下移）=====
\foreach \a/\b/\lab in {
    h4/h3/{\frac{\partial \mathbf{h}_4}{\partial \mathbf{h}_3}},
    h3/h2/{\frac{\partial \mathbf{h}_3}{\partial \mathbf{h}_2}},
    h2/h1/{\frac{\partial \mathbf{h}_2}{\partial \mathbf{h}_1}},
    h1/h0/{\frac{\partial \mathbf{h}_1}{\partial \mathbf{h}_0}}
}
{
    \draw[backward]
    ($(\a.south)+(0,-0.15)$)
    -- node[below=3pt] {$\lab$}
    ($(\b.south)+(0,-0.15)$);
}

% ===============================
% Inputs
% ===============================
\node[input, above=1.2cm of h1] (x1) {$\mathbf{x}_1$};
\node[input, above=1.2cm of h2] (x2) {$\mathbf{x}_2$};
\node[input, above=1.2cm of h3] (x3) {$\mathbf{x}_3$};
\node[input, above=1.2cm of h4] (x4) {$\mathbf{x}_4$};

\draw[->, gray!70] (x1) -- (h1);
\draw[->, gray!70] (x2) -- (h2);
\draw[->, gray!70] (x3) -- (h3);
\draw[->, gray!70] (x4) -- (h4);

% ===============================
% Loss
% ===============================
\node[loss, below=1.5cm of h4] (L) {$L$};
% ===== h4 -> L 前向（右偏）=====
\draw[->, purple!70, line width=1pt]
($(h4.south east)+(0.05,-0.05)$)
-- ($(L.north east)+(0.05,0)$);

% ===== L -> h4 反向（左偏）=====
\draw[backward]
($(L.north west)+(-0.05,0)$)
-- ($(h4.south west)+(-0.05,-0.05)$);

% ===============================
% Caption
% ===============================
\node[below=2.6cm of h2, font=\footnotesize] {
\color{red!70}梯度沿时间反向传播（每步乘一次 Jacobian）
};

\end{tikzpicture}
```

损失 $L$ 对 $\mathbf{h}_1$ 的梯度需要穿越所有中间隐状态：

$$
\frac{\partial L}{\partial \mathbf{h}_1} = \frac{\partial L}{\partial \mathbf{h}_4} \cdot \frac{\partial \mathbf{h}_4}{\partial \mathbf{h}_3} \cdot \frac{\partial \mathbf{h}_3}{\partial \mathbf{h}_2} \cdot \frac{\partial \mathbf{h}_2}{\partial \mathbf{h}_1}
$$

这正是 {ref}`gradient-vanishing-math` 中讨论的 **Jacobian 连乘**的序列版本——每个 $\frac{\partial \mathbf{h}_{t}}{\partial \mathbf{h}_{t-1}}$ 是一个 Jacobian 矩阵，经过 $T$ 步连乘后，梯度呈指数级衰减（或爆炸）。

```{important}
**RNN训练的根本矛盾**：要学习长程依赖，梯度必须穿越很长的路径。但路径越长，连乘的 Jacobian 越多，梯度消失越严重——网络"记不住"太远的信息。
```

(rnn-problems)=
## RNN的两个核心问题

### 问题一：长程依赖困难

从信息流动的直觉来理解：

- $\mathbf{h}_1$ 的信息要传递到 $\mathbf{h}_{100}$，需要经过 99 次 $\tanh(\mathbf{W}_h \cdot)$ 变换
- 每一次变换都是一次"有损压缩"——就像反复复印一份文件，复印100次后，原文已面目全非
- 原始信号的哪些部分被保留、哪些被丢弃，网络无法直接控制

```{admonition} 生活类比
:class: tip

**RNN的传话游戏**：100个人排成一列传一句话，第一个人说"今天天气很好"，但每个人传话时只能听到上一个人说的。到第100个人时，可能已经变成"今天有飞机"。

**注意力机制的传话游戏**：第100个人能同时听到前面99个人说的话，然后自己判断谁说的最关键。
```

### 问题二：梯度消失

{ref}`gradient-vanishing` 中讨论过，假设每层 Jacobian 的"放大倍数"为 $\gamma$，经过 $n$ 层连乘后为 $\gamma^n$。

RNN 中，这个 $n$ 不是网络深度，而是**序列长度**。考虑 $\tanh$ 的导数：

$$
\frac{d}{dx}\tanh(x) = 1 - \tanh^2(x) \in (0, 1]
$$

在大部分输入区域，$\tanh$ 的梯度远小于 1（饱和时接近 0）。假设平均 $\gamma = 0.5$：

| 序列长度 | 梯度衰减 | 能否学到依赖？ |
| -------- | -------------- | -------------- |
| 10个词 | $0.5^{10} \approx 0.001$ | 勉强可以 |
| 50个词 | $0.5^{50} \approx 8.9 \times 10^{-16}$ | 基本不可能 |
| 100个词 | $0.5^{100} \approx 7.9 \times 10^{-31}$ | 完全不可能 |

这意味着：**RNN 只能学会几十步以内的依赖关系**。更长的依赖（如段首的主题词与段尾的结论词之间的关联）在训练中几乎收不到梯度信号。

RNN 用一个隐状态同时承担"存储记忆"和"对外输出"——这个设计本身就是矛盾的。下一节 {doc}`lstm` 中，我们将看到如何通过**分离两个状态、引入三个门控**来大幅缓解梯度消失——为梯度创建一条不经过 $\tanh$ 的"高速公路"。

## 代码实践：RNN的梯度消失有多严重？

```python
import torch
import torch.nn as nn

# RNN参数: input_size=32（输入特征维度）, hidden_size=64（隐状态维度）
# 参数量 = W_h(64×64) + W_x(64×32) + b(64) + W_y(64×32:默认输出维度=输入维度)
#        = 4096 + 2048 + 64 + 2048 = 8256
rnn = nn.RNN(input_size=32, hidden_size=64, num_layers=1)

# 输入：50个时间步的序列 (seq_len=50, batch=1, features=32)
# 理论对应 {ref}`rnn-problems` 中的序列长度 vs 梯度衰减表
x = torch.randn(50, 1, 32)

# 前向传播——沿时间步串行执行（无法并行，{ref}`rnn-problems`）
#   output: (50, 1, 64) ——每一步的隐状态 h_1...h_50
#   h_n:    ( 1, 1, 64) ——最后一步的隐状态（等于 output[-1]）
output, h_n = rnn(x)

# 用最后一步的隐状态求和作为损失
# 这模拟了"基于整个序列的最终理解做预测"的典型RNN用法
loss = h_n.sum()
loss.backward()

# 查看 W_x 的梯度——它接收从第50步一路回传到第1步的信号
# 由于 {ref}`gradient-vanishing`，如果权重初始化不当或序列过长
# 靠近输入层的梯度几乎为0
print(f"W_ih grad norm: {rnn.weight_ih_l0.grad.norm():.10f}")
# 典型输出：~1e-6 或更小（序列越长，范数越小）
```

```{admonition} 本节小结
:class: note

- RNN 通过循环连接给了神经网络"记忆"——隐状态随输入逐步更新
- BPTT 是反向传播在时间展开的计算图上的应用，本质是 Jacobian 连乘
- RNN 的两个核心问题：**长程依赖困难**（信息经过多次变换后失真）和**梯度消失**（梯度指数级衰减到0）
- 这两个问题源于同一个根源：**信息必须一步步穿过循环连接**
- LSTM 缓解了梯度消失，但串行处理的根本限制仍在
```

下一节 {doc}`lstm` 中，我们将看到 LSTM 如何通过门控机制为梯度开辟一条"高速公路"——这让我们在处理序列时走得更远，但串行传递的根本矛盾仍然存在。

---

## 参考文献

```{bibliography}
:filter: docname in docnames
```
