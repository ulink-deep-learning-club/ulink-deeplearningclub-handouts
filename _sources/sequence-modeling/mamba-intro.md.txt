(mamba-intro)=
# Mamba 简述：对RNN思想的回归

{doc}`transformer` 告诉我们：Transformer 用 $O(n^2)$ 的代价换来了 $O(1)$ 的信息路径。这个权衡在大模型时代开始变得沉重——当序列长度达到百万级别（长视频、DNA序列、整本教科书），$n^2$ 的增长速度让 Transformer 望而却步。

**关键问题**：有没有办法在保持长程依赖能力的同时，把复杂度降回 $O(n)$？

Mamba {cite}`gu2023mamba` 给出的回答是：**有。而且答案就在 RNN 的 O(n) 骨架里——只是需要给它装上"选择性"。**

## 历史背景

Transformer 的成功让注意力成为默认选择，但 O(n²) 的代价从未消失。研究者一直在寻找"线性注意力"——能够在 O(n) 时间内实现类似效果的技术。2020年代，Albert Gu 等人提出了一系列**结构化状态空间模型（S4）** {cite}`gu2022efficiently`，展示了 SSM 在长序列建模上的潜力。

2023年，Albert Gu 和 Tri Dao 提出了 **Mamba** {cite}`gu2023mamba`，核心创新是**选择性机制**——让 SSM 的参数 $B_t, C_t, \Delta_t$ 成为输入的函数。结合硬件感知的并行实现，Mamba 在保持 O(n) 效率的同时，首次在语言建模上达到了与 Transformer 相当的水平。

```{note}
**历史意义**：Mamba 代表了序列建模的一个新方向——不是推翻 Transformer，而是在效率维度上超越它。它证明了 RNN 的骨架（O(n)、固定状态）只需"选择性"这剂药方，就能在现代硬件上重新焕发活力。
```

## 思想回归：RNN骨架从未消失

(state-space-model)=
### RNN的数学骨架

回顾 {doc}`rnn-basics` 中的 RNN 公式：

$$
\mathbf{h}_t = \tanh(\mathbf{W}_h \mathbf{h}_{t-1} + \mathbf{W}_x \mathbf{x}_t)
$$

这个公式有一个更一般的形式。考虑一个**连续的**微分方程——信号随时间平滑变化，而非一步步跳跃：

$$
\mathbf{h}'(t) = \mathbf{A} \mathbf{h}(t) + \mathbf{B} \mathbf{x}(t), \quad \mathbf{y}(t) = \mathbf{C} \mathbf{h}(t)
$$

这就是 **状态空间模型（State Space Model, SSM）** 的连续形式。其中：
- $\mathbf{h}(t) \in \mathbb{R}^d$：时刻 $t$ 的隐状态（$d$ 是状态维度，类比 RNN 的 `hidden_size`）
- $\mathbf{x}(t) \in \mathbb{R}^m$：时刻 $t$ 的输入（$m$ 是输入维度）
- $\mathbf{y}(t) \in \mathbb{R}^k$：时刻 $t$ 的输出
- $\mathbf{A} \in \mathbb{R}^{d \times d}$：状态转移矩阵——控制"旧记忆如何随时间衰减"
- $\mathbf{B} \in \mathbb{R}^{d \times m}$：输入投影矩阵——控制"新信息如何进入记忆"
- $\mathbf{C} \in \mathbb{R}^{k \times d}$：输出投影矩阵——控制"记忆如何转化为输出"

```{note}
**和 RNN 的关系**：如果去掉 $\tanh$，RNN 的 $\mathbf{h}_t = \mathbf{W}_h \mathbf{h}_{t-1} + \mathbf{W}_x \mathbf{x}_t$ 就是 SSM 的离散版本——其中 $\mathbf{A} \leftrightarrow \mathbf{W}_h$，$\mathbf{B} \leftrightarrow \mathbf{W}_x$。RNN 多了一个 $\tanh$ 来防止数值发散，而 SSM 用结构化的 $\mathbf{A}$ 矩阵来实现稳定。
```

**从连续到离散：离散化**

连续形式的 SSM 不能直接用于深度学习（我们需要处理的是离散的词序列，而非连续信号）。需要将其**离散化**——把微分方程转换为递推公式。

引入一个**步长 $\Delta$（step size）**，表示连续时间中两次采样之间的间隔。离散化后：

$$
\mathbf{A}_d = e^{\Delta \mathbf{A}} \approx \mathbf{I} + \Delta \mathbf{A}, \quad \mathbf{B}_d = (\Delta \mathbf{A})^{-1}(e^{\Delta \mathbf{A}} - \mathbf{I}) \cdot \Delta \mathbf{B} \approx \Delta \mathbf{B}
$$

```{note}
**不需要记住这些矩阵指数！** 关键是理解离散化的直觉：步长 $\Delta$ 越小，两次采样之间信号变化越小，$\mathbf{A}_d \approx \mathbf{I}$（几乎不衰减），$\mathbf{B}_d \approx \Delta \mathbf{B}$（新信息按步长比例进入）。
```

用近似后的离散参数，SSM 的递推公式变为：

$$
\mathbf{h}_t = \mathbf{A}_d \mathbf{h}_{t-1} + \mathbf{B}_d \mathbf{x}_t, \quad \mathbf{y}_t = \mathbf{C} \mathbf{h}_t
$$

这就是 SSM 的**离散形式**——和 RNN 的骨架几乎一模一样，只是 $\mathbf{A}_d, \mathbf{B}_d$ 由 $\Delta$ 和连续参数 $\mathbf{A}, \mathbf{B}$ 计算而来。

```{note}
**它和 RNN 有什么区别？** RNN 中 $\mathbf{W}_h$ 是直接学习的任意矩阵，训练时可能不稳定。SSM 中的 $\mathbf{A}$ 被约束为特殊结构（如对角矩阵），且通过 $\Delta$ 缩放后，数值上更可控——这为后面"选择性让这一切按输入动态变化"埋下了伏笔。
```

过去十年，SSM 之所以没有取代 Transformer，是因为一个致命缺陷：**$\mathbf{A}, \mathbf{B}, \mathbf{C}, \Delta$ 对所有输入都是一样的**——无论当前词是"但是"（暗示转折）还是"的"（结构助词），网络用同一套规则更新状态。这在需要**内容感知的推理**（Content-Based Reasoning）时捉襟见肘。

(selective-ssm)=
### Mamba的洞察：让参数依赖输入

Mamba 的核心创新是**选择性SSM（Selective SSM）**：让 $\mathbf{B}$、$\mathbf{C}$ 和步长 $\Delta$ 不再是固定的全局参数，而是**每个时间步根据当前输入 $\mathbf{x}_t$ 动态计算**：

$$
\mathbf{B}_t = \text{Linear}_B(\mathbf{x}_t), \quad \mathbf{C}_t = \text{Linear}_C(\mathbf{x}_t), \quad \Delta_t = \text{softplus}(\text{Linear}_\Delta(\mathbf{x}_t))
$$

其中：
- $\text{Linear}_B, \text{Linear}_C, \text{Linear}_\Delta$ 都是简单的**线性映射**——矩阵乘法加偏置（$Wx + b$），各有自己独立的可学习权重。$\text{Linear}_B$ 和 $\text{Linear}_C$ 将 $d_{\text{model}}$ 维输入投影到 $d_{\text{state}}$ 维（$d_{\text{state}}$ 是隐状态维度），$\text{Linear}_\Delta$ 投影为一个标量。
- $\text{softplus}(z) = \log(1 + e^z)$ 是一个平滑版的 ReLU（对照 {ref}`activation-functions`）：当 $z$ 很大时接近 $z$，当 $z$ 很小时接近 0。用于确保步长 $\Delta_t > 0$——状态更新不能"倒退"。

将这些输入依赖的参数代入 {ref}`state-space-model` 中的离散化递推，得到选择性的完整更新公式：

$$
\begin{aligned}
\mathbf{A}_{d,t} &= e^{\Delta_t \mathbf{A}} \approx \mathbf{I} + \Delta_t \mathbf{A} \\[4pt]
\mathbf{B}_{d,t} &\approx \Delta_t \mathbf{B}_t \\[4pt]
\mathbf{h}_t &= \mathbf{A}_{d,t} \mathbf{h}_{t-1} + \mathbf{B}_{d,t} \mathbf{x}_t \\[4pt]
\mathbf{y}_t &= \mathbf{C}_t \mathbf{h}_t
\end{aligned}
$$

```{note}
**和传统 SSM 的区别一目了然**：传统 SSM 中 $\Delta, \mathbf{B}, \mathbf{C}$ 是固定的训练参数，对所有时间步都一样。选择性 SSM 中 $\Delta_t, \mathbf{B}_t, \mathbf{C}_t$ 每一步都不同——由 $\mathbf{x}_t$ 决定。**选择性 = 参数随时间步变化，变化规则由输入内容驱动。**
```

在实际实现中（以及我们的代码中），做进一步的简化：假设 $\mathbf{A}$ 是对角矩阵（每个状态维度独立衰减），略去矩阵指数，直接用标量形式：

$$
\mathbf{h}_t = \mathbf{A} \odot \mathbf{h}_{t-1} + \Delta_t \cdot (\mathbf{B}_t \odot \mathbf{x}_t), \quad \mathbf{y}_t = \mathbf{C}_t^\top \mathbf{h}_t
$$

> 其中 $\odot$ 是逐元素乘法（{ref}`lstm-forget-gate` 中 $\mathbf{f}_t \odot \mathbf{c}_{t-1}$ 的同款操作）。$\mathbf{A} \in \mathbb{R}^{d_{\text{state}}}$ 是衰减率向量（每个维度独立衰减），$\mathbf{B}_t, \mathbf{C}_t \in \mathbb{R}^{d_{\text{state}}}$ 是选择性输入/输出门。注意这里 $\mathbf{B}_t, \mathbf{C}_t$ 是向量（而非矩阵）——这是对角 SSM 的简化，在教学中足够传达核心思想。

这个简化版本把公式变得极其清晰：**状态更新就是衰减率 $\mathbf{A}$ × 旧记忆 + 步长 $\Delta_t$ × 选择性写入的 $\mathbf{B}_t \odot \mathbf{x}_t$**。

#### 为什么"选择性"是关键？

传统 SSM 失败的原因可以用一个例子说明。考虑处理这句话：

> "今天天气很好，**所以**我决定去公园散步。"

当网络读到"所以"时，它需要知道"所以"是一个因果连接词，暗示前面的"天气很好"是后面"散步"的原因。但传统 SSM 对所有词都使用**同一套** $\mathbf{A}, \mathbf{B}, \mathbf{C}$——它无法区分"所以"（需要触发因果推理）和"的"（结构助词，可以忽略）。

选择性机制让网络能够**根据当前输入的内容，动态调整状态更新的策略**。

#### 逐步拆解三个选择性参数

**参数一：$\Delta_t$（步长）——"当前输入有多重要？"**

| 符号 | 含义 | 维度 | 直觉理解 |
|------|------|------|---------|
| $\text{Linear}_\Delta$ | 一个可学习的线性映射 | $\mathbb{R}^{d} \to \mathbb{R}$ | 学到"什么样的输入是重要的、需要大步伐更新" |
| $\text{softplus}$ | $\log(1 + e^x)$，输出始终为正 | 标量 | 确保步长为正，且平滑 |

$\Delta_t$ 控制的是**状态更新的"步幅"**。回顾状态更新公式：

$$
\mathbf{h}_t = \mathbf{A} \mathbf{h}_{t-1} + \Delta_t \cdot (\mathbf{B}_t \mathbf{x}_t)
$$

- $\Delta_t$ 大 → 网络认为当前输入很重要，大步更新状态（类似于 LSTM 中输入门大开）
- $\Delta_t$ 小 → 网络认为当前输入不重要，几乎不更新状态（类似于 LSTM 中输入门紧闭，遗忘门全开）

```{note}
**直觉**：读"的"时，$\Delta_t$ 可能很小——"保持当前状态，这个字不重要"。读"所以"时，$\Delta_t$ 可能很大——"注意！这里有关键转折，需要更新理解"。

这种机制让 Mamba 自动学会了**对关键信息"聚焦"、对冗余信息"忽略"**，而无需显式设计门控逻辑。
```

**参数二：$\mathbf{B}_t$（输入投影）——"新信息的哪些维度值得记？"**

| 符号 | 含义 | 维度 | 直觉理解 |
|------|------|------|---------|
| $\text{Linear}_B$ | 一个可学习的线性映射 | $\mathbb{R}^{d} \to \mathbb{R}^{d_{\text{state}}}$ | 学到"如何从当前输入中提取值得记录的信息" |
| $\mathbf{B}_t$ | 输入依赖的投影向量 | $\mathbb{R}^{d_{\text{state}}}$ | 决定输入 $\mathbf{x}_t$ 的哪些方向应该进入隐状态 |

在传统 SSM 中，$\mathbf{B}$ 是一个固定矩阵——所有输入用同一种方式投影。选择性 SSM 中，$\mathbf{B}_t$ 随输入变化——"所以"和"的"产生不同的 $\mathbf{B}_t$，"所以"的投影可能强调"转折/因果"相关维度，而"的"的投影可能接近零向量（"没什么值得记录的"）。

**参数三：$\mathbf{C}_t$（输出投影）——"记忆中的哪些部分现在该说出来？"**

| 符号 | 含义 | 维度 | 直觉理解 |
|------|------|------|---------|
| $\text{Linear}_C$ | 一个可学习的线性映射 | $\mathbb{R}^{d} \to \mathbb{R}^{d_{\text{state}}}$ | 学到"基于当前输入，哪些记忆应该被输出" |
| $\mathbf{C}_t$ | 输入依赖的输出投影 | $\mathbb{R}^{d_{\text{state}}}$ | 决定隐状态 $\mathbf{h}_t$ 的哪些维度暴露给输出 $\mathbf{y}_t$ |

回忆输出公式：$\mathbf{y}_t = \mathbf{C}_t \mathbf{h}_t$。$\mathbf{C}_t$ 根据当前输入决定输出什么——类似于 LSTM 的输出门。当读到一个问号时，$\mathbf{C}_t$ 可能重点暴露关于"疑问"的维度；当读取动词时，$\mathbf{C}_t$ 可能暴露关于"动作"的维度。

#### 三个参数的协同：一个具体例子

以"我昨天买的苹果很好吃"为例，追踪 Mamba 如何处理：

| 当前词 | $\Delta_t$（步长） | $\mathbf{B}_t$（写入什么） | $\mathbf{C}_t$（输出什么） | 状态发生了什么 |
|--------|-------------------|--------------------------|--------------------------|---------------|
| "我" | 中 | 记录"主语=我"的方向 | 输出"我"的语义 | 状态初始化 |
| "昨天" | 小 | 几乎为零 | 几乎为零 | 时间状语，状态几乎不变 |
| "买" | 大 | 重点记录"动作=购买" | 输出与"买"相关的语义 | 状态捕捉关键动词 |
| "的" | 极小 | 接近零 | 接近零 | 虚词，状态保持不变 |
| "苹果" | 大 | 记录"宾语=苹果" | 输出"苹果"的语义 | 状态补充宾语信息 |
| "很" | 中 | 记录"程度修饰" | 输出程度信息 | 状态微调 |
| "好吃" | 大 | 记录"评价=正面" | 重点输出评价信息 | 状态完成语义构建 |

```{note}
这个表展示了选择性的精髓：**每个词的 $\Delta_t, \mathbf{B}_t, \mathbf{C}_t$ 都不同**。虚词（"的"）获得极小 $\Delta$，几乎不影响状态；内容词（"买"、"苹果"、"好吃"）获得大 $\Delta$，深刻影响状态。网络通过大量训练学会了"什么词值得关注"——这是一个完全端到端学出来的能力。
```

```{admonition} 直觉：选择性 = 动态门控
:class: tip

把 SSM 想象成一个"信息滤网"：
- **普通SSM**：滤网的孔径是固定的——不管流过来的是沙子还是石块，都用同样的网眼过滤。该记住的记不住，该忘的忘不掉。
- **选择性SSM（Mamba）**：滤网根据当前流入的内容**动态调整**孔径。遇到重要信息（如段首的主题词），网眼缩小，精细保留；遇到不重要信息（如虚词），网眼放大，快速丢弃。

这种"基于输入内容决定保留或丢弃什么"的思想，本质上与 {doc}`lstm` 的门控机制 {cite}`hochreiter1997long` 一脉相承——遗忘门根据当前上下文决定丢弃旧记忆中的哪些内容，选择性SSM则是让整个状态更新规则都依赖输入。区别在于：LSTM 的门控是在固定规则框架内的微调（门控值变了，但 $W_f, W_i, W_o, W_c$ 本身是固定的），而选择性 SSM 让 $B$ 和 $C$ 本身就是输入的函数——**门控的程度更深、更彻底**。
```

## Mamba vs RNN vs Transformer：三者的位置

```{list-table} 序列建模三阶段
:header-rows: 1

* - 对比维度
  - RNN / LSTM
  - Transformer
  - Mamba / 选择性SSM
* - 灵感来源
  - 大脑的时序处理
  - 全局信息检索
  - **回归RNN骨架 + Transformer的"选择性"精神**
* - 时间复杂度
  - O(n) 串行
  - O(n²) 并行
  - O(n) 可并行训练
* - 长程依赖
  - 弱（梯度消失）
  - 强（O(1)信息路径）
  - 强（选择性保留关键信息）
* - 信息传递
  - $h_t \to h_{t+1}$ 链式
  - 所有位置全连接
  - $h_t \to h_{t+1}$ 链式 + 选择机制
* - 推理速度
  - 快（固定大小状态）
  - 慢（需要整个K/V缓存）
  - **快**（固定大小状态，5× Transformer 吞吐量 {cite}`gu2023mamba`）
```

```{mermaid}
graph TD
    A[RNN 1986-1997<br/>串行, O n<br/>但长程依赖差] --> B[LSTM 1997<br/>门控缓解梯度消失<br/>但仍是串行]
    B --> C[注意力 2014<br/>全局连接<br/>O 1 信息路径]
    C --> D[Transformer 2017<br/>纯注意力<br/>O n² 代价]
    D --> E[Mamba 2023<br/>回归RNN骨架<br/>选择性SSM, O n]
    E -.->|思想螺旋上升| A
```

```{note}
**Mamba 的历史意义**：35年后，我们回到了 RNN 的起点——一个通过固定大小隐状态（$O(1)$ 内存）处理序列的架构——但这一次，隐状态的更新规则不再是固定的，而是根据输入动态调整。选择性机制解决了 RNN 的长程依赖问题，而线性骨架保留了 $O(n)$ 的效率。

这就像一个思想走了很远的路，最后在更高的层次上回到了原点——**螺旋上升，而非简单重复**。
```

## 代码实践：选择性SSM的核心逻辑

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimplifiedSelectiveSSM(nn.Module):
    """
    简化的选择性SSM——展示Mamba的核心思想

    理论对应 {ref}`selective-ssm` 中"让参数依赖输入"的关键洞察
    不是完整Mamba实现，而是教学用的核心逻辑提取

    d_model: 输入特征维度（如256）
    d_state: 隐状态维度（如16）——相比d_model较小，体现了"压缩"的设计哲学
    """
    def __init__(self, d_model=256, d_state=16):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state

        # 状态转移矩阵A——这是SSM中唯一个不依赖输入的参数
        # A控制"旧记忆如何随时间衰减"，通常初始化为接近1的值（缓慢衰减）
        # 形状：(d_state,) — 对角形式，每个维度独立衰减
        self.A = nn.Parameter(torch.ones(d_state))

        # x_proj: 从输入x生成三个选择性参数 B_t, C_t, Delta_t
        # 输出维度 = d_state (B_t) + d_state (C_t) + 1 (Delta_t)
        # 这是选择性的核心——所有参数都是x的函数（{ref}`selective-ssm`）
        self.x_proj = nn.Linear(d_model, d_state * 2 + 1)

    def forward(self, x):
        """
        x: (batch, seq_len, d_model)
        返回: (batch, seq_len, 1)

        核心公式（{ref}`state-space-model`）：
            h_t = A * h_{t-1} + B_t * x_t
            y_t = C_t * h_t
        其中 B_t, C_t 是输入依赖的（选择性），A是固定的
        """
        B, T, D = x.shape

        # 步骤1: 从输入计算三个选择性参数
        proj = self.x_proj(x)  # (B, T, 2*d_state + 1)

        # B_t: 输入投影——"当前输入的哪些方向值得写入状态"（{ref}`selective-ssm` 参数二）
        # 形状: (B, T, d_state) ——每个时间步、每个batch有独立的B_t
        B_t = proj[..., :self.d_state]

        # C_t: 输出投影——"状态的哪些维度应该被读出"（{ref}`selective-ssm` 参数三）
        # 形状: (B, T, d_state) ——每个时间步独立决定输出
        C_t = proj[..., self.d_state:2*self.d_state]

        # Delta_t: 步长——"当前输入有多重要"（{ref}`selective-ssm` 参数一）
        # softplus 确保步长为正（log(1+e^x) > 0）
        # 大Delta → 大步更新状态（关键信息）；小Delta → 几乎不更新（冗余信息）
        Delta = F.softplus(proj[..., -1:])  # (B, T, 1)

        # 步骤2: 选择性扫描（概念上串行，实际Mamba用硬件感知并行算法）
        h = torch.zeros(B, self.d_state)  # h_0: 初始状态
        outputs = []

        for t in range(T):
            # 选择性地将当前输入映射到状态空间
            # B_t[t] 决定了当前输入在状态空间的"写入方向"
            # x[..., :d_state] 取d_state维的输入子空间（简化起见）
            input_contribution = B_t[:, t, :] * x[:, t, :self.d_state]

            # 状态更新: h_t = A * h_{t-1} + Delta * (B_t * x_t)
            #   A * h_{t-1}: 旧记忆按A的衰减率保留
            #   Delta * input: 新信息按当前步长写入
            # 当Delta→0时，h_t ≈ A * h_{t-1}（状态几乎不变——"忽略当前输入"）
            h = self.A * h + Delta[:, t, :] * input_contribution

            # 输出: y_t = C_t * h_t（C_t决定暴露哪些维度）
            y = (C_t[:, t, :] * h).sum(dim=-1, keepdim=True)
            outputs.append(y)

        return torch.stack(outputs, dim=1)  # (B, T, 1)
```

```{admonition} 本节小结
:class: note

- 状态空间模型（SSM）与 RNN 共享相同的数学骨架：$h_t = A h_{t-1} + B x_t$
- 传统 SSM 的 $A, B, C$ 是固定的——无法根据输入内容调整记忆策略
- Mamba 的关键创新：让 $B, C, \Delta$ 依赖输入 $\mathbf{x}_t$ ——**选择性**机制
- 这个选择性让 SSM 拥有了"基于内容决定记住什么、忘记什么"的能力
- 结果：$O(n)$ 复杂度 + 与 Transformer 相当的长程依赖能力——对 RNN 思想的螺旋式回归
```

```{admonition} 为什么不继续深入？
:class: caution

Mamba 的完整实现涉及更深层的数学工具：**矩阵指数** $e^{\Delta \mathbf{A}}$ 的精确计算需要线性代数谱分解，**结构化状态空间**的初始化依赖控制理论中的 HIPPO 矩阵，**硬件感知并行算法**涉及 GPU 内存层次和 kernel fusion 的底层优化。

这些工具已经远超本教程的目标——面向高中生的入门介绍。本章的目标是让你理解序列建模的**思想演化路径**和 Mamba 的**核心直觉**（选择性 = 让参数依赖输入），而非掌握其完整推导。如果你对细节感兴趣，可以阅读原论文 {cite}`gu2023mamba` 及其引用的 S4 系列工作 {cite}`gu2022efficiently`。
```

Mamba 的思想——用固定大小的隐状态高效处理长序列，同时通过选择性机制解决长程依赖——代表了序列建模的一个新方向。下一节 {doc}`the-end` 中，我们将系统对比这三种架构，并展望未来。

---

```{only} not pdf

~~~{rubric} 参考文献
:heading-level: 2
~~~

~~~{bibliography}
:filter: docname in docnames
~~~
```
