(task-formulations)=
# 学习任务的形式

{doc}`computational-graph`中介绍的计算图描述了数据如何在模型中流动。但数据流向什么样的目标？不同类型的"规律"需要不同的数学建模方式。

## 机器学习在找什么规律？

机器学习的本质是从数据中**发现规律**，然后用这些规律做预测 {cite}`bishop2006pattern`。但"规律"有不同形式：

| 规律类型 | 核心问题 | 例子 |
|---------|---------|------|
| **类别边界** | 这类事物有什么共同特征？ | 识别猫和狗 |
| **数值映射** | 输入输出是什么关系？ | 根据面积预测房价 |
| **序列依赖** | 下一步应该是什么？ | 续写句子 |

这三种规律对应三种任务类型：**分类**、**回归**和**自回归**。它们共享相同的{ref}`computational-graph`计算机制，但目标的数学形式截然不同。

---

## 分类任务：发现类别边界的规律

想象你在教小孩认动物。你不会给他看一张 checklist，而是指着图片说："这个有四条腿、有毛、会汪汪叫，所以是狗。"

**分类的本质**：找到区分不同类别的**边界**——边界这边是A类，那边是B类。

在特征空间中，每个样本是一个点，分类就是画一条**决策边界**把空间切开：

```{tikz} 分类任务的决策边界
\begin{tikzpicture}[scale=0.9]
    % 坐标轴
    \draw[->] (-0.5,0) -- (5,0) node[right] {$x_1$};
    \draw[->] (0,-0.5) -- (0,5) node[above] {$x_2$};
    
    % 类别A的点（蓝色）
    \foreach \x/\y in {0.5/1, 1/0.5, 0.8/1.2, 1.5/0.8}
        \fill[blue!60] (\x, \y) circle (3pt);
    
    % 类别B的点（红色）
    \foreach \x/\y in {3.5/4, 4/3.5, 3.8/3.8, 4.2/3.2}
        \fill[red!60] (\x, \y) circle (3pt);
    
    % 决策边界（曲线）
    \draw[thick, green!50!black, dashed] plot[smooth] coordinates {(0,3) (1.5,2.5) (3,2) (5,1.5)};
    \node[green!50!black] at (4.5, 2.5) {决策边界};
    
    % 标注
    \node[blue] at (1, -0.8) {类别A};
    \node[red] at (4, 4.5) {类别B};
\end{tikzpicture}
```

**关键洞察**：
- 边界这边聚集着同类样本
- 边界是"模糊"的（软边界），不是一刀切的硬线
- 新样本落在哪边，就预测为哪类

**数学形式**：分类模型输出类别概率分布 $p(y=k|x)$，其中 $k \in \{1, 2, ..., K\}$。决策规则是选择概率最大的类别：

$$\hat{y} = \arg\max_k \, p(y=k|x)$$

决策边界是两个类别概率相等的位置：$p(y=i|x) = p(y=j|x)$。

{ref}`activation-functions`中讨论的非线性激活函数，正是为了让神经网络能够学习**非线性的决策边界**。没有非线性激活，无论多少层网络都只能学习线性边界。

---

## 回归任务：发现数值映射的规律

想象你在做物理实验：测量不同拉力下弹簧的伸长量。你发现"拉力越大，弹簧伸得越长"。这不是分类问题，因为你不是把拉力分成"大"或"小"两类，而是要找出**具体的数值关系**。

**回归的本质**：学习一个**函数映射**，输入一个数值，输出另一个数值。

最简单的回归是**线性回归**：假设输入和输出之间是直线关系。类比：你收集了若干房屋的面积和价格数据，在纸上画散点图，然后画一条最"拟合"这些点的直线。

**数学形式**：对于单变量线性回归

$$\hat{y} = wx + b$$

- $x$：输入特征（如房屋面积）
- $w$：权重（斜率，表示"每平方米值多少钱"）
- $b$：偏置（截距，表示"基础价格"）
- $\hat{y}$：预测值（预测房价）

**几何意义**

在二维平面上，线性回归就是找一条直线，让所有数据点到直线的**垂直距离之和最小**。

```{tikz} 线性回归的几何意义
\begin{tikzpicture}[scale=0.9]
    % 坐标轴
    \draw[->] (-0.5,0) -- (6,0) node[right] {$x$ (面积)};
    \draw[->] (0,-0.5) -- (0,5) node[above] {$y$ (价格)};
    
    % 数据点
    \foreach \x/\y in {0.5/0.8, 1/1.5, 1.5/1.7, 2/2.5, 2.5/2.7, 3/3.5, 3.5/3.4, 4/4.2, 4.5/4.0, 5/4.8}
        \fill[blue!60] (\x, \y) circle (2.5pt);
    
    % 拟合直线 y = 0.9x + 0.5
    \draw[thick, red] (0, 0.5) -- (5.5, 5.45);
    \node[red] at (6.5, 3.5) {$\hat{y} = wx + b$};
    
    % 误差线示例
    \draw[dashed, gray] (2, 2.5) -- (2, 2.3);
    \draw[dashed, gray] (3, 3.5) -- (3, 3.2);
    \draw[dashed, gray] (4, 4.2) -- (4, 4.1);
    \node[gray, font=\small] at (4.5, 1.5) {误差 = $|y - \hat{y}|$};
\end{tikzpicture}
```

**扩展到多元**：现实中的房价不只由面积决定，还取决于位置、房龄、楼层等。多元线性回归：

$$\hat{y} = w_1x_1 + w_2x_2 + ... + w_dx_d + b = \mathbf{w}^T\mathbf{x} + b$$

这不再是二维平面上的直线，而是**高维空间中的超平面**。

**非线性回归**

线性回归假设关系是直线，但很多规律是曲线。神经网络通过堆叠非线性层，可以学习**任意复杂的函数映射**。

```{tikz} 从线性到非线性回归
\begin{tikzpicture}[scale=0.7]
    % 左侧：线性拟合
    \begin{scope}[shift={(0,0)}]
        \draw[->] (-0.3,0) -- (3,0) node[right] {$x$};
        \draw[->] (0,-0.3) -- (0,3) node[above] {$y$};
        \foreach \x/\y in {0.3/0.5, 0.8/1.2, 1.2/1.8, 1.8/2.0, 2.5/2.8}
            \fill[blue!60] (\x, \y) circle (2pt);
        \draw[thick, red] (0, 0.2) -- (2.8, 3);
        \node at (1.5, -0.8) {线性拟合};
    \end{scope}
    
    % 右侧：非线性拟合
    \begin{scope}[shift={(5,0)}]
        \draw[->] (-0.3,0) -- (3,0) node[right] {$x$};
        \draw[->] (0,-0.3) -- (0,3) node[above] {$y$};
        \foreach \x/\y in {0.2/0.3, 0.6/1.5, 1.0/2.5, 1.4/2.8, 2.1/2.2, 3.0/1.0, 2.9/0.5}
            \fill[blue!60] (\x, \y) circle (2pt);
        \draw[thick, red, domain=0:pi, smooth] plot (\x, {0.5 + 2.5*sin(\x r)});
        \node at (1.5, -0.8) {非线性拟合（神经网络）};
    \end{scope}
\end{tikzpicture}
```

### 回归 vs 分类的本质区别

| 维度 | 分类 | 回归 |
|------|------|------|
| **规律形式** | 类别边界（划分区域） | 函数映射（输入→输出） |
| **输出空间** | 离散有限集 $\{1,...,K\}$ | 连续无穷集 $\mathbb{R}$ |
| **预测内容** | "这是哪一类" | "数值是多少" |
| **误差度量** | 分类错误率 | 数值偏差 $\|y - \hat{y}\|$ |
| **几何形象** | 空间中的分界曲面 | 空间中的拟合曲面 |

**一句话总结**：分类是"切蛋糕"（划分空间），回归是"画曲线"（拟合关系）。

---

## 自回归任务：发现序列依赖的规律

想象你在听故事："从前有座山，山里有座..."你自然会接"庙"。为什么不是"城堡"或"学校"？因为你学过，"山里有座"后面最常接的是"庙"。

**自回归的本质**：基于**已生成的内容**预测**下一个内容**，一步一步"续写"。

**概率分解**：序列的联合概率可以用链式法则分解。对于两个事件，$p(A,B) = p(A) \cdot p(B|A)$——先发生A，再在A发生的条件下发生B。扩展到序列：

$$p(y_1, y_2, y_3) = p(y_1) \cdot p(y_2|y_1) \cdot p(y_3|y_1, y_2)$$

一般形式：

$$p(y_1, y_2, ..., y_T) = \prod_{t=1}^{T} p(y_t | y_{<t})$$

其中 $y_{<t}$ 表示 $y_1$ 到 $y_{t-1}$ 的所有历史内容。

**为什么这样分解有用？** 想象你要生成句子"猫坐在垫子上"。直接学习整个句子的概率很难，但分解后每步都是简单的分类：
- $p(\text{猫}|\text{<s>})$：句首是"猫"的概率
- $p(\text{坐}|\text{<s>猫})$：看到"猫"后接"坐"的概率
- $p(\text{在}|\text{<s>猫坐})$：看到"猫坐"后接"在"的概率

**因果性约束**：自回归只能依赖**过去**，不能"偷看"未来：

$$p(y_t | y_1, ..., y_{t-1}, \underbrace{y_{t+1}, ...}_{\text{不能用}})$$

这就像写作文——你不能用还没写的句子来决定现在写什么。这种"只看左边"的特性称为**因果掩码**。

**训练 vs 推理**：
- **训练时**：已知完整序列"猫坐在垫子上"，模型同时学习每个位置的条件概率。可以并行计算所有位置的损失。
- **推理时**：从<start>开始，生成第一个词，然后用这个词作为输入生成第二个词，依此类推。必须**串行**进行。

**序列生成的树状结构**：

```{tikz} 自回归的序列生成
\begin{tikzpicture}[scale=0.8]
    % 起始节点
    \node[draw, rounded corners, fill=blue!20] (start) at (0, 0) {\small 从前};
    
    % 第一层
    \node[draw, rounded corners, fill=green!20] (a1) at (2, 1.5) {\small 有座};
    \node[draw, rounded corners, fill=gray!20] (a2) at (2, 0) {\small 有个};
    \node[draw, rounded corners, fill=gray!20] (a3) at (2, -1.5) {\small 有个人};
    
    % 第二层
    \node[draw, rounded corners, fill=green!20] (b1) at (4, 1.5) {\small 山};
    \node[draw, rounded corners, fill=gray!20] (b2) at (4, 0.5) {\small 庙};
    
    % 第三层
    \node[draw, rounded corners, fill=green!20] (c1) at (6, 1.5) {\small 山里};
    
    % 第四层
    \node[draw, rounded corners, fill=green!20] (d1) at (8, 1.5) {\small 有座};
    
    % 第五层
    \node[draw, rounded corners, fill=orange!30] (e1) at (10, 1.5) {\small 庙};
    
    % 连接
    \draw[->, thick] (start) -- (a1);
    \draw[->, gray] (start) -- (a2);
    \draw[->, gray] (start) -- (a3);
    
    \draw[->, thick] (a1) -- (b1);
    \draw[->, gray] (a1) -- (b2);
    
    \draw[->, thick] (b1) -- (c1);
    \draw[->, thick] (c1) -- (d1);
    \draw[->, thick] (d1) -- (e1);
    
    % 标注
    \node[green!50!black, font=\small] at (5, 3) {实际生成的序列};
    \draw[->, green!50!black] (5, 2.7) -- (3, 1.8);
    
    \node[gray, font=\small] at (2, -2.5) {灰色：其他可能的生成路径};
\end{tikzpicture}
```

在每一步，模型输出一个概率分布，然后采样（或选择最大概率）得到下一个词。

**与分类的联系**：自回归的每一步本质上都是分类——从词汇表中选择一个词。区别在于：普通分类的每次预测独立进行；而自回归的每次预测**依赖所有历史选择**，是"**有记忆的分类**"。评估时也不只看单步准确率，而是看整个序列的质量（用困惑度衡量）。

---

## 三种任务的统一与差异

三类任务都遵循相同的计算框架（见{ref}`computational-graph`）：输入 → 变换 → 输出 → 损失计算。差异在于**输出层的设计**和**损失函数的选择**。

| 任务 | 规律形式 | 输出层 | 典型损失 | 几何直观 |
|------|---------|--------|---------|---------|
| **分类** | 类别边界 | Softmax | 交叉熵 | 空间区域划分 |
| **回归** | 函数映射 | 线性 | MSE/MAE | 曲面拟合 |
| **自回归** | 条件概率链 | Softmax（每步） | 累计交叉熵 | 树状展开 |

同一输入在不同任务中的处理流程：

```{tikz} 三种任务的处理流程对比
\begin{tikzpicture}[scale=0.75]
    % 输入
    \node[draw, rounded corners, fill=blue!20, minimum width=2cm] (input) at (0, 0) {输入 $x$};
    
    % 三个分支
    \node[draw, rounded corners, fill=orange!20, minimum width=2.5cm] (class) at (4, 2.5) {分类网络};
    \node[draw, rounded corners, fill=orange!20, minimum width=2.5cm] (regr) at (4, 0) {回归网络};
    \node[draw, rounded corners, fill=orange!20, minimum width=2.5cm] (auto) at (4, -2.5) {自回归网络};
    
    % 输出
    \node[draw, rounded corners, fill=green!20, minimum width=3cm, align=center] (out1) at (8.5, 2.5) {"这是猫"\\（离散标签）};
    \node[draw, rounded corners, fill=green!20, minimum width=3cm, align=center] (out2) at (8.5, 0) {24.5\\（连续数值）};
    \node[draw, rounded corners, fill=green!20, minimum width=3cm, align=center] (out3) at (8.5, -2.5) {"从前有座..."\\（序列）};
    
    % 连接
    \draw[->, thick] (input) -- (class);
    \draw[->, thick] (input) -- (regr);
    \draw[->, thick] (input) -- (auto);
    
    \draw[->, thick] (class) -- (out1);
    \draw[->, thick] (regr) -- (out2);
    \draw[->, thick] (auto) -- (out3);
    
    % 标注
    \node[font=\small] at (2, 3.2) {找边界};
    \node[font=\small] at (2, 0.7) {找映射};
    \node[font=\small] at (2, -1.8) {找依赖};
\end{tikzpicture}
```

---

## 总结

| 概念 | 实践 | 联系 |
|------|------|------|
| **分类** | 发现类别边界 | 用{ref}`activation-functions`学习非线性边界 |
| **回归** | 学习函数映射 | 线性回归是基础，神经网络扩展非线性 |
| **自回归** | 建模序列依赖 | 每步是分类，整体是序列 |

三类任务的区别不在于网络的内部结构（都可以用相同的层堆叠），而在于：
1. **输出层的设计**（Softmax vs 线性 vs 每步Softmax）
2. **损失函数的选择**（交叉熵 vs MSE vs 累计交叉熵）
3. **评估的方式**（准确率 vs 误差大小 vs 序列质量）

理解了三种任务的数学形式后，{doc}`loss-functions`将详细介绍如何用损失函数**量化**预测的好坏——分类用交叉熵，回归用MSE，自回归用累计交叉熵。这些损失函数塑造了不同的{ref}`gradient-descent`优化曲面。

---

## 参考文献

```{bibliography}
:filter: docname in docnames
```

