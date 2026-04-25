(loss-functions)=
# 损失函数

## 损失函数的本质：定义优化目标

损失函数衡量模型预测与真实值之间的差异，将训练转化为**优化问题**。不同的损失函数塑造了不同的**损失曲面几何**，影响优化的难易程度。

```{tikz} 损失函数定义误差曲面
\begin{tikzpicture}
  \begin{axis}[
    view={30}{30},                % 视角
    xlabel=$\theta_1$,
    ylabel=$\theta_2$,
    zlabel=$L$,
    zlabel style={rotate=-90},    % 让L标签竖起来
    colormap/viridis,
    mesh/cols=20,
    mesh/rows=20,
    domain=0.5:3.5,
    y domain=0.5:3.5,
    xtick=\empty, ytick=\empty, ztick=\empty,
  ]
    % 损失曲面：L = 0.5*(theta1-2)^2 + 0.5*(theta2-1.5)^2 + 0.2
    \addplot3[surf, opacity=0.7, faceted color=blue!40] 
        {0.5*(x-2)^2 + 0.5*(y-1.5)^2 + 0.2};

    % 最优点（碗底）
    \addplot3[only marks, mark=*, red, mark size=3pt] coordinates {(2,1.5,0.2)};
    \node[red] at (axis cs:2.4,2.0,0.5) {最优点};

    % 梯度下降路径（画在曲面上）
    \addplot3+[only marks, mark=*, mark size=2.5pt, orange, 
           mark options={fill=orange!80}] 
    coordinates {
        (0.5, 3.5, 2.125)  % 起点
        (1.0, 2.75, 0.606) % 第1步
        (1.25, 2.375, 0.320) % 第2步
        (1.375, 2.1875, 0.252) % 第3步
        (2, 1.5, 0.2)       % 最优点
    };

% 用箭头连接这些点
    \addplot3[->, orange, very thick, samples=2] 
    coordinates {
        (0.5, 3.5, 2.125) 
        (1.0, 2.75, 0.606) 
        (1.25, 2.375, 0.320) 
        (1.375, 2.1875, 0.252) 
        (2, 1.5, 0.2)
    };
    \node[orange] at (axis cs:3.2,0.5,3.5) {梯度下降};
  \end{axis}
\end{tikzpicture}
```

**核心作用**：
- **量化误差**：将预测好坏转化为可计算的数值
- **指导优化**：通过梯度告诉模型"如何调整"（详见{ref}`gradient-descent`）
- **塑造曲面**：不同损失函数对应不同的{ref}`gradient-descent`优化难度

---

## 回归问题的损失函数

### 均方误差（MSE）

$$\text{MSE} = \frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)^2$$

**特点**：
- 对**大误差惩罚更重**（平方增长）
- 处处可导，便于梯度计算
- 对**异常值敏感**

```{tikz} MSE损失函数形状
\begin{tikzpicture}[scale=0.8]
    \draw[->] (-3,0) -- (3,0) node[right] {$\hat{y} - y$};
    \draw[->] (0,-0.5) -- (0,4) node[above] {$\text{MSE}$};
    \draw[domain=-1.7:1.7,smooth,variable=\x,blue,thick] plot ({\x},{\x*\x});
    
    \node[red, font=\small] at (3, 4) {大误差被平方放大};
    \draw[->, red, thick] (3, 3.3) -- (2.5, 2.7);
\end{tikzpicture}
```

### 平均绝对误差（MAE）

$$\text{MAE} = \frac{1}{n} \sum_{i=1}^n |y_i - \hat{y}_i|$$

**特点**：
- 对**异常值更鲁棒**（线性惩罚）
- 在零点**不可导**（可用次梯度）

### Huber损失（Smooth L1）

结合MSE和MAE的优点：

$$L_\delta(a) = \begin{cases} \frac{1}{2}a^2 & |a| \leq \delta \\ \delta(|a| - \frac{1}{2}\delta) & \text{otherwise} \end{cases}$$

**特点**：
- 小误差：MSE（平滑）
- 大误差：MAE（鲁棒）
- 常用于目标检测等任务

---

## 分类问题的损失函数

### 交叉熵损失（Cross-Entropy）

#### 从编码角度理解

想象你要传递一条消息："这是一张猫的图片"。如果只有10个可能的类别，最直接的编码方式是给每个类别分配一个编号（0-9）。但这种方式有问题：**如果模型预测"狗"的概率是80%而"猫"是20%，我们丢失了预测信心信息**。

更好的方式是使用**概率分布**作为编码。模型输出 $[0.1, 0.7, 0.05, ...]$ 表示对每个类别的信心。

**One-Hot编码**：将类别标签转换为向量，对应类别的位置为1，其余为0。

```{tikz} One-Hot编码示例
\begin{tikzpicture}[scale=0.9]
    % 类别标签
    \node[font=\bfseries] at (-3, 2) {类别};
    \node[font=\bfseries] at (3, 2) {One-Hot编码};
    
    % 猫
    \node[fill=orange!30, rounded corners, minimum width=1.5cm] at (-3, 1) {猫};
    \draw[->, thick] (-2, 1) -- (-0.5, 1);
    \node[font=\ttfamily\small] at (3, 1) {[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]};
    
    % 狗
    \node[fill=blue!30, rounded corners, minimum width=1.5cm] at (-3, 0) {狗};
    \draw[->, thick] (-2, 0) -- (-0.5, 0);
    \node[font=\ttfamily\small] at (3, 0) {[0, 1, 0, 0, 0, 0, 0, 0, 0, 0]};
    
    % 鸟
    \node[fill=green!30, rounded corners, minimum width=1.5cm] at (-3, -1) {鸟};
    \draw[->, thick] (-2, -1) -- (-0.5, -1);
    \node[font=\ttfamily\small] at (3, -1) {[0, 0, 1, 0, 0, 0, 0, 0, 0, 0]};
    
    % 标注
    \node[gray, font=\small] at (1.5, -2) {10个类别中只有1个位置是1};
\end{tikzpicture}
```

**为什么叫"One-Hot"？**
- **One**：只有一个位置是1
- **Hot**：这个位置"激活"（热），其余都是"冷"（0）

#### 交叉熵公式

交叉熵衡量**用预测分布 $Q$ 来编码真实分布 $P$ 所需的平均信息量**：

$$\text{CE}(P, Q) = -\sum_{i} P(i) \log Q(i)$$

对于分类问题（$P$ 是one-hot编码），这简化为：

$$\text{CE} = -\log(\hat{y}_{\text{正确类别}})$$

**关键性质**：
- 预测越准（概率接近1），损失越低
- 对**错误预测惩罚很大**（预测接近0时损失→∞）
- 与softmax配合，梯度形式简洁

```{tikz} 交叉熵损失：惩罚错误预测
\begin{tikzpicture}[scale=0.8]
    % 当真实标签为1时，损失随预测概率的变化
    \draw[->] (0,0) -- (5.5,0) node[right] {$\hat{y}$ (预测概率)};
    \draw[->] (0,-0.5) -- (0,4.5) node[above] {$-\log(\hat{y})$};
    
    % 损失曲线
    \draw[domain=0.05:5,smooth,variable=\x,red,thick] plot ({\x},{-ln(\x/5)});
    
    % 标注
    \fill[green!70!black] (5, 0) circle (3pt);
    \node[green!70!black, font=\small] at (5, -0.6) {预测正确 (损失=0)};
    
    \draw[<-, red, thick] (1, 1.6) -- (1.5, 2.5) node[right, red, font=\small] {预测错误，损失急剧增加};
    
    % 示例标注
    \node[blue, font=\small] at (2.5, 3.5) {真实标签 = 1};
\end{tikzpicture}
```

**为什么不用MSE做分类？**
- MSE在预测概率接近0或1时梯度很小，学习缓慢
- 交叉熵在错误预测时提供更强的梯度信号

### KL散度（Kullback-Leibler Divergence）

#### 直觉：额外的编码代价

想象你有一个完美的天气预测模型 $P$（知道每天实际下雨的概率），但你只能用另一个模型 $Q$ 来做预测。KL散度衡量的是：**因为使用了不完美的 $Q$ 而不是真实的 $P$，你需要多传输多少信息**。

$$D_{KL}(P \| Q) = \sum_{i} P(i) \log \frac{P(i)}{Q(i)}$$

**直观例子**：
- 如果 $Q = P$（预测完美），额外代价为0
- 如果预测经常出错，代价就高
- **非对称**：用 $Q$ 近似 $P$ 的代价 ≠ 用 $P$ 近似 $Q$ 的代价

#### 与交叉熵的关系

$$\underbrace{\text{CrossEntropy}(P, Q)}_{\text{总编码长度}} = \underbrace{H(P)}_{\text{最优编码长度}} + \underbrace{D_{KL}(P \| Q)}_{\text{额外代价}}$$

其中 $H(P)$ 是分布 $P$ 的**熵**（信息论中最小可能编码长度）。因为真实分布 $P$ 是固定的，$H(P)$ 是常数。因此：

> **最小化交叉熵 = 最小化KL散度 = 让预测分布尽可能接近真实分布**

#### 应用场景

- **变分自编码器（VAE）**：约束潜在变量 $z$ 的分布接近标准正态分布
- **知识蒸馏**：让小模型学习大模型的"软预测"（概率分布），而非硬标签
- **强化学习**：限制策略更新幅度，防止模型突变

---

## 损失函数选择指南

```{list-table} 损失函数选择参考
:header-rows: 1
:widths: 25 25 25 25

* - **问题类型**
  - **推荐损失**
  - **优势**
  - **注意事项**
* - 回归问题
  - MSE / MAE
  - MSE光滑可导，MAE鲁棒
  - 数据有异常值时选MAE
* - 回归（需鲁棒性）
  - Huber损失
  - 结合两者优点
  - 需要调超参数$\delta$
* - 二分类
  - 二元交叉熵
  - 梯度强、收敛快
  - 输出需sigmoid激活
* - 多分类
  - 交叉熵
  - 配合softmax效果最佳
  - PyTorch的CE包含softmax
* - 分布匹配
  - KL散度
  - 衡量分布差异
  - 注意方向性$P\|Q$
```

---

## PyTorch实现示例

```python
import torch
import torch.nn as nn

# 回归损失
mse_loss = nn.MSELoss()
mae_loss = nn.L1Loss()  # MAE
huber_loss = nn.SmoothL1Loss(beta=1.0)

# 分类损失
ce_loss = nn.CrossEntropyLoss()  # 包含softmax
bce_loss = nn.BCELoss()  # 二分类，需手动sigmoid
kl_loss = nn.KLDivLoss(reduction='batchmean')

# 使用示例
predictions = model(inputs)
loss = ce_loss(predictions, targets)  # 多分类
loss.backward()  # 计算梯度
```

---

## 总结

损失函数定义了"什么是好的预测"，将训练转化为优化问题：

1. **回归问题**：MSE光滑易优化，MAE鲁棒抗异常值
2. **分类问题**：交叉熵提供强梯度，是深度学习的主流选择
3. **分布匹配**：KL散度衡量概率分布差异，VAE和知识蒸馏的核心工具

理解损失函数后，我们将探讨{ref}`back-propagation`——如何高效计算梯度，完成误差的"信用分配"。
