(gradient-descent)=
# 梯度下降与优化算法

## 沿着最陡方向下山

想象你在山上迷路了，想要下到山谷：

- **当前位置**：参数当前值 $\theta_t$
- **最陡下降方向**：负梯度方向（你脚下最陡的下坡方向）
- **步长**：学习率 $\eta$（你每一步迈多大）
- **目标**：到达山谷最低点（损失 $L$ 最小）

神经网络训练转化为优化问题后，我们需要找到损失函数的**最小值点**。梯度下降（Gradient Descent）就是解决这个问题的迭代算法。

**核心思想**：在当前位置，沿着**梯度下降方向**（损失减小最快的方向）迈出一小步。

$$
\theta_{t+1} = \theta_t - \eta \cdot \nabla_\theta J(\theta)
$$

其中 $\eta$ 是**学习率**（步长），$\nabla_\theta J(\theta)$ 是损失函数对参数的梯度。

```{tikz} 梯度下降：从初始点逐步逼近最小值
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

### 学习率的重要性

学习率的选择至关重要：

| 学习率 | 效果 | 类比 |
|--------|------|------|
| 太小 | 收敛极慢 | 婴儿学步 |
| 合适 | 快速收敛 | 正常行走 |
| 太大 | 震荡/发散 | 大跳，可能跳过山谷 |

```{tikz} 学习率过大导致震荡
\begin{tikzpicture}[scale=1.0]
    \draw[->] (-0.5,0) -- (5,0) node[right] {$\theta$};
    \draw[->] (0,-0.5) -- (0,4) node[above] {$J(\theta)$};
    
    % Parabola
    \draw[thick, blue] plot[domain=0:4.5, samples=100] (\x, {0.3*(\x-2)^2 + 0.5});
    \fill[red] (2, 0.5) circle (2pt) node[below] {最小值};
    
    % 震荡路径
    \fill[purple] (0.5, 1.175) circle (2pt);
    \draw[->, thick, purple] (0.5, 1.175) -- (3.5, 1.175);
    \fill[purple] (3.5, 1.175) circle (2pt);
    \draw[->, thick, purple] (3.5, 1.175) -- (0.5, 1.175);
    
    \node[purple] at (2, 2.5) {学习率过大：在山谷两侧震荡};
\end{tikzpicture}
```

---

## 梯度下降的变体

在{ref}`gradient-descent`中，我们根据{ref}`back-propagation`计算的梯度更新参数。根据使用多少数据计算梯度，有三种变体：

### 批量梯度下降（Batch GD）

使用**全部数据**计算梯度：

$$
\theta_{t+1} = \theta_t - \eta \cdot \frac{1}{N} \sum_{i=1}^N \nabla_\theta J(\theta; x_i, y_i)
$$

**特点**：

- 梯度估计准确，收敛稳定
- 每次迭代计算量大，不适合大数据集

### 随机梯度下降（SGD）

每次使用**一个样本**计算梯度：

$$
\theta_{t+1} = \theta_t - \eta \cdot \nabla_\theta J(\theta; x_i, y_i)
$$

**特点**：

- 计算快，适合在线学习
- 梯度有噪声，收敛不稳定
- **噪声可能帮助跳出局部最优**

### 小批量梯度下降（Mini-batch GD）

使用**一小批样本**（通常32-256个）计算梯度：

$$
\theta_{t+1} = \theta_t - \eta \cdot \frac{1}{m} \sum_{i=1}^m \nabla_\theta J(\theta; x_i, y_i)
$$

**特点**：

- **平衡计算效率和稳定性**
- 可以利用GPU并行计算
- **深度学习中最常用**

---

## 高级优化算法

### Loss Landscape：{ref}`loss-functions`塑造的优化地形

**Loss Landscape**（损失景观）描述了{ref}`loss-functions`在参数空间中的形状。想象一个高维的山地地形：山谷代表损失低的区域，山峰代表损失高的区域。

在深度学习中，这个景观通常是**非凸**的，包含：

- **局部最优**：局部山谷（浅坑）
- **全局最优**：最深的山谷（最优解）
- **鞍点**：马鞍形状的点（高维空间中非常常见）

```{tikz} Loss Landscape：局部最优 vs 全局最优
\begin{tikzpicture}
  \begin{axis}[
    view={55}{40},
    xlabel=$\theta_1$,
    ylabel=$\theta_2$,
    zlabel=$J(\theta)$,
    zlabel style={rotate=-90},
    colormap/viridis,
    mesh/cols=60,
    mesh/rows=60,
    domain=-4:4,
    y domain=-4:4,
    xtick=\empty, ytick=\empty, ztick=\empty,
    width=11cm,
    height=8cm,
  ]
    % 损失景观：一个浅局部最优 + 一个深全局最优
    \addplot3[surf, opacity=0.75, faceted color=blue!20] 
        {-0.8*exp(-((x-2)^2 + (y-1.5)^2)/0.8) - 2.5*exp(-((x+1.5)^2 + (y+1)^2)/2) + 0.05*(x^2 + y^2)};
    
    % 局部最优 (较浅，损失值较高)
    \addplot3[only marks, mark=*, orange, mark size=5pt] coordinates {(2, 1.5, -0.8)};
    
    % 全局最优 (更深，损失值更低)
    \addplot3[only marks, mark=*, red, mark size=5pt] coordinates {(-1.5, -1, -2.5)};
    
    % 鞍点 (两个山谷之间的山口)
    \addplot3[only marks, mark=*, purple, mark size=4pt] coordinates {(0.3, 0.2, -0.2)};
    
    % 标注
    \node[orange, font=\small, fill=white, inner sep=2pt] at (axis cs:2.5, 2, -1.5) {局部最优};
    \node[red, font=\small, fill=white, inner sep=2pt] at (axis cs:-2, -0.5, -3.7) {全局最优};
    \node[purple, font=\small] at (axis cs:1.2, 0.5, 0.3) {鞍点};
  \end{axis}
\end{tikzpicture}
```

**关键观察**：

- 全局最优比局部最优**明显更深**（红色点远低于橙色点）
- 两个最优之间由**鞍点**（紫色）分隔
- 优化算法需要"翻过"鞍点才能从局部最优到达全局最优

**鞍点的详细展示**：

鞍点像一个马鞍——沿一个方向是极小值，沿另一个方向是极大值：

```{tikz} 鞍点：马鞍形状
\begin{tikzpicture}
  \begin{axis}[
    view={30}{20},
    xlabel=$\theta_1$,
    ylabel=$\theta_2$,
    zlabel=$J(\theta)$,
    zlabel style={rotate=-90},
    colormap/cool,
    mesh/cols=40,
    mesh/rows=40,
    domain=-2:2,
    y domain=-2:2,
    xtick=\empty, ytick=\empty, ztick=\empty,
    width=10cm,
    height=7cm,
  ]
    % 鞍点函数: z = x^2 - y^2
    \addplot3[surf, opacity=0.8, faceted color=blue!30] 
        {x^2 - y^2};
    
    % 鞍点中心
    \addplot3[only marks, mark=*, purple, mark size=5pt] coordinates {(0, 0, 0)};
    
    % 方向标注
    \node[purple, font=\small] at (axis cs:1, -1.5, 6) {沿$\theta_1$: 极小};
    \node[purple, font=\small] at (axis cs:1, -1.5, 4) {沿$\theta_2$: 极大};
  \end{axis}
\end{tikzpicture}
```

**深度学习的特殊情况**：

- 高维空间中，**鞍点比局部最优更常见**
- 随机梯度下降的噪声有助于跳出局部最优和鞍点
- 现代网络通常能找到足够好的解，即使不是全局最优

### 动量法：积累速度翻越障碍

**直觉**：想象滚雪球下山，球会积累动量，越滚越快，同时可以滚过小的坑洼。在 loss landscape 中，动量帮助我们冲过鞍点和平坦区域。

**原理**：引入速度变量，累积历史梯度：

$$
v_{t+1} = \beta v_t + \nabla J(\theta_t) \\
\theta_{t+1} = \theta_t - \eta v_{t+1}
$$

**效果**：

- 加速收敛（尤其在梯度方向一致时）
- 减少震荡（梯度方向变化时动量抵消）
- 帮助跳出局部最优

### Adam（自适应矩估计）

**直觉**：为每个参数单独调整学习率。频繁更新的参数用较小学习率，稀疏更新的用较大学习率。

**特点**：

- 结合动量和自适应学习率
- 对超参数不敏感，**默认首选**
- 适合大多数深度学习任务

```{list-table} 优化算法选择建议
:header-rows: 1
:widths: 25 35 40

* - **算法**
  - **适用场景**
  - **特点**
* - SGD + Momentum
  - 图像分类、大模型
  - 泛化性能好，需调学习率
* - Adam
  - 默认首选、NLP、推荐系统
  - 自适应、收敛快、易用
* - AdamW
  - Transformer、大模型
  - 改进权重衰减，效果更好
```

---

## 学习率调度：为什么需要调整学习率？

### 核心问题

训练初期，参数远离最优，需要**大学习率**快速接近目标。
训练后期，参数接近最优，需要**小学习率**精细调整，避免在最优点附近震荡。

```{tikz} 固定学习率 vs 学习率调度
\begin{tikzpicture}[scale=0.9]
    % 左侧：固定学习率（震荡）
    \begin{scope}[xshift=-5cm]
        \draw[->] (-0.5,0) -- (5,0) node[right] {迭代};
        \draw[->] (0,-0.5) -- (0,4) node[above] {损失};
        
        % 震荡收敛曲线
        \draw[thick, red, domain=0:4.5, samples=100] 
            plot (\x, {0.5 + 2*exp(-0.5*\x) + 0.3*sin(10*\x*180/pi)*exp(-0.3*\x)});
        
        \node[red, font=\small] at (2.5, 3) {固定大学习率};
        \node[red, font=\small] at (2.5, 2.5) {后期在最优点附近震荡};
    \end{scope}
    
    % 右侧：学习率调度（平滑收敛）
    \begin{scope}[xshift=2cm]
        \draw[->] (-0.5,0) -- (5,0) node[right] {迭代};
        \draw[->] (0,-0.5) -- (0,4) node[above] {损失};
        
        % 平滑收敛曲线
        \draw[thick, green!70!black, domain=0:4.5, samples=100] 
            plot (\x, {0.5 + 2.5*exp(-1.2*\x)});
        
        \node[green!70!black, font=\small] at (2.5, 3) {学习率衰减};
        \node[green!70!black, font=\small] at (2.5, 2.5) {平滑收敛到最优};
    \end{scope}
\end{tikzpicture}
```

### 常见学习率调度策略

```{tikz} 学习率调度策略对比
\begin{tikzpicture}[scale=0.85]
    % 坐标轴
    \draw[->] (-0.5,0) -- (6,0) node[right] {epoch};
    \draw[->] (0,-0.5) -- (0,4) node[above] {学习率};
    
    % 步长衰减（Step Decay）
    \draw[thick, blue] (0, 3.5) -- (2, 3.5);
    \draw[thick, blue] (2, 1.75) -- (4, 1.75);
    \draw[thick, blue] (4, 0.875) -- (5.5, 0.875);
    \draw[dashed, blue] (2, 3.5) -- (2, 1.75);
    \draw[dashed, blue] (4, 1.75) -- (4, 0.875);
    
    % 指数衰减（Exponential）
    \draw[thick, red, domain=0:5.5, samples=50] 
        plot (\x, {3.5*exp(-0.4*\x)});
    
    % 余弦退火（Cosine）
    \draw[thick, orange, domain=0:5.5, samples=100] 
        plot (\x, {1.75 + 1.75*cos(30*\x)});
    
    % 图例（Legend）
    \begin{scope}[xshift=5.5cm, yshift=3cm]
        \draw[thick, blue] (0, 0) -- (0.5, 0);
        \node[right, font=\small] at (0.6, 0) {Step Decay};
        
        \draw[thick, red] (0, -0.4) -- (0.5, -0.4);
        \node[right, font=\small] at (0.6, -0.4) {Exponential};
        
        \draw[thick, orange] (0, -0.8) -- (0.5, -0.8);
        \node[right, font=\small] at (0.6, -0.8) {Cosine};
    \end{scope}
\end{tikzpicture}
```

#### 1. 步长衰减（Step Decay）

**策略**：每N个epoch，学习率乘以衰减系数（如0.1）

**适用场景**：
- 传统图像分类任务（ImageNet训练标配）
- 当验证集损失不再下降时手动降低学习率

```python
# 每30个epoch，学习率乘以0.1
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
```

**特点**：阶梯式下降，简单有效，但需要预设衰减时机。

#### 2. 指数衰减（Exponential Decay）

**策略**：每个epoch学习率按指数衰减

$$\eta_t = \eta_0 \cdot \gamma^t$$

**适用场景**：
- 需要平滑连续衰减
- 训练轮数较多时

```python
# 每个epoch学习率乘以0.95
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
```

#### 3. 余弦退火（Cosine Annealing）

**策略**：学习率按余弦函数从初始值降到接近0

$$\eta_t = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})(1 + \cos(\frac{t}{T_{max}}\pi))$$

**适用场景**：

- 现代Transformer模型（BERT、GPT等）
- 配合Warmup使用效果更佳

```python
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
```

### 4. 预热（Warmup）

**问题**：训练初期参数随机初始化，梯度可能很大且不稳定，大学习率会导致震荡。

**策略**：从很小的学习率开始，线性增加到目标学习率。

```{tikz} Warmup + Cosine Annealing
\begin{tikzpicture}[scale=0.9]
    \draw[->] (-0.5,0) -- (8,0) node[right] {Epoch};
    \draw[->] (0,-0.5) -- (0,4) node[above] {学习率};
    
    % Warmup阶段（0-1）
    \draw[thick, green!70!black] (0, 0.2) -- (1, 3.5);
    \node[green!70!black, font=\small] at (1, 0.8) {Warmup};
    
    % Cosine退火阶段（1-7）
    \draw[thick, blue, domain=1:7, samples=100] 
        plot (\x, {1.75 + 1.75*cos((\x-1)/6*180)});
    \node[blue, font=\small] at (4, 3.8) {Cosine Annealing};
    
    % 标注
    \draw[dashed, gray] (1, 0) -- (1, 3.5);
    \node[font=\scriptsize] at (1, -0.2) {Warmup Steps};
\end{tikzpicture}
```

```python
# 组合使用：Warmup + Cosine Annealing
scheduler1 = optim.lr_scheduler.LinearLR(
    optimizer, 
    start_factor=0.01,      # 从1%的目标学习率开始
    end_factor=1.0,
    total_iters=5           # 前5个epoch预热
)
scheduler2 = optim.lr_scheduler.CosineAnnealingLR(
    optimizer, 
    T_max=95                # 剩余95个epoch余弦退火
)
scheduler = optim.lr_scheduler.SequentialLR(
    optimizer, 
    schedulers=[scheduler1, scheduler2], 
    milestones=[5]
)
```

### 选择建议

| 策略 | 推荐场景 | 优点 | 缺点 |
|------|----------|------|------|
| Step Decay | 图像分类、CNN | 简单有效 | 需预设衰减时机 |
| Exponential | 长周期训练 | 平滑连续 | 衰减过快 |
| Cosine | Transformer、NLP | 效果通常最好 | 相对复杂 |
| Warmup + Cosine | 大模型训练 | 稳定+效果好 | 需调两个参数 |

---

## PyTorch实践

- **基本训练循环**

    ```python
    import torch
    import torch.nn as nn
    import torch.optim as optim

    # 定义模型、损失函数、优化器
    model = MyModel()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 训练循环
    for epoch in range(num_epochs):
        for data, target in dataloader:
            # 前向传播
            output = model(data)
            loss = criterion(output, target)
            
            # 反向传播
            optimizer.zero_grad()  # 清空梯度
            loss.backward()        # 计算梯度
            optimizer.step()       # 更新参数
    ```

- **学习率调度**

    ```python
    # 组合调度：预热 + 余弦退火
    scheduler1 = optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=5)
    scheduler2 = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=45)
    scheduler = optim.lr_scheduler.SequentialLR(optimizer, [scheduler1, scheduler2], milestones=[5])

    for epoch in range(num_epochs):
        train(...)
        scheduler.step()  # 更新学习率
    ```

---

## 训练技巧

1. **梯度裁剪**

    防止梯度爆炸：

    ```python
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    ```

2. **批归一化（BatchNorm）**

    稳定训练、加速收敛：

    ```python
    self.bn = nn.BatchNorm1d(num_features)
    ```

3. **早停（Early Stopping）**

    验证集损失不再下降时停止训练，防止过拟合。

---

## 总结

梯度下降是深度学习训练的基石：

1. **基本原理**：沿{ref}`back-propagation`计算的负梯度方向迭代更新参数
2. **学习率**：最关键的超参数，过大震荡，过小收敛慢
3. **Mini-batch**：平衡效率和稳定性，深度学习标配
4. **Adam**：默认首选优化器，自适应、易用
5. **学习率调度**：训练过程中调整学习率能提升效果

理解{ref}`loss-functions`和优化算法后，你就掌握了深度学习训练的全流程。接下来我们将通过{doc}`the-end`回顾本章内容，然后进入实践章节。
