(unet-loss)=
# 损失函数设计

## 直觉：分类损失 vs 分割损失

分类任务（{doc}`../neural-network-basics/cnn-basics`）的交叉熵损失看的是"每个像素分对了吗"。但分割任务有一个特殊问题：**类别极度不平衡**。

想象一张 CT 图像，肿瘤只占 1% 的像素。如果模型把所有像素都预测为"正常组织"，准确率是 99%，但分割结果毫无意义——没找到肿瘤。

分割损失的核心挑战是：**让模型关心"少数派像素"**，而不是只追求整体准确率。

## 交叉熵损失

### 标准交叉熵

```{math}
L_{\text{CE}} = -\frac{1}{N} \sum_{i=1}^N \sum_{c=1}^C y_{i,c} \log(p_{i,c})
```

其中 $N$ 是像素数，$C$ 是类别数，$y_{i,c}$ 是真实标签，$p_{i,c}$ 是预测概率。

### 加权交叉熵

缓解类别不平衡：给少数类更高的权重。

```{math}
L_{\text{WCE}} = -\frac{1}{N} \sum_{i=1}^N \sum_{c=1}^C w_c \cdot y_{i,c} \log(p_{i,c})
```

权重 $w_c$ 通常与类别频率成反比。肿瘤只有 1% 像素，$w_{\text{肿瘤}}$ = 99，$w_{\text{正常}}$ = 1。

### 焦点损失（Focal Loss）{cite}`lin2017focal`

```{math}
L_{\text{FL}} = -\frac{1}{N} \sum_{i=1}^N \sum_{c=1}^C (1 - p_{i,c})^\gamma \cdot y_{i,c} \log(p_{i,c})
```

$\gamma$ 控制聚焦程度：$\gamma=2$ 时，易分类样本的贡献被大幅降低，模型被迫关注难分类样本。

### 交叉熵的局限

交叉熵优化的是**像素级准确率**，而我们真正关心的是**区域重叠度**。这两个目标在类别不平衡时可能南辕北辙——把所有像素都预测为"背景"，像素级准确率可能 >99%，但分割完全无效。

## Dice 损失

### Dice 系数

Dice 系数直接衡量预测掩码与真实掩码的重叠程度：

```{math}
\text{Dice} = \frac{2|X \cap Y|}{|X| + |Y|} = \frac{2TP}{2TP + FP + FN}
```

Dice 系数 = 1 表示完美重叠，= 0 表示完全不重叠。

```{tikz} Dice 系数的可视化
\begin{tikzpicture}[scale=0.8, opacity=0.8]
  % 完全重合
  \begin{scope}[shift={(0,0)}]
    \fill[blue!30] (0,0) circle (1);
    \fill[red!30] (0,0) circle (1);
    \node at (0, -1.8) {Dice = 1.00};
    \node[font=\small] at (0, -2.4) {完全重合};
  \end{scope}
  % 部分重叠
  \begin{scope}[shift={(4.5,0)}]
    \fill[blue!30] (-0.5,0) circle (1);
    \fill[red!30] (0.5,0) circle (1);
    \node at (0, -1.8) {Dice $\approx$ 0.67};
    \node[font=\small] at (0, -2.4) {部分重叠};
  \end{scope}
  % 不重叠
  \begin{scope}[shift={(9,0)}]
    \fill[blue!30] (-1.1,0) circle (1);
    \fill[red!30] (1.1,0) circle (1);
    \node at (0, -1.8) {Dice = 0.00};
    \node[font=\small] at (0, -2.4) {完全不重叠};
  \end{scope}
\end{tikzpicture}
```

### Dice 损失

```{math}
L_{\text{Dice}} = 1 - \text{Dice} = 1 - \frac{2|X \cap Y|}{|X| + |Y|}
```

在二分类任务中的像素级实现：

```{math}
L_{\text{Dice}} = 1 - \frac{2\sum_{i=1}^N p_i y_i + \epsilon}{\sum_{i=1}^N p_i + \sum_{i=1}^N y_i + \epsilon}
```

### Dice 损失的梯度分析

Dice 损失为什么对类别不平衡鲁棒？答案在它的梯度里。

先定义软 Dice 损失（去掉 $\epsilon$ 简化分析）：

$$L = 1 - \frac{2\sum p_i y_i}{\sum p_i + \sum y_i}$$

对第 $j$ 个像素的预测 $p_j$ 求导（链式法则，假设 $y_j=1$ 即该像素属于目标类）：

```{math}
\frac{\partial L}{\partial p_j} = -2 \cdot \frac{y_j (\sum p_i + \sum y_i) - (\sum p_i y_i) \cdot 1}{(\sum p_i + \sum y_i)^2}
```

分子提出 $y_j=1$：

```{math}
\frac{\partial L}{\partial p_j} = -2 \cdot \frac{\sum y_i - \sum p_i y_i}{(\sum p_i + \sum y_i)^2}
```

分子 $\sum y_i - \sum p_i y_i$ = **真实面积 - 重叠面积** = 被漏检的区域大小。

```{admonition} 直觉：Dice 梯度在说什么？
:class: tip

Dice 的梯度大小正比于 **"漏了多少"**，反比于 **总面积的平方**。

这意味着：
- 一个像素梯度的大小取决于**全局**漏检了多少，而不只是这个像素本身分对分错
- 小目标（$\sum y_i$ 很小）的梯度**天然放大**——因为分母 $(\sum p_i + \sum y_i)^2$ 中的 $\sum y_i$ 很小
- 比交叉熵更均衡：交叉熵对每个像素给独立梯度，大目标有更多像素 → 梯度主导；Dice 把整张图作为一个整体来优化
```

#### 梯度符号表

| 符号 | 含义 | 对梯度方向的影响 |
|------|------|----------------|
| $\sum y_i$ | 真实目标总面积 | 分母越大，整体梯度越小 |
| $\sum p_i y_i$ | 预测与真实的重叠面积 | 重叠越大，分子越小（漏检越少） |
| $\sum p_i$ | 预测目标总面积 | 分母项，过分割会降低梯度 |
| $\sum y_i - \sum p_i y_i$ | 真实中被漏检的面积 | **驱动梯度大小的核心**—漏越多，梯度越大 |

Dice 损失对类别不平衡天然鲁棒。因为它在分母中**同时用预测面积和真实面积做归一化**。肿瘤只占 1%？没关系——Dice 算的是"你预测的肿瘤和真实肿瘤重叠了多少"，不是"你对了多少像素"。

### Dice vs 交叉熵

| 特性 | 交叉熵 | Dice |
|------|--------|------|
| 关注重点 | 每个像素预测准确性 | 整体区域重叠度 |
| 类别不平衡 | 敏感 | **鲁棒** |
| 梯度来源 | 单像素 $p_i$ 与 1 的差距 | 全局漏检面积与总面积之比 |
| 小目标 | 被大目标淹没 | 天然放大（小分母效应） |
| 梯度特性 | 平滑、凸、易优化 | 非凸、可能有局部最优 |
| 收敛速度 | 通常较快 | 可能较慢 |

## 组合损失

实践中效果最好的往往是 **交叉熵 + Dice 组合损失**：

```{math}
L_{\text{total}} = \alpha L_{\text{CE}} + \beta L_{\text{Dice}}
```

交叉熵提供稳定的梯度，Dice 提供区域级优化目标。通常 $\alpha = \beta = 0.5$ 已经是很好的起点。

```python
class DiceLoss(nn.Module):
    """Dice 损失"""
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth
        
    def forward(self, pred, target):
        # pred: (B, C, H, W) logits, target: (B, H, W) 类别索引
        pred = torch.softmax(pred, dim=1)
        
        dice_loss = 0.0
        for c in range(pred.shape[1]):
            pred_c = pred[:, c, :, :]
            target_c = (target == c).float()
            
            intersection = (pred_c * target_c).sum()
            dice = (2. * intersection + self.smooth) / (
                pred_c.sum() + target_c.sum() + self.smooth
            )
            dice_loss += 1 - dice
            
        return dice_loss / pred.shape[1]
```

```python
class CombinedLoss(nn.Module):
    """交叉熵 + Dice 组合损失"""
    def __init__(self, weight_ce=0.5, weight_dice=0.5):
        super().__init__()
        self.weight_ce = weight_ce
        self.weight_dice = weight_dice
        self.ce_loss = nn.CrossEntropyLoss()
        self.dice_loss = DiceLoss()
        
    def forward(self, pred, target):
        return (self.weight_ce * self.ce_loss(pred, target) +
                self.weight_dice * self.dice_loss(pred, target))
```

## 损失函数选择指南

| 场景 | 推荐 | 理由 |
|------|------|------|
| 类别大致平衡 | 交叉熵 | 简单有效 |
| 类别极不平衡（肿瘤 < 1%） | Dice 损失 | 小目标天然放大 |
| 小目标检测 | 焦点损失 + Dice | 聚焦难分类样本 |
| 一般情况（推荐） | CE + Dice 组合 | 兼顾稳定性和优化目标 |

---

## 参考文献

```{bibliography}
:filter: docname in docnames
```
