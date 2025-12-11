# 神经网络训练基础

## 基本训练术语

在深入探讨神经网络架构之前，我们需要明确一些基本的训练概念和术语：

```{note}
**核心训练术语**

- **Epoch（轮次）**：完整遍历整个训练数据集一次的过程
- **Batch（批次）**：一次训练迭代中使用的样本集合
- **Batch Size（批大小）**：每个批次中的样本数量，影响训练稳定性和内存使用
- **Iteration（迭代）**：完成一个批次的训练步骤
- **Learning Rate（学习率）**：控制模型参数更新步长的超参数
- **Loss Function（损失函数）**：衡量模型预测与真实标签差异的函数
```

## 损失函数的选择

损失函数是神经网络训练的核心，不同的任务需要不同的损失函数：

```{list-table} 常见损失函数对比
:header-rows: 1
:widths: 30 40 30

* - **任务类型**
  - **损失函数**
  - **数学表达式**
* - 回归任务
  - 均方误差（MSE）
  - $\displaystyle \frac{1}{n}\sum_{i=1}^n (y_i - \hat{y}_i)^2$
* - 分类任务
  - 交叉熵损失
  - $\displaystyle -\sum_{i=1}^n y_i \log(\hat{y}_i)$
* - 二分类任务
  - 二元交叉熵
  - $\displaystyle -\frac{1}{n}\sum_{i=1}^n [y_i \log(\hat{y}_i) + (1-y_i)\log(1-\hat{y}_i)]$
```

```{admonition} MNIST分类的损失函数
:class: example

对于MNIST手写数字识别这样的10类分类任务，我们使用**分类交叉熵损失函数**：

$$
\mathcal{L} = -\sum_{c=1}^{10} y_c \log(p_c)
$$

其中：
- $y_c$：真实标签的one-hot编码（0或1）
- $p_c$：模型预测属于类别$c$的概率
- 求和遍历所有10个数字类别

**为什么选择交叉熵而不是MSE？** 交叉熵对概率分布的差异更敏感，且在分类任务中梯度更稳定。
```

## 训练过程与验证

### 训练集与测试集的划分

在机器学习中，我们将数据划分为不同的集合：

- **训练集（Training Set）**：用于模型参数的学习
- **验证集（Validation Set）**：用于超参数调优和模型选择
- **测试集（Test Set）**：用于最终模型性能评估

```{warning}
**数据划分的重要性**

- 避免过拟合：确保模型在未见过的数据上表现良好
- 公平评估：测试集只能在最终评估时使用一次
- 模型选择：验证集帮助我们选择最佳超参数
```

### 训练过程的监控

有效的训练需要监控多个指标：

```{note}
**关键监控指标**

- **训练损失**：模型在训练集上的表现，应该逐渐下降
- **验证损失**：模型在验证集上的表现，反映泛化能力
- **训练准确率**：模型在训练集上的分类正确率
- **验证准确率**：模型在验证集上的分类正确率
```

## 过拟合与欠拟合

### 欠拟合（Underfitting）

当模型过于简单，无法捕捉数据中的基本模式时发生：

- 训练损失和验证损失都很高
- 模型在训练集上表现不佳
- 解决方法：增加模型复杂度、减少正则化、训练更长时间

### 过拟合（Overfitting）

想象你在学习一门新课程：

- **正常学习**：理解基本概念，能够解决类似问题
- **死记硬背**：记住所有例题，但遇到新题型就不会

当模型过于复杂，记忆了训练数据中的噪声而非真实模式时发生：

- 训练损失很低，但验证损失很高
- 模型在训练集上表现很好，但在新数据上表现差
- 解决方法：增加训练数据、使用正则化、早停法、降低模型复杂度

```{tikz} 模型复杂度与损失的关系（示意图）
\begin{tikzpicture}[scale=1.1]
  % Axes
  \draw[->] (0,0) -- (7,0) node[right] {模型复杂度};
  \draw[->] (0,0) -- (0,4) node[above] {损失};
  
  % Training loss curve (decreasing)
  \draw[thick, blue, domain=0.5:6.5, smooth] plot ({\x}, {2.8/(0.7*\x+1)});
  \node[blue] at (5.5,1.1) {训练损失};
  
  % Validation loss curve (U-shape)
  \draw[thick, red, domain=0.5:6.5, smooth] plot ({\x}, {1.2 + 0.2*\x - 0.25*exp(-0.5*(\x-3)^2)});
  \node[red] at (1.5,2.8) {验证损失};
  
  % Overfitting region
  \draw[dashed, gray] (4.5,0) -- (4.5,4);
  \node[gray] at (5.2,3.5) {过拟合区域};
  
  % Underfitting region
  \draw[dashed, gray] (1.5,0) -- (1.5,4);
  \node[gray] at (0.9,3.5) {欠拟合区域};
  
  % Optimal point
  \fill[green!70!black] (3,1.5) circle (0.08);
  \node[green!70!black] at (3,1.8) {最佳复杂度};
\end{tikzpicture}
```

```{note}
有点牵强的总结：学而不思则过拟合，思而不学则欠拟合。
```

## 正则化技术：防止过拟合的实用方法

### 什么是正则化？

在神经网络中，正则化就是帮助我们训练出“聪明”而不是“死记硬背”的模型的技术。

```{list-table} 正则化的直观理解
:header-rows: 1
:widths: 50 50

* - **学生学习**
  - **神经网络训练**
* - 理解概念
  - 学习通用特征
* - 举一反三
  - 泛化到新数据
* - 正则化帮助学生避免死记硬背
  - 正则化帮助模型避免过拟合
```

### L1和L2正则化：最简单的正则化方法

**核心思想：** 让模型的权重不要变得太大

正则化通过在原始损失函数中添加参数惩罚项来实现：

```{admonition} L1正则化（Lasso回归）
:class: note

```{math}
\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{original}} + \lambda \sum_{i=1}^{n} |w_i|
```

**特点：**
- 产生稀疏解，许多参数会变为0
- 具有特征选择功能，自动选择重要特征
- 对异常值相对鲁棒
- 不可导，需要特殊优化方法


```{admonition} L2正则化（Ridge回归）
:class: note

```{math}
\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{original}} + \frac{\lambda}{2} \sum_{i=1}^{n} w_i^2
```

**特点：**
- 参数趋向于小而非零的值
- 数学处理更简单，可导
- 对异常值敏感
- 是最常用的正则化形式


```{admonition} 类比
:class: example

想象你在调音：
- **L1正则化**：直接告诉某些旋钮“完全不要动”（变成0）
- **L2正则化**：告诉所有旋钮“不要调得太大”（保持小值）
```

```{warning}
**为什么有效？**

- 权重太大 → 模型过于敏感 → 容易记住训练数据
- 权重较小 → 模型更加平滑 → 能够泛化到新数据
- 就像用粗笔画画，不会画出过于复杂的细节
```

```{admonition} PyTorch中的L2正则化
:class: example

在PyTorch中，L2正则化被称为“权重衰减”（weight decay）：

```python
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001,
    weight_decay=0.0001  # L2正则化
)
```

**实践建议：** 从0.0001开始尝试，根据验证效果调整

### Dropout：随机“失忆”技术

**核心思想：** 训练时随机让一些神经元“罢工”，迫使网络学习冗余的特征

```{tikz} Dropout：随机让部分神经元“罢工”
\begin{tikzpicture}[scale=0.8]
    % Input layer
    \foreach \i in {1,2,3,4}
        \node[circle, draw=blue!50, fill=blue!20, minimum size=0.6cm] (in\i) at (0,\i) {};
    
    % Hidden layer with dropout
    \foreach \i in {1,2,3,4,5}
        \node[circle, draw=red!50, minimum size=0.6cm] (hid\i) at (2,\i-0.5) {};
    
    % Some nodes crossed out (dropout)
    \foreach \i in {3,5}
        \draw[thick, red!70] (1.5,\i-1) -- (2.5,\i);
    \foreach \i in {3,5}
        \draw[thick, red!70] (1.5,\i) -- (2.5,\i-1);
    
    % Output layer
    \foreach \i in {1,2,3}
        \node[circle, draw=green!40!black, fill=green!20, minimum size=0.6cm] (out\i) at (4,\i+0.5) {};
    
    % Connections
    \foreach \i in {1,2,3,4}
        \foreach \j in {1,3,5}
            \draw[->, gray!50] (in\i) -- (hid\j);
    
    \foreach \j in {1,3,5}
        \foreach \k in {1,2,3}
            \draw[->, gray!50] (hid\j) -- (out\k);
\end{tikzpicture}
```

```{admonition} 直观的理解
:class: example

**训练阶段：** “同学们，今天随机抽一半同学回答问题，其他人休息”
- 每个人都要准备好，因为不知道会不会被抽到
- 不能依赖某个“学霸”同学，必须自己理解

**测试阶段：** “现在全班一起回答问题，把大家的答案平均一下”
- 相当于多个“子班级”的集体智慧
- 结果更加稳定可靠
```

```{admonition} Dropout的PyTorch实现
:class: example

```python
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(784, 256)
        self.dropout = nn.Dropout(0.5)  # 50%的dropout率
        self.fc2 = nn.Linear(256, 10)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)  # 只在训练时随机“丢弃”
        x = self.fc2(x)
        return x
```

**参数选择：**
- 输入层：通常不用dropout（0%）
- 隐藏层：0.3-0.5（30%-50%）
- 输出层：不用dropout

### 早停法：聪明的“刹车”技术

**核心思想：** 看到验证效果开始变差时就停止训练

```{tikz} 早停法示意图
\begin{tikzpicture}[scale=0.9]
    % Axes
    \draw[->] (0,0) -- (8,0) node[right] {训练时间};
    \draw[->] (0,0) -- (0,5) node[left] {准确率};
    
    % Training accuracy curve (improving)
    \draw[thick, blue, domain=0:7, smooth] plot ({\x}, {0.2 + 0.2*\x + 0.03*\x*\x});
    
    % Validation accuracy curve (plateau then drop)
    \draw[thick, red, domain=0:7, smooth] plot ({\x}, {0.3 + 0.4*\x - 0.05*\x*\x});
    
    % Optimal stopping point
    \draw[dashed, green!70!black] (4,0) -- (4,5);
    
    % Labels
    \node[blue] at (1.5,3.5) {训练准确率};
    \node[red] at (5.5,3.5) {验证准确率};
    \node[green!70!black] at (4,-0.3) {最佳停止点};
    
    % Overfitting indication
    \node[text width=3cm, align=center] at (8.5,2) {开始过拟合\\验证效果下降};
\end{tikzpicture}
```

```{warning}
**什么时候该停止？**

- 验证损失连续几个epoch没有改善
- 验证准确率开始下降
- 训练损失还在下降，但验证损失开始上升

**就像考试前：** 发现模拟考试成绩开始下降，就应该停止“熬夜突击”，保持当前水平
```

### 数据增强：免费的“新数据”

**核心思想：** 通过对现有数据进行合理变换，创造“新”的训练样本

```{admonition} MNIST数据增强实例
:class: example

原始图像：手写数字“3”
- **轻微旋转：** 顺时针转5度，还是“3”
- **平移：** 向左移动2像素，还是“3”
- **轻微缩放：** 放大到1.1倍，还是“3”
- **加噪声：** 加一点“雪花点”，还是“3”
```

```{note}
**数据增强的注意事项**

- **保持类别不变：** 增强后的数据应该还是同一个数字
- **避免过度增强：** 不要把“6”转得看起来像“9”
- **任务相关：** 手写数字识别不需要颜色变换
- **渐进式：** 从轻微变换开始，逐步增加强度
```

```{admonition} PyTorch中的MNIST数据增强
:class: example

```python
from torchvision import transforms

# 定义数据增强变换
train_transform = transforms.Compose([
    transforms.RandomRotation(10),      # 随机旋转±10度
    transforms.RandomAffine(0, translate=(0.1, 0.1)),  # 随机平移10%
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # MNIST数据的标准化
])

# 应用到训练数据
train_dataset = MNIST(root='./data', train=True,
                     transform=train_transform, download=True)
```

### 如何选择正则化方法？

```{list-table} 正则化方法选择指南
:header-rows: 1
:widths: 40 60

* - **方法**
  - **什么时候用**
* - L2正则化
  - 几乎所有情况
* - Dropout
  - 深层网络，参数量大
* - 早停法
  - 总是使用
* - 数据增强
  - 图像、语音等数据
* - L1正则化
  - 需要特征选择时
```

```{warning}
**初学者建议（从简单到复杂）**

1. **第一步：** 只用早停法（最简单，零成本）
2. **第二步：** 加上L2正则化（weight_decay=0.0001）
3. **第三步：** 在隐藏层加Dropout（0.3-0.5）
4. **第四步：** 尝试数据增强（如果适用）
5. **高级：** 组合多种方法，交叉验证调参
```

## 批量大小与学习率调度

### 批量大小的影响

不同的批量大小对训练有不同影响：

```{list-table} 不同批量大小的对比
:header-rows: 1
:widths: 30 35 35

* - **批量大小**
  - **优点**
  - **缺点**
* - 小批量（如32）
  - 泛化性好，内存需求低
  - 训练不稳定，梯度噪声大
* - 大批量（如256）
  - 训练稳定，并行度高
  - 内存需求大，可能陷入局部最优
* - 全批量（所有数据）
  - 梯度准确，收敛稳定
  - 内存需求极大，不适合大数据集
```

### 学习率调度

学习率不必固定不变，可以采用不同的调度策略：

```{note}
**常见学习率调度策略**

- **步长衰减：** 每隔一定轮次将学习率乘以固定因子
- **指数衰减：** 学习率按指数函数逐渐减小
- **余弦退火：** 学习率按余弦函数周期性变化
- **自适应方法：** 根据训练进展自动调整学习率
```

## 评价指标

对于分类任务，我们需要多个评价指标来全面评估模型性能：

```{note}
**分类任务的核心指标**

- **准确率（Accuracy）：** 正确预测的样本比例
  ```{math}
  \text{Accuracy} = \frac{\text{正确预测数}}{\text{总预测数}}
  ```
  
- **精确率（Precision）：** 预测为正类中真正为正类的比例
  ```{math}
  \text{Precision} = \frac{TP}{TP + FP}
  ```
  
- **召回率（Recall）：** 真正为正类中被正确预测的比例
  ```{math}
  \text{Recall} = \frac{TP}{TP + FN}
  ```
  
- **F1分数：** 精确率和召回率的调和平均
  ```{math}
  \text{F1} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}
  ```


```{admonition} MNIST评价指标选择
:class: example

对于MNIST这样的平衡多分类任务，**准确率**是最直观和常用的指标，因为：
- 10个类别的样本数量大致相等
- 每个类别的重要性相同
- 易于理解和解释
```