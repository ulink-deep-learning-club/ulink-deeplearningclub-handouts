(seq-end)=
# 总结与展望

恭喜你完成了**序列建模**章节的全部内容！

从 {doc}`rnn-basics` 的大脑启发，到 {doc}`lstm` 的门控突破，再到 {doc}`from-rnn-to-attention` 的关键跳跃，{doc}`transformer` 的极致化，{doc}`mamba-intro` 的思想回归——我们追溯了一条横跨 35 年的架构演化之路。

## 知识回顾

### 四种架构的系统对比

| 对比维度 | RNN | LSTM | Transformer | Mamba |
| -------- | -------- | -------- | -------- | -------- |
| **年份** | 1986-1990 | 1997 | 2017 | 2023 |
| **灵感** | 大脑时序处理 | RNN + 记忆管理 | 信息检索 | RNN骨架 + 选择性 |
| **核心公式** | $h_t = \tanh(W_h h_{t-1} + W_x x_t)$ | $c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t$ | $\text{softmax}(QK^T/\sqrt{d_k})V$ | $h_t = A h_{t-1} + B_t x_t$ |
| **状态数量** | 1 个 | **2 个**（$\mathbf{c}_t$ + $\mathbf{h}_t$） | 无固定状态（K/V缓存） | 1 个（但选择性更强） |
| **信息路径** | O(n) 串行 | O(n) 串行 | O(1) 直接 | O(1) 通过状态 |
| **时间复杂度** | O(n) | O(n) | O(n²) | O(n) |
| **长程依赖** | 弱（梯度消失） | **中**（门控缓解但非根治） | 强 | 强（选择性机制） |
| **并行训练** | 不能 | 不能 | 能 | 能（硬件感知算法） |
| **推理速度** | 快 | 快 | 慢（K/V缓存） | 快（固定状态） |
| **参数效率** | 高 | 中（4×门控权重） | 低（FFN占大头） | 高 |
| **归纳偏置** | 强：时序因果 | 强 + 门控记忆 | 弱：等距连接 | 中：选择性时序 |

### 核心思想的演化

| 概念 | 来源 | 关键洞察 |
| -------- | -------- | -------- |
| **循环记忆** | {doc}`rnn-basics` | 通过隐状态传递让网络拥有对历史的压缩表示 |
| **BPTT** | {doc}`rnn-basics` | 反向传播在时间展开计算图上的应用，是梯度消失的根源 |
| **双状态分离** | {doc}`lstm` | 将"长期记忆"（$\mathbf{c}_t$）与"对外输出"（$\mathbf{h}_t$）分开管理 |
| **三重门控** | {doc}`lstm` | 遗忘门（擦除）、输入门（写入）、输出门（读取）——精细管理记忆的进与出 |
| **梯度高速公路** | {doc}`lstm` | $\frac{\partial \mathbf{c}_t}{\partial \mathbf{c}_{t-1}} = \mathbf{f}_t$——细胞状态提供了一条不经过非线性的梯度通路 |
| **因果注意力** | {doc}`from-rnn-to-attention` | 连接所有前序状态的直觉——从"传话"到"查档案" |
| **Q/K/V 分离** | {doc}`from-rnn-to-attention` | 将"查询"和"被查询"的角色分离，增强表达力 |
| **多头注意力** | {doc}`transformer` | 多个视角并行，捕捉不同类型的关系 |
| **FFN 知识存储** | {doc}`transformer` | 注意力负责跨位置交互，FFN 负责位置内部的特征变换和知识存储 |
| **位置编码** | {doc}`transformer` | 补偿注意力对位置信息的无知 |
| **选择性SSM** | {doc}`mamba-intro` | 让状态更新规则依赖输入——{ref}`lstm` 门控思想的彻底化 |

### 叙事弧线：一个思想的螺旋上升

```{mermaid}
graph LR
    A["RNN<br/>1986-1990<br/>模仿大脑, O(n)<br/>但长程依赖消失"]
    B["LSTM<br/>1997<br/>门控机制<br/>梯度高速公路"]
    C["注意力<br/>2014<br/>全局连接<br/>O(1)信息路径"]
    D["Transformer<br/>2017<br/>纯注意力<br/>O(n²)代价"]
    E["Mamba<br/>2023<br/>选择性SSM<br/>O(n)+长程依赖"]

    A -->|"问题驱动"| B
    B -->|"串行瓶颈仍在"| C
    C -->|"极致化"| D
    D -->|"效率反思"| E
    E -.->|"螺旋回归"| A
```

```{admonition} 为什么要理解这一条演化路径？
:class: tip

1. **避免只见树木不见森林**：单独的 RNN、Transformer、Mamba 都只是"点"，演化路径才是"线"——理解从哪里来，才能判断往哪里去
2. **培养架构判断力**：当你遇到新架构时，能迅速定位它在这条演化线上处于什么位置——是回归、是改进、还是全新的分支
3. **建立"问题→解决→新问题"的思维模式**：每个架构都不是凭空设计的，而是对前一代架构缺陷的回应
```

## 与前面章节的联系

| 前面章节 | 本章应用 |
| -------- | -------- |
| {ref}`gradient-vanishing` | RNN 中梯度消失的数学根源（Jacobian 连乘）；LSTM 如何通过 $\mathbf{c}_t$ 旁路解决 |
| {ref}`inductive-bias` | 四种架构的归纳偏置对比——从强（RNN/LSTM）到弱（Transformer）到中（Mamba） |
| {ref}`res-net` | 残差连接在 Transformer 中的延续；LSTM 细胞状态的梯度高速公路 |
| {ref}`attention-mechanisms` | CNN 的通道/空间注意力 vs Transformer 的自注意力——两种不同的"注意力" |
| {ref}`computational-graph` | BPTT 在时间展开计算图上的反向传播 |
| {ref}`activation-functions` | $\tanh$ 在 RNN/LSTM 中作为门控激活；ReLU 在 Transformer FFN 中 |

## 实践建议

```{admonition} 如何选择序列建模架构？
:class: tip

- **短序列（< 1K）+ 需要最快效果**：Transformer（生态成熟，预训练模型最多）
- **长序列（> 10K）+ 推理速度关键**：Mamba / SSM（O(n) 速度优势显著）
- **教学/理解序列建模**：从 RNN 开始——最直观，问题也最明显
- **生产环境**：优先选择社区活跃、预训练权重丰富的架构（当前仍是 Transformer 主导）
```

## 未来方向

序列建模远未结束。以下是正在发展中的方向：

1. **混合架构**：Transformer 层 + SSM 层组合使用——取 Transformer 的表达力和 SSM 的效率
2. **线性注意力**：通过核技巧将 $O(n^2)$ 的注意力近似为 $O(n)$——Mamba 之外的另一种降复杂度路径
3. **状态空间模型的扩展**：Mamba-2、Jamba 等后续工作进一步改进选择性机制和硬件效率
4. **多模态序列**：将序列建模从文本扩展到视频、音频、基因组——这些都是天然的长序列

---

## 推荐资源

### 入门理解

- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)（Jay Alammar 的可视化讲解，Transformer 入门首选）
- [The Annotated Transformer](http://nlp.seas.harvard.edu/annotated-transformer/)（Harvard NLP 的逐行注释实现）

### 深入阅读

- {cite}`elman1990finding` —— RNN 的奠基论文（1990）
- {cite}`hochreiter1997long` —— LSTM 的原始论文（1997）
- {cite}`bahdanau2014neural` —— 注意力机制的起源（2014, ICLR 2015）
- {cite}`vaswani2017attention` —— Transformer 的诞生（2017, NeurIPS）
- {cite}`gu2023mamba` —— Mamba 的提出（2023）

### 动手实践

- [Andrej Karpathy's makemore](https://github.com/karpathy/makemore)（从零实现 RNN、LSTM、Transformer）
- [Mamba 官方代码](https://github.com/state-spaces/mamba)（选择性 SSM 的完整实现）

---

## 本章完

通过本章的学习，你不仅掌握了四种序列建模架构，更重要的是理解了它们之间的**演化逻辑**——为什么会有 RNN、为什么需要 LSTM、为什么注意力能革命性地解决问题、为什么 Transformer 又催生了 Mamba。

**记住**：

- **RNN 的困境**不是设计失败，而是串行信息传递的固有矛盾——这是理解后续所有架构的起点
- **注意力是"连接一切"的思想**——它解决了长程依赖，但代价是 $O(n^2)$
- **Mamba 是思想的螺旋回归**——在更高的层次上重新拥抱 RNN 的效率哲学，并用选择性机制弥补了它的缺陷

好的架构不是凭空发明的，而是**对前一代问题深刻理解后的自然回应**。这正是 {ref}`inductive-bias` 中讨论的核心思想——把先验知识内置到架构中，让结构本身替模型做出正确的假设。

---

**下一步**：回到 {doc}`../index` 选择其他进阶章节，或者探索 {doc}`../model-architecture-design/index` 学习通用的架构设计心法。

---

```{only} not latex

~~~{rubric} 参考文献
:heading-level: 2
~~~

~~~{bibliography}
:filter: docname in docnames
~~~
```

