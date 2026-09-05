# Deep Learning Club 学习教程

~~~{toctree}
:hidden:

preface
math-fundamentals/index
cnn-expedition/index
pytorch-practice/index
model-serving/index
transfer-learning/index
unet-image-segmentation/index
sequence-modeling/index
appendix/index
postscript
~~~

```{only} not latex
> 欢迎来到深度学习：一群人在黑盒子里找规律，70%的数据清洗，20%的调参玄学，10%的祈祷。有时候能找到规律，有时候找到的是过拟合。但至少，我们的loss曲线很好看！

这是我们深度学习社的教学资料库，面向具备微积分基础的高中生，提供从理论到实践的深度学习课程。

~~~{rubric} 学习路径
:heading-level: 2
~~~

~~~{mermaid}
graph LR
    A[数学基础<br/>计算图与梯度] --> B[翻越两座山<br/>CNN架构演化]
    B --> C[PyTorch实践<br/>训练流程]
    C --> D{进阶方向<br/>任选方向}
    D --> E[模型部署<br/>工程实践]
    D --> F[迁移学习<br/>实用技术]
    D --> G[U-Net分割<br/>CV实战]
    D --> H[CNN架构改造<br/>设计心法]
    D --> I[序列建模<br/>RNN到Transformer到Mamba]
~~~

**核心路径**（必修）：数学基础 → 翻越两座山 → PyTorch实践
**进阶路径**（选修）：根据兴趣选择一个或多个方向深入研究

~~~{rubric} 前置知识
:heading-level: 2
~~~

| 领域 | 要求 | 具体内容 |
| ---------- | ---------- | ---------- |
| **数学** | 必修 | 微积分（导数、链式法则）、线性代数（矩阵乘法）、基础概率论 |
| **编程** | 必修 | Python基础（函数、类、NumPy数组操作） |
| **深度学习** | 无需 | 本课程从零开始，不需要前置知识 |

**配套教材**：斋藤康毅《深度学习入门：基于Python的理论与实现》（俗称"鱼书"）。

> 鱼书从Python/NumPy出发，**不依赖任何外部库**，手把手教你从零手写神经网络。本材料使用**PyTorch**，侧重"为什么"和"怎么用好"。两者是**互补关系**：
> - 鱼书第3章（神经网络）→ 本材料{doc}`cnn-expedition/index`
> - 鱼书第5章（计算图/反向传播）→ 本材料{doc}`math-fundamentals/index`
> - 鱼书第7章（手写CNN）→ 本材料{doc}`cnn-expedition/index`
>
> 详见{doc}`preface`中的完整对照说明与8周并行学习计划。

~~~{rubric} 时间规划
:heading-level: 2
~~~

| 阶段 | 章节 | 预计时间 | 核心目标 |
| ---------- | ---------- | ---------- | ---------- |
| **基础阶段** | 数学基础 | 2-3小时 | 理解计算图、反向传播和梯度下降的核心机制 |
| | 翻越两座山 | 7-9小时 | 全连接→CNN→LeNet（上篇），AlexNet到EfficientNet（下篇），消融研究，架构心法 |
| **实践阶段** | PyTorch实践 | 4-5小时 | 动手搭建并训练完整神经网络 |
| **进阶阶段** | 模型部署 | 3-4小时 | ONNX导出、量化剪枝、服务架构、Ferrinx部署上线 |
| | 迁移学习 | 2-3小时 | 掌握小数据集训练大模型的核心技术 |
| | U-Net分割 | 3-4小时 | 实现图像分割模型，理解编码器-解码器架构 |
| | CNN架构改造 | 2-3小时 | 掌握CNN架构设计心法（属翻越两座山章，独立选修） |
| | 序列建模 | 4-6小时 | 理解RNN/LSTM/Transformer/Mamba的演化逻辑 |

**学习建议**：理论理解 40% + 代码实践 60% = 最佳效果

~~~{rubric} 技术特点
:heading-level: 2
~~~

- **构建系统**：Sphinx + Markdown，支持HTML/PDF输出
- **数学支持**：LaTeX公式渲染
- **可视化**：TikZ图表、Mermaid流程图
- **代码实践**：完整PyTorch实现，每行代码带详细注释
- **Cross-Reference**：章节间紧密关联，形成知识网络

~~~{rubric} 附录：环境配置
:heading-level: 2
~~~

当你从"跑通代码"进阶到"实际项目"时，迟早需要一台GPU服务器。附录 {doc}`appendix/environment-setup/index` 不讲神经网络，只讲"让深度学习代码跑起来"的实操技能：

| 章节 | 内容 | 典型场景 |
| ---------- | ---------- | ---------- |
| {doc}`appendix/environment-setup/linux-basics` | Linux基础、文件系统、SSH连接 | 第一次接触Linux服务器 |
| {doc}`appendix/environment-setup/nvidia-setup` | NVIDIA驱动、CUDA、PyTorch GPU版 | `nvidia-smi`报错时 |
| {doc}`appendix/environment-setup/remote-access` | 内网穿透、Frp配置 | 服务器在机房/实验室 |
| {doc}`appendix/environment-setup/server-management` | tmux、systemd、开机自启 | 训练跑到一半终端关了 |

**使用建议**：这不是必修内容，遇到具体问题时按需查阅即可。

---

> ⚠️ **警告**：本课程包含大量梯度下降、反向传播和玄学调参。副作用可能包括：对 GPU 产生依赖、对过拟合产生恐惧、以及半夜醒来突然想到"是不是学习率太大"。

**最后更新**：2026-06-14
```
