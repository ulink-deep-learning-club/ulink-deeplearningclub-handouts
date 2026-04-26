# Deep Learning Club 讲座材料

> 欢迎来到深度学习：一群人在黑盒子里找规律，70%的数据清洗，20%的调参玄学，10%的祈祷。有时候能找到规律，有时候找到的是过拟合。但至少，我们的loss曲线很好看！

这是我们深度学习社的讲座材料库，包含了从基础到进阶的深度学习课程内容。


```{admonition} 目录
:class: note
~~~{toctree}
:maxdepth: 1

preface
math-fundamentals/index
neural-network-basics/index
pytorch-practice/index
cnn-ablation-study/index
transfer-learning/index
metric-learning/index
attention-mechanisms/index
unet-image-segmentation/index
postscript
~~~
```

## 学习路径

```{mermaid}
graph LR
    A[数学基础] --> B[神经网络基础]
    B --> C[PyTorch实践]
    C --> D{进阶方向}
    D --> E[CNN消融研究]
    D --> F[迁移学习]
    D --> G[注意力机制]
    D --> H[度量学习]
    D --> I[U-Net分割]
```

**建议顺序**：数学基础 → 神经网络基础 → PyTorch实践 → 选择感兴趣的进阶方向

## 前置知识检查

开始学习前，请确认你已掌握：

- [ ] **数学**：微积分（导数、链式法则）、线性代数（矩阵乘法）、基础概率论
- [ ] **编程**：Python基础（函数、类、NumPy数组操作）
- [ ] **工具**：能运行Jupyter Notebook或Python脚本的环境

**没达标？** 没关系，材料中会回顾必要概念，但预习会让你轻松很多。

## 📖 阅读建议

| 章节 | 预计时间 | 重点 |
|------|----------|------|
| 数学基础 | 2-3小时 | 理解自动微分和梯度下降的核心思想 |
| 神经网络基础 | 3-4小时 | 掌握归纳偏置概念，对比全连接与CNN |
| PyTorch实践 | 4-5小时 | 动手实现，熟悉训练流程 |
| 进阶方向 | 各2-3小时 | 根据兴趣选择，深入特定领域 |

**学习比例**：理论理解 40% + 代码实践 60% = 最佳效果

## 技术细节

- **构建系统**：基于Sphinx的现代文档系统，支持HTML、PDF等多种输出格式
- **数学支持**：LaTeX数学公式渲染
- **图表支持**：TikZ图表和自定义可视化
- **语言**：材料主要为中文，包含英文技术术语
- **代码示例**：带有详细解释的PyTorch实现

## 致谢

本材料在编写过程中使用了AI辅助工具，并参考了经典深度学习研究论文。

---

> ⚠️ **警告**：本课程包含大量梯度下降、反向传播和玄学调参。副作用可能包括：对GPU产生依赖、对过拟合产生恐惧、以及半夜醒来突然想到"是不是学习率太大"。

**最后更新**：2025-12-05
