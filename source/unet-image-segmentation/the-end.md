(unet-end)=
# 应用、变体与展望

## 应用领域

U-Net 自 2015 年提出以来，应用早已超出最初的细胞分割。

### 医学影像分割（核心领域）

| 任务 | 典型 Dice | 特殊挑战 | U-Net 解决方案 |
|------|-----------|----------|---------------|
| 细胞核分割 | 0.92 | 细胞粘连、边界模糊 | 跳跃连接保留边界 |
| 肝脏/肿瘤分割 | 0.95 | 形状变异大 | 弹性变形数据增强 |
| 脑肿瘤分割 | 0.88 | 对比度低、边界不清晰 | 多尺度特征融合 |
| 视网膜血管分割 | 0.95（准确率） | 细长结构 | 深层次特征融合 |

### 拓展领域

- **卫星图像分析**：土地覆盖分类（IoU 0.89）、洪水检测、灾害评估
- **自动驾驶**：道路分割、可行驶区域检测（Cityscapes 数据集 mIoU 0.82）
- **工业检测**：表面缺陷检测、焊点质量检查
- **图像修复**：老照片修复、物体移除（U-Net 作为生成器）

U-Net 在这些领域的共性成功因素：**标注数据少、边界精度要求高、需要端到端训练**。

## 关键变体

十年间，研究人员围绕 U-Net 的核心思想提出了大量改进。以下是影响力最大的几个：

### U-Net++ {cite}`unetplusplus`：嵌套密集跳跃连接

解决了原始 U-Net 跳跃连接的"语义差距"问题（编码器浅层和解码器深层的特征"不太匹配"）。U-Net++ 在每条跳跃连接路径中加入中间卷积层来逐步缩小语义差距：

```{mermaid}
graph LR
    subgraph 编码器
    X0["输入"] --> X1["层1"]
    X1 --> X2["层2"]
    X2 --> X3["层3"]
    X3 --> X4["层4"]
    end
    subgraph 解码器
    X4 --> X5["上采样"]
    X5 --> X6["层3'"]
    X6 --> X7["层2'"]
    X7 --> X8["层1'"]
    X8 --> Y["输出"]
    end
    %% 原始U-Net连接
    X1 -.->|"skip"| X8
    X2 -.->|"skip"| X7
    X3 -.->|"skip"| X6
    X4 -.->|"skip"| X5
    %% U-Net++ 嵌套连接
    X1 -->|"dense"| X7
    X2 -->|"dense"| X6
    X1 -->|"dense"| X6
```

- **改进**：在跳跃连接路径中加入中间卷积层，逐步缩小语义差距
- **额外能力**：**深度监督**——多个解码器输出都参与损失计算
- **性能**：Dice 提升 2-5%，尤其在小数据集上
- **代价**：参数量和计算量增加

### Attention U-Net {cite}`attentionunet`：让跳跃连接学会"选择性关注"

{ref}`unet-arch` 中我们讨论了跳跃连接直接传递编码器特征。但编码器特征中不是所有区域都重要——背景区域的特征传递过去只是噪声。Attention U-Net 在跳跃连接入口处加入**注意力门（Attention Gate）**，让解码器特征指导编码器特征"只看该看的地方"。

```python
class AttentionGate(nn.Module):
    """注意力门：让跳跃连接"选择性关注"
    
    用解码器特征指导编码器特征的注意力权重
    """
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Conv2d(F_g, F_int, kernel_size=1)  # 解码器特征
        self.W_x = nn.Conv2d(F_l, F_int, kernel_size=1)  # 编码器特征
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1),
            nn.Sigmoid()
        )
    
    def forward(self, g, x):
        attn = self.psi(F.relu(self.W_g(g) + self.W_x(x)))
        return x * attn
```

**好处**：注意力图可以可视化，知道模型在"看哪里"——对医学诊断的可信度很重要。

### 3D U-Net {cite}`cicek20163d`：从切片到体积

把 2D 卷积换成 3D 卷积，直接处理 CT/MRI 的 3D 体积数据。

| 对比 | 2D U-Net | 3D U-Net |
|------|----------|----------|
| 输入 | 单张切片 | 整个体积（或 patch） |
| 卷积 | 2D (k×k) | 3D (k×k×k) |
| 内存 | ~4GB | ~16GB+ |
| 适用场景 | 切片独立的图像 | CT/MRI 等体积数据 |

3D U-Net 捕捉到了**相邻切片之间的连续性**，但对 GPU 内存是巨大挑战（通常需要 patch-based 训练）。

### TransUNet {cite}`transunet`：Transformer + U-Net

{ref}`receptive-field` 中我们提到标准 CNN 感受野有限。Transformer 的**自注意力机制**（{doc}`../attention-mechanisms/introduction`）一个操作内看到全局——TransUNet 用 Transformer 做编码器，U-Net 做解码器，兼得两者的优势。

### 变体对比表

| 变体 | 发布年份 | 核心改进 | Dice 提升 | 参数量变化 |
|------|---------|---------|-----------|-----------|
| U-Net++ | 2018 | 嵌套密集跳跃连接 | +2-5% | 增加 30-50% |
| Attention U-Net | 2018 | 注意力门控跳跃连接 | +1-3% | 略有增加 |
| 3D U-Net | 2016 | 3D 卷积 | +3-8%（体积数据） | 大幅增加 |
| ResUNet | 2017 | 残差连接 | +1-2% | 略有增加 |
| TransUNet | 2021 | Transformer 编码器 | +3-6% | 大幅增加 |

## 局限性

```{admonition} U-Net 并非万能
:class: warning

1. **计算开销大**：跳跃连接需要存储所有编码器特征图，内存占用大
2. **长距离依赖有限**：CNN 本质上是局部操作，感受野靠深度堆叠，效率不如 Transformer
3. **超参数敏感**：深度、初始通道数、学习率需要仔细调整
4. **对 3D 数据不够友好**：直接扩展为 3D 后内存爆炸，需要 patch 策略
```

## 核心启示

回顾这一章，U-Net 教给我们的不仅是分割方法，更是一种**设计哲学**：

| 问题 | U-Net 的回答 | 可迁移的思想 |
|------|-------------|-------------|
| "位置 vs 语义"矛盾 | 编码器-解码器 + 跳跃连接 | 对称信息流 + 短路连接 |
| "数据太少" | 弹性变形数据增强 | 领域知识指导增强策略 |
| "小目标被忽略" | Dice 损失 | 用评估指标指导损失设计 |
| "浅层梯度消失" | 跳跃连接 | 梯度高速公路 |

{doc}`../neural-network-basics/cnn-basics` 的 {ref}`inductive-bias` 告诉我们：好的架构设计把先验知识内置到网络中。U-Net 做了一个极好的示范——把"分割需要位置+语义"这个先验翻译成了 U 形的对称结构。

---

## 常见问题速答

**Q: U-Net 的输入尺寸必须是 572×572 吗？**

A: 不是。原始论文用 572×572 是因为用了有效填充。现代实现用相同填充（padding=1），输入可以是任意尺寸（最好是 2 的幂次，如 256、512 等），因为下采样 4 次需要输入能被 16 整除。

**Q: 什么时候用 U-Net 而不是其他分割网络？**

A: U-Net 最适合**小数据、高精度边界**的场景。如果你的数据量很大（>10K 张），DeepLab 或 Mask R-CNN 可能更好。如果边界不是核心关注点，FCN 就够用。

**Q: 跳跃连接为什么用拼接而不是相加？**

A: 拼接保留编码器特征的**全部信息**。FCN 用相加相当于对编码器特征做了一个线性投影（压缩），可能丢失信息。拼接让解码器自己决定"从编码器特征中取多少"——学出来的效果更好。

**Q: Dice 损失训练不稳定怎么办？**

A: 两个常见原因：1) 梯度非凸导致局部震荡 → 加交叉熵做组合损失；2) 平滑项 $\epsilon$ 太小 → 增大到 $10^{-5}$ 或 $10^{-4}$。

**Q: 为什么我的 U-Net 只预测背景，不预测目标？**

A: 最可能的原因是类别极端不平衡 + 学习率太大。先用 Dice 损失（它对小目标更敏感），检查学习率是否合适（用 lr finder），确认数据增强没把目标区域完全扭曲掉。

**Q: 3D U-Net 太吃内存怎么办？**

A: 三个策略：1) 用 patch-based 训练（把体积切成小块）；2) 用混合精度训练减少一半显存；3) 用深度可分离卷积减少参数量。

## 推荐资源

### 英文

- Ronneberger et al., "U-Net: Convolutional Networks for Biomedical Image Segmentation", MICCAI 2015 — 原始论文
- Zhou et al., "UNet++: A Nested U-Net Architecture for Medical Image Segmentation", 2018
- Oktay et al., "Attention U-Net: Learning Where to Look for the Pancreas", MIDL 2018
- 3D U-Net: Çiçek et al., "3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation", MICCAI 2016

### 中文

- 知乎专栏"U-Net 系列解读"系列文章
- 《动手学深度学习》第 13 章"语义分割"
- PyTorch 官方教程中的 U-Net 实现

## 参考文献

```{bibliography}
:filter: docname in docnames
```

**本章完。**
