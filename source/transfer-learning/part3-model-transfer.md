# 第三部分：基于模型的迁移

基于模型的迁移是当前深度学习实践中最主流的方法。本章讲解预训练模型的原理和两种核心策略。

## 3.1 预训练模型

### 3.1.1 预训练-微调范式

2018年，BERT {cite}`devlin2018bert` 和 GPT {cite}`radford2018improving` 的出现彻底改变了深度学习的研究范式：

1. **预训练阶段**：在大规模数据上学习通用表示
2. **微调阶段**：在下游任务的小规模标注数据上调整模型

预训练模型学到的通用特征具有强大的迁移能力——视觉模型学习边缘、纹理、形状等低级特征，语言模型学习语法、语义、世界知识。

### 3.1.2 经典预训练模型

**计算机视觉**：

| 模型 | 参数量 | 特点 |
|------|--------|------|
| ResNet-50 | 25M | 残差连接，易于训练 {cite}`he2016deep` |
| EfficientNet-B0 | 5.3M | 复合缩放，效率最优 |
| ViT-Base | 86M | Transformer架构 |
| CLIP | 400M+ | 图文对齐，零样本能力 {cite}`radford2021learning` |

**自然语言处理**：

| 模型 | 参数量 | 特点 |
|------|--------|------|
| BERT-Base | 110M | 双向Transformer {cite}`devlin2018bert` |
| GPT-2 | 1.5B | 生成式预训练 {cite}`radford2019language` |
| GPT-3 | 175B | 大规模生成模型 {cite}`brown2020language` |
| LLaMA-2 | 7B-70B | 开源大语言模型 {cite}`touvron2023llama` |

### 3.1.3 分层特征学习

深度神经网络的特征具有层级性：

- **浅层**（靠近输入）：学习边缘、颜色、纹理等**通用**特征
- **深层**（靠近输出）：学习物体、语义等**任务特定**特征

这一特性决定了迁移学习中的策略选择——浅层特征可复用，深层特征需调整。

## 3.2 策略一：特征提取器

**核心思想**：冻结预训练模型所有层，仅训练新的分类器。

**工作流程**：

```python
import torchvision.models as models

# 加载预训练模型
model = models.resnet50(weights='IMAGENET1K_V1')

# 冻结所有层
for param in model.parameters():
    param.requires_grad = False

# 替换分类头
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, num_classes)

# 仅训练新分类层
optimizer = torch.optim.Adam(model.fc.parameters(), lr=0.001)
```

**适用场景**：
- ✅ 目标数据集很小（<1000样本）
- ✅ 目标域与预训练域高度相似
- ✅ 需要快速原型验证

**优势**：训练快、不易过拟合、计算资源要求低

## 3.3 策略二：微调

**核心思想**：在预训练权重基础上继续训练，可微调全部或部分层。

### 3.3.1 全量微调

```python
# 加载预训练模型
model = models.resnet50(weights='IMAGENET1K_V1')
model.fc = nn.Linear(model.fc.in_features, num_classes)

# 所有参数参与训练
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
```

### 3.3.2 分层微调

为不同层设置不同学习率，浅层用小学习率保持预训练知识，深层用大学习率适应新任务：

```python
optimizer = torch.optim.Adam([
    {'params': model.layer1.parameters(), 'lr': 1e-5},
    {'params': model.layer2.parameters(), 'lr': 1e-5},
    {'params': model.layer3.parameters(), 'lr': 1e-4},
    {'params': model.layer4.parameters(), 'lr': 1e-4},
    {'params': model.fc.parameters(), 'lr': 1e-3}
])
```

### 3.3.3 渐进解冻

从顶层开始逐步解冻各层：

1. 先冻结所有层，只训练新分类头
2. 解冻倒数第二层，继续训练（较小学习率）
3. 逐步解冻更多层

**优势**：避免对预训练权重的剧烈扰动，训练更稳定。

### 3.3.4 适用场景

- ✅ 目标数据集中等规模（1000-10000+样本）
- ✅ 目标域与预训练域有一定差异
- ✅ 追求最佳性能

**学习率设置原则**：
- 预训练权重：1e-5 到 1e-4
- 新初始化层：1e-3 到 1e-2
- 通常为从头训练学习率的1/10到1/100

## 3.4 策略选择

```{mermaid}
---
caption: 迁移学习策略选择流程
---
graph TD
    A[开始] --> B{目标数据集大小？}
    B -->|小<br/><1000| C{域相似度高？}
    B -->|中等<br/>1000-10000| D{域相似度高？}
    B -->|大<br/>>10000| E[全量微调]
    
    C -->|是| F[特征提取]
    C -->|否| G[数据增强 + 微调]
    
    D -->|是| H[部分微调]
    D -->|否| I[全量微调 + 分层学习率]
    
    E --> J[性能最佳]
    F --> K[快速稳定]
    G --> L[有挑战性]
    H --> M[平衡选择]
    I --> N[充分适应]
```

## 3.5 参数高效微调（PEFT）

随着模型规模急剧增长，参数高效微调成为研究热点。

### 3.5.1 LoRA

LoRA {cite}`hu2021lora` 在原始权重旁路添加低秩矩阵：

```{math}
W = W_0 + BA, \quad B \in \mathbb{R}^{d \times r}, A \in \mathbb{R}^{r \times k}, r \ll \min(d,k)
```

**优势**：参数量减少到原来的1-10%，不增加推理延迟。

```python
from peft import LoraConfig, get_peft_model

config = LoraConfig(r=16, target_modules=["q_proj", "v_proj"])
model = get_peft_model(base_model, config)
```

### 3.5.2 其他PEFT方法

| 方法 | 可训练参数 | 适用场景 |
|------|-----------|----------|
| LoRA | ~0.1-1% | 性能与效率平衡（推荐） |
| Adapter | ~2-4% | 多任务/持续学习 |
| Prefix Tuning | ~0.1% | 极端资源受限 |
| BitFit | ~0.1% | 快速实验验证 |

## 3.6 本章小结

- **特征提取**：冻结预训练层，只训练分类头，适合小数据集
- **微调**：继续训练预训练权重，适合中大数据集
  - 分层学习率：浅层小lr，深层大lr
  - 渐进解冻：逐层解冻，训练更稳定
- **PEFT**：LoRA等方法大幅降低训练成本

---

**参考文献**

```{bibliography}
:filter: docname in docnames
```
