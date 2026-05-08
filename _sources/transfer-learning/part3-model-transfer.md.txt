(transfer-learning-model)=
# 基于模型的迁移

{ref}`transfer-learning-taxonomy` 中，我们学习了迁移学习的分类体系。在实际工程中，**基于模型的迁移**是最常用、最有效的方法——直接复用预训练模型的参数和网络结构。

{doc}`../neural-network-basics/cnn-basics` 中我们讨论了 {ref}`inductive-bias`：CNN 通过架构设计将"局部相关性"的先验知识内置到模型中。预训练模型则更进一步——它不仅包含架构设计，还包含从海量数据中学到的**权重参数**，这些参数编码了边缘、纹理、形状等通用视觉特征。

**关键问题**：如何有效地利用这些预训练权重？是直接用它们提取特征，还是在目标任务上继续微调？本章将讲解两种核心策略及其适用场景。

## 预训练模型

### 预训练-微调范式

2018年，BERT {cite}`devlin2018bert` 和 GPT {cite}`radford2018improving` 的出现彻底改变了深度学习的研究范式：

1. **预训练阶段**：在大规模数据上学习通用表示
2. **微调阶段**：在下游任务的小规模标注数据上调整模型

预训练模型学到的通用特征具有强大的迁移能力——视觉模型学习边缘、纹理、形状等低级特征，语言模型学习语法、语义、世界知识。

### 经典预训练模型

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

### 分层特征学习——网络的"视觉系统"

深度神经网络就像一个分层的视觉系统，每一层负责不同层次的特征识别：

#### 生活中的类比：识别一只猫

想象你走进一个房间，看到一只猫。你的大脑如何处理这个视觉信息？

~~~{mermaid}
flowchart LR
    A[视网膜接收<br/>光线信号] --> B[边缘检测<br/>「这是轮廓」]
    B --> C[纹理识别<br/>「毛茸茸的」]
    C --> D[形状组合<br/>「有耳朵尾巴」]
    D --> E[物体识别<br/>「这是一只猫」]
    E --> F[语义理解<br/>「可能是宠物」]
    
    style A fill:#e3f2fd
    style B fill:#e8f5e9
    style C fill:#e8f5e9
    style D fill:#fff3e0
    style E fill:#ffebee
    style F fill:#f3e5f5
~~~

**神经网络也是这样的分层结构**：

| 层级 | 学到的特征 | 通用性 | 迁移时的处理 |
|-----|-----------|--------|-------------|
| **浅层**（Layer1-2）| 边缘、颜色、简单纹理 | 🔵 最通用 | 🟢 保留，不训练 |
| **中层**（Layer3）| 形状、纹理组合 | 🟡 较通用 | 🟡 轻微调整 |
| **深层**（Layer4）| 物体部件、复杂模式 | 🟠 任务相关 | 🟠 适度训练 |
| **分类层**（FC）| 具体类别判断 | 🔴 最特定 | 🔴 完全替换 |

**关键洞察**：
- 浅层学到的"边缘检测"在任何视觉任务中都有用（猫的边缘和汽车的边缘是一样的）
- 深层学到的是"这像猫"的判断，换任务可能就不适用了
- **这就是为什么迁移学习能工作**：底层技能通用，只需调整上层应用

## 策略一：特征提取器

**核心思想**：冻结预训练模型所有层，仅训练新的分类器。

**工作流程**：

```python
import torchvision.models as models
import torch.nn as nn

# 加载预训练模型（来自 ImageNet 训练的 ResNet50）
# 这些权重编码了边缘、纹理、形状等通用视觉特征
model = models.resnet50(weights='IMAGENET1K_V1')

# 冻结所有层：requires_grad=False 表示不计算这些参数的梯度
# 这会"锁定"预训练知识，防止在训练中被改变
for param in model.parameters():
    param.requires_grad = False

# 替换分类头：原模型输出 1000 类（ImageNet），替换为目标类别数
# 只有这一层是新初始化的，需要从头学习
num_features = model.fc.in_features  # 获取全连接层输入维度（ResNet50 是 2048）
model.fc = nn.Linear(num_features, num_classes)  # num_classes 是你的任务类别数

# 优化器只传入新分类层的参数
# 见 {doc}`../pytorch-practice/optimiser` 中优化器参数设置
optimizer = torch.optim.Adam(model.fc.parameters(), lr=0.001)
```

**适用场景**：
- ✅ 目标数据集很小（<1000样本）
- ✅ 目标域与预训练域高度相似
- ✅ 需要快速原型验证

**优势**：训练快、不易过拟合、计算资源要求低

## 策略二：微调

**核心思想**：在预训练权重基础上继续训练，可微调全部或部分层。

### 全量微调

```python
import torchvision.models as models
import torch.nn as nn

# 加载预训练模型
model = models.resnet50(weights='IMAGENET1K_V1')

# 替换分类头为目标类别数
# 新分类层随机初始化，其他层保留预训练权重
model.fc = nn.Linear(model.fc.in_features, num_classes)

# 所有参数参与训练（不冻结任何层）
# 学习率通常比从头训练小一个数量级（1e-4 vs 1e-3）
# 防止过大的更新破坏预训练知识
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
```

### 分层微调——不同层用不同"学习速度"

**直觉理解**：

> 想象你在学习开车（新任务），但你已经会骑自行车（预训练）。
> 
> - **平衡感**（浅层）：骑车和开车都需要的技能 → 基本不用改
> - **转向控制**（中层）：原理相似但操作不同 → 稍微调整
> - **油门刹车**（深层）：全新的操作方式 → 重点学习
> - **交通规则**（分类层）：完全不同的知识 → 从头学起

分层微调就是给不同层设置不同的"学习速度"：

```python
import torch.optim as optim

# 分层学习率配置
# 原理：浅层（layer1-2）学习边缘/纹理等通用特征，应尽量保持不变
#      深层（layer3-4）学习物体部件等半通用特征，可适度调整
#      分类层（fc）完全针对新任务，需要大幅学习
optimizer = optim.Adam([
    {'params': model.layer1.parameters(), 'lr': 1e-5},  # 最浅层，最小学习率
    {'params': model.layer2.parameters(), 'lr': 1e-5},  # 浅层特征
    {'params': model.layer3.parameters(), 'lr': 1e-4},  # 中层特征
    {'params': model.layer4.parameters(), 'lr': 1e-4},  # 深层特征
    {'params': model.fc.parameters(), 'lr': 1e-3}       # 新分类层，最大学习率
])

# 这种配置平衡了稳定性（浅层不变）和适应性（深层可调）
# 参见 {doc}`../pytorch-practice/optimiser` 中参数组配置
```

**学习率设置的直觉**：

| 层级 | 学习率 | 为什么这样设置 | 类比 |
|-----|--------|---------------|------|
| Layer1-2 | 1e-5（很小） | 边缘检测技能通用，别破坏它 | 平衡感，骑车开车都一样 |
| Layer3-4 | 1e-4（中等） | 形状识别可以微调 | 转向控制，原理相似 |
| FC层 | 1e-3（正常） | 分类头针对新任务，需要大学习 | 交通规则，全新知识 |

### 渐进解冻（Progressive Unfreezing）——循序渐进地学习

**直觉理解**：

> 想象你在准备数学竞赛（新任务），但你已经学完了高中数学（预训练）。
> 
> **渐进解冻就像是分阶段复习**：
> - **第1阶段**：只刷竞赛真题（只训练分类层）
> - **第2阶段**：复习竞赛核心技巧（解冻深层）
> - **第3阶段**：回顾相关高中知识（解冻中层）
> - **第4阶段**：全面复习所有基础（解冻浅层）
> 
> 如果一开始就全面复习，可能会因为内容太多而混乱。**分阶段解冻让学习更有序**。

**核心思想**：
- 深层特征更接近任务特定，应该优先调整
- 浅层特征更通用，应该后期才解冻
- 逐步解冻避免了"一次性放开所有参数"导致的训练不稳定

**实现步骤**：

```python
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim

def get_optimizer_for_layers(model, unfrozen_layers, base_lr=1e-4):
    """
    为不同层组设置参数
    unfrozen_layers: 要训练的层名称列表
    """
    params_to_update = []
    for name, param in model.named_parameters():
        # 检查参数是否属于解冻的层
        is_unfrozen = any(layer in name for layer in unfrozen_layers)
        if is_unfrozen:
            param.requires_grad = True
            # 分类层使用更大的学习率
            if 'fc' in name:
                params_to_update.append({'params': param, 'lr': base_lr * 10})
            else:
                params_to_update.append({'params': param, 'lr': base_lr})
        else:
            param.requires_grad = False
    
    return optim.Adam(params_to_update)

# 加载预训练模型
model = models.resnet50(weights='IMAGENET1K_V1')
model.fc = nn.Linear(model.fc.in_features, num_classes)

# 定义ResNet50的层结构（从浅到深）
# 见 {ref}`transfer-learning-model` 中分层特征学习的讨论
layer_groups = [
    ['fc'],                    # 阶段1：只训练分类层
    ['layer4', 'fc'],          # 阶段2：解冻最深特征层
    ['layer3', 'layer4', 'fc'], # 阶段3：解冻中层
    ['layer2', 'layer3', 'layer4', 'fc'],  # 阶段4：解冻更多
    ['layer1', 'layer2', 'layer3', 'layer4', 'fc']  # 阶段5：全部解冻
]

# 渐进解冻训练
epochs_per_stage = 5  # 每个阶段训练5个epoch
for stage, layers in enumerate(layer_groups):
    print(f"阶段 {stage + 1}: 训练层 {layers}")
    
    # 为当前阶段创建优化器
    optimizer = get_optimizer_for_layers(model, layers, base_lr=1e-5)
    
    # 训练当前阶段
    for epoch in range(epochs_per_stage):
        train_epoch(model, train_loader, optimizer, criterion)
        val_acc = validate(model, val_loader)
        print(f"  Epoch {epoch+1}/{epochs_per_stage}, Val Acc: {val_acc:.2f}%")
```

**为什么从顶层开始解冻？**

~~~{mermaid}
flowchart TB
    subgraph 浅层 [浅层: 边缘/颜色/纹理]
        L1[Layer1]
        L2[Layer2]
    end
    
    subgraph 中层 [中层: 形状/部件]
        L3[Layer3]
    end
    
    subgraph 深层 [深层: 物体/语义]
        L4[Layer4]
        FC[分类层]
    end
    
    Input[输入图像] --> L1 --> L2 --> L3 --> L4 --> FC
    
    style L1 fill:#e3f2fd
    style L2 fill:#e3f2fd
    style L3 fill:#fff3e0
    style L4 fill:#ffebee
    style FC fill:#ffebee
~~~

- **Layer1-2**（浅层）：学习边缘、颜色、纹理——**最通用**，应该最后解冻
- **Layer3**（中层）：学习形状、部件——**半通用**，中期解冻
- **Layer4+FC**（深层）：学习物体、语义——**最任务相关**，最先解冻

**渐进解冻 vs 分层学习率**：

| 方法 | 原理 | 训练速度 | 稳定性 | 适用场景 |
|------|------|---------|--------|---------|
| 分层学习率 | 所有层同时训练，但用不同lr | 快 | 中等 | 大多数场景 |
| 渐进解冻 | 逐层解冻，分阶段训练 | 慢 | 高 | 数据集小、对稳定性要求高 |

**优势**：避免对预训练权重的剧烈扰动，训练更稳定，特别适合小数据集。

### 适用场景

- ✅ 目标数据集中等规模（1000-10000+样本）
- ✅ 目标域与预训练域有一定差异
- ✅ 追求最佳性能

**学习率设置原则**：
- 预训练权重：1e-5 到 1e-4
- 新初始化层：1e-3 到 1e-2
- 通常为从头训练学习率的1/10到1/100

## 策略选择

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

## 参数高效微调（PEFT）

随着模型规模急剧增长，参数高效微调成为研究热点。

### LoRA——只训练"差异"而不是全部

**直觉理解**：

> 想象你要学一门新方言（新任务），但你已经会普通话（预训练）。
> 
> **传统微调** = 重新学说话（调整所有参数）
> **LoRA** = 只学「普通话和方言的差异」（只调整少量参数）
> 
> 比如：
> - 普通话："吃饭了吗？"
> - 方言："食咗饭未？"
> - **差异**：发音不同，但语法结构相似
> 
> LoRA 就是只学习这些"差异规则"，而不是重新学习整个语言。

**技术原理**（简化版）：

假设预训练模型的某个权重是 $W_0$（已经训练好的）。LoRA 不直接修改 $W_0$，而是学习一个"调整量"：

$$W = W_0 + \underbrace{BA}_{\text{小的调整}}$$

其中 $B$ 和 $A$ 是两个小矩阵，它们的乘积 $BA$ 表示"如何调整"。

**为什么这样有效？**
- 原来的 $W_0$ 有 $d \times k$ 个参数（比如100万）
- $BA$ 只有 $(d \times r) + (r \times k)$ 个参数（比如1万，因为 $r$ 很小）
- **只训练1%的参数，达到相似的效果**

```python
from peft import LoraConfig, get_peft_model

# r=16 表示只学习16个"调整方向"
# 就像只学16条方言转换规则，而不是重新学所有词汇
config = LoraConfig(r=16, target_modules=["q_proj", "v_proj"])
model = get_peft_model(base_model, config)
```

**适用场景**：
- 模型太大，显卡存不下全部梯度
- 需要训练很多个不同任务（每个任务只存小的BA矩阵）
- 想快速实验不同任务（训练更快）

### 其他PEFT方法

| 方法 | 可训练参数 | 适用场景 |
|------|-----------|----------|
| LoRA | ~0.1-1% | 性能与效率平衡（推荐） |
| Adapter | ~2-4% | 多任务/持续学习 |
| Prefix Tuning | ~0.1% | 极端资源受限 |
| BitFit | ~0.1% | 快速实验验证 |

## 本章小结

- **特征提取**：冻结预训练层，只训练分类头，适合小数据集
- **微调**：继续训练预训练权重，适合中大数据集
  - 分层学习率：浅层小lr，深层大lr
  - 渐进解冻：逐层解冻，训练更稳定
- **PEFT**：LoRA等方法大幅降低训练成本

### 下一步

掌握了特征提取与微调的理论后，{doc}`part4-practical-guide` 我们将进入**实战环节**——如何判断你的数据集适合哪种策略？如何避免灾难性遗忘和过拟合？学习率到底该设多少？这些都是实践中必须面对的问题。

---

## 参考文献

```{bibliography}
:filter: docname in docnames
```
