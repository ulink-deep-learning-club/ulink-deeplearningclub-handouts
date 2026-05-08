(transfer-learning-practical)=
# 实操指南

{ref}`transfer-learning-model` 中我们学习了特征提取和微调两种策略。但实践中还有很多具体问题：

- 我的数据集应该用特征提取还是微调？
- 学习率设置多大？冻结哪些层？
- 为什么验证集准确率比训练集低这么多？

**本章就是深度学习的"避坑指南"**——聚焦实际应用中的关键决策点和常见陷阱，帮助你在真实项目中少走弯路。

## 数据集相似性判断

1. 图像领域

    | 目标领域 | 迁移难度 | 建议 |
    |----------|----------|------|
    | 自然图像分类 | 低 | 特征提取或轻微微调 |
    | 医学影像 | 中等 | 医学专用预训练模型 + 强数据增强 |
    | 卫星遥感 | 中等 | 遥感专用预训练模型 |
    | 工业缺陷 | 高 | 特征提取 + 数据增强 + 强正则化 |

2. 任务粒度

    - **粗粒度分类**（猫 vs 狗）：域相似度高，特征提取即可
    - **中等粒度**（犬种分类）：部分微调
    - **细粒度分类**（鸟类物种）：全量微调 + 数据增强

3. 实用判断方法

    1. **直接测试**：用预训练模型提取特征，训练简单分类器，性能>70%说明相似度高
    2. **对比实验**：同时尝试特征提取和微调，选择验证集性能更好的策略

## 常见陷阱与对策

1. 灾难性遗忘——学新忘旧

    **直觉理解**：

    > 想象你学了很多年英语（预训练），然后突然开始密集学法语（微调）。
    > 
    > 如果学得太猛，你可能会发现：
    > - ✅ 法语进步很快
    > - ❌ 英语开始遗忘（以前记得的单词想不起来了）
    > 
    > 这就是**灾难性遗忘**——新任务学得越好，旧知识忘得越多。

    **为什么会发生？**

    神经网络的权重就像是"记忆"。当你用新数据训练时，权重会被更新。如果学习率太大，新数据会"覆盖"旧记忆。

    **生活中的类比**：

    > 你的大脑有100个抽屉存储知识：
    > - 预训练时：英语知识分散存储在80个抽屉里
    > - 微调时：如果暴力地把法语知识塞进这些抽屉
    > - 结果：英语知识被挤掉了（灾难性遗忘）
    > 
    > **解决方案**：
    > - 只打开几个抽屉放法语（小学习率）
    > - 或者只增加新的小抽屉（LoRA）

    **缓解策略**：
    1. **使用较小学习率**（预训练权重：1e-5 到 1e-4）
    - 就像学法语时，每天只学1小时，保留英语的时间
    2. **分层学习率**（浅层更小lr）
    - 基础的语法知识（浅层）不改动，只调整词汇用法（深层）
    3. **使用PEFT方法**（LoRA、Adapter）
    - 不为法语占用原有抽屉，而是新建一个小抽屉专门存差异

2. 过拟合——死记硬背而不是真正学会

    **直觉理解**：

    > 想象你在准备数学考试，但只有10道练习题（小数据集）。
    > 
    > **过拟合的表现**：
    > - 你把这10道题的答案背得滚瓜烂熟（训练准确率99%）
    > - 考试时遇到新题目，完全不会做（验证准确率50%）
    > - 你只是"记住"了答案，没有"学会"解题方法

    **为什么迁移学习容易过拟合？**

    预训练模型有2500万个参数（ResNet50），而你的数据集可能只有100张图片。**参数太多，数据太少**——模型有足够的"记忆力"把每张图都背下来。

    **识别信号**：
    - 训练准确率持续上升（模型在学答案）
    - 验证准确率停滞或下降（泛化能力差）
    - 两者的差距越来越大（过拟合严重）

    **应对策略**：

    | 策略 | 直觉解释 | 怎么做 |
    |-----|---------|--------|
    | **数据增强** | 让10道题变100道（变换角度、裁剪） | 随机裁剪、翻转、颜色抖动 |
    | **早停** | 不要背太狠，适可而止 | 验证不提升就停止 |
    | **正则化** | 惩罚"死记硬背"，鼓励"理解" | weight_decay=1e-4 |
    | **减少参数** | 减少"记忆力"，迫使"理解" | 特征提取或LoRA |

3. 学习率设置不当

    - **学习率过大**：损失震荡不收敛，破坏预训练权重 → 降低学习率
    - **学习率过小**：收敛极慢，陷入局部最优 → 增大学习率，使用调度器

    ```python
    import torch.optim as optim

    # 学习率调度器：当验证损失不再下降时自动降低学习率
    # 这在微调时特别有用，防止后期学习率过大破坏预训练权重
    # 参见 {ref}`learning-rate-scheduling` 中学习率调整策略
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,      # 要调整的优化器
        mode='min',     # 监控指标越小越好（如损失）
        factor=0.1,     # 学习率衰减因子（新lr = 旧lr × 0.1）
        patience=5      # 等待5个epoch，如果指标不改善就降低学习率
    )

    # 在训练循环中使用
    for epoch in range(num_epochs):
        train(...)           # 训练
        val_loss = validate(...)  # 验证
        scheduler.step(val_loss)  # 根据验证损失调整学习率
    ```

4. 预处理不一致

    预训练模型对输入数据有特定预处理要求，必须匹配：

    ```python
    from torchvision import transforms

    # ImageNet标准化参数（必须与预训练时使用的完全一致）
    # 这些参数来自 ImageNet 数据集统计：mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    # 参见 {doc}`../pytorch-practice/train-workflow` 中数据预处理部分
    transform = transforms.Compose([
        transforms.Resize(256),      # 短边调整为256，保持长宽比
        transforms.CenterCrop(224),  # 中心裁剪为224×224（ResNet输入尺寸）
        transforms.ToTensor(),       # PIL Image → Tensor，并归一化到[0,1]
        transforms.Normalize(        # 标准化：(x - mean) / std
            mean=[0.485, 0.456, 0.406],   # ImageNet R,G,B 通道均值
            std=[0.229, 0.224, 0.225]     # ImageNet R,G,B 通道标准差
        )
    ])
    ```

    **关键检查点**：
    - [ ] 输入尺寸匹配（通常224×224）
    - [ ] 使用正确的归一化参数（必须与预训练一致）
    - [ ] 颜色通道顺序正确（RGB，不是BGR）

## 项目检查清单

### 项目启动前
- [ ] 评估目标数据集大小与质量
- [ ] 判断与预训练域的相似性
- [ ] 选择合适的预训练模型
- [ ] 确定迁移策略

### 模型构建
- [ ] 正确加载预训练权重
- [ ] 替换输出层为目标类别数
- [ ] 冻结/解冻正确的层
- [ ] 实现正确的数据预处理

### 训练阶段
- [ ] 设置合理的学习率（分层或统一）
- [ ] 添加早停机制
- [ ] 使用数据增强
- [ ] 监控训练和验证指标

### 评估迭代
- [ ] 在独立测试集上评估
- [ ] 分析错误样本
- [ ] 对比不同策略效果

## 高级技巧

### 测试时增强（TTA）

**核心思想**：对同一张测试图像应用不同的数据增强，取多个预测的平均，降低预测方差。

**为什么有效？**
- 模型对图像的微小变化敏感
- 多视角预测取平均，减少随机性
- 类似于"投票机制"，更稳定

```python
import torch
import torchvision.transforms as transforms
from PIL import Image

def tta_predict(model, image_path, num_augmentations=5):
    """
    测试时增强预测
    
    Args:
        model: 训练好的模型
        image_path: 测试图像路径
        num_augmentations: TTA增强次数
    
    Returns:
        平均预测概率
    """
    model.eval()
    
    # 基础预处理（必须保持与训练一致）
    base_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # TTA增强变换（在基础变换上增加随机性）
    tta_transforms = [
        # 原始图像
        base_transform,
        # 水平翻转
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(p=1.0),  # 必定翻转
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        # 轻微旋转
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.RandomRotation(degrees=(-5, 5)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        # 偏移裁剪（左上角）
        transforms.Compose([
            transforms.Resize(256),
            transforms.Lambda(lambda img: transforms.functional.crop(img, 0, 0, 224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        # 偏移裁剪（右下角）
        transforms.Compose([
            transforms.Resize(256),
            transforms.Lambda(lambda img: transforms.functional.crop(img, 32, 32, 224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    ]
    
    # 加载原始图像
    image = Image.open(image_path).convert('RGB')
    
    # 收集所有预测
    all_predictions = []
    with torch.no_grad():
        for transform in tta_transforms[:num_augmentations]:
            input_tensor = transform(image).unsqueeze(0).to(device)
            output = model(input_tensor)
            probs = torch.softmax(output, dim=1)
            all_predictions.append(probs)
    
    # 取平均
    avg_prediction = torch.stack(all_predictions).mean(dim=0)
    return avg_prediction

# 使用示例
prediction = tta_predict(model, 'test_image.jpg', num_augmentations=5)
predicted_class = prediction.argmax(dim=1).item()
confidence = prediction.max().item()
print(f"预测类别: {predicted_class}, 置信度: {confidence:.4f}")
```

**TTA 使用建议**：

| 场景 | 推荐增强数 | 预期提升 |
|------|-----------|---------|
| 追求速度 | 2（原图+水平翻转） | 0.5-1% |
| 平衡方案 | 5（上述代码） | 1-2% |
| 追求精度 | 10+（更多裁剪/旋转） | 2-3% |

**注意事项**：
- TTA只在**测试阶段**使用，不增加训练时间
- 对于已经过拟合的模型，TTA提升有限
- 分类任务效果最明显，检测/分割任务需要适配

### 集成学习

使用多个预训练模型集成，进一步提升性能：

```python
import torch
import torchvision.models as models

def ensemble_predict(model_paths, image_path, weights=None):
    """
    多模型集成预测
    
    Args:
        model_paths: 多个模型路径列表
        image_path: 测试图像
        weights: 各模型权重（None表示等权重）
    """
    # 加载多个模型
    models_list = []
    for path in model_paths:
        model = models.resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        model.load_state_dict(torch.load(path))
        model.eval()
        models_list.append(model.to(device))
    
    # 预处理
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0).to(device)
    
    # 收集各模型预测
    all_probs = []
    with torch.no_grad():
        for model in models_list:
            output = model(input_tensor)
            probs = torch.softmax(output, dim=1)
            all_probs.append(probs)
    
    # 加权平均
    if weights is None:
        weights = [1.0 / len(models_list)] * len(models_list)
    
    ensemble_prob = sum(w * p for w, p in zip(weights, all_probs))
    return ensemble_prob

# 使用不同架构集成（更强的集成效果）
# ResNet50 + EfficientNet-B0 + ViT-Base
ensemble_models = [
    'resnet50_best.pth',
    'efficientnet_best.pth',
    'vit_best.pth'
]

prediction = ensemble_predict(ensemble_models, 'test.jpg')
```

**集成策略对比**：

| 集成类型 | 方法 | 效果 | 计算成本 |
|---------|------|------|---------|
| 同架构不同初始化 | 相同模型，不同随机种子 | ★★☆ | 低 |
| 不同架构 | ResNet + EfficientNet + ViT | ★★★ | 中 |
| 不同预训练数据 | ImageNet + 医学数据 + 领域数据 | ★★★ | 高 |
| TTA + 集成 | 先TTA再集成 | ★★★★ | 很高 |

**集成学习的前提**：
- 单个模型性能不能太差（至少>70%）
- 模型间要有**差异性**（不能是完全相同的预测）
- 模型数量不是越多越好，通常3-5个足够

### 项目案例：医学影像分类实战

下面通过一个完整案例，展示迁移学习在真实项目中的应用。

#### 问题背景

**任务**：从胸部X光片中识别肺炎
**数据**：
- 训练集：正常 1,341张，肺炎 3,876张（类别不平衡）
- 验证集：正常 234张，肺炎 390张
- 测试集：正常 234张，肺炎 390张
- 图像尺寸：原始1024×1024，预处理为224×224

**挑战**：
1. 数据量小（总计约5,000张）
2. 类别不平衡（正常:肺炎 ≈ 1:3）
3. 医学图像与自然图像差异大

#### 方案设计与实现

~~~{mermaid}
flowchart LR
    A[数据准备] --> B[选择预训练模型]
    B --> C[迁移策略]
    C --> D[训练优化]
    D --> E[评估测试]
    
    A1[数据增强<br/>类别平衡] --> A
    B1[ResNet50<br/>ImageNet预训练] --> B
    C1[分层微调<br/>渐进解冻] --> C
    D1[Focal Loss<br/>学习率调度] --> D
    E1[TTA<br/>混淆矩阵分析] --> E
~~~

##### 步骤1：数据准备与增强

```python
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, WeightedRandomSampler

# 训练时增强（强增强）
train_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=(-10, 10)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 验证/测试时预处理（无增强）
val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 处理类别不平衡：计算每个类别的权重
class_counts = [1341, 3876]  # 正常, 肺炎
class_weights = 1. / torch.tensor(class_counts, dtype=torch.float)
sample_weights = [class_weights[label] for _, label in train_dataset]
sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

train_loader = DataLoader(train_dataset, batch_size=32, sampler=sampler)
```

##### 步骤2：模型构建（分层微调）

```python
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim

# 加载预训练ResNet50
model = models.resnet50(weights='IMAGENET1K_V1')

# 替换分类头为2类（正常/肺炎）
model.fc = nn.Linear(model.fc.in_features, 2)

# 分层学习率配置
# 医学影像与自然影像差异较大，需要适度调整浅层
optimizer = optim.Adam([
    {'params': model.layer1.parameters(), 'lr': 1e-5},
    {'params': model.layer2.parameters(), 'lr': 1e-5},
    {'params': model.layer3.parameters(), 'lr': 5e-5},
    {'params': model.layer4.parameters(), 'lr': 1e-4},
    {'params': model.fc.parameters(), 'lr': 1e-3}
])

# 使用Focal Loss处理类别不平衡
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()

criterion = FocalLoss(alpha=0.25, gamma=2.0)
```

##### 步骤3：训练循环与早停

```python
best_val_acc = 0
patience = 5
patience_counter = 0

for epoch in range(50):  # 最多训练50轮
    # 训练阶段
    model.train()
    train_loss = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
    
    # 验证阶段
    model.eval()
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            val_total += labels.size(0)
            val_correct += predicted.eq(labels).sum().item()
    
    val_acc = 100. * val_correct / val_total
    print(f"Epoch {epoch}: Train Loss={train_loss/len(train_loader):.4f}, Val Acc={val_acc:.2f}%")
    
    # 早停机制
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), 'best_model.pth')
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"早停触发，最佳验证准确率: {best_val_acc:.2f}%")
            break
```

##### 步骤4：评估与TTA

```python
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np

# 加载最佳模型
model.load_state_dict(torch.load('best_model.pth'))
model.eval()

# 使用TTA进行测试
def evaluate_with_tta(model, test_loader, num_tta=5):
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            
            # TTA预测
            batch_preds = []
            for _ in range(num_tta):
                # 添加轻微噪声进行增强
                noisy_images = images + torch.randn_like(images) * 0.01
                outputs = model(noisy_images)
                probs = torch.softmax(outputs, dim=1)
                batch_preds.append(probs)
            
            # 平均预测
            avg_probs = torch.stack(batch_preds).mean(dim=0)
            _, predicted = avg_probs.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    return np.array(all_preds), np.array(all_labels)

predictions, labels = evaluate_with_tta(model, test_loader, num_tta=5)

# 打印详细评估报告
print("分类报告：")
print(classification_report(labels, predictions, target_names=['正常', '肺炎']))

# 混淆矩阵
cm = confusion_matrix(labels, predictions)
print("混淆矩阵：")
print(cm)
```

#### 实验结果

| 方法 | 准确率 | 召回率（肺炎） | F1分数 |
|------|--------|---------------|--------|
| 从头训练 | 82.3% | 78.5% | 0.81 |
| 特征提取 | 85.1% | 82.3% | 0.84 |
| **分层微调（本方案）** | **91.7%** | **90.2%** | **0.91** |
| + TTA | 92.4% | 91.0% | 0.92 |

**关键洞察**：
1. 分层学习率有效平衡了预训练知识的保留与新任务的适应
2. Focal Loss显著改善了类别不平衡问题
3. TTA在测试阶段提供了稳定的性能提升

#### 调试记录

**问题1**：初期验证准确率停留在75%左右
- **诊断**：学习率过大，破坏了预训练权重
- **解决**：将预训练层学习率从1e-3降至1e-5

**问题2**：肺炎类别召回率低（仅70%）
- **诊断**：类别不平衡导致模型偏向多数类
- **解决**：使用WeightedRandomSampler + Focal Loss

**问题3**：训练集准确率>95%，验证集<80%
- **诊断**：过拟合，数据增强不够
- **解决**：增加ColorJitter和RandomRotation

### 持续学习

模型需持续适应新任务时：
- 使用PEFT方法（LoRA、Adapter）
- 考虑经验回放或EWC等持续学习方法 {cite}`kirkpatrick2017overcoming`

## 本章小结

**数据集相似性判断**：内容相似性、任务粒度、量化方法

**常见陷阱**：
- 灾难性遗忘 → 小学习率、PEFT
- 过拟合 → 数据增强、早停、正则化
- 学习率问题 → 分层学习率、调度器
- 预处理 → 使用预训练时的预处理参数

**核心思想**：站在巨人的肩膀上，善用预训练知识。

### 下一步

完成了实操指南的学习，你已经掌握了迁移学习的完整技能栈。{doc}`the-end` 中，我们将回顾本章知识、整理学习资源、推荐动手项目，并展望接下来的学习路径。

---

## 参考文献

```{bibliography}
:filter: docname in docnames
```
