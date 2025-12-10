# 训练技巧与最佳实践

成功训练U-Net模型需要综合考虑数据准备、模型架构、训练策略和评估方法。本章将分享经过实践验证的训练技巧和最佳实践，帮助读者避免常见陷阱，获得最佳性能。

## 数据准备策略

### 数据预处理

#### 图像标准化

```python
def normalize_image(image):
    """医学图像标准化"""
    # 去除异常值
    image = np.clip(image, np.percentile(image, 0.5), np.percentile(image, 99.5))
    # 标准化到[0, 1]
    image = (image - image.min()) / (image.max() - image.min() + 1e-8)
    return image

def standardize_image(image):
    """标准化到零均值和单位方差"""
    mean = image.mean()
    std = image.std()
    return (image - mean) / (std + 1e-8)
```

#### 医学图像特定处理

- **CT图像**：应用窗宽窗位调整，通常使用肝脏窗（窗宽150-200，窗位40-60）
- **MRI图像**：强度归一化，去除偏置场
- **X光图像**：对比度增强，直方图均衡化

### 数据划分

```python
def split_dataset(data_dir, val_ratio=0.2, test_ratio=0.1):
    """划分训练集、验证集、测试集"""
    all_files = sorted(glob.glob(os.path.join(data_dir, '*.png')))
    
    # 确保患者级别的划分（如果文件名包含患者ID）
    patient_ids = [extract_patient_id(f) for f in all_files]
    unique_patients = list(set(patient_ids))
    
    # 随机划分患者
    random.shuffle(unique_patients)
    num_val = int(len(unique_patients) * val_ratio)
    num_test = int(len(unique_patients) * test_ratio)
    
    val_patients = unique_patients[:num_val]
    test_patients = unique_patients[num_val:num_val+num_test]
    train_patients = unique_patients[num_val+num_test:]
    
    # 根据患者ID划分文件
    train_files = [f for f in all_files if extract_patient_id(f) in train_patients]
    val_files = [f for f in all_files if extract_patient_id(f) in val_patients]
    test_files = [f for f in all_files if extract_patient_id(f) in test_patients]
    
    return train_files, val_files, test_files
```

## 训练策略

### 学习率调度

#### 余弦退火

```python
from torch.optim.lr_scheduler import CosineAnnealingLR

scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
```

#### 带热重启的余弦退火

```python
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)
```

#### 基于指标的学习率调整

```python
from torch.optim.lr_scheduler import ReduceLROnPlateau

scheduler = ReduceLROnPlateau(
    optimizer, mode='max', factor=0.5, patience=10, 
    verbose=True, threshold=0.0001, threshold_mode='abs'
)
```

### 优化器选择

#### Adam优化器

```python
optimizer = torch.optim.Adam(
    model.parameters(), 
    lr=1e-4,
    betas=(0.9, 0.999),
    eps=1e-8,
    weight_decay=1e-5
)
```

#### AdamW优化器（推荐）

```python
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-4,
    betas=(0.9, 0.999),
    eps=1e-8,
    weight_decay=0.01
)
```

#### SGD with Momentum

```python
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.01,
    momentum=0.9,
    weight_decay=1e-4,
    nesterov=True
)
```

### 早停策略

```python
class EarlyStopping:
    """早停类"""
    def __init__(self, patience=20, delta=0, mode='max'):
        self.patience = patience
        self.delta = delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
        elif (self.mode == 'max' and score < self.best_score + self.delta) or \
             (self.mode == 'min' and score > self.best_score - self.delta):
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0
            
        return self.early_stop
```

### 混合精度训练

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for epoch in range(num_epochs):
    for images, masks in train_loader:
        images, masks = images.cuda(), masks.cuda()
        
        optimizer.zero_grad()
        
        # 混合精度前向传播
        with autocast():
            outputs = model(images)
            loss = criterion(outputs, masks)
        
        # 混合精度反向传播
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
```

### 梯度累积

```python
accumulation_steps = 4  # 累积4个批次的梯度

for i, (images, masks) in enumerate(train_loader):
    images, masks = images.cuda(), masks.cuda()
    
    # 前向传播
    outputs = model(images)
    loss = criterion(outputs, masks)
    
    # 缩放损失
    loss = loss / accumulation_steps
    
    # 反向传播
    loss.backward()
    
    # 每accumulation_steps步更新一次
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

## 模型初始化

### 权重初始化

```python
def init_weights(m):
    """初始化模型权重"""
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, 0, 0.01)
        nn.init.constant_(m.bias, 0)

model.apply(init_weights)
```

### 预训练权重

```python
def load_pretrained_weights(model, pretrained_path):
    """加载预训练权重"""
    pretrained_dict = torch.load(pretrained_path)
    model_dict = model.state_dict()
    
    # 过滤不匹配的键
    pretrained_dict = {k: v for k, v in pretrained_dict.items() 
                      if k in model_dict and v.shape == model_dict[k].shape}
    
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    
    print(f'加载了 {len(pretrained_dict)}/{len(model_dict)} 层预训练权重')
    return model
```

## 评估指标

### 分割指标计算

```python
def calculate_segmentation_metrics(pred, target, threshold=0.5):
    """计算分割评估指标"""
    # 二值化
    pred_bin = (pred > threshold).float()
    target_bin = (target > threshold).float()
    
    # 基本统计
    tp = (pred_bin * target_bin).sum().item()
    fp = (pred_bin * (1 - target_bin)).sum().item()
    fn = ((1 - pred_bin) * target_bin).sum().item()
    tn = ((1 - pred_bin) * (1 - target_bin)).sum().item()
    
    # 计算指标
    metrics = {}
    
    # Dice系数
    metrics['dice'] = (2 * tp) / (2 * tp + fp + fn + 1e-8)
    
    # IoU (Jaccard指数)
    metrics['iou'] = tp / (tp + fp + fn + 1e-8)
    
    # 精确率、召回率、F1分数
    metrics['precision'] = tp / (tp + fp + 1e-8)
    metrics['recall'] = tp / (tp + fn + 1e-8)
    metrics['f1'] = 2 * metrics['precision'] * metrics['recall'] / (metrics['precision'] + metrics['recall'] + 1e-8)
    
    # 准确率
    metrics['accuracy'] = (tp + tn) / (tp + tn + fp + fn + 1e-8)
    
    # 特异性
    metrics['specificity'] = tn / (tn + fp + 1e-8)
    
    return metrics
```

### 多类别指标

```python
def calculate_multiclass_metrics(pred, target, num_classes):
    """计算多类别分割指标"""
    metrics = {}
    
    for c in range(num_classes):
        pred_c = (pred == c).float()
        target_c = (target == c).float()
        
        tp = (pred_c * target_c).sum().item()
        fp = (pred_c * (1 - target_c)).sum().item()
        fn = ((1 - pred_c) * target_c).sum().item()
        
        dice = (2 * tp) / (2 * tp + fp + fn + 1e-8)
        iou = tp / (tp + fp + fn + 1e-8)
        
        metrics[f'class_{c}_dice'] = dice
        metrics[f'class_{c}_iou'] = iou
    
    # 平均指标
    metrics['mean_dice'] = np.mean([metrics[f'class_{c}_dice'] for c in range(num_classes)])
    metrics['mean_iou'] = np.mean([metrics[f'class_{c}_iou'] for c in range(num_classes)])
    
    return metrics
```

### 边界质量指标

```python
def calculate_boundary_metrics(pred, target):
    """计算边界质量指标"""
    from scipy.ndimage import distance_transform_edt
    
    # 转换为二值图像
    pred_bin = (pred > 0.5).astype(np.uint8)
    target_bin = (target > 0.5).astype(np.uint8)
    
    # 计算距离变换
    pred_dist = distance_transform_edt(1 - pred_bin)
    target_dist = distance_transform_edt(1 - target_bin)
    
    # 平均表面距离
    asd_pred = np.mean(pred_dist[target_bin == 1])
    asd_target = np.mean(target_dist[pred_bin == 1])
    asd = (asd_pred + asd_target) / 2
    
    # Hausdorff距离
    hd = max(np.max(pred_dist[target_bin == 1]), np.max(target_dist[pred_bin == 1]))
    
    return {'asd': asd, 'hd': hd}
```

## 调试技巧

### 训练监控

#### TensorBoard日志

```python
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('runs/experiment1')

for epoch in range(num_epochs):
    # 训练
    train_loss = train_one_epoch(model, train_loader, criterion, optimizer)
    
    # 验证
    val_loss, val_metrics = validate(model, val_loader, criterion)
    
    # 记录到TensorBoard
    writer.add_scalar('Loss/train', train_loss, epoch)
    writer.add_scalar('Loss/val', val_loss, epoch)
    writer.add_scalar('Metrics/dice', val_metrics['dice'], epoch)
    writer.add_scalar('Metrics/iou', val_metrics['iou'], epoch)
    
    # 记录学习率
    writer.add_scalar('Learning_rate', optimizer.param_groups[0]['lr'], epoch)
```

#### 可视化预测

```python
def visualize_predictions(model, dataloader, num_samples=3):
    """可视化模型预测"""
    model.eval()
    
    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4*num_samples))
    
    with torch.no_grad():
        for i, (images, masks) in enumerate(dataloader):
            if i >= num_samples:
                break
                
            images, masks = images.cuda(), masks.cuda()
            outputs = model(images)
            preds = torch.sigmoid(outputs) > 0.5
            
            # 显示
            axes[i, 0].imshow(images[0].cpu().permute(1, 2, 0))
            axes[i, 0].set_title('输入图像')
            axes[i, 0].axis('off')
            
            axes[i, 1].imshow(masks[0].cpu(), cmap='gray')
            axes[i, 1].set_title('真实掩码')
            axes[i, 1].axis('off')
            
            axes[i, 2].imshow(preds[0].cpu(), cmap='gray')
            axes[i, 2].set_title('预测掩码')
            axes[i, 2].axis('off')
    
    plt.tight_layout()
    plt.show()
```

### 常见问题诊断

| 问题 | 可能原因 | 解决方案 |
|------|----------|----------|
| 损失不下降 | 学习率太小 | 增加学习率，使用学习率搜索 |
| 损失震荡 | 学习率太大 | 减小学习率，使用学习率调度 |
| 过拟合 | 模型太复杂或数据太少 | 增加数据增强，使用正则化，早停 |
| 欠拟合 | 模型太简单 | 增加模型容量，减少正则化 |
| 梯度爆炸 | 初始化不当 | 使用合适的初始化，梯度裁剪 |
| 内存不足 | 批次太大或模型太大 | 减小批次大小，使用梯度累积，混合精度 |

## 部署考虑

### 模型优化

#### 模型量化

```python
# 动态量化
model_quantized = torch.quantization.quantize_dynamic(
    model, {nn.Conv2d, nn.Linear}, dtype=torch.qint8
)

# 静态量化
model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
model_prepared = torch.quantization.prepare(model)
# 校准...
model_quantized = torch.quantization.convert(model_prepared)
```

#### ONNX导出

```python
torch.onnx.export(
    model,
    dummy_input,
    "unet.onnx",
    export_params=True,
    opset_version=11,
    do_constant_folding=True,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
)
```

### 推理优化

```python
@torch.no_grad()
def inference_optimized(model, image, device='cuda'):
    """优化推理"""
    model.eval()
    
    # 移动到设备
    image = image.to(device)
    
    # 使用torch.inference_mode（PyTorch 1.9+）
    with torch.inference_mode():
        output = model(image)
    
    return output
```

## 总结

训练高性能的U-Net模型需要系统的方法和细致的调优。关键要点包括：

1. **数据为王**：高质量的数据和适当的数据增强是成功的基础
2. **渐进调优**：从简单配置开始，逐步增加复杂度
3. **全面监控**：使用多种指标和可视化工具监控训练过程
4. **耐心调试**：遇到问题时系统诊断，不要盲目更改参数
5. **实践验证**：最终性能需要在独立的测试集上验证

通过遵循这些最佳实践，你可以最大化U-Net模型的性能，并在实际应用中取得良好效果。记住，没有"一刀切"的解决方案，每个任务都需要根据具体情况进行调整和优化。
