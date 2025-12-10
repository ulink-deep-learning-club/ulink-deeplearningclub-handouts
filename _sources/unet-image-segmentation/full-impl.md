# 完整U-Net实现

本章将展示完整的U-Net实现，包括网络架构定义、前向传播逻辑、配置选项和使用示例。我们将基于前面章节定义的核心组件构建一个可扩展的U-Net模型。

## 基础U-Net实现

### 导入依赖

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from .core_components import DoubleConv, DownSample, UpSample, OutConv
```

### 完整U-Net类

```python
class UNet(nn.Module):
    """完整的U-Net实现"""
    def __init__(self, n_channels=1, n_classes=2, bilinear=False):
        super(UNet, self).__init__()
        
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        
        # 编码器路径
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = DownSample(64, 128)
        self.down2 = DownSample(128, 256)
        self.down3 = DownSample(256, 512)
        self.down4 = DownSample(512, 1024)
        
        # 解码器路径
        factor = 2 if bilinear else 1
        self.up1 = UpSample(1024, 512 // factor, bilinear)
        self.up2 = UpSample(512, 256 // factor, bilinear)
        self.up3 = UpSample(256, 128 // factor, bilinear)
        self.up4 = UpSample(128, 64, bilinear)
        
        # 输出层
        self.outc = OutConv(64, n_classes)
        
    def forward(self, x):
        # 编码器路径
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        # 解码器路径（带跳跃连接）
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        
        # 输出
        logits = self.outc(x)
        return logits
```

## 可配置U-Net实现

### 灵活深度配置

```python
class FlexibleUNet(nn.Module):
    """可配置深度和宽度的U-Net"""
    def __init__(self, n_channels=1, n_classes=2, depth=4, 
                 start_channels=64, bilinear=False):
        super(FlexibleUNet, self).__init__()
        
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.depth = depth
        self.bilinear = bilinear
        
        # 计算每层通道数
        encoder_channels = [start_channels * (2 ** i) for i in range(depth + 1)]
        decoder_channels = encoder_channels[::-1]
        
        # 编码器路径
        self.encoder = nn.ModuleList()
        self.encoder.append(DoubleConv(n_channels, encoder_channels[0]))
        
        for i in range(depth):
            self.encoder.append(
                DownSample(encoder_channels[i], encoder_channels[i + 1])
            )
        
        # 解码器路径
        self.decoder = nn.ModuleList()
        factor = 2 if bilinear else 1
        
        for i in range(depth):
            in_ch = decoder_channels[i]
            out_ch = decoder_channels[i + 1] // factor
            self.decoder.append(UpSample(in_ch, out_ch, bilinear))
        
        # 输出层
        self.outc = OutConv(decoder_channels[-1], n_classes)
        
    def forward(self, x):
        # 编码器路径
        encoder_features = []
        for i, layer in enumerate(self.encoder):
            x = layer(x)
            if i < len(self.encoder) - 1:  # 除了最后一层，都保存特征
                encoder_features.append(x)
        
        # 解码器路径（带跳跃连接）
        for i, layer in enumerate(self.decoder):
            x = layer(x, encoder_features[-(i + 1)])
        
        # 输出
        logits = self.outc(x)
        return logits
```

### 3D U-Net实现

```python
class UNet3D(nn.Module):
    """3D U-Net，用于体积数据分割"""
    def __init__(self, n_channels=1, n_classes=2):
        super(UNet3D, self).__init__()
        
        self.n_channels = n_channels
        self.n_classes = n_classes
        
        # 3D双卷积块
        def double_conv_3d(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm3d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm3d(out_channels),
                nn.ReLU(inplace=True)
            )
        
        # 编码器路径
        self.inc = double_conv_3d(n_channels, 64)
        self.down1 = nn.Sequential(
            nn.MaxPool3d(2),
            double_conv_3d(64, 128)
        )
        self.down2 = nn.Sequential(
            nn.MaxPool3d(2),
            double_conv_3d(128, 256)
        )
        self.down3 = nn.Sequential(
            nn.MaxPool3d(2),
            double_conv_3d(256, 512)
        )
        self.down4 = nn.Sequential(
            nn.MaxPool3d(2),
            double_conv_3d(512, 1024)
        )
        
        # 解码器路径
        self.up1 = nn.ConvTranspose3d(1024, 512, kernel_size=2, stride=2)
        self.conv1 = double_conv_3d(1024, 512)
        
        self.up2 = nn.ConvTranspose3d(512, 256, kernel_size=2, stride=2)
        self.conv2 = double_conv_3d(512, 256)
        
        self.up3 = nn.ConvTranspose3d(256, 128, kernel_size=2, stride=2)
        self.conv3 = double_conv_3d(256, 128)
        
        self.up4 = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2)
        self.conv4 = double_conv_3d(128, 64)
        
        # 输出层
        self.outc = nn.Conv3d(64, n_classes, kernel_size=1)
        
    def forward(self, x):
        # 编码器路径
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        # 解码器路径
        x = self.up1(x5)
        x = torch.cat([x, x4], dim=1)
        x = self.conv1(x)
        
        x = self.up2(x)
        x = torch.cat([x, x3], dim=1)
        x = self.conv2(x)
        
        x = self.up3(x)
        x = torch.cat([x, x2], dim=1)
        x = self.conv3(x)
        
        x = self.up4(x)
        x = torch.cat([x, x1], dim=1)
        x = self.conv4(x)
        
        # 输出
        logits = self.outc(x)
        return logits
```

## 模型初始化

### 权重初始化

正确的权重初始化对训练稳定性至关重要。

```python
def init_weights(model, init_type='normal', gain=0.02):
    """初始化模型权重"""
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                nn.init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                nn.init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                nn.init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('初始化方法 [%s] 未实现' % init_type)
            
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
        
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, gain)
            nn.init.constant_(m.bias.data, 0.0)
    
    model.apply(init_func)
    print('模型使用 %s 初始化' % init_type)
    return model
```

### 预训练权重加载

```python
def load_pretrained_weights(model, pretrained_path, strict=True):
    """加载预训练权重"""
    if pretrained_path is not None:
        print(f'加载预训练权重: {pretrained_path}')
        checkpoint = torch.load(pretrained_path, map_location='cpu')
        
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        # 处理可能的键名不匹配
        model_dict = model.state_dict()
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items() 
                     if k.replace('module.', '') in model_dict}
        
        model_dict.update(state_dict)
        model.load_state_dict(model_dict, strict=strict)
    
    return model
```

## 模型使用示例

### 基本使用

```python
# 创建模型
model = UNet(n_channels=3, n_classes=2)

# 初始化权重
model = init_weights(model, init_type='kaiming')

# 输入数据
batch_size, channels, height, width = 4, 3, 256, 256
x = torch.randn(batch_size, channels, height, width)

# 前向传播
with torch.no_grad():
    output = model(x)
    print(f'输入尺寸: {x.shape}')
    print(f'输出尺寸: {output.shape}')
    print(f'预测类别数: {output.shape[1]}')
```

### 训练配置

```python
def configure_training(model, device='cuda'):
    """配置训练参数"""
    model = model.to(device)
    
    # 损失函数
    criterion = nn.CrossEntropyLoss()
    
    # 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, verbose=True
    )
    
    return model, criterion, optimizer, scheduler
```

### 推理示例

```python
def predict(model, image, device='cuda'):
    """单张图像推理"""
    model.eval()
    
    # 预处理
    if len(image.shape) == 3:
        image = image.unsqueeze(0)  # 添加批次维度
    
    image = image.to(device)
    
    # 推理
    with torch.no_grad():
        output = model(image)
        if model.n_classes > 1:
            prediction = torch.argmax(output, dim=1)
        else:
            prediction = (torch.sigmoid(output) > 0.5).float()
    
    return prediction.cpu().squeeze()
```

## 性能优化

### 内存优化版本

```python
class MemoryEfficientUNet(UNet):
    """内存高效的U-Net，使用梯度检查点"""
    def forward(self, x):
        # 编码器路径
        x1 = torch.utils.checkpoint.checkpoint(self.inc, x)
        x2 = torch.utils.checkpoint.checkpoint(self.down1, x1)
        x3 = torch.utils.checkpoint.checkpoint(self.down2, x2)
        x4 = torch.utils.checkpoint.checkpoint(self.down3, x3)
        x5 = torch.utils.checkpoint.checkpoint(self.down4, x4)
        
        # 解码器路径
        x = torch.utils.checkpoint.checkpoint(self.up1, x5, x4)
        x = torch.utils.checkpoint.checkpoint(self.up2, x, x3)
        x = torch.utils.checkpoint.checkpoint(self.up3, x, x2)
        x = torch.utils.checkpoint.checkpoint(self.up4, x, x1)
        
        # 输出
        logits = self.outc(x)
        return logits
```

### 混合精度训练

```python
from torch.cuda.amp import autocast, GradScaler

def train_mixed_precision(model, train_loader, val_loader, num_epochs=100):
    """混合精度训练"""
    model = model.cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scaler = GradScaler()
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        
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
            
            train_loss += loss.item()
        
        # 验证
        val_loss = validate(model, val_loader, criterion)
        print(f'Epoch {epoch}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
```

## 模型评估

### 参数统计

```python
def count_parameters(model):
    """统计模型参数数量"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f'总参数: {total_params:,}')
    print(f'可训练参数: {trainable_params:,}')
    print(f'不可训练参数: {total_params - trainable_params:,}')
    
    return total_params, trainable_params

# 示例
model = UNet()
total_params, trainable_params = count_parameters(model)
```

### 计算复杂度分析

```python
def analyze_complexity(model, input_size=(1, 256, 256)):
    """分析模型计算复杂度"""
    from thop import profile  # 需要安装thop: pip install thop
    
    input_tensor = torch.randn(1, *input_size)
    flops, params = profile(model, inputs=(input_tensor,))
    
    print(f'FLOPs: {flops / 1e9:.2f} G')
    print(f'参数数量: {params / 1e6:.2f} M')
    print(f'每张图像推理内存: {params * 4 / 1e6:.2f} MB (float32)')
    
    return flops, params
```

## 部署准备

### 模型导出

```python
def export_model(model, input_shape, export_path='unet.onnx'):
    """导出模型为ONNX格式"""
    model.eval()
    dummy_input = torch.randn(1, *input_shape)
    
    torch.onnx.export(
        model,
        dummy_input,
        export_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    
    print(f'模型已导出到: {export_path}')
```

### 模型量化

```python
def quantize_model(model, calibration_data):
    """模型量化"""
    model.eval()
    model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    
    # 准备量化
    model_prepared = torch.quantization.prepare(model)
    
    # 校准
    for data in calibration_data:
        model_prepared(data)
    
    # 转换
    model_quantized = torch.quantization.convert(model_prepared)
    
    return model_quantized
```

## 总结

完整的U-Net实现不仅包括网络架构的定义，还涉及权重初始化、训练配置、性能优化和部署准备。本章提供了从基础到高级的多种U-Net变体实现，并展示了如何在实际应用中使用这些模型。通过灵活配置网络深度、宽度和组件类型，可以针对不同的任务需求和硬件约束定制合适的U-Net模型。

在实际项目中，建议：
1. 从基础U-Net开始，验证模型在任务上的可行性
2. 根据性能需求调整网络深度和宽度
3. 使用适当的初始化方法和优化策略
4. 考虑内存和计算约束，选择是否需要内存优化或量化
5. 充分测试模型在不同输入尺寸和条件下的表现

U-Net作为一种经典而强大的分割架构，其实现相对简单但效果显著，是图像分割任务的良好起点。
