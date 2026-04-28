(debug-and-visualise)=
# 调试与可视化技巧

在{doc}`train-workflow`中，我们搭建了一个完整的训练流程。但当你开始训练自己的模型时，可能会遇到各种问题：损失不下降、准确率上不去、模型不收敛...

**本章就是深度学习开发的"急救手册"**——教你诊断问题、可视化训练过程、定位bug。

## 为什么需要调试技巧？

深度学习代码有几个特点让调试变得困难：

1. **错误延迟暴露**：数据预处理的问题可能在训练10个epoch后才显现
2. **梯度流动不可见**：神经网络是黑盒，中间状态难以观察
3. **数值问题隐蔽**：梯度爆炸/消失不会报错，只会让模型学不好
4. **随机性干扰**：同样的代码运行多次结果可能不同

**好消息**：PyTorch 提供了丰富的工具帮我们"透视"训练过程。

---

## 梯度检查：诊断训练的"生命线"

### 梯度范数监控

{ref}`gradient-descent`告诉我们，梯度是参数更新的基础。如果梯度有问题，训练必然失败。

```python
def check_gradients(model, print_details=False):
    """
    检查模型梯度状态
    
    参数:
        model: PyTorch模型
        print_details: 是否打印每层梯度详情
    
    返回:
        total_norm: 全局梯度范数
    """
    total_norm = 0.0
    grad_stats = {}
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            # 计算该层梯度的L2范数
            param_norm = param.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
            
            # 收集统计信息
            grad_stats[name] = {
                'mean': param.grad.mean().item(),
                'std': param.grad.std().item(),
                'max': param.grad.max().item(),
                'min': param.grad.min().item(),
            }
            
            # 检查数值异常
            if torch.isnan(param.grad).any():
                print(f"❌ 错误: {name} 包含NaN梯度！")
                print(f"   该层参数形状: {param.shape}")
                print(f"   建议: 检查前向传播是否有除以0或log(0)")
                
            if torch.isinf(param.grad).any():
                print(f"❌ 错误: {name} 包含Inf梯度！")
                print(f"   建议: 使用梯度裁剪 (gradient clipping)")
    
    total_norm = total_norm ** 0.5
    
    # 判断梯度状态
    if total_norm < 1e-6:
        print(f"⚠️ 警告: 梯度范数过小 ({total_norm:.2e})，可能出现梯度消失")
        print(f"   建议: 检查激活函数、网络深度，或使用残差连接")
    elif total_norm > 100:
        print(f"⚠️ 警告: 梯度范数过大 ({total_norm:.2f})，可能出现梯度爆炸")
        print(f"   建议: 使用梯度裁剪或减小学习率")
    else:
        print(f"✅ 梯度范数正常: {total_norm:.4f}")
    
    if print_details:
        print("\n各层梯度统计:")
        for name, stats in grad_stats.items():
            print(f"  {name}: mean={stats['mean']:.4f}, std={stats['std']:.4f}")
    
    return total_norm
```

**使用场景**：在每个epoch或每隔几个batch调用，监控梯度健康状态。

### 梯度爆炸与消失的检测

{ref}`back-propagation`中提到梯度消失/爆炸是深层网络的常见问题。如何在代码中检测？

```python
class GradientMonitor:
    """梯度监控器：记录训练过程中的梯度变化"""
    
    def __init__(self, model):
        self.model = model
        self.history = []  # 记录每个step的梯度信息
        
        # 注册hook，捕获每层的梯度
        self.hooks = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                hook = param.register_hook(
                    lambda grad, name=name: self._log_gradient(grad, name)
                )
                self.hooks.append(hook)
    
    def _log_gradient(self, grad, name):
        """记录梯度信息"""
        if grad is not None:
            self.history.append({
                'name': name,
                'step': len(self.history),
                'mean': grad.abs().mean().item(),
                'max': grad.abs().max().item(),
            })
    
    def plot_gradient_flow(self):
        """可视化梯度在各层的流动"""
        import matplotlib.pyplot as plt
        
        # 按层分组计算平均梯度
        layer_grads = {}
        for record in self.history:
            layer = record['name'].split('.')[0]  # 提取层名
            if layer not in layer_grads:
                layer_grads[layer] = []
            layer_grads[layer].append(record['mean'])
        
        # 绘图
        plt.figure(figsize=(12, 6))
        for layer, grads in layer_grads.items():
            plt.plot(grads, label=layer, alpha=0.7)
        
        plt.xlabel('Training Step')
        plt.ylabel('Mean Gradient')
        plt.title('Gradient Flow Across Layers')
        plt.legend()
        plt.yscale('log')  # 对数尺度更容易看出差异
        plt.grid(True, alpha=0.3)
        plt.show()
        
        # 诊断建议
        first_layer_grad = list(layer_grads.values())[0]
        last_layer_grad = list(layer_grads.values())[-1]
        
        ratio = np.mean(first_layer_grad) / (np.mean(last_layer_grad) + 1e-10)
        
        if ratio > 100:
            print(f"⚠️ 检测到梯度消失: 第一层梯度是最后一层的 {ratio:.1f} 倍")
            print(f"   建议: 使用BatchNorm、残差连接或更小的学习率")
        elif ratio < 0.01:
            print(f"⚠️ 检测到梯度爆炸: 最后一层梯度是第一层的 {1/ratio:.1f} 倍")
            print(f"   建议: 使用梯度裁剪或权重衰减")
    
    def remove_hooks(self):
        """移除所有hook，释放内存"""
        for hook in self.hooks:
            hook.remove()
```

**核心洞察**：通过可视化梯度在各层的流动，可以快速判断是否存在梯度消失/爆炸问题。

---

## 训练过程可视化：TensorBoard

TensorBoard 是 TensorFlow 团队开发的可视化工具，但 PyTorch 也完全支持。它能让我们"看到"训练过程中发生的一切。

### 基础设置

```python
from torch.utils.tensorboard import SummaryWriter
import datetime

# 创建writer，日志目录包含时间戳，方便区分不同实验
log_dir = f'runs/mnist_experiment_{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}'
writer = SummaryWriter(log_dir)

print(f"TensorBoard日志目录: {log_dir}")
print(f"启动TensorBoard: tensorboard --logdir={log_dir}")
```

### 记录训练指标

```python
def train_with_logging(model, device, train_loader, optimizer, epoch):
    """带TensorBoard日志记录的训练函数"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        # 累计统计
        running_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
        
        # 每100个batch记录一次
        if batch_idx % 100 == 99:
            step = epoch * len(train_loader) + batch_idx
            
            # 记录损失和准确率
            writer.add_scalar('Training/Loss', running_loss / 100, step)
            writer.add_scalar('Training/Accuracy', 100. * correct / total, step)
            
            # 记录学习率
            current_lr = optimizer.param_groups[0]['lr']
            writer.add_scalar('Training/Learning_Rate', current_lr, step)
            
            running_loss = 0.0
            correct = 0
            total = 0
```

### 记录模型结构和参数分布

```python
# 记录模型计算图
sample_data = next(iter(train_loader))[0].to(device)
writer.add_graph(model, sample_data)

# 每5个epoch记录参数和梯度的直方图
def log_histograms(model, epoch):
    """记录参数分布，帮助诊断问题"""
    for name, param in model.named_parameters():
        # 记录参数分布
        writer.add_histogram(f'Parameters/{name}', param, epoch)
        
        # 记录梯度分布
        if param.grad is not None:
            writer.add_histogram(f'Gradients/{name}', param.grad, epoch)
            
            # 计算并记录梯度/参数比值（重要诊断指标）
            ratio = (param.grad.abs() / (param.abs() + 1e-8)).mean()
            writer.add_scalar(f'GradParamRatio/{name}', ratio, epoch)
```

**梯度/参数比值的意义**：
- 比值太小（<0.001）：学习率可能太小，参数更新缓慢
- 比值适中（0.001~0.1）：健康状态
- 比值太大（>1）：学习率可能太大，参数震荡

### 可视化训练数据

```python
def visualize_data(writer, train_loader, num_images=64):
    """可视化训练数据，检查预处理是否正确"""
    images, labels = next(iter(train_loader))
    
    # 创建网格图像
    img_grid = torchvision.utils.make_grid(
        images[:num_images], 
        nrow=8, 
        normalize=True,
        value_range=(0, 1)
    )
    
    writer.add_image('Training_Data', img_grid, 0)
    
    # 记录类别分布
    class_counts = torch.bincount(labels, minlength=10)
    for i, count in enumerate(class_counts):
        writer.add_scalar(f'Class_Distribution/class_{i}', count, 0)
```

### 嵌入可视化（高维数据投影）

```python
def visualize_embeddings(model, device, test_loader, writer):
    """
    可视化模型学到的特征表示
    将高维特征投影到3D空间，看不同类别是否分开
    """
    model.eval()
    embeddings = []
    labels = []
    images_list = []
    
    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device)
            # 提取倒数第二层的特征
            features = model.features(data)  # 假设模型有features方法
            embeddings.append(features.cpu())
            labels.append(target)
            images_list.append(data.cpu())
            
            if len(embeddings) >= 100:  # 只取100个batch
                break
    
    embeddings = torch.cat(embeddings, dim=0)
    labels = torch.cat(labels, dim=0)
    images_list = torch.cat(images_list, dim=0)
    
    writer.add_embedding(
        embeddings[:1000],  # 最多1000个点
        metadata=labels[:1000].tolist(),
        label_img=images_list[:1000],
        global_step=0,
        tag='Feature_Embeddings'
    )
```

---

## 常见训练问题诊断

### 问题1：损失不下降

**症状**：训练了多个epoch，损失几乎不变。

```python
def diagnose_no_decrease_loss(model, loader, device):
    """诊断损失不下降的原因"""
    
    print("=" * 50)
    print("🔍 诊断损失不下降...")
    print("=" * 50)
    
    # 1. 检查数据
    data, target = next(iter(loader))
    print(f"\n1. 数据检查:")
    print(f"   输入范围: [{data.min():.2f}, {data.max():.2f}]")
    print(f"   标签唯一值: {target.unique()}")
    
    if data.max() > 10:
        print("   ⚠️ 警告: 输入未归一化，建议除以255或标准化")
    
    # 2. 检查模型输出
    model.eval()
    with torch.no_grad():
        output = model(data.to(device))
        print(f"\n2. 模型输出检查:")
        print(f"   输出形状: {output.shape}")
        print(f"   输出范围: [{output.min():.2f}, {output.max():.2f}]")
        
        if torch.allclose(output, output[0]):
            print("   ❌ 错误: 所有输出相同，模型未学习！")
            print("   可能原因: 权重初始化问题或学习率太小")
    
    # 3. 检查梯度
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    output = model(data.to(device))
    loss = nn.CrossEntropyLoss()(output, target.to(device))
    loss.backward()
    
    print(f"\n3. 梯度检查:")
    has_grad = False
    for name, param in model.named_parameters():
        if param.grad is not None and param.grad.abs().sum() > 0:
            has_grad = True
            print(f"   ✅ {name}: 梯度范数={param.grad.norm():.4f}")
        elif param.grad is None:
            print(f"   ⚠️ {name}: 无梯度！")
    
    if not has_grad:
        print("   ❌ 所有参数都没有梯度！")
        print("   检查: forward中是否使用了requires_grad=False的输入")
    
    # 4. 学习率检查
    print(f"\n4. 学习率检查:")
    test_lr = [1e-5, 1e-3, 1e-1]
    for lr in test_lr:
        print(f"   学习率={lr}: ", end="")
        # 简化的参数更新测试
        param = list(model.parameters())[0].clone()
        grad = list(model.parameters())[0].grad
        if grad is not None:
            param_update = (grad * lr).abs().mean()
            print(f"平均更新量={param_update:.2e}")
```

**常见原因和解决方案**：

| 原因 | 检查方法 | 解决方案 |
|------|----------|----------|
| 学习率太小 | 观察参数更新量 | 增大到0.001~0.01 |
| 梯度消失 | 检查深层梯度范数 | 加BatchNorm、换ReLU |
| 数据未归一化 | 检查输入范围 | 标准化到[0,1]或[-1,1] |
| 标签错误 | 检查target值 | 确保标签从0开始 |
| 权重初始化问题 | 检查初始输出 | 使用Xavier/He初始化 |

### 问题2：过拟合

**症状**：训练准确率很高，验证准确率很低。

```python
def diagnose_overfitting(train_acc_history, val_acc_history):
    """诊断过拟合程度"""
    
    train_acc = np.array(train_acc_history)
    val_acc = np.array(val_acc_history)
    
    # 计算gap
    gap = train_acc - val_acc
    
    print(f"\n📊 过拟合诊断报告:")
    print(f"   最终训练准确率: {train_acc[-1]:.2f}%")
    print(f"   最终验证准确率: {val_acc[-1]:.2f}%")
    print(f"   准确率差距: {gap[-1]:.2f}%")
    
    if gap[-1] < 3:
        print(f"   ✅ 状态良好，无明显过拟合")
    elif gap[-1] < 10:
        print(f"   ⚠️ 轻度过拟合，建议增加正则化")
        print(f"      - 添加Dropout (rate=0.2~0.5)")
        print(f"      - 增加weight_decay (1e-4~1e-3)")
    else:
        print(f"   ❌ 严重过拟合，必须采取措施！")
        print(f"      - 收集更多训练数据")
        print(f"      - 使用更强的数据增强")
        print(f"      - 减小模型容量")
    
    # 检查验证集准确率是否下降
    if len(val_acc) > 5:
        recent_trend = val_acc[-1] - val_acc[-5]
        if recent_trend < -2:
            print(f"   ⚠️ 验证准确率下降 {abs(recent_trend):.1f}%，建议早停")
```

**参考**：{ref}`regularization`中详细讨论了各种正则化技术。

### 问题3：学习率不当

```python
def find_lr(model, train_loader, optimizer, criterion, device, 
            start_lr=1e-7, end_lr=10, num_iter=100):
    """
    学习率范围测试 (Learning Rate Range Test)
    参考: Cyclical Learning Rates for Training Neural Networks
    """
    
    model.train()
    lrs = []
    losses = []
    
    # 指数增长学习率
    lr_mult = (end_lr / start_lr) ** (1 / num_iter)
    lr = start_lr
    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    
    for batch_idx, (data, target) in enumerate(train_loader):
        if batch_idx >= num_iter:
            break
            
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        lrs.append(lr)
        losses.append(loss.item())
        
        # 更新学习率
        lr *= lr_mult
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    
    # 绘图
    plt.figure(figsize=(10, 6))
    plt.plot(lrs, losses)
    plt.xscale('log')
    plt.xlabel('Learning Rate')
    plt.ylabel('Loss')
    plt.title('Learning Rate Range Test')
    plt.axvline(x=lrs[np.argmin(losses)], color='r', linestyle='--', 
                label=f'Min Loss @ lr={lrs[np.argmin(losses)]:.2e}')
    plt.legend()
    plt.show()
    
    # 建议最优学习率
    min_idx = np.argmin(losses)
    suggested_lr = lrs[max(0, min_idx - 1)]  # 取最小损失前一个点
    print(f"建议学习率: {suggested_lr:.2e}")
    
    return suggested_lr
```

---

## 下一步

掌握了调试和可视化技巧后，你已经具备了独立训练神经网络的能力。

{doc}`best-practices`中，我们将学习工程最佳实践——如何让代码更规范、更易复现、更易于协作。这些技巧来自于{doc}`scaling-law`中的效率优化思想，是走向专业深度学习开发的必经之路。
