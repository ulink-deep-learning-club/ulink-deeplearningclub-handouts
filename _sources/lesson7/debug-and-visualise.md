# 调试与可视化

## 梯度检查

```python
def check_gradients(model):
    """检查梯度是否存在问题"""
    total_norm = 0
    for name, param in model.named_parameters():
        if param.grad is not None:
            param_norm = param.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
            
            # 检查梯度爆炸/消失
            if torch.isnan(param.grad).any():
                print(f"警告: {name} 包含NaN梯度")
            if torch.isinf(param.grad).any():
                print(f"警告: {name} 包含Inf梯度")
                
    total_norm = total_norm ** 0.5
    print(f"梯度范数: {total_norm}")
    
    return total_norm
```

## 模型可视化

```python
import torchviz
from torchsummary import summary

# 可视化计算图
x = torch.randn(1, 1, 28, 28, requires_grad=True)
model = MNISTNet()
y = model(x)

# 生成计算图
dot = torchviz.make_dot(y, params=dict(model.named_parameters()))
dot.render("mnist_net", format="png")

# 模型摘要
summary(model, input_size=(1, 28, 28))
```

## TensorBoard集成

```python
from torch.utils.tensorboard import SummaryWriter

# 创建Writer
writer = SummaryWriter('runs/mnist_experiment')

# 记录标量
for epoch in range(num_epochs):
    # ... 训练和测试
    writer.add_scalar('Loss/train', train_loss, epoch)
    writer.add_scalar('Accuracy/train', train_acc, epoch)
    writer.add_scalar('Loss/test', test_loss, epoch)
    writer.add_scalar('Accuracy/test', test_acc, epoch)
    
    # 记录直方图
    for name, param in model.named_parameters():
        writer.add_histogram(name, param, epoch)
        if param.grad is not None:
            writer.add_histogram(f'{name}.grad', param.grad, epoch)

# 记录图像
images, labels = next(iter(train_loader))
writer.add_images('training_images', images, 0)

# 记录模型图
writer.add_graph(model, images)

# 关闭Writer
writer.close()
```
