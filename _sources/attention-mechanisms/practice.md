(attn-practice)=
# 实践指南

掌握了注意力机制的原理和选择策略后，本章聚焦实际实现中的关键问题。

## 快速上手：在已有模型中加入注意力

在 PyTorch 中，给已有模型添加注意力模块只需要三步：

```python
import torchvision.models as models
from se_block import SEBlock  # 你的SE模块实现

# 1. 加载预训练模型
model = models.resnet18(weights='IMAGENET1K_V1')

# 2. 替换残差块：在每个BasicBlock的最后一个卷积后插入SE
def add_se_to_block(block, reduction=16):
    """给一个残差块添加SE模块"""
    block.se = SEBlock(block.conv2.out_channels, reduction)
    # 修改forward逻辑——需要在BasicBlock中增加se的调用
    return block

# 3. 应用到所有层（或只应用到最后几层）
for name, module in model.named_children():
    if 'layer' in name:  # 只给残差层添加
        for i in range(len(module)):
            add_se_to_block(module[i])
```

**关键点**：不需要修改模型的整体架构，只需在残差块内部插入。

## 超参数设置

### 压缩比 $r$

- **默认值**：$r=16$，适合大多数情况
- **模型较大时**（ResNet-101+）：可尝试 $r=8$，保留更多通道信息
- **模型较小时**（MobileNet）：建议 $r=16$ 或更大，避免参数量增加过多

### 插入位置

- **分类任务**：在 stage3 和 stage4 添加即可（深层特征语义丰富）
- **检测/分割**：从 stage2 开始添加（需要保留空间分辨率）
- **小数据集**：只加一层，通常加在最后一个 stage

## 学习率调整

添加注意力模块后，建议：
- 使用与骨干网络**相同的学习率**或略微更小（0.1×~1×）
- 注意力模块的权重是随机初始化的，初期需要较大的学习率来快速收敛
- 如果训练不稳定，可以给注意力模块单独设置学习率：

```python
optimizer = torch.optim.Adam([
    {'params': model.backbone.parameters(), 'lr': 1e-4},
    {'params': model.attention.parameters(), 'lr': 1e-3},  # 注意力模块用更大的lr
])
```

## 调试技巧

### 可视化注意力权重

训练过程中，定期检查注意力权重的分布：

```python
def inspect_attention(model, name='se'):
    """打印注意力权重的统计信息"""
    for module_name, module in model.named_modules():
        if name in module_name.lower() and hasattr(module, 'scale'):
            weights = module.scale.data
            print(f"{module_name}: "
                  f"mean={weights.mean():.3f}, "
                  f"std={weights.std():.3f}, "
                  f"min={weights.min():.3f}, "
                  f"max={weights.max():.3f}")
```

**判断标准**：
- 权重均匀分布在 0.5 左右 → 注意力没学到东西（可能是学习率太小）
- 权重大量集中在 0 或 1 → 注意力过于极端（可能是学习率太大）
- 权重分布在 0.2~0.8 之间，且有变化 → 正常

## 常见问题

### Q1：加了注意力后准确率反而下降？

可能的原因：
1. **学习率不合适**：注意力模块需要单独调学习率
2. **插入位置不对**：尝试不同的插入位置
3. **数据集太小**：注意力模块增加了模型容量，小数据集上容易过拟合

### Q2：训练损失震荡不收敛？

可能的原因：
1. 注意力权重初始值不合适 → 尝试不同的初始化方法
2. Sigmoid 输出接近 0 导致梯度消失 → 检查权重初始化
3. 学习率过大 → 降低学习率

### Q3：注意力权重全为 0.5 左右（均匀分布）？

说明注意力没有学到有效信息。检查：
- SE 模块的压缩比是否过大（$r$ 太大导致表达能力不足）
- 训练是否充分（可能需要更多 epoch）
- 任务本身是否不需要注意力（某些简单任务上注意力提升有限）

## 本章小结

- 注意力模块可以即插即用，只需在残差块中插入几行代码
- 压缩比 $r=16$ 是安全起点，根据任务和资源调整
- 定期可视化注意力权重分布，诊断训练问题
- 注意力和 BN、残差连接等技术互补，不是替代关系

### 下一步

{doc}`the-end` 将总结本章的核心知识点，并给出进一步学习的建议。
