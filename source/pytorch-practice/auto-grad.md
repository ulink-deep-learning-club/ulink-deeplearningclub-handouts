# 4. 自动微分：PyTorch的核心机制

```{admonition} 本章要点
:class: important

- 理解动态计算图的工作原理
- 掌握梯度计算的基本规则
- 学习梯度累积与控制技巧
- 了解何时以及如何禁用梯度计算
- 应用自动微分解决实际问题
```

## 4.1 计算图基础

PyTorch使用**动态计算图**记录张量操作。计算图是深度学习框架的核心，它跟踪所有操作并自动计算梯度。

### 4.1.1 什么是计算图？

计算图是有向无环图（DAG），其中：
- **节点**：表示张量或操作
- **边**：表示数据依赖关系
- **叶子节点**：用户创建的张量（`requires_grad=True`）
- **非叶子节点**：通过操作生成的张量

```{mermaid}
flowchart TD
    A[x: requires_grad=True] --> C[乘法]
    B[y: requires_grad=True] --> C
    C --> D[z = x * y]
    D --> E[加法]
    F[常数 2] --> E
    E --> G[result = z + 2]
    
    style A fill:#e1f5fe
    style B fill:#e1f5fe
    style F fill:#f3e5f5
    style G fill:#e8f5e8
```

### 4.1.2 动态计算图示例

```{literalinclude} code/auto-grad-dyn-comp-diagram.py
:language: python
:linenos:
:caption: 动态计算图构建与可视化
```

```{admonition} 动态计算图的优势
:class: note

1. **灵活性**：图结构在运行时动态构建
2. **易调试**：可以随时检查中间结果
3. **Python集成**：与Python控制流无缝结合
4. **内存效率**：可以释放不再需要的中间结果
```

## 4.2 梯度计算规则

PyTorch使用反向传播算法自动计算梯度。当调用`.backward()`时，框架会沿着计算图反向传播，计算所有叶子节点的梯度。

### 4.2.1 基本梯度计算

```{literalinclude} code/auto-grad-calc.py
:language: python
:linenos:
:caption: 复杂函数的梯度计算与验证
```

### 4.2.2 梯度计算原理

对于标量输出 $L$，PyTorch计算每个叶子节点 $x_i$ 的梯度：

$$
\frac{\partial L}{\partial x_i} = \sum_{j \in \text{路径}} \frac{\partial L}{\partial y_j} \cdot \frac{\partial y_j}{\partial x_i}
$$

其中 $y_j$ 是计算图中的中间变量。

```{admonition} 梯度计算注意事项
:class: caution

1. **标量输出**：默认情况下，`.backward()` 只适用于标量输出
2. **梯度累积**：多次调用 `.backward()` 会累积梯度
3. **梯度清零**：训练循环中需要手动清零梯度
4. **内存管理**：计算图会占用内存，及时释放
```

## 4.3 梯度累积与控制

在实际训练中，我们经常需要控制梯度的计算和累积。

### 4.3.1 梯度累积技巧

```{literalinclude} code/auto-grad-acc.py
:language: python
:linenos:
:caption: 梯度累积与控制策略
```

### 4.3.2 梯度累积的应用场景

```{admonition} 何时使用梯度累积？
:class: important

1. **大批次训练**：当GPU内存不足时，使用小批次累积梯度
2. **稳定训练**：累积多个小批次的梯度，减少噪声
3. **模拟大批次**：用小批次模拟大批次的效果
4. **分布式训练**：在多个设备间同步梯度
```

## 4.4 禁用梯度计算

在某些情况下，我们不需要计算梯度，这时可以禁用自动微分以提高性能。

### 4.4.1 禁用梯度的方法

```{literalinclude} code/auto-grad-disable-calc.py
:language: python
:linenos:
:caption: 禁用梯度计算的多种方法
```

### 4.4.2 何时禁用梯度？

```{admonition} 禁用梯度的场景
:class: note

1. **模型推理**：预测阶段不需要梯度
2. **特征提取**：仅使用预训练模型提取特征
3. **冻结参数**：训练部分层时冻结其他层
4. **性能优化**：减少内存占用和计算时间
5. **数值稳定性**：避免梯度计算中的数值问题
```

## 4.5 高级主题

### 4.5.1 自定义自动微分

PyTorch允许定义自定义函数的梯度计算规则：

```python
class CustomFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        # 前向传播计算
        ctx.save_for_backward(input)
        return input * 2
    
    @staticmethod
    def backward(ctx, grad_output):
        # 反向传播计算梯度
        input, = ctx.saved_tensors
        return grad_output * 2
```

### 4.5.2 梯度检查

使用`torch.autograd.gradcheck`验证梯度计算的正确性：

```python
def test_function(x):
    return x ** 3 + x ** 2 + x

# 验证梯度计算是否正确
test_input = torch.randn(3, 4, dtype=torch.double, requires_grad=True)
test = torch.autograd.gradcheck(test_function, test_input, eps=1e-6, atol=1e-4)
print(f"梯度检查结果: {test}")
```

```{admonition} 本章总结
:class: success

自动微分是PyTorch的核心特性，它使得深度学习模型的训练变得简单高效。通过本章学习，您应该能够：
1. 理解动态计算图的工作原理
2. 正确使用梯度计算和累积
3. 在适当的时候禁用梯度计算
4. 验证梯度计算的正确性

在下一章中，我们将学习优化器，了解如何使用梯度来更新模型参数。
```
