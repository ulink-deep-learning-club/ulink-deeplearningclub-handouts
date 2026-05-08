(pytorch-autograd)=
# 自动微分：PyTorch 的核心魔法

{doc}`./neural-network-module`中我们学会了构建神经网络结构。现在的问题是：**如何让网络"学习"？**

在 {doc}`../math-fundamentals/back-propagation`中，我们学习了反向传播算法的原理——通过链式法则将损失梯度从输出层传回输入层。但如果手动实现这个算法，代码会非常复杂且容易出错。

**PyTorch 的自动微分（autograd）就是解决方案**：它自动构建计算图并执行反向传播，让我们只需关注前向计算，梯度会自动计算。

## 直觉理解：自动微分是什么？

**类比：自动记账系统**

想象你经营一家连锁餐厅：

- **手动反向传播**：每道菜卖了多少钱，你需要手动追踪每一笔成本（食材、人工、租金），然后计算每个分店应该调整什么——极其繁琐且容易出错。
- **自动微分**：你只需记录每笔交易（前向传播），系统自动生成财务报表（梯度），告诉你每个分店该如何调整。

**核心洞察**：
自动微分 = 自动构建计算图 + 自动执行反向传播 + 自动存储梯度

## 从理论到代码

| {doc}`../math-fundamentals/back-propagation` 理论 | PyTorch 实现 | 作用 |
|--------------------------------------------------|-------------|------|
| {ref}`computational-graph` | `requires_grad=True` | 标记需要计算梯度的张量 |
| 链式法则 | `.backward()` | 自动回传梯度 |
| 梯度存储 | `.grad` | 存储计算得到的梯度 |
| 梯度清零 | `.zero_()` | 清除旧梯度，避免累积 |

## 计算图的自动构建

### 什么是动态计算图？

PyTorch 使用**动态计算图**（Dynamic Computational Graph），这意味着：

1. **图在运行时构建**：每次前向传播都会重新构建图
2. **图结构可以变化**：支持 Python 控制流（if、for、while）
3. **内存高效**：反向传播后可以释放中间结果

```{mermaid}
flowchart TD
    A["x<br/>requires_grad=True<br/>叶子节点"] --> C["乘法 op"]
    B["w<br/>requires_grad=True<br/>叶子节点"] --> C
    C --> D["z = x × w<br/>中间节点"]
    D --> E["加法 op"]
    F["b<br/>requires_grad=True"] --> E
    E --> G["y = z + b<br/>输出节点"]
    
    style A fill:#e3f2fd
    style B fill:#e3f2fd
    style F fill:#e3f2fd
    style G fill:#e8f5e9
```

**关键区别**：
- **叶子节点（Leaf Node）**：用户创建的张量（`requires_grad=True`），梯度会保存
- **中间节点**：运算产生的张量，默认不保存梯度（除非设置 `retain_graph=True`）

### 创建可追踪的张量

```{literalinclude} code/auto-grad-trackable-tensors.py
:language: python
:linenos:
```

## 梯度计算：.backward() 的魔力

### 标量输出的梯度计算

```{literalinclude} code/auto-grad-scalar-calc.py
:language: python
:linenos:
```

**代码解释**：
- `requires_grad=True`：告诉 PyTorch 跟踪这个张量的所有操作
- `.backward()`：从输出节点开始，沿计算图反向传播，计算所有叶子节点的梯度
- `.grad`：存储计算得到的梯度值

### 非标量输出的处理

当输出不是标量时，需要指定权重向量：

```python
import torch

# 输出是向量而非标量
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
y = x * 2  # y = [2, 4, 6]

# 错误：y 是向量，不能直接 backward()
# y.backward()  # RuntimeError!

# 正确：提供一个权重向量（相当于计算 v^T × J）
# 这里我们计算 y 每个元素对 x 的梯度的加权和
v = torch.tensor([1.0, 1.0, 1.0])  # 权重向量
y.backward(v)  # 等价于计算 sum(y) 的梯度

print(f"x.grad = {x.grad}")  # [2, 2, 2]

# 实际应用：通常我们会将损失降为标量
loss = y.sum()  # 标量
loss.backward()  # 可以直接调用
```

## 梯度控制技巧

### 梯度累积与清零

**关键问题**：PyTorch 的 `.backward()` 默认会**累积**梯度！

```{literalinclude} code/auto-grad-accumulation.py
:language: python
:linenos:
```

**训练循环中的标准模式**：

```python
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

for batch in dataloader:
    # 1. 清零旧梯度
    optimizer.zero_grad()  # 或 model.zero_grad()
    
    # 2. 前向传播
    output = model(batch.input)
    loss = criterion(output, batch.target)
    
    # 3. 反向传播
    loss.backward()
    
    # 4. 更新参数
    optimizer.step()
```

### 禁用梯度计算

在某些情况下，我们不需要计算梯度：

```{literalinclude} code/auto-grad-disable.py
:language: python
:linenos:
```

**何时禁用梯度？**

| 场景 | 原因 | 代码 |
|------|------|------|
| 模型推理 | 不需要更新参数 | `with torch.no_grad()` |
| 特征提取 | 冻结预训练模型 | `param.requires_grad = False` |
| 数值计算 | 避免梯度开销 | `.detach()` |
| 保存张量 | 避免保留计算图 | `tensor.detach().cpu().numpy()` |

## 实际应用：神经网络训练

### 完整示例：线性回归

```python
import torch
import torch.nn as nn

# 生成数据：y = 2x + 1 + 噪声
torch.manual_seed(42)
X = torch.randn(100, 1)          # 100个样本，1个特征
y_true = 2 * X + 1 + 0.1 * torch.randn(100, 1)

# 定义模型：对应 {doc}`../math-fundamentals/gradient-descent` 中的线性模型
model = nn.Linear(1, 1)          # 输入1维，输出1维

# 损失函数和优化器
criterion = nn.MSELoss()         # 均方误差
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

# 训练循环
for epoch in range(100):
    # 1. 清零梯度
    optimizer.zero_grad()
    
    # 2. 前向传播
    y_pred = model(X)            # 计算预测值
    loss = criterion(y_pred, y_true)  # 计算损失
    
    # 3. 反向传播（自动计算梯度）
    loss.backward()
    
    # 4. 更新参数
    optimizer.step()
    
    if (epoch + 1) % 20 == 0:
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

# 查看学习到的参数
print(f"\n学习到的权重: {model.weight.item():.4f}（真实值: 2.0）")
print(f"学习到的偏置: {model.bias.item():.4f}（真实值: 1.0）")
```

### 查看计算图

```python
import torch

x = torch.tensor(2.0, requires_grad=True)
y = x ** 2

# 查看梯度函数（对应计算图中的边）
print(f"y.grad_fn: {y.grad_fn}")          # <PowBackward0 object>
print(f"y.grad_fn.next_functions: {y.grad_fn.next_functions}")

# 查看是否需要梯度
print(f"x.requires_grad: {x.requires_grad}")  # True
print(f"y.requires_grad: {y.requires_grad}")  # True
```

## 高级主题

### 自定义梯度计算

偶尔需要自定义梯度计算规则：

```python
import torch
from torch.autograd import Function

class MyReLU(Function):
    """
    自定义 ReLU 激活函数，带梯度裁剪
    对应 {doc}`../math-fundamentals/activation-functions` 中的 ReLU
    """
    
    @staticmethod
    def forward(ctx, input):
        """前向传播：max(0, x)"""
        ctx.save_for_backward(input)  # 保存用于反向传播的张量
        return input.clamp(min=0)
    
    @staticmethod
    def backward(ctx, grad_output):
        """反向传播：梯度裁剪"""
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input < 0] = 0  # ReLU 的梯度规则
        return grad_input

# 使用自定义函数
my_relu = MyReLU.apply
x = torch.tensor([-1.0, 2.0, -3.0, 4.0], requires_grad=True)
y = my_relu(x)
y.sum().backward()

print(f"输入: {x}")
print(f"输出: {y}")
print(f"梯度: {x.grad}")  # [0, 1, 0, 1]
```

### 梯度检查

验证梯度计算是否正确：

```python
import torch
from torch.autograd import gradcheck

# 定义需要测试的函数
def func(x):
    return x ** 3 + x ** 2

# 使用双精度浮点数进行数值梯度检查
test_input = torch.randn(3, 4, dtype=torch.double, requires_grad=True)

# 验证梯度
result = gradcheck(func, test_input, eps=1e-6, atol=1e-4)
print(f"梯度检查通过: {result}")  # True 表示梯度计算正确
```

## 总结

### 核心概念回顾

| 概念 | 解释 | 代码 |
|------|------|------|
| 计算图 | 记录张量操作的 DAG | 自动构建 |
| 叶子节点 | 用户创建的可训练参数 | `requires_grad=True` |
| 反向传播 | 从输出回传梯度 | `.backward()` |
| 梯度存储 | 存储在 `.grad` 属性中 | `tensor.grad` |
| 梯度清零 | 避免梯度累积 | `.zero_grad()` |
| 禁用梯度 | 推理时节省内存 | `torch.no_grad()` |

### 下一步

掌握了自动微分后，下一节 {doc}`./optimiser` 我们将学习如何使用这些梯度来更新模型参数——从简单的 SGD 到自适应学习率的 Adam，掌握优化算法的核心原理。

**从"计算梯度"到"使用梯度优化"，让我们继续深入！**
