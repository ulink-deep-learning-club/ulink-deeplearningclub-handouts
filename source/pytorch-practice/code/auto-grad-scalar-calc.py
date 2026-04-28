import torch

# 创建叶子节点
x = torch.tensor(2.0, requires_grad=True)  # 对应 back-propagation 中的示例
y = torch.tensor(3.0, requires_grad=True)

# 前向传播：f = (x + y) × y
a = x + y      # a = 5
f = a * y      # f = 15

print(f"前向结果: f = {f.item()}")  # 15.0

# 反向传播：自动计算梯度
f.backward()

# 查看梯度
print(f"∂f/∂x = {x.grad}")  # 3.0（与理论一致！）
print(f"∂f/∂y = {y.grad}")  # 8.0（与理论一致！）

# 梯度验证（手工计算）：
# ∂f/∂x = ∂f/∂a × ∂a/∂x = y × 1 = 3 ✓
# ∂f/∂y = ∂f/∂a × ∂a/∂y + ∂f/∂y（直接） = y + a = 3 + 5 = 8 ✓
