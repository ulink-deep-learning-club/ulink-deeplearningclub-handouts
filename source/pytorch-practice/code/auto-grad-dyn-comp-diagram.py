import torch

# 创建需要梯度的张量
x = torch.tensor(2.0, requires_grad=True)
y = torch.tensor(3.0, requires_grad=True)

# 前向计算
z = x * y + torch.sin(x)

# 反向传播
z.backward()

print(f"∂z/∂x = {x.grad}")   # y + cos(x) = 3 + cos(2) ≈ 3 - 0.416 = 2.584
print(f"∂z/∂y = {y.grad}")   # x = 2
