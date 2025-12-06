import torch

# 复杂函数的梯度
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
y = torch.tensor([4.0, 5.0, 6.0], requires_grad=True)

# 向量运算
z = torch.dot(x, y) + torch.norm(x) ** 2

# 计算梯度
z.backward()

print(f"x.grad = {x.grad}")  # y + 2x
print(f"y.grad = {y.grad}")  # x
