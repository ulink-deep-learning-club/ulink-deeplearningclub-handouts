import torch

x = torch.tensor(2.0, requires_grad=True)

# 第一次前向+反向
y1 = x ** 2      # y1 = 4
y1.backward()
print(f"第一次反向后 x.grad = {x.grad}")  # 4.0（dy1/dx = 2x = 4）

# 第二次前向+反向（不清零）
y2 = x ** 3      # y2 = 8
y2.backward()
print(f"第二次反向后 x.grad = {x.grad}")  # 16.0！（4 + 12，累积了！）

# 正确做法：每次反向前清零
x.grad.zero_()   # 清零梯度
y3 = x ** 2
y3.backward()
print(f"清零后 x.grad = {x.grad}")  # 4.0
