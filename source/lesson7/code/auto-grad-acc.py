import torch

# 梯度累积问题
x = torch.tensor(1.0, requires_grad=True)

for i in range(3):
    y = x ** 2
    y.backward()  # 每次调用backward()  # 梯度会累积

print(f"累积梯度: {x.grad}")  # 2 + 2 + 2 = 6

# 正确做法：清零梯度
x = torch.tensor(1.0, requires_grad=True)

for i in range(3):
    if x.grad is not None:
        x.grad.zero_()  # 清零梯度

    y = x ** 2
    y.backward()
    print(f"第{i}次梯度: {x.grad}")
