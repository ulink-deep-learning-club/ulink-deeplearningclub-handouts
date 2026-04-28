import torch

x = torch.tensor(2.0, requires_grad=True)

# 方式1：使用 torch.no_grad() 上下文管理器（推荐）
with torch.no_grad():
    y = x * 2
    print(f"y.requires_grad: {y.requires_grad}")  # False
    # y.backward()  # 错误！无法反向传播

# 方式2：使用 .detach() 从计算图中分离
z = x * 3
z_detached = z.detach()
print(f"z_detached.requires_grad: {z_detached.requires_grad}")  # False

# 方式3：设置 requires_grad=False（全局）
w = torch.tensor(2.0, requires_grad=False)
output = w * x
print(f"output.requires_grad: {output.requires_grad}")  # True（x 需要梯度）
