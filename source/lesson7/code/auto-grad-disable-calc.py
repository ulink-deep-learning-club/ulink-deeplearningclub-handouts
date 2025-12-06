import torch

# 方法1：torch.no_grad()上下文管理器
x = torch.tensor(2.0, requires_grad=True)

with torch.no_grad():
    y = x ** 2  # 不会记录计算图
    y.requires_grad = False

# 方法2：detach()方法
x = torch.tensor(2.0, requires_grad=True)
y = x ** 2
y_detached = y.detach()  # 创建不需要梯度的副本

# 方法3：requires_grad_()方法
x = torch.tensor(2.0)
x.requires_grad_(True)   # 启用梯度
x.requires_grad_(False)  # 禁用梯度
