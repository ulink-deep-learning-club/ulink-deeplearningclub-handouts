import torch

# 叶子节点：requires_grad=True 告诉 PyTorch"我要跟踪这个张量"
x = torch.tensor([2.0, 3.0], requires_grad=True)  # 输入特征
w = torch.tensor([1.0, 2.0], requires_grad=True)  # 权重
b = torch.tensor(0.5, requires_grad=True)          # 偏置

print(f"x.is_leaf: {x.is_leaf}")  # True：用户创建的叶子节点
print(f"w.is_leaf: {w.is_leaf}")  # True

# 中间节点：通过运算产生
z = x * w              # 逐元素乘法，形状 [2]
y = z.sum() + b        # 求和后加偏置，标量

print(f"z.is_leaf: {z.is_leaf}")  # False：运算产生
print(f"y.is_leaf: {y.is_leaf}")  # False
