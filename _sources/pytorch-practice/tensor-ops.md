# 张量操作详解

## 什么是张量

张量是PyTorch中的基本数据结构，是多维数组的推广。标量（0维）、向量（1维）、矩阵（2维）都是张量（$n$维）的特例。张量可以在CPU或GPU上运行，支持自动微分，是构建深度学习模型的核心组件。与NumPy数组不同，PyTorch张量可以利用GPU加速计算，并能够跟踪梯度用于反向传播。

```{figure} ../../_static/images/scalar-to-tensor.png
:width: 80%
:align: center

从标量到张量
```

## 创建张量

```python
import torch

# 多种创建方式
zeros = torch.zeros(2, 3)          # 全零张量
ones = torch.ones(2, 3)            # 全一张量
rand = torch.rand(2, 3)            # 均匀分布随机数
randn = torch.randn(2, 3)          # 标准正态分布随机数
arange = torch.arange(0, 10, 2)    # 类似range的序列
linspace = torch.linspace(0, 1, 5) # 线性间隔序列

# 从已有数据创建
data = [[1, 2], [3, 4]]
tensor_from_list = torch.tensor(data)
tensor_from_numpy = torch.from_numpy(np.array(data))
```

## 张量属性

```python
# 查看张量属性
x = torch.randn(3, 4, 5)
print(f"形状: {x.shape}")          # torch.Size([3, 4, 5])
print(f"维度: {x.ndim}")           # 3
print(f"元素总数: {x.numel()}")     # 60
print(f"数据类型: {x.dtype}")       # torch.float32
print(f"设备: {x.device}")         # cpu 或 cuda:0
print(f"是否需要梯度: {x.requires_grad}")  # False
```

## 索引和切片

PyTorch支持NumPy风格的索引：

```python
x = torch.randn(4, 5)

# 基本索引
first_row = x[0]           # 第一行
first_col = x[:, 0]        # 第一列
submatrix = x[1:3, 2:4]    # 子矩阵

# 高级索引
indices = torch.tensor([0, 2, 3])
selected_rows = x[indices]  # 选择第0,2,3行

# 布尔索引
mask = x > 0.5
positive_values = x[mask]
```

## 形状操作

```python
x = torch.randn(2, 3, 4)

# 改变形状
reshaped = x.reshape(6, 4)      # 重塑为6×4
flattened = x.flatten()         # 展平为一维
squeezed = x.squeeze()          # 移除维度为1的轴
unsqueezed = x.unsqueeze(0)     # 在指定位置添加维度

# 转置和重排
transposed = x.transpose(0, 1)  # 交换维度0和1
permuted = x.permute(2, 0, 1)   # 重排维度顺序
```

## 广播机制

广播允许不同形状的张量进行运算：

```python
# 标量与张量
x = torch.ones(2, 3)
y = x + 1  # 标量1被广播为2×3的全1张量

# 不同形状张量
A = torch.ones(3, 1, 4)
B = torch.ones(2, 4)
C = A + B  # B被广播为(3, 2, 4)

# 显式广播
B_expanded = B.unsqueeze(0).expand(3, 2, 4)
```

