# 从NumPy到PyTorch

## NumPy基础回顾

NumPy是Python科学计算的基础库，提供多维数组对象和数学函数：

```python
import numpy as np

# 创建数组
arr = np.array([1, 2, 3, 4, 5])
print(f"数组: {arr}")
print(f"形状: {arr.shape}")
print(f"数据类型: {arr.dtype}")

# 基本操作
arr_squared = arr ** 2
arr_sum = np.sum(arr)
arr_mean = np.mean(arr)

# 矩阵运算
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])
C = np.dot(A, B)  # 矩阵乘法
```

## PyTorch张量：NumPy的扩展

PyTorch张量（Tensor）与NumPy数组类似，但支持GPU加速和自动微分：

```python
import torch

# 创建张量
tensor = torch.tensor([1, 2, 3, 4, 5])
print(f"张量: {tensor}")
print(f"形状: {tensor.shape}")
print(f"数据类型: {tensor.dtype}")
print(f"设备: {tensor.device}")

# NumPy与PyTorch互转
numpy_array = tensor.numpy()  # 张量转NumPy
torch_tensor = torch.from_numpy(numpy_array)  # NumPy转张量
```

## 关键差异

```{list-table} NumPy数组 vs PyTorch张量
:header-rows: 1
:widths: 30 35 35

* - **特性**
  - **NumPy数组**
  - **PyTorch张量**
* - 设备支持
  - 仅CPU
  - CPU和GPU
* - 自动微分
  - 不支持
  - 支持（requires_grad=True）
* - 广播规则
  - 类似但不同
  - 类似NumPy
* - 内存布局
  - 行优先（C风格）
  - 行优先（默认）
* - 与深度学习集成
  - 需要额外库
  - 原生支持
```
