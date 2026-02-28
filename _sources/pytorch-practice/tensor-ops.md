# 3. 张量操作详解：PyTorch的核心数据结构

```{admonition} 本章要点
:class: important

- 理解张量的概念和数学基础
- 掌握张量的创建和基本操作
- 熟练使用索引、切片和形状操作
- 理解广播机制和内存布局
- 学习张量的设备管理和性能优化
```

## 3.1 什么是张量？

张量是PyTorch中的基本数据结构，是多维数组的推广。在数学和物理中，张量是描述线性关系的几何对象，而在深度学习中，张量是存储和操作数据的主要方式。

### 3.1.1 张量的数学定义

从数学角度看，张量是多重线性函数：

$$
T: V_1 \times V_2 \times \cdots \times V_n \rightarrow \mathbb{R}
$$

其中 $V_i$ 是向量空间。在PyTorch中，我们主要关注张量的数值表示和计算。

### 3.1.2 张量的维度

```{mermaid}
flowchart TD
    A[张量维度] --> B["0维：标量 Scalar (e.g., 3.14)"]
    A --> C["1维：向量 Vector (e.g., [1, 2, 3])"]
    A --> D["2维：矩阵 Matrix (e.g., [[1,2],[3,4]])"]
    A --> E["3维：立方体 Cube (e.g., RGB图像)"]
    A --> F["n维：高阶张量 (e.g., 批量数据)"]
```

### 3.1.3 PyTorch张量 vs NumPy数组

```{admonition} 关键区别
:class: caution

| 特性 | PyTorch张量 | NumPy数组 |
|------|-------------|-----------|
| **GPU支持** | ✅ 原生支持 | ❌ 需要额外库 |
| **自动微分** | ✅ 内置支持 | ❌ 不支持 |
| **动态计算图** | ✅ 运行时构建 | ❌ 静态计算 |
| **设备管理** | ✅ 自动迁移 | ❌ 手动处理 |
| **性能优化** | ✅ 编译优化 | ⚠️ 有限优化 |

PyTorch张量专为深度学习设计，提供了NumPy数组的所有功能，并增加了GPU加速和自动微分等关键特性。
```

```{figure} ../../_static/images/scalar-to-tensor.png
:width: 80%
:align: center
:caption: 从标量到张量：不同维度的数据表示

标量 → 向量 → 矩阵 → 3D张量 → nD张量
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
