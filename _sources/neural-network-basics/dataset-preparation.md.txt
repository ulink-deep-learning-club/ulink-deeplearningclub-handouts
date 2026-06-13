(dataset-preparation)=
# 数据准备：从原始文件到训练流水线

{ref}`cnn-basics`和{ref}`pooling`中我们学习了 CNN 如何提取特征——但有一个问题：**数据从哪里来？怎么从硬盘上的文件变成 GPU 里的张量？**

本节回答这个问题，打造一条完整的数据流水线：从原始数据 → 预处理 → Dataset → DataLoader → GPU。

(data-preparation-flow)=
## 数据流水线概览

```{mermaid}
graph LR
    A[原始数据<br/>磁盘上的文件] --> B[Transform<br/>预处理/增强]
    B --> C[Dataset<br/>Python可索引对象]
    C --> D[DataLoader<br/>自动批量化+并行加载]
    D --> E[GPU<br/>训练循环]
```

从右往左看：**PyTorch 不需要你知道文件存在哪里，它只需要一个能`按索引返回(图片, 标签)`的对象**——这就是 Dataset。DataLoader 负责在这个对象上自动打包成 batch、打乱顺序、多进程预加载。

## 第一步：数据从哪里来

### 内置数据集（最省心）

对于 MNIST、CIFAR-10、ImageNet 等经典基准数据集，PyTorch 通过 `torchvision.datasets` 一行代码下载并准备好：

```python
import torchvision

# 下载 MNIST（第一次自动下载，之后复用本地缓存）
train_data = torchvision.datasets.MNIST(
    root='./data',      # 本地缓存目录
    train=True,         # 训练集
    download=True       # 第一次运行时设为 True
)
```

```{admonition} 常用内置数据集
:class: note

| 数据集 | PyTorch 类 | 任务 | 规模 |
|--------|-----------|------|------|
| MNIST | `torchvision.datasets.MNIST` | 手写数字分类（10类） | 60K训练 / 10K测试 |
| Fashion-MNIST | `torchvision.datasets.FashionMNIST` | 服装分类（10类） | 60K训练 / 10K测试 |
| CIFAR-10 | `torchvision.datasets.CIFAR10` | 通用物体分类（10类） | 50K训练 / 10K测试 |
| CIFAR-100 | `torchvision.datasets.CIFAR100` | 通用物体分类（100类） | 50K训练 / 10K测试 |
| ImageNet | `torchvision.datasets.ImageNet` | 大规模分类（1000类） | 1.2M训练 / 50K验证 |
```

### 自定义数据集（真实项目）

真实场景中你的数据通常是以文件夹形式组织的：

```
data/
├── train/
│   ├── cat/        # 按类别分文件夹
│   │   ├── cat_001.jpg
│   │   ├── cat_002.jpg
│   │   └── ...
│   ├── dog/
│   └── ...
└── test/
    ├── cat/
    ├── dog/
    └── ...
```

对于这种结构，`torchvision.datasets.ImageFolder` 直接读取：

```python
from torchvision.datasets import ImageFolder

# 自动按文件夹名分配标签：cat→0, dog→1, ...
dataset = ImageFolder(root='data/train')
print(dataset.classes)      # ['cat', 'dog', ...]
print(dataset.class_to_idx) # {'cat': 0, 'dog': 1, ...}
```

```{warning}
ImageFolder 要求数据按**文件夹名 = 类别名**组织。如果你的数据是 CSV/JSON/X光片等非图像格式，就需要自己写 Dataset 类（见第六步）。
```

## 第二步：Transform——从图像到张量

原始图像是 JPEG/PNG 文件（像素值 0\~255），不能直接喂给神经网络，需要**三步标准处理**：

```{mermaid}
graph LR
    A["JPEG文件<br/>PIL Image"] --> B["ToTensor<br/>H×W×C → C×H×W<br/>0-255 → 0.0-1.0"]
    B --> C["Normalize<br/>(x - mean) / std<br/>中心化+缩放到"]
    C --> D["最终输入<br/>分布≈N(0,1)"]
```

### ToTensor：格式转换 + 归一化

```
transforms.ToTensor()  # 做了两件事：
                      # 1. PIL Image (H×W×C) → Tensor (C×H×W)
                      # 2. [0, 255] → [0.0, 1.0]
```

### Normalize：标准化

图像数据中，像素强度差异很大（背景暗、文字亮）。标准化让每个通道的均值为 0、方差为 1，**帮助梯度下降更稳定地收敛**。

```python
from torchvision import transforms

transform = transforms.Compose([
    transforms.ToTensor(),
    # (x - mean) / std，mean 和 std 是数据集的统计值
    # MNIST 是灰度图，所以只有 1 个通道
    transforms.Normalize((0.1307,), (0.3081,))
])
```

这里的 `(0.1307,)` 和 `(0.3081,)` 是从整个 MNIST 训练集计算出来的**全局均值和标准差**。它们的作用是让模型输入具有统一的数值范围，不受亮度、对比度的影响。

**对于彩色图像**（如 CIFAR-10）每个通道分别标准化：

```python
# CIFAR-10 的 RGB 三通道统计值
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.4914, 0.4822, 0.4465),  # R/G/B 各通道均值
        std=(0.2023, 0.1994, 0.2010)    # R/G/B 各通道标准差
    )
])
```

### 训练集与验证集的 transform 不同

这是实践中最常犯的错误之一：**验证集和测试集不能做数据增强**，只能用标准的 ToTensor + Normalize。

```python
# 训练集 transform：可做数据增强
train_transform = transforms.Compose([
    transforms.RandomRotation(10),        # 随机旋转 ±10°
    transforms.RandomAffine(0, translate=(0.1, 0.1)),  # 随机平移 ±10%
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# 验证/测试集 transform：只做标准化
val_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
```

```{admonition} 为什么验证集不能做数据增强？
:class: warning

数据增强（旋转、平移、翻转）本质上是在**制造"假样本"**来扩增训练数据。但在验证时，我们需要**评估模型在真实数据上的表现**——如果验证集也被增强了，评估结果就不反映真实性能了。就像你平时用模拟题练习（数据增强），但考试时题目必须是原题（无增强）。
```

### 常见数据增强操作一览

| 操作 | transforms 类 | 典型参数 | 效果 |
|------|---------------|---------|------|
| 随机旋转 | `RandomRotation` | `degrees=10` | 小幅旋转，增加旋转不变性 |
| 随机平移 | `RandomAffine` | `translate=(0.1, 0.1)` | 小幅平移，增加平移不变性 |
| 随机翻转 | `RandomHorizontalFlip` | `p=0.5` | 水平镜像，增加翻转不变性 |
| 随机裁剪 | `RandomCrop` / `RandomResizedCrop` | `size=224` | 局部特征，增加尺度不变性 |
| 色彩抖动 | `ColorJitter` | `brightness=0.2` | 调整亮度/对比度，增加光照不变性 |

(albumentations)=
### Albumentations：更强大的图像增强库

上面表格中的操作可以应付 MNIST/CIFAR 等基础任务，但在真实项目中（特别是竞赛和生产环境），你可能需要**更强、更快、更丰富的增强策略**——这就是 {doc}`../../pytorch-practice/train-workflow` 中提到的 {doc}`../../pytorch-practice/best-practices` 里会再次讨论的 Albumentations。

#### 为什么用 Albumentations？

| 对比维度 | torchvision.transforms | Albumentations |
|---------|----------------------|---------------|
| **增强种类** | ~20种 | **~70种**，涵盖像素级、空间级、噪声模拟 |
| **速度** | 一般 | **快 2~5 倍**（基于 OpenCV 实现） |
| **API 一致性** | 每个类参数风格不同 | 统一的 `always_apply` / `p=0.5` 概率控制 |
| **边界框/掩码同步增强** | ❌ 不支持 | ✅ 目标检测/分割的标注同步变换 |
| **pip install** | 随 torchvision 自带 | `pip install albumentations` 单独安装 |

#### 安装

```bash
pip install albumentations
```

#### 基本用法

Albumentations 的 API 与 `transforms.Compose` 类似，但参数更一致，并且所有操作都支持统一的概率控制：

```python
import albumentations as A
from albumentations.pytorch import ToTensorV2

# 训练集增强
# 注意每个操作都有 p=... 指定概率，默认 p=1
# A.Compose 按顺序应用，与 torchvision 的 transforms.Compose 相同
train_transform = A.Compose([
    A.HorizontalFlip(p=0.5),               # 50% 概率翻转
    A.Rotate(limit=15, p=0.8),             # 80% 概率旋转 ±15°
    A.RandomBrightnessContrast(p=0.5),     # 50% 概率调整亮度对比度
    A.Normalize(mean=(0.1307,), std=(0.3081,)),
    ToTensorV2(),
])

# 验证集：只做标准化（与 torchvision 一致）
val_transform = A.Compose([
    A.Normalize(mean=(0.1307,), std=(0.3081,)),
    ToTensorV2(),
])
```

#### 与 PyTorch Dataset 集成

Albumentations 的输入输出格式与 torchvision 不同，需要在 Dataset 的 `__getitem__` 中做适配：

```python
from torch.utils.data import Dataset
import cv2

class AlbumentationsDataset(Dataset):
    """支持 Albumentations 增强的 Dataset"""
    
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # 1. 读取图像（Albumentations 需要 numpy 数组格式）
        image = cv2.imread(self.image_paths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = self.labels[idx]
        
        # 2. 应用增强（传入 dict，返回 dict）
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        
        return image, label
```

#### Albumentations 的独家增强操作

以下是一些 torchvision 没有、但在竞赛中非常常用的增强：

| 操作 | 类 | 效果 |
|------|-------|------|
| **网格失真** | `A.GridDistortion` | 模拟镜头畸变，增加几何鲁棒性 |
| **光学畸变** | `A.OpticalDistortion` | 模拟鱼眼镜头效果 |
| **Cutout / CoarseDropout** | `A.CoarseDropout` | 随机遮罩部分区域，防止过拟合到特定纹理 |
| **像素级噪声** | `A.GaussNoise` | 模拟传感器噪声，增加抗噪能力 |
| **锐化/模糊** | `A.Sharpen` / `A.Blur` / `A.MotionBlur` | 模拟对焦不准或运动模糊 |
| **通道偏移** | `A.ChannelShuffle` | 随机打乱颜色通道，增加色彩鲁棒性 |

#### 何时用 torchvision，何时用 Albumentations？

| 场景 | 推荐 | 理由 |
|------|------|------|
| MNIST/CIFAR 学习入门 | torchvision | 零额外依赖，够用 |
| ImageNet 级分类 | torchvision 或 Albumentations | 两者均可 |
| **目标检测 / 语义分割** | **Albumentations** | 标注（边界框、掩码）同步变换是关键能力 |
| **比赛 / 生产调优** | **Albumentations** | 70+ 增强策略，精细控制，可反复组合 |
| 自定义数据加载逻辑 | 两者结合 | torchvision 处理基础，Albumentations 扩展增强 |

```{admonition} 数据增强哲学：适度优于过度
:class: tip

数据增强不是越多越好。过多的增强（如同时做旋转+畸变+Cutout+噪声）可能导致：
1. **样本失真**：增强后的图像人类都认不出，模型更不可能学会
2. **训练不稳定**：输入分布变化过大，梯度难以收敛
3. **性价比递减**：前 3~5 种增强效果最显著，超过 8 种几乎没有额外收益

一个实用的起点是：**旋转 + 翻转 + 颜色抖动 + Cutout**，根据任务逐步追加。
```

## 第三步：Dataset——数据结构的统一接口

**Dataset 是 PyTorch 数据系统的核心抽象**。它定义了一个简单的协议："给我一个索引，返回一个 (数据, 标签) 对"。

```python
from torch.utils.data import Dataset

# 任何 Dataset 都只需要实现两个方法
class MyDataset(Dataset):
    def __len__(self):
        """返回数据集总大小"""
        pass
    
    def __getitem__(self, idx):
        """返回第 idx 个样本的 (数据, 标签) """
        pass
```

对于内置数据集（如 MNIST），PyTorch 已经实现了这两个方法，你不需要自己写。

**Dataset 只负责"有什么数据"，不负责"怎么组织成 batch"**——那是 DataLoader 的事。

```{admonition} Dataset vs DataLoader 的职责边界
:class: important

| 组件 | 职责 | 不负责 |
|------|------|--------|
| **Dataset** | 索引 → (数据, 标签) | 打乱顺序、打包batch、多进程 |
| **DataLoader** | Dataset → batch | 文件读取、预处理 |
```

## 第四步：DataLoader——从单样本到批量

有了 Dataset 之后，DataLoader 把它变成一个**可迭代的批量数据源**：

```python
from torch.utils.data import DataLoader

train_loader = DataLoader(
    train_dataset,
    batch_size=64,     # 每批 64 个样本
    shuffle=True,      # 每 epoch 打乱顺序
    num_workers=2,     # 2 个子进程预加载数据
    pin_memory=True    # 加速 GPU 数据传输
)
```

### 关键参数详解

**`batch_size`**：每批打包多少个样本。模型是**每批更新一次参数**，所以 batch size 直接影响：

- 内存占用：batch size × 单个样本大小
- 梯度稳定性：batch 越大，梯度估计越准确
- 泛化能力：过大的 batch 可能降低泛化性能

**`shuffle`**：每轮训练前打乱数据。如果不打乱，模型会"记住"数据顺序而非真正学习特征。

**`num_workers`**：用几个子进程预加载下一批数据。设置为 2\~8 可以显著加速数据加载，但过大反而增加开销。

**`pin_memory`**：将数据固定在 CPU 内存的特定区域，**加速 CPU → GPU 传输**。设为 `True` 几乎总是更好。

### 实际效果演示

```python
# 假设 train_dataset 有 60,000 张 MNIST 图片
train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True,
    num_workers=2
)

# 每次迭代返回一个 batch
for batch_idx, (images, labels) in enumerate(train_loader):
    # images.shape: [64, 1, 28, 28]  — 64张灰度图
    # labels.shape: [64]             — 64个标签
    # batch_idx: 0, 1, 2, ..., 937   — 共 938 个 batch
    
    if batch_idx == 0:
        print(f"image shape: {images.shape}")  # [64, 1, 28, 28]
        print(f"label shape: {labels.shape}")   # [64]
        print(f"labels: {labels}")               # tensor([3, 7, 1, ...])
        print(f"total batches per epoch: {len(train_loader)}")  # 938
        break
```

```{admonition} GPU 数据传输路径
:class: tip

流式加载机制保证了训练循环不会因为 I/O 而停滞——GPU 在算第 N 批时，CPU 已经在加载第 N+1 批了。
```

## 第五步：数据集拆分——train / val / test

**模型训练不能只看训练集**，需要三组数据各司其职：

| 数据集 | 用途 | 使用频率 | 核心要求 |
|--------|------|---------|---------|
| **训练集** | 更新模型参数（权重+偏置） | 每个 epoch 反复使用 | 数据量大，覆盖多样性 |
| **验证集** | 选择超参数、检测过拟合 | 每 epoch 后评估一次 | 与训练集独立分布，但不可参与训练 |
| **测试集** | 最终报告模型性能 | 只在全部调参完成后用一次 | 完全隔离，绝不可泄露 |

### 最常用：随机拆分（train_test_split）

很多内置数据集只给了训练/测试两份，**需要自己从中分出验证集**。

```python
from torch.utils.data import random_split

# MNIST 标准划分：60K 训练 / 10K 测试
# 从 60K 训练集中分出 10% 作为验证集
train_dataset = torchvision.datasets.MNIST(root='./data', train=True)
test_dataset  = torchvision.datasets.MNIST(root='./data', train=False)

val_size = int(0.1 * len(train_dataset))  # 6,000 张
train_size = len(train_dataset) - val_size  # 54,000 张

# random_split 按比例随机拆分，不会破坏类别平衡
train_subset, val_subset = random_split(
    train_dataset, 
    [train_size, val_size],
    generator=torch.Generator().manual_seed(42)  # 固定种子，确保可复现
)

print(f"训练集: {len(train_subset)}")   # 54,000
print(f"验证集: {len(val_subset)}")     # 6,000
print(f"测试集: {len(test_dataset)}")   # 10,000
```

```{admonition} 为什么需要固定 random_split 种子？
:class: tip

不固定种子的话，每次运行程序验证集会不同——今天在 A 组上验证好，明天可能在 B 组上不好，实验结果不可复现。固定 `manual_seed(42)` 确保每次拆分结果一致。
```

### 分层拆分（保持类别比例）

`random_split` 是随机抽取，在小数据集上可能破坏类别平衡。如果原始数据分布不均（如猫 90%，狗 10%），随机抽取的验证集可能恰好没有狗。

```python
from sklearn.model_selection import train_test_split
import numpy as np

# 获取所有数据的标签
labels = [train_dataset[i][1] for i in range(len(train_dataset))]

# 分层拆分：按标签比例分配
train_idx, val_idx = train_test_split(
    np.arange(len(train_dataset)),
    test_size=0.1,
    stratify=labels,    # 按标签分层
    random_state=42
)

from torch.utils.data import Subset
train_subset = Subset(train_dataset, train_idx)
val_subset   = Subset(train_dataset, val_idx)
```

```{admonition} scikit-learn vs torch 的选择
:class: note

- 数据量大且类别均衡 → `random_split` 就够了（简单、无额外依赖）
- 小数据集或类别不均衡 → `train_test_split(stratify=...)`（需要 sklearn）
- 做交叉验证 → `KFold` / `StratifiedKFold`（也需要 sklearn）
```

### 完整数据流水线模板

把上述所有步骤组合成一个可复用的函数：

```python
def prepare_mnist_data(batch_size=64, val_split=0.1, num_workers=2):
    """
    返回 train_loader, val_loader, test_loader
    
    Args:
        batch_size: 每批样本数
        val_split:  从训练集中分出的验证集比例
        num_workers: 并行加载进程数
    """
    # 1. 定义 transform（训练集有增强，验证/测试集无增强）
    train_transform = transforms.Compose([
        transforms.RandomRotation(10),
        transforms.RandomAffine(0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # 2. 加载原始数据
    full_train = torchvision.datasets.MNIST(
        root='./data', train=True, download=True
    )
    test_dataset = torchvision.datasets.MNIST(
        root='./data', train=False, transform=val_transform
    )
    
    # 3. 拆分 train / val
    val_size = int(val_split * len(full_train))
    train_size = len(full_train) - val_size
    
    train_dataset, val_dataset = random_split(
        full_train, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # 4. 分别应用 transform（注意：random_split 之后才 apply transform）
    # 因为 transform 已经在 Dataset 初始化时应用了，
    # 所以这里我们预先在 full_train 上应用了 transform（略）
    # 
    # 实际做法：用 Subset + 重写 __getitem__，或提前 apply
    # 这里简化为：在初始化 Dataset 时直接传 transform
    # 详见下面的"最佳实践"完整示例
    
    # 5. 创建 DataLoader
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size,
        shuffle=True, num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size,
        shuffle=False, num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size,
        shuffle=False, num_workers=num_workers, pin_memory=True
    )
    
    return train_loader, val_loader, test_loader
```

### 验证集与测试集的严格区分

在训练循环中，验证集和测试集的使用方式截然不同：

```python
for epoch in range(epochs):
    # —— 训练阶段 ——
    model.train()
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        loss = criterion(model(images), labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    
    # —— 验证阶段 ——
    model.eval()
    val_loss = 0
    with torch.no_grad():  # 验证时不计算梯度
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            val_loss += criterion(model(images), labels).item()
    print(f"Epoch {epoch}: val_loss = {val_loss / len(val_loader):.4f}")
    
    # —— 早停：验证集 loss 不再下降时停止训练 ——
    if val_loss > best_val_loss:
        patience_counter += 1
        if patience_counter >= 5:  # 连续 5 个 epoch 没改善
            break                   # 停止训练
    
# —— 最终测试（只用一次！）——
model.eval()
test_acc = 0
with torch.no_grad():
    for images, labels in test_loader:
        test_acc += (model(images).argmax(1) == labels).sum().item()
print(f"Test accuracy: {test_acc / len(test_loader.dataset):.2%}")
```

## 第六步：自定义 Dataset——处理任意格式的数据

当 torchvision 和 ImageFolder 都满足不了需求时（比如你需要读取 CSV 文件、医疗 DICOM 图像、自定义二进制格式），就自己写 Dataset 类：

```python
import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

class CustomImageDataset(Dataset):
    """从 CSV 文件 + 图片文件夹读取数据"""
    
    def __init__(self, csv_file, img_dir, transform=None):
        """
        Args:
            csv_file: CSV 文件路径，包含 'filename', 'label' 两列
            img_dir:  图片所在文件夹
            transform: 预处理流水线
        """
        self.labels = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        """返回第 idx 个样本的 (image, label)"""
        # 1. 读取图片
        img_name = self.labels.iloc[idx, 0]
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        
        # 2. 读取标签
        label = self.labels.iloc[idx, 1]
        
        # 3. 应用预处理
        if self.transform:
            image = self.transform(image)
        
        return image, label
```

```{admonition} 关键设计原则
:class: warning

- `__getitem__` **在每次被调用时即时读取文件**，而不是在 `__init__` 中一次性把所有图片加载到内存。如果数据集有 10,000 张图，一次性加载需要数十 GB 内存，按需加载只需要几 MB。
- 一种常见的优化是**缓存到内存**：如果内存够大且频繁访问，可以在第一次读取后缓存结果。PyTorch 也提供了 `torchdata` 库来处理这类场景。
```

## 总结

| 步骤 | 组件 | 一句话 | 常见坑 |
|------|------|--------|--------|
| 数据源 | `torchvision.datasets` | 下载即用，不用自己找 | 忘记设 `download=True` |
| 预处理 | `transforms.Compose` | 图像→张量→标准化 | 验证集误用了数据增强 |
| 数据集 | `Dataset` / `ImageFolder` | 按索引返回 (数据, 标签) | 自定义 Dataset 时忘记 `convert('RGB')` |
| 拆分 | `random_split` / `Subset` | 训练 / 验证 / 测试三份 | 忘记固定种子，不可复现 |
| 批量化 | `DataLoader` | 自动打包 batch + 并行加载 | `num_workers=0` 导致训练速度瓶颈 |

### 下一步

掌握了数据准备后，下一节{doc}`le-net`将带你把数据跑进第一个完整的 CNN 架构——LeNet-5。你会看到，理论上的卷积操作+数据准备的Pytorch代码如何**串联成一个完整可训练的模型**。
