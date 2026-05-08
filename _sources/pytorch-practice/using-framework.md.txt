(using-framework)=
# 使用训练框架

{doc}`train-workflow`中我们手写了完整训练流程，{doc}`best-practices`中我们讨论了工程规范。现在的问题是：**这些实践如何落地成可复用的工具——而不是每次新项目都重写一遍？**

社团的 `mnist-helloworld` 框架就是答案。它不是新知识，而是把{doc}`train-workflow`的流程和{doc}`best-practices`的规范，封装成了一个**工程化系统**。

```{admonition} 本节定位
:class: important

- {doc}`train-workflow`：**学原理**——手写每一行代码，理解背后发生了什么
- {doc}`best-practices`：**学规范**——理解好的工程长什么样、为什么需要它
- **本节**：**用工具**——看前面的原理和规范如何整合成一个可用的框架
```

本节不教你新概念，而是带你走一遍：**"你在第 X 节手写的那段代码，框架里对应的模块是什么，为什么那样设计"**。

---

## 从"每次重写"到"一次封装"

### 核心矛盾

{doc}`train-workflow`最后我们写了一个可运行的训练脚本。它能在 MNIST 上训练分类器。但想象你接下来要做三个实验：

1. 在 CIFAR-10 上测试 LeNet
2. 比较 Adam 和 SGD 的效果
3. 从 MNIST 预训练模型迁移到 CIFAR-10

手动版本的问题：

```python
# 实验一：复制文件，改数据集路径、模型名...
# 实验二：改 optimizer，重新跑...
# 实验三：手动保存权重，加载时小心维度匹配...
# —— 每次改动都要修改代码本体，容易引入 bug，难以回溯
```

这就是{doc}`best-practices`中讨论的"工程规范"要解决的问题。框架把这些规范变成了代码：

```bash
# 三个实验，零代码修改
python train.py --dataset cifar10 --model lenet
python train.py --dataset mnist --model lenet --optimizer sgd --scheduler cosine
python train.py --fork exp1 --dataset cifar10 --freeze "0-0" "0-1" --learning-rate 0.0001
```

### 设计原则回顾

{doc}`best-practices`中我们讨论了四条规范，框架将它们变成了具体实现：

| {doc}`best-practices` 规范 | 框架实现 | 效果 |
|---------------------------|---------|------|
| 配置不硬编码 | `src/config/config.py`：YAML + CLI 双配置 | 改实验改配置不改代码 |
| 模块化项目结构 | `src/datasets/`、`src/models/`、`src/training/` | 新增模型/数据集只需加一个文件 |
| 实验可复现 | `ExperimentManager`：runs/expN/ 自动管理 | 每次运行完整存档，永不覆盖 |
| 检查点管理 | `CheckpointManager`：4 种保存策略 | 断点续训、最佳模型自动筛选 |

---

(framework-config)=
## 配置系统：从硬编码到配置文件

### 你之前做的

{doc}`train-workflow`中，我们把超参数写死在代码开头：

```python
batch_size = 64
epochs = 10
lr = 0.001
```

{doc}`best-practices`中，我们改进为 `@dataclass Config`——参数集中管理，但仍然和代码在一起。

### 框架的做法

框架更进一步：**参数和代码完全分离**。

项目根目录的 `config.yaml` 定义所有默认值：

```yaml
dataset:
  name: mnist
  root: ./data
model:
  name: mynet
  num_classes: 10
training:
  epochs: 20
  batch_size: 64
optimization:
  learning_rate: 1e-3
  optimizer: adamw
  weight_decay: 0.01
  scheduler: cosine
  scheduler_t_max: 20
checkpointing:
  save_frequency: 1
```

运行时用 CLI 参数覆盖：

```bash
python train.py --dataset cifar10 --model lenet --epochs 50
```

优先级规则：

```
CLI 参数（最高） > 自定义 YAML > 默认 config.yaml（最低）
```

**效果**：想尝试 10 组超参数，不再需要改 10 次代码或维护 10 份 Config 子类——10 条命令就够了。

```{admonition} 注意
:class: caution

CLI 参数是扁平的（`--epochs`），YAML 里是嵌套的（`training.epochs`）。两种方式完全等价，`src/config/config.py` 负责两者的合并解析。
```

---

(framework-models)=
## 模型管理：从 nn.Module 到 BaseModel

### 你之前做的

{doc}`neural-network-module`中，我们继承 `nn.Module` 写网络：

```python
class LeNet5(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.c1 = nn.Conv2d(1, 6, 5, padding=2)
        # ...

    def forward(self, x):
        x = torch.tanh(self.c1(x))
        return x
```

### 框架的做法

框架定义了一个 `BaseModel` 抽象类，在 `nn.Module` 之上增加了统一接口：

```python
class BaseModel(nn.Module):
    @property
    def model_type(self): ...        # "classification" / "siamese"
    def get_criterion(self): ...     # 模型自带的损失函数
    def get_model_info(self): ...    # 参数量等信息
```

所有模型通过 **注册机制** 管理：

```bash
# 查看所有可用模型
python -c "from src.models import ModelRegistry; print(ModelRegistry.list_available())"
```

框架内置 4 个系列，共 26 个变体：

**经典 CNN**（用于入门和小数据量）

| 模型 | 参数量 | 你已经在哪学过 | 特点 |
|------|--------|--------------|------|
| `lenet` | ~62K | {doc}`../neural-network-basics/le-net` | 经典 5 层，MNIST 标准 |
| `mynet` | ~122K | — | 不对称卷积 + SiLU + BN，默认模型 |
| `alexnet` | ~2.5M | {doc}`../neural-network-basics/cnn-basics` | 适配小输入的 AlexNet |

**Vision Transformer**（用于更大数据集）

| 模型 | 参数量 | 说明 |
|------|--------|------|
| `bottleneck_vit` | ~9M | CNN 提取特征 + ViT 头部 |
| `fpn_vit_tiny/small/large` | 0.9M/1.7M/3.8M | 金字塔多尺度融合 + ViT |

**Mixture of Experts**（在 FPN-ViT 基础上把 MLP 替换为 MoE）

| 模型 | 机制 |
|------|------|
| `fpn_moe_vit` 系列 | 每个 token 动态选择 Top-K 专家，附带负载均衡损失 |

**Siamese 系列**（输出嵌入向量而非 logits，配合 Triplet Loss）

| 模型 | 用途 |
|------|------|
| `siamese` | 基础孪生网络 |
| `siamese_fpn_vit` 系列 | FPN-ViT backbone 做嵌入 |

```bash
# 使用：一行切换
python train.py --model lenet
python train.py --dataset cifar10 --model fpn_vit_tiny
```

更换模型时，框架自动读取模型的 `input_channels` 和 `input_size`，适配对应的数据集预处理。

(framework-add-model)=
### 添加新模型：3 步

{doc}`best-practices`说"模块化"——框架用注册模式实现：

```python
# 第一步：src/models/my_model.py
from src.models.base import BaseModel

class MyModel(BaseModel):
    def __init__(self, num_classes=10, input_channels=1, **kwargs):
        super().__init__(num_classes, input_channels)
        self.net = nn.Sequential(
            nn.Conv2d(input_channels, 32, 3, padding=1),
            nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(), nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(), nn.Linear(64, num_classes)
        )
    def forward(self, x):
        return self.net(x)
```

```python
# 第二步：在 src/models/__init__.py 注册
from src.models.my_model import MyModel
ModelRegistry.register("my_model", MyModel)
```

```bash
# 第三步：使用
python train.py --model my_model
```

不需要修改框架的任何已有代码。

---

(framework-datasets)=
## 数据集管理：从 DataLoader 到 BaseDataset

### 你之前做的

{doc}`train-workflow`中，用 `torchvision.datasets.MNIST` 加载数据，手动写 transform：

```python
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
train_dataset = datasets.MNIST(root='./data', train=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
```

换数据集？全部重写。

### 框架的做法

每个数据集是一个类，transform 封装在内：

```python
# MNIST 自带 transform（src/datasets/mnist.py）
def get_train_transform(self):
    return transforms.Compose([
        transforms.RandomAffine(degrees=15, translate=0.1, scale=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
```

通过 `DatasetRegistry` 统一管理：

```bash
# 切换数据集，一行命令
python train.py --dataset mnist
python train.py --dataset cifar10
python train.py --dataset subset_631

# 模型自动适配输入尺寸、通道数、类别数
python train.py --dataset cifar10 --model lenet
```

内置 7 个数据集：

| 数据集 | 类型 | 类别 | 输入 | 说明 |
|--------|------|------|------|------|
| `mnist` | 分类 | 10 | 28×28 灰度 | 快速实验首选 |
| `cifar10` | 分类 | 10 | 32×32 彩色 | CNN 入门标准基准 |
| `subset_631` | 分类 | 631 | 64×64 灰度 | 汉字识别，进阶 |
| `subset_1000` | 分类 | 1000 | 64×64 灰度 | 更多汉字 |
| `triplet_mnist` | 度量学习 | 10 | 28×28 灰度 | 在线生成三元组 |
| `balanced_triplet_mnist` | 度量学习 | 10 | 28×28 灰度 | 预均衡三元组 |
| `triplet_subset_1000` | 度量学习 | 1000 | 64×64 灰度 | 汉字 triplet |

(framework-add-dataset)=
### 添加新数据集

与模型注册同理：继承 `BaseDataset` → 实现 `load_data()`、`get_train_transform()`、`get_test_transform()` → 注册。

```python
from src.datasets.base import ClassificationDataset
from src.datasets import DatasetRegistry

class MyDataset(ClassificationDataset):
    def __init__(self, root="./data", **kwargs):
        super().__init__(num_classes=5, input_channels=3, input_size=(64, 64))
        # 实现数据加载逻辑...

DatasetRegistry.register("my_dataset", MyDataset)
```

---

## 训练引擎：从手写循环到 Trainer

### 你之前做的

{doc}`train-workflow`中，我们手写了完整的训练循环——约 50 行代码处理一个 epoch 的训练和验证：

```python
for epoch in range(num_epochs):
    model.train()
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        for inputs, targets in test_loader:
            outputs = model(inputs)
            # 统计准确率...
```

这套代码在 LeNet 上能跑。但如果要：
- 加上早停？加 ~15 行
- 切换 Triplet Loss？重写数据读取
- 用混合精度？包 autocast + GradScaler
- 保存最佳模型和最新模型？加 ~20 行状态管理

**每加一个功能，循环的复杂度翻倍**——这正是框架要封装的部分。

### 框架的做法

`Trainer` 类封装了完整的训练逻辑：

- **自动调度**：分类模式 vs Triplet 模式（根据 `model_type` 和 `dataset_type` 自动判断）
- **早停**：`--patience 5`，验证损失 N 轮不降停止
- **混合精度**：`--mixed-precision`，自动使用 `autocast` + `GradScaler`
- **学习率调度**：`--scheduler cosine`，每轮自动 `scheduler.step()`
- **指标追踪**：loss、accuracy、positive/negative distance（triplet 模式额外追踪）
- **进度显示**：tqdm 进度条
- **曲线绘制**：每轮结束后更新 4 面板训练曲线图

```bash
# 一键启用多个功能
python train.py --dataset cifar10 --model lenet \
    --optimizer sgd --momentum 0.9 \
    --scheduler cosine --scheduler-t-max 50 \
    --mixed-precision --patience 10 \
    --save-frequency 5
```

一行命令等价于之前手写的 ~200 行代码 + 早停逻辑 + 混合精度 + 调度器 + 曲线绘制。

---

(framework-experiments)=
## 实验管理：从手动记录到自动追踪

### 你之前做的

{doc}`train-workflow`中，我们手动创建目录、保存模型：

```python
torch.save({'model_state_dict': model.state_dict(), ...}, 'checkpoint.pt')
```

{doc}`best-practices`中，我们用 `ExperimentLogger` 管理实验目录和 git 信息。

### 框架的做法

`ExperimentManager` 实现了同样的逻辑，并增加了 **YOLO 风格自动编号**：

```bash
python train.py   # → runs/exp1/
python train.py   # → runs/exp2/
python train.py   # → runs/exp3/
```

每次运行生成：

```
runs/exp1/
├── config.yaml              # 本次实验完整配置
├── checkpoints/
│   ├── latest_checkpoint.pt  # 每轮更新，用于断点续训
│   ├── best_model.pt         # 验证准确率最高时保存
│   ├── final_model.pt        # 训练结束时保存
│   └── epoch_10.pt           # 每 save_frequency 轮保存
├── training_curves.png       # loss + accuracy + LR + speed
└── logs/training.log         # 逐行日志
```

**对比实验变得极其简单**：

```bash
# 基线
python train.py --model lenet --dataset mnist
# → runs/exp1/

# 换模型对比
python train.py --model mynet --dataset mnist
# → runs/exp2/

# 直接比较两个训练曲线
open runs/exp1/training_curves.png
open runs/exp2/training_curves.png
```

---

## Checkpoint 管理：两种恢复策略

{doc}`train-workflow`中，我们只保存了权重。框架的 `CheckpointManager` 保存完整训练状态：

```
checkpoint = {
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict(),
    'best_acc': best_acc,
    'loss': val_loss
}
```

基于此，框架支持两种恢复方式：

| 操作 | 命令 | 行为 | 适用场景 |
|------|------|------|---------|
| **Resume** | `--resume exp1` | 恢复完整状态（模型+优化器+调度器+epoch计数），**覆盖原目录** | 训练被中断，继续跑完 |
| **Fork** | `--fork exp1` | 只加载模型权重，创建**新目录** runs/exp2，支持改配置 | 迁移学习、调参 |

Fork 的 **非严格加载** 值得一提：如果新模型架构与 checkpoint 不完全一致（比如分类头从 10 类改成 5 类），框架会自动匹配兼容的层、跳过不兼容的层，并打印详细报告。

---

## 迁移学习：层冻结

{doc}`../transfer-learning/index`中我们学到：迁移学习的核心是**冻结通用特征提取层，只训练任务特定层**。框架的 `--freeze` 参数实现了这一点。

```bash
# 冻结方式一：按层 ID
python train.py --model lenet --freeze "0-0" "0-1"
# 冻结方式二：按 ID 范围
python train.py --model lenet --freeze "0-0:0-2"
# 冻结方式三：按层名模式
python train.py --model lenet --freeze "features"
```

层 ID 映射在训练开始时打印：

```
Layer ID Mapping:
  0-0: conv1
  0-1: conv2
  1-0: fc1
  1-1: fc2
```

冻结后，对应参数不在 optimizer 中注册，`.backward()` 跳过它们——这正是{doc}`auto-grad`中 `requires_grad=False` 的实际应用。

**实战：MNIST → CIFAR-10 迁移**

```bash
# 第 1 步：MNIST 上训练
python train.py --model lenet --dataset mnist --epochs 20
# → runs/exp1/checkpoints/best_model.pt

# 第 2 步：冻结卷积层，在 CIFAR-10 上微调全连接层
python train.py --fork exp1 --dataset cifar10 --model lenet \
    --epochs 15 --learning-rate 0.0001 \
    --freeze "0-0" "0-1"
```

---

## 进阶功能

### 度量学习（Siamese + Triplet Loss）

{doc}`../neural-network-basics/neural-training-basics`中讨论过度量学习。框架原生支持：

```bash
python train.py --dataset balanced_triplet_mnist --model siamese \
    --embedding-dim 128 --epochs 30
```

Trainer 检测到 `model_type="siamese"` 或 `dataset_type="triplet"` 时，自动切换训练模式：
- 加载 (anchor, positive, negative, label) 四元组
- 使用 `OnlineTripletLoss`（batch 内挖掘难例）
- 追踪 positive distance 和 negative distance——理想状态是前者远小于后者

### 向量搜索

框架集成了 Qdrant 向量数据库，训练好的 Siamese 模型可以用于相似性搜索：

```bash
# 索引：提取特征存入向量库
python -m src.utils.qdrant_search index \
    --checkpoint runs/exp1/checkpoints/best_model.pt \
    --dataset mnist --collection mnist_digits

# 搜索：找与输入图片最相似的样本
python -m src.utils.qdrant_search search \
    --checkpoint runs/exp1/checkpoints/best_model.pt \
    --collection mnist_digits --image path/to/digit.png
```

### GUI 实时验证

框架附带 Tkinter GUI，用于在摄像头前实时测试模型：

```bash
cd gui-example
pip install -r requirements.txt
python main_gui.py
```

支持两种模式：
- **手动拍照**：按一次识别一次
- **连续检测**：每 500ms 识别一次，适合演示

预测结果以彩色柱状图展示：绿色（>80% 置信度）、橙色（>50%）、红色（<50%）。

---

## 对照总结

```{mermaid}
graph LR
    subgraph 你手写的
        A1[DataLoader] --> A2[训练循环]
        A2 --> A3[torch.save]
        A3 --> A4[手动对比]
    end
    subgraph 框架封装的
        B1[DatasetRegistry] --> B2[Trainer]
        B2 --> B3[CheckpointManager]
        B3 --> B4[ExperimentManager]
    end
    A1 -.->|封装| B1
    A2 -.->|封装| B2
    A3 -.->|封装| B3
    A4 -.->|封装| B4
```

| 你手写的 | 框架封装的 | 你现在可以专注的 |
|---------|-----------|----------------|
| `for epoch` 循环 + early stopping 逻辑 | `Trainer` 类 | 模型架构设计 |
| `torch.save` + 手动管理文件 | `CheckpointManager` | 实验方案设计 |
| 手动建目录、记日志 | `ExperimentManager` | 结果分析 |
| 改代码换数据集/模型 | `YAML + CLI 配置` | 超参数调优 |
| 手动 `model.to(device)` | 自动设备检测 | 算法创新 |
| 手写数据增强和预处理 | 每个数据集自带 transform pipeline | 数据探索 |

---

## 下一步

你在{doc}`../cnn-ablation-study/index`中将用框架做真正的对比实验——每个消融变体对应一个 `--config` 文件，`runs/expN/` 自动记录结果，ExperimentManager 保障实验可复现。

{doc}`the-end`中列出了更多学习资源。

如果想深入理解框架实现，读以下三个文件和你的手写代码做对照：

- `src/training/trainer.py`：你手写的训练循环 → 框架的封装
- `src/training/checkpoint.py`：你手写的 `torch.save` → 框架的自动管理
- `src/config/config.py`：你手写的 `@dataclass Config` → 框架的 YAML + CLI 解析

**从"每次重写"到"一次封装，反复使用"——这就是工程化的意义。**

## 参与到社团项目的开发

本章介绍的项目是UCS 深度学习社维护的开源项目，欢迎你贡献代码、报告问题或提出改进建议：

- [**mnist-helloworld**](https://github.com/ulink-deep-learning-club/mnist-helloworld)（训练框架）：本课程使用的训练框架，涵盖数据集注册、模型管理、实验追踪等完整功能。如果你注册了新的数据集或模型，欢迎提交 Pull Request 让更多人受益。

---

**贡献的方式很简单**：发现问题 → 提 Issue → 讨论方案 → 提交 PR。即使只是修正一个文档错别字，也是对社区有意义的贡献。**真正理解一个框架最好的方式，就是尝试改进它。**
