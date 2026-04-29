(cnn-ablation-impl)=
# PyTorch实现

本章提供完整的消融实验代码。代码基于 {doc}`../pytorch-practice/neural-network-module` 和 {doc}`../pytorch-practice/train-workflow` 中的知识。

```{admonition} 两种实现方式
:class: important

本章提供两种层次的实现：

1. **模型定义**（`code/base-model.py`、`code/batch-normal.py`、`code/dropout.py`）
   — 展示每个消融变体的架构差异，是理解"改了哪里"的关键

2. **训练运行**（使用社团的 `mnist-helloworld` 框架）
   — 训练循环、实验管理、结果对比由框架自动完成，专注实验设计本身

建议先阅读模型定义理解架构差异，再用框架运行实验。
```

## 基线模型实现

```{literalinclude} code/base-model.py
:language: python
:linenos:
:caption: 基线CNN模型代码（含 Dropout）
```

```{admonition} 参数量计算
:class: note

| 层 | 计算公式 | 参数量 |
|---|---------|--------|
| Conv1 | $(3 \times 3 \times 3 + 1) \times 32$ | 896 |
| Conv2 | $(3 \times 3 \times 32 + 1) \times 64$ | 18,496 |
| FC1 | $(4,096 + 1) \times 512$ | 2,097,664 |
| FC2 | $(512 + 1) \times 10$ | 5,130 |
| **总计** | | **2,122,186** |

全连接层占 85% 以上参数，卷积层仅占 1.6%。这就是为什么可以冻结卷积层做迁移学习——特征提取器本身很轻量。
```

## 批归一化实现

```{literalinclude} code/batch-normal.py
:language: python
:linenos:
:caption: 带 BatchNorm 的 CNN 代码
```

```{admonition} 消融实验对比：BatchNorm
:class: tip

**预期结果**（参见 {ref}`cnn-ablation-experiment`）：
- 有 BN 的模型收敛更快（约 50% 的训练时间）
- 有 BN 的模型允许使用更大学习率
- 准确率可能略有提升或持平

**实验方法**：
1. 训练基线模型和无 BN 模型各 20 个 epoch
2. 记录每轮的训练/测试准确率
3. 绘制对比曲线，观察 BN 对收敛速度的影响
```

## Dropout实现

```{literalinclude} code/dropout.py
:language: python
:linenos:
:caption: 带 Dropout 的 CNN 代码
```

```{admonition} 消融实验对比：Dropout
:class: tip

| Dropout率 | 训练准确率 | 测试准确率 | 过拟合差距 |
|-----------|------------|------------|------------|
| 0.0（无） | 95% | 78% | 17% |
| 0.5（有） | 89% | 78% | 11% |

**关键观察**：
- 无 Dropout：训练准确率高但测试低 = 过拟合
- 有 Dropout：训练准确率降低但测试不变 = 更好的泛化

**建议实验**：
1. 训练 `dropout_rate=0.0` 的模型
2. 训练 `dropout_rate=0.5` 的模型
3. 对比两者的训练/测试准确率差距
4. 体会"正则化"的实际意义
```

## 使用框架运行消融实验

模型定义好之后，训练由社团的 `mnist-helloworld` 框架处理。详见{doc}`../pytorch-practice/using-framework`中的{ref}`framework-add-model`。

### 注册模型

将消融实验的模型注册到框架的 `ModelRegistry`：

```python
from code.base_model import BaselineCNN
from code.batch_normal import CNNWithBN
from code.dropout import CNNWithDropout

ModelRegistry.register("baseline_cnn", BaselineCNN)
ModelRegistry.register("cnn_with_bn", CNNWithBN)
ModelRegistry.register("cnn_with_dropout", CNNWithDropout)
```

### 实验配置

每个消融变体对应一个 YAML 配置文件（见 `code/configs/`）：

```yaml
# configs/baseline.yaml
model:
  name: baseline_cnn
dataset:
  name: cifar10
training:
  epochs: 20
  batch_size: 64
optimization:
  optimizer: adam
  learning_rate: 0.001
```

### 运行实验

```bash
# 基线模型
python train.py --config code/configs/baseline.yaml

# 消融：移除 Dropout
python train.py --config code/configs/abl_no_dropout.yaml

# 消融：添加 BatchNorm
python train.py --config code/configs/abl_with_bn.yaml

# 或直接在命令行覆盖参数
python train.py --model baseline_cnn --dataset cifar10 --epochs 20 --learning-rate 0.001
```

每次运行自动创建 `runs/exp1/`、`runs/exp2/`... 目录，保存完整配置、checkpoint 和训练曲线。

### 对比实验结果

```bash
# 训练完成后，直接比较两个实验的准确率
cat runs/exp1/logs/training.log | grep "Test Acc"
cat runs/exp2/logs/training.log | grep "Test Acc"

# 或查看训练曲线
open runs/exp1/training_curves.png
open runs/exp2/training_curves.png
```

### 对照：手写 vs 框架

| 环节 | 手写实现 | 框架实现 |
|------|---------|---------|
| 模型定义 | 继承 `nn.Module` | 继承 `nn.Module`（不变） |
| 训练循环 | ~90 行 `train_model()` | `Trainer` 类自动处理 |
| 实验管理 | 手动建目录、命文件名 | `ExperimentManager`，YOLO 自动编号 |
| 超参数 | 硬编码在脚本中 | YAML 配置 + CLI 覆盖 |
| 结果对比 | 手动记录准确率、画图 | 自动保存训练曲线 |

## 代码使用指南

```{admonition} 实验流程
:class: tip

1. 阅读 `base-model.py`、`batch-normal.py`、`dropout.py`，理解每个变体的架构差异
2. 阅读 {ref}`cnn-ablation-experiment` 中的实验设计
3. 将模型注册到框架，创建对应的 YAML 配置文件
4. 运行基线实验，记录结果
5. 逐一运行消融实验（每次只改一个组件）
6. 对比各实验的训练曲线和最终准确率
7. 分析组件重要性，撰写实验报告
```
