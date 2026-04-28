(cnn-ablation-impl)=
# PyTorch实现

本章提供完整的 PyTorch 代码实现。代码基于 {doc}`../pytorch-practice/neural-network-module` 和 {doc}`../pytorch-practice/train-workflow` 中的知识，建议先复习这些前置内容。

```{admonition} 代码使用指南
:class: tip

1. **基线模型** (`base-model.py`)：先跑通这个，确保环境配置正确
2. **逐一修改**：每次只修改一个组件，观察变化
3. **记录结果**：用表格记录每个实验的配置和结果
4. **对比分析**：与 {ref}`cnn-ablation-experiment` 中的示例数据对比，理解差异
```

## 基线模型实现

```{literalinclude} code/base-model.py
:language: python
:linenos:
:caption: 基线CNN模型代码
```

## 训练循环

```{literalinclude} code/training-cycle.py
:language: python
:linenos:
:caption: 训练循环代码
```

## 批归一化实现

```{literalinclude} code/batch-normal.py
:language: python
:linenos:
:caption: 批归一化实现代码
```

## Dropout实现

```{literalinclude} code/dropout.py
:language: python
:linenos:
:caption: Dropout实现代码