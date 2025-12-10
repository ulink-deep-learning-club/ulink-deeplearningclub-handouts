# 自动微分

## 计算图基础

PyTorch使用动态计算图记录张量操作：

```{literalinclude} code/auto-grad-dyn-comp-diagram.py
:language: python
:linenos:
:caption: 动态计算图
```

## 梯度计算规则

```{literalinclude} code/auto-grad-calc.py
:language: python
:linenos:
:caption: 梯度计算
```

## 梯度累积与控制

```{literalinclude} code/auto-grad-acc.py
:language: python
:linenos:
:caption: 梯度累积与控制
```

## 禁用梯度计算

```{literalinclude} code/auto-grad-disable-calc.py
:language: python
:linenos:
:caption: 禁用梯度计算
```
