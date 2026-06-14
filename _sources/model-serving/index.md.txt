(model-serving)=
# 模型部署与服务：从训练到生产

```{only} html
模型训练好了，**怎么让别人用？** 本章从 ONNX 导出到 Ferrinx 部署，打通从训练到生产的最后一公里。

~~~{rubric} 本章概览
:heading-level: 2
~~~

| 章节 | 内容 | 与前面章节的联系 |
| ---------- | ---------- | ---------- |
| {doc}`introduction` | 从训练到生产的完整链路 | {doc}`../pytorch-practice/train-workflow` 中的模型 → 这里导出 |
| {doc}`onnx-export` | PyTorch 模型导出为 ONNX 格式 | {doc}`../cnn-expedition/practice-peak/le-net` 的 LeNet → 导出示例 |
| {doc}`model-optimization` | 量化与剪枝：让模型又轻又快 | 导出的 ONNX 模型 → 压缩加速 |
| {doc}`serving-architecture` | 模型服务架构设计 | 对比 {doc}`../cnn-expedition/ablation-study/experiment-design` 的实验架构 |
| {doc}`deployment-practice` | Ferrinx 服务的部署操作 | {doc}`../pytorch-practice/using-framework` 的框架使用模式 |
| {doc}`the-end` | 总结与完整 MLOps 链路 | 串联所有章节 |

~~~{rubric} 学习路径
:heading-level: 2
~~~

本章是模型从"实验"到"生产"的**最后一公里**：

~~~{mermaid}
graph LR
    A[PyTorch训练<br/>学会训练] --> B[ONNX导出<br/>跨平台格式]
    B --> C[模型优化<br/>量化与剪枝]
    C --> D[服务架构<br/>同步/异步]
    D --> E[部署实践<br/>Ferrinx CLI]
    E --> F[生产环境<br/>API服务]
~~~

**核心认知**：模型部署不是训练的附属品，而是让 AI 产生价值的**必要环节**——没有部署，再高精度的模型也只是实验室的玩具。

~~~{rubric} 前置知识
:heading-level: 2
~~~

| 前置章节 | 本章应用 |
| ---------- | ---------- |
| {doc}`../pytorch-practice/train-workflow` | 训练好的模型 → ONNX 导出 |
| {doc}`../pytorch-practice/using-framework` | 框架使用模式 → 部署实践 |
| {doc}`../cnn-expedition/ablation-study/experiment-design` | 实验管理思维 → 生产环境管理 |
```

```{toctree}
:maxdepth: 2
:hidden:

introduction
onnx-export
model-optimization
serving-architecture
deployment-practice
the-end
```
