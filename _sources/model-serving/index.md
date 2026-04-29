(model-serving)=
# 模型部署与服务：从训练到生产

## 摘要

上一章我们用消融实验验证了 CNN 各组件的贡献，但一个模型在实验环境跑出好结果之后呢？

在 {doc}`../pytorch-practice/index` 中你学会了训练模型，在 {doc}`../cnn-ablation-study/index` 中你学会了验证模型——现在我们将学习如何把模型部署到生产环境，让它真正对外提供服务。

本章将回答四个核心问题：
1. 训练好的 PyTorch 模型如何跨平台导出？
2. 生产环境需要支持多少并发请求？
3. 同步推理和异步推理各适合什么场景？
4. 如何用 CLI 和配置文件管理模型的生命周期？

我们将以 **Ferrinx**——深度学习社团开源的轻量级 ONNX 推理服务（[GitHub](https://github.com/ulink-deep-learning-club/ferrinx.git)）——作为教学工具，让你理解从训练到生产的完整链路。

```{admonition} 学习目标
:class: important

完成本章后，你将能够：
1. 用 `torch.onnx.export` 将 PyTorch 模型导出为 ONNX 格式
2. 理解模型服务架构：同步推理 vs 异步推理、简单模式 vs 分布式模式
3. 使用 Ferrinx 部署并管理模型的生命周期
4. 通过 CLI 完成模型注册、推理调用等操作
5. 为模型配置预处理/后处理流水线
6. 理解 API 认证、限流等生产环境必备的安全机制
```

## 本章概览

| 章节 | 内容 | 与前面章节的联系 |
|------|------|-----------------|
| {doc}`introduction` | 从训练到生产的完整链路 | {doc}`../pytorch-practice/train-workflow` 中的模型 → 这里导出 |
| {doc}`onnx-export` | PyTorch 模型导出为 ONNX 格式 | {doc}`../neural-network-basics/le-net` 的 LeNet → 导出示例 |
| {doc}`serving-architecture` | 模型服务架构设计 | 对比 {doc}`../cnn-ablation-study/experiment-design` 的实验架构 |
| {doc}`deployment-practice` | Ferrinx 服务的部署操作 | {doc}`../pytorch-practice/using-framework` 的框架使用模式 |
| {doc}`the-end` | 总结与完整 MLOps 链路 | 串联所有章节 |

## 学习路径

本章是模型从"实验"到"生产"的**最后一公里**：

```{mermaid}
graph LR
    A[PyTorch训练<br/>学会训练] --> B[ONNX导出<br/>跨平台格式]
    B --> C[服务架构<br/>同步/异步]
    C --> D[部署实践<br/>Ferrinx CLI]
    D --> E[生产环境<br/>API服务]
```

**核心认知**：模型部署不是训练的附属品，而是让 AI 产生价值的**必要环节**——没有部署，再高精度的模型也只是实验室的玩具。

## 本章定位

本章和之前章节的关系：

- {doc}`../neural-network-basics/index` 给出了网络架构的理论
- {doc}`../pytorch-practice/index` 让你能用 PyTorch 实现训练
- {doc}`../cnn-ablation-study/index` 让你能系统验证模型设计
- 本章则将这些训练的模型**推向生产环境**

Ferrinx 是社团开源的项目，目标让 ONNX 模型的部署"一个二进制文件，一个配置文件，就能跑起来"——这与 {doc}`../pytorch-practice/using-framework` 中框架的设计哲学一脉相承。你可以在 [GitHub](https://github.com/ulink-deep-learning-club/ferrinx) 找到源码并贡献代码。欢迎提 Issue 和 PR 参与贡献！

| 前置章节 | 本章应用 |
|---------|---------|
| {doc}`../pytorch-practice/train-workflow` | 训练好的模型 → ONNX 导出 |
| {doc}`../pytorch-practice/using-framework` | 框架使用模式 → 部署实践 |
| {doc}`../cnn-ablation-study/experiment-design` | 实验管理思维 → 生产环境管理 |

## 前置要求

```{admonition} 学习本章前，请确保你已经掌握
:class: caution

学习本章前，请确保你已经掌握：

1. {doc}`../pytorch-practice/index`：用 PyTorch 训练和导出模型
2. 熟悉基础的 REST API 概念（HTTP 方法、JSON 格式）
3. 了解 {doc}`../cnn-ablation-study/experiment-design` 中的实验管理思维
```

## 目录

```{toctree}
:maxdepth: 2

introduction
onnx-export
serving-architecture
deployment-practice
the-end
```
