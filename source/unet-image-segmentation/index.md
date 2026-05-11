(unet-image-segmentation)=
# U-Net：图像分割的革命

```{only} html
分类问题是"看整体"，分割问题是"看每个点"——每个像素属于什么？U-Net 的答案很巧妙：**用对称的解码器恢复空间，再用"抄近道"（跳跃连接）把丢失的细节传回来**。

~~~{rubric} 本章概览
:heading-level: 2
~~~

| 章节 | 内容 | 与前面章节的联系 |
| ------ | --------- | --------- |
| {doc}`introduction` | 分割问题与 U-Net 的核心思想 | "在哪"和"是什么"必须同时解决 |
| {doc}`u-net` | U 形架构：编码器、解码器、跳跃连接 | {doc}`../neural-network-basics/le-net` 的对称扩展 |
| {doc}`loss-func` | Dice 损失与交叉熵损失 | {ref}`loss-functions` 的进阶应用 |
| {doc}`core-impl` | 从零实现 U-Net | {doc}`../pytorch-practice/neural-network-module` 的实践 |
| {doc}`practice` | 数据增强与训练技巧 | {doc}`../neural-network-basics/neural-training-basics` 的迁移 |
| {doc}`the-end` | 应用、变体、局限与展望 | U-Net 为什么十年不过时 |

~~~{rubric} 学习路径
:heading-level: 2
~~~

本章是从"分类"到"分割"的**范式跃迁**：

~~~{mermaid}
graph LR
    A[CNN分类<br/>这是什么？] --> B[图像分割<br/>每个像素是什么？]
    B --> C[编码器<br/>提取语义]
    C --> D[解码器<br/>恢复空间]
    D --> E[跳跃连接<br/>保留细节]
~~~

**核心认知**：分割不是"输出更多的分类"，而是需要同时解决"是什么"和"在哪"两个互补问题。

~~~{rubric} 前置知识
:heading-level: 2
~~~

| 前置章节 | 本章应用 |
| ------ | --------- |
| {doc}`../neural-network-basics/cnn-basics` | 卷积/池化 → 编码器的构建块 |
| {doc}`../neural-network-basics/le-net` | 编码器-分类器 → 编码器-解码器 |
| {doc}`../attention-mechanisms/index` | 注意力 → Attention U-Net |
```

```{toctree}
:maxdepth: 2
:hidden:

introduction
u-net
loss-func
core-impl
practice
the-end
```
