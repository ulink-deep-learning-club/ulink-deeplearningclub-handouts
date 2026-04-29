(appendix-env)=
# 环境配置番外篇

前面章节的所有内容都可以在笔记本或Google Colab上完成——有PyTorch、有Jupyter、有GPU，就够了。但当你开始认真做深度学习项目，迟早会遇到一个绕不开的问题：**我需要一台自己的GPU服务器**。

这篇附录不讲神经网络，不讲训练技巧，只讲"搞一台GPU服务器并让它稳定跑起来"需要的那些非技术知识。它们不属于深度学习本身，但没有它们，你的深度学习代码就跑不起来。

## 内容概览

| 章节 | 解决什么问题 | 适合谁 |
|------|-------------|--------|
| {doc}`linux-basics` | Linux比Windows好在哪？装完系统该干什么？怎么远程连接？ | 第一次接触Linux服务器的读者 |
| {doc}`nvidia-setup` | 驱动装了没？为什么nvidia-smi报错？CUDA版本怎么选？ | 有GPU但不知道怎么配置的读者 |
| {doc}`remote-access` | 服务器在机房怎么访问？没有公网IP怎么办？ | 需要远程管理服务器的读者 |
| {doc}`server-management` | 训练跑到一半终端关了怎么办？服务怎么开机自启？ | 部署了服务需要维护的读者 |

```{admonition} 阅读建议
:class: note

这篇附录是**按需查阅**的参考材料，不是线性章节。遇到具体问题再回来翻对应部分即可。如果你是第一次接触Linux服务器，建议从{doc}`linux-basics`开始顺序阅读。
```

## 学习路径

```{mermaid}
graph TD
    A[拿到一台GPU服务器] --> B{操作系统?}
    B -->|Windows| C[考虑装Linux双系统<br/>或使用WSL2]
    B -->|已有Linux| D[配置环境]
    D --> E{需要什么?}
    E --> F[NVIDIA驱动+CUDA]
    E --> G[远程访问]
    E --> H[服务管理]
    F --> I[可以训练了]
    G --> I
    H --> I
```

## 前置知识

- 能区分"操作系统"和"软件"的区别
- 知道什么是终端/命令行

没有更多前置要求了——这篇附录就是从零开始的。

```{toctree}
:hidden:
:maxdepth: 2

linux-basics
nvidia-setup
remote-access
server-management
the-end
```
