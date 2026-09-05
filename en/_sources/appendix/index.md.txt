# 附录

```{only} html
以下是深度学习正文之外的补充材料——不讲模型，只讲"让这一切跑起来"的周边技能。

~~~{rubric} 附录概览
:heading-level: 2
~~~

| 章节 | 内容 | 适合谁 |
| -------- | ---------- | ---------- |
| {doc}`environment-setup/index` | Linux 基础、NVIDIA 驱动、CUDA 配置、远程访问、服务器管理 | 需要自己搭建 GPU 训练环境的读者 |
| {doc}`sphinx-guide/index` | Sphinx Markdown 语法、图表、交叉引用、文献管理 | 需要为本书贡献内容的写作者 |
| {doc}`build-docs` | 编译文档（macOS / Linux） | 需要在本地构建 HTML/PDF 的贡献者 |

~~~{rubric} 阅读建议
:heading-level: 2
~~~

~~~{admonition} 按需查阅
:class: note

附录不是线性章节，遇到具体问题再翻对应部分即可：
- 训练环境搞不定 → {doc}`environment-setup/index`
- 不知道怎么写文档 → {doc}`sphinx-guide/index`
- 不知道怎么编译 → {doc}`build-docs`
~~~

~~~{rubric} 前置知识
:heading-level: 2
~~~

- **环境配置**：能区分操作系统和软件，知道什么是终端/命令行
- **Sphinx 指南**：会用 Markdown 即可

不涉及深度学习知识——附录就是为降低工程摩擦而存在的。
```

```{toctree}
:hidden:
:maxdepth: 2

environment-setup/index
sphinx-guide/index
build-docs
```