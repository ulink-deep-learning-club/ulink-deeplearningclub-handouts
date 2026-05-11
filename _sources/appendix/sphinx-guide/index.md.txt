(sphinx-guide)=
# Sphinx Markdown 语法参考

本附录收录了在这本书中常见的 Sphinx Markdown 语法，面向需要撰写教程内容的贡献者，而非学习者。如果你只想学习深度学习，可以跳过。

```{note}
基本所有示例代码存放在 `examples/` 子目录中，通过 `literalinclude` 引用。这样做可以避免 MyST 指令在代码块内被误解析。
```

## 嵌套指令

Markdown 不支持在代码块内再嵌套代码块，Sphinx 用 `~~~`（波浪线）作为内层分隔符：

```{literalinclude} examples/nested-directive.md
:language: markdown
```

换一种分隔符：`` ``` `` → `~~~`。

## 图表

### Mermaid（流程图）

```{literalinclude} examples/mermaid.md
:language: markdown
```

```{tip}
Mermaid 适合用于展示简单的图表，如流程图、序列图等（或者你不想调试 TikZ 代码）。

Mermaid 图表语法参考：[https://mermaid.js.org/](https://mermaid.js.org/)。

正确渲染需要 Chrome 系浏览器 （如 Chrome/Chromium、Edge 等）和 mermaid-cli (mmdc) 工具，安装方法如下：

~~~bash
npm install -g @mermaid-js/mermaid-cli  # Linux 可能需要 sudo
~~~
```

### TikZ（复杂图表）

```{literalinclude} examples/tikz.md
:language: markdown
```

```{tip}
TikZ 适合用于展示复杂的图表，如网络架构等。

需要 TeXLive + pdf2svg 才能渲染。缺少依赖时文档仍可编译，只是图表不显示。
```

## 图片

```{literalinclude} examples/figure.md
:language: markdown
```

路径相对于 Markdown 文件所在位置。

```{tip}

不推荐设定 width 为 100%，因为这会导致图片在不同设备上显示不一致。
图片宽度建议设置为 400-600 px，根据需要调整。
```

## 代码

大段代码推荐放在单独文件中，用 `literalinclude` 引用：

`````markdown
```{literalinclude} /path/to/code.py
:language: python
:linenos:
:caption: 代码标题
```
`````

这是本指南本身使用的技巧——用 `literalinclude` 展示代码示例，避免 MyST 解析冲突。

`linenos` 启用行号。

## 数学公式

行内公式：

```{literalinclude} examples/math.md
:language: markdown
:lines: 1
```

行间公式：

```{literalinclude} examples/math.md
:language: markdown
:lines: 3-11
```

## 彩色提示框

```{literalinclude} examples/admonition.md
:language: markdown
```

可用 class：`attention`、`caution`、`danger`、`error`、`hint`、`important`、`note`、`tip`、`warning`。

## 目录树

在 admonition 内嵌套 toctree（用 `~~~` 分隔）：

```{literalinclude} examples/toctree.md
:language: markdown
```

每行是 `.md` 文件路径（不含 `.md` 扩展名）。

## Cross-Reference

### `{ref}` — 引用章节/标签

**定义标签**（放在标题上方）：

```{literalinclude} examples/label.md
:language: markdown
```

**引用标签**：

```{literalinclude} examples/ref.md
:language: markdown
```

### `{doc}` — 引用整个文档

```{literalinclude} examples/doc-ref.md
:language: markdown
```

不含 `.md` 扩展名。

### `{cite}` — 引用文献

```{literalinclude} examples/cite-ref.md
:language: markdown
```

### 最佳实践

1. 标签用描述性名称：`(gradient-vanishing)=` 而非 `(section-5)=`
2. 文件顶部定义标签：`(label)=` 紧邻 `# 标题` 之上
3. 用 `{ref}` 引用章节/概念，用 `{doc}` 引用整篇文档
4. 重命名文件后更新所有 cross-reference

## 文献管理（BibTeX）

所有引用存放在 `source/references.bib` 中，格式为 BibTeX。

**添加一条引用**：在 `references.bib` 末尾追加，key 命名规则为 `作者姓氏年份关键词`：

```bibtex
@inproceedings{vaswani2017attention,
  title={Attention is all you need},
  author={Vaswani, Ashish and ...},
  booktitle={Advances in Neural Information Processing Systems},
  volume={30},
  year={2017}
}
```

**在正文中引用**：

```markdown
{cite}`vaswani2017attention`
```

**在文件末尾渲染参考文献列表**（只显示该文件实际引用的条目，仅引用了 `references.bib` 中的条目时需要添加）：

``````markdown
## 参考文献

```{bibliography}
:filter: docname in docnames
```
``````

```{note}
`:filter: docname in docnames` 确保只列出本文档中通过 `{cite}` 实际引用过的条目，而非整个 `references.bib`。
```

## 表格

使用 `{list-table}` 指令，支持多级表头：

`````markdown
```{list-table} 表格标题
:header-rows: 1

* - 列一
  - 列二
  - 列三
* - 数据1
  - 数据2
  - 数据3
```
`````

第一行（`* -`）为表头，后续行为数据。仅在 Markdown 表格不满足需求时使用（如需要 directive 嵌入或更复杂的格式）。

## 主目录树

每个章节的 `index.md` 头部应有隐藏的 `{toctree}` 定义子文档顺序：

`````markdown
```{toctree}
:maxdepth: 2
:hidden:

introduction
cnn-basics
le-net
...
```
`````

而在主 `source/index.md` 中定义全书的顶层目录树。条目为文件路径（不含 `.md` 扩展名）。

## 脚注

使用标准 Markdown 脚注语法：

```markdown
一些文字[^note-key]。

[^note-key]: 这是脚注内容。
```

脚注适合补充细节或提供扩展阅读链接，避免正文冗长。

## 条件渲染

条件渲染使得一部分内容仅在满足条件时渲染。

``````markdown
```{only} html
只在 HTML 输出中渲染的内容。
```
``````

``````markdown
```{only} html or epub
只在 HTML 或 EPUB 输出中渲染的内容。
```
``````

或者

``````markdown
```{only} not pdf
只在非 PDF 输出中渲染的内容。
```
``````

```{admonition} 注意
:class: warn

为了兼容 Sphinx 的 LaTeX 输出，index 除了隐藏的 ToC，其他内容都必须通过此方法取消渲染。
其余页面需要取消参考文献列表的渲染。
```
