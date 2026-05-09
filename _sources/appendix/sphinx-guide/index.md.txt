(sphinx-guide)=
# Sphinx Markdown 写作指南

本附录面向需要撰写教程内容的贡献者，而非学习者。如果你只想学习深度学习，可以跳过。

```{note}
所有示例代码存放在 `examples/` 子目录中，通过 `literalinclude` 引用。这样做可以避免 MyST 指令在代码块内被误解析。
```

## 嵌套指令

Markdown 不支持在代码块内再嵌套代码块，Sphinx 用 `~~~`（波浪线）作为内层分隔符：

```{literalinclude} examples/nested-directive.md
:language: markdown
```

每多一层嵌套，换一种分隔符：`` ``` `` → `~~~`。

## 图表

### Mermaid（流程图）

```{literalinclude} examples/mermaid.md
:language: markdown
```

### TikZ（复杂图表）

```{literalinclude} examples/tikz.md
:language: markdown
```

需要 TeXLive + pdf2svg 才能渲染。缺少依赖时文档仍可编译，只是图表不显示。

## 图片

```{literalinclude} examples/figure.md
:language: markdown
```

路径相对于 Markdown 文件所在位置。

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

```{literalinclude} examples/math.md
:language: markdown
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

**在文件末尾渲染参考文献列表**（只显示该文件实际引用的条目）：

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

第一行（`* -`）为表头，后续行为数据。对 Markdown 表格不满足需求时使用（如需要 directive 嵌入或更复杂的格式）。

## 主目录树

每个章节的 `index.md` 末尾应有 `{toctree}` 定义子文档顺序：

`````markdown
```{toctree}
:maxdepth: 2

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
