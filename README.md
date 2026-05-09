# Deep Learning Club 学习教程

**Ulink 深度学习社**讲义。面向具备微积分基础的高中生，从理论到实践的深度学习课程。在线文档：[ulink-deep-learning-club.github.io/ulink-deeplearningclub-handouts](https://ulink-deep-learning-club.github.io/ulink-deeplearningclub-handouts/)

## 项目结构

```
lecture-material/
├── source/                       # 课程文档（Markdown + MyST）
│   ├── index.md                  # 课程主页
│   ├── preface.md                # 前言
│   ├── postscript.md             # 后记
│   ├── references.bib            # 全局参考文献
│   ├── math-fundamentals/        # 数学基础
│   ├── neural-network-basics/    # 神经网络基础
│   ├── pytorch-practice/         # PyTorch 实践
│   ├── sequence-modeling/        # 序列建模（RNN → Transformer → Mamba）
│   ├── cnn-ablation-study/       # CNN 消融研究
│   ├── transfer-learning/        # 迁移学习
│   ├── attention-mechanisms/     # CNN 注意力机制
│   ├── model-architecture-design/# CNN 架构设计心法
│   ├── unet-image-segmentation/  # U-Net 图像分割
│   ├── model-serving/            # 模型部署
│   └── appendix/                 # 附录（环境配置、Sphinx 写作指南）
├── legacy-doc/                   # 已归档的 LaTeX 文档
└── _static/                      # 静态资源（图片等）
```

## 本地构建

```bash
# 创建虚拟环境并安装依赖
uv venv
uv sync

# 构建 HTML
source .venv/bin/activate
make html

# 预览
cd build/html
python -m http.server 8000
# 浏览器访问 http://localhost:8000
```

> **图表渲染依赖**：TikZ 图表需要 TeXLive + pdf2svg，Mermaid 图表需要 `@mermaid-js/mermaid-cli`。缺少这些依赖时文档仍可编译，但对应图表不会渲染。
>
> Ubuntu 下安装：
> ```bash
> sudo apt install texlive latex-cjk-all texlive-pictures texlive-latex-extra pdf2svg texlive-xetex fonts-noto-cjk
> sudo npm install -g @mermaid-js/mermaid-cli
> ```

## 参与贡献

### 流程

1. **先开 Issue**：讨论你希望做的修改（内容增删、结构变更、错误修复等），确保方向一致后再动手
2. Fork 本仓库，创建功能分支（如 `feat/lstm-section`）
3. 修改完成后提交 PR 到 `main` 分支，描述改动内容和原因
4. PR 合并后 GitHub Actions 自动构建部署到 gh-pages

### 写作约定

- 新建章节请参照现有结构：`index.md`（章节概览 + toctree）→ 内容文件 → `the-end.md`（总结）
- **Cross-Reference**：每个关键概念至少一个 `{ref}` 或 `{doc}` 链接到相关章节，建立知识网络
- **文献管理**：引用统一追加到 `source/references.bib`，使用 `{cite}` 在正文引用，文件末尾用 `{bibliography}` 渲染
- **代码注释**：代码块中每行应有注释（维度说明、参数计算、cross-reference 到理论）
- **Label 规范**：文件顶部 `(label)=` 紧邻 `# 标题` 之上，子节用 `(subsection-label)=` 在 `##` 之上
- 更多细节见附录 `source/appendix/sphinx-guide/index.md`

### 内容风格

- **直觉优先**：每个概念先建立直觉理解（生活类比、几何直觉），再深入数学
- **对比驱动**：使用表格对比不同方法（如全连接 vs CNN、RNN vs LSTM vs Transformer）
- **数据说话**：用具体数字支撑观点（参数量、时间、准确率）
- **适度诚实**：当数学深度超出目标读者范围时，坦诚说明，指出核心直觉即可

## 技术栈

- **构建工具**：Sphinx + MyST Markdown
- **数学渲染**：MathJax（LaTeX）
- **图表**：TikZ（复杂图）+ Mermaid（流程图）
- **引用管理**：sphinxcontrib-bibtex

## 许可证

CC BY-SA 4.0
