# Deep Learning Club 学习教程

**Ulink 深度学习社**讲义。面向具备微积分基础的高中生，从理论到实践的深度学习课程。

> **配套参考**：建议与斋藤康毅《深度学习入门：基于Python的理论与实现》（"鱼书"）配合阅读。鱼书教你从零手写神经网络，本材料教你用PyTorch理解原理并动手做项目——两者互补，效果最佳。详见在线文档的[前言](https://ulink-deep-learning-club.github.io/ulink-deeplearningclub-handouts/preface.html)。

- 在线文档：[ulink-deep-learning-club.github.io/ulink-deeplearningclub-handouts](https://ulink-deep-learning-club.github.io/ulink-deeplearningclub-handouts/)
- PDF 版本：[deeplearningclubhandouts.pdf](https://github.com/ulink-deep-learning-club/ulink-deeplearningclub-handouts/blob/gh-pages/deeplearningclubhandouts.pdf)

## 双语版本

中文 Markdown 是唯一的内容源；英语版本由 Sphinx gettext 翻译目录生成。英语内容通过完整性审查后，在线站点会在每页页头提供语言切换，并在 `/en/` 下发布与中文完全对应的页面；英文 PDF 位于 `/en/deep-learning-club-handouts.pdf`。现有中文链接及 PDF 文件名保持不变。

翻译贡献者请遵循 [本地化工作指南](docs/i18n/AGENT_GUIDE.md) 和 [术语表](docs/i18n/glossary.md)。完成并审校的英语章节会逐章发布到 `/en/`；未完成页面会安全地回退到对应中文页面。全书完成后才发布英文 PDF。

## 项目结构

```
lecture-material/
├── source/                       # 课程文档（Markdown + MyST）
│   ├── index.md                  # 课程主页
│   ├── preface.md                # 前言
│   ├── postscript.md             # 后记
│   ├── references.bib            # 全局参考文献
│   ├── math-fundamentals/        # 数学基础（计算图、反向传播、梯度下降）
│   ├── cnn-expedition/           # CNN 架构演化（全书最长的章节）
│   │   ├── practice-peak/        #   上篇：练习峰（全连接→CNN→LeNet）
│   │   ├── ablation-study/       #   消融研究（验证每个组件的贡献）
│   │   ├── image-net-era/        #   下篇：未登峰（AlexNet→Inception→ResNet→MobileNet→EfficientNet）
│   │   └── model-architecture-design/  # 架构心法（感受野、深度连接、注意力、效率四维度）
│   ├── pytorch-practice/         # PyTorch 实践（训练流程、框架使用、最佳实践）
│   ├── model-serving/            # 模型部署（ONNX 导出、量化剪枝、服务架构、Ferrinx 部署）
│   ├── transfer-learning/        # 迁移学习（微调、域适应、小样本学习）
│   ├── sequence-modeling/        # 序列建模（RNN→LSTM→Transformer→Mamba）
│   ├── unet-image-segmentation/  # U-Net 图像分割
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

# 构建隔离的中英版本；中文发布根目录，英文发布到 /en/
bash scripts/build-i18n-site.sh

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
4. PR 合并后 GitHub Actions 自动构建部署到 gh-pages；翻译变更还必须更新对应 `.po` 目录并通过本地化检查

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
