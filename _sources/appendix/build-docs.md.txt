(appendix-build-docs)=

# 编译文档

本书使用 Sphinx + MyST Markdown 构建，输出 HTML 网页、PDF 和 EPUB 三个版本。本附录说明如何在本地搭建编译环境并生成文档。

如果你是在为本书贡献内容，先阅读 {doc}`sphinx-guide/index` 了解语法规范。

## 为什么需要"编译"？

如果你用过 Word 或 Google Docs，你对"所见即所得"（WYSIWYG）很熟悉——你在编辑界面里看到的格式，**就是**最终打印出来的样子。文件直接打开就是成品。

但本书用的是另一种模式：**写代码，再编译**。你编辑的 Markdown 文件只是"蓝图"或"底片"，必须经过构建工具链的处理，才能变成可阅读的最终产物。

| | Word 模式 | Sphinx 模式 |
| --- | --- | --- |
| **编辑对象** | `.docx` 文件 | `.md` 文件（纯文本） |
| **所见即所得** | 是，格式直接可见 | 否，纯文本 + 标记语法 |
| **交叉引用** | 手动粘贴 | 编译时自动解析 |
| **数学公式** | 公式编辑器 | LaTeX 源码，编译后渲染 |
| **多格式输出** | 另存为 PDF（可能乱码） | 一次编译，输出 HTML/PDF/EPUB |
| **版本控制** | 二进制文件，难 diff | 纯文本，Git 友好 |

~~~{mermaid}
graph LR
    subgraph Word["Word 模式"]
        A["docx 文件"] --> B["直接打开 = 成品"]
    end

    subgraph Sphinx["Sphinx 模式"]
        C["Markdown 源文件<br/>纯文本 + MyST 语法"] --> D["Sphinx 编译引擎"]
        D --> E["HTML 网页"]
        D --> F["PDF 电子书"]
        D --> G["EPUB 电子书"]
    end
~~~

### 为什么中间要多一步编译？

**因为你写的不是文档，而是文档的"生成指令"**。构建过程中，Sphinx 帮你做了这些繁重工作：

1. **解析交叉引用** — `{ref}`、`{doc}` 等标签在编译时自动转成链接和页码
2. **渲染数学公式** — `$E = mc^2$` 在 HTML 中变成 MathJax，在 PDF 中变成 LaTeX
3. **渲染图表** — Mermaid 流程图需要 Chromium 转成 SVG，TikZ 需要 TeX 编译
4. **统一排版** — 同一套内容，HTML 用网页主题，PDF 用 LaTeX 排版，EPUB 用电子书规范

没有编译这一步，你手里只有一堆 Markdown 源码，看不到流程图、看不到公式、看不到链接——就像只有底片，没有照片。

## 系统依赖

编译需要以下工具链：

- **Python 3.11+** — 运行 Sphinx
- **TeXLive** — PDF 输出（latex + lualatex）
- **Chromium** — Mermaid 图表渲染（通过 Puppeteer 调用）
- **Mermaid CLI** (`mmdc`) — 命令行渲染 Mermaid 图表
- **中文字体** — 中文 PDF 输出（Noto Sans CJK、Noto Emoji）
- **Roboto Flex 字体** — PDF 正文排版

```{admonition} 为什么需要 Chromium？
:class: note

本书的 Mermaid 流程图在 HTML 中由浏览器实时渲染，但 PDF 输出需要将 Mermaid 图表先转为 SVG——这个转换依赖 {doc}`sphinx-guide/index` 中提到的 `mmdc` 工具，而 `mmdc` 实际调用的是 Chromium 的无头模式（headless browser）。因此即使不在本地开发前端页面，构建 PDF 也需要安装 Chromium。
```

### macOS（Homebrew）

```bash
# 1. TeXLive（完整版约 4GB，时间较长）
brew install texlive

# 2. Chromium
brew install --cask chromium

# 3. 中文字体和英文字体
brew install font-noto-sans-cjk font-noto-emoji
# Roboto Flex（PDF 英文正文用）
brew install --cask font-roboto-flex

# 4. Node.js（Mermaid CLI 依赖）
brew install node

# 5. Mermaid CLI
npm install -g @mermaid-js/mermaid-cli

# 6. uv（Python 包管理器）
curl -LsSf https://astral.sh/uv/install.sh | sh
```

```{tip}
macOS 上 `brew install texlive` 安装的是 TeX Live small scheme，编译中文文档可能缺 `ctex` 等包。推荐安装完整版 MacTeX：

~~~bash
brew install --cask mactex  # 完整版，约 4GB
~~~

如果已经装了 basic 版，可通过 `tlmgr install <包名>` 补充缺失的 LaTeX 包。
```

### Debian / Ubuntu

```bash
# 1. 系统包（TeXLive + Chromium + 字体）
sudo apt update
sudo apt install texlive latex-cjk-all texlive-pictures \
    texlive-latex-extra texlive-extra-utils latexmk pdf2svg \
    texlive-xetex texlive-luatex fonts-noto-cjk fonts-noto-core \
    fonts-noto-extra chromium

# 2. Noto Emoji 字体（apt 包版本可能较旧，建议手动安装最新版）
mkdir -p ~/.local/share/fonts
wget "https://github.com/google/fonts/raw/refs/heads/main/ofl/notoemoji/NotoEmoji%5Bwght%5D.ttf" \
    -O ~/.local/share/fonts/NotoEmoji.ttf
fc-cache -f ~/.local/share/fonts

# 3. Roboto Flex 字体（PDF 英文正文用）
wget "https://github.com/google/fonts/raw/refs/heads/main/ofl/robotoflex/RobotoFlex%5BGRAD,XOPQ,XTRA,YOPQ,YTAS,YTDE,YTFI,YTLC,YTUC,opsz,slnt,wdth,wght%5D.ttf" \
    -O ~/.local/share/fonts/RobotoFlex.ttf
fc-cache -f ~/.local/share/fonts

# 4. Mermaid CLI（需要 Node.js）
sudo apt install nodejs npm
sudo npm install -g @mermaid-js/mermaid-cli

# 5. uv（Python 包管理器）
curl -LsSf https://astral.sh/uv/install.sh | sh
```

```{warning}
Debian/Ubuntu 自带的 `chromium` 包可能因为 Snap 冲突导致安装失败。如果遇到这个问题，参考 CI 脚本中的做法——先 `sudo apt purge snapd`，再从第三方仓库安装：

~~~bash
sudo apt purge snapd
sudo apt-mark hold snapd
sudo wget 'https://keyserver.ubuntu.com/pks/lookup?op=get&search=0x869689FE09306074' \
    -O '/etc/apt/trusted.gpg.d/phd-chromium.asc'
echo "deb https://freeshell.de/phd/chromium/$(lsb_release -sc) /" | \
    sudo tee /etc/apt/sources.list.d/phd-chromium.list
sudo apt update
sudo apt install chromium
~~~

详细的 CI 配置可参考 [`.github/workflows/build-docs.yml`](https://github.com/ulink-deep-learning-club/ulink-deeplearningclub-handouts/blob/main/lecture-material/.github/workflows/build-docs.yml)。
```

```{admonition} 磁盘空间提示
:class: warning

TeXLive 完整安装约需 4-5 GB 磁盘空间。如果空间有限，可以仅安装必要组件：

~~~bash
# Debian/Ubuntu 最小安装
sudo apt install texlive-latex-base texlive-latex-recommended \
    texlive-latex-extra texlive-luatex latexmk
~~~

但建议先完整安装一次——缺失包导致的编译错误往往比磁盘空间更耗费时间。
```

## Python 依赖

克隆仓库并安装 Python 包：

```bash
git clone <仓库地址>
cd lecture-material

# 使用 uv 创建虚拟环境并安装依赖
uv venv
uv sync

# 每次构建前激活环境
source .venv/bin/activate  # Linux/macOS
# 或
.venv\Scripts\activate     # Windows PowerShell
```

依赖清单见 `pyproject.toml`，主要包括：

| 包 | 用途 |
| --- | --- |
| `sphinx` | 文档构建引擎 |
| `myst-parser` | Markdown 解析（MyST 语法支持） |
| `sphinxcontrib-bibtex` | BibTeX 文献管理 |
| `sphinxcontrib-mermaid` | Mermaid 图表 |
| `sphinxcontrib-tikz` | TikZ 图表 |
| `sphinx-design` | 增强型 UI 组件 |
| `sphinx-breeze-theme` | Breeze 主题 |
| `sphinx-serve` | 本地预览服务器 |

## 构建文档

所有构建命令在项目根目录执行，且需要先激活虚拟环境。

### 构建 HTML

```bash
source .venv/bin/activate
make html
```

输出在 `build/html/`，直接用浏览器打开 `build/html/index.html` 即可预览。

### 构建 PDF

```bash
source .venv/bin/activate
make latexpdf
```

输出在 `build/latex/`，文件名为 `deeplearningclubhandouts.pdf`。

### 构建 EPUB

```bash
source .venv/bin/activate
make epub
```

输出在 `build/epub/`，文件名为 `DeepLearningClubHandouts.epub`。EPUB 格式适合在电子书阅读器或手机上阅读。

### 构建技巧

```{admonition} ToC 没有刷新？
:class: tip

如果修改了 `{toctree}` 或文件结构后页面导航没有更新，先删除已生成的 HTML 再重新构建：

~~~bash
rm -rf build/html
make html
~~~

只改内容不改结构时不需要这么做。
```

~~~{admonition} 保留 `_images` 目录加速重复构建
:class: tip

Mermaid 和 TikZ 图表的编译非常耗时（每次都要启动 Chromium 或 LaTeX）。反复构建时不要删除 `build/html/_images/`，Sphinx 会跳过已生成的图片。

同样地，构建 EPUB 前可以将 `build/html/_images/` 复制到 `build/epub/` 下，这样 EPUB 构建时也会跳过图片重生成：

```bash
cp -r build/html/_images build/epub/
```
~~~

```{admonition} EPUB 数学公式 SVG 缓存
:class: tip

EPUB 构建时会将数学公式编译为 SVG 图片，这一步同样耗时。如果之前构建过 EPUB，`build/epub/_images/` 中已经缓存了这些 SVG，保留它们可以显著加速后续构建。
```

### 本地预览

使用 `sphinx-serve` 启动本地 HTTP 服务器：

```bash
source .venv/bin/activate
uv run sphinx-serve -b build
```

`sphinx-serve` 默认使用 8000 端口，打开 `http://localhost:8000` 即可浏览。修改源文件后刷新页面即可看到更新。

```{tip}
`-b build` 指定编译输出目录。如果不在项目根目录运行，需要调整为 `uv run sphinx-serve -b /path/to/build`。
```

```{tip}
开发时建议只构建 HTML——速度比 PDF 快得多。PDF 在提交前或发布时构建一次即可。
```

### 完整构建流程（参考 CI）

以下是一个端到端的构建流程，与 GitHub Actions CI 中的步骤一致：

```bash
# 1. 激活环境
source .venv/bin/activate

# 2. 构建 HTML
make html

# 3. 构建 PDF（并移动到 HTML 目录方便一起发布）
make latexpdf
mv build/latex/deeplearningclubhandouts.pdf ./build/html/
```

完整的 CI 配置见 [`.github/workflows/build-docs.yml`](https://github.com/ulink-deep-learning-club/ulink-deeplearningclub-handouts/blob/main/lecture-material/.github/workflows/build-docs.yml)。

## 常见问题

### Mermaid 图表不渲染

检查 Chromium 是否安装且可执行：

```bash
# 查看 Sphinx 构建日志中有没有以下警告
# "Chrome not found, Mermaid diagram rendering will get switched to raw mode."

# 检查 Chromium 路径
which chromium          # Linux
ls /Applications/Chromium.app/Contents/MacOS/Chromium  # macOS
```

Sphinx 会自动检测系统中的 Chrome/Chromium/Edge，检测逻辑见 `source/conf.py`。

### PDF 编译报错（LaTeX 包缺失）

```bash
# Debian/Ubuntu：安装完整 TeXLive
sudo apt install texlive-latex-extra texlive-pictures texlive-luatex

# macOS：使用 tlmgr 补充单个包
tlmgr install <包名>
```

### 中文字体显示为方框

检查中文字体是否安装：

```bash
# 查看已安装的 CJK 字体
fc-list :lang=zh | head -5

# 如果为空，安装 Noto Sans CJK
# macOS
brew install font-noto-sans-cjk
# Debian/Ubuntu
sudo apt install fonts-noto-cjk
```

## 参考

- {doc}`sphinx-guide/index` — Sphinx Markdown 语法与写作规范
- [`.github/workflows/build-docs.yml`](https://github.com/ulink-deep-learning-club/ulink-deeplearningclub-handouts/blob/main/lecture-material/.github/workflows/build-docs.yml) — CI 构建配置
- [Sphinx 官方文档](https://www.sphinx-doc.org/)
- [MyST Parser 文档](https://myst-parser.readthedocs.io/)
