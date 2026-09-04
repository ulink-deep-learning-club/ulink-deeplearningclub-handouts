# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import sys
import os
from pathlib import Path
import logging
import platform
import shutil
import posixpath
import json
from urllib.parse import urlsplit

from bs4 import BeautifulSoup
from docutils import nodes
from markupsafe import Markup

def list_file_with_extension(directory: Path, extension: str) -> list[Path]:
    """
    Lists files in a directory that match the given extension.
    """
    if not os.path.isdir(directory):
        return []

    matching_files = []
    for item in os.listdir(directory):
        if os.path.isfile(directory / item) and item.endswith(extension):
            matching_files.append(directory / item)
        elif os.path.isdir(directory / item):
            matching_files += list_file_with_extension(directory / item, extension)

    return matching_files

def get_potential_paths():
    """
    Returns a list of potential browser paths based on the operating system.
    """
    system = platform.system()

    if system == "Windows":
        # Windows paths (System and User installs)
        return [
            # Chrome
            r"C:\Program Files\Google\Chrome\Application\chrome.exe",
            r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
            os.path.expandvars(r"%LOCALAPPDATA%\Google\Chrome\Application\chrome.exe"),
            # Edge
            r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
            r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
            # Chromium
            os.path.expandvars(r"%LOCALAPPDATA%\Chromium\Application\chrome.exe"),
        ]

    elif system == "Darwin":  # macOS
        return [
            "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
            "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
            "/Applications/Chromium.app/Contents/MacOS/Chromium",
            "/Applications/Brave Browser.app/Contents/MacOS/Brave Browser"
        ]

    elif system == "Linux":
        # On Linux, we primarily look for binaries in the global PATH
        return [
            "google-chrome",
            "google-chrome-stable",
            "microsoft-edge",
            "microsoft-edge-stable",
            "chromium",
            "chromium-browser",
            "/usr/bin/google-chrome" # Fallback explicit path
        ]

    return []

def find_browser_executable():
    """
    Iterates through potential paths and checks if they exist/are executable.
    """
    potential_paths = get_potential_paths()

    print(f"[*] Detecting OS: {platform.system()}")
    print("[*] Searching for Chromium-based browsers...")

    for path in potential_paths:
        # Check if it is a full path (Windows/Mac)
        if os.path.isabs(path):
            if os.path.exists(path) and os.access(path, os.X_OK):
                return path
        # Check if it is a command name (Linux primarily)
        else:
            cmd_path = shutil.which(path)
            if cmd_path:
                return cmd_path

    return None


logger = logging.getLogger(__name__)

chrome_path = find_browser_executable()
if not chrome_path:
    logger.warning("Chrome not found, Mermaid diagram rendering will get switched to raw mode.")
else:
    print("Using Chrome at: ", chrome_path)
    os.environ["PUPPETEER_EXECUTABLE_PATH"] = str(chrome_path)

sys.path.append(str(Path('../exts').resolve()))

project = 'Deep Learning Club Handouts'
copyright = '2025, UCS Deep Learning Club and its contributors, licensed under CC BY-SA 4.0'
author = 'UCS Deep Learning Club'
release = '0.0.1'
html_title = "Deep Learning Club Handouts"

# English pages become public chapter by chapter. The reviewed prefix list is
# maintained outside the source tree so the publishing script and templates use
# the same release decision.
I18N_RELEASE_MANIFEST = Path(__file__).resolve().parents[1] / "docs" / "i18n" / "release-manifest.json"
SITE_BASEURL = "https://ulink-deep-learning-club.github.io/ulink-deeplearningclub-handouts/"

# conf.py

latex_engine = "lualatex"

latex_elements = {
    'passoptionstopackages': r'''
\PassOptionsToPackage{svgnames}{xcolor}
\PassOptionsToPackage{nocheck}{fancyhdr}
''',
    'sphinxsetup': r'HeaderFamily=\rmfamily\bfseries',
    'fontpkg': r'''
\usepackage{fontspec}
\usepackage[UTF8, fontset=none]{ctex}
\usepackage{times}

% 声明 HarfBuzz 渲染器
\defaultfontfeatures{Renderer=HarfBuzz}

% Emoji Fallback 设置
\directlua{
  luaotfload.add_fallback("emojifallback", {
    "NotoEmoji-Regular:mode=harf;",
    "NotoSansCJKsc-Regular:mode=harf;"
  })
}

% 设置英文字体 (必须紧跟着 ctex 声明)
\setmainfont{Noto Serif CJK SC}[Scale=MatchLowercase, RawFeature={fallback=emojifallback}]
\setsansfont{Noto Sans CJK SC}[Scale=MatchLowercase, RawFeature={fallback=emojifallback}]
\setmonofont{Noto Sans Mono}[Scale=MatchLowercase, RawFeature={fallback=emojifallback}]

% 设置中文字体
\setCJKmainfont{Noto Serif CJK SC}[AutoFakeSlant=true]
\setCJKsansfont{Noto Sans CJK SC}[AutoFakeSlant=true]
\setCJKmonofont{Noto Sans Mono CJK SC}[AutoFakeSlant=true]
''',
    'preamble': r'''
\renewcommand{\familydefault}{\rmdefault}
\pdfstringdefDisableCommands{%
  \def\times{×}%
}
''',
}
latex_show_urls = 'footnote'


html_css_files = [
    'style-fixes.css',
    'i18n.css',
]

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'myst_parser',
    'sphinx_design',
    'sphinxcontrib.tikz',
    'tikz_fallback',
    'sphinxcontrib.mermaid',
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.githubpages',
    'fix_mermaid_svgs',
    'sphinxcontrib.bibtex'
]
myst_heading_anchors = 3


in_html = False
if 'html' in sys.argv:
    print("MathJax is enabled for html product")
    extensions.append('sphinx.ext.mathjax')
    extensions.append('contributors')
    in_html = True
elif 'epub' in sys.argv:
    print("math-svg conversion is enabled for epub product")
    extensions.append('math2svg')

if not in_html:
    mermaid_output_format = 'svg'
    mermaid_params = [
        '--theme', 'neutral',
        '--backgroundColor', 'transparent',
        '--scale', '1'             # 1 = actual size, 2 = 2x, etc.
    ]
    mermaid_pdfcrop = 'pdfcrop'
    mermaid_cmd = "mmdc -f"
else:
    # MathJax configuration
    mathjax3_config = {
        'tex': {
            'inlineMath': [['$', '$'], ['\\(', '\\)']],
            'displayMath': [['$$', '$$'], ['\\[', '\\]']],
        },
        'loader': {
          'load': ['input/tex', 'output/svg']
        }
    }

    mermaid_init_config = '''
    {
      "theme": "neutral"
    }
    '''
    mermaid_height = "360px"

    mermaid_d3_zoom = True

exclude_patterns = [
    '_build', 'build', 'Thumbs.db', '.DS_Store', '.cache',
    # These are literalinclude fixtures rendered by the guide, not standalone
    # documentation pages. Parsing them independently intentionally produces
    # broken-example warnings.
    'appendix/sphinx-guide/examples/**',
]

# TikZ configuration
tikz_proc_suite = 'pdf2svg'
tikz_resolution = 92
tikz_latex_preamble = r'''
\usepackage[UTF8, fontset=none]{ctex}
\usepackage{fontspec}
\usetikzlibrary{shapes,arrows,arrows.meta,positioning,shapes.geometric,calc,decorations.pathreplacing,trees,backgrounds,fit}
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{xcolor}

\setCJKmainfont{Noto Serif CJK SC}
\setCJKsansfont{Noto Sans CJK SC}
\setCJKmonofont{Noto Sans Mono CJK SC}
\renewcommand{\familydefault}{\sfdefault}
'''

# MyST configuration
myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "deflist",
    "dollarmath",
    "fieldlist",
    "html_admonition",
    "html_image",
    "replacements",
    "smartquotes",
    "strikethrough",
    "substitution",
    "tasklist",
]

# Internationalization
language = 'zh_CN'
locale_dirs = ['locale/']
gettext_compact = False
gettext_additional_targets = {'literal-block'}
figure_language_filename = '{root}.{language}{ext}'

templates_path = ['_templates']

# Existing chapters deliberately repeat bibliography entries on one page. Keep
# strict warnings for unresolved references and malformed source while omitting
# this known, non-rendering bibliography noise.
suppress_warnings = ['bibtex.duplicate_citation', 'bibtex.duplicate_label']

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'breeze'
html_static_path = ['_static']

# Sphinx Book Theme configuration
html_context = {
    'github_user': 'ulink-deep-learning-club',
    'github_repo': 'ulink-deeplearningclub-handouts',
    'github_version': 'main',
    "doc_path": "source",
}

html_theme_options = {
    "header_start": ["header-brand.html"],
    "header_end": ["search-button.html", "language-switcher.html", "theme-switcher.html", "external-links.html"],
    "sidebar_primary": ["sidebar-nav.html"],
    "sidebar_secondary": ["repo-stats.html", "edit-this-page.html", "sidebar-toc.html"],
    "article_header": ["breadcrumbs.html"],
    "article_footer": ["related-pages.html"],
    "footer": ["footer-copyright.html", "external-links.html"],
    "header_tabs": False,
    "page_actions": False,
    "external_links": [
        "https://github.com/ulink-deep-learning-club/ulink-deeplearningclub-handouts"
    ],
    "default_mode": "auto",
}




imgmath_latex = 'lualatex'
imgmath_image_format = 'svg'
imgmath_latex_preamble = r'''
\usepackage{unicode-math}
\usepackage[UTF8, fontset=none]{ctex}
\setCJKmainfont{Noto Serif CJK SC}
'''

bibtex_bibfiles = ["references.bib"]


def _page_path(language_code: str, pagename: str) -> str:
    """Return the published HTML path for a Sphinx page in one language."""
    prefix = "en/" if language_code == "en" else ""
    return f"{prefix}{pagename}.html"


def _released_english_prefixes() -> tuple[str, ...]:
    try:
        manifest = json.loads(I18N_RELEASE_MANIFEST.read_text(encoding="utf-8"))
        prefixes = manifest["released_prefixes"]
        if not all(isinstance(prefix, str) and prefix for prefix in prefixes):
            raise ValueError("released_prefixes must contain non-empty strings")
        return tuple(prefixes)
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        logger.warning("Invalid English release manifest: %s", error)
        return ()


def _english_page_is_released(pagename: str) -> bool:
    return any(pagename == prefix or pagename.startswith(f"{prefix}/")
               for prefix in _released_english_prefixes())


ENGLISH_NAVIGATION_TITLES = {
    'cnn-expedition/index': 'CNN Expedition: Thirty Years of Architectures',
    'pytorch-practice/index': 'PyTorch Practice: Turning Theory into Code',
    'model-serving/index': 'Model Deployment and Serving',
    'transfer-learning/index': 'Transfer Learning and Fine-Tuning',
    'unet-image-segmentation/index': 'U-Net Image Segmentation',
    'sequence-modeling/index': 'Sequence Modeling: RNN to Transformer to Mamba',
    'appendix/index': 'Appendix',
    'postscript': 'Postscript',
}


def _configure_language(app, config):
    """Apply options that depend on the final -D language override."""
    if config.language == "en":
        config.html_title = "Deep Learning Club Handouts"
        config.html_baseurl = f"{SITE_BASEURL}en/"
        config.latex_documents = [
            ('index', 'deep-learning-club-handouts.tex', 'Deep Learning Club Handouts',
             'UCS Deep Learning Club', 'manual'),
        ]

    else:
        config.html_title = "Deep Learning Club 学习教程"
        config.html_baseurl = SITE_BASEURL
        config.latex_documents = [
            ('index', 'deeplearningclubhandouts.tex', 'Deep Learning Club 学习教程',
             'UCS Deep Learning Club', 'manual'),
        ]

    # TikZ node labels are not gettext messages.  Keep both labels in the
    # canonical source and select the rendered one here so diagrams do not
    # remain in Chinese in the English edition.
    localized_label = r'\newcommand{\localizedlabel}[2]{#2}' if config.language == "en" else r'\newcommand{\localizedlabel}[2]{#1}'
    config.tikz_latex_preamble += f"\n{localized_label}\n"

    # MyST does not extract Markdown tables nested in directive bodies into
    # gettext catalogs. These substitutions keep such structural content in
    # the canonical Chinese source while rendering it in the active language.
    backprop_table_labels = {
        "zh_CN": {
            "bp_table_concept": "概念",
            "bp_table_notation": "数学表示",
            "bp_table_input_dimension": "输入维度",
            "bp_table_output_dimension": "输出维度",
            "bp_table_matrix_shape": "矩阵形状",
            "bp_table_scalar_derivative": "**标量导数**",
            "bp_table_gradient": "**梯度**",
            "bp_table_jacobian": "**Jacobian 矩阵**",
            "bp_table_row_vector": "行向量",
            "bp_table_matrix": "**M×N 矩阵**",
            "bp_computational_efficiency": "1. 计算效率",
            "bp_modular_design": "2. 模块化设计",
        },
        "en": {
            "bp_table_concept": "Concept",
            "bp_table_notation": "Mathematical notation",
            "bp_table_input_dimension": "Input dimension",
            "bp_table_output_dimension": "Output dimension",
            "bp_table_matrix_shape": "Matrix shape",
            "bp_table_scalar_derivative": "**Scalar derivative**",
            "bp_table_gradient": "**Gradient**",
            "bp_table_jacobian": "**Jacobian matrix**",
            "bp_table_row_vector": "row vector",
            "bp_table_matrix": "**M×N matrix**",
            "bp_computational_efficiency": "1. Computational Efficiency",
            "bp_modular_design": "2. Modular Design",
        },
    }
    config.myst_substitutions.update(backprop_table_labels["en" if config.language == "en" else "zh_CN"])


def _add_language_switch_context(app, pagename, templatename, context, doctree):
    """Expose same-page language menu data to the header template."""
    language_code = app.config.language
    target_language = 'zh_CN' if language_code == 'en' else 'en'
    current_path = _page_path(language_code, pagename)
    target_path = _page_path(target_language, pagename)
    context['language_switcher_enabled'] = (
        language_code == 'en' or _english_page_is_released(pagename)
    )
    context['language_switch_label'] = '中文' if language_code == 'en' else 'English'
    context['language_switch_url'] = posixpath.relpath(
        target_path, start=posixpath.dirname(current_path)
    )
    context['language_code'] = language_code
    context['language_current_label'] = 'English' if language_code == 'en' else '中文'
    context['language_current_lang'] = 'en' if language_code == 'en' else 'zh-CN'
    context['language_switch_lang'] = 'zh-CN' if language_code == 'en' else 'en'
    # Keep the control name stable across editions. The choices below it still
    # use their native names (English and 中文).
    context['language_menu_label'] = 'Language'

    def localized_toctree(content: str | None) -> Markup:
        """Describe unreleased chapters without presenting them as English pages."""
        if language_code != 'en' or not content:
            return Markup(content or '')

        soup = BeautifulSoup(content, 'html.parser')
        current_dir = posixpath.dirname(pagename)
        for item in soup.select('li.toctree-l1'):
            link = item.find('a', href=True)
            if not link:
                continue
            target = urlsplit(link['href']).path
            target_docname = posixpath.normpath(
                posixpath.join(current_dir, target)
            ).removesuffix('.html')
            if _english_page_is_released(target_docname):
                continue

            english_title = ENGLISH_NAVIGATION_TITLES.get(target_docname)
            if not english_title:
                # Only top-level entries are transformed here. If a new one is
                # added without a label, hiding it is safer than leaking an
                # untranslated title into the English interface.
                item.decompose()
                continue

            link.clear()
            link.append(english_title)
            link['href'] = posixpath.relpath(
                _page_path('zh_CN', target_docname),
                start=posixpath.dirname(_page_path('en', pagename)),
            )
            link['hreflang'] = 'zh-CN'
            link['title'] = 'English translation unavailable; open the Chinese edition'
            link['aria-label'] = f'{english_title}. English translation unavailable; open the Chinese edition'
            item['class'] = [*item.get('class', []), 'i18n-unavailable']
            # Do not expose untranslated child-page titles below this English
            # catalog entry. The top-level label still tells readers what the
            # course contains and offers the Chinese edition intentionally.
            for child in list(item.find_all(['details', 'ul'], recursive=False)):
                child.decompose()
            badge = soup.new_tag('span', attrs={
                'class': 'i18n-unavailable-badge',
                'aria-hidden': 'true',
            })
            badge.string = 'Chinese only'
            link.append(badge)

        # Sphinx omits the current top-level leaf from the generated toctree.
        # Without this, the preface vanishes precisely while it is being read.
        if pagename == 'preface':
            root = soup.find('ul')
            if root and not root.select_one('li > a[href="#"]'):
                item = soup.new_tag('li', attrs={'class': 'toctree-l1 current'})
                link = soup.new_tag('a', attrs={
                    'class': 'current reference internal',
                    'href': '#',
                    'aria-current': 'page',
                })
                link.string = 'Preface: About Deep Learning'
                item.append(link)
                root.insert(0, item)
        return Markup(str(soup))

    context['localized_toctree'] = localized_toctree


UNRELEASED_REFERENCE_LABELS = {
    'cnn-expedition/image-net-era/res-net': 'ResNet (Chinese)',
    'unet-image-segmentation/u-net': 'U-Net (Chinese)',
    'sequence-modeling/rnn-basics': 'RNN Basics (Chinese)',
    'cnn-expedition/practice-peak/neural-training-basics': 'Neural Training Basics (Chinese)',
}


def _localize_unreleased_references(app, doctree, docname):
    """Keep English pages from displaying untranslated cross-reference titles."""
    if app.config.language != 'en' or docname != 'math-fundamentals/back-propagation':
        return

    for reference in doctree.findall(nodes.reference):
        # BibTeX citations can be linked to anchors in another document; they
        # are not document cross-references and must keep their citation key.
        if reference.get('refdomain') == 'cite' or reference.get('reftitle'):
            continue
        target_docname = reference.get('refdocname')
        if not target_docname and reference.get('refuri'):
            target_path = urlsplit(reference['refuri']).path
            target_docname = posixpath.normpath(
                posixpath.join(posixpath.dirname(docname), target_path)
            ).removesuffix('.html')
        label = UNRELEASED_REFERENCE_LABELS.get(target_docname)
        if not label or _english_page_is_released(target_docname):
            continue
        reference.clear()
        reference += nodes.Text(label)
        target_path = _page_path('zh_CN', target_docname)
        current_path = _page_path('en', docname)
        reference['refuri'] = posixpath.relpath(
            target_path, start=posixpath.dirname(current_path)
        )


def setup(app):
    app.connect('config-inited', _configure_language)
    app.connect('html-page-context', _add_language_switch_context)
    app.connect('doctree-resolved', _localize_unreleased_references)
