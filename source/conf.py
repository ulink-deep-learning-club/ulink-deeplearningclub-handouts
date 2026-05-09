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
\addtolength{\topmargin}{-1pt}
\renewcommand{\familydefault}{\rmdefault}

% 顺便解决你刚才提到的 \times 书签警告问题
\pdfstringdefDisableCommands{%
  \def\times{×}% 
}
''',
}
latex_show_urls = 'footnote'


html_css_files = [
    'style-fixes.css'
]

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'myst_parser',
    'sphinx_design',
    'sphinxcontrib.tikz',
    'sphinxcontrib.mermaid',
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.githubpages',
    'fix_mermaid_svgs',
    'sphinxcontrib.bibtex'
]
myst_heading_anchors = 3

if 'html' in sys.argv:
    print("MathJax is enabled for html product")
    extensions.append('sphinx.ext.mathjax')
    extensions.append('contributors')
elif 'epub' in sys.argv:
    print("math-svg conversion is enabled for epub product")
    extensions.append('math2svg')

mermaid_output_format = 'svg'
mermaid_params = [
    '--theme', 'neutral',
    '--backgroundColor', 'transparent',
    '--scale', '1'             # 1 = actual size, 2 = 2x, etc.
]
mermaid_pdfcrop = 'pdfcrop'
mermaid_cmd = "mmdc -f"

exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', '.cache']

# TikZ configuration
tikz_proc_suite = 'pdf2svg'
tikz_resolution = 92
tikz_latex_preamble = r'''
\usepackage[UTF8]{ctex}
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

templates_path = ['_templates']
exclude_patterns = ['build', 'Thumbs.db', '.DS_Store']

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
    "header_end": ["search-button.html", "theme-switcher.html", "external-links.html"],
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


imgmath_latex = 'lualatex'
imgmath_image_format = 'svg'
imgmath_latex_preamble = r'''
\usepackage{unicode-math}
\usepackage[UTF8, fontset=none]{ctex}
\setCJKmainfont{Noto Serif CJK SC}
'''

bibtex_bibfiles = ["references.bib"]
