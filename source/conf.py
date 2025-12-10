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
    logger.info("Using Chrome at: ", chrome_path)
    os.environ["PUPPETEER_EXECUTABLE_PATH"] = str(chrome_path)
sys.path.append(str(Path('../exts').resolve()))

project = 'Deep Learning Club Lectures'
copyright = '2025, UCS Deep Learning Club and Contributors'
author = 'UCS Deep Learning Club'
release = '0.0.1'

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
    'sphinx.ext.mathjax',
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.githubpages',
    'contributors',
    'fix_mermaid_svgs'
]

mermaid_output_format = 'svg'
mermaid_params = [
    '--theme', 'neutral',
    '--backgroundColor', 'white',
    '--scale', '1'             # 1 = actual size, 2 = 2x, etc.
]

exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', '.cache']

# TikZ configuration
tikz_proc_suite = 'pdf2svg'
tikz_resolution = 92
tikz_latex_preamble = r'''
\usepackage[UTF8]{ctex}
\usepackage{fontspec}
\usetikzlibrary{shapes,arrows,arrows.meta,positioning,shapes.geometric,calc,decorations.pathreplacing}
\usepackage{amsmath}
\usepackage{amssymb}

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

html_theme = 'sphinx_book_theme'
html_static_path = ['_static']

# Sphinx Book Theme configuration
html_theme_options = {
    'repository_url': 'https://github.com/ulink-deep-learning-club/ulink-deeplearningclub-handouts',
    'use_repository_button': True,
    'use_issues_button': True,
    'use_edit_page_button': True,
    'navigation_depth': 4,
    'show_toc_level': 3,
    'logo': {
        'text': 'Deep Learning Club Lectures',
    },
}

# MathJax configuration
mathjax3_config = {
    'tex': {
        'inlineMath': [['$', '$'], ['\\(', '\\)']],
        'displayMath': [['$$', '$$'], ['\\[', '\\]']],
    }
}
