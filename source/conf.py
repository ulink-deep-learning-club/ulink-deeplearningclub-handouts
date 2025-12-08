# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import sys
from pathlib import Path

sys.path.append(str(Path('../exts').resolve()))

project = 'Deep Learning Club Lectures'
copyright = '2025, UCS Deep Learning Club'
author = 'UCS Deep Learning Club'
release = '0.0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'myst_parser',
    'sphinx_design',
    'sphinxcontrib.tikz',
    'sphinx.ext.mathjax',
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.githubpages',
    'contributors'
]

exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', '.cache']

# TikZ configuration
tikz_proc_suite = 'Ghostscript'
tikz_latex_preamble = r'''
\usepackage{tikz}
\usetikzlibrary{shapes,arrows,positioning,calc,decorations.pathreplacing}
\usepackage{amsmath}
\usepackage{amssymb}
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
    'repository_url': 'https://github.com/yourusername/deep-learning-club-lecture-material',
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
