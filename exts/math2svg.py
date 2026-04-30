"""
Render math in HTML via LuaLaTeX + pdf2svg.

- LuaLaTeX handles Unicode correctly.
- pdf2svg converts the resulting PDF into clean SVG.
- Depth (vertical alignment) information is **not** available with pdf2svg;
- Error output is always printed (stdout/stderr of the failing command).
"""
from __future__ import annotations

__all__ = ()

import base64
import contextlib
import os
import os.path
import shutil
import subprocess
import tempfile
from hashlib import sha1
from pathlib import Path
from subprocess import CalledProcessError
from typing import TYPE_CHECKING

from docutils import nodes

import sphinx
from sphinx import package_dir
from sphinx.errors import SphinxError
from sphinx.locale import _, __
from sphinx.util import logging
from sphinx.util.math import get_node_equation_number, wrap_displaymath
from sphinx.util.template import LaTeXRenderer

if TYPE_CHECKING:
    from docutils.nodes import Element

    from sphinx.application import Sphinx
    from sphinx.builders import Builder
    from sphinx.config import Config
    from sphinx.util._pathlib import _StrPath
    from sphinx.util.typing import ExtensionMetadata
    from sphinx.writers.html5 import HTML5Translator

logger = logging.getLogger(__name__)

templates_path = package_dir.joinpath('templates', 'imgmath')


class MathExtError(SphinxError):
    category = 'Math extension error'

    def __init__(
        self, msg: str, stderr: str | None = None, stdout: str | None = None
    ) -> None:
        if stderr:
            msg += '\n[stderr]\n' + str(stderr)
        if stdout:
            msg += '\n[stdout]\n' + str(stdout)
        super().__init__(msg)


class InvokeError(SphinxError):
    """Errors on invoking converters."""


# Only SVG is supported.
SUPPORTED_FORMAT = 'svg'


def generate_latex_macro(
    image_format: str,
    math: str,
    config: Config,
    confdir: _StrPath,
) -> str:
    """Generate LaTeX source for a math snippet.

    The ``tightpage`` driver is set to ``pdftex`` because we produce PDF
    directly with LuaLaTeX.  (Depth information is not used by pdf2svg.)
    """
    variables = {
        'fontsize': config.imgmath_font_size,
        'baselineskip': round(config.imgmath_font_size * 1.2),
        'preamble': config.imgmath_latex_preamble,
        'tightpage': ',pdftex,tightpage',
        'math': math,
    }

    template_name = 'preview.tex'

    for template_dir in config.templates_path:
        for template_suffix in ('.jinja', '_t'):
            template = confdir / template_dir / (template_name + template_suffix)
            if template.exists():
                return LaTeXRenderer().render(template, variables)

    return LaTeXRenderer([templates_path]).render(template_name + '.jinja', variables)


def ensure_tempdir(builder: Builder) -> Path:
    """Return (and create once) the temporary directory for this build."""
    if not hasattr(builder, '_imgmath_tempdir'):
        builder._imgmath_tempdir = Path(tempfile.mkdtemp())  # type: ignore[attr-defined]
    return builder._imgmath_tempdir  # type: ignore[attr-defined]


def compile_math(latex: str, builder: Builder) -> Path:
    """Compile the LaTeX snippet with LuaLaTeX → PDF.

    Returns the path to ``math.pdf`` inside the temporary directory.
    """
    tempdir = ensure_tempdir(builder)
    tex_file = tempdir / 'math.tex'
    with open(tex_file, 'w', encoding='utf-8') as f:
        f.write(latex)

    command = [builder.config.imgmath_latex]
    # Use errorstopmode so compilation stops on any error (no malformed PDF)
    command.append('-interaction=nonstopmode')
    command.extend(builder.config.imgmath_latex_args)
    command.append('math.tex')

    try:
        subprocess.run(command, capture_output=True, cwd=tempdir, check=True)
    except OSError as exc:
        logger.warning(
            __('LaTeX command %r cannot be run (needed for math display), '
               'check the imgmath_latex setting'),
            builder.config.imgmath_latex,
        )
        raise InvokeError from exc
    except CalledProcessError as exc:
        logger.warning(
            __('LuaLaTeX exited with error'),
            builder.config.imgmath_latex,
        )
        stderr = exc.stderr.decode(errors='replace') if exc.stderr else ''
        stdout = exc.stdout.decode(errors='replace') if exc.stdout else ''
        msg = 'lualatex exited with error'
        raise MathExtError(msg, stderr, stdout) from exc

    pdf_path = tempdir / 'math.pdf'
    if not pdf_path.is_file() or pdf_path.stat().st_size == 0:
        raise MathExtError('lualatex did not produce a valid PDF (empty or missing)')
    return pdf_path


def convert_pdf_to_svg(pdfpath: Path, builder: Builder, out_path: Path) -> None:
    """Convert ``math.pdf`` to SVG using pdf2svg.

    pdf2svg does **not** provide depth (vertical alignment) information.
    """
    command = [builder.config.imgmath_pdf2svg, str(pdfpath), str(out_path)]
    try:
        subprocess.run(command, capture_output=True, check=True)
    except OSError as exc:
        logger.warning(
            __('pdf2svg command %r cannot be run (needed for math display), '
               'check the imgmath_pdf2svg setting'),
            builder.config.imgmath_pdf2svg,
        )
        raise InvokeError from exc
    except CalledProcessError as exc:
        stderr = exc.stderr.decode(errors='replace') if exc.stderr else ''
        stdout = exc.stdout.decode(errors='replace') if exc.stdout else ''
        msg = 'pdf2svg exited with error'
        raise MathExtError(msg, stderr, stdout) from exc

    if not out_path.is_file() or out_path.stat().st_size == 0:
        raise MathExtError('pdf2svg did not produce a valid SVG output')


def render_math(
    self: HTML5Translator,
    math: str,
) -> tuple[_StrPath | None, int | None]:
    """Render *math* as an SVG image (LuaLaTeX → PDF → pdf2svg).

    Returns the image filename and ``None`` for depth (not supported).
    """
    image_format = self.builder.config.imgmath_image_format.lower()
    if image_format != SUPPORTED_FORMAT:
        raise MathExtError(
            f'imgmath_image_format must be "{SUPPORTED_FORMAT}" '
            f'(got "{image_format}"). '
            'Only SVG is supported with the lualatex+pdf2svg workflow.'
        )

    latex = generate_latex_macro(
        image_format, math, self.builder.config, self.builder.confdir
    )

    filename = (
        f'{sha1(latex.encode(), usedforsecurity=False).hexdigest()}.svg'
    )
    generated_path = self.builder.outdir / self.builder.imagedir / 'math' / filename
    generated_path.parent.mkdir(parents=True, exist_ok=True)

    if generated_path.is_file():
        # Depth is always None for pdf2svg.
        return generated_path, None

    # If any tool has already failed once, skip further attempts.
    latex_failed = hasattr(self.builder, '_imgmath_warned_latex')
    trans_failed = hasattr(self.builder, '_imgmath_warned_image_translator')
    if latex_failed or trans_failed:
        return None, None

    # .tex → .pdf
    try:
        pdfpath = compile_math(latex, self.builder)
    except InvokeError:
        self.builder._imgmath_warned_latex = True  # type: ignore[attr-defined]
        return None, None

    # .pdf → .svg
    try:
        convert_pdf_to_svg(pdfpath, self.builder, generated_path)
    except InvokeError:
        self.builder._imgmath_warned_image_translator = True  # type: ignore[attr-defined]
        return None, None

    return generated_path, None


def render_maths_to_base64(generated_path: Path) -> str:
    """Return a base64 data URI for the SVG."""
    with open(generated_path, 'rb') as f:
        content = f.read()
    encoded = base64.b64encode(content).decode(encoding='utf-8')
    return f'data:image/svg+xml;base64,{encoded}'


def clean_up_files(app: Sphinx, exc: Exception) -> None:
    if exc:
        return

    if hasattr(app.builder, '_imgmath_tempdir'):
        with contextlib.suppress(Exception):
            shutil.rmtree(app.builder._imgmath_tempdir)

    if app.builder.config.imgmath_embed:
        with contextlib.suppress(Exception):
            shutil.rmtree(app.builder.outdir / app.builder.imagedir / 'math')


def get_tooltip(self: HTML5Translator, node: Element) -> str:
    if self.builder.config.imgmath_add_tooltips:
        return f' alt="{self.encode(node.astext()).strip()}"'
    return ''


def html_visit_math(self: HTML5Translator, node: nodes.math) -> None:
    try:
        rendered_path, depth = render_math(self, '$' + node.astext() + '$')
    except MathExtError as exc:
        msg = str(exc)
        sm = nodes.system_message(
            msg, type='WARNING', level=2, backrefs=[], source=node.astext()
        )
        sm.walkabout(self)
        logger.warning(__('display latex %r: %s'), node.astext(), msg)
        raise nodes.SkipNode from exc

    if rendered_path is None:
        self.body.append(
            f'<span class="math">{self.encode(node.astext()).strip()}</span>'
        )
    else:
        if self.builder.config.imgmath_embed:
            img_src = render_maths_to_base64(rendered_path)
        else:
            bname = os.path.basename(rendered_path)
            relative_path = Path(self.builder.imgpath, 'math', bname)
            img_src = relative_path.as_posix()
        # No depth => no vertical-align style.
        self.body.append(
            f'<img class="math" src="{img_src}"{get_tooltip(self, node)}/>'
        )
    raise nodes.SkipNode


def html_visit_displaymath(self: HTML5Translator, node: nodes.math_block) -> None:
    if node.get('no-wrap', node.get('nowrap', False)):
        latex = node.astext()
    else:
        latex = wrap_displaymath(node.astext(), None, False)
    try:
        rendered_path, depth = render_math(self, latex)
    except MathExtError as exc:
        msg = str(exc)
        sm = nodes.system_message(
            msg, type='WARNING', level=2, backrefs=[], source=node.astext()
        )
        sm.walkabout(self)
        logger.warning(__('inline latex %r: %s'), node.astext(), msg)
        raise nodes.SkipNode from exc

    self.body.append(self.starttag(node, 'div', CLASS='math'))
    self.body.append('<p>')
    if node['number']:
        number = get_node_equation_number(self, node)
        self.body.append('<span class="eqno">(%s)' % number)
        self.add_permalink_ref(node, _('Link to this equation'))
        self.body.append('</span>')

    if rendered_path is None:
        self.body.append(
            f'<span class="math">{self.encode(node.astext()).strip()}</span></p>\n</div>'
        )
    else:
        if self.builder.config.imgmath_embed:
            img_src = render_maths_to_base64(rendered_path)
        else:
            bname = os.path.basename(rendered_path)
            relative_path = Path(self.builder.imgpath, 'math', bname)
            img_src = relative_path.as_posix()
        self.body.append(f'<img src="{img_src}"{get_tooltip(self, node)}/></p>\n</div>')
    raise nodes.SkipNode


def setup(app: Sphinx) -> ExtensionMetadata:
    app.add_html_math_renderer(
        'imgmath',
        inline_renderers=(html_visit_math, None),
        block_renderers=(html_visit_displaymath, None),
    )

    # Core settings
    app.add_config_value('imgmath_image_format', 'svg', 'html', types=frozenset({str}))
    app.add_config_value('imgmath_latex', 'lualatex', 'html', types=frozenset({str}))
    app.add_config_value('imgmath_pdf2svg', 'pdf2svg', 'html', types=frozenset({str}))

    # LaTeX behaviour
    app.add_config_value('imgmath_latex_args', [], 'html', types=frozenset({list, tuple}))
    app.add_config_value('imgmath_latex_preamble', '', 'html', types=frozenset({str}))
    app.add_config_value('imgmath_font_size', 12, 'html', types=frozenset({int}))

    # Presentation
    app.add_config_value('imgmath_add_tooltips', True, 'html', types=frozenset({bool}))
    app.add_config_value('imgmath_embed', False, 'html', types=frozenset({bool}))

    app.connect('build-finished', clean_up_files)

    return {
        'version': sphinx.__display_version__,
        'parallel_read_safe': True,
    }
