"""Keep HTML structurally valid when a local preview cannot render TikZ."""

from docutils import nodes
from sphinxcontrib.tikz import TikzExtError, render_tikz, tikz


def html_visit_tikz(translator, node):
    """Render a TikZ figure, or leave a contained fallback on missing TeX.

    sphinxcontrib-tikz reports a render error but does not skip the node.  Its
    departure handler then writes a closing ``</div>`` without a matching
    opening tag, causing the Breeze theme's article layout to escape after the
    first failed diagram.
    """
    libs = translator.builder.config.tikz_tikzlibraries + ',' + node['libs']
    libs = libs.replace(' ', '').replace('\t', '').strip(', ')

    unavailable_message = (
        'This diagram is unavailable in the local preview (LaTeX and pdf2svg are required).'
        if translator.builder.config.language == 'en'
        else '该图在本地预览中不可用（需要 LaTeX 和 pdf2svg）。'
    )

    try:
        filename = render_tikz(translator, node, libs, node['stringsubst'])
    except TikzExtError as exc:
        translator.document.reporter.warning(str(exc), line=node.line)
        translator.body.append(
            '<div class="figure tikz-unavailable" role="note">'
            f'<p>{unavailable_message}</p>'
        )
        return

    # The upstream extension suppresses repeated conversion attempts after a
    # failure. Keep subsequent figures valid as well.
    if filename is None:
        translator.body.append(
            '<div class="figure tikz-unavailable" role="note">'
            f'<p>{unavailable_message}</p>'
        )
        return

    scale = f' width="{node["xscale"]}%"' if node['xscale'] else ''
    style = f'text-align: {translator.encode(node["align"])}'
    translator.body.append(translator.starttag(node, 'div', CLASS='figure', STYLE=style).strip())
    translator.body.append(
        f'<p><img{scale} src="{filename}" '
        f'alt="{translator.encode(node["alt"]).strip()}" /></p>\n'
    )


def setup(app):
    # Preserve the node's existing figure-numbering registration and replace
    # only its HTML visitor installed by sphinxcontrib-tikz.
    app.add_node(
        tikz,
        override=True,
        html=(html_visit_tikz, lambda translator, node: translator.body.append('</div>')),
    )
    return {'parallel_read_safe': True, 'parallel_write_safe': True}
