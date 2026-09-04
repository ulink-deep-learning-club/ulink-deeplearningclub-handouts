# Raster Asset Language Audit

Audited 2026-09-04. All current raster assets are language-neutral or already
contain English/math-only labels, so no `*.en.png` sibling is required for the
initial English edition. Re-audit any asset that is replaced or newly added.

| Asset | Result |
| --- | --- |
| `cbam-block.png` | English labels; shared |
| `cbam-channel-spatial-module.png` | English labels; shared |
| `cbam-resnet.png` | English labels; shared |
| `common-conv.jpg` | Diagram only; shared |
| `conv-param-share.png` | English labels; shared |
| `conv-process.png` | Symbols/numerals only; shared |
| `depthwise-separable-conv.jpg` | Diagram only; shared |
| `gradient_descent.png` | English title; shared |
| `lstm.png` | Math symbols only; shared |
| `mnist.png` | English labels; shared |
| `rfb-module.png` | English labels; shared |
| `scalar-to-tensor.png` | English labels; shared |
| `se-block.png` | Math symbols only; shared |
| `se-resnet.png` | English labels; shared |
| `simple-nn-demo.png` | Numerals only; shared |
| `squeeze-and-unsqueeze.png` | English labels; shared |
| `u-net-architecture.png` | English labels; shared |

For a future Chinese-text image, add an English variant alongside it using
`<basename>.en.<extension>` (for example, `pipeline.en.png`). Sphinx selects it
for `-D language=en` through `figure_language_filename` in `source/conf.py`.
