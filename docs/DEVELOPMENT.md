# Local Development and Serving

Run all build commands from the repository root, never from `build/` or a
previously served output directory. From anywhere inside this checkout, return
there with:

```bash
cd "$(git rev-parse --show-toplevel)"
```

There are two useful local modes.

## Fast Chinese-only preview

Use this while editing ordinary Chinese Markdown:

```bash
cd "$(git rev-parse --show-toplevel)"
uv sync
source .venv/bin/activate
make html
cd build/html
python -m http.server 8000 --bind 127.0.0.1
```

Open <http://localhost:8000>.

## Bilingual staging preview

Use this to test language switching, English catalogs, and the same site
layout that GitHub Pages receives. `--html-only` avoids the slower TeX/PDF
build:

```bash
cd "$(git rev-parse --show-toplevel)"
uv sync
source .venv/bin/activate
bash scripts/build-i18n-site.sh --html-only
cd build/i18n/site
python -m http.server 8000 --bind 127.0.0.1
```

Open <http://localhost:8000>. Approved English chapter prefixes appear at
`/en/`; unreleased English URLs redirect to their corresponding Chinese page.

This relaxed preview mode still works without TeX or Mermaid installed, but
TikZ/Mermaid diagrams may be missing. Full and CI builds remain strict.
The build script always uses this repository’s `.venv`, even if Conda is active.

To build PDFs too, omit `--html-only`. This requires the TeX, Chromium, and
Mermaid dependencies documented in [the build appendix](../source/appendix/build-docs.md).

## Releasing an English chapter locally

1. Complete and review all catalogs in a chapter prefix.
2. Add the prefix, for example `math-fundamentals`, to
   `docs/i18n/release-manifest.json`.
3. Validate and rebuild:

   ```bash
   python scripts/check-i18n.py --require-complete --prefix math-fundamentals
   bash scripts/build-i18n-site.sh --html-only
   ```

See [the localization agent guide](i18n/AGENT_GUIDE.md) for translation and
review requirements.

## If `uv` is unavailable

Install it with `brew install uv`, then rerun the commands above. Alternatively,
create the virtual environment with Python:

```bash
cd "$(git rev-parse --show-toplevel)"
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```
