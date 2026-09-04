#!/usr/bin/env bash
# Build isolated Chinese and English documentation trees, then assemble the
# GitHub Pages layout: Chinese at / and English at /en/.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SOURCE_DIR="$ROOT_DIR/source"
BUILD_ROOT="$ROOT_DIR/build/i18n"
ZH_BUILD="$BUILD_ROOT/zh"
EN_BUILD="$BUILD_ROOT/en"
SITE_DIR="$BUILD_ROOT/site"
MANIFEST="$ROOT_DIR/docs/i18n/release-manifest.json"
SPHINX_BUILD="$ROOT_DIR/.venv/bin/sphinx-build"
HTML_ONLY=0

if [[ ! -x "$SPHINX_BUILD" ]]; then
  echo "Missing project virtual environment. Run: uv sync" >&2
  exit 2
fi

if [[ "${1:-}" == "--html-only" ]]; then
  HTML_ONLY=1
elif [[ $# -gt 0 ]]; then
  echo "Usage: $0 [--html-only]" >&2
  exit 2
fi

# Local HTML previews must remain usable on machines without TeX or Mermaid.
# Release/CI builds stay strict and therefore catch every Sphinx warning.
HTML_BUILD_OPTIONS=(--keep-going)
if [[ "$HTML_ONLY" == "0" ]]; then
  HTML_BUILD_OPTIONS=(-W --keep-going)
fi

rm -rf "$BUILD_ROOT"
mkdir -p "$SITE_DIR/en"

"$SPHINX_BUILD" -W --keep-going -b gettext "$SOURCE_DIR" "$BUILD_ROOT/gettext"
"$SPHINX_BUILD" "${HTML_BUILD_OPTIONS[@]}" -D language=zh_CN -b html "$SOURCE_DIR" "$ZH_BUILD/html"
"$SPHINX_BUILD" "${HTML_BUILD_OPTIONS[@]}" -D language=en -b html "$SOURCE_DIR" "$EN_BUILD/html"

RELEASED_PREFIXES=()
RELEASED_PREFIX_COUNT=0
while IFS= read -r prefix; do
  if [[ -n "$prefix" ]]; then
    RELEASED_PREFIXES+=("$prefix")
    RELEASED_PREFIX_COUNT=$((RELEASED_PREFIX_COUNT + 1))
  fi
done < <(python3 - "$MANIFEST" <<'PY'
import json
import sys

for prefix in json.load(open(sys.argv[1], encoding="utf-8"))["released_prefixes"]:
    print(prefix)
PY
)
for prefix in "${RELEASED_PREFIXES[@]:-}"; do
  [[ -z "$prefix" ]] && continue
  "$ROOT_DIR/scripts/check-i18n.py" --require-complete --prefix "$prefix"
done

cp -a "$ZH_BUILD/html/." "$SITE_DIR/"
python3 "$ROOT_DIR/scripts/assemble_i18n_site.py" \
  --source "$SOURCE_DIR" \
  --english-output "$EN_BUILD/html" \
  --site "$SITE_DIR" \
  --manifest "$MANIFEST"

if [[ "$HTML_ONLY" == "0" ]]; then
  "$SPHINX_BUILD" -W --keep-going -D language=zh_CN -b latex "$SOURCE_DIR" "$ZH_BUILD/latex"
  make -C "$ZH_BUILD/latex" all-pdf
  cp "$ZH_BUILD/latex/deeplearningclubhandouts.pdf" "$SITE_DIR/deeplearningclubhandouts.pdf"

  # A full English PDF is meaningful only once every published document is
  # covered by the reviewed release manifest.
  if [[ "$RELEASED_PREFIX_COUNT" -gt 0 ]] && "$ROOT_DIR/scripts/check-i18n.py" --require-complete; then
    "$SPHINX_BUILD" -W --keep-going -D language=en -b latex "$SOURCE_DIR" "$EN_BUILD/latex"
    make -C "$EN_BUILD/latex" all-pdf
    cp "$EN_BUILD/latex/deep-learning-club-handouts.pdf" "$SITE_DIR/en/deep-learning-club-handouts.pdf"
  fi
fi

printf 'Bilingual site assembled at %s\n' "$SITE_DIR"
