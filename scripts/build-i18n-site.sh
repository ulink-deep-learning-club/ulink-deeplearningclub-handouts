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
HTML_ONLY=0

if [[ "${1:-}" == "--html-only" ]]; then
  HTML_ONLY=1
elif [[ $# -gt 0 ]]; then
  echo "Usage: $0 [--html-only]" >&2
  exit 2
fi

rm -rf "$BUILD_ROOT"
mkdir -p "$SITE_DIR/en"

sphinx-build -W --keep-going -b gettext "$SOURCE_DIR" "$BUILD_ROOT/gettext"
sphinx-build -W --keep-going -D language=zh_CN -b html "$SOURCE_DIR" "$ZH_BUILD/html"
sphinx-build -W --keep-going -D language=en -b html "$SOURCE_DIR" "$EN_BUILD/html"

RELEASED_PREFIXES=()
while IFS= read -r prefix; do
  [[ -n "$prefix" ]] && RELEASED_PREFIXES+=("$prefix")
done < <(python3 - "$MANIFEST" <<'PY'
import json
import sys

for prefix in json.load(open(sys.argv[1], encoding="utf-8"))["released_prefixes"]:
    print(prefix)
PY
)
for prefix in "${RELEASED_PREFIXES[@]}"; do
  "$ROOT_DIR/scripts/check-i18n.py" --require-complete --prefix "$prefix"
done

cp -a "$ZH_BUILD/html/." "$SITE_DIR/"
python3 "$ROOT_DIR/scripts/assemble_i18n_site.py" \
  --source "$SOURCE_DIR" \
  --english-output "$EN_BUILD/html" \
  --site "$SITE_DIR" \
  --manifest "$MANIFEST"

if [[ "$HTML_ONLY" == "0" ]]; then
  sphinx-build -W --keep-going -D language=zh_CN -b latex "$SOURCE_DIR" "$ZH_BUILD/latex"
  make -C "$ZH_BUILD/latex" all-pdf
  cp "$ZH_BUILD/latex/deeplearningclubhandouts.pdf" "$SITE_DIR/deeplearningclubhandouts.pdf"

  # A full English PDF is meaningful only once every published document is
  # covered by the reviewed release manifest.
  if [[ ${#RELEASED_PREFIXES[@]} -gt 0 ]] && "$ROOT_DIR/scripts/check-i18n.py" --require-complete; then
    sphinx-build -W --keep-going -D language=en -b latex "$SOURCE_DIR" "$EN_BUILD/latex"
    make -C "$EN_BUILD/latex" all-pdf
    cp "$EN_BUILD/latex/deep-learning-club-handouts.pdf" "$SITE_DIR/en/deep-learning-club-handouts.pdf"
  fi
fi

printf 'Bilingual site assembled at %s\n' "$SITE_DIR"
