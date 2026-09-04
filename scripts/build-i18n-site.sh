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

if [[ "${I18N_RELEASE_READY:-0}" == "1" ]]; then
  "$ROOT_DIR/scripts/check-i18n.py" --require-complete
fi

rm -rf "$BUILD_ROOT"
mkdir -p "$SITE_DIR/en"

sphinx-build -W --keep-going -b gettext "$SOURCE_DIR" "$BUILD_ROOT/gettext"
sphinx-build -W --keep-going -D language=zh_CN -b html "$SOURCE_DIR" "$ZH_BUILD/html"
sphinx-build -W --keep-going -D language=en -b html "$SOURCE_DIR" "$EN_BUILD/html"

sphinx-build -W --keep-going -D language=zh_CN -b latex "$SOURCE_DIR" "$ZH_BUILD/latex"
make -C "$ZH_BUILD/latex" all-pdf
sphinx-build -W --keep-going -D language=en -b latex "$SOURCE_DIR" "$EN_BUILD/latex"
make -C "$EN_BUILD/latex" all-pdf

cp -a "$ZH_BUILD/html/." "$SITE_DIR/"
cp "$ZH_BUILD/latex/deeplearningclubhandouts.pdf" "$SITE_DIR/deeplearningclubhandouts.pdf"

if [[ "${I18N_RELEASE_READY:-0}" == "1" ]]; then
  cp -a "$EN_BUILD/html/." "$SITE_DIR/en/"
  cp "$EN_BUILD/latex/deep-learning-club-handouts.pdf" "$SITE_DIR/en/deep-learning-club-handouts.pdf"
else
  rmdir "$SITE_DIR/en"
  printf 'English preview built but not published; set I18N_RELEASE_READY=1 after completion review.\n'
fi

printf 'Bilingual site assembled at %s\n' "$SITE_DIR"
