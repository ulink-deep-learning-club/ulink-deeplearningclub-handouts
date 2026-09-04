# English Localization Agent Guide

Chinese Markdown under `source/` is canonical. Do not create an English source
tree and do not edit `legacy-doc/`. English translations live only in
`source/locale/en/LC_MESSAGES/<source-relative-path>.po`.

## Setup and catalog refresh

Run these commands from the repository root after installing project
dependencies:

```bash
sphinx-build -W --keep-going -b gettext source build/i18n/gettext
sphinx-intl update -p build/i18n/gettext -l en
python scripts/check-i18n.py
```

Commit changed `.po` files. Never commit `build/`, `.pot`, or `.mo` files.
When Chinese content changes, refresh catalogs in the same pull request before
editing the affected English entries.

## Translation rules

- Translate headings, prose, tables, admonitions, captions, Mermaid/TikZ
  labels, and explanatory comments in code examples into natural technical
  English.
- Do not change MyST syntax, `{doc}`/`{ref}` targets, label IDs, citation keys,
  URLs, formulas, filenames, imports, identifiers, commands, or program logic.
- Keep source code executable. English may change comments only after a reviewer
  confirms the non-comment code is identical in behavior.
- Use [the shared glossary](glossary.md); add a reviewed entry before inventing
  a new recurring translation.
- Do not mark an entry translated by copying its Chinese `msgid` into `msgstr`.

## Ownership and review

Work in disjoint catalog domains: one agent owns one top-level section at a
time. The localization reviewer owns `conf.py`, templates, CI, this glossary,
and terminology shared across sections. A fluent English reviewer must approve
each translated domain before it merges.

## Required checks

For an in-progress domain:

```bash
python scripts/check-i18n.py
sphinx-build -W --keep-going -D language=en -b html source build/i18n/en-preview
```

Before enabling the public language switcher, all 107 document catalogs must
exist, have no fuzzy or untranslated messages, and pass:

```bash
python scripts/check-i18n.py --require-complete
I18N_RELEASE_READY=1 scripts/build-i18n-site.sh
```

Inspect the rendered homepage and one page from every top-level section in both
languages. Audit all raster assets referenced from Markdown: retain
language-neutral files once; add an English sibling named `image.en.png` for
each image containing Chinese text. The configured Sphinx figure substitution
selects the English sibling automatically. The current baseline is recorded in
[the raster asset audit](asset-audit.md).

Only then set `I18N_RELEASE_READY=1` in the deployment workflow. This makes the
header switcher visible; until then, the hidden English preview must not be
linked from production.
