#!/usr/bin/env python3
"""Validate tracked Sphinx English catalogs for a release build."""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "source"
CATALOG_ROOT = SOURCE / "locale" / "en" / "LC_MESSAGES"
MESSAGE_ID = re.compile(r'^msgid\s+"(.*)"$', re.MULTILINE)


def catalog_for(document: Path) -> Path:
    return CATALOG_ROOT / document.relative_to(SOURCE).with_suffix(".po")


def has_non_header_message(output: str) -> bool:
    return any(match.group(1) for match in MESSAGE_ID.finditer(output))


def run(*args: str) -> str:
    completed = subprocess.run(args, check=False, capture_output=True, text=True)
    if completed.returncode:
        raise RuntimeError(completed.stderr.strip() or " ".join(args))
    return completed.stdout


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()

    if shutil.which("msgfmt") is None or shutil.which("msgattrib") is None:
        print("msgfmt and msgattrib (GNU gettext) are required.", file=sys.stderr)
        return 2

    errors: list[str] = []
    documents = sorted(SOURCE.rglob("*.md"))
    for document in documents:
        catalog = catalog_for(document)
        if not catalog.exists():
            if args.require_complete:
                errors.append(f"missing catalog: {catalog.relative_to(ROOT)}")
            continue
        try:
            run("msgfmt", "--check", "--output-file=/dev/null", str(catalog))
            untranslated = run("msgattrib", "--untranslated", "--no-obsolete", "--output-file=-", str(catalog))
            fuzzy = run("msgattrib", "--only-fuzzy", "--no-obsolete", "--output-file=-", str(catalog))
        except RuntimeError as error:
            errors.append(f"invalid catalog {catalog.relative_to(ROOT)}: {error}")
            continue
        if args.require_complete and has_non_header_message(untranslated):
            errors.append(f"untranslated messages: {catalog.relative_to(ROOT)}")
        if args.require_complete and has_non_header_message(fuzzy):
            errors.append(f"fuzzy messages: {catalog.relative_to(ROOT)}")

    if errors:
        print("i18n validation failed:", file=sys.stderr)
        print("\n".join(f"- {error}" for error in errors), file=sys.stderr)
        return 1
    print("i18n catalogs are valid" + (" and complete" if args.require_complete else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
