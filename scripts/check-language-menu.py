#!/usr/bin/env python3
"""Ensure every published language selector uses the shared menu template."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


MENU_CLASS = 'class="i18n-language-menu"'
LEGACY_CLASS = 'i18n-language-switcher'
EXCLUDED_PREFIX = Path("appendix/sphinx-guide/examples")


def is_released(docname: str, prefixes: list[str]) -> bool:
    return any(docname == prefix or docname.startswith(f"{prefix}/") for prefix in prefixes)


def check_page(path: Path, should_have_menu: bool, failures: list[str]) -> None:
    html = path.read_text(encoding="utf-8")
    menu_count = html.count(MENU_CLASS)
    if LEGACY_CLASS in html:
        failures.append(f"{path}: contains the retired language-switch button")
    if should_have_menu and menu_count != 1:
        failures.append(f"{path}: expected one shared language menu, found {menu_count}")
    if should_have_menu and (
        "<summary>Language</summary>" not in html
        or '<ul aria-label="Language">' not in html
    ):
        failures.append(f"{path}: language menu must always be labelled Language")
    if not should_have_menu and menu_count:
        failures.append(f"{path}: unexpected language menu")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--site", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    prefixes = manifest["released_prefixes"]
    if not isinstance(prefixes, list) or not all(isinstance(prefix, str) and prefix for prefix in prefixes):
        raise ValueError("release manifest must contain non-empty released_prefixes strings")

    failures: list[str] = []
    for source_file in args.source.rglob("*.md"):
        relative = source_file.relative_to(args.source)
        if relative.is_relative_to(EXCLUDED_PREFIX):
            continue
        docname = relative.with_suffix("").as_posix()
        released = is_released(docname, prefixes)
        check_page(args.site / relative.with_suffix(".html"), released, failures)
        check_page(args.site / "en" / relative.with_suffix(".html"), released, failures)

    stylesheet = args.site / "_static" / "i18n.css"
    css = stylesheet.read_text(encoding="utf-8")
    if ".i18n-language-menu" not in css or LEGACY_CLASS in css:
        failures.append(f"{stylesheet}: does not contain the shared menu styles")

    if failures:
        raise SystemExit("Language menu check failed:\n" + "\n".join(failures))
    print("All published language selectors use the shared menu")


if __name__ == "__main__":
    main()
