#!/usr/bin/env python3
"""Publish released English pages and redirect the remainder to Chinese."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import posixpath
import shutil


def is_released(docname: str, prefixes: list[str]) -> bool:
    return any(docname == prefix or docname.startswith(f"{prefix}/") for prefix in prefixes)


def redirect_page(target: str) -> str:
    return f'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta http-equiv="refresh" content="0; url={target}">
<link rel="canonical" href="{target}"><title>Redirecting…</title>
<script>location.replace({target!r});</script></head>
<body><p>This page is not translated yet. <a href="{target}">Continue in Chinese</a>.</p></body></html>\n'''


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--english-output", type=Path, required=True)
    parser.add_argument("--site", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    prefixes = manifest["released_prefixes"]
    if not isinstance(prefixes, list) or not all(isinstance(prefix, str) and prefix for prefix in prefixes):
        raise ValueError("release manifest must contain non-empty released_prefixes strings")

    english_site = args.site / "en"
    shutil.copytree(args.english_output, english_site, dirs_exist_ok=True)
    excluded = Path("appendix/sphinx-guide/examples")
    for source_file in args.source.rglob("*.md"):
        relative = source_file.relative_to(args.source)
        if relative.is_relative_to(excluded):
            continue
        docname = relative.with_suffix("").as_posix()
        if is_released(docname, prefixes):
            continue
        output = english_site / relative.with_suffix(".html")
        output.parent.mkdir(parents=True, exist_ok=True)
        target = posixpath.relpath(
            f"{docname}.html", start=posixpath.dirname(f"en/{docname}.html")
        )
        output.write_text(redirect_page(target), encoding="utf-8")


if __name__ == "__main__":
    main()
