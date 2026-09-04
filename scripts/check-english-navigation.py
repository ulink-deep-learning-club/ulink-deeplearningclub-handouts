#!/usr/bin/env python3
"""Validate the English sidebar's released and Chinese-only chapter catalog."""

from __future__ import annotations

import argparse
from pathlib import Path
import re

from bs4 import BeautifulSoup


EXPECTED_TITLES = [
    "Preface: About Deep Learning",
    "Core Mathematical Foundations of Deep Learning",
    "CNN Expedition: Thirty Years of Architectures",
    "PyTorch Practice: Turning Theory into Code",
    "Model Deployment and Serving",
    "Transfer Learning and Fine-Tuning",
    "U-Net Image Segmentation",
    "Sequence Modeling: RNN to Transformer to Mamba",
    "Appendix",
    "Postscript",
]
CHINESE_ONLY_COUNT = 8
CJK = re.compile(r"[\u3400-\u9fff]")


def check_page(path: Path, failures: list[str]) -> None:
    soup = BeautifulSoup(path.read_text(encoding="utf-8"), "html.parser")
    nav = soup.select_one("nav.bz-sidebar-nav")
    if nav is None:
        failures.append(f"{path}: missing primary navigation")
        return

    items = nav.select(":scope > ul > li.toctree-l1")
    titles = []
    for item in items:
        link = item.find("a", recursive=False)
        if link:
            title = link.find(string=True, recursive=False)
            titles.append(title.strip() if title else "")

    if titles != EXPECTED_TITLES:
        failures.append(f"{path}: unexpected top-level entries: {titles!r}")

    unavailable = nav.select("li.i18n-unavailable")
    if len(unavailable) != CHINESE_ONLY_COUNT:
        failures.append(
            f"{path}: expected {CHINESE_ONLY_COUNT} Chinese-only entries, "
            f"found {len(unavailable)}"
        )
    for item in unavailable:
        link = item.find("a", recursive=False)
        badge = item.select_one(".i18n-unavailable-badge")
        if not link or link.get("hreflang") != "zh-CN" or not badge:
            failures.append(f"{path}: malformed Chinese-only entry")

    if CJK.search(nav.get_text(" ", strip=True)):
        failures.append(f"{path}: sidebar contains untranslated Chinese text")

    if path.name == "preface.html":
        current = nav.select_one('li.current > a[aria-current="page"]')
        if not current or current.get_text(" ", strip=True) != EXPECTED_TITLES[0]:
            failures.append(f"{path}: preface is not shown as the current page")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--site", type=Path, required=True)
    args = parser.parse_args()

    failures: list[str] = []
    english_root = args.site / "en"
    check_page(english_root / "preface.html", failures)
    for page in sorted((english_root / "math-fundamentals").glob("*.html")):
        check_page(page, failures)

    if failures:
        raise SystemExit("English navigation check failed:\n" + "\n".join(failures))
    print("English navigation lists released and Chinese-only content correctly")


if __name__ == "__main__":
    main()
