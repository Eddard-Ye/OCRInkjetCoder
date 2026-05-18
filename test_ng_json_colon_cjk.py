# -*- coding: utf-8 -*-
"""Batch-evaluate NG OCR JSON snapshots with ColonCjkPhraseMatchStrategy."""

from __future__ import annotations

import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from production_phrase_strategy import ColonCjkPhraseMatchStrategy

JSON_DIR = Path(r"C:\Users\Admin\HikCameraPhotos\ng\withJson\json")


def texts_from_json(path: Path) -> list[str]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    boxes = data.get("boxes") or []
    texts: list[str] = []
    for b in boxes:
        if isinstance(b, dict):
            t = str(b.get("text") or "").strip()
            if t:
                texts.append(t)
    return texts


def main() -> None:
    if not JSON_DIR.is_dir():
        print(f"Directory not found: {JSON_DIR}", file=sys.stderr)
        sys.exit(1)

    strategy = ColonCjkPhraseMatchStrategy(
        exclude_lines_without_year=False,
        max_cjk_length_diff=2,
        min_cjk_lcs_matches=3,
    )

    paths = sorted(JSON_DIR.glob("*.json"))
    true_names: list[str] = []
    false_names: list[str] = []
    empty_boxes: list[str] = []

    for path in paths:
        texts = texts_from_json(path)
        if not texts:
            empty_boxes.append(path.name)
            false_names.append(path.name)
            continue
        if strategy.match(texts):
            true_names.append(path.name)
        else:
            false_names.append(path.name)

    n = len(paths)
    n_true = len(true_names)
    n_false = len(false_names)

    print(f"JSON directory: {JSON_DIR}")
    print(f"Strategy: ColonCjkPhraseMatchStrategy (defaults: x=2, y=3, no year filter)")
    print(f"Total JSON files: {n}")
    print(f"True:  {n_true}")
    print(f"False: {n_false}")
    if empty_boxes:
        print(f"  (includes {len(empty_boxes)} with empty/missing box text -> counted as False)")
    print()
    print("--- False filenames ---")
    for name in false_names:
        print(name)


if __name__ == "__main__":
    main()
