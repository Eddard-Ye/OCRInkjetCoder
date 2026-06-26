# -*- coding: utf-8 -*-
"""Pixel-size checks on filtered PaddleOCR detection boxes."""

from __future__ import annotations

from typing import Any, Optional, Sequence

import numpy as np

from pixel_match_config import PixelMatchConfig


def box_screen_width_height(box: dict[str, Any]) -> tuple[float, float]:
    """
    Return (horizontal_length, vertical_width) in image pixels.

    Facing the screen: length = horizontal (x), width = vertical (y).
    """
    bb = box.get("bbox_xyxy")
    if isinstance(bb, (list, tuple)) and len(bb) >= 4:
        x1, y1, x2, y2 = (float(bb[0]), float(bb[1]), float(bb[2]), float(bb[3]))
        return max(0.0, x2 - x1), max(0.0, y2 - y1)

    poly = box.get("poly")
    if poly is None:
        return 0.0, 0.0
    pts = np.asarray(poly, dtype=np.float32).reshape(-1, 2)
    if pts.size == 0:
        return 0.0, 0.0
    xs = pts[:, 0]
    ys = pts[:, 1]
    return float(xs.max() - xs.min()), float(ys.max() - ys.min())


def diagnose_pixel_match(
    boxes: Sequence[dict[str, Any]],
    cfg: PixelMatchConfig,
) -> dict[str, Any]:
    """Check count and per-box pixel dimensions."""
    params = cfg.to_dict()
    if not cfg.enabled:
        return {
            "enabled": False,
            "skipped": True,
            "passed": None,
            "params": params,
            "box_count": len(boxes),
            "valid_box_count": None,
            "boxes": [],
            "failure": None,
        }

    rows: list[dict[str, Any]] = []
    all_size_ok = True
    for i, box in enumerate(boxes):
        horiz, vert = box_screen_width_height(box)
        width_ok = vert > float(cfg.min_pixel_width)
        length_ok = horiz > float(cfg.min_pixel_length)
        ok = width_ok and length_ok
        if not ok:
            all_size_ok = False
        rows.append(
            {
                "index": i,
                "text": str(box.get("text", "") or "")[:64],
                "horizontal_length_px": round(horiz, 1),
                "vertical_width_px": round(vert, 1),
                "width_ok": width_ok,
                "length_ok": length_ok,
                "passed": ok,
            }
        )

    count_ok = len(boxes) > int(cfg.min_window_count)
    passed = count_ok and all_size_ok and len(boxes) > 0

    failure: Optional[dict[str, Any]] = None
    if not passed:
        if len(boxes) == 0:
            failure = {
                "code": "no_boxes",
                "message": "No OCR boxes after filtering",
            }
        elif not count_ok:
            failure = {
                "code": "insufficient_window_count",
                "message": (
                    f"Box count {len(boxes)} is not greater than min_window_count "
                    f"{cfg.min_window_count}"
                ),
            }
        else:
            bad = [r for r in rows if not r["passed"]]
            failure = {
                "code": "box_size",
                "message": f"{len(bad)} box(es) below min pixel width/length",
                "detail": bad[:8],
            }

    return {
        "enabled": True,
        "skipped": False,
        "passed": passed,
        "params": params,
        "box_count": len(boxes),
        "valid_box_count": sum(1 for r in rows if r["passed"]),
        "boxes": rows,
        "failure": failure,
    }
