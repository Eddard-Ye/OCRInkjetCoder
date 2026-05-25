# -*- coding: utf-8 -*-
"""Date-check global config (JSON) and OCR line date validation."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any, Optional, Sequence

# YYYY.MM.DD / YYYY-MM-DD / YYYY/MM/DD / YYYY年MM月DD日
_DATE_RE = re.compile(
    r"(?P<y>20\d{2})"
    r"(?:[\./\-]|[\u5e74])?"
    r"(?P<m>\d{1,2})"
    r"(?:[\./\-]|[\u6708])?"
    r"(?P<d>\d{1,2})"
)


def _system_today() -> date:
    """Local calendar date (production date when date check is enabled)."""
    return date.today()


@dataclass
class DateCheckGlobalConfig:
    enable_date_check: bool = False
    shelf_life_normal: int = 5
    shelf_life_frozen: int = 180

    def to_dict(self) -> dict[str, Any]:
        return {
            "enable_date_check": bool(self.enable_date_check),
            "shelf_life_normal": int(self.shelf_life_normal),
            "shelf_life_frozen": int(self.shelf_life_frozen),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> DateCheckGlobalConfig:
        if not data:
            return cls()
        return cls(
            enable_date_check=bool(data.get("enable_date_check", False)),
            shelf_life_normal=max(0, int(data.get("shelf_life_normal", 5))),
            shelf_life_frozen=max(0, int(data.get("shelf_life_frozen", 180))),
        )

    @classmethod
    def load(cls, path: str) -> DateCheckGlobalConfig:
        if not os.path.isfile(path):
            return cls()
        try:
            with open(path, encoding="utf-8") as f:
                raw = json.load(f)
            if isinstance(raw, dict):
                return cls.from_dict(raw)
        except Exception:
            pass
        return cls()

    def save(self, path: str) -> None:
        parent = os.path.dirname(path)
        if parent and not os.path.isdir(parent):
            os.makedirs(parent, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)


def parse_first_date_in_text(text: str) -> Optional[date]:
    m = _DATE_RE.search(str(text))
    if not m:
        return None
    try:
        return date(int(m.group("y")), int(m.group("m")), int(m.group("d")))
    except ValueError:
        return None


def _parsed_dates_by_text_index(texts: Sequence[str]) -> dict[int, date]:
    """Map each OCR line index to its first parsed date (skip lines without date)."""
    out: dict[int, date] = {}
    for i, t in enumerate(texts):
        dt = parse_first_date_in_text(t)
        if dt is not None:
            out[i] = dt
    return out


def diagnose_shelf_life_dates(
    texts: Sequence[str],
    cfg: DateCheckGlobalConfig,
) -> dict[str, Any]:
    """
    Structured diagnosis: three distinct OCR lines must parse to today / +normal / +frozen.
    """
    today = _system_today()
    expect_normal = today + timedelta(days=int(cfg.shelf_life_normal))
    expect_frozen = today + timedelta(days=int(cfg.shelf_life_frozen))
    expected = {
        "production": today.isoformat(),
        "normal_expiry": expect_normal.isoformat(),
        "frozen_expiry": expect_frozen.isoformat(),
    }

    by_idx = _parsed_dates_by_text_index(texts)
    parsed_lines = [
        {
            "index": i,
            "text": str(texts[i]),
            "date": dt.isoformat(),
        }
        for i, dt in sorted(by_idx.items())
    ]

    roles = (
        ("production", today, "生产日(系统当天)"),
        ("normal_expiry", expect_normal, f"常温到期(+{cfg.shelf_life_normal}天)"),
        ("frozen_expiry", expect_frozen, f"冷冻到期(+{cfg.shelf_life_frozen}天)"),
    )

    failures: list[dict[str, Any]] = []
    if len(by_idx) < 3:
        failures.append(
            {
                "code": "TOO_FEW_DATE_LINES",
                "role": "all",
                "message": f"至少 3 行可解析日期，当前仅 {len(by_idx)} 行",
                "expected_date": None,
            }
        )

    used: set[int] = set()
    for role, target, desc in roles:
        matched = False
        for idx, dt in by_idx.items():
            if idx in used:
                continue
            if dt == target:
                used.add(idx)
                matched = True
                break
        if not matched:
            failures.append(
                {
                    "code": f"MISSING_{role.upper()}",
                    "role": role,
                    "message": (
                        f"未找到独立 OCR 行解析为 {target.isoformat()}（{desc}）"
                    ),
                    "expected_date": target.isoformat(),
                }
            )

    passed = len(failures) == 0
    return {
        "passed": passed,
        "expected_dates": expected,
        "system_today": today.isoformat(),
        "parsed_lines": parsed_lines,
        "failure": failures[0] if failures else None,
        "failures": failures,
    }


def validate_shelf_life_dates(
    texts: Sequence[str],
    cfg: DateCheckGlobalConfig,
) -> bool:
    return bool(diagnose_shelf_life_dates(texts, cfg)["passed"])
