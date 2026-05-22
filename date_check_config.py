# -*- coding: utf-8 -*-
"""Date-check global config (JSON) and OCR line date validation."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any, Optional, Sequence

# YYYY.MM.DD / YYYY-MM-DD / YYYY/MM/DD / YYYY��MM��DD��
_DATE_RE = re.compile(
    r"(?P<y>20\d{2})"
    r"(?:[\./\-]|[\u5e74])?"
    r"(?P<m>\d{1,2})"
    r"(?:[\./\-]|[\u6708])?"
    r"(?P<d>\d{1,2})"
)

_KEY_PRODUCTION = "\u751f\u4ea7\u65e5\u671f"
_KEY_NORMAL = "\u5e38\u6e29"
_KEY_FROZEN = "\u51b7\u51bb"


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


def _line_date_for_keyword(texts: Sequence[str], keyword: str) -> Optional[date]:
    for t in texts:
        if keyword in t:
            dt = parse_first_date_in_text(t)
            if dt is not None:
                return dt
    return None


def validate_shelf_life_dates(
    texts: Sequence[str],
    cfg: DateCheckGlobalConfig,
) -> bool:
    prod = _line_date_for_keyword(texts, _KEY_PRODUCTION)
    normal_exp = _line_date_for_keyword(texts, _KEY_NORMAL)
    frozen_exp = _line_date_for_keyword(texts, _KEY_FROZEN)
    if prod is None or normal_exp is None or frozen_exp is None:
        return False
    expect_normal = prod + timedelta(days=int(cfg.shelf_life_normal))
    expect_frozen = prod + timedelta(days=int(cfg.shelf_life_frozen))
    return normal_exp == expect_normal and frozen_exp == expect_frozen
