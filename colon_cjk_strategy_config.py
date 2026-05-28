# -*- coding: utf-8 -*-
"""Persisted UI config for ColonCjkPhraseMatchStrategy thresholds."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any


@dataclass
class ColonCjkStrategyConfig:
    max_cjk_length_diff: int = 2
    min_match_percentage_limit: float = 0.75

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_cjk_length_diff": int(self.max_cjk_length_diff),
            "min_match_percentage_limit": float(self.min_match_percentage_limit),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> ColonCjkStrategyConfig:
        if not data:
            return cls()
        try:
            xd = int(data.get("max_cjk_length_diff", 2))
        except (TypeError, ValueError):
            xd = 2
        try:
            pct = float(data.get("min_match_percentage_limit", 0.75))
        except (TypeError, ValueError):
            pct = 0.75
        if xd < 0:
            xd = 0
        pct = max(0.0, min(1.0, pct))
        return cls(max_cjk_length_diff=xd, min_match_percentage_limit=pct)

    @classmethod
    def load(cls, path: str) -> ColonCjkStrategyConfig:
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
