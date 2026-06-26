# -*- coding: utf-8 -*-
"""Persisted UI config for OCR box pixel-dimension matching."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any


@dataclass
class PixelMatchConfig:
    enabled: bool = False
    min_window_count: int = 0
    min_pixel_width: int = 10
    min_pixel_length: int = 20

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": bool(self.enabled),
            "min_window_count": int(self.min_window_count),
            "min_pixel_width": int(self.min_pixel_width),
            "min_pixel_length": int(self.min_pixel_length),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> PixelMatchConfig:
        if not data:
            return cls()
        try:
            min_window_count = int(data.get("min_window_count", 0))
        except (TypeError, ValueError):
            min_window_count = 0
        try:
            min_pixel_width = int(data.get("min_pixel_width", 10))
        except (TypeError, ValueError):
            min_pixel_width = 10
        try:
            min_pixel_length = int(data.get("min_pixel_length", 20))
        except (TypeError, ValueError):
            min_pixel_length = 20
        return cls(
            enabled=bool(data.get("enabled", False)),
            min_window_count=max(0, min_window_count),
            min_pixel_width=max(0, min_pixel_width),
            min_pixel_length=max(0, min_pixel_length),
        )

    @classmethod
    def load(cls, path: str) -> PixelMatchConfig:
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
