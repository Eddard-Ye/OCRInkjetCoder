# -*- coding: utf-8 -*-
"""Median spatial filter for camera BGR frames (production pipeline)."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

STRENGTH_MAX = 100.0
_KERNEL_SIZE_MIN = 3
_KERNEL_SIZE_MAX = 9


@dataclass
class MedianLowpassConfig:
    """User-facing strength 0-100; requires ``enabled`` and strength > 0 to apply."""

    enabled: bool = False
    strength: float = 0.0

    def active(self) -> bool:
        return bool(self.enabled) and float(self.strength) > 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": bool(self.enabled),
            "strength": float(self.strength),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> MedianLowpassConfig:
        if not data:
            return cls()
        try:
            strength = float(data.get("strength", 0.0))
        except (TypeError, ValueError):
            strength = 0.0
        return cls(
            enabled=bool(data.get("enabled", False)),
            strength=clamp_strength(strength),
        )

    @classmethod
    def load(cls, path: str) -> MedianLowpassConfig:
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


def clamp_strength(strength: float) -> float:
    return float(np.clip(float(strength), 0.0, STRENGTH_MAX))


def kernel_size_from_strength(strength: float) -> int:
    """Map UI strength 0-100 to odd median kernel size (same as spatial_lowpass_test)."""
    s = float(np.clip(float(strength) / 100.0, 0.0, 1.0))
    k = int(round(_KERNEL_SIZE_MIN + s * 6.0)) | 1
    return max(_KERNEL_SIZE_MIN, min(k, _KERNEL_SIZE_MAX))


def _ensure_bgr_u8_contiguous(image_bgr: np.ndarray) -> np.ndarray:
    img = np.asarray(image_bgr)
    if img.size == 0:
        return img
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.ndim == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(img)


def apply_median_lowpass(image_bgr: np.ndarray, strength: float) -> np.ndarray:
    """
    Apply ``cv2.medianBlur`` when ``strength > 0``; otherwise return input unchanged.

    ``strength`` is 0-100 (mapped to kernel 3x3 .. 9x9, same as offline test tool).
    """
    src = _ensure_bgr_u8_contiguous(image_bgr)
    if src.size == 0:
        return src
    st = clamp_strength(strength)
    if st <= 0.0:
        return src.copy()
    k = kernel_size_from_strength(st)
    return cv2.medianBlur(src, k)
