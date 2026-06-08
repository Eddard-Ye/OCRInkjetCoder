# -*- coding: utf-8 -*-
"""Grayscale morphological close for camera BGR frames (production pipeline)."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

STRENGTH_MAX = 100.0
_LEGACY_MEDIAN_CONFIG_BASENAME = "hik_camera_ui_median_lowpass.json"


@dataclass
class MorphCloseLowpassConfig:
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
    def from_dict(cls, data: dict[str, Any] | None) -> MorphCloseLowpassConfig:
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
    def load(cls, path: str) -> MorphCloseLowpassConfig:
        if os.path.isfile(path):
            try:
                with open(path, encoding="utf-8") as f:
                    raw = json.load(f)
                if isinstance(raw, dict):
                    return cls.from_dict(raw)
            except Exception:
                pass
        legacy = os.path.join(os.path.dirname(path), _LEGACY_MEDIAN_CONFIG_BASENAME)
        if legacy != path and os.path.isfile(legacy):
            try:
                with open(legacy, encoding="utf-8") as f:
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


def morph_close_kernel_from_strength(strength: float) -> int:
    """Closing kernel size (same mapping as ``spatial_lowpass_test`` morph_close)."""
    s = float(np.clip(float(strength) / 100.0, 0.0, 1.0))
    return max(3, int(round(3 + s * 10.0)) | 1)


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


def apply_morph_close_lowpass(image_bgr: np.ndarray, strength: float) -> np.ndarray:
    """
    Grayscale morphological close when ``strength > 0``; otherwise return input unchanged.

    Same as offline ``spatial_lowpass_test`` ``morph_close`` (optional pre-blur, then close).
    """
    src = _ensure_bgr_u8_contiguous(image_bgr)
    if src.size == 0:
        return src
    st = clamp_strength(strength)
    if st <= 0.0:
        return src.copy()
    s = float(st / 100.0)
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    if s > 0.05:
        gk = max(3, int(round(3 + s * 4)) | 1)
        gray = cv2.GaussianBlur(gray, (gk, gk), 0)
    ks = morph_close_kernel_from_strength(st)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ks, ks))
    closed = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
    return cv2.cvtColor(closed, cv2.COLOR_GRAY2BGR)
