# -*- coding: utf-8 -*-
"""Gaussian spatial low-pass for camera BGR frames (production pipeline)."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

# Same effective upper bound as spatial_lowpass_test (strength 100 -> 8.3).
SIGMA_MAX = 8.3
_KERNEL_SIZE_MAX = 51


@dataclass
class GaussianLowpassConfig:
    """User-facing sigma in OpenCV pixels; 0 means filter off."""

    sigma: float = 0.0
    debug_mode: bool = False

    def enabled(self) -> bool:
        return float(self.sigma) > 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "sigma": float(self.sigma),
            "debug_mode": bool(self.debug_mode),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> GaussianLowpassConfig:
        if not data:
            return cls()
        try:
            sigma = float(data.get("sigma", 0.0))
        except (TypeError, ValueError):
            sigma = 0.0
        return cls(sigma=clamp_sigma(sigma), debug_mode=bool(data.get("debug_mode", False)))

    @classmethod
    def load(cls, path: str) -> GaussianLowpassConfig:
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


def clamp_sigma(sigma: float) -> float:
    return float(np.clip(float(sigma), 0.0, SIGMA_MAX))


def kernel_size_for_sigma(sigma: float) -> int:
    k = int(round(float(sigma) * 6.0)) | 1
    return max(3, min(k, _KERNEL_SIZE_MAX))


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


def apply_gaussian_lowpass(image_bgr: np.ndarray, sigma: float) -> np.ndarray:
    """
    Apply isotropic Gaussian blur when ``sigma > 0``; otherwise return input unchanged.

    ``sigma`` is in pixels (OpenCV ``GaussianBlur`` sigmaX/sigmaY).
    """
    src = _ensure_bgr_u8_contiguous(image_bgr)
    if src.size == 0:
        return src
    sig = clamp_sigma(sigma)
    if sig <= 0.0:
        return src.copy()
    k = kernel_size_for_sigma(sig)
    return cv2.GaussianBlur(src, (k, k), sigmaX=sig, sigmaY=sig)
