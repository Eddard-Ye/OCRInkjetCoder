# -*- coding: utf-8 -*-
"""HSV white-background segmentation for Hik camera OCR filtering."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

import cv2
import numpy as np

MIN_AREA_MAX = 500_000
MIN_AREA_SCALE = 100


@dataclass
class WhiteBgSegmentConfig:
    h_min: int = 0
    h_max: int = 179
    s_min: int = 0
    s_max: int = 30
    v_min: int = 100
    v_max: int = 255
    close_k: int = 15
    open_k: int = 5
    min_area: int = 5000
    enable_validation: bool = False
    enable_aux_overlay: bool = False

    def normalized(self) -> WhiteBgSegmentConfig:
        cfg = WhiteBgSegmentConfig(**asdict(self))
        cfg.h_min, cfg.h_max = min(cfg.h_min, cfg.h_max), max(cfg.h_min, cfg.h_max)
        cfg.s_min, cfg.s_max = min(cfg.s_min, cfg.s_max), max(cfg.s_min, cfg.s_max)
        cfg.v_min, cfg.v_max = min(cfg.v_min, cfg.v_max), max(cfg.v_min, cfg.v_max)
        cfg.close_k = clamp_close_k(cfg.close_k)
        cfg.open_k = clamp_open_k(cfg.open_k)
        cfg.min_area = clamp_min_area(cfg.min_area)
        cfg.enable_validation = bool(cfg.enable_validation)
        cfg.enable_aux_overlay = bool(cfg.enable_aux_overlay)
        return cfg

    def to_dict(self) -> dict[str, Any]:
        cfg = self.normalized()
        return asdict(cfg)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> WhiteBgSegmentConfig:
        if not data:
            return cls()
        cfg = cls(
            h_min=clamp_h(int(data.get("h_min", 0))),
            h_max=clamp_h(int(data.get("h_max", 179))),
            s_min=clamp_sv(int(data.get("s_min", 0))),
            s_max=clamp_sv(int(data.get("s_max", 30))),
            v_min=clamp_sv(int(data.get("v_min", 100))),
            v_max=clamp_sv(int(data.get("v_max", 255))),
            close_k=clamp_close_k(int(data.get("close_k", 15))),
            open_k=clamp_open_k(int(data.get("open_k", 5))),
            min_area=clamp_min_area(int(data.get("min_area", 5000))),
            enable_validation=bool(data.get("enable_validation", False)),
            enable_aux_overlay=bool(data.get("enable_aux_overlay", False)),
        )
        return cfg.normalized()

    @classmethod
    def load(cls, path: str) -> WhiteBgSegmentConfig:
        if os.path.isfile(path):
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
            json.dump(self.normalized().to_dict(), f, ensure_ascii=False, indent=2)


@dataclass
class WhiteBgBox:
    x: int
    y: int
    w: int
    h: int
    area: float

    def as_xyxy(self) -> list[int]:
        return [self.x, self.y, self.x + self.w, self.y + self.h]


@dataclass
class WhiteBgSegmentResult:
    mask: np.ndarray
    rect_xyxy: Optional[list[int]]
    primary_box: Optional[WhiteBgBox]
    boxes: list[WhiteBgBox] = field(default_factory=list)

    @property
    def found(self) -> bool:
        return self.rect_xyxy is not None


def clamp_h(value: int) -> int:
    return int(np.clip(int(value), 0, 179))


def clamp_sv(value: int) -> int:
    return int(np.clip(int(value), 0, 255))


def clamp_close_k(value: int) -> int:
    return int(np.clip(int(value), 0, 51))


def clamp_open_k(value: int) -> int:
    return int(np.clip(int(value), 0, 31))


def clamp_min_area(value: int) -> int:
    return int(np.clip(int(value), 0, MIN_AREA_MAX))


def min_area_from_slider(slider_value: int) -> int:
    return clamp_min_area(int(slider_value) * MIN_AREA_SCALE)


def min_area_to_slider(min_area: int) -> int:
    return int(np.clip(int(min_area) // MIN_AREA_SCALE, 0, MIN_AREA_MAX // MIN_AREA_SCALE))


def build_hsv_mask(img_bgr: np.ndarray, cfg: WhiteBgSegmentConfig) -> np.ndarray:
    cfg = cfg.normalized()
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    lower = np.array([cfg.h_min, cfg.s_min, cfg.v_min], dtype=np.uint8)
    upper = np.array([cfg.h_max, cfg.s_max, cfg.v_max], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)
    if cfg.close_k > 0:
        k = cfg.close_k | 1
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    if cfg.open_k > 0:
        k = cfg.open_k | 1
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return mask


def find_white_bg_boxes(mask: np.ndarray, min_area: int) -> list[WhiteBgBox]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes: list[WhiteBgBox] = []
    for cnt in contours:
        area = float(cv2.contourArea(cnt))
        if area < min_area:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        boxes.append(WhiteBgBox(x=x, y=y, w=w, h=h, area=area))
    boxes.sort(key=lambda b: b.area, reverse=True)
    return boxes


def segment_white_background(
    img_bgr: np.ndarray,
    cfg: WhiteBgSegmentConfig,
) -> WhiteBgSegmentResult:
    cfg = cfg.normalized()
    if img_bgr is None or img_bgr.size == 0:
        empty = np.zeros((1, 1), dtype=np.uint8)
        return WhiteBgSegmentResult(mask=empty, rect_xyxy=None, primary_box=None, boxes=[])

    mask = build_hsv_mask(img_bgr, cfg)
    boxes = find_white_bg_boxes(mask, cfg.min_area)
    primary = boxes[0] if boxes else None
    rect = primary.as_xyxy() if primary is not None else None
    return WhiteBgSegmentResult(
        mask=mask,
        rect_xyxy=rect,
        primary_box=primary,
        boxes=boxes,
    )


def ocr_box_inside_rect(box: dict[str, Any], rect_xyxy: list[int]) -> bool:
    poly = box.get("poly")
    if poly is None:
        return False
    pts = np.asarray(poly, dtype=np.float32).reshape(-1, 2)
    if pts.size == 0:
        return False
    x1, y1, x2, y2 = [float(v) for v in rect_xyxy]
    pb_x1 = float(np.min(pts[:, 0]))
    pb_y1 = float(np.min(pts[:, 1]))
    pb_x2 = float(np.max(pts[:, 0]))
    pb_y2 = float(np.max(pts[:, 1]))
    return pb_x1 >= x1 and pb_y1 >= y1 and pb_x2 <= x2 and pb_y2 <= y2


def filter_ocr_boxes_by_white_rect(
    boxes: list[dict[str, Any]],
    rect_xyxy: Optional[list[int]],
) -> tuple[list[dict[str, Any]], int]:
    if rect_xyxy is None:
        return [], len(boxes)
    kept: list[dict[str, Any]] = []
    rejected = 0
    for box in boxes:
        if ocr_box_inside_rect(box, rect_xyxy):
            kept.append(box)
        else:
            rejected += 1
    return kept, rejected


def apply_aux_segment_overlay(
    img_bgr: np.ndarray,
    seg: WhiteBgSegmentResult,
    *,
    mask_alpha: float = 0.28,
    image_alpha: float = 0.72,
) -> np.ndarray:
    if img_bgr is None or img_bgr.size == 0:
        return img_bgr
    vis = img_bgr.copy()
    if seg.mask is not None and seg.mask.size > 0 and seg.mask.shape[:2] == vis.shape[:2]:
        tint = np.zeros_like(vis)
        tint[:, :, 1] = seg.mask
        vis = cv2.addWeighted(vis, image_alpha, tint, mask_alpha, 0)
    if seg.rect_xyxy is not None:
        x1, y1, x2, y2 = seg.rect_xyxy
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 255), 2)
        if seg.primary_box is not None:
            label = f"white area={int(seg.primary_box.area)}"
            cv2.putText(
                vis,
                label,
                (x1, max(y1 - 8, 16)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
    return vis


def draw_white_rect_on_bgr(
    img_bgr: np.ndarray,
    rect_xyxy: Optional[list[int]],
    *,
    color: tuple[int, int, int] = (0, 255, 255),
    thickness: int = 2,
) -> np.ndarray:
    if img_bgr is None or img_bgr.size == 0 or rect_xyxy is None:
        return img_bgr
    vis = img_bgr.copy()
    x1, y1, x2, y2 = rect_xyxy
    cv2.rectangle(vis, (x1, y1), (x2, y2), color, thickness)
    return vis
