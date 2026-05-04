"""
Classical ROI extraction for bright, low-saturation regions (e.g. white label strips).

HSV mask (low S, high V) + morphology + either:
- largest inscribed rectangle in the binary mask (single stable crop), or
- connected components -> multiple bounding boxes (optional aspect filter).
"""

from __future__ import annotations

import cv2
import numpy as np


def _largest_inscribed_rectangle(mask01: np.ndarray) -> tuple[int, int, int, int] | None:
    """Histogram-stack largest rectangle in a binary mask. Returns (x, y, w, h) or None."""
    h, w = mask01.shape[:2]
    heights = [0] * w
    best_area = 0
    best_box = (0, 0, 0, 0)
    for y in range(h):
        row = mask01[y]
        for x in range(w):
            heights[x] = heights[x] + 1 if row[x] else 0

        stack: list[int] = []
        x = 0
        while x <= w:
            cur_h = heights[x] if x < w else 0
            if not stack or cur_h >= heights[stack[-1]]:
                stack.append(x)
                x += 1
                continue

            top = stack.pop()
            hh = heights[top]
            left = stack[-1] + 1 if stack else 0
            ww = x - left
            area = hh * ww
            if area > best_area:
                best_area = area
                best_box = (left, y - hh + 1, ww, hh)

    x1, y1, bw, bh = best_box
    if bw <= 1 or bh <= 1:
        return None
    return (int(x1), int(y1), int(bw), int(bh))


def build_uniform_mask(
    image_bgr: np.ndarray,
    roi_sat_max: int,
    roi_val_min: int,
    close_ksize: int = 15,
    open_ksize: int = 5,
    close_iterations: int = 2,
    open_iterations: int = 1,
) -> np.ndarray:
    """Binary mask 0/255: low saturation + high brightness."""
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]
    mask = ((s <= roi_sat_max) & (v >= roi_val_min)).astype(np.uint8) * 255
    ck = max(3, close_ksize | 1)
    ok = max(3, open_ksize | 1)
    close_k = np.ones((ck, ck), np.uint8)
    open_k = np.ones((ok, ok), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_k, iterations=close_iterations)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_k, iterations=open_iterations)
    return mask


def extract_uniform_roi(
    image_bgr: np.ndarray,
    roi_sat_max: int,
    roi_val_min: int,
    roi_min_area_ratio: float,
    roi_shrink_ratio: float,
    close_ksize: int = 15,
    open_ksize: int = 5,
) -> tuple[np.ndarray, tuple[int, int, int, int] | None]:
    """
    Largest inscribed rectangle inside the low-S high-V mask.
    Returns (mask_uint8, (x, y, w, h)) or (mask, None).
    """
    h, w = image_bgr.shape[:2]
    mask = build_uniform_mask(
        image_bgr, roi_sat_max, roi_val_min, close_ksize, open_ksize
    )
    mask01 = (mask > 0).astype(np.uint8)
    if mask01.sum() == 0:
        return mask, None

    best = _largest_inscribed_rectangle(mask01)
    if best is None:
        return mask, None

    x1, y1, bw, bh = best
    shrink_x = int(max(0, bw * roi_shrink_ratio))
    shrink_y = int(max(0, bh * roi_shrink_ratio))
    x1 += shrink_x
    y1 += shrink_y
    bw -= 2 * shrink_x
    bh -= 2 * shrink_y

    if (bw * bh) < (h * w * roi_min_area_ratio):
        return mask, None
    if bw <= 10 or bh <= 10:
        return mask, None
    return mask, (int(x1), int(y1), int(bw), int(bh))


def extract_uniform_components(
    image_bgr: np.ndarray,
    roi_sat_max: int,
    roi_val_min: int,
    min_area_ratio: float,
    close_ksize: int = 15,
    open_ksize: int = 5,
    min_aspect_w_over_h: float | None = None,
    max_boxes: int = 32,
) -> tuple[np.ndarray, list[tuple[int, int, int, int]]]:
    """
    Connected components on the same uniform mask; return sorted bounding boxes (area desc).
    Optional filter: width/height >= min_aspect_w_over_h (horizontal strips).
    """
    h, w = image_bgr.shape[:2]
    area_img = h * w
    min_area = max(1, int(area_img * min_area_ratio))
    mask = build_uniform_mask(
        image_bgr, roi_sat_max, roi_val_min, close_ksize, open_ksize
    )
    mask01 = (mask > 0).astype(np.uint8)
    if mask01.sum() == 0:
        return mask, []

    n_lab, labels, stats, _ = cv2.connectedComponentsWithStats(mask01, connectivity=8)
    boxes: list[tuple[int, int, int, int, int]] = []
    for lab in range(1, n_lab):
        x = int(stats[lab, cv2.CC_STAT_LEFT])
        y = int(stats[lab, cv2.CC_STAT_TOP])
        bw = int(stats[lab, cv2.CC_STAT_WIDTH])
        bh = int(stats[lab, cv2.CC_STAT_HEIGHT])
        a = int(stats[lab, cv2.CC_STAT_AREA])
        if a < min_area or bw < 2 or bh < 2:
            continue
        if min_aspect_w_over_h is not None and bh > 0:
            if (bw / bh) < min_aspect_w_over_h:
                continue
        boxes.append((a, x, y, bw, bh))

    boxes.sort(key=lambda t: -t[0])
    out = [(x, y, bw, bh) for _, x, y, bw, bh in boxes[:max_boxes]]
    return mask, out


def refine_strip_to_text_bbox(
    roi_bgr: np.ndarray,
    roi_bin_bgr: np.ndarray,
    *,
    horiz_frac: float = 0.028,
    vert_kernel_px: int = 3,
    close_iterations: int = 1,
    fg_thresh: int = 128,
    pad_frac: float = 0.018,
    pad_min_px: int = 4,
    min_fg_pixels: int = 80,
    max_inner_area_ratio: float = 0.992,
) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int, int]] | None:
    """
    Layer-2 crop: after layer-1 uniform strip + preprocess binarization, find a tight
    axis-aligned box around ink (dot-matrix text). Uses horizontal morphological closing
    on foreground so dotted glyphs merge into bands for a stable boundingRect.

    Expects roi_bin_bgr from preprocess_roi_for_ocr (text dark, background bright).

    Returns (cropped_color_roi, cropped_binary_bgr, (x,y,w,h) in layer-1 ROI coords)
    or None if refinement is unsafe / useless.
    """
    if roi_bgr.shape[:2] != roi_bin_bgr.shape[:2]:
        return None
    h, w = roi_bgr.shape[:2]
    if h < 8 or w < 8:
        return None

    gray = cv2.cvtColor(roi_bin_bgr, cv2.COLOR_BGR2GRAY)
    # preprocess_roi_for_ocr inverts so mean high -> background white.
    fg = gray < fg_thresh
    if int(fg.sum()) < min_fg_pixels:
        return None

    kw = max(9, int(w * horiz_frac))
    kh = max(3, vert_kernel_px | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kw, kh))
    mask_u8 = (fg.astype(np.uint8) * 255)
    closed = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel, iterations=close_iterations)
    n_lab, _, stats, _ = cv2.connectedComponentsWithStats((closed > 0).astype(np.uint8), connectivity=8)
    if n_lab <= 1:
        return None

    border_margin = max(2, int(min(h, w) * 0.01))
    min_comp_area = max(20, int(h * w * 0.00008))
    candidates: list[tuple[int, int, int, int, int]] = []
    for lab in range(1, n_lab):
        x = int(stats[lab, cv2.CC_STAT_LEFT])
        y = int(stats[lab, cv2.CC_STAT_TOP])
        bw = int(stats[lab, cv2.CC_STAT_WIDTH])
        bh = int(stats[lab, cv2.CC_STAT_HEIGHT])
        area = int(stats[lab, cv2.CC_STAT_AREA])
        if area < min_comp_area or bw < 4 or bh < 4:
            continue
        # Border-touching blobs are often strip edges / background artifacts.
        if (
            x <= border_margin
            or y <= border_margin
            or (x + bw) >= (w - border_margin)
            or (y + bh) >= (h - border_margin)
        ):
            continue
        candidates.append((area, x, y, bw, bh))

    if not candidates:
        return None
    candidates.sort(key=lambda t: -t[0])
    max_area = candidates[0][0]
    keep = [c for c in candidates[:24] if c[0] >= max(15, int(max_area * 0.03))]
    if not keep:
        keep = candidates[:8]

    x1 = min(x for _, x, _, _, _ in keep)
    y1 = min(y for _, _, y, _, _ in keep)
    x2 = max(x + bw for _, x, _, bw, _ in keep)
    y2 = max(y + bh for _, _, y, _, bh in keep)
    cw, ch = x2 - x1, y2 - y1
    if cw < 8 or ch < 8:
        return None

    pad_x = max(pad_min_px, int(cw * pad_frac))
    pad_y = max(pad_min_px, int(ch * pad_frac))
    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(w, x2 + pad_x)
    y2 = min(h, y2 + pad_y)

    iw, ih = x2 - x1, y2 - y1
    if iw < 16 or ih < 12:
        return None
    if (iw * ih) >= (h * w * max_inner_area_ratio):
        # Still almost full strip — refinement did not shrink meaningfully.
        return None

    crop_bgr = roi_bgr[y1:y2, x1:x2].copy()
    crop_bin = roi_bin_bgr[y1:y2, x1:x2].copy()
    return crop_bgr, crop_bin, (int(x1), int(y1), int(iw), int(ih))
