#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from paddleocr import PaddleOCR

from cv_roi import extract_uniform_roi, refine_strip_to_text_bbox


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def _log_preview(text: str, max_len: int = 120) -> str:
    t = text.replace("\n", " ").strip()
    if len(t) <= max_len:
        return t
    return t[: max_len - 1] + "…"


@dataclass
class OCRRecord:
    file_name: str
    region_file: str
    line_count: int
    text: str
    normalized_text: str
    production_date: str
    normal_expiry: str
    frozen_expiry: str
    confidence: float
    elapsed_seconds: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract a large uniform-color region, then segment text and run OCR."
    )
    parser.add_argument(
        "--input-dir",
        default="data",
        type=Path,
        help="Input image folder. Default: data",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs",
        type=Path,
        help="Output folder for masks/crops/results. Default: outputs",
    )
    parser.add_argument("--roi-sat-max", type=int, default=70, help="Max saturation for uniform ROI.")
    parser.add_argument("--roi-val-min", type=int, default=95, help="Min brightness for uniform ROI.")
    parser.add_argument(
        "--roi-min-area-ratio",
        type=float,
        default=0.12,
        help="Min ROI area ratio in full image.",
    )
    parser.add_argument(
        "--roi-shrink-ratio",
        type=float,
        default=0.04,
        help="Inward shrink ratio to remove noisy borders and reduce OCR area.",
    )
    parser.add_argument(
        "--ocr-min-conf",
        type=float,
        default=0.55,
        help="Minimum confidence to keep OCR line.",
    )
    parser.add_argument(
        "--ocr-upscale",
        type=float,
        default=1.6,
        help="Upscale ratio for OCR enhanced branch.",
    )
    parser.add_argument(
        "--ocr-mode",
        choices=["fast", "high_accuracy"],
        default="fast",
        help="fast: raw+one enhanced pass. high_accuracy: multi-scale + rotation, pick best by score (slower).",
    )
    parser.add_argument(
        "--ocr-model-size",
        choices=["auto", "mobile", "server"],
        default="auto",
        help="OCR model size. auto: mobile in fast, server in high_accuracy.",
    )
    parser.add_argument(
        "--ocr-lang",
        default="ch",
        help="PaddleOCR language. Example: ch / en / latin",
    )
    parser.add_argument(
        "--det-limit-side-len",
        type=int,
        default=2560,
        help="OCR text detector input side limit. Larger keeps tiny dot-matrix details (slower).",
    )
    parser.add_argument(
        "--det-thresh",
        type=float,
        default=0.2,
        help="Text detector pixel threshold (lower -> more sensitive).",
    )
    parser.add_argument(
        "--det-box-thresh",
        type=float,
        default=0.35,
        help="Text detector box confidence threshold (lower -> keep more boxes).",
    )
    parser.add_argument(
        "--det-unclip-ratio",
        type=float,
        default=2.0,
        help="Text detector unclip ratio. Higher expands detected boxes.",
    )
    parser.add_argument(
        "--use-angle-cls",
        action="store_true",
        default=True,
        help="Enable angle classification for rotated text.",
    )
    parser.add_argument(
        "--ocr-verbose",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Log each OCR branch / scale+rotation (default: on). Use --no-ocr-verbose to reduce output.",
    )
    parser.add_argument(
        "--input-upright",
        action="store_true",
        help="Input photos are already correctly oriented. In high_accuracy, only try rot=0 (no extra 90°/180°/270° passes); faster. Textline angle model still on unless you disable it.",
    )
    parser.add_argument(
        "--no-debug-vis",
        action="store_true",
        help="Disable auxiliary debug image outputs (preprocessed image and det visualization).",
    )
    parser.add_argument(
        "--hi-acc-scales",
        type=str,
        default="1.0",
        help="high_accuracy only: comma-separated scale factors. 1.0 = original ROI (no CLAHE upscale). "
        "Example: 1.0 or 1.0,1.4,1.7,1.9 for multi-scale voting.",
    )
    parser.add_argument(
        "--no-text-inner-crop",
        action="store_true",
        help="Disable layer-2 crop: OCR uses full uniform strip after preprocess (like older behavior). "
        "Default: refine to tight bbox around ink on binarized strip.",
    )
    parser.add_argument(
        "--text-inner-horiz-frac",
        type=float,
        default=0.028,
        help="Layer-2: horizontal closing kernel width ~ frac * strip width (dot-matrix bridge).",
    )
    parser.add_argument(
        "--text-inner-pad-frac",
        type=float,
        default=0.018,
        help="Layer-2: pad around ink bbox as fraction of box width/height.",
    )
    parser.add_argument(
        "--no-det-vis-merge",
        action="store_true",
        help="Do not merge adjacent detector fragments in *_det_vis.png (show raw Paddle boxes).",
    )
    return parser.parse_args()


def parse_hi_acc_scales(s: str) -> tuple[float, ...]:
    out: list[float] = []
    for part in s.split(","):
        p = part.strip()
        if not p:
            continue
        v = float(p)
        if v <= 0:
            raise ValueError(f"hi-acc-scales must be > 0, got {v}")
        out.append(v)
    if not out:
        raise ValueError("hi-acc-scales: need at least one value")
    return tuple(out)


def iter_images(folder: Path) -> Iterable[Path]:
    for p in sorted(folder.rglob("*")):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            yield p


def build_ocr_engine(
    ocr_lang: str,
    use_angle_cls: bool,
    ocr_model_size: str,
    det_limit_side_len: int,
    det_thresh: float,
    det_box_thresh: float,
    det_unclip_ratio: float,
    device: str | None = None,
) -> PaddleOCR:
    det_model = f"PP-OCRv5_{ocr_model_size}_det"
    rec_model = f"PP-OCRv5_{ocr_model_size}_rec"
    kwargs: dict = {}
    if device is not None:
        kwargs["device"] = device
        kwargs["use_gpu"] = device.startswith("gpu")
    return PaddleOCR(
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=use_angle_cls,
        text_detection_model_name=det_model,
        text_recognition_model_name=rec_model,
        text_det_limit_side_len=det_limit_side_len,
        text_det_limit_type="max",
        text_det_thresh=det_thresh,
        text_det_box_thresh=det_box_thresh,
        text_det_unclip_ratio=det_unclip_ratio,
        lang=ocr_lang,
        **kwargs,
    )


def ocr_roi_lines(
    roi_bgr: np.ndarray, ocr_engine: PaddleOCR
) -> list[tuple[str, float, tuple[int, int, int, int]]]:
    if hasattr(ocr_engine, 'predict'):
        result = ocr_engine.predict(roi_bgr)
        first = result[0] if result else {}
        texts = first.get("rec_texts", []) if hasattr(first, "get") else []
        scores = first.get("rec_scores", []) if hasattr(first, "get") else []
        polys = first.get("rec_polys", []) if hasattr(first, "get") else []
        if len(polys) == 0 and hasattr(first, "get"):
            polys = first.get("rec_boxes", [])
    else:
        result = ocr_engine.ocr(roi_bgr, cls=True)
        texts = []
        scores = []
        polys = []
        if result and len(result) > 0 and result[0]:
            for line in result[0]:
                if len(line) >= 2:
                    box = line[0]
                    text = line[1][0]
                    score = line[1][1]
                    texts.append(text)
                    scores.append(score)
                    polys.append(box)

    lines: list[tuple[str, float, tuple[int, int, int, int]]] = []
    n = min(len(texts), len(scores), len(polys))
    for i in range(n):
        txt = str(texts[i]).strip()
        conf = float(scores[i])
        if not txt:
            continue
        poly = np.asarray(polys[i])
        if poly.size == 0:
            continue
        if poly.ndim == 1:
            if poly.shape[0] >= 4:
                x1, y1, x2, y2 = float(poly[0]), float(poly[1]), float(poly[2]), float(poly[3])
            else:
                continue
        else:
            xs = poly[:, 0].astype(np.float32)
            ys = poly[:, 1].astype(np.float32)
            x1, y1, x2, y2 = float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())
        x = int(max(0, x1))
        y = int(max(0, y1))
        w = int(max(1, x2 - x1))
        h = int(max(1, y2 - y1))
        lines.append((txt, conf, (x, y, w, h)))
    return lines


def merge_fragment_det_boxes_for_vis(
    lines: list[tuple[str, float, tuple[int, int, int, int]]],
    roi_hw: tuple[int, int],
) -> list[tuple[str, float, tuple[int, int, int, int]]]:
    """
    Merge adjacent detector fragments on the same text row into one box (visualization only).

    Dot-matrix inkjet often splits one logical line into two boxes (e.g. label vs date).
    """
    if len(lines) <= 1:
        return lines

    _, iw = roi_hw[0], roi_hw[1]

    def cy_center(box: tuple[int, int, int, int]) -> float:
        _x, y, _bw, bh = box
        return y + bh * 0.5

    sorted_lines = sorted(lines, key=lambda t: (cy_center(t[2]), t[2][0]))

    rows: list[list[tuple[str, float, tuple[int, int, int, int]]]] = []
    for item in sorted_lines:
        cc = cy_center(item[2])
        if not rows:
            rows.append([item])
            continue
        last_in_row = rows[-1][-1][2]
        rh = float(last_in_row[3])
        prev_cy = cy_center(last_in_row)
        row_tol = max(9.0, 0.48 * rh)
        if abs(cc - prev_cy) <= row_tol:
            rows[-1].append(item)
        else:
            rows.append([item])

    out: list[tuple[str, float, tuple[int, int, int, int]]] = []
    for row in rows:
        row_sorted = sorted(row, key=lambda t: t[2][0])
        cur_txt, cur_conf, cur_box = row_sorted[0]
        cur_x, cur_y, cur_w, cur_h = cur_box
        conf_sum = float(cur_conf)
        conf_cnt = 1

        for next_txt, next_conf, nb in row_sorted[1:]:
            nx, ny, nw, nh = nb
            gap = nx - (cur_x + cur_w)
            mh = max(cur_h, nh, 1)
            oy = max(0, min(cur_y + cur_h, ny + nh) - max(cur_y, ny))
            overlap_ratio = float(oy / min(cur_h, nh)) if min(cur_h, nh) > 0 else 0.0
            gap_tol = max(16, int(0.024 * iw), int(0.42 * mh))
            if gap <= gap_tol and overlap_ratio >= 0.28:
                rx2 = max(cur_x + cur_w, nx + nw)
                ry2 = max(cur_y + cur_h, ny + nh)
                cur_x = min(cur_x, nx)
                cur_y = min(cur_y, ny)
                cur_w = rx2 - cur_x
                cur_h = ry2 - cur_y
                cur_txt = cur_txt + next_txt
                conf_sum += float(next_conf)
                conf_cnt += 1
                cur_conf = conf_sum / float(conf_cnt)
                cur_box = (cur_x, cur_y, cur_w, cur_h)
                continue
            out.append((cur_txt, cur_conf, cur_box))
            cur_txt, cur_conf, cur_box = next_txt, next_conf, nb
            cur_x, cur_y, cur_w, cur_h = nb
            conf_sum = float(next_conf)
            conf_cnt = 1

        out.append((cur_txt, cur_conf, cur_box))

    return out


def save_det_debug_outputs(
    roi_bgr: np.ndarray,
    lines: list[tuple[str, float, tuple[int, int, int, int]]],
    regions_dir: Path,
    stem: str,
    suffix: str = "uniform_region_det_vis",
) -> None:
    vis = roi_bgr.copy()
    h, w = vis.shape[:2]
    for i, (txt, conf, (x, y, bw, bh)) in enumerate(lines, start=1):
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(w - 1, x + bw)
        y2 = min(h - 1, y + bh)
        # Visual tweak only: shrink height a little, keep full width.
        sh = max(1, int((y2 - y1) * 0.12))
        if (y2 - y1) > 8:
            y1 = min(y2 - 1, y1 + sh)
            y2 = max(y1 + 1, y2 - sh)
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.imwrite(str(regions_dir / f"{stem}_{suffix}.png"), vis)


def preprocess_roi_for_ocr(
    roi_bgr: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Fixed preprocessing pipeline (same idea as debug image #07):
    1) grayscale
    2) large-blur background normalization (divide)
    3) CLAHE local contrast enhancement
    4) light median denoise
    5) Otsu binarization
    Returns (gray, bg_norm, clahe, median, otsu_bgr_for_ocr).
    """
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    bg = cv2.GaussianBlur(gray, (0, 0), sigmaX=21, sigmaY=21)
    bg_norm = cv2.divide(gray, bg, scale=255)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(bg_norm)
    median = cv2.medianBlur(clahe, 3)
    _, otsu = cv2.threshold(median, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if float(np.mean(otsu)) < 127.0:
        otsu = 255 - otsu
    return gray, bg_norm, clahe, median, cv2.cvtColor(otsu, cv2.COLOR_GRAY2BGR)


def enhance_roi_for_ocr(roi_bgr: np.ndarray, scale: float) -> np.ndarray:
    h, w = roi_bgr.shape[:2]
    nh = max(16, int(h * scale))
    nw = max(16, int(w * scale))
    up = cv2.resize(roi_bgr, (nw, nh), interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(up, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    eq = clahe.apply(gray)
    blur = cv2.GaussianBlur(eq, (0, 0), 1.1)
    sharp = cv2.addWeighted(eq, 1.35, blur, -0.35, 0)
    return cv2.cvtColor(sharp, cv2.COLOR_GRAY2BGR)


def resolve_ocr_model_size(ocr_mode: str, ocr_model_size: str) -> str:
    if ocr_model_size == "auto":
        return "server" if ocr_mode == "high_accuracy" else "mobile"
    return ocr_model_size


def rotate_quarters(roi_bgr: np.ndarray, quarters: int) -> np.ndarray:
    out = roi_bgr
    for _ in range(quarters % 4):
        out = cv2.rotate(out, cv2.ROTATE_90_CLOCKWISE)
    return out


def prepare_roi_at_scale(roi_bgr: np.ndarray, scale: float) -> np.ndarray:
    if scale <= 1.001:
        return roi_bgr
    return enhance_roi_for_ocr(roi_bgr, scale)


def score_ocr_candidate(merged_text: str, avg_conf: float, line_count: int) -> float:
    t = merged_text.replace(" ", "")
    s = 0.0
    if "生产日期" in t:
        s += 2.0
    if "常温储存保质期至" in t or ("常温" in t and "保质期" in t):
        s += 1.5
    if "冷冻储存保质期至" in t or ("冷冻" in t and "保质期" in t):
        s += 1.5
    if re.search(r"20\d{2}.*\d{1,2}.*\d{1,2}", t):
        s += 0.6
    s += float(avg_conf) * 1.2
    s += min(line_count, 12) * 0.07
    if not t.strip():
        s -= 5.0
    return s


def score_lines_branch(lines: list[tuple[str, float, tuple[int, int, int, int]]]) -> float:
    if not lines:
        return 0.0
    texts = [t for t, _, _ in lines]
    conf = float(np.mean([c for _, c, _ in lines]))
    keyword_hits = 0
    for t in texts:
        if "生产日期" in t:
            keyword_hits += 1
        if "常温储存保质期至" in t:
            keyword_hits += 1
        if "冷冻储存保质期至" in t:
            keyword_hits += 1
    return conf + keyword_hits * 0.35 + len(lines) * 0.03


def pick_better_lines(
    raw_lines: list[tuple[str, float, tuple[int, int, int, int]]],
    enhanced_lines: list[tuple[str, float, tuple[int, int, int, int]]],
) -> list[tuple[str, float, tuple[int, int, int, int]]]:
    return (
        enhanced_lines
        if score_lines_branch(enhanced_lines) >= score_lines_branch(raw_lines)
        else raw_lines
    )


def run_ocr_fast(
    roi_bgr: np.ndarray,
    ocr_engine: PaddleOCR,
    ocr_upscale: float,
    ocr_min_conf: float,
    log_prefix: str = "",
    verbose: bool = True,
) -> tuple[str, float, int, list[tuple[str, float, tuple[int, int, int, int]]]]:
    raw_lines = ocr_roi_lines(roi_bgr, ocr_engine)
    merged, conf, n = merge_lines_to_text(raw_lines, ocr_min_conf)
    if verbose:
        print(
            f"{log_prefix}  [ocr fast] single-pass raw: det_lines={len(raw_lines)} "
            f"merged_lines={n} conf={conf:.3f}"
        )
        print(f"{log_prefix}  [ocr fast] text: {_log_preview(merged)}")
    return merged, conf, n, raw_lines


def run_ocr_high_accuracy(
    roi_bgr: np.ndarray,
    ocr_engine: PaddleOCR,
    ocr_min_conf: float,
    log_prefix: str = "",
    verbose: bool = True,
    input_upright: bool = False,
    hi_acc_scales: tuple[float, ...] = (1.0,),
) -> tuple[str, float, int, str]:
    """Multi-scale; optional 4-way rotation. Return (text, conf, line_count, debug_tag)."""
    # Dot-matrix inkjet text often has uneven per-fragment confidence.
    # Use a softer keep threshold in high-accuracy mode to reduce dropped fragments.
    min_c = max(0.30, ocr_min_conf * 0.75)
    best: tuple[str, float, int, float, int] | None = None
    best_score = -1e9
    trial = 0
    rots = (0,) if input_upright else (0, 1, 2, 3)
    scales = hi_acc_scales
    total_trials = len(scales) * len(rots)
    rh, rw = roi_bgr.shape[:2]
    if verbose:
        print(
            f"{log_prefix}  [ocr hi] ROI {rw}x{rh} px, {total_trials} forward passes. "
            f"First infer can take several minutes on CPU; not frozen.",
            flush=True,
        )
    for sc in scales:
        base = prepare_roi_at_scale(roi_bgr, sc)
        for r in rots:
            trial += 1
            img = rotate_quarters(base, r)
            if verbose:
                print(
                    f"{log_prefix}  [ocr hi] trial {trial:02d}/{total_trials} running det+rec "
                    f"(scale={sc:.2f}, rot90x={r}, img {img.shape[1]}x{img.shape[0]})...",
                    flush=True,
                )
            det_lines = ocr_roi_lines(img, ocr_engine)
            merged, conf, n = merge_lines_to_text(det_lines, min_c)
            scv = score_ocr_candidate(merged, conf, n)
            is_best = scv > best_score
            if is_best:
                best_score = scv
                best = (merged, conf, n, sc, r)
            if verbose:
                mark = "*" if is_best else " "
                print(
                    f"{log_prefix}  [ocr hi {trial:02d}/{total_trials}]{mark} scale={sc:.2f} rot90x{r} "
                    f"det={len(det_lines)} kept={n} conf={conf:.3f} candidate_score={scv:.3f}",
                    flush=True,
                )
                print(f"{log_prefix}      text: {_log_preview(merged)}", flush=True)
    if best and best[0].strip():
        sc, r = best[3], best[4]
        tag = f"scale={sc:.2f} rot_90x{r} score={best_score:.2f}"
        if verbose:
            print(
                f"{log_prefix}  [ocr hi] BEST candidate_score={best_score:.3f} "
                f"(min_conf_used={min_c:.2f}) -> {tag}",
                flush=True,
            )
            print(f"{log_prefix}  [ocr hi] final text: {_log_preview(best[0])}", flush=True)
        return best[0], best[1], best[2], tag
    lines = ocr_roi_lines(roi_bgr, ocr_engine)
    merged, conf, n = merge_lines_to_text(lines, ocr_min_conf)
    if verbose:
        print(f"{log_prefix}  [ocr hi] fallback to raw ROI (no good candidate)", flush=True)
        print(f"{log_prefix}  [ocr hi] text: {_log_preview(merged)}", flush=True)
    return merged, conf, n, "fallback_raw"


def merge_lines_to_text(
    lines: list[tuple[str, float, tuple[int, int, int, int]]], ocr_min_conf: float
) -> tuple[str, float, int]:
    kept: list[tuple[str, float, tuple[int, int, int, int]]] = []
    for txt, conf, box in lines:
        if conf < ocr_min_conf:
            continue
        # Drop likely OCR noise such as single punctuation.
        if len(txt.strip()) <= 1 and conf < 0.8:
            continue
        kept.append((txt, conf, box))
    if not kept and lines:
        # Safety net: detector found text boxes but threshold filtered all.
        # Keep top-confidence non-empty lines so output is not blank.
        backup = [(txt, conf, box) for txt, conf, box in lines if txt.strip() and conf >= 0.2]
        backup.sort(key=lambda item: item[1], reverse=True)
        kept = backup[: min(6, len(backup))]
    if not kept:
        return "", 0.0, 0

    # Read order heuristic: vertical print is usually right-to-left by columns.
    vertical_ratio = float(np.mean([1.0 if b[3] > b[2] * 1.25 else 0.0 for _, _, b in kept]))
    if vertical_ratio >= 0.5:
        kept.sort(key=lambda item: (-item[2][0], item[2][1]))
    else:
        # Horizontal text: first cluster by row (tolerate y-jitter), then sort each row by x.
        heights = [b[3] for _, _, b in kept]
        row_tol = max(6, int(np.median(heights) * 0.6))
        by_center = sorted(kept, key=lambda item: item[2][1] + item[2][3] * 0.5)
        rows: list[list[tuple[str, float, tuple[int, int, int, int]]]] = []
        row_anchors: list[float] = []
        for item in by_center:
            cy = item[2][1] + item[2][3] * 0.5
            if not rows:
                rows.append([item])
                row_anchors.append(cy)
                continue
            if abs(cy - row_anchors[-1]) <= row_tol:
                rows[-1].append(item)
                # Keep row anchor stable but adaptive to mild drift.
                row_anchors[-1] = row_anchors[-1] * 0.7 + cy * 0.3
            else:
                rows.append([item])
                row_anchors.append(cy)
        kept = []
        for row in rows:
            row.sort(key=lambda item: item[2][0])
            kept.extend(row)
    merged_text = ", ".join([item[0] for item in kept])
    avg_conf = float(np.mean([item[1] for item in kept]))
    return merged_text, avg_conf, len(kept)


def normalize_datetime(raw: str) -> str:
    s = re.sub(r"[^0-9:/]", "", raw)
    if not s:
        return ""
    m = re.search(r"(20\d{2})\D*(\d{1,2})\D*(\d{1,2})(?:\D*(\d{1,2})\D*(\d{1,2}))?", s)
    if not m:
        return ""
    year = m.group(1)
    month = m.group(2).zfill(2)
    day = m.group(3).zfill(2)
    hh = m.group(4)
    mm = m.group(5)
    if hh and mm:
        return f"{year}/{month}/{day} {hh.zfill(2)}:{mm.zfill(2)}"
    return f"{year}/{month}/{day}"


def extract_structured_fields(text: str) -> tuple[str, str, str, str]:
    compact = text.replace(" ", "")
    production = ""
    normal_expiry = ""
    frozen_expiry = ""

    m_prod = re.search(r"生产日期[:：]?(.*?)(?:,|$)", compact)
    if m_prod:
        production = normalize_datetime(m_prod.group(1))
    else:
        # fallback: first datetime with time
        m_any_prod = re.search(r"(20\d{2}\D*\d{1,2}\D*\d{1,2}\D*\d{1,2}\D*\d{1,2})", compact)
        if m_any_prod:
            production = normalize_datetime(m_any_prod.group(1))

    # Tolerate OCR confusions like "常温/南温/温储存保质期至".
    m_normal = re.search(r"(?:常温|南温|温)储存保质期至[:：]?(.*?)(?:,|$)", compact)
    if m_normal:
        normal_expiry = normalize_datetime(m_normal.group(1))

    m_frozen = re.search(r"冷冻储存保质期至[:：]?(.*?)(?:,|$)", compact)
    if m_frozen:
        frozen_expiry = normalize_datetime(m_frozen.group(1))

    # fallback for date-only values if one field missing
    all_dates = re.findall(r"20\d{2}\D*\d{1,2}\D*\d{1,2}", compact)
    parsed_dates = [normalize_datetime(x) for x in all_dates if normalize_datetime(x)]
    parsed_dates = list(dict.fromkeys(parsed_dates))
    # Exclude production date from expiry candidates when available.
    if production:
        parsed_dates = [d for d in parsed_dates if d != production]
    if not normal_expiry and len(parsed_dates) >= 1:
        normal_expiry = parsed_dates[0]
    if not frozen_expiry and len(parsed_dates) >= 2:
        frozen_expiry = parsed_dates[1]

    parts: list[str] = []
    if production:
        parts.append(f"生产日期：{production}")
    if normal_expiry:
        parts.append(f"常温储存保质期至：{normal_expiry}")
    if frozen_expiry:
        parts.append(f"冷冻储存保质期至：{frozen_expiry}")
    normalized_text = ", ".join(parts)
    return normalized_text, production, normal_expiry, frozen_expiry


def save_results(records: list[OCRRecord], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "ocr_results.json"
    csv_path = output_dir / "ocr_results.csv"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump([asdict(r) for r in records], f, ensure_ascii=False, indent=2)

    fieldnames = list(OCRRecord.__annotations__.keys())
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in records:
            writer.writerow(asdict(r))

    xlsx_path = output_dir / "ocr_results.xlsx"
    try:
        from openpyxl import Workbook

        wb = Workbook()
        ws = wb.active
        ws.title = "ocr_results"
        ws.append(fieldnames)
        for r in records:
            row = [asdict(r)[k] for k in fieldnames]
            ws.append(row)
        wb.save(xlsx_path)
    except ImportError:
        print(
            "Warning: openpyxl not installed; skipped ocr_results.xlsx. "
            "Install with: pip install openpyxl"
        )


def run_pipeline(args: argparse.Namespace) -> None:
    input_dir = args.input_dir
    output_dir = args.output_dir
    regions_dir = output_dir / "regions"
    regions_dir.mkdir(parents=True, exist_ok=True)

    try:
        hi_scales = parse_hi_acc_scales(args.hi_acc_scales)
    except ValueError as e:
        print(f"Invalid --hi-acc-scales: {e}")
        return

    records: list[OCRRecord] = []
    model_size = resolve_ocr_model_size(args.ocr_mode, args.ocr_model_size)
    ocr_engine = build_ocr_engine(
        args.ocr_lang,
        args.use_angle_cls,
        model_size,
        args.det_limit_side_len,
        args.det_thresh,
        args.det_box_thresh,
        args.det_unclip_ratio,
    )
    images = list(iter_images(input_dir))
    if not images:
        print(f"No images found in: {input_dir}")
        return

    try:
        _run_images_loop(images, regions_dir, args, model_size, ocr_engine, hi_scales, records)
    except KeyboardInterrupt:
        print(
            "\nStopped: KeyboardInterrupt (SIGINT). "
            "Same as Ctrl+C — also sent if an IDE/tool stops the terminal job or the session ends.",
            flush=True,
        )
        if records:
            save_results(records, output_dir)
            print(
                f"Partial results saved: {len(records)} row(s) -> {output_dir / 'ocr_results.csv'}",
                flush=True,
            )
        print(
            "Tip: large ROI + server model on CPU can run many minutes per image; "
            "try '--ocr-model-size mobile' for faster runs.",
            flush=True,
        )
        sys.exit(130)


def _run_images_loop(
    images: list[Path],
    regions_dir: Path,
    args: argparse.Namespace,
    model_size: str,
    ocr_engine: PaddleOCR,
    hi_scales: tuple[float, ...],
    records: list[OCRRecord],
) -> None:
    output_dir = args.output_dir
    for img_path in images:
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"Skip unreadable image: {img_path}")
            continue

        t0 = time.perf_counter()
        t_seg0 = time.perf_counter()
        uniform_mask, roi_box = extract_uniform_roi(
            image,
            args.roi_sat_max,
            args.roi_val_min,
            args.roi_min_area_ratio,
            args.roi_shrink_ratio,
        )
        t_seg = time.perf_counter() - t_seg0
        if roi_box is None:
            elapsed = time.perf_counter() - t0
            print(
                f"[{img_path.name}] no large uniform-color region found. (elapsed {elapsed:.2f}s)"
            )
            continue

        x0, y0, rw, rh = roi_box
        print(f"[{img_path.name}] uniform_roi=({x0}, {y0}, {rw}, {rh})")
        cv2.imwrite(str(regions_dir / f"{img_path.stem}_uniform_mask.png"), uniform_mask)
        roi = image[y0 : y0 + rh, x0 : x0 + rw]
        region_file = f"{img_path.stem}_uniform_region.png"
        cv2.imwrite(str(regions_dir / region_file), roi)
        lp = f"[{img_path.name}]"
        t_pre0 = time.perf_counter()
        _, _, _, _, roi_ocr = preprocess_roi_for_ocr(roi)
        t_pre = time.perf_counter() - t_pre0
        if not args.no_debug_vis:
            # Layer 1: full strip binarization (same as before).
            cv2.imwrite(str(regions_dir / f"{img_path.stem}_uniform_region_preproc_bin.png"), roi_ocr)
            print(
                f"{lp} preprocessing: gray -> bg_norm(divide) -> CLAHE -> median3 -> Otsu "
                f"(saved *_preproc_bin.png)",
                flush=True,
            )
        else:
            print(
                f"{lp} preprocessing: gray -> bg_norm(divide) -> CLAHE -> median3 -> Otsu",
                flush=True,
            )

        region_file = f"{img_path.stem}_uniform_region.png"
        roi_for_ocr = roi_ocr
        inner_meta = ""
        if not args.no_text_inner_crop:
            inner = refine_strip_to_text_bbox(
                roi,
                roi_ocr,
                horiz_frac=args.text_inner_horiz_frac,
                pad_frac=args.text_inner_pad_frac,
            )
            if inner is not None:
                roi_inner_bgr, roi_inner_bin, (ix, iy, iw, ih) = inner
                roi_for_ocr = roi_inner_bin
                region_file = f"{img_path.stem}_text_inner_crop.png"
                inner_meta = f" layer2_inner=({ix},{iy},{iw}x{ih})"
                if not args.no_debug_vis:
                    cv2.imwrite(
                        str(regions_dir / f"{img_path.stem}_text_inner_crop.png"),
                        roi_inner_bgr,
                    )
                    cv2.imwrite(
                        str(regions_dir / f"{img_path.stem}_text_inner_preproc_bin.png"),
                        roi_inner_bin,
                    )
                    print(
                        f"{lp} layer-2 text bbox in strip coords: x={ix} y={iy} w={iw} h={ih} "
                        f"(saved *_text_inner_*.png)",
                        flush=True,
                    )
                else:
                    print(
                        f"{lp} layer-2 text bbox in strip coords: x={ix} y={iy} w={iw} h={ih}",
                        flush=True,
                    )
            else:
                print(f"{lp} layer-2 skipped (use full strip for OCR)", flush=True)

        ih2, iw2 = roi_for_ocr.shape[:2]
        print(
            f"{lp} running OCR (mode={args.ocr_mode}, model={model_size}, ROI={iw2}x{ih2}px{inner_meta}) ...",
            flush=True,
        )
        t_ocr0 = time.perf_counter()
        t_detvis = 0.0
        if args.ocr_mode == "high_accuracy":
            if args.ocr_verbose and args.input_upright:
                print(f"{lp} high_accuracy: --input-upright -> no 90°/180°/270° image rotation")
            if args.ocr_verbose:
                print(
                    f"{lp} high_accuracy: scales={hi_scales} (1.0=原图不放大; 多值=多尺度投票)",
                    flush=True,
                )
            merged_text, avg_conf, line_count, ocr_tag = run_ocr_high_accuracy(
                roi_for_ocr,
                ocr_engine,
                args.ocr_min_conf,
                log_prefix=lp,
                verbose=args.ocr_verbose,
                input_upright=args.input_upright,
                hi_acc_scales=hi_scales,
            )
            print(f"{lp} ocr_best: {ocr_tag}")
            if not args.no_debug_vis:
                t_detvis0 = time.perf_counter()
                det_debug_lines = ocr_roi_lines(roi_for_ocr, ocr_engine)
                vis_inner = (
                    merge_fragment_det_boxes_for_vis(det_debug_lines, roi_for_ocr.shape[:2])
                    if not args.no_det_vis_merge
                    else det_debug_lines
                )
                det_suffix = (
                    "text_inner_det_vis"
                    if region_file.endswith("_text_inner_crop.png")
                    else "uniform_region_det_vis"
                )
                save_det_debug_outputs(
                    roi_for_ocr, vis_inner, regions_dir, img_path.stem, suffix=det_suffix
                )
                # Also keep full-strip det vis for easier diagnosis.
                if region_file.endswith("_text_inner_crop.png"):
                    full_strip_lines = ocr_roi_lines(roi_ocr, ocr_engine)
                    vis_full = (
                        merge_fragment_det_boxes_for_vis(full_strip_lines, roi_ocr.shape[:2])
                        if not args.no_det_vis_merge
                        else full_strip_lines
                    )
                    save_det_debug_outputs(
                        roi_ocr,
                        vis_full,
                        regions_dir,
                        img_path.stem,
                        suffix="uniform_region_det_vis",
                    )
                t_detvis = time.perf_counter() - t_detvis0
                print(
                    f"{lp} det debug saved: {img_path.stem}_{det_suffix}.png "
                    f"(det_lines={len(det_debug_lines)} vis_boxes={len(vis_inner)})",
                    flush=True,
                )
        else:
            merged_text, avg_conf, line_count, det_debug_lines = run_ocr_fast(
                roi_for_ocr,
                ocr_engine,
                args.ocr_upscale,
                args.ocr_min_conf,
                log_prefix=lp,
                verbose=args.ocr_verbose,
            )
            if not args.no_debug_vis:
                t_detvis0 = time.perf_counter()
                vis_inner = (
                    merge_fragment_det_boxes_for_vis(det_debug_lines, roi_for_ocr.shape[:2])
                    if not args.no_det_vis_merge
                    else det_debug_lines
                )
                det_suffix = (
                    "text_inner_det_vis"
                    if region_file.endswith("_text_inner_crop.png")
                    else "uniform_region_det_vis"
                )
                save_det_debug_outputs(
                    roi_for_ocr, vis_inner, regions_dir, img_path.stem, suffix=det_suffix
                )
                if region_file.endswith("_text_inner_crop.png"):
                    full_strip_lines = ocr_roi_lines(roi_ocr, ocr_engine)
                    vis_full = (
                        merge_fragment_det_boxes_for_vis(full_strip_lines, roi_ocr.shape[:2])
                        if not args.no_det_vis_merge
                        else full_strip_lines
                    )
                    save_det_debug_outputs(
                        roi_ocr,
                        vis_full,
                        regions_dir,
                        img_path.stem,
                        suffix="uniform_region_det_vis",
                    )
                t_detvis = time.perf_counter() - t_detvis0
                print(
                    f"{lp} det debug saved: {img_path.stem}_{det_suffix}.png "
                    f"(det_lines={len(det_debug_lines)} vis_boxes={len(vis_inner)})",
                    flush=True,
                )
        t_ocr = time.perf_counter() - t_ocr0
        t_parse0 = time.perf_counter()
        normalized_text, production_date, normal_expiry, frozen_expiry = extract_structured_fields(
            merged_text
        )
        t_parse = time.perf_counter() - t_parse0
        print(f"[{img_path.name}] lines_kept={line_count}")
        elapsed = time.perf_counter() - t0
        print(
            f"{lp} timing: seg={t_seg:.3f}s preproc={t_pre:.3f}s "
            f"ocr={t_ocr:.3f}s det_vis={t_detvis:.3f}s parse={t_parse:.3f}s total={elapsed:.3f}s"
        )
        print(f"{lp} done in {elapsed:.2f}s")
        records.append(
            OCRRecord(
                file_name=img_path.name,
                region_file=region_file,
                line_count=line_count,
                text=merged_text,
                normalized_text=normalized_text,
                production_date=production_date,
                normal_expiry=normal_expiry,
                frozen_expiry=frozen_expiry,
                confidence=round(avg_conf, 2),
                elapsed_seconds=round(elapsed, 3),
            )
        )

    save_results(records, output_dir)
    if not records:
        print(
            "No OCR records produced. Try lowering '--roi-val-min' or increasing '--roi-sat-max'."
        )
    print(f"Done. OCR records: {len(records)}")
    xlsx_path = output_dir / "ocr_results.xlsx"
    has_xlsx = xlsx_path.exists()
    print(
        f"Result files: {output_dir / 'ocr_results.json'}, {output_dir / 'ocr_results.csv'}"
        + (f", {xlsx_path}" if has_xlsx else " (install openpyxl for .xlsx)")
    )


if __name__ == "__main__":
    run_pipeline(parse_args())
