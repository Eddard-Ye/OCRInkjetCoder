#!/usr/bin/env python3
"""
Segment the inkjet **white label panel** with Meta **Segment Anything (SAM)**,
using Paddle detection boxes (from `paddle_full_image_detect.py` JSON) as prompts.
Only boxes whose Paddle recognition `text` has **more than three characters**
(length ≥ `--min-text-chars`, default **4**) are used.

Typical pipeline:
  1. Run Paddle → `*_det_boxes.json`
  2. Run this script → mask / RGBA crop / `*_sam_prompts.png` (box + 前景点) / overlay;
     per-image `*_sam_meta.json` includes `timing_seconds` (read → json load → prepare →
     model load → SAM predict → mask post → render → write + **total**).

SAM tuning (native decode knobs are few; most fixes are prompts + post-process):
  - **`--model-type vit_h`**: larger backbone than vit_b, often cleaner masks (needs matching .pth).
  - **`--multimask` + `--mask-pick largest`**: when SAM returns 3 candidates, pick by area instead of score.
  - **`--pad-ratio` / `--point-expand-ratio`**: larger prompts tend to include more of the white panel.
  - **`--mask-close` / `--mask-fill-holes` / `--mask-dilate`**: heal jagged or incomplete masks after SAM.

Prompt strategy (see `--prompt-mode`, `--point-expand-ratio`):
  - **union_box**: union of all kept `bbox_xyxy`, padded with `--pad-ratio` → SAM **box** prompt.
  - **points**: for each kept box, take `bbox_xyxy` **outward-expanded** by `--point-expand-ratio`,
    then use the **four corners** of that rectangle as SAM **foreground** points (not box centers).
  - **both**: union **box** (as above) + per-box expanded **corner** points.

Dependencies (install once):
  pip install torch torchvision
  pip install "segment_anything @ https://github.com/facebookresearch/segment-anything/archive/refs/heads/main.zip"
  # If GitHub is unreachable, configure proxy/mirror or download the zip manually and:
  # pip install /path/to/segment-anything-main.zip

Download a SAM checkpoint (e.g. ViT-B, ~375MB):
  https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth

Example (single image):
  python sam_white_label.py \\
    --image data3/20260429154940.jpg \\
    --det-json outputs_loose/20260429154940_det_boxes.json \\
    --checkpoint ~/models/sam_vit_b_01ec64.pth \\
    --output-dir outputs_sam_label

Batch (folder of images + folder of ``{{stem}}_det_boxes.json``, e.g. from ``--json-copy-dir``):
  python sam_white_label.py \\
    --image-dir data3 --det-json-dir all_det_json \\
    --checkpoint ~/models/sam_vit_b_01ec64.pth \\
    --output-dir outputs_sam_label
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Literal

import cv2
import numpy as np

PromptMode = Literal["union_box", "points", "both"]

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def iter_images(folder: Path) -> list[Path]:
    out: list[Path] = []
    for p in sorted(folder.rglob("*")):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            out.append(p)
    return out


def filter_boxes_min_recognition_chars(
    entries: list[dict[str, Any]], *, min_chars: int
) -> list[dict[str, Any]]:
    """Keep boxes where Paddle `text` length (stripped) is at least `min_chars`."""
    out: list[dict[str, Any]] = []
    for e in entries:
        t = e.get("text")
        if t is None:
            continue
        s = str(t).strip()
        if len(s) >= min_chars:
            out.append(e)
    return out


def load_det_boxes(json_path: Path) -> tuple[list[dict[str, Any]], int, int]:
    with json_path.open(encoding="utf-8") as f:
        data = json.load(f)
    boxes = data.get("boxes") or []
    w = int(data.get("width", 0))
    h = int(data.get("height", 0))
    return boxes, w, h


def bbox_xyxy_from_entry(entry: dict[str, Any]) -> np.ndarray:
    bb = entry.get("bbox_xyxy")
    if bb is None or len(bb) < 4:
        poly = entry.get("poly")
        if not poly or len(poly) < 2:
            raise ValueError("box entry missing bbox_xyxy and valid poly")
        pts = np.asarray(poly, dtype=np.float32)
        return np.array(
            [
                float(pts[:, 0].min()),
                float(pts[:, 1].min()),
                float(pts[:, 0].max()),
                float(pts[:, 1].max()),
            ],
            dtype=np.float32,
        )
    return np.array([float(bb[0]), float(bb[1]), float(bb[2]), float(bb[3])], dtype=np.float32)


def box_area(xyxy: np.ndarray) -> float:
    return float(max(0.0, xyxy[2] - xyxy[0]) * max(0.0, xyxy[3] - xyxy[1]))


def filter_boxes_by_area(
    entries: list[dict[str, Any]], *, min_area: float
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for e in entries:
        bb = bbox_xyxy_from_entry(e)
        if box_area(bb) >= min_area:
            out.append(e)
    return out


def union_xyxy(boxes: list[np.ndarray]) -> np.ndarray:
    if not boxes:
        raise ValueError("no boxes for union")
    ar = np.stack(boxes, axis=0)
    return np.array(
        [
            float(ar[:, 0].min()),
            float(ar[:, 1].min()),
            float(ar[:, 2].max()),
            float(ar[:, 3].max()),
        ],
        dtype=np.float32,
    )


def pad_xyxy(
    xyxy: np.ndarray, shape_hw: tuple[int, int], pad_ratio: float
) -> np.ndarray:
    h, w = shape_hw[0], shape_hw[1]
    x1, y1, x2, y2 = [float(x) for x in xyxy]
    bw, bh = max(1.0, x2 - x1), max(1.0, y2 - y1)
    px, py = bw * pad_ratio, bh * pad_ratio
    return np.array(
        [
            max(0.0, x1 - px),
            max(0.0, y1 - py),
            min(float(w - 1), x2 + px),
            min(float(h - 1), y2 + py),
        ],
        dtype=np.float32,
    )


def corner_foreground_points_from_boxes(
    xyxys: list[np.ndarray],
    shape_hw: tuple[int, int],
    expand_ratio: float,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """
    Per kept detection box: outward-expand bbox_xyxy by expand_ratio, then take the
    four axis-aligned corners as SAM foreground (positive) points.

    Returns:
        points (4*N, 2), expanded_xyxy list (one per input box) for visualization.
    """
    pts: list[list[float]] = []
    expanded_list: list[np.ndarray] = []
    for bb in xyxys:
        ex = pad_xyxy(bb, shape_hw, expand_ratio)
        expanded_list.append(ex)
        x1, y1, x2, y2 = (float(ex[0]), float(ex[1]), float(ex[2]), float(ex[3]))
        pts.extend(
            [
                [x1, y1],
                [x2, y1],
                [x2, y2],
                [x1, y2],
            ]
        )
    return np.asarray(pts, dtype=np.float32), expanded_list


def rgba_with_mask(
    image_bgr: np.ndarray, mask_bool: np.ndarray
) -> np.ndarray:
    """BGR image + HxW bool mask → BGRA."""
    if mask_bool.shape[:2] != image_bgr.shape[:2]:
        raise ValueError("mask shape must match image")
    bgra = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2BGRA)
    bgra[:, :, 3] = (mask_bool.astype(np.uint8) * 255)
    return bgra


def tight_crop_from_mask(image_bgr: np.ndarray, mask_bool: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    ys, xs = np.where(mask_bool)
    if len(xs) == 0:
        return image_bgr.copy(), (0, 0, image_bgr.shape[1], image_bgr.shape[0])
    x1, x2 = int(xs.min()), int(xs.max()) + 1
    y1, y2 = int(ys.min()), int(ys.max()) + 1
    crop = image_bgr[y1:y2, x1:x2].copy()
    mcrop = mask_bool[y1:y2, x1:x2]
    bgra = cv2.cvtColor(crop, cv2.COLOR_BGR2BGRA)
    bgra[:, :, 3] = (mcrop.astype(np.uint8) * 255)
    return bgra, (x1, y1, x2, y2)


def draw_sam_prompt_visualization(
    img_bgr: np.ndarray,
    *,
    box_xyxy: np.ndarray | None,
    foreground_points: np.ndarray | None,
    per_box_expanded_xyxy: list[np.ndarray] | None = None,
) -> np.ndarray:
    """
    Draw exactly what was passed to SAM: optional box prompt + optional positive (FG) points.

    Box: orange rectangle. Per-box expanded rects: thin mint outline (optional).
    Points: red disk + white ring + index (SAM foreground label = 1).
    """
    vis = img_bgr.copy()
    h, w = vis.shape[:2]

    if per_box_expanded_xyxy:
        for ex in per_box_expanded_xyxy:
            x1, y1, x2, y2 = [float(x) for x in ex]
            ix1 = int(np.clip(round(x1), 0, w - 1))
            iy1 = int(np.clip(round(y1), 0, h - 1))
            ix2 = int(np.clip(round(x2), 0, w - 1))
            iy2 = int(np.clip(round(y2), 0, h - 1))
            cv2.rectangle(vis, (ix1, iy1), (ix2, iy2), (180, 255, 180), 1)

    if box_xyxy is not None:
        x1, y1, x2, y2 = [float(x) for x in box_xyxy]
        ix1 = int(np.clip(round(x1), 0, w - 1))
        iy1 = int(np.clip(round(y1), 0, h - 1))
        ix2 = int(np.clip(round(x2), 0, w - 1))
        iy2 = int(np.clip(round(y2), 0, h - 1))
        cv2.rectangle(vis, (ix1, iy1), (ix2, iy2), (0, 165, 255), 2)
        cv2.putText(
            vis,
            "SAM box prompt",
            (ix1, max(16, iy1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 165, 255),
            2,
            cv2.LINE_AA,
        )

    if foreground_points is not None and len(foreground_points) > 0:
        for i, row in enumerate(foreground_points, start=1):
            px, py = float(row[0]), float(row[1])
            cx = int(np.clip(round(px), 0, w - 1))
            cy = int(np.clip(round(py), 0, h - 1))
            cv2.circle(vis, (cx, cy), 9, (255, 255, 255), 2)
            cv2.circle(vis, (cx, cy), 6, (0, 0, 255), -1)
            cv2.putText(
                vis,
                f"FG{i}",
                (cx + 10, cy + 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                1,
                cv2.LINE_AA,
            )
        cv2.putText(
            vis,
            "red = expanded-bbox corners (FG)",
            (8, h - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            1,
            cv2.LINE_AA,
        )

    return vis


def build_predictor(
    checkpoint: Path,
    model_type: str,
    device: str,
):
    try:
        from segment_anything import sam_model_registry, SamPredictor
        import torch
    except ImportError as e:
        raise SystemExit(
            "Missing dependency. Install:\n"
            "  pip install torch torchvision\n"
            "  pip install git+https://github.com/facebookresearch/segment-anything.git\n"
            f"Import error: {e}"
        ) from e

    if not checkpoint.is_file():
        raise SystemExit(f"SAM checkpoint not found: {checkpoint}")

    sam = sam_model_registry[model_type](checkpoint=str(checkpoint))
    sam.to(device=device)
    return SamPredictor(sam), torch


def run_sam(
    predictor,
    image_rgb: np.ndarray,
    *,
    union_box: np.ndarray | None,
    point_coords: np.ndarray | None,
    prompt_mode: PromptMode,
    multimask_output: bool,
    mask_pick: Literal["score", "largest"],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns:
        mask_best: HxW bool
        scores: (K,) if multimask else (1,)
    """
    predictor.set_image(image_rgb)

    box_arg = None
    pts_arg = None
    lbl_arg = None

    if prompt_mode == "union_box":
        box_arg = union_box
    elif prompt_mode == "points":
        pts_arg = point_coords
        lbl_arg = np.ones(len(point_coords), dtype=np.int32)
    else:
        box_arg = union_box
        pts_arg = point_coords
        lbl_arg = np.ones(len(point_coords), dtype=np.int32)

    masks, scores, _ = predictor.predict(
        point_coords=pts_arg,
        point_labels=lbl_arg,
        box=box_arg,
        multimask_output=multimask_output,
    )
    # masks: (N, H, W); scores: (N,)
    if multimask_output and mask_pick == "largest":
        areas = masks.reshape(masks.shape[0], -1).sum(axis=1)
        best = int(np.argmax(areas))
    else:
        best = int(np.argmax(scores))
    return masks[best].astype(bool), scores


def refine_mask_bool(
    mask_bool: np.ndarray,
    *,
    close_kernel: int,
    dilate_kernel: int,
    fill_holes: bool,
) -> np.ndarray:
    """
    Post-process SAM mask: close gaps, optional hole fill, optional dilate to recover edges.

    ``close_kernel`` / ``dilate_kernel``: odd pixel size, or 0 to disable that step.
    """
    m = (mask_bool.astype(np.uint8)) * 255

    def _odd_k(k: int) -> int:
        if k <= 0:
            return 0
        return k if k % 2 == 1 else k + 1

    ck = _odd_k(close_kernel)
    if ck > 0:
        ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ck, ck))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, ker)

    if fill_holes:
        h, w = m.shape
        im_ff = m.copy()
        flood_mask = np.zeros((h + 2, w + 2), np.uint8)
        cv2.floodFill(im_ff, flood_mask, (0, 0), 255)
        im_inv = cv2.bitwise_not(im_ff)
        m = cv2.bitwise_or(m, im_inv)

    dk = _odd_k(dilate_kernel)
    if dk > 0:
        ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dk, dk))
        m = cv2.dilate(m, ker)

    return m > 127


def process_one(
    image_path: Path,
    det_json: Path,
    out_dir: Path,
    *,
    checkpoint: Path,
    model_type: str,
    device: str,
    prompt_mode: PromptMode,
    pad_ratio: float,
    point_expand_ratio: float,
    min_text_chars: int,
    min_area: float,
    multimask: bool,
    mask_pick: Literal["score", "largest"],
    mask_close_kernel: int,
    mask_dilate_kernel: int,
    mask_fill_holes: bool,
    stem: str | None = None,
    predictor: Any | None = None,
) -> dict[str, Any]:
    """
    If ``predictor`` is None, loads SAM once via ``build_predictor``.
    Pass a shared predictor from ``main`` when batch-processing many images.
    """
    t_wall0 = time.perf_counter()
    out_dir.mkdir(parents=True, exist_ok=True)

    t_a = time.perf_counter()
    img_bgr = cv2.imread(str(image_path))
    t_b = time.perf_counter()
    read_seconds = round(float(t_b - t_a), 4)
    if img_bgr is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    ih, iw = img_bgr.shape[:2]

    t_a = time.perf_counter()
    entries, jw, jh = load_det_boxes(det_json)
    t_b = time.perf_counter()
    load_det_json_seconds = round(float(t_b - t_a), 4)
    if jw and jh and (jw != iw or jh != ih):
        print(
            f"Warning: JSON size {jw}x{jh} != image {iw}x{ih} ({image_path.name}); "
            "using image dimensions.",
            file=sys.stderr,
        )

    t_a = time.perf_counter()
    entries = filter_boxes_min_recognition_chars(entries, min_chars=min_text_chars)
    if not entries:
        raise ValueError(
            f"No boxes with recognition text length ≥ {min_text_chars}. "
            "Use `--with-recognition` JSON from paddle_full_image_detect, "
            "or lower --min-text-chars."
        )

    entries = filter_boxes_by_area(entries, min_area=min_area)
    if not entries:
        raise ValueError(
            f"No boxes left after min_area={min_area} filter. "
            "Lower --min-box-area or fix JSON."
        )

    xyxys = [bbox_xyxy_from_entry(e) for e in entries]
    uni = union_xyxy(xyxys)
    uni_pad = pad_xyxy(uni, (ih, iw), pad_ratio)

    pts_for_sam: np.ndarray | None = None
    expanded_per_box: list[np.ndarray] | None = None
    if prompt_mode in ("points", "both"):
        pts_for_sam, expanded_per_box = corner_foreground_points_from_boxes(
            xyxys, (ih, iw), point_expand_ratio
        )

    box_for_sam = None
    if prompt_mode in ("union_box", "both"):
        box_for_sam = uni_pad

    image_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    t_b = time.perf_counter()
    prepare_prompts_seconds = round(float(t_b - t_a), 4)

    sam_model_load_seconds = 0.0
    if predictor is None:
        t_a = time.perf_counter()
        predictor, _torch = build_predictor(checkpoint, model_type, device)
        t_b = time.perf_counter()
        sam_model_load_seconds = round(float(t_b - t_a), 4)

    t_a = time.perf_counter()
    mask_raw, sc = run_sam(
        predictor,
        image_rgb,
        union_box=box_for_sam,
        point_coords=pts_for_sam,
        prompt_mode=prompt_mode,
        multimask_output=multimask,
        mask_pick=mask_pick,
    )
    t_b = time.perf_counter()
    sam_predict_seconds = round(float(t_b - t_a), 4)

    t_a = time.perf_counter()
    mask_b = refine_mask_bool(
        mask_raw,
        close_kernel=mask_close_kernel,
        dilate_kernel=mask_dilate_kernel,
        fill_holes=mask_fill_holes,
    )
    t_b = time.perf_counter()
    sam_mask_postprocess_seconds = round(float(t_b - t_a), 4)

    name = stem or image_path.stem

    t_a = time.perf_counter()
    prompt_vis = draw_sam_prompt_visualization(
        img_bgr,
        box_xyxy=box_for_sam,
        foreground_points=pts_for_sam,
        per_box_expanded_xyxy=expanded_per_box,
    )
    mask_u8 = (mask_b.astype(np.uint8) * 255)
    rgba_full = rgba_with_mask(img_bgr, mask_b)
    tight, xyxy = tight_crop_from_mask(img_bgr, mask_b)
    overlay = img_bgr.copy()
    green = np.zeros_like(img_bgr)
    green[:, :] = (0, 255, 0)
    overlay = np.where(mask_b[..., None], (0.5 * overlay + 0.5 * green).astype(np.uint8), overlay)
    t_b = time.perf_counter()
    render_seconds = round(float(t_b - t_a), 4)

    t_a = time.perf_counter()
    cv2.imwrite(str(out_dir / f"{name}_sam_prompts.png"), prompt_vis)
    cv2.imwrite(str(out_dir / f"{name}_sam_mask.png"), mask_u8)
    cv2.imwrite(str(out_dir / f"{name}_label_rgba.png"), rgba_full)
    cv2.imwrite(str(out_dir / f"{name}_label_crop.png"), tight)
    cv2.imwrite(str(out_dir / f"{name}_sam_overlay.png"), overlay)
    t_b = time.perf_counter()
    write_png_seconds = round(float(t_b - t_a), 4)

    meta = {
        "image": image_path.name,
        "det_json": str(det_json),
        "prompt_mode": prompt_mode,
        "pad_ratio": pad_ratio,
        "point_expand_ratio": point_expand_ratio,
        "min_box_area": min_area,
        "union_xyxy": uni_pad.tolist(),
        "det_boxes_used": len(entries),
        "min_text_chars": min_text_chars,
        "mask_pixels": int(mask_b.sum()),
        "tight_bbox_xyxy": list(xyxy),
        "sam_score_best": float(np.max(sc)) if sc.size else None,
        "mask_pick": mask_pick,
        "mask_postprocess": {
            "close_kernel": mask_close_kernel,
            "dilate_kernel": mask_dilate_kernel,
            "fill_holes": mask_fill_holes,
        },
        "sam_prompts_png": f"{name}_sam_prompts.png",
        "sam_foreground_points_xy": (
            pts_for_sam.tolist() if pts_for_sam is not None else []
        ),
        "sam_box_prompt_xyxy": (
            box_for_sam.tolist() if box_for_sam is not None else None
        ),
    }

    sam_pipeline_seconds = round(
        sam_predict_seconds + sam_mask_postprocess_seconds, 4
    )

    timing_seconds: dict[str, float] = {
        "read": read_seconds,
        "load_det_json": load_det_json_seconds,
        "prepare_prompts": prepare_prompts_seconds,
        "sam_model_load": sam_model_load_seconds,
        "sam_predict": sam_predict_seconds,
        "sam_mask_postprocess": sam_mask_postprocess_seconds,
        "sam_pipeline": sam_pipeline_seconds,
        "render": render_seconds,
        "write_png": write_png_seconds,
    }
    meta["timing_seconds"] = dict(timing_seconds)

    t_json0 = time.perf_counter()
    with (out_dir / f"{name}_sam_meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    t_json1 = time.perf_counter()
    write_json_seconds = round(float(t_json1 - t_json0), 4)

    timing_seconds["write_json"] = write_json_seconds
    timing_seconds["write"] = round(write_png_seconds + write_json_seconds, 4)
    parts_sum = (
        read_seconds
        + load_det_json_seconds
        + prepare_prompts_seconds
        + sam_model_load_seconds
        + sam_predict_seconds
        + sam_mask_postprocess_seconds
        + render_seconds
        + write_png_seconds
        + write_json_seconds
    )
    timing_seconds["parts_sum"] = round(parts_sum, 4)
    timing_seconds["total"] = round(float(time.perf_counter() - t_wall0), 4)
    meta["timing_seconds"] = timing_seconds

    with (out_dir / f"{name}_sam_meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    return meta


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "SAM white-label segmentation from Paddle det JSON. "
            "Only entries with boxes[].text length ≥ --min-text-chars (default: >3 chars) "
            "are used as prompts."
        ),
    )
    parser.add_argument(
        "--image",
        type=Path,
        default=None,
        help="Single image path (use with --det-json).",
    )
    parser.add_argument(
        "--det-json",
        type=Path,
        default=None,
        help="Single *_det_boxes.json (use with --image).",
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=None,
        help="Folder of images; pairs with --det-json-dir as {stem}_det_boxes.json per file.",
    )
    parser.add_argument(
        "--det-json-dir",
        type=Path,
        default=None,
        help="Folder containing Paddle *_det_boxes.json (e.g. all_det_json).",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs_sam_label"))
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Path to sam_vit_*.pth (default: env SAM_CHECKPOINT or ./checkpoints/sam_vit_b_01ec64.pth)",
    )
    parser.add_argument(
        "--model-type",
        choices=["vit_h", "vit_l", "vit_b"],
        default="vit_b",
        help="Must match checkpoint (default vit_b).",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="cuda / cuda:0 / cpu (default: cuda if available else cpu)",
    )
    parser.add_argument(
        "--prompt-mode",
        choices=["union_box", "points", "both"],
        default="both",
        help="How to turn Paddle boxes into SAM prompts (default: both).",
    )
    parser.add_argument(
        "--pad-ratio",
        type=float,
        default=0.12,
        help="Expand union box by this fraction before SAM (default 0.12).",
    )
    parser.add_argument(
        "--point-expand-ratio",
        type=float,
        default=0.04,
        help="For each kept bbox_xyxy, expand outward by this fraction before taking "
        "the four corner points as SAM foreground prompts (default 0.04). "
        "Ignored for --prompt-mode union_box.",
    )
    parser.add_argument(
        "--min-text-chars",
        type=int,
        default=4,
        help="Minimum Paddle recognition character count (after strip). "
        "Default 4 means strictly more than three characters. "
        "Requires recognition JSON (not det-only).",
    )
    parser.add_argument(
        "--min-box-area",
        type=float,
        default=0.0,
        help="Drop det boxes smaller than this area (pixels). "
        "Suggest ~5000–15000 to remove spurious tiny boxes.",
    )
    parser.add_argument(
        "--multimask",
        action="store_true",
        help="Ask SAM for 3 masks, then pick one (see --mask-pick).",
    )
    parser.add_argument(
        "--mask-pick",
        choices=["score", "largest"],
        default="score",
        help="With --multimask: keep highest IoU score (default) or the largest-area mask.",
    )
    parser.add_argument(
        "--mask-close",
        type=int,
        default=0,
        help="Post-process: morphological close kernel size (odd; e.g. 7–15), 0=off. "
        "Fills small gaps / jagged edges in the mask.",
    )
    parser.add_argument(
        "--mask-dilate",
        type=int,
        default=0,
        help="Post-process: final dilate kernel (odd; e.g. 3–5), 0=off. Slightly grows the mask "
        "to recover under-segmented borders.",
    )
    parser.add_argument(
        "--mask-fill-holes",
        action="store_true",
        help="Post-process: flood-fill to remove internal holes in the white region.",
    )
    args = parser.parse_args()

    single = args.image is not None and args.det_json is not None
    batch = args.image_dir is not None and args.det_json_dir is not None
    if single and batch:
        parser.error("Use either single mode (--image + --det-json) or batch (--image-dir + --det-json-dir), not both.")
    if not single and not batch:
        parser.error(
            "Provide --image and --det-json, OR --image-dir and --det-json-dir."
        )

    ck = args.checkpoint
    if ck is None:
        import os

        env_ck = os.environ.get("SAM_CHECKPOINT")
        if env_ck:
            ck = Path(env_ck)
        else:
            ck = Path("checkpoints/sam_vit_b_01ec64.pth")

    device = args.device
    if device is None:
        try:
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"

    out_root = args.output_dir.resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    if single:
        meta = process_one(
            args.image.resolve(),
            args.det_json.resolve(),
            out_root,
            checkpoint=ck.resolve(),
            model_type=args.model_type,
            device=device,
            prompt_mode=args.prompt_mode,
            pad_ratio=args.pad_ratio,
            point_expand_ratio=args.point_expand_ratio,
            min_text_chars=args.min_text_chars,
            min_area=args.min_box_area,
            multimask=args.multimask,
            mask_pick=args.mask_pick,
            mask_close_kernel=args.mask_close,
            mask_dilate_kernel=args.mask_dilate,
            mask_fill_holes=args.mask_fill_holes,
        )
        print(json.dumps(meta, ensure_ascii=False, indent=2))
        print(f"Done -> {out_root}")
        return

    image_dir = args.image_dir.resolve()
    det_dir = args.det_json_dir.resolve()
    paths = iter_images(image_dir)
    if not paths:
        raise SystemExit(f"No images under {image_dir}")

    predictor, _torch = build_predictor(
        ck.resolve(), args.model_type, device
    )
    ok: list[str] = []
    failed: list[dict[str, str]] = []
    for img_path in paths:
        stem = img_path.stem
        dj = det_dir / f"{stem}_det_boxes.json"
        if not dj.is_file():
            failed.append(
                {"stem": stem, "error": f"missing JSON: {dj.name}"}
            )
            print(f"Skip {img_path.name}: no {dj.name}", file=sys.stderr)
            continue
        try:
            m = process_one(
                img_path,
                dj,
                out_root,
                checkpoint=ck.resolve(),
                model_type=args.model_type,
                device=device,
                prompt_mode=args.prompt_mode,
                pad_ratio=args.pad_ratio,
                point_expand_ratio=args.point_expand_ratio,
                min_text_chars=args.min_text_chars,
                min_area=args.min_box_area,
                multimask=args.multimask,
                mask_pick=args.mask_pick,
                mask_close_kernel=args.mask_close,
                mask_dilate_kernel=args.mask_dilate,
                mask_fill_holes=args.mask_fill_holes,
                stem=stem,
                predictor=predictor,
            )
            ok.append(stem)
            ts = m.get("timing_seconds") or {}
            print(
                f"OK {img_path.name} "
                f"sam_predict={ts.get('sam_predict', 0):.3f}s "
                f"total={ts.get('total', 0):.3f}s -> {stem}_*.png",
                flush=True,
            )
        except (ValueError, FileNotFoundError, OSError) as e:
            failed.append({"stem": stem, "error": str(e)})
            print(f"Fail {img_path.name}: {e}", file=sys.stderr)
        except Exception as e:
            failed.append({"stem": stem, "error": f"{type(e).__name__}: {e}"})
            print(f"Fail {img_path.name}:", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)

    summary = {
        "image_dir": str(image_dir),
        "det_json_dir": str(det_dir),
        "output_dir": str(out_root),
        "total_images": len(paths),
        "ok_count": len(ok),
        "fail_count": len(failed),
        "ok": ok,
        "failed": failed,
    }
    with (out_root / "batch_sam_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(
        f"Batch done -> {out_root} | ok={len(ok)} fail={len(failed)} "
        f"(see batch_sam_summary.json)"
    )


if __name__ == "__main__":
    main()
