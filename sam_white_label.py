#!/usr/bin/env python3
"""
Segment the inkjet **white label panel** with Meta **Segment Anything (SAM)**,
using Paddle detection boxes (from `paddle_full_image_detect.py` JSON) as prompts.
Only boxes whose `text` contains at least one CJK character (中文) are used.

Typical pipeline:
  1. Run Paddle → `*_det_boxes.json`
  2. Run this script → mask / RGBA crop / optional tight PNG

Prompt strategy (see `--prompt-mode`):
  - **union_box**: axis-aligned union of all (filtered) `bbox_xyxy`, padded → SAM **box** prompt.
  - **points**: center of each box → SAM **positive point** prompts (good when boxes sit on the label).
  - **both**: union box + points (often most stable).

Dependencies (install once):
  pip install torch torchvision
  pip install "segment_anything @ https://github.com/facebookresearch/segment-anything/archive/refs/heads/main.zip"
  # If GitHub is unreachable, configure proxy/mirror or download the zip manually and:
  # pip install /path/to/segment-anything-main.zip

Download a SAM checkpoint (e.g. ViT-B, ~375MB):
  https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth

Example:
  python sam_white_label.py \\
    --image data3/20260429154940.jpg \\
    --det-json outputs_loose/20260429154940_det_boxes.json \\
    --checkpoint ~/models/sam_vit_b_01ec64.pth \\
    --output-dir outputs_sam_label
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Literal

import cv2
import numpy as np

PromptMode = Literal["union_box", "points", "both"]

# CJK Unified Ideographs — matches typical Chinese characters in recognition text.
_RE_HAS_CJK = re.compile(r"[\u4e00-\u9fff]")


def text_has_chinese(text: str | None) -> bool:
    if text is None:
        return False
    s = str(text).strip()
    if not s:
        return False
    return _RE_HAS_CJK.search(s) is not None


def filter_boxes_chinese_text(
    entries: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return [e for e in entries if text_has_chinese(e.get("text"))]


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


def centers_from_boxes(boxes: list[np.ndarray]) -> np.ndarray:
    pts = []
    for b in boxes:
        x1, y1, x2, y2 = b
        pts.append([(x1 + x2) * 0.5, (y1 + y2) * 0.5])
    return np.array(pts, dtype=np.float32)


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
    best = int(np.argmax(scores))
    return masks[best].astype(bool), scores


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
    min_area: float,
    multimask: bool,
    stem: str | None = None,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    ih, iw = img_bgr.shape[:2]
    entries, jw, jh = load_det_boxes(det_json)
    if jw and jh and (jw != iw or jh != ih):
        print(
            f"Warning: JSON size {jw}x{jh} != image {iw}x{ih} ({image_path.name}); "
            "using image dimensions.",
            file=sys.stderr,
        )

    entries = filter_boxes_chinese_text(entries)
    if not entries:
        raise ValueError(
            "No boxes with Chinese in `text`. "
            "Use recognition output JSON (e.g. paddle_full_image_detect --with-recognition), "
            "or ensure at least one box has 中文 in `text`."
        )

    entries = filter_boxes_by_area(entries, min_area=min_area)
    if not entries:
        raise ValueError(
            f"No Chinese-text boxes left after min_area={min_area} filter. "
            "Lower --min-box-area or fix JSON."
        )

    xyxys = [bbox_xyxy_from_entry(e) for e in entries]
    uni = union_xyxy(xyxys)
    uni_pad = pad_xyxy(uni, (ih, iw), pad_ratio)
    centers = centers_from_boxes(xyxys)

    predictor, _torch = build_predictor(checkpoint, model_type, device)
    image_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    box_for_sam = None
    if prompt_mode in ("union_box", "both"):
        box_for_sam = uni_pad
    pts_for_sam = None
    if prompt_mode in ("points", "both"):
        pts_for_sam = centers

    mask_b, sc = run_sam(
        predictor,
        image_rgb,
        union_box=box_for_sam,
        point_coords=pts_for_sam,
        prompt_mode=prompt_mode,
        multimask_output=multimask,
    )

    name = stem or image_path.stem
    mask_u8 = (mask_b.astype(np.uint8) * 255)
    cv2.imwrite(str(out_dir / f"{name}_sam_mask.png"), mask_u8)

    rgba_full = rgba_with_mask(img_bgr, mask_b)
    cv2.imwrite(str(out_dir / f"{name}_label_rgba.png"), rgba_full)

    tight, xyxy = tight_crop_from_mask(img_bgr, mask_b)
    cv2.imwrite(str(out_dir / f"{name}_label_crop.png"), tight)

    overlay = img_bgr.copy()
    green = np.zeros_like(img_bgr)
    green[:, :] = (0, 255, 0)
    overlay = np.where(mask_b[..., None], (0.5 * overlay + 0.5 * green).astype(np.uint8), overlay)
    cv2.imwrite(str(out_dir / f"{name}_sam_overlay.png"), overlay)

    meta = {
        "image": image_path.name,
        "det_json": str(det_json),
        "prompt_mode": prompt_mode,
        "pad_ratio": pad_ratio,
        "min_box_area": min_area,
        "union_xyxy": uni_pad.tolist(),
        "det_boxes_used": len(entries),
        "chinese_text_boxes_only": True,
        "mask_pixels": int(mask_b.sum()),
        "tight_bbox_xyxy": list(xyxy),
        "sam_score_best": float(np.max(sc)) if sc.size else None,
    }
    with (out_dir / f"{name}_sam_meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    return meta


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "SAM white-label segmentation from Paddle det JSON. "
            "Only entries with Chinese in boxes[].text are used as prompts."
        ),
    )
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--det-json", type=Path, required=True)
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
        "--min-box-area",
        type=float,
        default=0.0,
        help="Drop det boxes smaller than this area (pixels). "
        "Suggest ~5000–15000 to remove spurious tiny boxes.",
    )
    parser.add_argument(
        "--multimask",
        action="store_true",
        help="Ask SAM for 3 masks and take highest score (sometimes better).",
    )
    args = parser.parse_args()

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

    meta = process_one(
        args.image.resolve(),
        args.det_json.resolve(),
        args.output_dir.resolve(),
        checkpoint=ck.resolve(),
        model_type=args.model_type,
        device=device,
        prompt_mode=args.prompt_mode,
        pad_ratio=args.pad_ratio,
        min_area=args.min_box_area,
        multimask=args.multimask,
    )
    print(json.dumps(meta, ensure_ascii=False, indent=2))
    print(f"Done -> {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
