#!/usr/bin/env python3
"""
YOLO instance segmentation: cut each detected instance into separate image blocks.

Uses Ultralytics YOLO (e.g. yolov8n-seg.pt). First run downloads weights.

Note: Default COCO classes are things like person, car — not packaging strips.
For product/date regions you need a custom-trained .pt. This script still outputs
whatever the model segments (masks + bbox crops + transparent PNGs).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def safe_filename_part(s: str) -> str:
    for c in '/\\:*?"<>|\n\r\t':
        s = s.replace(c, "_")
    return s.replace(" ", "_")


def iter_images(folder: Path) -> list[Path]:
    out: list[Path] = []
    for p in sorted(folder.rglob("*")):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            out.append(p)
    return out


def mask_to_fullsize(
    mask_hw: np.ndarray, target_h: int, target_w: int
) -> np.ndarray:
    """Resize single-channel mask to (target_h, target_w), values in [0,1]."""
    if mask_hw.shape[0] == target_h and mask_hw.shape[1] == target_w:
        return mask_hw.astype(np.float32)
    return cv2.resize(
        mask_hw.astype(np.float32),
        (target_w, target_h),
        interpolation=cv2.INTER_LINEAR,
    )


def process_one(
    image_path: Path,
    output_root: Path,
    model,
    save_overlay: bool,
    save_transparent: bool,
) -> int:
    """Return number of instances saved."""
    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        print(f"Skip (unreadable): {image_path}")
        return 0

    h0, w0 = img_bgr.shape[:2]
    results = model.predict(source=str(image_path), verbose=False)
    r = results[0]

    stem = image_path.stem
    out_dir = output_root / stem
    out_dir.mkdir(parents=True, exist_ok=True)

    if r.masks is None or len(r.boxes) == 0:
        print(f"{image_path.name}: no instances segmented.")
        # still save copy for traceability
        cv2.imwrite(str(out_dir / f"{stem}_original.png"), img_bgr)
        return 0

    masks_obj = r.masks
    # data: (N, Hm, Wm) float 0..1 — may need resize to orig
    masks_t = masks_obj.data.cpu().numpy()
    n = masks_t.shape[0]

    names = model.names if hasattr(model, "names") else {}

    for i in range(n):
        box = r.boxes.xyxy[i].cpu().numpy().astype(np.float32)
        x1, y1, x2, y2 = box
        x1i = max(0, int(np.floor(x1)))
        y1i = max(0, int(np.floor(y1)))
        x2i = min(w0, int(np.ceil(x2)))
        y2i = min(h0, int(np.ceil(y2)))

        cls_id = int(r.boxes.cls[i].item()) if r.boxes.cls is not None else -1
        label = names.get(cls_id, str(cls_id)) if isinstance(names, dict) else str(cls_id)
        conf = float(r.boxes.conf[i].item()) if r.boxes.conf is not None else 0.0

        m_full = mask_to_fullsize(masks_t[i], h0, w0)
        m_crop = m_full[y1i:y2i, x1i:x2i]
        crop_bgr = img_bgr[y1i:y2i, x1i:x2i].copy()

        base = f"{stem}_inst_{i:02d}_cls{cls_id}_{safe_filename_part(label)}_conf{conf:.2f}"

        # 1) BBox crop (always)
        cv2.imwrite(str(out_dir / f"{base}_bbox_crop.png"), crop_bgr)

        # 2) Masked crop (background black)
        if m_crop.size > 0:
            m3 = np.clip(m_crop[..., None], 0.0, 1.0)
            masked = (crop_bgr.astype(np.float32) * m3).astype(np.uint8)
            cv2.imwrite(str(out_dir / f"{base}_masked.png"), masked)

        # 3) Binary mask patch (same crop extent)
        if m_crop.size > 0:
            m_u8 = (np.clip(m_crop, 0, 1) * 255).astype(np.uint8)
            cv2.imwrite(str(out_dir / f"{base}_mask.png"), m_u8)

        # 4) Transparent PNG (RGBA)
        if save_transparent and m_crop.size > 0:
            bgra = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2BGRA)
            alpha = (np.clip(m_crop, 0, 1) * 255).astype(np.uint8)
            bgra[:, :, 3] = alpha
            cv2.imwrite(str(out_dir / f"{base}_rgba.png"), bgra)

    if save_overlay:
        plotted = r.plot()
        if plotted is not None:
            cv2.imwrite(str(out_dir / f"{stem}_overlay.png"), plotted)

    print(f"{image_path.name}: saved {n} instance(s) -> {out_dir}")
    return n


def main() -> None:
    parser = argparse.ArgumentParser(description="YOLO instance segmentation -> export each mask block.")
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input image file or directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs_yolo_seg"),
        help="Root folder for outputs (per-image subfolders). Default: outputs_yolo_seg",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8n-seg.pt",
        help="Ultralytics YOLO-seg weights (e.g. yolov8n-seg.pt or your custom.pt).",
    )
    parser.add_argument("--no-overlay", action="store_true", help="Do not save overlay preview.")
    parser.add_argument(
        "--no-rgba",
        action="store_true",
        help="Do not save transparent PNG crops.",
    )
    args = parser.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError as e:
        raise SystemExit(
            "Missing ultralytics. Install with:\n"
            "  pip install ultralytics\n"
        ) from e

    inp = args.input.resolve()
    out_root = args.output_dir.resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"Loading model: {args.model}")
    model = YOLO(args.model)

    paths: list[Path]
    if inp.is_file():
        paths = [inp]
    elif inp.is_dir():
        paths = iter_images(inp)
        if not paths:
            raise SystemExit(f"No images under: {inp}")
    else:
        raise SystemExit(f"Not a file or directory: {inp}")

    total_inst = 0
    for p in paths:
        total_inst += process_one(
            p,
            out_root,
            model,
            save_overlay=not args.no_overlay,
            save_transparent=not args.no_rgba,
        )

    print(f"Done. Images: {len(paths)}, total instances: {total_inst}. Root: {out_root}")


if __name__ == "__main__":
    main()
