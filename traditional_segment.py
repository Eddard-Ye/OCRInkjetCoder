#!/usr/bin/env python3
"""
Traditional computer vision segmentation for bright low-saturation regions (e.g. white strips).

Same mask idea as ocr_segment.py: HSV (low S, high V) + morphology, then either:
  strip       — largest inscribed rectangle in the mask (default, stable single ROI)
  components  — connected-component bounding boxes (optional horizontal-strip filter)

Outputs per image under --output-dir/<stem>/: mask, overlay, crops, masked patches.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from cv_roi import extract_uniform_components, extract_uniform_roi

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def iter_images(folder: Path) -> list[Path]:
    out: list[Path] = []
    for p in sorted(folder.rglob("*")):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            out.append(p)
    return out


def save_block(
    img_bgr: np.ndarray,
    mask_full: np.ndarray,
    x: int,
    y: int,
    bw: int,
    bh: int,
    out_dir: Path,
    stem: str,
    tag: str,
    save_rgba: bool,
) -> None:
    h0, w0 = img_bgr.shape[:2]
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(w0, x + bw)
    y2 = min(h0, y + bh)
    crop = img_bgr[y1:y2, x1:x2].copy()
    m_crop = (mask_full[y1:y2, x1:x2] > 0).astype(np.float32)
    if crop.size == 0:
        return
    m3 = np.clip(m_crop[..., None], 0.0, 1.0)
    masked = (crop.astype(np.float32) * m3).astype(np.uint8)
    cv2.imwrite(str(out_dir / f"{stem}_{tag}_bbox_crop.png"), crop)
    cv2.imwrite(str(out_dir / f"{stem}_{tag}_masked.png"), masked)
    cv2.imwrite(
        str(out_dir / f"{stem}_{tag}_mask.png"),
        (m_crop * 255).astype(np.uint8),
    )
    if save_rgba:
        bgra = cv2.cvtColor(crop, cv2.COLOR_BGR2BGRA)
        bgra[:, :, 3] = (np.clip(m_crop, 0, 1) * 255).astype(np.uint8)
        cv2.imwrite(str(out_dir / f"{stem}_{tag}_rgba.png"), bgra)


def process_strip_mode(
    img_bgr: np.ndarray,
    out_dir: Path,
    stem: str,
    args: argparse.Namespace,
) -> int:
    mask, box = extract_uniform_roi(
        img_bgr,
        args.roi_sat_max,
        args.roi_val_min,
        args.roi_min_area_ratio,
        args.roi_shrink_ratio,
        close_ksize=args.close_ksize,
        open_ksize=args.open_ksize,
    )
    cv2.imwrite(str(out_dir / f"{stem}_uniform_mask.png"), mask)
    if box is None:
        cv2.imwrite(str(out_dir / f"{stem}_original.png"), img_bgr)
        return 0

    x0, y0, rw, rh = box
    vis = img_bgr.copy()
    cv2.rectangle(vis, (x0, y0), (x0 + rw, y0 + rh), (0, 255, 0), 2)
    cv2.imwrite(str(out_dir / f"{stem}_overlay.png"), vis)
    save_block(
        img_bgr,
        mask,
        x0,
        y0,
        rw,
        rh,
        out_dir,
        stem,
        "strip",
        save_rgba=not args.no_rgba,
    )
    return 1


def process_components_mode(
    img_bgr: np.ndarray,
    out_dir: Path,
    stem: str,
    args: argparse.Namespace,
) -> int:
    mask, boxes = extract_uniform_components(
        img_bgr,
        args.roi_sat_max,
        args.roi_val_min,
        args.component_min_area_ratio,
        close_ksize=args.close_ksize,
        open_ksize=args.open_ksize,
        min_aspect_w_over_h=args.min_aspect
        if args.min_aspect and args.min_aspect > 0
        else None,
        max_boxes=args.max_boxes,
    )
    cv2.imwrite(str(out_dir / f"{stem}_uniform_mask.png"), mask)
    if not boxes:
        cv2.imwrite(str(out_dir / f"{stem}_original.png"), img_bgr)
        return 0

    vis = img_bgr.copy()
    for i, (x, y, bw, bh) in enumerate(boxes):
        cv2.rectangle(vis, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
        cv2.putText(
            vis,
            str(i),
            (x, max(0, y - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )
    cv2.imwrite(str(out_dir / f"{stem}_overlay.png"), vis)

    for i, (x, y, bw, bh) in enumerate(boxes):
        tag = f"block_{i:02d}"
        save_block(
            img_bgr,
            mask,
            x,
            y,
            bw,
            bh,
            out_dir,
            stem,
            tag,
            save_rgba=not args.no_rgba,
        )
    return len(boxes)


def process_one(image_path: Path, output_root: Path, args: argparse.Namespace) -> int:
    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        print(f"Skip (unreadable): {image_path}")
        return 0

    stem = image_path.stem
    out_dir = output_root / stem
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.strategy == "strip":
        n = process_strip_mode(img_bgr, out_dir, stem, args)
    else:
        n = process_components_mode(img_bgr, out_dir, stem, args)

    strat = args.strategy
    print(f"{image_path.name}: strategy={strat} -> {n} segment(s), dir={out_dir}")
    return n


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Classical CV segmentation: bright uniform regions (HSV + morphology)."
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input image or directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs_traditional_seg"),
        help="Output root (one subfolder per image). Default: outputs_traditional_seg",
    )
    parser.add_argument(
        "--strategy",
        choices=["strip", "components"],
        default="strip",
        help="strip: single ROI via largest inscribed rect in mask (same idea as ocr_segment). "
        "components: each connected component as a block.",
    )
    parser.add_argument("--roi-sat-max", type=int, default=70, help="Max HSV saturation.")
    parser.add_argument("--roi-val-min", type=int, default=95, help="Min HSV value (brightness).")
    parser.add_argument(
        "--roi-min-area-ratio",
        type=float,
        default=0.12,
        help="strip mode: min ROI area vs full image.",
    )
    parser.add_argument(
        "--roi-shrink-ratio",
        type=float,
        default=0.04,
        help="strip mode: shrink inscribed rect inward (fraction of w/h).",
    )
    parser.add_argument(
        "--component-min-area-ratio",
        type=float,
        default=0.002,
        help="components mode: min CC area vs full image.",
    )
    parser.add_argument(
        "--min-aspect",
        type=float,
        default=0.0,
        help="components mode: if >0, keep only boxes with width/height >= this (horizontal strips).",
    )
    parser.add_argument(
        "--max-boxes",
        type=int,
        default=32,
        help="components mode: max number of boxes to export.",
    )
    parser.add_argument(
        "--close-ksize",
        type=int,
        default=15,
        help="Morphology close kernel size (odd >= 3).",
    )
    parser.add_argument(
        "--open-ksize",
        type=int,
        default=5,
        help="Morphology open kernel size (odd >= 3).",
    )
    parser.add_argument(
        "--no-rgba",
        action="store_true",
        help="Do not write transparent PNG crops.",
    )
    args = parser.parse_args()

    inp = args.input.resolve()
    out_root = args.output_dir.resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    if inp.is_file():
        paths = [inp]
    elif inp.is_dir():
        paths = iter_images(inp)
        if not paths:
            raise SystemExit(f"No images under {inp}")
    else:
        raise SystemExit(f"Not a file or directory: {inp}")

    total = 0
    for p in paths:
        total += process_one(p, out_root, args)
    print(f"Done. Images: {len(paths)}, total segments: {total}. Root: {out_root}")


if __name__ == "__main__":
    main()
