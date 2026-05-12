#!/usr/bin/env python3
"""
Full-image **text detection** on the original photo (no strip ROI / no cv_roi crop).

**Default: detection only** — uses PaddleX `PP-OCRv5_*_det` via `paddlex.create_model` (no recognition
model is loaded). Boxes with detector score below `--min-det-score` (default **0.8**) are dropped from PNG + JSON.

Optional `--with-recognition` loads the full PaddleOCR pipeline (det + rec) like before.

Intermediate PNGs per image under --output-dir:

  {stem}_01_input.png
  {stem}_02_det_boxes_only.png
  {stem}_03_det_overlay.png
  {stem}_04_compare_lr.png
  {stem}_full_image_det_vis.png

Example:
  python paddle_full_image_detect.py --input data3/20260429154940.jpg --output-dir outputs_full_det
  python paddle_full_image_detect.py --input data3 --with-recognition --output-dir outputs_full_ocr
  python paddle_full_image_detect.py --input data3 --output-dir out --use-cuda --gpu-id 0
  python paddle_full_image_detect.py --input data3 --output-dir out --json-copy-dir all_det_json
  python paddle_full_image_detect.py --input data3 --output-dir out --with-recognition --json-only
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import paddle

PADDLEX_AVAILABLE = False

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def resolve_paddle_device(
    *, use_cuda: bool, gpu_id: int
) -> tuple[str | None, dict[str, Any]]:
    """
    Map CLI flags to paddlex `device` and collect metadata for JSON output.

    Returns:
        (device_kwarg_for_create_model, static_device_info)
        device_kwarg: None = leave default (often GPU if available), else 'cpu' or 'gpu:N'.
    """
    compiled = bool(paddle.device.is_compiled_with_cuda())
    try:
        cuda_count = int(paddle.device.cuda.device_count()) if compiled else 0
    except Exception:
        cuda_count = 0

    info: dict[str, Any] = {
        "use_cuda_requested": use_cuda,
        "paddle_compiled_with_cuda": compiled,
        "cuda_device_count": cuda_count,
    }

    if use_cuda:
        info["gpu_id_requested"] = int(gpu_id)
        if not compiled or cuda_count == 0:
            info["device_arg_passed_to_model"] = "cpu"
            info["fallback_reason"] = (
                "paddle not built with CUDA"
                if not compiled
                else "no CUDA GPU visible to Paddle"
            )
            return "cpu", info
        gid = max(0, min(int(gpu_id), cuda_count - 1))
        if gid != int(gpu_id):
            info["gpu_id_effective"] = gid
        dev = f"gpu:{gid}"
        info["device_arg_passed_to_model"] = dev
        return dev, info

    info["device_arg_passed_to_model"] = None
    return None, info


def finalize_device_json(static_info: dict[str, Any]) -> dict[str, Any]:
    """Merge Paddle runtime device into export dict (call after model(s) are created)."""
    out = dict(static_info)
    try:
        rt = str(paddle.device.get_device())
    except Exception as e:
        rt = ""
        out["paddle_runtime_device_error"] = str(e)
    out["paddle_runtime_device"] = rt
    rt_lower = rt.lower()
    out["cuda_acceleration_active"] = rt_lower.startswith("gpu") or ":gpu" in rt_lower
    return out


def iter_images(folder: Path) -> list[Path]:
    out: list[Path] = []
    for p in sorted(folder.rglob("*")):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            out.append(p)
    return out


def _poly_to_points(poly: np.ndarray) -> np.ndarray | None:
    if poly.size == 0:
        return None
    if poly.ndim == 1:
        if poly.shape[0] >= 8:
            return poly.reshape(-1, 2).astype(np.int32)
        if poly.shape[0] >= 4:
            x1, y1, x2, y2 = [int(poly[i]) for i in range(4)]
            return np.array(
                [[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.int32
            )
        return None
    return poly.astype(np.int32)


def predict_boxes_det_only(
    det_model,
    image_bgr: np.ndarray,
    *,
    det_limit_side_len: int,
    det_thresh: float,
    det_box_thresh: float,
    det_unclip_ratio: float,
    max_side_limit: int = 4000,
    min_det_score: float = 0.8,
) -> list[dict]:
    """Run PP-OCR text detector only; returns polygons + detector confidence."""
    from paddleocr import PaddleOCR
    
    if isinstance(det_model, PaddleOCR):
        result = det_model.ocr(image_bgr, cls=False)
        if not result or len(result) == 0:
            return []
        
        boxes: list[dict] = []
        for line in result[0]:
            if line is None:
                continue
            pts = np.array(line[0], dtype=np.float32)
            text = line[1][0] if len(line) > 1 else ""
            score = float(line[1][1]) if len(line) > 1 else 0.0
            
            if score < min_det_score:
                continue
            
            xs = pts[:, 0]
            ys = pts[:, 1]
            boxes.append({
                "text": text,
                "det_score": score,
                "score": score,
                "poly": pts.astype(np.int32).tolist(),
                "bbox_xyxy": [
                    float(xs.min()),
                    float(ys.min()),
                    float(xs.max()),
                    float(ys.max()),
                ],
            })
        return boxes
    
    gen = det_model(
        [image_bgr],
        limit_side_len=det_limit_side_len,
        limit_type="max",
        thresh=det_thresh,
        box_thresh=det_box_thresh,
        unclip_ratio=det_unclip_ratio,
        max_side_limit=max_side_limit,
    )
    first = next(iter(gen), None)
    if first is None:
        return []

    polys_raw = first["dt_polys"]
    scores_raw = first["dt_scores"]
    if isinstance(polys_raw, np.ndarray):
        if polys_raw.ndim == 3:
            polys = [polys_raw[i] for i in range(polys_raw.shape[0])]
        elif polys_raw.ndim == 2:
            polys = [polys_raw]
        else:
            polys = []
    elif polys_raw is None:
        polys = []
    else:
        polys = list(polys_raw)

    if isinstance(scores_raw, np.ndarray):
        scores = [float(scores_raw[i]) for i in range(scores_raw.shape[0])]
    elif scores_raw is None:
        scores = []
    else:
        scores = [float(x) for x in list(scores_raw)]
    boxes: list[dict] = []
    for i, poly in enumerate(polys):
        pts = _poly_to_points(np.asarray(poly))
        if pts is None:
            continue
        xs = pts[:, 0].astype(np.float32)
        ys = pts[:, 1].astype(np.float32)
        det_sc = float(scores[i]) if i < len(scores) else 0.0
        if det_sc < min_det_score:
            continue
        boxes.append(
            {
                "text": "",
                "det_score": det_sc,
                "score": det_sc,
                "poly": pts.tolist(),
                "bbox_xyxy": [
                    float(xs.min()),
                    float(ys.min()),
                    float(xs.max()),
                    float(ys.max()),
                ],
            }
        )
    return boxes


def predict_boxes_full_ocr(
    ocr_engine,
    image_bgr: np.ndarray,
) -> list[dict]:
    """Full PaddleOCR predict (det + rec). Lazy-import caller supplies engine."""
    if hasattr(ocr_engine, 'predict'):
        result = ocr_engine.predict(image_bgr)
        first = result[0] if result else {}
        texts = list(first.get("rec_texts", []) or [])
        scores = list(first.get("rec_scores", []) or [])
        polys = list(first.get("rec_polys", []) or [])
        if not polys:
            polys = list(first.get("rec_boxes", []) or [])
    else:
        result = ocr_engine.ocr(image_bgr, cls=True)
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

    boxes: list[dict] = []
    n = min(len(texts), len(scores), len(polys))
    for i in range(n):
        poly = np.asarray(polys[i])
        pts = _poly_to_points(poly)
        if pts is None:
            continue
        xs = pts[:, 0].astype(np.float32)
        ys = pts[:, 1].astype(np.float32)
        boxes.append(
            {
                "text": str(texts[i]).strip(),
                "det_score": None,
                "score": float(scores[i]),
                "poly": pts.tolist(),
                "bbox_xyxy": [
                    float(xs.min()),
                    float(ys.min()),
                    float(xs.max()),
                    float(ys.max()),
                ],
            }
        )
    return boxes


def draw_boxes_only_canvas(
    height: int,
    width: int,
    boxes: list[dict],
) -> np.ndarray:
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)
    for b in boxes:
        pts = np.asarray(b["poly"], dtype=np.int32)
        if pts.ndim != 2 or pts.shape[0] < 2:
            continue
        cv2.polylines(canvas, [pts.reshape(-1, 1, 2)], True, (0, 255, 0), 2)
    return canvas


def draw_detections(
    image_bgr: np.ndarray,
    boxes: list[dict],
    *,
    draw_label: bool,
    det_only: bool,
) -> np.ndarray:
    vis = image_bgr.copy()
    h, w = vis.shape[:2]
    for i, b in enumerate(boxes, start=1):
        pts = np.asarray(b["poly"], dtype=np.int32)
        if pts.ndim != 2 or pts.shape[0] < 2:
            continue
        cv2.polylines(vis, [pts.reshape(-1, 1, 2)], True, (0, 255, 0), 2)
        if not draw_label:
            continue
        x1 = int(np.clip(np.min(pts[:, 0]), 0, w - 1))
        y1 = int(np.clip(np.min(pts[:, 1]) - 4, 0, h - 1))
        if det_only or not (b.get("text") or "").strip():
            raw_ds = b.get("det_score")
            if raw_ds is None:
                raw_ds = b.get("score")
            if raw_ds is None:
                raw_ds = 0.0
            ds = float(raw_ds)
            label = f"{i}: det={ds:.2f}"
        else:
            label = f"{i}: {b['text'][:40]}"
        cv2.putText(
            vis,
            label,
            (x1, y1),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 200, 0),
            1,
            cv2.LINE_AA,
        )
    return vis


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Full-image PP-OCR **text detection** (default: det-only, no recognition)."
    )
    parser.add_argument("--input", type=Path, required=True, help="Image file or folder.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs_full_det"),
        help="Output directory. Default: outputs_full_det",
    )
    parser.add_argument(
        "--with-recognition",
        action="store_true",
        help="Also run recognition (loads det+rec; slower). Default is detection only.",
    )
    parser.add_argument(
        "--ocr-model-size",
        choices=["mobile", "server"],
        default="mobile",
        help="PP-OCRv5 text_detection model variant (and rec if --with-recognition).",
    )
    parser.add_argument("--ocr-lang", default="ch", help="Only used with --with-recognition.")
    parser.add_argument(
        "--use-angle-cls",
        action="store_true",
        default=True,
        help="Only used with --with-recognition.",
    )
    parser.add_argument("--det-limit-side-len", type=int, default=2560)
    parser.add_argument("--det-thresh", type=float, default=0.2)
    parser.add_argument("--det-box-thresh", type=float, default=0.35)
    parser.add_argument("--det-unclip-ratio", type=float, default=2.0)
    parser.add_argument(
        "--min-det-score",
        type=float,
        default=0.5,
        help="det-only mode: drop boxes with detector score below this (default 0.8). "
        "Ignored for --with-recognition (det scores not exposed per box).",
    )
    parser.add_argument(
        "--no-labels",
        action="store_true",
        help="Draw boxes only (no index / score / text labels).",
    )
    parser.add_argument("--no-json", action="store_true")
    parser.add_argument(
        "--json-copy-dir",
        type=Path,
        default=None,
        help="If set, copy each *_det_boxes.json into this folder as well (same filename).",
    )
    parser.add_argument(
        "--no-intermediate-png",
        action="store_true",
        help="Skip 01..04 intermediate PNGs; still write *_full_image_det_vis.png + json.",
    )
    parser.add_argument(
        "--json-only",
        action="store_true",
        help="No PNG output (skips all visualization). Only *_det_boxes.json; "
        "batch runs also write summary.json. Incompatible with --no-json.",
    )
    parser.add_argument(
        "--use-cuda",
        action="store_true",
        help="Request GPU via PaddleX (e.g. device=gpu:0). Falls back to CPU if unavailable.",
    )
    parser.add_argument(
        "--gpu-id",
        type=int,
        default=0,
        help="CUDA device index when --use-cuda (default: 0).",
    )
    args = parser.parse_args()
    if args.json_only and args.no_json:
        raise SystemExit("Use either --json-only (JSON only) or --no-json, not both.")

    inp = args.input.resolve()
    out_root = args.output_dir.resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    device_kw, device_info_static = resolve_paddle_device(
        use_cuda=args.use_cuda, gpu_id=args.gpu_id
    )
    
    from paddleocr import PaddleOCR
    det_name = f"PP-OCRv5_{args.ocr_model_size}_det"
    det_model = PaddleOCR(
        lang=args.ocr_lang,
        use_angle_cls=False,
        use_gpu=args.use_cuda,
        gpu_id=args.gpu_id,
        det_limit_side_len=args.det_limit_side_len,
    )
    print(f"Using PaddleOCR with {'GPU' if args.use_cuda else 'CPU'} mode")

    ocr_device: str | None = None
    if args.use_cuda:
        ocr_device = device_info_static.get("device_arg_passed_to_model")
    else:
        ocr_device = "cpu"
    ocr_engine = None
    if args.with_recognition:
        from ocr_segment import build_ocr_engine

        ocr_engine = build_ocr_engine(
            args.ocr_lang,
            args.use_angle_cls,
            args.ocr_model_size,
            args.det_limit_side_len,
            args.det_thresh,
            args.det_box_thresh,
            args.det_unclip_ratio,
            device=ocr_device,
        )

    device_info = finalize_device_json(device_info_static)

    json_copy_root: Path | None = None
    if args.json_copy_dir is not None and not args.no_json:
        json_copy_root = args.json_copy_dir.resolve()
        json_copy_root.mkdir(parents=True, exist_ok=True)

    paths: list[Path]
    if inp.is_file():
        paths = [inp]
    elif inp.is_dir():
        paths = iter_images(inp)
        if not paths:
            raise SystemExit(f"No images under {inp}")
    else:
        raise SystemExit(f"Not a file or directory: {inp}")

    summary: list[dict] = []
    mode = "det_only" if not args.with_recognition else "det_plus_rec"
    for p in paths:
        t0 = time.perf_counter()
        img = cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_COLOR)
        t_read = time.perf_counter() - t0
        if img is None:
            print(f"Skip unreadable: {p}")
            continue

        t0 = time.perf_counter()
        if args.with_recognition:
            boxes = predict_boxes_full_ocr(ocr_engine, img)
        else:
            boxes = predict_boxes_det_only(
                det_model,
                img,
                det_limit_side_len=args.det_limit_side_len,
                det_thresh=args.det_thresh,
                det_box_thresh=args.det_box_thresh,
                det_unclip_ratio=args.det_unclip_ratio,
                min_det_score=args.min_det_score,
            )
        t_infer = time.perf_counter() - t0

        h0, w0 = img.shape[:2]
        if args.json_only:
            t_render = 0.0
        else:
            t0 = time.perf_counter()
            vis = draw_detections(
                img,
                boxes,
                draw_label=not args.no_labels,
                det_only=not args.with_recognition,
            )
            boxes_only = None
            compare = None
            if not args.no_intermediate_png:
                boxes_only = draw_boxes_only_canvas(h0, w0, boxes)
                compare = np.hstack([img, vis])
            t_render = time.perf_counter() - t0
        stem = p.stem

        out_png = out_root / f"{stem}_full_image_det_vis.png"
        t0 = time.perf_counter()
        if not args.json_only:
            if not args.no_intermediate_png:
                cv2.imwrite(str(out_root / f"{stem}_01_input.png"), img)
                cv2.imwrite(str(out_root / f"{stem}_02_det_boxes_only.png"), boxes_only)
                cv2.imwrite(str(out_root / f"{stem}_03_det_overlay.png"), vis)
                cv2.imwrite(str(out_root / f"{stem}_04_compare_lr.png"), compare)
            cv2.imwrite(str(out_png), vis)
        json_path = out_root / f"{stem}_det_boxes.json"
        if not args.no_json:
            pass  # written after record dict is built
        t_write_part1 = time.perf_counter() - t0

        timing = {
            "read": round(float(t_read), 4),
            "infer": round(float(t_infer), 4),
            "render": round(float(t_render), 4),
            "write": round(float(t_write_part1), 4),
            "total": round(
                float(t_read + t_infer + t_render + t_write_part1), 4
            ),
        }

        record = {
            "file": p.name,
            "mode": mode,
            "det_model": det_name,
            "width": int(w0),
            "height": int(h0),
            "box_count": len(boxes),
            "infer_seconds": timing["infer"],
            "timing_seconds": timing,
            "device": device_info,
            "boxes": boxes,
            "debug_pngs": []
            if args.json_only or args.no_intermediate_png
            else [
                f"{stem}_01_input.png",
                f"{stem}_02_det_boxes_only.png",
                f"{stem}_03_det_overlay.png",
                f"{stem}_04_compare_lr.png",
                f"{stem}_full_image_det_vis.png",
            ],
        }
        summary.append(record)
        if not args.no_json:
            tj0 = time.perf_counter()
            with json_path.open("w", encoding="utf-8") as f:
                json.dump(record, f, ensure_ascii=False, indent=2)
            if json_copy_root is not None:
                shutil.copy2(json_path, json_copy_root / json_path.name)
            timing["write"] = round(float(timing["write"] + (time.perf_counter() - tj0)), 4)
            timing["total"] = round(
                float(timing["read"] + timing["infer"] + timing["render"] + timing["write"]),
                4,
            )
            record["timing_seconds"] = timing

        if args.json_only:
            print(
                f"{p.name}: {len(boxes)} box(es) [{mode}] "
                f"read={timing['read']:.3f}s infer={timing['infer']:.3f}s "
                f"render={timing['render']:.3f}s write={timing['write']:.3f}s "
                f"total={timing['total']:.3f}s -> {stem}_det_boxes.json"
            )
        elif args.no_intermediate_png:
            print(
                f"{p.name}: {len(boxes)} box(es) [{mode}] "
                f"read={timing['read']:.3f}s infer={timing['infer']:.3f}s "
                f"render={timing['render']:.3f}s write={timing['write']:.3f}s "
                f"total={timing['total']:.3f}s -> {out_png.name}"
            )
        else:
            print(
                f"{p.name}: {len(boxes)} box(es) [{mode}] "
                f"read={timing['read']:.3f}s infer={timing['infer']:.3f}s "
                f"render={timing['render']:.3f}s write={timing['write']:.3f}s "
                f"total={timing['total']:.3f}s -> {stem}_01..04 + {out_png.name}"
            )

    if len(summary) > 1:
        with (out_root / "summary.json").open("w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"Done. Root: {out_root}")
    if json_copy_root is not None:
        print(f"JSON copies: {json_copy_root}")


if __name__ == "__main__":
    main()
