#!/usr/bin/env python3
"""
Full-image **PaddleOCR** (PP-OCRv5 detection + recognition) on the original photo (no strip ROI).

Writes polygons, recognized text, and recognition scores per box to JSON and optional debug PNGs.

Intermediate PNGs per image under --output-dir:

  {stem}_01_input.png
  {stem}_02_det_boxes_only.png
  {stem}_03_det_overlay.png
  {stem}_04_compare_lr.png
  {stem}_full_image_det_vis.png

Example:
  python paddle_full_image_detect.py --input data3/20260429154940.jpg --output-dir outputs
  python paddle_full_image_detect.py --input data3 --output-dir outputs
  python paddle_full_image_detect.py --input data3 --output-dir out --use-cuda --gpu-id 0
  python paddle_full_image_detect.py --input data3 --output-dir out --json-copy-dir all_det_json
  python paddle_full_image_detect.py --input data3 --output-dir out --json-only
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import paddle
from paddleocr import PaddleOCR

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def coerce_box_text(val: Any) -> str:
    """Turn OCR box ``text`` into ``str`` without NumPy truth-value checks."""
    if val is None:
        return ""
    if isinstance(val, np.ndarray):
        if val.size == 0:
            return ""
        if val.ndim == 0:
            return str(val.item()).strip()
        return str(val).strip()
    return str(val).strip()


def coerce_box_score(val: Any) -> float:
    """Turn OCR box ``score`` into ``float`` without ``ndarray or 0`` ambiguity."""
    if val is None:
        return 0.0
    if isinstance(val, np.ndarray):
        if val.size == 0:
            return 0.0
        return float(np.asarray(val, dtype=np.float64).ravel()[0])
    try:
        return float(val)
    except (TypeError, ValueError):
        return 0.0


def _paddle_ocr_seq(val: Any) -> list[Any]:
    """Normalize PaddleOCR ``predict`` list fields; never ``bool(ndarray)``."""
    if val is None:
        return []
    if isinstance(val, np.ndarray):
        if val.ndim == 0:
            return [val.item()]
        return val.tolist()
    if isinstance(val, (list, tuple)):
        return list(val)
    return [val]


def resolve_paddle_device(
    *, use_cuda: bool, gpu_id: int
) -> tuple[str | None, dict[str, Any]]:
    """
    Map CLI flags to PaddleOCR `device` and collect metadata for JSON output.

    Returns:
        (device_kwarg_for_paddleocr, static_device_info)
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
    """Construct PaddleOCR (PP-OCRv5 det+rec) for full-image predict."""
    det_model = f"PP-OCRv5_{ocr_model_size}_det"
    rec_model = f"PP-OCRv5_{ocr_model_size}_rec"
    kwargs: dict[str, Any] = {}
    if device is not None:
        kwargs["device"] = device
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


def prepare_bgr_for_predict(image_bgr: np.ndarray) -> np.ndarray:
    """
  Normalize camera / OpenCV buffers for ``PaddleOCR.predict``.

  Live grab buffers may be non-contiguous or non-uint8; ``cv2.imwrite`` + ``imread``
  often hides that. UI and CLI should call this before predict.
    """
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


def predict_boxes(
    ocr_engine: PaddleOCR,
    image_bgr: np.ndarray,
    *,
    return_debug: bool = False,
) -> list[dict] | tuple[list[dict], dict[str, int]]:
    """Run PaddleOCR predict (det + rec); returns boxes with text and recognition score.

    Drops boxes whose stripped text has at most 3 non-whitespace characters.
  If ``return_debug`` is True, also return raw recognition count before filtering.
    """
    image_bgr = prepare_bgr_for_predict(image_bgr)
    result = ocr_engine.predict(image_bgr)
    first = result[0] if result else {}
    texts = _paddle_ocr_seq(first.get("rec_texts", []))
    scores = _paddle_ocr_seq(first.get("rec_scores", []))
    polys = _paddle_ocr_seq(first.get("rec_polys", []))
    if not polys:
        polys = _paddle_ocr_seq(first.get("rec_boxes", []))

    raw_rec_count = min(len(texts), len(scores), len(polys))
    boxes: list[dict] = []
    n = raw_rec_count
    for i in range(n):
        text = coerce_box_text(texts[i])
        # 有效字符：去掉空白后的长度；仅保留超过 3 个有效字符的框
        effective = "".join(text.split())
        if len(effective) <= 3:
            continue
        poly = np.asarray(polys[i])
        pts = _poly_to_points(poly)
        if pts is None:
            continue
        xs = pts[:, 0].astype(np.float32)
        ys = pts[:, 1].astype(np.float32)
        boxes.append(
            {
                "text": text,
                "det_score": None,
                "score": coerce_box_score(scores[i]),
                "poly": pts.tolist(),
                "bbox_xyxy": [
                    float(xs.min()),
                    float(ys.min()),
                    float(xs.max()),
                    float(ys.max()),
                ],
            }
        )
    if return_debug:
        return boxes, {
            "raw_rec_count": int(raw_rec_count),
            "boxes_after_filter": len(boxes),
        }
    return boxes


def _load_cjk_label_font(size: int = 18) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """TrueType font that can render Chinese labels (``cv2.putText`` cannot)."""
    if sys.platform == "win32":
        windir = os.environ.get("WINDIR", r"C:\Windows")
        candidates = [
            os.path.join(windir, "Fonts", "msyh.ttc"),
            os.path.join(windir, "Fonts", "msyhbd.ttc"),
            os.path.join(windir, "Fonts", "simhei.ttf"),
            os.path.join(windir, "Fonts", "simsun.ttc"),
        ]
    else:
        candidates = [
            "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
            "/System/Library/Fonts/PingFang.ttc",
            "/System/Library/Fonts/STHeiti Light.ttc",
        ]
    for path in candidates:
        if path and os.path.isfile(path):
            try:
                return ImageFont.truetype(path, size=size)
            except OSError:
                continue
    return ImageFont.load_default()


def _draw_label_on_bgr(
    vis_bgr: np.ndarray,
    text: str,
    x: int,
    y: int,
    *,
    font_size: int = 18,
) -> np.ndarray:
    """Draw UTF-8 label with PIL; ``y`` is the desired top edge (clamped in-image)."""
    if not text:
        return vis_bgr
    h, w = vis_bgr.shape[:2]
    font = _load_cjk_label_font(font_size)
    rgb = cv2.cvtColor(vis_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    draw = ImageDraw.Draw(pil)
    # textbbox: (left, top, right, bottom) relative to anchor
    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    tx = int(np.clip(x, 0, max(0, w - tw - 1)))
    ty = int(np.clip(y, 0, max(0, h - th - 1)))
    pad = 2
    draw.rectangle(
        (tx - pad, ty - pad, tx + tw + pad, ty + th + pad),
        fill=(0, 0, 0),
    )
    draw.text((tx, ty), text, font=font, fill=(0, 220, 0))
    return cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)


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
        txt = coerce_box_text(b.get("text"))
        if txt:
            label = f"{i}: {txt[:40]}"
        else:
            sc = coerce_box_score(b.get("score"))
            label = f"{i}: (empty) rec={sc:.2f}"
        font_px = max(14, min(22, int(round(h / 120))))
        vis = _draw_label_on_bgr(vis, label, x1, max(0, y1 - font_px - 4), font_size=font_px)
    return vis


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Full-image PaddleOCR (PP-OCRv5 detection + recognition)."
    )
    parser.add_argument("--input", type=Path, required=True, help="Image file or folder.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Output directory. Default: outputs",
    )
    parser.add_argument(
        "--ocr-model-size",
        choices=["mobile", "server"],
        default="mobile",
        help="PP-OCRv5 detector and recognizer variant (mobile or server).",
    )
    parser.add_argument("--ocr-lang", default="ch", help="PaddleOCR lang (e.g. ch, en).")
    parser.add_argument(
        "--use-angle-cls",
        action="store_true",
        default=True,
        help="Enable text-line orientation (angle classification).",
    )
    parser.add_argument("--det-limit-side-len", type=int, default=2560)
    parser.add_argument("--det-thresh", type=float, default=0.10)
    parser.add_argument("--det-box-thresh", type=float, default=0.35)
    parser.add_argument("--det-unclip-ratio", type=float, default=1.8)
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
        help="Request GPU for PaddleOCR (e.g. device=gpu:0). Falls back to CPU if unavailable.",
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

    _, device_info_static = resolve_paddle_device(
        use_cuda=args.use_cuda, gpu_id=args.gpu_id
    )

    ocr_device: str | None = None
    if args.use_cuda:
        ocr_device = device_info_static.get("device_arg_passed_to_model")
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

    det_name = f"PP-OCRv5_{args.ocr_model_size}_det"
    rec_name = f"PP-OCRv5_{args.ocr_model_size}_rec"
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
    mode = "paddleocr"
    for p in paths:
        t0 = time.perf_counter()
        img = cv2.imread(str(p))
        t_read = time.perf_counter() - t0
        if img is None:
            print(f"Skip unreadable: {p}")
            continue

        t0 = time.perf_counter()
        boxes = predict_boxes(ocr_engine, img)
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
            "rec_model": rec_name,
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
