# -*- coding: utf-8 -*-
"""HSV threshold box demo: tune H/S/V ranges and draw bounding boxes."""

from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
from dataclasses import asdict, dataclass
from typing import Any, Optional

import cv2
import numpy as np

from label_cropper import imread_unicode, imwrite_unicode


@dataclass
class HsvThresholds:
    h_min: int = 0
    h_max: int = 179
    s_min: int = 0
    s_max: int = 30
    v_min: int = 100
    v_max: int = 255
    close_ksize: int = 15
    open_ksize: int = 5
    min_area: int = 5000


@dataclass
class BoxResult:
    x: int
    y: int
    w: int
    h: int
    area: float

    def as_xyxy(self) -> list[int]:
        return [self.x, self.y, self.x + self.w, self.y + self.h]


def build_hsv_mask(img_bgr: np.ndarray, th: HsvThresholds) -> np.ndarray:
    """Mask pixels inside [h_min,h_max] x [s_min,s_max] x [v_min,v_max]."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    lower = np.array([th.h_min, th.s_min, th.v_min], dtype=np.uint8)
    upper = np.array([th.h_max, th.s_max, th.v_max], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)

    if th.close_ksize > 0:
        k = th.close_ksize | 1
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    if th.open_ksize > 0:
        k = th.open_ksize | 1
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return mask


def find_boxes(mask: np.ndarray, min_area: int) -> list[BoxResult]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes: list[BoxResult] = []
    for cnt in contours:
        area = float(cv2.contourArea(cnt))
        if area < min_area:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        boxes.append(BoxResult(x=x, y=y, w=w, h=h, area=area))
    boxes.sort(key=lambda b: b.area, reverse=True)
    return boxes


def draw_overlay(
    img_bgr: np.ndarray,
    mask: np.ndarray,
    boxes: list[BoxResult],
    th: HsvThresholds,
) -> np.ndarray:
    vis = img_bgr.copy()
    tint = np.zeros_like(vis)
    tint[:, :, 1] = mask
    vis = cv2.addWeighted(vis, 0.72, tint, 0.28, 0)

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box.as_xyxy()
        color = (0, 255, 255) if i == 0 else (0, 180, 255)
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        label = f"#{i + 1} area={int(box.area)}"
        cv2.putText(
            vis,
            label,
            (x1, max(y1 - 8, 16)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )

    info = (
        f"H[{th.h_min},{th.h_max}] S[{th.s_min},{th.s_max}] "
        f"V[{th.v_min},{th.v_max}] boxes={len(boxes)}"
    )
    cv2.putText(
        vis,
        info,
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return vis


def process_image(img_bgr: np.ndarray, th: HsvThresholds) -> dict[str, Any]:
    mask = build_hsv_mask(img_bgr, th)
    boxes = find_boxes(mask, th.min_area)
    overlay = draw_overlay(img_bgr, mask, boxes, th)
    return {
        "thresholds": asdict(th),
        "boxes": [asdict(b) for b in boxes],
        "mask": mask,
        "overlay": overlay,
    }


def _save_result(out_dir: str, stem: str, result: dict[str, Any]) -> None:
    os.makedirs(out_dir, exist_ok=True)
    imwrite_unicode(os.path.join(out_dir, f"{stem}_hsv_mask.png"), result["mask"])
    imwrite_unicode(os.path.join(out_dir, f"{stem}_hsv_boxes.png"), result["overlay"])
    meta = {"thresholds": result["thresholds"], "boxes": result["boxes"]}
    json_path = os.path.join(out_dir, f"{stem}_hsv_boxes.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def _pick_image_macos(initial_dir: Optional[str] = None) -> Optional[str]:
    lines: list[str] = []
    if initial_dir and os.path.isdir(initial_dir):
        esc = initial_dir.replace("\\", "\\\\").replace('"', '\\"')
        lines.append(f'set defaultLocation to POSIX file "{esc}"')
        lines.append(
            'set picked to choose file with prompt "Select image" '
            "default location defaultLocation "
            'of type {"public.image", "jpg", "jpeg", "png", "bmp", "webp"}'
        )
    else:
        lines.append(
            'set picked to choose file with prompt "Select image" '
            'of type {"public.image", "jpg", "jpeg", "png", "bmp", "webp"}'
        )
    lines.append("return POSIX path of picked")
    script = "\n".join(lines)
    try:
        proc = subprocess.run(
            ["osascript", "-e", script],
            capture_output=True,
            text=True,
            timeout=300,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    path = proc.stdout.strip()
    return path if path else None


def _pick_image_tk_subprocess(initial_dir: Optional[str] = None) -> Optional[str]:
    """Run tk filedialog in a child process to avoid OpenCV/Tk conflicts."""
    init = (initial_dir or os.getcwd()).replace("\\", "\\\\").replace('"', '\\"')
    code = f"""
import tkinter as tk
from tkinter import filedialog
root = tk.Tk()
root.withdraw()
root.attributes("-topmost", True)
path = filedialog.askopenfilename(
    title="Select image",
    initialdir=r"{init}",
    filetypes=[
        ("Image files", "*.jpg *.jpeg *.png *.bmp *.webp"),
        ("All files", "*.*"),
    ],
)
print(path or "")
root.destroy()
"""
    try:
        proc = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=300,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    path = proc.stdout.strip()
    return path if path else None


def pick_image_file(initial_dir: Optional[str] = None) -> Optional[str]:
    if sys.platform == "darwin":
        path = _pick_image_macos(initial_dir)
        if path:
            return path
    if sys.platform.startswith("linux"):
        try:
            proc = subprocess.run(
                [
                    "zenity",
                    "--file-selection",
                    "--title=Select image",
                    "--file-filter=Images | *.jpg *.jpeg *.png *.bmp *.webp",
                ],
                capture_output=True,
                text=True,
                timeout=300,
                check=False,
            )
            if proc.returncode == 0 and proc.stdout.strip():
                return proc.stdout.strip()
        except (OSError, subprocess.SubprocessError):
            pass
    return _pick_image_tk_subprocess(initial_dir)


# Toolbar button layout in the OpenCV window (x1, y1, x2, y2, action)
_TOOLBAR_H = 48
_TOOLBAR_ACTIONS: tuple[tuple[int, int, int, int, str, str], ...] = (
    (16, 8, 140, 40, "import", "Import"),
    (156, 8, 260, 40, "save", "Save"),
    (276, 8, 340, 40, "quit", "Quit"),
)


def _draw_toolbar(canvas: np.ndarray, current_path: str) -> None:
    h, w = canvas.shape[:2]
    bar = canvas[:_TOOLBAR_H, :w]
    bar[:] = (40, 40, 40)
    cv2.line(canvas, (0, _TOOLBAR_H), (w, _TOOLBAR_H), (80, 80, 80), 1)

    for x1, y1, x2, y2, _action, label in _TOOLBAR_ACTIONS:
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (70, 130, 180), -1)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (120, 170, 220), 1)
        scale = 0.5
        tw = int(cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)[0][0])
        tx = x1 + max(4, ((x2 - x1) - tw) // 2)
        ty = y1 + (y1 + y2) // 2 + 5
        cv2.putText(
            canvas,
            label,
            (tx, ty),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    hint = current_path if current_path else "No image loaded - click Import or press o"
    if len(hint) > 90:
        hint = "..." + hint[-87:]
    cv2.putText(
        canvas,
        hint,
        (360, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (210, 210, 210),
        1,
        cv2.LINE_AA,
    )


def _toolbar_action_at(x: int, y: int) -> Optional[str]:
    if y < 0 or y >= _TOOLBAR_H:
        return None
    for x1, y1, x2, y2, action, _label in _TOOLBAR_ACTIONS:
        if x1 <= x <= x2 and y1 <= y <= y2:
            return action
    return None


def _compose_panel(content: np.ndarray, current_path: str) -> np.ndarray:
    toolbar = np.zeros((_TOOLBAR_H, content.shape[1], 3), dtype=np.uint8)
    _draw_toolbar(toolbar, current_path)
    return np.vstack([toolbar, content])


def _placeholder_panel() -> np.ndarray:
    panel = np.zeros((480, 1280, 3), dtype=np.uint8)
    cv2.putText(
        panel,
        "Adjust H/S/V trackbars after loading an image",
        (300, 240),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (180, 180, 180),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        panel,
        "Click [Import] above, or press o / s / q",
        (360, 290),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (140, 140, 140),
        2,
        cv2.LINE_AA,
    )
    return _compose_panel(panel, "")


def run_interactive(
    img_path: Optional[str],
    th: HsvThresholds,
    out_dir: Optional[str],
) -> None:
    initial_img: Optional[np.ndarray] = None
    initial_path = img_path or ""
    initial_stem = "untitled"
    initial_save_dir = out_dir or os.path.join(os.getcwd(), "hsv_demo")

    if img_path:
        initial_img = imread_unicode(img_path)
        if initial_img is None:
            print(f"Cannot read: {img_path}")
            sys.exit(1)
        initial_stem = os.path.splitext(os.path.basename(img_path))[0]
        initial_save_dir = out_dir or os.path.join(
            os.path.dirname(os.path.abspath(img_path)), "hsv_demo"
        )

    win = "HSV Threshold Demo"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 1280, 720)

    state: dict[str, Any] = {
        "th": th,
        "img": initial_img,
        "path": initial_path,
        "stem": initial_stem,
        "save_dir": initial_save_dir,
    }

    def _on_change(_: int) -> None:
        pass

    cv2.createTrackbar("H_min", win, th.h_min, 179, _on_change)
    cv2.createTrackbar("H_max", win, th.h_max, 179, _on_change)
    cv2.createTrackbar("S_min", win, th.s_min, 255, _on_change)
    cv2.createTrackbar("S_max", win, th.s_max, 255, _on_change)
    cv2.createTrackbar("V_min", win, th.v_min, 255, _on_change)
    cv2.createTrackbar("V_max", win, th.v_max, 255, _on_change)
    cv2.createTrackbar("close_k", win, th.close_ksize, 51, _on_change)
    cv2.createTrackbar("open_k", win, th.open_ksize, 31, _on_change)
    cv2.createTrackbar("min_area", win, th.min_area // 100, 500, _on_change)

    def _load_image(path: str) -> bool:
        img = imread_unicode(path)
        if img is None:
            print(f"Failed to read: {path}")
            return False
        state["img"] = img
        state["path"] = path
        state["stem"] = os.path.splitext(os.path.basename(path))[0]
        state["save_dir"] = out_dir or os.path.join(
            os.path.dirname(os.path.abspath(path)), "hsv_demo"
        )
        print(f"Loaded: {path}")
        return True

    def _import_image() -> None:
        initial_dir = os.path.dirname(state["path"]) if state["path"] else os.getcwd()
        path = pick_image_file(initial_dir)
        if path:
            _load_image(path)

    def _save_current() -> None:
        if state["img"] is None or state.get("last_result") is None:
            print("Nothing to save (load an image first)")
            return
        _save_result(state["save_dir"], state["stem"], state["last_result"])
        print(f"Saved -> {state['save_dir']}/{state['stem']}_hsv_*")

    def _on_mouse(event: int, x: int, y: int, _flags: int, _userdata: Any) -> None:
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        action = _toolbar_action_at(x, y)
        if action == "import":
            _import_image()
        elif action == "save":
            _save_current()
        elif action == "quit":
            state["quit"] = True

    cv2.setMouseCallback(win, _on_mouse)

    print("Interactive UI open. Click [Import] in the window toolbar or press o.")
    print("White label hint: lower S_max, raise V_min, H=0..179.")

    state["quit"] = False

    while not state["quit"]:
        t = state["th"]
        t.h_min = cv2.getTrackbarPos("H_min", win)
        t.h_max = cv2.getTrackbarPos("H_max", win)
        t.s_min = cv2.getTrackbarPos("S_min", win)
        t.s_max = cv2.getTrackbarPos("S_max", win)
        t.v_min = cv2.getTrackbarPos("V_min", win)
        t.v_max = cv2.getTrackbarPos("V_max", win)
        t.close_ksize = cv2.getTrackbarPos("close_k", win)
        t.open_ksize = cv2.getTrackbarPos("open_k", win)
        t.min_area = max(100, cv2.getTrackbarPos("min_area", win) * 100)

        t.h_min, t.h_max = min(t.h_min, t.h_max), max(t.h_min, t.h_max)
        t.s_min, t.s_max = min(t.s_min, t.s_max), max(t.s_min, t.s_max)
        t.v_min, t.v_max = min(t.v_min, t.v_max), max(t.v_min, t.v_max)

        if state["img"] is not None:
            result = process_image(state["img"], t)
            state["last_result"] = result
            mask_bgr = cv2.cvtColor(result["mask"], cv2.COLOR_GRAY2BGR)
            content = np.hstack([result["overlay"], mask_bgr])
            panel = _compose_panel(content, state["path"])
            cv2.imshow(win, panel)
        else:
            cv2.imshow(win, _placeholder_panel())

        key = cv2.waitKey(30) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord("s"):
            _save_current()
        if key == ord("o"):
            _import_image()

    cv2.destroyAllWindows()


def run_batch(input_dir: str, out_dir: str, th: HsvThresholds) -> None:
    patterns = ["*.png", "*.jpg", "*.jpeg", "*.bmp"]
    files: list[str] = []
    for pat in patterns:
        files.extend(glob.glob(os.path.join(input_dir, pat)))
    files = sorted(set(files))

    ok = 0
    for fp in files:
        img = imread_unicode(fp)
        if img is None:
            continue
        stem = os.path.splitext(os.path.basename(fp))[0]
        result = process_image(img, th)
        _save_result(out_dir, stem, result)
        ok += 1

    print(f"Batch done: {ok}/{len(files)} -> {out_dir}")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="HSV threshold box demo")
    p.add_argument("-i", "--input", help="Image or directory (optional for interactive UI)")
    p.add_argument("-o", "--output", help="Output directory")
    p.add_argument("--batch", action="store_true", help="Batch mode for directory")
    p.add_argument("--no-gui", action="store_true", help="Export without GUI window")
    p.add_argument("--h-min", type=int, default=0)
    p.add_argument("--h-max", type=int, default=179)
    p.add_argument("--s-min", type=int, default=0)
    p.add_argument("--s-max", type=int, default=30)
    p.add_argument("--v-min", type=int, default=100)
    p.add_argument("--v-max", type=int, default=255)
    p.add_argument("--close-k", type=int, default=15)
    p.add_argument("--open-k", type=int, default=5)
    p.add_argument("--min-area", type=int, default=5000)
    return p


def main() -> None:
    args = _build_parser().parse_args()
    th = HsvThresholds(
        h_min=args.h_min,
        h_max=args.h_max,
        s_min=args.s_min,
        s_max=args.s_max,
        v_min=args.v_min,
        v_max=args.v_max,
        close_ksize=args.close_k,
        open_ksize=args.open_k,
        min_area=args.min_area,
    )

    is_dir = args.input and (args.batch or os.path.isdir(args.input))

    if is_dir:
        if not args.input:
            print("Batch mode requires --input directory")
            sys.exit(1)
        out = args.output or os.path.join(args.input, "hsv_demo")
        run_batch(args.input, out, th)
        return

    if args.no_gui:
        if not args.input:
            print("Export mode requires --input image")
            sys.exit(1)
        img = imread_unicode(args.input)
        if img is None:
            print(f"Cannot read: {args.input}")
            sys.exit(1)
        stem = os.path.splitext(os.path.basename(args.input))[0]
        out = args.output or os.path.join(os.path.dirname(args.input) or ".", "hsv_demo")
        result = process_image(img, th)
        _save_result(out, stem, result)
        print(
            json.dumps(
                {"boxes": result["boxes"], "thresholds": result["thresholds"]},
                ensure_ascii=False,
                indent=2,
            )
        )
        print(f"Saved -> {out}")
        return

    run_interactive(args.input, th, args.output)


if __name__ == "__main__":
    main()
