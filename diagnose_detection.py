#!/usr/bin/env python3
"""
诊断脚本：分析为什么 paddle_ocr_detect 检测不到文本框
可视化每个处理阶段，帮助找出问题
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from paddleocr import PaddleOCR


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


def _largest_inscribed_rectangle(mask01: np.ndarray) -> tuple[int, int, int, int] | None:
    """Histogram-stack largest rectangle in a binary mask."""
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


def extract_uniform_roi(
    image_bgr: np.ndarray,
    roi_sat_max: int,
    roi_val_min: int,
    roi_min_area_ratio: float,
    roi_shrink_ratio: float,
) -> tuple[np.ndarray, tuple[int, int, int, int] | None]:
    """Extract uniform ROI."""
    h, w = image_bgr.shape[:2]
    mask = build_uniform_mask(image_bgr, roi_sat_max, roi_val_min)
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


def preprocess_roi_for_ocr(roi_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """ROI preprocessing."""
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    bg = cv2.GaussianBlur(gray, (0, 0), sigmaX=21, sigmaY=21)
    bg_norm = cv2.divide(gray, bg, scale=255)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(bg_norm)
    median = cv2.medianBlur(clahe, 3)
    _, otsu = cv2.threshold(median, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if float(np.mean(otsu)) < 127.0:
        otsu = 255 - otsu
    return gray, bg_norm, clahe, median, cv2.cvtColor(otsu, cv2.COLOR_GRAY2BGR)


def main():
    parser = argparse.ArgumentParser(description="诊断文本检测问题")
    parser.add_argument("--input", type=Path, required=True, help="输入图片路径")
    parser.add_argument("--output-dir", type=Path, default=Path("diagnose_outputs"), help="输出目录")
    parser.add_argument("--roi-sat-max", type=int, default=70)
    parser.add_argument("--roi-val-min", type=int, default=95)
    parser.add_argument("--det-limit-side-len", type=int, default=2560)
    parser.add_argument("--det-thresh", type=float, default=0.2)
    parser.add_argument("--det-box-thresh", type=float, default=0.35)
    parser.add_argument("--det-unclip-ratio", type=float, default=2.0)
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    img_path = args.input
    image = cv2.imread(str(img_path))
    if image is None:
        print(f"无法读取图片: {img_path}")
        return

    print(f"图片尺寸: {image.shape[1]}x{image.shape[0]}")
    cv2.imwrite(str(output_dir / "01_original.png"), image)

    # Step 1: 检查 ROI 提取
    print("\n=== 步骤 1: ROI 提取 ===")
    mask, roi_box = extract_uniform_roi(
        image,
        args.roi_sat_max,
        args.roi_val_min,
        0.12,
        0.04,
    )

    cv2.imwrite(str(output_dir / "02_mask.png"), mask)
    print(f"掩码像素总数: {np.sum(mask > 0)}")

    if roi_box is None:
        print("❌ 没有检测到 ROI！")
        print("尝试放宽参数: --roi-sat-max 90 --roi-val-min 80")
    else:
        x0, y0, rw, rh = roi_box
        print(f"✅ 检测到 ROI: ({x0}, {y0}, {rw}, {rh})")

        roi = image[y0:y0+rh, x0:x0+rw]
        cv2.imwrite(str(output_dir / "03_roi.png"), roi)

        # Step 2: 检查预处理
        print("\n=== 步骤 2: ROI 预处理 ===")
        gray, bg_norm, clahe, median, roi_ocr = preprocess_roi_for_ocr(roi)

        cv2.imwrite(str(output_dir / "04_gray.png"), gray)
        cv2.imwrite(str(output_dir / "05_bg_norm.png"), bg_norm)
        cv2.imwrite(str(output_dir / "06_clahe.png"), clahe)
        cv2.imwrite(str(output_dir / "07_median.png"), median)
        cv2.imwrite(str(output_dir / "08_otsu_bin.png"), roi_ocr)

        print(f"Otsu 二值图均值: {np.mean(cv2.cvtColor(roi_ocr, cv2.COLOR_BGR2GRAY)):.2f}")

        # Step 3: 尝试在不同阶段进行检测
        print("\n=== 步骤 3: 文本检测测试 ===")

        print("初始化 PaddleOCR...")
        ocr_engine = PaddleOCR(
            use_angle_cls=True,
            lang="ch",
            text_det_limit_side_len=args.det_limit_side_len,
            text_det_limit_type="max",
            text_det_thresh=args.det_thresh,
            text_det_box_thresh=args.det_box_thresh,
            text_det_unclip_ratio=args.det_unclip_ratio,
            show_log=True
        )

        test_images = [
            ("original_roi", roi),
            ("bg_norm", cv2.cvtColor(bg_norm, cv2.COLOR_GRAY2BGR)),
            ("clahe", cv2.cvtColor(clahe, cv2.COLOR_GRAY2BGR)),
            ("median", cv2.cvtColor(median, cv2.COLOR_GRAY2BGR)),
            ("otsu_bin", roi_ocr),
        ]

        for name, img in test_images:
            print(f"\n测试 {name}...")
            try:
                result = ocr_engine.ocr(img, cls=True)
                boxes = []
                if result and result[0]:
                    for line in result[0]:
                        if line:
                            boxes.append(line)

                print(f"检测到 {len(boxes)} 个文本框")

                vis_img = img.copy()
                for i, box in enumerate(boxes):
                    poly = np.array(box[0], dtype=np.int32)
                    cv2.polylines(vis_img, [poly.reshape(-1, 1, 2)], True, (0, 255, 0), 2)
                    text = box[1][0] if len(box) > 1 else ""
                    conf = box[1][1] if len(box) > 1 else 0.0
                    cv2.putText(vis_img, f"{i}: {conf:.2f}", (poly[0][0], poly[0][1] - 5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                cv2.imwrite(str(output_dir / f"09_detect_{name}.png"), vis_img)

                if boxes:
                    print(f"  前3个结果:")
                    for i, box in enumerate(boxes[:3]):
                        print(f"    {i}: 文本='{box[1][0]}', 置信度={box[1][1]:.3f}")
            except Exception as e:
                print(f"检测错误: {e}")

    # 尝试降低检测阈值直接检测全图
    print("\n=== 步骤 4: 全图直接检测（降低阈值）===")
    ocr_low_thresh = PaddleOCR(
        use_angle_cls=True,
        lang="ch",
        text_det_limit_side_len=args.det_limit_side_len,
        text_det_thresh=0.1,
        text_det_box_thresh=0.1,
        text_det_unclip_ratio=args.det_unclip_ratio,
        show_log=False
    )

    try:
        result = ocr_low_thresh.ocr(image, cls=True)
        boxes = []
        if result and result[0]:
            for line in result[0]:
                if line:
                    boxes.append(line)

        print(f"全图检测到 {len(boxes)} 个文本框")

        vis_full = image.copy()
        for i, box in enumerate(boxes):
            poly = np.array(box[0], dtype=np.int32)
            cv2.polylines(vis_full, [poly.reshape(-1, 1, 2)], True, (0, 0, 255), 2)

        cv2.imwrite(str(output_dir / "10_full_image_low_thresh.png"), vis_full)
    except Exception as e:
        print(f"全图检测错误: {e}")

    print(f"\n诊断完成！结果保存在: {output_dir}")


if __name__ == "__main__":
    main()
