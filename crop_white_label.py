#!/usr/bin/env python3
"""
自动识别并裁剪图片中亮度最高的白色矩形区域（白色标签）
优化版本：包含二值化处理步骤，针对白色标签优化
"""

import cv2
import numpy as np

def find_white_label_region(image_path: str):
    img = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"无法读取图片: {image_path}")
    
    height, width = img.shape[:2]
    print(f"原始图片尺寸: {width} x {height}")
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # ========== 二值化处理步骤 ==========
    print("执行二值化处理...")
    
    # 使用较高的固定阈值（只保留接近纯白的区域）
    white_threshold = 220
    _, binary = cv2.threshold(gray, white_threshold, 255, cv2.THRESH_BINARY)
    print(f"二值化阈值: {white_threshold}")
    
    # 形态学操作：先开运算去除小噪点，再闭运算填充标签内部空洞
    kernel_small = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    kernel_large = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 30))
    
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_small, iterations=2)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_large, iterations=2)
    
    # ========== 使用二值化结果进行投影分析 ==========
    
    # 垂直投影（每行白色像素数量）
    vertical_proj = np.sum(binary, axis=1)
    
    # 找到白色像素占比超过阈值的行
    row_threshold = width * 0.4
    white_rows = np.where(vertical_proj >= row_threshold)[0]
    
    print(f"找到 {len(white_rows)} 行白色像素占比超过 {row_threshold/width*100:.0f}%")
    
    if len(white_rows) < 10:
        row_threshold = width * 0.25
        white_rows = np.where(vertical_proj >= row_threshold)[0]
        print(f"降低阈值到 {row_threshold/width*100:.0f}% 后找到 {len(white_rows)} 行")
    
    if len(white_rows) < 10:
        print("二值化结果不理想，切换到亮度统计方法")
        return find_by_brightness(img)
    
    # 找到最长的连续白色区域
    clusters = []
    current_cluster = [white_rows[0]]
    
    for i in range(1, len(white_rows)):
        if white_rows[i] - white_rows[i-1] < 5:
            current_cluster.append(white_rows[i])
        else:
            clusters.append(current_cluster)
            current_cluster = [white_rows[i]]
    clusters.append(current_cluster)
    
    clusters.sort(key=len, reverse=True)
    
    if clusters:
        main_cluster = clusters[0]
        y_start = main_cluster[0]
        y_end = main_cluster[-1] + 1
        print(f"最长连续白色区域: {len(main_cluster)} 行")
    else:
        y_start, y_end = height // 4, height * 3 // 4
    
    # 水平投影（在垂直范围内）
    horizontal_proj = np.sum(binary[y_start:y_end, :], axis=0)
    col_threshold = (y_end - y_start) * 0.3
    white_cols = np.where(horizontal_proj >= col_threshold)[0]
    
    if len(white_cols) >= 10:
        x_start = white_cols[0]
        x_end = white_cols[-1] + 1
    else:
        x_start, x_end = 0, width
    
    # 添加边距
    margin = 20
    x_start = max(0, x_start - margin)
    x_end = min(width, x_end + margin)
    y_start = max(0, y_start - margin)
    y_end = min(height, y_end + margin)
    
    # 计算当前尺寸
    current_width = x_end - x_start
    current_height = y_end - y_start
    
    print(f"二值化检测结果 - 宽: {current_width}, 高: {current_height}")
    
    # 如果宽度太小，扩展到几乎全宽，同时调整高度以保持宽高比
    if current_width < width * 0.7:
        original_height = current_height
        
        x_start = 30
        x_end = width - 30
        current_width = x_end - x_start
        
        target_aspect = 2.5
        target_height = int(current_width / target_aspect)
        
        expand_amount = max(0, target_height - original_height)
        expand_top = expand_amount // 2
        expand_bottom = expand_amount - expand_top
        
        new_y_start = max(0, y_start - expand_top)
        new_y_end = min(height, y_end + expand_bottom)
        
        if new_y_start == 0 and y_start > 0:
            extra_expand = y_start
            new_y_end = min(height, new_y_end + extra_expand)
        
        if new_y_end == height and y_end < height:
            extra_expand = height - y_end
            new_y_start = max(0, new_y_start - extra_expand)
        
        y_start, y_end = new_y_start, new_y_end
        current_height = y_end - y_start
    
    # 调整宽高比（放宽到1.5-3.5）
    aspect_ratio = current_width / current_height
    
    print(f"初步检测 - 宽: {current_width}, 高: {current_height}, 宽高比: {aspect_ratio:.2f}")
    
    target_min_aspect = 1.5  # 放宽下限
    target_max_aspect = 3.5  # 放宽上限
    
    if aspect_ratio > target_max_aspect:
        target_height = int(current_width / target_max_aspect) + 10
        expand_amount = max(0, target_height - current_height)
        
        expand_top = expand_amount // 2
        expand_bottom = expand_amount - expand_top
        
        new_y_start = max(0, y_start - expand_top)
        new_y_end = min(height, y_end + expand_bottom)
        
        if new_y_start == 0 and y_start > 0:
            extra_expand = y_start
            new_y_end = min(height, new_y_end + extra_expand)
        
        if new_y_end == height and y_end < height:
            extra_expand = height - y_end
            new_y_start = max(0, new_y_start - extra_expand)
        
        y_start, y_end = new_y_start, new_y_end
        current_height = y_end - y_start
        aspect_ratio = current_width / current_height
        
        print(f"调整后 - 宽: {current_width}, 高: {current_height}, 宽高比: {aspect_ratio:.2f}")
    
    elif aspect_ratio < target_min_aspect:
        target_height = int(current_width / target_min_aspect)
        expand_amount = max(0, target_height - current_height)
        
        y_start = max(0, y_start - expand_amount // 2)
        y_end = min(height, y_end + (expand_amount - expand_amount // 2))
        
        current_height = y_end - y_start
        aspect_ratio = current_width / current_height
        
        print(f"调整后 - 宽: {current_width}, 高: {current_height}, 宽高比: {aspect_ratio:.2f}")
    
    cropped_img = img[y_start:y_end, x_start:x_end]
    
    w = x_end - x_start
    h = y_end - y_start
    final_aspect = w / h
    
    print(f"最终裁剪区域: ({x_start}, {y_start}) - ({x_end}, {y_end})")
    print(f"裁剪后尺寸: {w} x {h}")
    print(f"最终宽高比: {final_aspect:.2f}")
    
    return cropped_img, (x_start, y_start, w, h)

def find_by_brightness(img):
    height, width = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    row_brightness = np.mean(gray, axis=1)
    threshold = np.percentile(row_brightness, 65)
    bright_rows = np.where(row_brightness >= threshold)[0]
    
    if len(bright_rows) < 10:
        threshold = np.percentile(row_brightness, 55)
        bright_rows = np.where(row_brightness >= threshold)[0]
    
    if len(bright_rows) < 10:
        y_start = height // 4
        y_end = height * 3 // 4
    else:
        y_start = max(0, bright_rows.min() - 20)
        y_end = min(height, bright_rows.max() + 20)
    
    x_start, x_end = 30, width - 30
    
    cropped_img = img[y_start:y_end, x_start:x_end]
    w = x_end - x_start
    h = y_end - y_start
    
    print(f"基于亮度统计的裁剪: ({x_start}, {y_start}) - ({x_end}, {y_end})")
    return cropped_img, (x_start, y_start, w, h)

def main():
    input_path = r"d:\软件开发\OCRInkjetCoder\data3\20260429154940.jpg"
    output_path = "output.jpg"
    
    try:
        cropped_img, region = find_white_label_region(input_path)
        
        cv2.imwrite(output_path, cropped_img)
        print(f"裁剪完成！结果已保存到: {output_path}")
        print(f"裁剪区域坐标: x={region[0]}, y={region[1]}, w={region[2]}, h={region[3]}")
        
    except Exception as e:
        print(f"处理失败: {str(e)}")

if __name__ == "__main__":
    main()