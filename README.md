# OCR Inkjet: 纯色区域提取 + OCR

这个项目会做两件事：

1. 从 `data` 图片中提取一块“大面积纯色区域”（如白色包装袋区域）；
2. 在该区域内做文字分割，再做 OCR 识别。

## 1) 安装

推荐使用 conda 环境 `OCRInkjetCoder`：

```bash
conda create -n OCRInkjetCoder python=3.10 -y
conda activate OCRInkjetCoder
pip install -r requirements.txt
```

本地 OCR 采用 `PaddleOCR`（支持旋转文本方向分类）。
默认语言参数是 `--ocr-lang ch`（中文）。
如果只识别英文可改为 `--ocr-lang en`。

## 2) 运行

```bash
conda run -n OCRInkjetCoder python ocr_segment.py --input-dir data --output-dir outputs
```

常用参数：

- `--roi-sat-max`：纯色区域最大饱和度（默认 `70`）
- `--roi-val-min`：纯色区域最小亮度（默认 `95`）
- `--roi-min-area-ratio`：纯色区域最小面积占比（默认 `0.12`）
- `--roi-shrink-ratio`：ROI 向内收窄比例（默认 `0.02`）
- `--ocr-lang`：PaddleOCR 语言（默认 `ch`）
- `--use-angle-cls`：开启旋转方向分类（默认开启）
- `--ocr-min-conf`：OCR 最低保留置信度（默认 `0.55`）
- `--ocr-upscale`：OCR 增强分支放大倍率（默认 `1.6`）
- `--ocr-mode`：`fast`（默认）或 `high_accuracy`（多尺度+旋转集成，更慢更准）
- `--hi-acc-scales`：仅 `high_accuracy` 有效。逗号分隔的放大倍率，默认 **`1.0`**（只用原 ROI，不再做 CLAHE 放大）。需要多尺度投票时可写 `1.0,1.4,1.7,1.9`
- `--ocr-model-size`：`auto`（默认）、`mobile`、`server`；`auto` 在 `fast` 下为 mobile，在 `high_accuracy` 下为 server
- `--ocr-verbose` / `--no-ocr-verbose`：是否打印每次 OCR 尝试的日志（默认真）
- `--input-upright`：原图已摆正方向时加上；`high_accuracy` 下不再做 0/90/180/270 的整图旋转尝试，只保留多尺度（4 次推理，更快）。Paddle 文字行方向分类仍默认开启。

> **CPU 上** `high_accuracy` + `server` 每张大 ROI 的**单次** `det+rec` 可能就要几分钟；日志若停在 `trial 01/04 running det+rec...` 是正在推理，并非死机。可改用 `--ocr-model-size mobile` 或 `fast` 模式加快。

如果没找到纯色区域，可尝试放宽参数：

```bash
python ocr_segment.py --input-dir data --output-dir outputs --roi-sat-max 90 --roi-val-min 80
```

## 3) 输出结果

运行后会得到：

- `outputs/masks/*_mask.png`：前景分割掩码
- `outputs/regions/*_uniform_mask.png`：纯色区域检测掩码
- `outputs/regions/*_uniform_region.png`：截取出的纯色区域 ROI
- `outputs/ocr_results.json`：每张图一条汇总文本结果
- `outputs/ocr_results.csv`：便于表格查看的 OCR 结果（含结构化字段与 `elapsed_seconds`）
- `outputs/ocr_results.xlsx`：与 CSV 内容相同的 Excel 表（需已安装 `openpyxl`）

## 4) 处理逻辑说明

- 先在 HSV 空间中找“低饱和度 + 高亮度”的大面积区域；
- 在纯色掩码中寻找“最大内接纯色矩形”作为 ROI，避免把杂色边缘裁进来；
- 对 `*_uniform_region.png` 整图执行 PaddleOCR 检测+识别（支持旋转文字）；
- 按阅读顺序拼接文本并输出为每图一条结果；
- 通过规则提取 `生产日期` / `常温储存保质期至` / `冷冻储存保质期至`，输出标准化文本。
