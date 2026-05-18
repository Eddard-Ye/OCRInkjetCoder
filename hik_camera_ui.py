# -*- coding: utf-8 -*-
import sys
import os

_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import json
import random
import re
import time
import ctypes
import queue
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
from ctypes import *
import threading
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext

import numpy as np
import cv2

from paddle_full_image_detect import (
    build_ocr_engine,
    coerce_box_score,
    coerce_box_text,
    draw_detections,
    predict_boxes,
    prepare_bgr_for_predict,
    resolve_paddle_device,
)
from relay_controller import RelayController
from production_phrase_strategy import (
    ColonCjkPhraseMatchStrategy,
    StrictExclusiveSubstringStrategy,
)

HIK_SDK_PATH = r"D:\海康威视\MVS\Development\Samples\Python"
SDK_PATH = os.path.join(HIK_SDK_PATH, "MvImport")
sys.path.append(SDK_PATH)

from MvCameraControl_class import *
from CameraParams_header import *

# GenICam keys we snapshot / edit (unsupported keys are skipped at read time).
_CAMERA_FEATURE_SCHEMA: List[Tuple[str, str]] = [
    ("Width", "int"),
    ("Height", "int"),
    ("OffsetX", "int"),
    ("OffsetY", "int"),
    ("PixelFormat", "enum"),
    ("AcquisitionMode", "enum"),
    ("ExposureAuto", "enum"),
    ("ExposureTime", "float"),
    ("GainAuto", "enum"),
    ("Gain", "float"),
    ("AcquisitionFrameRateEnable", "bool"),
    ("AcquisitionFrameRate", "float"),
    ("TriggerMode", "enum"),
    ("TriggerSource", "enum"),
    ("TriggerActivation", "enum"),
    ("BalanceWhiteAuto", "enum"),
    ("DeviceUserID", "string"),
]

# GenICam set order: size/offset after pixel format; manual exposure/gain after auto switches.
_CAMERA_PARAM_APPLY_ORDER: tuple[str, ...] = (
    "AcquisitionMode",
    "PixelFormat",
    "OffsetX",
    "OffsetY",
    "Width",
    "Height",
    "ExposureAuto",
    "GainAuto",
    "BalanceWhiteAuto",
    "ExposureTime",
    "Gain",
    "AcquisitionFrameRateEnable",
    "AcquisitionFrameRate",
    "TriggerMode",
    "TriggerSource",
    "TriggerActivation",
    "DeviceUserID",
)


def _camera_param_apply_sort_key(key: str) -> tuple[int, str]:
    try:
        return (_CAMERA_PARAM_APPLY_ORDER.index(key), key)
    except ValueError:
        return (999, key)


# Hik cameras often implement these as IInteger / int64 (us); GetFloatValue may lie (e.g. 1.0).
_FLAKY_FLOAT_GENICAM_KEYS = frozenset({"ExposureTime", "Gain"})


# OCR 产线三语校验：下拉框取值（与 ``production_phrase_strategy`` 类名一致便于对照代码）
_PRODUCTION_STRATEGY_STRICT = "StrictExclusiveSubstringStrategy"
_PRODUCTION_STRATEGY_COLON_CJK = "ColonCjkPhraseMatchStrategy"

# 主窗口启动时是否最大化（Windows 下 ``state("zoomed")``；不支持则保持 geometry）。
_START_WINDOW_MAXIMIZED = True


def _decode_bayer8_to_bgr(
    img_bayer: np.ndarray, pixel_type: int
) -> np.ndarray | None:
    """
    Map Hik GVSP Bayer8 pixel type to OpenCV demosaic (standard + HB_ 压缩格式)。
    HB_* 与海康枚举一一对应，与标准 Bayer 共用同一马赛克相位时使用相同 cv2 常数。
    """
    pt = int(pixel_type)
    m = {
        int(PixelType_Gvsp_BayerRG8): cv2.COLOR_BAYER_RG2BGR,
        int(PixelType_Gvsp_BayerGR8): cv2.COLOR_BAYER_GR2BGR,
        int(PixelType_Gvsp_BayerGB8): cv2.COLOR_BAYER_GB2BGR,
        int(PixelType_Gvsp_BayerBG8): cv2.COLOR_BAYER_BG2BGR,
        int(PixelType_Gvsp_HB_BayerRG8): cv2.COLOR_BAYER_RG2BGR,
        int(PixelType_Gvsp_HB_BayerGR8): cv2.COLOR_BAYER_GR2BGR,
        int(PixelType_Gvsp_HB_BayerGB8): cv2.COLOR_BAYER_GB2BGR,
        int(PixelType_Gvsp_HB_BayerBG8): cv2.COLOR_BAYER_BG2BGR,
    }
    cv_code = m.get(pt)
    if cv_code is None:
        return None
    return cv2.cvtColor(img_bayer, cv_code)


def _normalize_genicam_enum_string_for_set(key: str, raw: str) -> str:
    """
    GenICam ``SetEnumValueByString`` 通常要求与相机 XML 中 **Symbolic** 完全一致。

    MVS 界面常显示为 ``Bayer RG 8``（带空格）；XML 里多为 ``BayerRG8``。
    用户若粘贴 ``bayerrg8`` 等大小写不一致写法，此处规范为 ``BayerXX8`` 形式。
    """
    s = "".join(str(raw).strip().split())
    if not s or key != "PixelFormat":
        return s
    m = re.match(r"(?i)^bayer(rg|gr|gb|bg)(\d+)$", s)
    if m:
        return f"Bayer{m.group(1).upper()}{m.group(2)}"
    return s


def _format_genicam_float_for_display(fv: float) -> str:
    """
    Turn a GenICam float into a stable decimal string for the UI.

    Never do ``str(...).rstrip("0")`` on values without a decimal point: e.g. ``"80000"`` → ``"8"``.
    """
    if fv != fv:  # NaN
        return str(fv)
    if fv.is_integer():
        return str(int(fv))
    s = f"{fv:.10f}".rstrip("0").rstrip(".")
    return s if s else "0"


class HikCameraApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("海康工业相机 - 实时监控系统")
        self.root.geometry("1100x800")
        self.root.configure(bg="#2b2b2b")
        if _START_WINDOW_MAXIMIZED:
            try:
                self.root.state("zoomed")
            except tk.TclError:
                pass

        self.cam = MvCamera()
        self.device_list = MV_CC_DEVICE_INFO_LIST()
        self.b_open_device = False
        self.b_start_grabbing = False
        self.b_exit = False
        self.current_frame = None
        self.frame_lock = threading.Lock()
        self.grab_thread: Optional[threading.Thread] = None
        self.use_hw_trigger = False
        self._mode_switch_lock = threading.Lock()
        self._video_preview_photo: Optional[ImageTk.PhotoImage] = None
        self._preview_queue: queue.Queue = queue.Queue(maxsize=2)
        self._stats_lock = threading.Lock()
        self._manual_capture_ocr_count = 0
        self._hw_trigger_capture_count = 0
        self._ocr_total_count = 0
        self._ocr_ok_count = 0
        self._ocr_ng_count = 0
        self._stats_json_path = os.path.join(_PROJECT_ROOT, "hik_camera_ui_stats.json")

        self.photo_save_dir = os.path.join(os.path.expanduser("~"), "HikCameraPhotos")
        self.ocr_output_dir = os.path.join(os.path.expanduser("~"), "HikCameraOCR")
        if not os.path.exists(self.photo_save_dir):
            os.makedirs(self.photo_save_dir)
        if not os.path.exists(self.ocr_output_dir):
            os.makedirs(self.ocr_output_dir)

        self.current_ocr_result = None
        self.ocr_result_image = None
        self.ocr_engine = None
        self._ocr_init_error: Optional[str] = None

        relay_exe = os.path.join(
            _PROJECT_ROOT,
            "4roadcontrol",
            "TestApp",
            "CommandApp_USBRelay.exe",
        )
        self.relay_controller = RelayController(
            exe_path=os.path.normpath(relay_exe),
            serial_number="HW341",
        )

        # 三语 OCR 校验策略（默认严格子串；与 ColonCjk 并存，下拉切换）
        self._strict_strategy_cfg: Dict[str, Any] = {"strict_year_suffix": False}
        self._colon_cjk_strategy_cfg: Dict[str, Any] = {
            "exclude_lines_without_year": False,
            "max_cjk_length_diff": 2,
            "min_cjk_lcs_matches": 3,
        }
        self._strategy_strict = StrictExclusiveSubstringStrategy(
            strict_year_suffix=self._strict_strategy_cfg["strict_year_suffix"],
        )
        self._strategy_colon_cjk = ColonCjkPhraseMatchStrategy(
            exclude_lines_without_year=self._colon_cjk_strategy_cfg[
                "exclude_lines_without_year"
            ],
            max_cjk_length_diff=self._colon_cjk_strategy_cfg["max_cjk_length_diff"],
            min_cjk_lcs_matches=self._colon_cjk_strategy_cfg["min_cjk_lcs_matches"],
        )
        self._production_strategy_var = tk.StringVar(
            master=self.root,
            value=_PRODUCTION_STRATEGY_STRICT,
        )

        self._camera_serial_for_config = ""
        self._camera_config_dir = os.path.join(_PROJECT_ROOT, "camera_configs")

        self.setup_ui()
        self.protocol()
        self._load_stats_from_disk()
        self._init_ocr_engine()

    def setup_ui(self):
        title_label = tk.Label(
            self.root,
            text="海康工业相机 + OCR识别系统",
            font=("微软雅黑", 20, "bold"),
            bg="#2b2b2b",
            fg="#00ff00"
        )
        title_label.pack(pady=10)

        control_frame = tk.Frame(self.root, bg="#2b2b2b")
        control_frame.pack(pady=5)

        btn_style = {
            "font": ("微软雅黑", 10),
            "width": 12,
            "height": 2,
            "relief": "raised",
            "bd": 2
        }

        self.btn_connect = tk.Button(
            control_frame,
            text="连接相机",
            command=self.connect_camera,
            bg="#4CAF50",
            fg="white",
            **btn_style
        )
        self.btn_connect.grid(row=0, column=0, padx=5, pady=5)

        self.btn_disconnect = tk.Button(
            control_frame,
            text="断开连接",
            command=self.disconnect_camera,
            state="disabled",
            bg="#f44336",
            fg="white",
            **btn_style
        )
        self.btn_disconnect.grid(row=0, column=1, padx=5, pady=5)

        self.btn_capture = tk.Button(
            control_frame,
            text="📷 拍照+OCR",
            command=self.capture_and_ocr,
            state="disabled",
            bg="#9C27B0",
            fg="white",
            **btn_style
        )
        self.btn_capture.grid(row=0, column=2, padx=5, pady=5)

        self.btn_open_folder = tk.Button(
            control_frame,
            text="📁 照片文件夹",
            command=self.open_photo_folder,
            bg="#FF9800",
            fg="white",
            **btn_style
        )
        self.btn_open_folder.grid(row=0, column=3, padx=5, pady=5)

        self.btn_config_camera = tk.Button(
            control_frame,
            text="配置相机",
            command=self.open_camera_config_dialog,
            state="disabled",
            bg="#009688",
            fg="white",
            **btn_style
        )
        self.btn_config_camera.grid(row=0, column=4, padx=5, pady=5)

        self.btn_toggle_trigger = tk.Button(
            control_frame,
            text="切换到硬触发(Line0)",
            command=self.toggle_hardware_trigger_mode,
            state="disabled",
            bg="#607D8B",
            fg="white",
            font=("微软雅黑", 10),
            width=28,
            height=1,
            relief="raised",
            bd=2,
        )
        self.btn_toggle_trigger.grid(row=1, column=0, columnspan=5, padx=5, pady=(0, 5))

        strat_frame = tk.Frame(control_frame.master, bg="#2b2b2b")
        strat_frame.pack(pady=(0, 4), fill="x", padx=10)

        tk.Label(
            strat_frame,
            text="产线/保质期校验策略:",
            font=("微软雅黑", 9),
            bg="#2b2b2b",
            fg="#cccccc",
        ).pack(side=tk.LEFT, padx=(0, 6))

        self._combo_production_strategy = ttk.Combobox(
            strat_frame,
            textvariable=self._production_strategy_var,
            state="readonly",
            width=36,
            values=(_PRODUCTION_STRATEGY_STRICT, _PRODUCTION_STRATEGY_COLON_CJK),
        )
        self._combo_production_strategy.pack(side=tk.LEFT, padx=4)

        btn_cfg_strict = tk.Button(
            strat_frame,
            text="编辑 Strict 配置",
            command=self._open_strict_strategy_config_dialog,
            font=("微软雅黑", 9),
            bg="#37474f",
            fg="#ffffff",
            padx=8,
            pady=2,
        )
        btn_cfg_strict.pack(side=tk.LEFT, padx=(12, 4))

        btn_cfg_colon = tk.Button(
            strat_frame,
            text="编辑 ColonCjk 配置",
            command=self._open_colon_cjk_strategy_config_dialog,
            font=("微软雅黑", 9),
            bg="#37474f",
            fg="#ffffff",
            padx=8,
            pady=2,
        )
        btn_cfg_colon.pack(side=tk.LEFT, padx=4)

        self.status_label = tk.Label(
            self.root,
            text="状态: 未连接",
            font=("微软雅黑", 10),
            bg="#2b2b2b",
            fg="#ffff00"
        )
        self.status_label.pack(pady=3)

        main_frame = tk.Frame(self.root, bg="#2b2b2b")
        main_frame.pack(fill="both", expand=True, padx=10, pady=5)

        display_frame = tk.LabelFrame(
            main_frame,
            text="相机预览",
            font=("微软雅黑", 11),
            bg="#1a1a1a",
            fg="#00ff00",
            bd=2
        )
        display_frame.pack(side="left", fill="both", expand=True)

        self.video_label = tk.Label(
            display_frame,
            text="相机预览区域",
            bg="#1a1a1a",
            fg="#666666",
            font=("微软雅黑", 14)
        )
        self.video_label.pack(fill="both", expand=True, padx=5, pady=5)

        result_frame = tk.LabelFrame(
            main_frame,
            text="OCR识别结果",
            font=("微软雅黑", 11),
            bg="#1a1a1a",
            fg="#00ff00",
            bd=2
        )
        result_frame.pack(side="right", fill="both", expand=False, padx=(10, 0))

        self.result_text = scrolledtext.ScrolledText(
            result_frame,
            width=40,
            height=15,
            font=("微软雅黑", 9),
            bg="#1a1a1a",
            fg="#00ff00",
            insertbackground="white",
            relief="sunken",
            bd=1
        )
        self.result_text.pack(fill="both", expand=True, padx=5, pady=5)

        ocr_result_frame = tk.LabelFrame(
            result_frame,
            text="识别结果图像",
            font=("微软雅黑", 10),
            bg="#1a1a1a",
            fg="#00ff00",
            bd=1
        )
        ocr_result_frame.pack(fill="both", expand=True, padx=5, pady=5)

        self.ocr_result_label = tk.Label(
            ocr_result_frame,
            text="OCR结果预览",
            bg="#1a1a1a",
            fg="#666666",
            font=("微软雅黑", 10)
        )
        self.ocr_result_label.pack(fill="both", expand=True, padx=5, pady=5)

        info_frame = tk.Frame(self.root, bg="#2b2b2b")
        info_frame.pack(pady=3, fill="x", padx=20)

        self.camera_info_label = tk.Label(
            info_frame,
            text="相机信息: 未连接",
            font=("微软雅黑", 9),
            bg="#2b2b2b",
            fg="#aaaaaa",
            anchor="w"
        )
        self.camera_info_label.pack(side="left")

        stats_wrap = tk.Frame(info_frame, bg="#2b2b2b")
        stats_wrap.pack(side="right")

        self.btn_stats_reset = tk.Button(
            stats_wrap,
            text="统计清零",
            command=self._clear_all_stats,
            font=("微软雅黑", 9),
            bg="#444444",
            fg="#ffffff",
            padx=8,
            pady=2,
        )
        self.btn_stats_reset.pack(side="right", padx=(8, 0))

        self.photo_count_label = tk.Label(
            stats_wrap,
            text="拍照+OCR: 0 次  |  硬触发: 0 次\nOCR总计: 0  |  OK: 0  |  NG: 0  |  OK率: 0%  |  NG率: 0%",
            font=("微软雅黑", 9),
            bg="#2b2b2b",
            fg="#aaaaaa",
            justify=tk.RIGHT,
            anchor="e",
        )
        self.photo_count_label.pack(side="right")

    def protocol(self):
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def update_status(self, text, color="#ffff00"):
        self.status_label.config(text=f"状态: {text}", fg=color)
        self.root.update_idletasks()

    def update_camera_info(self, info):
        self.camera_info_label.config(text=f"相机信息: {info}")

    def update_capture_stats_display(self) -> None:
        """Refresh bottom-right counters (backed by ``hik_camera_ui_stats.json``)."""
        with self._stats_lock:
            m = self._manual_capture_ocr_count
            h = self._hw_trigger_capture_count
            t = self._ocr_total_count
            ok = self._ocr_ok_count
            ng = self._ocr_ng_count
        ok_rate = (100.0 * ok / t) if t > 0 else 0.0
        ng_rate = (100.0 * ng / t) if t > 0 else 0.0
        self.photo_count_label.config(
            text=(
                f"拍照+OCR: {m} 次  |  硬触发: {h} 次\n"
                f"OCR总计: {t}  |  OK: {ok}  |  NG: {ng}  |  "
                f"OK率: {ok_rate:.1f}%  |  NG率: {ng_rate:.1f}%"
            )
        )

    def _load_stats_from_disk(self) -> None:
        path = self._stats_json_path
        if not os.path.isfile(path):
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                d = json.load(f)
            with self._stats_lock:
                self._manual_capture_ocr_count = max(
                    0, int(d.get("manual_capture_ocr_count", 0))
                )
                self._hw_trigger_capture_count = max(
                    0, int(d.get("hw_trigger_capture_count", 0))
                )
                self._ocr_total_count = max(0, int(d.get("ocr_total_count", 0)))
                self._ocr_ok_count = max(0, int(d.get("ocr_ok_count", 0)))
                self._ocr_ng_count = max(0, int(d.get("ocr_ng_count", 0)))
        except Exception:
            pass

    def _persist_stats_to_disk(self) -> None:
        with self._stats_lock:
            payload = {
                "manual_capture_ocr_count": self._manual_capture_ocr_count,
                "hw_trigger_capture_count": self._hw_trigger_capture_count,
                "ocr_total_count": self._ocr_total_count,
                "ocr_ok_count": self._ocr_ok_count,
                "ocr_ng_count": self._ocr_ng_count,
                "saved_at": datetime.now(timezone.utc).isoformat(),
            }
        try:
            with open(self._stats_json_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def _clear_all_stats(self) -> None:
        if not messagebox.askyesno("统计清零", "确定清零所有统计计数吗？"):
            return
        with self._stats_lock:
            self._manual_capture_ocr_count = 0
            self._hw_trigger_capture_count = 0
            self._ocr_total_count = 0
            self._ocr_ok_count = 0
            self._ocr_ng_count = 0
        self._persist_stats_to_disk()
        self.update_capture_stats_display()

    def display_ocr_result(self, ocr_data):
        self.result_text.delete(1.0, tk.END)

        if not ocr_data or "boxes" not in ocr_data:
            self.result_text.insert(tk.END, "未识别到文字\n")
            return

        boxes = ocr_data.get("boxes", [])
        file_name = ocr_data.get("file", "unknown")
        mode = ocr_data.get("mode", "unknown")
        infer_time = ocr_data.get("infer_seconds", 0)

        self.result_text.insert(tk.END, f"文件: {file_name}\n")
        self.result_text.insert(tk.END, f"模式: {mode}\n")
        self.result_text.insert(tk.END, f"识别数量: {len(boxes)} 个文本框\n")
        self.result_text.insert(tk.END, f"推理时间: {infer_time:.3f}s\n")
        self.result_text.insert(tk.END, "-" * 30 + "\n\n")

        long_texts = []
        for i, box in enumerate(boxes, 1):
            text = coerce_box_text(box.get("text"))
            score = coerce_box_score(box.get("score"))
            bbox_raw = box.get("bbox_xyxy", [])
            bb = np.asarray(bbox_raw, dtype=np.float64).ravel()
            if bb.size < 4:
                bb = np.zeros(4, dtype=np.float64)

            if len(text) >= 2:
                long_texts.append(text)
                self.result_text.insert(tk.END, f"[{i}] {text} ", "green")
                self.result_text.insert(tk.END, f"(置信度: {score:.2f})\n", "yellow")
                self.result_text.insert(tk.END, f"    位置: [{bb[0]:.0f}, {bb[1]:.0f}, "
                                               f"{bb[2]:.0f}, {bb[3]:.0f}]\n", "gray")

        self.result_text.tag_config("green", foreground="#00ff00")
        self.result_text.tag_config("yellow", foreground="#ffff00")
        self.result_text.tag_config("gray", foreground="#888888")

        self.result_text.insert(tk.END, "\n" + "=" * 30 + "\n")
        self.result_text.insert(tk.END, f"长度>=2的文本 ({len(long_texts)}个):\n", "bold")
        self.result_text.tag_config("bold", foreground="#ffffff", font=("微软雅黑", 10, "bold"))

        for i, text in enumerate(long_texts, 1):
            self.result_text.insert(tk.END, f"  {i}. {text}\n", "cyan")
        self.result_text.tag_config("cyan", foreground="#00ffff")

        return long_texts

    def display_ocr_image(self, image_path):
        if not os.path.exists(image_path):
            return

        img = cv2.imread(image_path)
        if img is None:
            return

        self.display_ocr_image_from_bgr(img)

    def display_ocr_image_from_bgr(self, img_bgr: np.ndarray) -> None:
        """Show a BGR ndarray in the OCR preview panel (resized for layout)."""
        if img_bgr is None or img_bgr.size == 0:
            return
        if img_bgr.ndim == 2:
            img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2BGR)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)

        display_width = 350
        display_height = int(display_width * img_pil.height / img_pil.width)
        img_pil = img_pil.resize((display_width, display_height), Image.Resampling.LANCZOS)

        img_tk = ImageTk.PhotoImage(img_pil)

        self.ocr_result_label.config(image=img_tk, text="")
        self.ocr_result_label.image = img_tk

    def _init_ocr_engine(self) -> None:
        """Load PaddleOCR once at startup; default request GPU (same det defaults as CLI)."""
        self.ocr_engine = None
        self._ocr_init_error = None
        try:
            self.update_status("正在加载 OCR 模型（首次可能较慢）...", "#ffff00")
            self.root.update_idletasks()
            device_kw, dev_info = resolve_paddle_device(use_cuda=True, gpu_id=0)
            self.ocr_engine = build_ocr_engine(
                ocr_lang="ch",
                use_angle_cls=True,
                ocr_model_size="mobile",
                det_limit_side_len=2560,
                det_thresh=0.10,
                det_box_thresh=0.35,
                det_unclip_ratio=2.0,
                device=device_kw,
            )
            if dev_info.get("fallback_reason"):
                self.update_status(
                    "OCR 模型已就绪（请求 GPU，已回退 CPU）", "#ffff00"
                )
            else:
                self.update_status("OCR 模型已就绪（GPU）", "#00ff00")
        except Exception as e:
            self._ocr_init_error = str(e)
            self.ocr_engine = None
            self.update_status(f"OCR 初始化失败: {e}", "#ff0000")

    @staticmethod
    def _ensure_bgr_u8(frame: np.ndarray | None) -> np.ndarray | None:
        if frame is None or frame.size == 0:
            return None
        if frame.ndim == 2:
            return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        if frame.ndim == 3 and frame.shape[2] == 3:
            return frame
        if frame.ndim == 3 and frame.shape[2] == 4:
            return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        return None

    def callRelayAction(self) -> None:
        """Hook: invoked when required label phrases are not all found in ``boxes``."""
        try:
            self.relay_controller.turn_on(1)
            time.sleep(0.5)
            self.relay_controller.turn_off(1)
        except Exception as e:
            messagebox.showwarning("继电器", f"继电器动作失败: {e}")

    def _active_production_strategy(self):
        """当前下拉选中的三语校验策略实例。"""
        name = self._production_strategy_var.get()
        if name == _PRODUCTION_STRATEGY_COLON_CJK:
            strat = self._strategy_colon_cjk
            cfg = dict(self._colon_cjk_strategy_cfg)
        else:
            strat = self._strategy_strict
            cfg = dict(self._strict_strategy_cfg)
        print(
            "[HikCameraApp] active_production_strategy "
            f"combo={name!r} class={type(strat).__name__} config={cfg!r}",
            flush=True,
        )
        return strat

    def _open_strict_strategy_config_dialog(self) -> None:
        top = tk.Toplevel(self.root)
        top.title("StrictExclusiveSubstringStrategy")
        top.configure(bg="#2b2b2b")
        top.transient(self.root)
        top.grab_set()

        var = tk.BooleanVar(
            value=bool(self._strict_strategy_cfg.get("strict_year_suffix", False))
        )
        tk.Label(
            top,
            text="严格子串（三行互斥）参数",
            font=("微软雅黑", 11, "bold"),
            bg="#2b2b2b",
            fg="#00ff00",
        ).pack(anchor="w", padx=12, pady=(10, 4))

        tk.Checkbutton(
            top,
            text="strict_year_suffix：排除整行不含今年数字(如2026)子串的行",
            variable=var,
            font=("微软雅黑", 9),
            bg="#2b2b2b",
            fg="#eeeeee",
            selectcolor="#444444",
            activebackground="#2b2b2b",
            activeforeground="#ffffff",
        ).pack(anchor="w", padx=12, pady=6)

        btn_row = tk.Frame(top, bg="#2b2b2b")
        btn_row.pack(pady=14)

        def on_ok() -> None:
            self._strict_strategy_cfg["strict_year_suffix"] = bool(var.get())
            self._strategy_strict = StrictExclusiveSubstringStrategy(
                strict_year_suffix=self._strict_strategy_cfg["strict_year_suffix"],
            )
            top.destroy()

        tk.Button(
            btn_row,
            text="确定",
            command=on_ok,
            font=("微软雅黑", 9),
            bg="#4CAF50",
            fg="white",
            padx=14,
            pady=4,
        ).pack(side=tk.LEFT, padx=6)
        tk.Button(
            btn_row,
            text="取消",
            command=top.destroy,
            font=("微软雅黑", 9),
            bg="#666666",
            fg="white",
            padx=14,
            pady=4,
        ).pack(side=tk.LEFT, padx=6)

    def _open_colon_cjk_strategy_config_dialog(self) -> None:
        top = tk.Toplevel(self.root)
        top.title("ColonCjkPhraseMatchStrategy")
        top.configure(bg="#2b2b2b")
        top.transient(self.root)
        top.grab_set()

        cfg = self._colon_cjk_strategy_cfg
        var_year = tk.BooleanVar(value=bool(cfg.get("exclude_lines_without_year", False)))
        var_x = tk.StringVar(value=str(int(cfg.get("max_cjk_length_diff", 2))))
        var_y = tk.StringVar(value=str(int(cfg.get("min_cjk_lcs_matches", 3))))

        tk.Label(
            top,
            text="冒号前 CJK + 长度/LCS 阈值策略",
            font=("微软雅黑", 11, "bold"),
            bg="#2b2b2b",
            fg="#00ff00",
        ).pack(anchor="w", padx=12, pady=(10, 4))

        tk.Checkbutton(
            top,
            text="exclude_lines_without_year：同 Strict（行内须含今年数字子串）",
            variable=var_year,
            font=("微软雅黑", 9),
            bg="#2b2b2b",
            fg="#eeeeee",
            selectcolor="#444444",
            activebackground="#2b2b2b",
            activeforeground="#ffffff",
        ).pack(anchor="w", padx=12, pady=4)

        row_x = tk.Frame(top, bg="#2b2b2b")
        row_x.pack(fill="x", padx=12, pady=4)
        tk.Label(
            row_x,
            text="max_cjk_length_diff (x):",
            font=("微软雅黑", 9),
            bg="#2b2b2b",
            fg="#cccccc",
            width=22,
            anchor="w",
        ).pack(side=tk.LEFT)
        tk.Entry(row_x, textvariable=var_x, width=8, font=("微软雅黑", 9)).pack(
            side=tk.LEFT, padx=4
        )

        row_y = tk.Frame(top, bg="#2b2b2b")
        row_y.pack(fill="x", padx=12, pady=4)
        tk.Label(
            row_y,
            text="min_cjk_lcs_matches (y):",
            font=("微软雅黑", 9),
            bg="#2b2b2b",
            fg="#cccccc",
            width=22,
            anchor="w",
        ).pack(side=tk.LEFT)
        tk.Entry(row_y, textvariable=var_y, width=8, font=("微软雅黑", 9)).pack(
            side=tk.LEFT, padx=4
        )

        btn_row = tk.Frame(top, bg="#2b2b2b")
        btn_row.pack(pady=14)

        def on_ok() -> None:
            try:
                xd = int(str(var_x.get()).strip())
                yd = int(str(var_y.get()).strip())
            except ValueError:
                messagebox.showerror("错误", "x / y 请输入非负整数", parent=top)
                return
            if xd < 0 or yd < 0:
                messagebox.showerror("错误", "x / y 须 >= 0", parent=top)
                return
            self._colon_cjk_strategy_cfg["exclude_lines_without_year"] = bool(
                var_year.get()
            )
            self._colon_cjk_strategy_cfg["max_cjk_length_diff"] = xd
            self._colon_cjk_strategy_cfg["min_cjk_lcs_matches"] = yd
            self._strategy_colon_cjk = ColonCjkPhraseMatchStrategy(
                exclude_lines_without_year=self._colon_cjk_strategy_cfg[
                    "exclude_lines_without_year"
                ],
                max_cjk_length_diff=xd,
                min_cjk_lcs_matches=yd,
            )
            top.destroy()

        tk.Button(
            btn_row,
            text="确定",
            command=on_ok,
            font=("微软雅黑", 9),
            bg="#4CAF50",
            fg="white",
            padx=14,
            pady=4,
        ).pack(side=tk.LEFT, padx=6)
        tk.Button(
            btn_row,
            text="取消",
            command=top.destroy,
            font=("微软雅黑", 9),
            bg="#666666",
            fg="white",
            padx=14,
            pady=4,
        ).pack(side=tk.LEFT, padx=6)

    def check_required_production_expiry_boxes(self, boxes: list[dict]) -> bool:
        """
        使用界面选中的 ``production_phrase_strategy`` 策略，对 ``boxes`` 的 ``text``
        做三语校验。通过则返回 True；否则 :meth:`callRelayAction` 并返回 False。
        """
        texts = [str(b.get("text", "") or "") for b in boxes]
        strategy = self._active_production_strategy()
        if strategy.match(texts):
            self.callRelayAction()
            return True
        return False

    def _trigger_sdk_ints(self) -> Tuple[int, int, int]:
        """TriggerMode / TriggerSource ints from MVS headers (with safe fallbacks)."""
        g = globals()
        mode_on = int(g.get("MV_TRIGGER_MODE_ON", 1))
        mode_off = int(g.get("MV_TRIGGER_MODE_OFF", 0))
        line0 = int(g.get("MV_TRIGGER_SOURCE_LINE0", 0))
        return mode_on, mode_off, line0

    def _apply_internal_free_run_trigger(self) -> tuple[bool, str]:
        """Continuous acquisition: internal clock, no external line trigger."""
        mode_on, mode_off, _line0 = self._trigger_sdk_ints()
        ret = self.cam.MV_CC_SetEnumValue("TriggerMode", mode_off)
        if ret != 0:
            return False, f"TriggerMode=Off 失败 0x{ret:x}"
        return True, ""

    def _apply_hardware_line0_trigger(self) -> tuple[bool, str]:
        """Hardware trigger on physical Line0 (rising edge default on many models)."""
        mode_on, _mode_off, line0 = self._trigger_sdk_ints()
        ret = self.cam.MV_CC_SetEnumValue("TriggerMode", mode_on)
        if ret != 0:
            return False, f"TriggerMode=On 失败 0x{ret:x}"
        ret = self.cam.MV_CC_SetEnumValue("TriggerSource", line0)
        if ret != 0:
            return False, f"TriggerSource=Line0 失败 0x{ret:x}"
        return True, ""

    def _sdk_convert_raw_to_bgr(
        self,
        frame_data,
        width: int,
        height: int,
        pixel_type: int,
    ) -> Optional[np.ndarray]:
        """海康 SDK 转 BGR8（色彩/去马赛克与 MVS 预览更一致）；需已打开设备。"""
        if not self.b_open_device or width <= 0 or height <= 0:
            return None
        if width > 65535 or height > 65535:
            return None
        try:
            buf_len = len(frame_data)
            out_pixels = width * height * 3
            row_tight = width * 3
            row_stride = (row_tight + 3) // 4 * 4
            buf_sz = max(out_pixels, int(height * row_stride))
            st = MV_CC_PIXEL_CONVERT_PARAM()
            st.nWidth = width
            st.nHeight = height
            st.enSrcPixelType = int(pixel_type)
            st.pSrcData = cast(frame_data, POINTER(c_ubyte))
            st.nSrcDataLen = buf_len
            st.enDstPixelType = int(PixelType_Gvsp_BGR8_Packed)
            dst_buf = (c_ubyte * buf_sz)()
            st.pDstBuffer = cast(dst_buf, POINTER(c_ubyte))
            st.nDstBufferSize = buf_sz
            st.nDstLen = 0
            ret = int(self.cam.MV_CC_ConvertPixelType(byref(st)))
            if ret != int(MV_OK):
                return None
            nbytes = int(st.nDstLen) if int(st.nDstLen) > 0 else out_pixels
            if nbytes < out_pixels:
                return None
            rawv = np.frombuffer(memoryview(dst_buf)[:nbytes], dtype=np.uint8)
            if height > 0 and nbytes % height == 0:
                stride = nbytes // height
                if stride >= row_tight and stride != row_tight:
                    wide = rawv.reshape((height, stride))
                    img = wide[:, :row_tight].reshape((height, width, 3))
                    return np.ascontiguousarray(img)
            return np.ascontiguousarray(rawv[:out_pixels].reshape((height, width, 3)))
        except Exception:
            return None

    def _decode_raw_to_bgr(
        self,
        frame_data,
        width: int,
        height: int,
        pixel_type: int,
    ) -> Optional[np.ndarray]:
        """
        Decode one camera buffer to BGR.

        Prefer explicit packed formats, then MV_CC_ConvertPixelType（与 MVS 一致），
        再退回 OpenCV Bayer8。
        """
        try:
            pt = int(pixel_type)
            npx = width * height
            buf_len = len(frame_data)
            _bgr_packed = {
                int(PixelType_Gvsp_BGR8_Packed),
                int(PixelType_Gvsp_HB_BGR8_Packed),
            }
            _rgb_packed = {
                int(PixelType_Gvsp_RGB8_Packed),
                int(PixelType_Gvsp_HB_RGB8_Packed),
            }
            _bayer8opencv = frozenset(
                {
                    int(PixelType_Gvsp_BayerRG8),
                    int(PixelType_Gvsp_BayerGR8),
                    int(PixelType_Gvsp_BayerGB8),
                    int(PixelType_Gvsp_BayerBG8),
                    int(PixelType_Gvsp_HB_BayerRG8),
                    int(PixelType_Gvsp_HB_BayerGR8),
                    int(PixelType_Gvsp_HB_BayerGB8),
                    int(PixelType_Gvsp_HB_BayerBG8),
                }
            )

            if pt in _bgr_packed and buf_len >= npx * 3:
                img = np.array(frame_data, dtype=np.uint8, copy=True).reshape(
                    (height, width, 3)
                )
                return np.ascontiguousarray(img)
            if pt in _rgb_packed and buf_len >= npx * 3:
                img = np.array(frame_data, dtype=np.uint8, copy=True).reshape(
                    (height, width, 3)
                )
                return np.ascontiguousarray(
                    cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                )
            if pt == int(PixelType_Gvsp_Mono8) and buf_len >= npx:
                img = np.array(frame_data, dtype=np.uint8, copy=True).reshape(
                    (height, width)
                )
                return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

            sdk_bgr = self._sdk_convert_raw_to_bgr(
                frame_data, width, height, pixel_type
            )
            if sdk_bgr is not None and sdk_bgr.size > 0:
                return sdk_bgr

            if buf_len >= npx and pt in _bayer8opencv:
                img = np.frombuffer(frame_data, dtype=np.uint8, count=npx).reshape(
                    (height, width)
                )
                decoded = _decode_bayer8_to_bgr(img, pixel_type)
                if decoded is not None:
                    return decoded

            if buf_len >= npx:
                img = np.frombuffer(frame_data, dtype=np.uint8, count=npx).reshape(
                    (height, width)
                )
                return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        except Exception:
            return None
        return None

    @staticmethod
    def _bgr_resize_to_preview(
        bgr: np.ndarray, max_w: int = 640, max_h: int = 480
    ) -> np.ndarray:
        """Resize BGR image to fit preview area (keeps aspect, max max_w x max_h)."""
        if bgr is None or bgr.size == 0:
            return bgr
        if bgr.ndim == 2:
            bgr = cv2.cvtColor(bgr, cv2.COLOR_GRAY2BGR)
        h, w = bgr.shape[:2]
        if w <= 0 or h <= 0:
            return bgr
        scale = min(max_w / float(w), max_h / float(h), 1.0)
        nw = max(1, int(round(w * scale)))
        nh = max(1, int(round(h * scale)))
        if nw == w and nh == h:
            return np.ascontiguousarray(bgr)
        return cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_AREA)

    def _draw_ok_ng_overlay(self, img_bgr: np.ndarray, pass_check: bool) -> np.ndarray:
        """
        Draw an OK (green) / NG (red) badge in the **lower-middle** of the image:
        rectangle bottom sits above ~20% bottom margin; label centered inside the box.
        """
        out = img_bgr.copy()
        if out.ndim == 2:
            out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
        h, w = out.shape[:2]
        if w < 2 or h < 2:
            return out

        box_w = max(72, int(w * 0.18))
        box_h = max(44, int(h * 0.10))
        # ~20% of image height reserved as empty space below the badge
        margin_bottom = max(1, int(round(h * 0.20)))
        y2 = h - 1 - margin_bottom
        y1 = max(0, y2 - box_h + 1)
        rect_h = y2 - y1 + 1
        cx = w // 2
        x1 = max(0, cx - box_w // 2)
        x2 = min(w - 1, x1 + box_w - 1)
        if x2 - x1 < 2 or y2 <= y1:
            return out

        color = (0, 200, 0) if pass_check else (0, 0, 230)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness=-1)

        label = "OK" if pass_check else "NG"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = float(max(0.9, min(3.2, 0.35 * rect_h / 48.0 + 0.6)))
        thickness = max(2, int(round(font_scale * 2)))
        (tw, th), _baseline = cv2.getTextSize(label, font, font_scale, thickness)
        rect_w = x2 - x1 + 1
        tx = x1 + (rect_w - tw) // 2
        tx = max(x1, min(x2 - tw + 1, tx))
        # putText y = baseline; (rect_h + th) // 2 matches common OpenCV centering in a box
        ty = y1 + (rect_h + th) // 2
        ty = min(y2, max(y1 + 1, ty))
        cv2.putText(
            out,
            label,
            (tx, ty),
            font,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )
        return out

    def _record_hardware_trigger_capture(self) -> None:
        """Bump hard-trigger counter; persist stats; call from Tk main thread."""
        with self._stats_lock:
            self._hw_trigger_capture_count += 1
            n = self._hw_trigger_capture_count
        self._persist_stats_to_disk()
        self.update_capture_stats_display()
        self.update_status(f"硬触发已采集（累计 {n} 次）", "#00ff00")

    def _ocr_snapshot_json_dict(self, ocr_data: Dict[str, Any]) -> Dict[str, Any]:
        """Build a JSON-serializable OCR summary for disk (NG archive)."""
        rows: List[Dict[str, Any]] = []
        for b in ocr_data.get("boxes", []) or []:
            if not isinstance(b, dict):
                continue
            bbox_raw = b.get("bbox_xyxy", [])
            try:
                bb = np.asarray(bbox_raw, dtype=np.float64).ravel()
                bb_list = [float(x) for x in bb.tolist()] if bb.size else []
            except Exception:
                bb_list = []
            rows.append(
                {
                    "text": str(coerce_box_text(b.get("text"))),
                    "score": float(coerce_box_score(b.get("score"))),
                    "bbox_xyxy": bb_list,
                }
            )
        return {
            "file": ocr_data.get("file"),
            "mode": ocr_data.get("mode"),
            "infer_seconds": float(ocr_data.get("infer_seconds") or 0.0),
            "required_phrases_pass": bool(ocr_data.get("required_phrases_pass")),
            "paddle_raw_rec_count": ocr_data.get("paddle_raw_rec_count"),
            "paddle_boxes_after_filter": ocr_data.get("paddle_boxes_after_filter"),
            "boxes": rows,
        }

    def _save_ocr_original_by_result(
        self,
        frame_bgr: np.ndarray,
        pass_check: bool,
        meta_file: str,
        ocr_data: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Queue saving full-resolution original capture (no boxes / OK-NG overlay).

        NG: always save JPEG; also save OCR result JSON alongside when ``ocr_data`` is set.
        OK: 1/10 probability JPEG only. Directory: ``photo_save_dir``.
        Runs disk I/O on a short-lived daemon thread so the Tk main thread is not blocked.
        """
        try:
            if pass_check and random.random() >= 0.1:
                return
            tag = "OK" if pass_check else "NG"
            ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            safe = "".join(
                c if (c.isalnum() or c in "-_") else "_" for c in (meta_file or "shot")
            )[:48]
            fname = f"ocr_{safe}_{tag}_{ts}.jpg"
            path = os.path.join(self.photo_save_dir, fname)
            img = np.ascontiguousarray(frame_bgr).copy()
            json_payload: Optional[Dict[str, Any]] = None
            json_path: Optional[str] = None
            if ocr_data is not None:
                json_payload = self._ocr_snapshot_json_dict(ocr_data)
                json_path = os.path.join(
                    self.photo_save_dir, f"ocr_{safe}_{tag}_{ts}.json"
                )

            def _worker() -> None:
                try:
                    if not os.path.isdir(self.photo_save_dir):
                        os.makedirs(self.photo_save_dir, exist_ok=True)
                    cv2.imwrite(
                        path,
                        img,
                        [int(cv2.IMWRITE_JPEG_QUALITY), 95],
                    )
                    if json_payload is not None and json_path is not None:
                        with open(json_path, "w", encoding="utf-8") as f:
                            json.dump(json_payload, f, ensure_ascii=False, indent=2)
                except Exception:
                    pass

            threading.Thread(target=_worker, daemon=True).start()
        except Exception:
            pass

    def run_ocr_pipeline_on_frame(
        self,
        frame_bgr: np.ndarray,
        *,
        meta_file: str = "camera_live",
        update_camera_preview: bool = True,
        increment_manual_capture_counter: bool = False,
        dialog_on_success: bool = False,
        dialog_on_error: bool = True,
        persist_original_capture: bool = False,
    ) -> bool:
        """
        Run predict_boxes + UI updates (same logic as manual capture OCR core).

        Must run on the Tk main thread if it touches widgets / messagebox.

        If ``persist_original_capture`` is True (拍照+OCR / 硬触发), after OCR the
        original ``frame_bgr`` may be written under ``photo_save_dir`` (NG always, OK 1/10).
        NG saves a sibling ``.json`` with OCR boxes / scores / pass flag next to the JPEG.
        """
        if self.ocr_engine is None:
            if dialog_on_error:
                messagebox.showerror(
                    "错误",
                    self._ocr_init_error or "OCR 未初始化",
                )
            return False

        try:
            frame_bgr = prepare_bgr_for_predict(frame_bgr)
            t0 = time.perf_counter()
            boxes, paddle_debug = predict_boxes(
                self.ocr_engine, frame_bgr, return_debug=True
            )
            pass_check = self.check_required_production_expiry_boxes(boxes)
            infer_s = time.perf_counter() - t0

            ocr_data = {
                "boxes": boxes,
                "file": meta_file,
                "mode": "paddle_full_image_detect.predict_boxes",
                "infer_seconds": float(infer_s),
                "required_phrases_pass": pass_check,
                "paddle_raw_rec_count": paddle_debug.get("raw_rec_count"),
                "paddle_boxes_after_filter": paddle_debug.get("boxes_after_filter"),
            }
            vis_det = draw_detections(frame_bgr, boxes, draw_label=True)
            vis_ocr_ui = self._draw_ok_ng_overlay(vis_det, pass_check)

            if persist_original_capture:
                self._save_ocr_original_by_result(
                    frame_bgr,
                    pass_check,
                    meta_file,
                    ocr_data=None if pass_check else ocr_data,
                )

            self.current_ocr_result = ocr_data
            self.display_ocr_image_from_bgr(vis_ocr_ui)
            if update_camera_preview:
                prev = self._bgr_resize_to_preview(vis_det)
                self._apply_video_preview(prev)
                with self.frame_lock:
                    self.current_frame = prev.copy()

            with self._stats_lock:
                self._ocr_total_count += 1
                if pass_check:
                    self._ocr_ok_count += 1
                else:
                    self._ocr_ng_count += 1
                if increment_manual_capture_counter:
                    self._manual_capture_ocr_count += 1
            self._persist_stats_to_disk()
            self.update_capture_stats_display()

            long_texts = self.display_ocr_result(ocr_data)
            n = len(boxes)
            text_summary = f"合法文本框 {n} 个"
            if long_texts:
                text_summary += f"（列表中长度>=2 的共 {len(long_texts)} 个）"
            self.update_status(f"✅ OCR 完成 — {text_summary}", "#00ff00")
            if dialog_on_success:
                messagebox.showinfo(
                    "成功",
                    f"拍照+OCR 完成（不落盘）。\n\n{text_summary}\n耗时: {infer_s:.2f}s",
                )
            return True
        except Exception as e:
            if dialog_on_error:
                messagebox.showerror("错误", f"OCR 识别出错:\n{e}")
            self.update_status(f"OCR 错误: {e}", "#ff0000")
            return False

    def _handle_hardware_trigger_frame(self, bgr: np.ndarray) -> None:
        """Main thread: count trigger, then OCR on the full-resolution frame."""
        self._record_hardware_trigger_capture()
        self.run_ocr_pipeline_on_frame(
            bgr,
            meta_file="hardware_trigger",
            update_camera_preview=True,
            increment_manual_capture_counter=False,
            dialog_on_success=False,
            dialog_on_error=True,
            persist_original_capture=True,
        )

    def toggle_hardware_trigger_mode(self) -> None:
        """Switch between continuous preview and Line0 hardware trigger (saves JPEG on each frame)."""
        if not self.b_open_device:
            messagebox.showwarning("提示", "请先连接相机。")
            return
        if not self._mode_switch_lock.acquire(blocking=False):
            return
        try:
            self.b_start_grabbing = False
            if self.grab_thread is not None:
                self.grab_thread.join(timeout=3.0)

            self.cam.MV_CC_StopGrabbing()

            want_hw = not self.use_hw_trigger
            if want_hw:
                ok, msg = self._apply_hardware_line0_trigger()
                if not ok:
                    self._apply_internal_free_run_trigger()
                    ret0 = self.cam.MV_CC_StartGrabbing()
                    if ret0 != 0:
                        messagebox.showerror("错误", f"恢复采集失败: 0x{ret0:x}")
                    self.b_start_grabbing = True
                    self.grab_thread = threading.Thread(
                        target=self.grab_thread_func, daemon=True
                    )
                    self.grab_thread.start()
                    messagebox.showerror("硬触发", f"无法切换到硬触发:\n{msg}")
                    return
            else:
                ok, msg = self._apply_internal_free_run_trigger()
                if not ok:
                    messagebox.showerror("连续采集", msg)

            ret = self.cam.MV_CC_StartGrabbing()
            if ret != 0:
                messagebox.showerror("错误", f"开始采集失败: 0x{ret:x}")
                self.update_status("开始采集失败，尝试恢复连续模式…", "#ff0000")
                self._apply_internal_free_run_trigger()
                ret2 = self.cam.MV_CC_StartGrabbing()
                if ret2 != 0:
                    messagebox.showerror("错误", f"恢复连续采集失败: 0x{ret2:x}")
                self.use_hw_trigger = False
                self.btn_toggle_trigger.config(text="切换到硬触发(Line0)")
                self.b_start_grabbing = True
                self.grab_thread = threading.Thread(
                    target=self.grab_thread_func, daemon=True
                )
                self.grab_thread.start()
                return

            if want_hw:
                self.use_hw_trigger = True
                self.btn_toggle_trigger.config(text="切换到连续采集")
                self.update_status("硬触发: Line0，等待触发脉冲…", "#ffff00")
            else:
                self.use_hw_trigger = False
                self.btn_toggle_trigger.config(text="切换到硬触发(Line0)")
                self.update_status("连续采集（内部触发）", "#00ff00")

            self.b_start_grabbing = True
            self.grab_thread = threading.Thread(
                target=self.grab_thread_func, daemon=True
            )
            self.grab_thread.start()
        finally:
            self._mode_switch_lock.release()

    @staticmethod
    def decoding_char(ctypes_char_array):
        byte_str = memoryview(ctypes_char_array).tobytes()
        null_index = byte_str.find(b'\x00')
        if null_index != -1:
            byte_str = byte_str[:null_index]
        for encoding in ['gbk', 'utf-8', 'latin-1']:
            try:
                return byte_str.decode(encoding)
            except UnicodeDecodeError:
                continue
        return byte_str.decode('latin-1', errors='ignore')

    @staticmethod
    def ip_int_to_str(ip_int):
        nip1 = ((ip_int & 0xff000000) >> 24)
        nip2 = ((ip_int & 0x00ff0000) >> 16)
        nip3 = ((ip_int & 0x0000ff00) >> 8)
        nip4 = (ip_int & 0x000000ff)
        return f"{nip1}.{nip2}.{nip3}.{nip4}"

    def _safe_config_filename_serial(self, serial: str) -> str:
        s = (serial or "").strip() or "unknown"
        bad = '<>:"/\\|?*'
        out = "".join("_" if c in bad or ord(c) < 32 else c for c in s)
        return out[:120]

    def _camera_config_json_path(self) -> str:
        if not os.path.isdir(self._camera_config_dir):
            os.makedirs(self._camera_config_dir, exist_ok=True)
        sn = self._safe_config_filename_serial(self._camera_serial_for_config)
        return os.path.join(self._camera_config_dir, f"{sn}.json")

    def _read_floatish_genicam_numeric(self, key: str, row: Dict[str, Any]) -> None:
        """
        Read nodes like ``ExposureTime`` / ``Gain`` that MVS shows as large integers (e.g. µs)
        but may report misleading values via :meth:`MV_CC_GetFloatValue`.
        Prefer ``GetIntValueEx`` → ``GetIntValue`` → ``GetFloatValue``.
        """
        last_err = 0
        stex = MVCC_INTVALUE_EX()
        ret_ex = int(self.cam.MV_CC_GetIntValueEx(key, stex))
        if ret_ex == 0:
            row["ftype"] = "int"
            row["value"] = str(int(stex.nCurValue))
            row["range"] = (
                f"[{int(stex.nMin)}, {int(stex.nMax)}] step {int(stex.nInc)}"
            )
            row["read_ok"] = True
            return
        last_err = ret_ex

        sti = MVCC_INTVALUE()
        ret_i = int(self.cam.MV_CC_GetIntValue(key, sti))
        if ret_i == 0:
            row["ftype"] = "int"
            row["value"] = str(int(sti.nCurValue))
            row["range"] = f"[{sti.nMin}, {sti.nMax}] step {sti.nInc}"
            row["read_ok"] = True
            return
        last_err = ret_i

        stf = MVCC_FLOATVALUE()
        ret_f = int(self.cam.MV_CC_GetFloatValue(key, stf))
        if ret_f == 0:
            fv = float(stf.fCurValue)
            row["ftype"] = "float"
            row["value"] = _format_genicam_float_for_display(fv)
            row["range"] = f"[{stf.fMin:g}, {stf.fMax:g}]"
            row["read_ok"] = True
            return
        last_err = ret_f
        row["read_err"] = last_err

    def _collect_camera_config_rows(self) -> List[Dict[str, Any]]:
        """Read GenICam features from the open device (best-effort)."""
        rows: List[Dict[str, Any]] = []
        for key, ftype in _CAMERA_FEATURE_SCHEMA:
            row: Dict[str, Any] = {
                "key": key,
                "ftype": ftype,
                "value": "",
                "range": "",
                "read_ok": False,
                "read_err": 0,
            }
            if ftype == "int":
                st = MVCC_INTVALUE()
                ret = self.cam.MV_CC_GetIntValue(key, st)
                if ret == 0:
                    row["value"] = str(int(st.nCurValue))
                    row["range"] = f"[{st.nMin}, {st.nMax}] step {st.nInc}"
                    row["read_ok"] = True
                else:
                    row["read_err"] = int(ret)
            elif ftype == "float":
                if key in _FLAKY_FLOAT_GENICAM_KEYS:
                    self._read_floatish_genicam_numeric(key, row)
                else:
                    stf = MVCC_FLOATVALUE()
                    ret = self.cam.MV_CC_GetFloatValue(key, stf)
                    if ret == 0:
                        fv = float(stf.fCurValue)
                        row["value"] = _format_genicam_float_for_display(fv)
                        row["range"] = f"[{stf.fMin:g}, {stf.fMax:g}]"
                        row["read_ok"] = True
                    else:
                        sti = MVCC_INTVALUE()
                        ret2 = self.cam.MV_CC_GetIntValue(key, sti)
                        if ret2 == 0:
                            row["ftype"] = "int"
                            row["value"] = str(int(sti.nCurValue))
                            row["range"] = (
                                f"[{sti.nMin}, {sti.nMax}] step {sti.nInc}"
                            )
                            row["read_ok"] = True
                        else:
                            row["read_err"] = int(ret)
            elif ftype == "bool":
                bv = c_bool(False)
                ret = self.cam.MV_CC_GetBoolValue(key, bv)
                if ret == 0:
                    row["value"] = "true" if bv.value else "false"
                    row["range"] = "true | false"
                    row["read_ok"] = True
                else:
                    row["read_err"] = int(ret)
            elif ftype == "string":
                st = MVCC_STRINGVALUE()
                ret = self.cam.MV_CC_GetStringValue(key, st)
                if ret == 0:
                    raw = bytes(st.chCurValue).split(b"\x00", 1)[0]
                    row["value"] = raw.decode("utf-8", errors="replace")
                    row["range"] = f"max_len={int(st.nMaxLength)}"
                    row["read_ok"] = True
                else:
                    row["read_err"] = int(ret)
            elif ftype == "enum":
                ev = MVCC_ENUMVALUE()
                ret = self.cam.MV_CC_GetEnumValue(key, ev)
                if ret == 0:
                    entry = MVCC_ENUMENTRY()
                    ctypes.memset(byref(entry), 0, ctypes.sizeof(entry))
                    entry.nValue = int(ev.nCurValue)
                    ret2 = self.cam.MV_CC_GetEnumEntrySymbolic(key, entry)
                    sym = ""
                    if ret2 == 0:
                        sym = (
                            bytes(entry.chSymbolic)
                            .split(b"\x00", 1)[0]
                            .decode("ascii", errors="replace")
                        )
                    row["value"] = sym or str(int(ev.nCurValue))
                    row["range"] = f"enum cur={int(ev.nCurValue)}"
                    row["read_ok"] = True
                else:
                    row["read_err"] = int(ret)
            rows.append(row)
        return rows

    def _write_camera_config_json_file(self, rows: List[Dict[str, Any]]) -> None:
        payload = {
            "serial": self._camera_serial_for_config,
            "saved_at": datetime.now(timezone.utc).isoformat(),
            "rows": [
                {
                    "key": r["key"],
                    "ftype": r["ftype"],
                    "value": r.get("value", ""),
                    "range": r.get("range", ""),
                    "read_ok": bool(r.get("read_ok")),
                }
                for r in rows
            ],
        }
        path = self._camera_config_json_path()
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    def _snapshot_and_save_camera_config(self) -> None:
        """After connect: pull GenICam snapshot and store under ``camera_configs/``."""
        if not self.b_open_device:
            return
        try:
            rows = self._collect_camera_config_rows()
            self._write_camera_config_json_file(rows)
        except Exception as e:
            messagebox.showwarning(
                "相机配置",
                f"已连接相机，但写入本地配置备份失败:\n{e}",
            )

    def _pause_camera_grabbing(self, join_timeout: float = 4.0) -> None:
        """Stop grab thread and SDK grabbing (safe before ROI / exposure writes)."""
        self.b_start_grabbing = False
        if self.grab_thread is not None:
            self.grab_thread.join(timeout=join_timeout)
            self.grab_thread = None
        if self.b_open_device:
            self.cam.MV_CC_StopGrabbing()

    def _resume_camera_grabbing(self) -> tuple[int, str]:
        """Restart SDK grabbing and preview thread. Returns (ret_code, message)."""
        if not self.b_open_device:
            return 0, ""
        ret = self.cam.MV_CC_StartGrabbing()
        if ret != 0:
            return int(ret), f"MV_CC_StartGrabbing 失败 0x{int(ret):x}"
        self.b_start_grabbing = True
        self.grab_thread = threading.Thread(target=self.grab_thread_func, daemon=True)
        self.grab_thread.start()
        try:
            self.root.after(0, self._pump_preview_queue)
        except Exception:
            pass
        return 0, ""

    def _try_set_enum_by_string(self, key: str, symbolic: str) -> int:
        """Return MV_OK (0) if SetEnumValueByString succeeds."""
        return int(self.cam.MV_CC_SetEnumValueByString(key, symbolic))

    def _ensure_exposure_auto_off(self) -> None:
        """Manual exposure time requires auto exposure off (GenICam)."""
        for sym in ("Off", "off", "OFF"):
            if self._try_set_enum_by_string("ExposureAuto", sym) == 0:
                return
        try:
            self.cam.MV_CC_SetEnumValue(
                "ExposureAuto", int(MV_EXPOSURE_AUTO_MODE_OFF)
            )
        except Exception:
            pass

    def _ensure_gain_auto_off(self) -> None:
        for sym in ("Off", "off", "OFF"):
            if self._try_set_enum_by_string("GainAuto", sym) == 0:
                return
        try:
            self.cam.MV_CC_SetEnumValue("GainAuto", int(MV_GAIN_MODE_OFF))
        except Exception:
            pass

    def _apply_camera_config_rows(self, rows: List[Dict[str, Any]]) -> List[str]:
        """
        Push edited values to the device.

        Caller should normally have called :meth:`_pause_camera_grabbing` first so
        Width/Height/Offset/PixelFormat can be applied. ExposureTime needs
        ExposureAuto off; Gain needs GainAuto off.
        """
        errors: List[str] = []
        sorted_rows = sorted(rows, key=lambda r: _camera_param_apply_sort_key(str(r["key"])))

        for r in sorted_rows:
            key = str(r["key"])
            ftype = str(r["ftype"])
            s = str(r.get("value", "")).strip()
            if not s:
                continue
            ret = 0xFFFFFFFF
            try:
                if key == "ExposureTime":
                    self._ensure_exposure_auto_off()
                if key == "Gain":
                    self._ensure_gain_auto_off()

                if ftype == "int":
                    iv = int(s)
                    if key in _FLAKY_FLOAT_GENICAM_KEYS:
                        ret = int(self.cam.MV_CC_SetIntValueEx(key, iv))
                        if ret != 0:
                            ret = int(self.cam.MV_CC_SetIntValue(key, iv))
                    else:
                        ret = int(self.cam.MV_CC_SetIntValue(key, iv))
                elif ftype == "float":
                    v = float(s)
                    if key in _FLAKY_FLOAT_GENICAM_KEYS:
                        ret = int(self.cam.MV_CC_SetFloatValue(key, v))
                        if ret != 0:
                            iv = int(round(v))
                            ret = int(self.cam.MV_CC_SetIntValueEx(key, iv))
                            if ret != 0:
                                ret = int(self.cam.MV_CC_SetIntValue(key, iv))
                    else:
                        ret = int(self.cam.MV_CC_SetFloatValue(key, v))
                        if ret != 0:
                            sti = MVCC_INTVALUE()
                            get_ret = int(self.cam.MV_CC_GetIntValue(key, sti))
                            if get_ret == 0:
                                ret = int(
                                    self.cam.MV_CC_SetIntValue(
                                        key, int(round(v))
                                    )
                                )
                elif ftype == "bool":
                    low = s.lower()
                    b = low in ("1", "true", "yes", "on")
                    ret = int(self.cam.MV_CC_SetBoolValue(key, b))
                elif ftype == "string":
                    ret = int(self.cam.MV_CC_SetStringValue(key, s))
                elif ftype == "enum":
                    sym = _normalize_genicam_enum_string_for_set(key, s)
                    if sym.isdigit():
                        ret = int(self.cam.MV_CC_SetEnumValue(key, int(sym)))
                    else:
                        ret = int(self.cam.MV_CC_SetEnumValueByString(key, sym))
                else:
                    errors.append(f"{key}: 未知类型 {ftype}")
                    continue
            except ValueError as ve:
                errors.append(f"{key}: 数值无效 ({ve})")
                continue
            if ret != 0:
                errors.append(f"{key}: 写入失败 0x{int(ret):x}")
        return errors

    def open_camera_config_dialog(self) -> None:
        if not self.b_open_device:
            messagebox.showwarning("配置相机", "请先连接相机。")
            return

        win = tk.Toplevel(self.root)
        win.title("配置相机")
        win.geometry("900x560")
        win.configure(bg="#2b2b2b")

        hint = tk.Label(
            win,
            text=(
                "保存时会短暂停止取流再写入相机（否则 Width/Height 等多数机型无法在采集中修改）。"
                "修改曝光时间前会自动尝试关闭 ExposureAuto；修改 Gain 前会尝试关闭 GainAuto。"
                "若仍失败请对照「范围」检查步进与单位（曝光多为微秒）。"
                "PixelFormat 请填相机 XML 的 Symbolic（如 BayerRG8），与 MVS 里「Bayer RG 8」对应但无空格；"
                "程序保存时会自动去掉空格并纠正 Bayer 大小写。"
            ),
            font=("微软雅黑", 9),
            bg="#2b2b2b",
            fg="#cccccc",
            wraplength=860,
            justify="left",
        )
        hint.pack(fill="x", padx=10, pady=(8, 4))

        outer = tk.Frame(win, bg="#2b2b2b")
        outer.pack(fill="both", expand=True, padx=8, pady=4)

        canvas = tk.Canvas(outer, bg="#1a1a1a", highlightthickness=0)
        vsb = ttk.Scrollbar(outer, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=vsb.set)
        vsb.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)

        inner = tk.Frame(canvas, bg="#1a1a1a")
        cid = canvas.create_window((0, 0), window=inner, anchor="nw")

        def _on_inner_configure(_event=None):
            canvas.configure(scrollregion=canvas.bbox("all"))

        def _on_canvas_configure(e):
            canvas.itemconfigure(cid, width=e.width)

        inner.bind("<Configure>", _on_inner_configure)
        canvas.bind("<Configure>", _on_canvas_configure)

        hdrs = ("参数", "类型", "值", "范围 / 状态")
        for c, t in enumerate(hdrs):
            tk.Label(
                inner,
                text=t,
                font=("微软雅黑", 9, "bold"),
                bg="#1a1a1a",
                fg="#00ff88",
            ).grid(row=0, column=c, sticky="w", padx=6, pady=4)

        row_widgets: List[Tuple[str, str, tk.StringVar]] = []

        def rebuild_table(rows_in: List[Dict[str, Any]]) -> None:
            for w in inner.grid_slaves():
                if int(w.grid_info().get("row", 0)) > 0:
                    w.destroy()
            row_widgets.clear()
            for i, r in enumerate(rows_in, start=1):
                tk.Label(
                    inner,
                    text=r["key"],
                    font=("Consolas", 9),
                    bg="#1a1a1a",
                    fg="#e0e0e0",
                ).grid(row=i, column=0, sticky="w", padx=6, pady=1)
                tk.Label(
                    inner,
                    text=str(r["ftype"]),
                    font=("Consolas", 9),
                    bg="#1a1a1a",
                    fg="#a0c4ff",
                ).grid(row=i, column=1, sticky="w", padx=4, pady=1)
                v = tk.StringVar(value=str(r.get("value", "")))
                row_widgets.append((str(r["key"]), str(r["ftype"]), v))
                tk.Entry(
                    inner,
                    textvariable=v,
                    font=("Consolas", 10),
                    width=26,
                    bg="#2b2b2b",
                    fg="#ffffff",
                    insertbackground="white",
                ).grid(row=i, column=2, sticky="w", padx=4, pady=1)
                hint_txt = (
                    str(r.get("range", ""))
                    if r.get("read_ok")
                    else f"读取失败 0x{int(r.get('read_err', 0)):x}"
                )
                tk.Label(
                    inner,
                    text=hint_txt,
                    font=("微软雅黑", 8),
                    bg="#1a1a1a",
                    fg="#888888",
                    wraplength=420,
                    justify="left",
                ).grid(row=i, column=3, sticky="w", padx=4, pady=1)
            _on_inner_configure()

        def refresh_from_camera():
            try:
                rows = self._collect_camera_config_rows()
            except Exception as e:
                messagebox.showerror("配置相机", f"读取失败: {e}")
                return
            rebuild_table(rows)

        def save_apply():
            rows_out: List[Dict[str, Any]] = []
            for key, ftype, var in row_widgets:
                rows_out.append(
                    {
                        "key": key,
                        "ftype": ftype,
                        "value": var.get(),
                        "read_ok": True,
                    }
                )
            if not self._mode_switch_lock.acquire(blocking=True, timeout=30.0):
                messagebox.showwarning("保存", "正在切换触发模式，请稍后再试保存。")
                return
            errs: List[str] = []
            resume_err = ""
            try:
                self._pause_camera_grabbing()
                try:
                    errs = self._apply_camera_config_rows(rows_out)
                finally:
                    rcode, rmsg = self._resume_camera_grabbing()
                    if rcode != 0:
                        resume_err = rmsg
                        errs = list(errs) + [f"恢复采集: {rmsg}"]
            finally:
                self._mode_switch_lock.release()

            if errs:
                messagebox.showwarning(
                    "保存",
                    "部分参数未成功写入相机或恢复采集异常:\n"
                    + "\n".join(errs[:14])
                    + ("\n..." if len(errs) > 14 else ""),
                )
            try:
                refreshed = self._collect_camera_config_rows()
                self._write_camera_config_json_file(refreshed)
                rebuild_table(refreshed)
                tail = f"\n{resume_err}" if resume_err else ""
                messagebox.showinfo(
                    "保存",
                    f"已尝试写入相机（保存时已暂停取流）；本地备份已更新:\n"
                    f"{self._camera_config_json_path()}{tail}",
                )
            except Exception as e:
                messagebox.showerror("保存", f"写入相机后刷新/保存 JSON 失败: {e}")

        btn_bar = tk.Frame(win, bg="#2b2b2b")
        btn_bar.pack(fill="x", padx=10, pady=8)
        tk.Button(
            btn_bar,
            text="从相机刷新",
            command=refresh_from_camera,
            font=("微软雅黑", 10),
            width=14,
        ).pack(side="left", padx=4)
        tk.Button(
            btn_bar,
            text="保存（写入相机 + 本机 JSON）",
            command=save_apply,
            font=("微软雅黑", 10),
            width=26,
            bg="#1976D2",
            fg="white",
        ).pack(side="left", padx=4)
        tk.Button(
            btn_bar,
            text="关闭",
            command=win.destroy,
            font=("微软雅黑", 10),
            width=10,
        ).pack(side="right", padx=4)

        refresh_from_camera()

    def enum_devices(self):
        self.device_list = MV_CC_DEVICE_INFO_LIST()
        ret = self.cam.MV_CC_EnumDevices(MV_GIGE_DEVICE, self.device_list)
        if ret != 0:
            return None
        return self.device_list.nDeviceNum > 0

    def connect_camera(self):
        self.update_status("正在枚举设备...", "#ffff00")
        self.root.update()

        if not self.enum_devices():
            messagebox.showerror("错误", "未找到任何设备，请检查相机连接！")
            self.update_status("未找到设备", "#ff0000")
            return

        device_num = self.device_list.nDeviceNum
        self.update_status(f"找到 {device_num} 个设备，正在连接...", "#ffff00")
        self.root.update()

        stDevInfo = cast(
            self.device_list.pDeviceInfo[0],
            POINTER(MV_CC_DEVICE_INFO)
        ).contents

        stGigEInfo = stDevInfo.SpecialInfo.stGigEInfo
        camera_info = (
            f"{self.decoding_char(stGigEInfo.chModelName)} | "
            f"IP: {self.ip_int_to_str(stGigEInfo.nCurrentIp)} | "
            f"序列号: {self.decoding_char(stGigEInfo.chSerialNumber)}"
        )
        self._camera_serial_for_config = (
            self.decoding_char(stGigEInfo.chSerialNumber).strip() or "unknown"
        )
        self.update_camera_info(camera_info)

        ret = self.cam.MV_CC_CreateHandle(stDevInfo)
        if ret != 0:
            messagebox.showerror("错误", f"创建设备句柄失败，错误码: 0x{ret:x}")
            self.update_status("创建句柄失败", "#ff0000")
            return

        ret = self.cam.MV_CC_OpenDevice(MV_ACCESS_Control, 0)
        if ret != 0:
            messagebox.showerror("错误", f"打开设备失败，错误码: 0x{ret:x}\n可能原因：\n1. MVS客户端正在占用相机\n2. 权限不足")
            self.cam.MV_CC_DestroyHandle()
            self.update_status("打开设备失败", "#ff0000")
            return

        if stDevInfo.nTLayerType == MV_GIGE_DEVICE:
            nPacketSize = self.cam.MV_CC_GetOptimalPacketSize()
            if int(nPacketSize) > 0:
                self.cam.MV_CC_SetIntValue("GevSCPSPacketSize", nPacketSize)

        # 强制内部连续采集，避免相机/MVS 上次留在硬触发导致取不到帧、预览卡住
        ok_trig, trig_msg = self._apply_internal_free_run_trigger()
        if not ok_trig:
            messagebox.showwarning(
                "触发模式",
                f"未能确认连续采集(TriggerMode=Off):\n{trig_msg}\n若预览异常请在 MVS 中检查触发。",
            )

        ret = self.cam.MV_CC_StartGrabbing()
        if ret != 0:
            messagebox.showerror("错误", f"开始采集失败，错误码: 0x{ret:x}")
            self.cam.MV_CC_CloseDevice()
            self.cam.MV_CC_DestroyHandle()
            self.update_status("开始采集失败", "#ff0000")
            return

        self.b_open_device = True
        self.b_start_grabbing = True
        self.use_hw_trigger = False
        try:
            self.cam.MV_CC_SetBayerCvtQuality(2)
        except Exception:
            pass
        self.btn_toggle_trigger.config(
            state="normal", text="切换到硬触发(Line0)"
        )

        self.btn_connect.config(state="disabled")
        self.btn_disconnect.config(state="normal")
        self.btn_capture.config(state="normal")
        self.btn_config_camera.config(state="normal")

        self.update_status("相机已连接，正在采集...", "#00ff00")

        try:
            while True:
                self._preview_queue.get_nowait()
        except queue.Empty:
            pass

        self.grab_thread = threading.Thread(target=self.grab_thread_func, daemon=True)
        self.grab_thread.start()
        self.root.after(0, self._pump_preview_queue)
        self._snapshot_and_save_camera_config()

    def _apply_video_preview(self, bgr: np.ndarray) -> None:
        """Update camera preview from a BGR image (Tk main thread only)."""
        try:
            if bgr is None or bgr.size == 0:
                return
            if bgr.ndim == 2:
                bgr = cv2.cvtColor(bgr, cv2.COLOR_GRAY2BGR)
            if bgr.ndim != 3 or bgr.shape[2] != 3:
                return
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(rgb)
            img_tk = ImageTk.PhotoImage(img_pil)
            self._video_preview_photo = img_tk
            self.video_label.config(image=img_tk, text="")
        except Exception:
            pass

    def _pump_preview_queue(self) -> None:
        """Drain preview frames on Tk main thread (started from connect, ~30 FPS)."""
        if self.b_exit:
            return
        last: Optional[np.ndarray] = None
        try:
            while True:
                last = self._preview_queue.get_nowait()
        except queue.Empty:
            pass
        if last is not None:
            self._apply_video_preview(last)
        if self.b_start_grabbing and not self.b_exit:
            self.root.after(33, self._pump_preview_queue)

    def _schedule_video_preview(self, bgr: np.ndarray) -> None:
        """Grab thread enqueues BGR preview; main thread pump paints ``video_label``."""
        snap = np.ascontiguousarray(bgr).copy()
        try:
            self._preview_queue.put_nowait(snap)
        except queue.Full:
            try:
                self._preview_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._preview_queue.put_nowait(snap)
            except queue.Full:
                pass

    def grab_thread_func(self):
        stFrameInfo = MV_FRAME_OUT()

        while self.b_start_grabbing and not self.b_exit:
            memset(byref(stFrameInfo), 0, sizeof(stFrameInfo))
            timeout_ms = 5000 if self.use_hw_trigger else 100
            ret = self.cam.MV_CC_GetImageBuffer(stFrameInfo, timeout_ms)
            if ret == 0 and stFrameInfo.pBufAddr is not None:
                width = stFrameInfo.stFrameInfo.nWidth
                height = stFrameInfo.stFrameInfo.nHeight
                pixel_type = stFrameInfo.stFrameInfo.enPixelType
                img_size = stFrameInfo.stFrameInfo.nFrameLen

                frame_data = (c_ubyte * img_size)()
                memmove(frame_data, stFrameInfo.pBufAddr, img_size)

                self.cam.MV_CC_FreeImageBuffer(stFrameInfo)

                if self.use_hw_trigger:
                    try:
                        bgr = self._decode_raw_to_bgr(
                            frame_data, width, height, pixel_type
                        )
                        if bgr is not None and bgr.size > 0:
                            preview = self._bgr_resize_to_preview(bgr)
                            with self.frame_lock:
                                self.current_frame = preview.copy()
                            self._schedule_video_preview(preview)
                            snap = np.ascontiguousarray(bgr).copy()
                            try:
                                self.root.after(
                                    0,
                                    lambda img=snap: self._handle_hardware_trigger_frame(
                                        img
                                    ),
                                )
                            except Exception:
                                pass
                    except Exception:
                        pass
                    continue

                img = self._decode_raw_to_bgr(
                    frame_data, width, height, pixel_type
                )
                if img is None or img.size == 0:
                    continue

                img = self._bgr_resize_to_preview(img)

                with self.frame_lock:
                    self.current_frame = img.copy()
                self._schedule_video_preview(img)

            elif ret != 0:
                time.sleep(0.01)

    def get_highres_frame(self):
        stFrameInfo = MV_FRAME_OUT()
        memset(byref(stFrameInfo), 0, sizeof(stFrameInfo))

        timeout_ms = 5000 if self.use_hw_trigger else 1000
        ret = self.cam.MV_CC_GetImageBuffer(stFrameInfo, timeout_ms)
        if ret == 0 and stFrameInfo.pBufAddr is not None:
            width = stFrameInfo.stFrameInfo.nWidth
            height = stFrameInfo.stFrameInfo.nHeight
            pixel_type = stFrameInfo.stFrameInfo.enPixelType
            img_size = stFrameInfo.stFrameInfo.nFrameLen

            frame_data = (c_ubyte * img_size)()
            memmove(frame_data, stFrameInfo.pBufAddr, img_size)

            self.cam.MV_CC_FreeImageBuffer(stFrameInfo)

            decoded = self._decode_raw_to_bgr(
                frame_data, width, height, pixel_type
            )
            if decoded is not None:
                return np.ascontiguousarray(decoded.copy())

        return self.current_frame if self.current_frame is not None else np.zeros((480, 640, 3), dtype=np.uint8)

    def capture_and_ocr(self):
        if self.ocr_engine is None:
            msg = self._ocr_init_error or "OCR 未初始化"
            messagebox.showerror("错误", msg)
            return

        self.update_status("正在拍照...", "#ffff00")
        self.root.update()

        frame = self.get_highres_frame()
        frame_bgr = self._ensure_bgr_u8(frame)
        if frame_bgr is None:
            messagebox.showerror("错误", "无法获取图像！")
            self.update_status("获取图像失败", "#ff0000")
            return

        self.btn_capture.config(state="disabled")
        self.update_status("正在 OCR 识别...", "#ffff00")
        self.root.update()

        try:
            self.run_ocr_pipeline_on_frame(
                frame_bgr,
                meta_file="camera_live",
                update_camera_preview=True,
                increment_manual_capture_counter=True,
                dialog_on_success=False,
                dialog_on_error=True,
                persist_original_capture=True,
            )
        finally:
            self.btn_capture.config(state="normal")

    def disconnect_camera(self):
        self.b_start_grabbing = False
        if self.grab_thread is not None:
            self.grab_thread.join(timeout=2.0)

        if self.b_open_device:
            self.cam.MV_CC_StopGrabbing()
            if self.use_hw_trigger:
                self._apply_internal_free_run_trigger()
            self.cam.MV_CC_CloseDevice()
            self.cam.MV_CC_DestroyHandle()
            self.b_open_device = False

        self.use_hw_trigger = False
        self.grab_thread = None
        self.btn_toggle_trigger.config(
            state="disabled", text="切换到硬触发(Line0)"
        )

        self.btn_connect.config(state="normal")
        self.btn_disconnect.config(state="disabled")
        self.btn_capture.config(state="disabled")
        self.btn_config_camera.config(state="disabled")
        self.video_label.config(image="", text="相机预览区域")
        self.video_label.image = None

        self.update_status("相机已断开", "#ffff00")
        self.update_camera_info("相机信息: 未连接")

    def open_photo_folder(self):
        if os.path.exists(self.photo_save_dir):
            os.startfile(self.photo_save_dir)
        else:
            messagebox.showwarning("警告", f"保存目录不存在: {self.photo_save_dir}")

    def on_closing(self):
        if messagebox.askokcancel("退出", "确定要退出程序吗？"):
            self.b_exit = True
            self.b_start_grabbing = False
            if self.grab_thread is not None:
                self.grab_thread.join(timeout=2.0)

            if self.b_open_device:
                self.cam.MV_CC_StopGrabbing()
                if self.use_hw_trigger:
                    self._apply_internal_free_run_trigger()
                self.cam.MV_CC_CloseDevice()
                self.cam.MV_CC_DestroyHandle()

            self._persist_stats_to_disk()
            self.root.destroy()
            sys.exit(0)

    def run(self):
        self.update_capture_stats_display()
        self.root.mainloop()


if __name__ == "__main__":
    print("=" * 60)
    print("海康工业相机 + OCR识别系统")
    print("=" * 60)
    app = HikCameraApp()
    app.run()
