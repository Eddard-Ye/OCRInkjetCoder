# -*- coding: utf-8 -*-
"""Structured OK/NG reports for phrase strategy + date check."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from date_check_config import DateCheckGlobalConfig, diagnose_shelf_life_dates
from pixel_match_config import PixelMatchConfig
from pixel_match_strategy import diagnose_pixel_match
from production_phrase_strategy import ColonCjkPhraseMatchStrategy


def _strategy_block(
    strategy: ColonCjkPhraseMatchStrategy,
    strategy_cfg: Dict[str, Any],
    texts: Sequence[str],
) -> Dict[str, Any]:
    enabled = bool(strategy_cfg.get("enabled", True))
    params = {
        "enabled": enabled,
        "required_phrases": list(strategy.phrases),
        "max_cjk_length_diff": int(strategy_cfg.get("max_cjk_length_diff", 2)),
        "min_match_percentage_limit": float(
            strategy_cfg.get("min_match_percentage_limit", 0.75)
        ),
    }
    if not enabled:
        return {
            "class_name": "ColonCjkPhraseMatchStrategy",
            "enabled": False,
            "skipped": True,
            "params": params,
            "passed": None,
            "text_box_count": len(texts),
            "diagnosis": None,
        }

    diag = strategy.diagnose(texts)
    return {
        "class_name": "ColonCjkPhraseMatchStrategy",
        "enabled": True,
        "skipped": False,
        "params": params,
        "passed": bool(diag.get("passed")),
        "text_box_count": diag.get("text_box_count"),
        "diagnosis": diag,
    }


def _phrase_checks_pass(
    strategy_block: Dict[str, Any],
    pixel_block: Dict[str, Any],
) -> tuple[bool, Optional[str], str]:
    """
    Combine CJK and pixel match with OR when both enabled.

    Returns (passed, ng_trigger, combine_mode) where combine_mode is
    ``or``, ``cjk_only``, ``pixel_only``, or ``none``.
    """
    cjk_enabled = bool(strategy_block.get("enabled"))
    pixel_enabled = bool(pixel_block.get("enabled"))

    cjk_pass = strategy_block.get("passed") if cjk_enabled else None
    pixel_pass = pixel_block.get("passed") if pixel_enabled else None

    if cjk_enabled and pixel_enabled:
        if bool(cjk_pass) or bool(pixel_pass):
            return True, None, "or"
        return False, "cjk_and_pixel", "or"

    if cjk_enabled:
        ok = bool(cjk_pass)
        return ok, (None if ok else "strategy"), "cjk_only"

    if pixel_enabled:
        ok = bool(pixel_pass)
        return ok, (None if ok else "pixel_match"), "pixel_only"

    return True, None, "none"


def build_ocr_check_report(
    texts: Sequence[str],
    *,
    strategy: ColonCjkPhraseMatchStrategy,
    strategy_cfg: Dict[str, Any],
    date_cfg: DateCheckGlobalConfig,
    boxes: Optional[Sequence[dict[str, Any]]] = None,
    pixel_cfg: Optional[PixelMatchConfig] = None,
) -> Dict[str, Any]:
    """Full structured report for UI + NG JSON."""
    strat = _strategy_block(strategy, strategy_cfg, texts)
    pixel = diagnose_pixel_match(boxes or [], pixel_cfg or PixelMatchConfig())

    date_block: Dict[str, Any] = {
        "enabled": bool(date_cfg.enable_date_check),
        "params": date_cfg.to_dict(),
        "passed": None,
        "skipped": not bool(date_cfg.enable_date_check),
        "diagnosis": None,
    }
    if date_cfg.enable_date_check:
        date_diag = diagnose_shelf_life_dates(texts, date_cfg)
        date_block["passed"] = bool(date_diag.get("passed"))
        date_block["diagnosis"] = date_diag
        date_block["skipped"] = False

    phrase_pass, phrase_trigger, phrase_combine_mode = _phrase_checks_pass(strat, pixel)
    passed = phrase_pass and (
        not date_cfg.enable_date_check or bool(date_block["passed"])
    )

    ng_trigger: Optional[str] = None
    if not passed:
        if not phrase_pass:
            ng_trigger = phrase_trigger
        elif date_cfg.enable_date_check and not date_block["passed"]:
            ng_trigger = "date_check"

    return {
        "verdict": "OK" if passed else "NG",
        "passed": passed,
        "ng_trigger": ng_trigger,
        "phrase_combine_mode": phrase_combine_mode,
        "phrase_passed": phrase_pass,
        "strategy": strat,
        "pixel_match": pixel,
        "date_check": date_block,
    }


def format_check_report_for_ui(report: Dict[str, Any]) -> str:
    """Human-readable block for the OCR result text panel."""
    _pass = "\u901a\u8fc7"
    _fail = "\u672a\u901a\u8fc7"
    _yes = "\u662f"
    _no = "\u5426"
    lines: List[str] = []
    lines.append("=" * 32)
    lines.append(f"\u6821\u9a8c\u7ed3\u8bba: {report.get('verdict', '?')}")
    combine_mode = str(report.get("phrase_combine_mode") or "")
    if combine_mode == "or":
        lines.append(
            "  \u77ed\u8bed\u7ea7\u5224\u5b9a: CJK \u4e0e\u50cf\u7d20\u5339\u914d\u4e3a OR"
            "（任一通过即可）"
        )
    lines.append("")

    strat = report.get("strategy") or {}
    lines.append("[\u4ea7\u7ebf\u4e09\u8bed\u7b56\u7565 / CJK]")
    lines.append(f"  \u542f\u7528: {_yes if strat.get('enabled') else _no}")
    lines.append(f"  \u7b56\u7565: {strat.get('class_name')}")
    params = strat.get("params") or {}
    if "max_cjk_length_diff" in params:
        lines.append(
            f"  \u53c2\u6570: max_cjk_length_diff={params.get('max_cjk_length_diff')}, "
            f"min_match_percentage_limit={params.get('min_match_percentage_limit')}"
        )
    phrases = params.get("required_phrases") or []
    if phrases:
        lines.append(f"  \u76ee\u6807\u77ed\u8bed: {' | '.join(phrases)}")
    if strat.get("enabled"):
        lines.append(f"  \u6587\u672c\u6846\u6570: {strat.get('text_box_count')}")
        lines.append(
            f"  \u7b56\u7565\u7ed3\u679c: {_pass if strat.get('passed') else _fail}"
        )
        sf = (strat.get("diagnosis") or {}).get("failure")
        if sf:
            lines.append(
                f"  \u5931\u8d25\u539f\u56e0: {sf.get('message', sf.get('code', ''))}"
            )
            detail = sf.get("detail")
            if isinstance(detail, list):
                for item in detail[:6]:
                    if isinstance(item, dict):
                        lines.append(f"    - {item.get('summary') or item}")
                    else:
                        lines.append(f"    - {item}")
    else:
        lines.append("  \uff08\u672a\u542f\u7528\uff0c\u8df3\u8fc7\uff09")

    pixel_blk = report.get("pixel_match") or {}
    lines.append("")
    lines.append("[\u50cf\u7d20\u5339\u914d]")
    lines.append(f"  \u542f\u7528: {_yes if pixel_blk.get('enabled') else _no}")
    if pixel_blk.get("enabled"):
        pp = pixel_blk.get("params") or {}
        lines.append(
            f"  \u53c2\u6570: min_window_count={pp.get('min_window_count')}, "
            f"min_pixel_width={pp.get('min_pixel_width')}, "
            f"min_pixel_length={pp.get('min_pixel_length')}"
        )
        lines.append(f"  \u6587\u672c\u6846\u6570: {pixel_blk.get('box_count')}")
        lines.append(
            f"  \u50cf\u7d20\u5339\u914d\u7ed3\u679c: "
            f"{_pass if pixel_blk.get('passed') else _fail}"
        )
        pf = pixel_blk.get("failure")
        if pf:
            lines.append(
                f"  \u5931\u8d25\u539f\u56e0: {pf.get('message', pf.get('code', ''))}"
            )
            detail = pf.get("detail") or []
            for item in detail[:6]:
                if isinstance(item, dict):
                    lines.append(
                        f"    - #{item.get('index')} "
                        f"\u957f={item.get('horizontal_length_px')} "
                        f"\u5bbd={item.get('vertical_width_px')} "
                        f"{item.get('text', '')[:24]}"
                    )
    else:
        lines.append("  \uff08\u672a\u542f\u7528\uff0c\u8df3\u8fc7\uff09")

    date_blk = report.get("date_check") or {}
    lines.append("")
    lines.append("[\u65e5\u671f\u68c0\u6d4b]")
    lines.append(f"  \u542f\u7528: {_yes if date_blk.get('enabled') else _no}")
    if date_blk.get("enabled"):
        dp = date_blk.get("params") or {}
        lines.append(
            f"  \u53c2\u6570: shelf_life_normal={dp.get('shelf_life_normal')} "
            f"\u5929, shelf_life_frozen={dp.get('shelf_life_frozen')} \u5929"
        )
        diag = date_blk.get("diagnosis") or {}
        expected = diag.get("expected_dates") or {}
        if expected:
            lines.append(
                f"  \u671f\u671b\u65e5\u671f: "
                f"\u751f\u4ea7={expected.get('production')} "
                f"\u5e38\u6e29={expected.get('normal_expiry')} "
                f"\u51b7\u51bb={expected.get('frozen_expiry')}"
            )
        parsed = diag.get("parsed_lines") or []
        if parsed:
            lines.append("  OCR \u89e3\u6790\u65e5\u671f:")
            for row in parsed[:12]:
                lines.append(
                    f"    [{row.get('index')}] {row.get('date')}  "
                    f"{row.get('text', '')[:48]}"
                )
        lines.append(
            f"  \u65e5\u671f\u7ed3\u679c: {_pass if date_blk.get('passed') else _fail}"
        )
        df = diag.get("failure")
        if df:
            lines.append(
                f"  \u5931\u8d25\u539f\u56e0: {df.get('message', df.get('code', ''))}"
            )
        extra = diag.get("failures") or []
        if len(extra) > 1:
            for item in extra[1:]:
                lines.append(f"    - {item.get('message', item.get('code'))}")
    else:
        lines.append("  \uff08\u672a\u542f\u7528\uff0c\u8df3\u8fc7\uff09")

    trigger = report.get("ng_trigger")
    if trigger:
        label_map = {
            "strategy": "CJK 匹配",
            "pixel_match": "像素匹配",
            "cjk_and_pixel": "CJK 与像素匹配均未通过",
            "date_check": "日期检测",
        }
        label = label_map.get(str(trigger), str(trigger))
        lines.append("")
        lines.append(f"NG \u89e6\u53d1\u9636\u6bb5: {label}")

    lines.append("=" * 32)
    return "\n".join(lines)
