# -*- coding: utf-8 -*-
"""Structured OK/NG reports for phrase strategy + date check."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Union

from date_check_config import DateCheckGlobalConfig, diagnose_shelf_life_dates
from production_phrase_strategy import (
    ColonCjkPhraseMatchStrategy,
    StrictExclusiveSubstringStrategy,
)


def _strategy_block(
    combo_name: str,
    strategy: Union[StrictExclusiveSubstringStrategy, ColonCjkPhraseMatchStrategy],
    strategy_cfg: Dict[str, Any],
    texts: Sequence[str],
) -> Dict[str, Any]:
    if isinstance(strategy, ColonCjkPhraseMatchStrategy):
        diag = strategy.diagnose(texts)
        params = {
            "required_phrases": list(strategy.phrases),
            "max_cjk_length_diff": int(strategy_cfg.get("max_cjk_length_diff", 2)),
            "min_match_percentage_limit": float(
                strategy_cfg.get("min_match_percentage_limit", 0.75)
            ),
        }
        class_name = "ColonCjkPhraseMatchStrategy"
    else:
        diag = strategy.diagnose(texts)
        params = {"required_phrases": list(strategy.phrases)}
        class_name = "StrictExclusiveSubstringStrategy"

    return {
        "combo_selection": combo_name,
        "class_name": class_name,
        "params": params,
        "passed": bool(diag.get("passed")),
        "text_box_count": diag.get("text_box_count"),
        "diagnosis": diag,
    }


def build_ocr_check_report(
    texts: Sequence[str],
    *,
    combo_name: str,
    strategy: Union[StrictExclusiveSubstringStrategy, ColonCjkPhraseMatchStrategy],
    strategy_cfg: Dict[str, Any],
    date_cfg: DateCheckGlobalConfig,
) -> Dict[str, Any]:
    """Full structured report for UI + NG JSON."""
    strat = _strategy_block(combo_name, strategy, strategy_cfg, texts)

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

    passed = bool(strat["passed"]) and (
        not date_cfg.enable_date_check or bool(date_block["passed"])
    )

    ng_trigger: Optional[str] = None
    if not passed:
        ng_trigger = "strategy" if not strat["passed"] else "date_check"

    return {
        "verdict": "OK" if passed else "NG",
        "passed": passed,
        "ng_trigger": ng_trigger,
        "strategy": strat,
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
    lines.append("")

    strat = report.get("strategy") or {}
    lines.append("[\u4ea7\u7ebf\u4e09\u8bed\u7b56\u7565]")
    lines.append(
        f"  \u7b56\u7565: {strat.get('combo_selection')} ({strat.get('class_name')})"
    )
    params = strat.get("params") or {}
    if "max_cjk_length_diff" in params:
        lines.append(
            f"  \u53c2\u6570: max_cjk_length_diff={params.get('max_cjk_length_diff')}, "
            f"min_match_percentage_limit={params.get('min_match_percentage_limit')}"
        )
    phrases = params.get("required_phrases") or []
    if phrases:
        lines.append(f"  \u76ee\u6807\u77ed\u8bed: {' | '.join(phrases)}")
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
        label = (
            "\u4ea7\u7ebf\u4e09\u8bed\u7b56\u7565"
            if trigger == "strategy"
            else "\u65e5\u671f\u68c0\u6d4b"
        )
        lines.append("")
        lines.append(f"NG \u89e6\u53d1\u9636\u6bb5: {label}")

    lines.append("=" * 32)
    return "\n".join(lines)
