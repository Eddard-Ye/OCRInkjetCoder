# -*- coding: utf-8 -*-
"""Core tests: CJK OR pixel match (never AND) + pixel dimension rules."""

import unittest

from colon_cjk_strategy_config import ColonCjkStrategyConfig
from date_check_config import DateCheckGlobalConfig
from ocr_check_report import (
    _phrase_checks_pass,
    build_ocr_check_report,
    format_check_report_for_ui,
)
from pixel_match_config import PixelMatchConfig
from pixel_match_strategy import box_screen_width_height, diagnose_pixel_match
from production_phrase_strategy import ColonCjkPhraseMatchStrategy

GOOD_CJK_TEXTS = [
    "\u751f\u4ea7\u65e5\u671f: 2026/06/25",
    "\u5e38\u6e29\u50a8\u5b58\u4fdd\u8d28\u671f\u81f3: 2026/07/01",
    "\u51b7\u51bb\u50a8\u5b58\u4fdd\u8d28\u671f\u81f3: 2026/12/01",
]

LOOSE_STRATEGY_CFG = {
    "enabled": True,
    "max_cjk_length_diff": 5,
    "min_match_percentage_limit": 0.0,
}


def _box(text: str, w: float, h: float) -> dict:
    return {
        "text": text,
        "bbox_xyxy": [0.0, 0.0, float(w), float(h)],
        "poly": [[0, 0], [w, 0], [w, h], [0, h]],
    }


def _good_boxes() -> list[dict]:
    return [_box("line-a", 80, 20), _box("line-b", 90, 25)]


def _bad_boxes() -> list[dict]:
    return [_box("tiny", 10, 5)]


class PhraseChecksPassOrTest(unittest.TestCase):
    """Direct unit tests for OR combiner (core contract)."""

    def test_or_both_enabled_cjk_pass_pixel_fail(self) -> None:
        ok, trigger, mode = _phrase_checks_pass(
            {"enabled": True, "passed": True},
            {"enabled": True, "passed": False},
        )
        self.assertTrue(ok)
        self.assertIsNone(trigger)
        self.assertEqual(mode, "or")

    def test_or_both_enabled_cjk_fail_pixel_pass(self) -> None:
        ok, trigger, mode = _phrase_checks_pass(
            {"enabled": True, "passed": False},
            {"enabled": True, "passed": True},
        )
        self.assertTrue(ok)
        self.assertIsNone(trigger)
        self.assertEqual(mode, "or")

    def test_or_both_enabled_both_fail(self) -> None:
        ok, trigger, mode = _phrase_checks_pass(
            {"enabled": True, "passed": False},
            {"enabled": True, "passed": False},
        )
        self.assertFalse(ok)
        self.assertEqual(trigger, "cjk_and_pixel")
        self.assertEqual(mode, "or")

    def test_or_both_enabled_both_pass(self) -> None:
        ok, trigger, mode = _phrase_checks_pass(
            {"enabled": True, "passed": True},
            {"enabled": True, "passed": True},
        )
        self.assertTrue(ok)
        self.assertIsNone(trigger)
        self.assertEqual(mode, "or")

    def test_cjk_only(self) -> None:
        ok, trigger, mode = _phrase_checks_pass(
            {"enabled": True, "passed": True},
            {"enabled": False, "passed": None},
        )
        self.assertTrue(ok)
        self.assertEqual(mode, "cjk_only")

        ok, trigger, mode = _phrase_checks_pass(
            {"enabled": True, "passed": False},
            {"enabled": False, "passed": None},
        )
        self.assertFalse(ok)
        self.assertEqual(trigger, "strategy")

    def test_pixel_only(self) -> None:
        ok, trigger, mode = _phrase_checks_pass(
            {"enabled": False, "passed": None},
            {"enabled": True, "passed": True},
        )
        self.assertTrue(ok)
        self.assertEqual(mode, "pixel_only")

    def test_neither_enabled(self) -> None:
        ok, trigger, mode = _phrase_checks_pass(
            {"enabled": False, "passed": None},
            {"enabled": False, "passed": None},
        )
        self.assertTrue(ok)
        self.assertEqual(mode, "none")


class PixelMatchStrategyTest(unittest.TestCase):
    def test_box_screen_width_height(self) -> None:
        horiz, vert = box_screen_width_height(_box("x", 100, 30))
        self.assertEqual(horiz, 100.0)
        self.assertEqual(vert, 30.0)

    def test_strict_greater_than_min_window_count(self) -> None:
        one_box = [_box("a", 80, 20)]
        cfg = PixelMatchConfig(
            enabled=True,
            min_window_count=1,
            min_pixel_width=10,
            min_pixel_length=50,
        )
        self.assertFalse(diagnose_pixel_match(one_box, cfg)["passed"])

        two_boxes = [_box("a", 80, 20), _box("b", 80, 20)]
        self.assertTrue(diagnose_pixel_match(two_boxes, cfg)["passed"])

    def test_each_box_must_meet_width_and_length(self) -> None:
        boxes = [_box("a", 80, 20), _box("b", 40, 25)]
        cfg = PixelMatchConfig(
            enabled=True,
            min_window_count=0,
            min_pixel_width=10,
            min_pixel_length=50,
        )
        self.assertFalse(diagnose_pixel_match(boxes, cfg)["passed"])


class OcrCheckReportOrIntegrationTest(unittest.TestCase):
    def _report(
        self,
        texts: list[str],
        boxes: list[dict],
        *,
        cjk_enabled: bool = True,
        pixel_enabled: bool = True,
        date_enabled: bool = False,
    ) -> dict:
        return build_ocr_check_report(
            texts,
            strategy=ColonCjkPhraseMatchStrategy(
                max_cjk_length_diff=5,
                min_match_percentage_limit=0.0,
            ),
            strategy_cfg={
                **LOOSE_STRATEGY_CFG,
                "enabled": cjk_enabled,
            },
            date_cfg=DateCheckGlobalConfig(enable_date_check=date_enabled),
            boxes=boxes,
            pixel_cfg=PixelMatchConfig(
                enabled=pixel_enabled,
                min_window_count=1,
                min_pixel_width=10,
                min_pixel_length=50,
            ),
        )

    def test_cjk_pass_pixel_fail_is_ok(self) -> None:
        report = self._report(GOOD_CJK_TEXTS, _bad_boxes())
        self.assertTrue(report["strategy"]["passed"])
        self.assertFalse(report["pixel_match"]["passed"])
        self.assertEqual(report["phrase_combine_mode"], "or")
        self.assertTrue(report["phrase_passed"])
        self.assertEqual(report["verdict"], "OK")

    def test_cjk_fail_pixel_pass_is_ok(self) -> None:
        report = self._report(["bad"], _good_boxes())
        self.assertFalse(report["strategy"]["passed"])
        self.assertTrue(report["pixel_match"]["passed"])
        self.assertEqual(report["verdict"], "OK")

    def test_both_fail_is_ng(self) -> None:
        report = self._report(["bad"], _bad_boxes())
        self.assertEqual(report["verdict"], "NG")
        self.assertEqual(report["ng_trigger"], "cjk_and_pixel")
        self.assertFalse(report["phrase_passed"])

    def test_both_pass_is_ok(self) -> None:
        report = self._report(GOOD_CJK_TEXTS, _good_boxes())
        self.assertEqual(report["verdict"], "OK")

    def test_cjk_only_fail_is_ng(self) -> None:
        report = self._report(["bad"], _good_boxes(), pixel_enabled=False)
        self.assertEqual(report["phrase_combine_mode"], "cjk_only")
        self.assertEqual(report["verdict"], "NG")
        self.assertEqual(report["ng_trigger"], "strategy")

    def test_pixel_only_fail_is_ng(self) -> None:
        report = self._report(GOOD_CJK_TEXTS, _bad_boxes(), cjk_enabled=False)
        self.assertEqual(report["phrase_combine_mode"], "pixel_only")
        self.assertEqual(report["verdict"], "NG")
        self.assertEqual(report["ng_trigger"], "pixel_match")

    def test_ui_shows_or_hint_when_both_enabled(self) -> None:
        report = self._report(GOOD_CJK_TEXTS, _good_boxes())
        ui = format_check_report_for_ui(report)
        self.assertIn("OR", ui)


if __name__ == "__main__":
    unittest.main()
