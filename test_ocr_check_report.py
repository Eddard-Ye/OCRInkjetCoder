# -*- coding: utf-8 -*-
"""Tests for :mod:`ocr_check_report`."""

import unittest
from datetime import date
from unittest.mock import patch

from date_check_config import DateCheckGlobalConfig
from ocr_check_report import build_ocr_check_report, format_check_report_for_ui
from production_phrase_strategy import ColonCjkPhraseMatchStrategy


class TestOcrCheckReport(unittest.TestCase):
    @patch("date_check_config._system_today")
    def test_ng_strategy_and_date_sections(self, mock_today) -> None:
        mock_today.return_value = date(2026, 5, 19)
        texts = [
            "\u751fO\u65e5\u671f: 2026/5/25",
            "\u5e38\u6e29\u50a8\u5b58\u4fdd\u8d28\u671f\u81f3: 2026/5/30",
            "\u51b7\u51bb\u50a8\u5b58\u4fdd\u8d28\u671f\u81f3:2026/11/21",
        ]
        report = build_ocr_check_report(
            texts,
            combo_name="ColonCjkPhraseMatchStrategy",
            strategy=ColonCjkPhraseMatchStrategy(
                max_cjk_length_diff=1, min_match_percentage_limit=0.0
            ),
            strategy_cfg={"max_cjk_length_diff": 1, "min_match_percentage_limit": 0.0},
            date_cfg=DateCheckGlobalConfig(
                enable_date_check=True, shelf_life_normal=5, shelf_life_frozen=180
            ),
        )
        self.assertEqual(report["verdict"], "NG")
        self.assertTrue(report["strategy"]["passed"])
        self.assertFalse(report["date_check"]["passed"])
        self.assertEqual(report["ng_trigger"], "date_check")
        ui = format_check_report_for_ui(report)
        self.assertIn("max_cjk_length_diff", ui)
        self.assertIn("shelf_life_normal", ui)


if __name__ == "__main__":
    unittest.main()
