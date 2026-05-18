# -*- coding: utf-8 -*-
"""Tests for :mod:`production_phrase_strategy`."""

import io
import unittest
from contextlib import redirect_stdout
from datetime import date
from unittest.mock import patch

from production_phrase_strategy import (
    ColonCjkPhraseMatchStrategy,
    StrictExclusiveSubstringStrategy,
)


def _three_ok_lines_2026() -> list[str]:
    return [
        "\u751f\u4ea7\u65e5\u671f\uff1a2026/04/2915:54",
        "\u5e38\u6e29\u50a8\u5b58\u4fdd\u8d28\u671f\u81f3\uff1a2026/05/06",
        "\u51b7\u51bb\u9500\u5b58\u4fdd\u8d28\u671f\u81f3\uff1a2026/09/5",
    ]

def _three_label_only_lines() -> list[str]:
    """Labels + colon only (no year digits); year filter drops all three when year is 2026."""
    return [
        "\u751f\u4ea7\u65e5\u671f\uff1a",
        "\u5e38\u6e29\u50a8\u5b58\u4fdd\u8d28\u671f\u81f3\uff1a",
        "\u51b7\u51bb\u9500\u5b58\u4fdd\u8d28\u671f\u81f3\uff1a",
    ]


class TestStrictExclusiveSubstringStrategy(unittest.TestCase):
    def test_three_realistic_lines_match(self) -> None:
        texts = _three_ok_lines_2026()
        self.assertTrue(StrictExclusiveSubstringStrategy().match(texts))

    def test_strict_year_suffix_false_when_no_date_after_colon(self) -> None:
        texts = _three_label_only_lines()
        self.assertFalse(StrictExclusiveSubstringStrategy(strict_year_suffix=True).match(texts))


@patch("production_phrase_strategy.date")
class TestStrictYearSuffixMode(unittest.TestCase):
    """``strict_year_suffix=True``: each kept line must contain ``str(today.year)``."""

    # ``datetime.date`` is a C type: cannot patch ``date.today`` directly; replace module ``date``.
    @staticmethod
    def _fixed_today_year_2026(mock_date) -> None:
        mock_date.today.return_value = date(2026, 1, 1)

    def test_strict_year_suffix_all_current_year_still_matches(self, mock_date) -> None:
        self._fixed_today_year_2026(mock_date)
        s = StrictExclusiveSubstringStrategy(strict_year_suffix=True)
        self.assertTrue(s.match(_three_ok_lines_2026()))

    def test_strict_year_suffix_accepts_ascii_colon(self, mock_date) -> None:
        self._fixed_today_year_2026(mock_date)
        texts = [
            "\u751f\u4ea7\u65e5\u671f:2026/04/29",
            "\u5e38\u6e29\u50a8\u5b58\u4fdd\u8d28\u671f\u81f3\uff1a2026/05/06",
            "\u51b7\u51bb\u9500\u5b58\u4fdd\u8d28\u671f\u81f3\uff1a2026/09/5",
        ]
        self.assertTrue(StrictExclusiveSubstringStrategy(strict_year_suffix=True).match(texts))

    def test_strict_year_suffix_false_when_too_few_lines_after_filter(self, mock_date) -> None:
        self._fixed_today_year_2026(mock_date)
        base = _three_ok_lines_2026()
        texts = [base[0].replace("2026", "2025"), base[1], base[2]]
        s = StrictExclusiveSubstringStrategy(strict_year_suffix=True)
        self.assertFalse(s.match(texts))

    def test_strict_year_suffix_prints_once_when_filtered(self, mock_date) -> None:
        self._fixed_today_year_2026(mock_date)
        texts = _three_ok_lines_2026() + [
            "\u5e38\u6e29\u50a8\u5b58\u4fdd\u8d28\u671f\u81f3\uff1a2025/05/06",
        ]
        buf = io.StringIO()
        s = StrictExclusiveSubstringStrategy(strict_year_suffix=True)
        with redirect_stdout(buf):
            self.assertTrue(s.match(texts))
        out = buf.getvalue()
        self.assertIn("filtered 1 of 4", out)
        self.assertIn("strict_year_suffix=True", out)


class TestColonCjkPhraseMatchStrategy(unittest.TestCase):
    def test_ok_lines_loose_thresholds(self) -> None:
        s = ColonCjkPhraseMatchStrategy(
            max_cjk_length_diff=2,
            min_cjk_lcs_matches=3,
        )
        self.assertTrue(s.match(_three_ok_lines_2026()))

    def test_impossible_when_min_lcs_too_high(self) -> None:
        s = ColonCjkPhraseMatchStrategy(min_cjk_lcs_matches=50)
        self.assertFalse(s.match(_three_ok_lines_2026()))

    @patch("production_phrase_strategy.date")
    def test_exclude_without_year_like_strict(self, mock_date) -> None:
        mock_date.today.return_value = date(2026, 1, 1)
        s = ColonCjkPhraseMatchStrategy(
            exclude_lines_without_year=True,
            max_cjk_length_diff=2,
            min_cjk_lcs_matches=3,
        )
        self.assertTrue(s.match(_three_ok_lines_2026()))
        base = _three_ok_lines_2026()
        self.assertFalse(
            s.match([base[0].replace("2026", "2025"), base[1], base[2]])
        )


if __name__ == "__main__":
    unittest.main()
