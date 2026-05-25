# -*- coding: utf-8 -*-
"""Tests for :mod:`production_phrase_strategy`."""

import unittest

from production_phrase_strategy import (
    ColonCjkPhraseMatchStrategy,
    StrictExclusiveSubstringStrategy,
)


def _three_ok_lines_2026() -> list[str]:
    return [
        "\u751f\u4ea7\u65e5\u671f\uff1a2026/04/2915:54",
        "\u5e38\u6e29\u50a8\u5b58\u4fdd\u8d28\u671f\u81f3\uff1a2026/05/06",
        "\u51b7\u51bb\u50a8\u5b58\u4fdd\u8d28\u671f\u81f3\uff1a2026/09/5",
    ]


def _three_label_only_lines() -> list[str]:
    return [
        "\u751f\u4ea7\u65e5\u671f\uff1a",
        "\u5e38\u6e29\u50a8\u5b58\u4fdd\u8d28\u671f\u81f3\uff1a",
        "\u51b7\u51bb\u50a8\u5b58\u4fdd\u8d28\u671f\u81f3\uff1a",
    ]


class TestStrictExclusiveSubstringStrategy(unittest.TestCase):
    def test_three_realistic_lines_match(self) -> None:
        texts = _three_ok_lines_2026()
        self.assertTrue(StrictExclusiveSubstringStrategy().match(texts))

    def test_label_only_lines_still_match(self) -> None:
        texts = _three_label_only_lines()
        self.assertTrue(StrictExclusiveSubstringStrategy().match(texts))


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


if __name__ == "__main__":
    unittest.main()
