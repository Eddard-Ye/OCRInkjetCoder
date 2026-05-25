# -*- coding: utf-8 -*-
"""Tests for :mod:`production_phrase_strategy`."""

import unittest

from production_phrase_strategy import (
    ColonCjkPhraseMatchStrategy,
    StrictExclusiveSubstringStrategy,
    _cjk_match_percentage,
    _extract_cjk_unified,
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


class TestMatchPercentage(unittest.TestCase):
    def test_half_match_for_two_of_four_chars(self) -> None:
        phrase = _extract_cjk_unified("\u751f\u4ea7\u65e5\u671f")
        for cand_text in (
            "\u751f\u65e5",
            "\u4ea7\u671f",
            "\u65e5\u671f",
            "\u751f\u671f",
        ):
            cand = _extract_cjk_unified(cand_text)
            self.assertAlmostEqual(_cjk_match_percentage(cand, phrase), 0.5)


class TestStrictExclusiveSubstringStrategy(unittest.TestCase):
    def test_three_realistic_lines_match(self) -> None:
        texts = _three_ok_lines_2026()
        self.assertTrue(StrictExclusiveSubstringStrategy().match(texts))

    def test_label_only_lines_still_match(self) -> None:
        texts = _three_label_only_lines()
        self.assertTrue(StrictExclusiveSubstringStrategy().match(texts))


class TestColonCjkPhraseMatchStrategy(unittest.TestCase):
    def test_ok_lines_default_thresholds(self) -> None:
        s = ColonCjkPhraseMatchStrategy(
            max_cjk_length_diff=2,
            min_match_percentage_limit=0.75,
        )
        self.assertTrue(s.match(_three_ok_lines_2026()))

    def test_impossible_when_percentage_too_high(self) -> None:
        s = ColonCjkPhraseMatchStrategy(
            max_cjk_length_diff=0,
            min_match_percentage_limit=0.99,
        )
        texts = [
            "\u751f\u4ea7\u57fa\u672c",
            "\u5357\u6e29\u50a8\u5b58\u4fdd\u8d28\u671f\u81f3",
            "\u51b7\u51bb\u50a8\u5b58\u4fdd\u8d28\u671f\u81f3",
        ]
        self.assertFalse(s.match(texts))

    def test_limit_zero_accepts_partial_production_line(self) -> None:
        s = ColonCjkPhraseMatchStrategy(
            max_cjk_length_diff=0,
            min_match_percentage_limit=0.0,
        )
        texts = [
            "\u751f\u4ea7\u57fa\u672c",
            "\u65e5\u671f:/2026/5/25",
            "\u5357\u6e29\u50a8\u5b58\u4fdd\u8d28\u671f\u81f3:",
            "\u51b7\u51bb\u50a8\u5b58\u4fdd\u8d28\u671f\u81f3",
        ]
        self.assertTrue(s.match(texts))

    def test_limit_half_passes_two_of_four(self) -> None:
        s = ColonCjkPhraseMatchStrategy(
            max_cjk_length_diff=2,
            min_match_percentage_limit=0.5,
        )
        texts = [
            "\u751f\u4ea7\u65e5\u671f",
            "\u5357\u6e29\u50a8\u5b58\u4fdd\u8d28\u671f\u81f3",
            "\u51b7\u51bb\u50a8\u5b58\u4fdd\u8d28\u671f\u81f3",
        ]
        self.assertTrue(s.match(texts))

    def test_limit_above_half_fails_two_of_four(self) -> None:
        s = ColonCjkPhraseMatchStrategy(
            max_cjk_length_diff=0,
            min_match_percentage_limit=0.51,
        )
        texts = [
            "\u751f\u4ea7\u57fa\u672c",
            "\u5357\u6e29\u50a8\u5b58\u4fdd\u8d28\u671f\u81f3",
            "\u51b7\u51bb\u50a8\u5b58\u4fdd\u8d28\u671f\u81f3",
        ]
        self.assertFalse(s.match(texts))

    def test_limit_half_exactly_passes_two_of_four(self) -> None:
        s = ColonCjkPhraseMatchStrategy(
            max_cjk_length_diff=0,
            min_match_percentage_limit=0.5,
        )
        texts = [
            "\u751f\u4ea7\u57fa\u672c",
            "\u5357\u6e29\u50a8\u5b58\u4fdd\u8d28\u671f\u81f3",
            "\u51b7\u51bb\u50a8\u5b58\u4fdd\u8d28\u671f\u81f3",
        ]
        self.assertTrue(s.match(texts))


if __name__ == "__main__":
    unittest.main()
