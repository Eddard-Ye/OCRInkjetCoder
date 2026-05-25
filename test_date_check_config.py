# -*- coding: utf-8 -*-
"""Tests for :mod:`date_check_config`."""

import unittest
from datetime import date, timedelta
from unittest.mock import patch

from date_check_config import DateCheckGlobalConfig, validate_shelf_life_dates


def _ok_lines_for(prod: date) -> list[str]:
    dn = prod + timedelta(days=5)
    df = prod + timedelta(days=180)
    return [
        f"\u751f\u4ea7\u65e5\u671f: {prod.year}/{prod.month}/{prod.day}",
        f"\u5e38\u6e29\u50a8\u5b58\u4fdd\u8d28\u671f\u81f3: {dn.year}/{dn.month}/{dn.day}",
        f"\u51b7\u51bb\u50a8\u5b58\u4fdd\u8d28\u671f\u81f3:{df.year}/{df.month}/{df.day}",
    ]


class TestValidateShelfLifeDates(unittest.TestCase):
    @patch("date_check_config._system_today")
    def test_passes_three_distinct_dates_today(self, mock_today) -> None:
        mock_today.return_value = date(2026, 5, 19)
        texts = _ok_lines_for(date(2026, 5, 19))
        cfg = DateCheckGlobalConfig(shelf_life_normal=5, shelf_life_frozen=180)
        self.assertTrue(validate_shelf_life_dates(texts, cfg))

    @patch("date_check_config._system_today")
    def test_passes_without_production_keyword(self, mock_today) -> None:
        mock_today.return_value = date(2026, 5, 25)
        prod = date(2026, 5, 25)
        texts = [
            f"\u751fO\u65e5\u671f: {prod.year}/{prod.month}/{prod.day}",
            f"\u5e38\u6e29\u50a8\u5b58\u4fdd\u8d28\u671f\u81f3: 2026/5/30",
            f"\u51b7\u51bb\u50a8\u5b58\u4fdd\u8d28\u671f\u81f3:2026/11/21",
        ]
        cfg = DateCheckGlobalConfig(shelf_life_normal=5, shelf_life_frozen=180)
        self.assertTrue(validate_shelf_life_dates(texts, cfg))

    @patch("date_check_config._system_today")
    def test_fails_when_production_not_today(self, mock_today) -> None:
        mock_today.return_value = date(2026, 5, 19)
        texts = _ok_lines_for(date(2026, 5, 25))
        cfg = DateCheckGlobalConfig(shelf_life_normal=5, shelf_life_frozen=180)
        self.assertFalse(validate_shelf_life_dates(texts, cfg))

    @patch("date_check_config._system_today")
    def test_fails_when_only_two_date_lines(self, mock_today) -> None:
        mock_today.return_value = date(2026, 5, 19)
        texts = _ok_lines_for(date(2026, 5, 19))[:2]
        cfg = DateCheckGlobalConfig(shelf_life_normal=5, shelf_life_frozen=180)
        self.assertFalse(validate_shelf_life_dates(texts, cfg))

    @patch("date_check_config._system_today")
    def test_fails_wrong_shelf_dates(self, mock_today) -> None:
        mock_today.return_value = date(2026, 5, 19)
        texts = [
            "\u751f\u4ea7\u65e5\u671f: 2026/5/19",
            "\u5e38\u6e29\u50a8\u5b58\u4fdd\u8d28\u671f\u81f3: 2026/5/29",
            "\u51b7\u51bb\u50a8\u5b58\u4fdd\u8d28\u671f\u81f3:2026/11/21",
        ]
        cfg = DateCheckGlobalConfig(shelf_life_normal=5, shelf_life_frozen=180)
        self.assertFalse(validate_shelf_life_dates(texts, cfg))


if __name__ == "__main__":
    unittest.main()
