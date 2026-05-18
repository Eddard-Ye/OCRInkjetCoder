# -*- coding: utf-8 -*-
"""
Production / expiry phrase strategies (same three strings as ``hik_camera_ui``).

- Strategy 1: strict exclusive substring match (optional year-suffix filter).
- Strategy 2: colon-prefix CJK length + LCS threshold vs three phrases (injective).
"""

from __future__ import annotations

import sys
from datetime import date
from itertools import combinations, permutations
from typing import Mapping, Sequence, Union

# Same three strings as hik_camera_ui.check_required_production_expiry_boxes.
DEFAULT_REQUIRED_PHRASES: tuple[str, ...] = (
    "生产日期",
    "常温储存保质期至",
    "冷冻储存保质期至"
)


def _normalize_strings(data: Union[Sequence[str], Mapping[str, str]]) -> list[str]:
    if isinstance(data, Mapping):
        return [str(v) for v in data.values()]
    return [str(x) for x in data]


def _split_once_on_colon(s: str) -> tuple[str, str] | None:
    r"""Split on the first ASCII ':' or fullwidth colon (U+FF1A). Return None if absent."""
    idxs = [i for i in (s.find(":"), s.find("\uff1a")) if i >= 0]
    if not idxs:
        return None
    i = min(idxs)
    return s[:i].strip(), s[i + 1 :].strip()


def _extract_cjk_unified(s: str) -> str:
    """Keep CJK Unified + Ext. A code points (typical printed Chinese)."""
    out: list[str] = []
    for ch in s:
        o = ord(ch)
        if 0x3400 <= o <= 0x4DBF or 0x4E00 <= o <= 0x9FFF:
            out.append(ch)
    return "".join(out)


def _candidate_cjk_before_colon(line: str) -> str:
    """
    If a colon is present, CJK are taken only from the segment before the first
    ASCII ``:`` or fullwidth U+FF1A. Otherwise CJK are taken from the whole line.
    """
    parts = _split_once_on_colon(line)
    if parts is not None:
        left, _right = parts
        return _extract_cjk_unified(left)
    return _extract_cjk_unified(line)


def _lcs_length(a: str, b: str) -> int:
    """Longest common subsequence length (ordered exact character matches)."""
    m, n = len(a), len(b)
    if m == 0 or n == 0:
        return 0
    prev = [0] * (n + 1)
    cur = [0] * (n + 1)
    for i in range(1, m + 1):
        ai = a[i - 1]
        for j in range(1, n + 1):
            if ai == b[j - 1]:
                cur[j] = prev[j - 1] + 1
            else:
                cur[j] = max(prev[j], cur[j - 1])
        prev, cur = cur, prev
    return prev[n]


def _line_contains_calendar_year(s: str, year: int) -> bool:
    """True iff ``str(year)`` occurs as a substring anywhere in ``s``."""
    return str(year) in s


class StrictExclusiveSubstringStrategy:
    """
    Strategy 1: three fixed phrases vs a list of strings (or dict values).

    - Substring match per phrase (same rule as the original UI check).
    - Each string entry may be assigned to at most one phrase; phrases must use
      three distinct entries when all three match.
    - Optional ``strict_year_suffix``: keep only strings that contain the current
      calendar year as a decimal substring (``str(date.today().year) in line``).
      Filtered rows are reported once on stdout.
    """

    __slots__ = ("_phrases", "_strict_year_suffix")

    def __init__(
        self,
        phrases: tuple[str, ...] | None = None,
        *,
        strict_year_suffix: bool = False,
    ) -> None:
        self._phrases: tuple[str, ...] = (
            tuple(phrases) if phrases is not None else DEFAULT_REQUIRED_PHRASES
        )
        if len(self._phrases) != 3:
            raise ValueError("StrictExclusiveSubstringStrategy expects exactly 3 phrases")
        self._strict_year_suffix: bool = bool(strict_year_suffix)

    @property
    def phrases(self) -> tuple[str, ...]:
        return self._phrases

    def match(self, strings: Union[Sequence[str], Mapping[str, str]]) -> bool:
        """
        Return True iff there is an injective assignment: each phrase p is a substring
        of a distinct string among ``strings``.
        """
        texts = _normalize_strings(strings)
        if self._strict_year_suffix:
            year = date.today().year
            before = len(texts)
            texts = [t for t in texts if _line_contains_calendar_year(t, year)]
            if len(texts) < before:
                print(
                    "StrictExclusiveSubstringStrategy(strict_year_suffix=True): "
                    f"filtered {before - len(texts)} of {before} string(s) "
                    f"(no substring of calendar year {year} in line).",
                    file=sys.stdout,
                    flush=True,
                )
        phrases = self._phrases
        k = len(phrases)
        n = len(texts)
        if n < k:
            return False

        for idxs in combinations(range(n), k):
            chosen = [texts[i] for i in idxs]
            for perm in permutations(range(k)):
                if all(phrases[p] in chosen[perm[p]] for p in range(k)):
                    return True
        return False

    def __call__(self, strings: Union[Sequence[str], Mapping[str, str]]) -> bool:
        return self.match(strings)


class ColonCjkPhraseMatchStrategy:
    """
    Strategy 2: same three phrases; assign each phrase to a distinct string candidate.

    - Optional ``exclude_lines_without_year``: same rule as
      ``StrictExclusiveSubstringStrategy(strict_year_suffix=True)``: the line must
      contain ``str(date.today().year)`` as a substring. One stdout line when any
      row is dropped.
    - If a colon (ASCII ``:`` or fullwidth U+FF1A) exists, compare only CJK taken from
      the substring *before* that colon; otherwise CJK are taken from the full line.
    - Let ``L_p`` / ``L_c`` be CJK lengths for phrase vs candidate region. Require
      ``abs(L_c - L_p) <= max_cjk_length_diff``.
    - Require at least ``min_cjk_lcs_matches`` CJK that match in order between
      candidate CJK and phrase CJK, implemented as LCS length on those two strings.
    """

    __slots__ = (
        "_phrases",
        "_exclude_lines_without_year",
        "_max_cjk_length_diff",
        "_min_cjk_lcs_matches",
    )

    def __init__(
        self,
        phrases: tuple[str, ...] | None = None,
        *,
        exclude_lines_without_year: bool = False,
        max_cjk_length_diff: int = 2,
        min_cjk_lcs_matches: int = 3,
    ) -> None:
        self._phrases: tuple[str, ...] = (
            tuple(phrases) if phrases is not None else DEFAULT_REQUIRED_PHRASES
        )
        if len(self._phrases) != 3:
            raise ValueError("ColonCjkPhraseMatchStrategy expects exactly 3 phrases")
        self._exclude_lines_without_year: bool = bool(exclude_lines_without_year)
        self._max_cjk_length_diff: int = int(max_cjk_length_diff)
        self._min_cjk_lcs_matches: int = int(min_cjk_lcs_matches)
        if self._max_cjk_length_diff < 0 or self._min_cjk_lcs_matches < 0:
            raise ValueError("max_cjk_length_diff and min_cjk_lcs_matches must be >= 0")

    @property
    def phrases(self) -> tuple[str, ...]:
        return self._phrases

    def _phrase_cjk(self) -> tuple[str, str, str]:
        a, b, c = self._phrases
        return (
            _extract_cjk_unified(a),
            _extract_cjk_unified(b),
            _extract_cjk_unified(c),
        )

    def _pair_ok(self, phrase_cjk: str, line: str) -> bool:
        cand = _candidate_cjk_before_colon(line)
        if abs(len(cand) - len(phrase_cjk)) > self._max_cjk_length_diff:
            return False
        if _lcs_length(cand, phrase_cjk) < self._min_cjk_lcs_matches:
            return False
        return True

    def match(self, strings: Union[Sequence[str], Mapping[str, str]]) -> bool:
        texts = _normalize_strings(strings)
        if self._exclude_lines_without_year:
            year = date.today().year
            before = len(texts)
            texts = [t for t in texts if _line_contains_calendar_year(t, year)]
            if len(texts) < before:
                print(
                    "ColonCjkPhraseMatchStrategy(exclude_lines_without_year=True): "
                    f"filtered {before - len(texts)} of {before} string(s) "
                    f"(no substring of calendar year {year} in line).",
                    file=sys.stdout,
                    flush=True,
                )
        phrases = self._phrases
        phrase_cjk = self._phrase_cjk()
        k = len(phrases)
        n = len(texts)
        if n < k:
            return False

        for idxs in combinations(range(n), k):
            chosen = [texts[i] for i in idxs]
            for perm in permutations(range(k)):
                if all(
                    self._pair_ok(phrase_cjk[p], chosen[perm[p]]) for p in range(k)
                ):
                    return True
        return False

    def __call__(self, strings: Union[Sequence[str], Mapping[str, str]]) -> bool:
        return self.match(strings)
