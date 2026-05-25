# -*- coding: utf-8 -*-
"""
Production / expiry phrase strategies (same three strings as ``hik_camera_ui``).

- Strategy 1: strict exclusive substring match.
- Strategy 2: colon-prefix CJK length + LCS threshold vs three phrases (injective).

Year / shelf-life date rules live in ``date_check_config`` (not in these strategies).
"""

from __future__ import annotations

from itertools import combinations, permutations
from typing import Any, Mapping, Optional, Sequence, Union

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


class StrictExclusiveSubstringStrategy:
    """
    Strategy 1: three fixed phrases vs a list of strings (or dict values).

    - Substring match per phrase (same rule as the original UI check).
    - Each string entry may be assigned to at most one phrase; phrases must use
      three distinct entries when all three match.
    """

    __slots__ = ("_phrases",)

    def __init__(
        self,
        phrases: tuple[str, ...] | None = None,
    ) -> None:
        self._phrases: tuple[str, ...] = (
            tuple(phrases) if phrases is not None else DEFAULT_REQUIRED_PHRASES
        )
        if len(self._phrases) != 3:
            raise ValueError("StrictExclusiveSubstringStrategy expects exactly 3 phrases")

    @property
    def phrases(self) -> tuple[str, ...]:
        return self._phrases

    def match(self, strings: Union[Sequence[str], Mapping[str, str]]) -> bool:
        """
        Return True iff there is an injective assignment: each phrase p is a substring
        of a distinct string among ``strings``.
        """
        texts = _normalize_strings(strings)
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

    def diagnose(self, strings: Union[Sequence[str], Mapping[str, str]]) -> dict[str, Any]:
        texts = _normalize_strings(strings)
        n = len(texts)
        base: dict[str, Any] = {
            "passed": False,
            "text_box_count": n,
            "strategy": "StrictExclusiveSubstringStrategy",
            "failure": None,
        }
        if n < 3:
            base["failure"] = {
                "code": "INSUFFICIENT_TEXT_BOXES",
                "message": f"至少需要 3 个 OCR 文本框，当前 {n} 个",
            }
            return base

        if self.match(strings):
            base["passed"] = True
            return base

        phrase_hits: list[dict[str, Any]] = []
        for phrase in self._phrases:
            idxs = [i for i, t in enumerate(texts) if phrase in t]
            phrase_hits.append(
                {
                    "phrase": phrase,
                    "matching_line_indices": idxs,
                    "summary": (
                        f"短语「{phrase}」: "
                        + (f"命中行 {idxs}" if idxs else "无任何行包含该子串")
                    ),
                }
            )

        missing = [p for p in phrase_hits if not p["matching_line_indices"]]
        if missing:
            detail = [p["summary"] for p in missing]
            base["failure"] = {
                "code": "PHRASE_SUBSTRING_MISSING",
                "message": "部分目标短语在 OCR 文本中找不到子串匹配",
                "detail": detail,
                "phrase_hits": phrase_hits,
            }
            return base

        base["failure"] = {
            "code": "NO_INJECTIVE_ASSIGNMENT",
            "message": "各短语虽单独可匹配，但无法分配到 3 个互不相同的文本框",
            "detail": [p["summary"] for p in phrase_hits],
            "phrase_hits": phrase_hits,
        }
        return base


class ColonCjkPhraseMatchStrategy:
    """
    Strategy 2: same three phrases; assign each phrase to a distinct string candidate.

    - If a colon (ASCII ``:`` or fullwidth U+FF1A) exists, compare only CJK taken from
      the substring *before* that colon; otherwise CJK are taken from the full line.
    - Let ``L_p`` / ``L_c`` be CJK lengths for phrase vs candidate region. Require
      ``abs(L_c - L_p) <= max_cjk_length_diff``.
    - Require at least ``min_cjk_lcs_matches`` CJK that match in order between
      candidate CJK and phrase CJK, implemented as LCS length on those two strings.
    """

    __slots__ = (
        "_phrases",
        "_max_cjk_length_diff",
        "_min_cjk_lcs_matches",
    )

    def __init__(
        self,
        phrases: tuple[str, ...] | None = None,
        *,
        max_cjk_length_diff: int = 2,
        min_cjk_lcs_matches: int = 3,
    ) -> None:
        self._phrases: tuple[str, ...] = (
            tuple(phrases) if phrases is not None else DEFAULT_REQUIRED_PHRASES
        )
        if len(self._phrases) != 3:
            raise ValueError("ColonCjkPhraseMatchStrategy expects exactly 3 phrases")
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

    def _pair_detail(self, phrase: str, phrase_cjk: str, line: str) -> dict[str, Any]:
        cand = _candidate_cjk_before_colon(line)
        length_diff = abs(len(cand) - len(phrase_cjk))
        lcs = _lcs_length(cand, phrase_cjk)
        length_ok = length_diff <= self._max_cjk_length_diff
        lcs_ok = lcs >= self._min_cjk_lcs_matches
        failed_on: Optional[str] = None
        if not length_ok:
            failed_on = "max_cjk_length_diff"
        elif not lcs_ok:
            failed_on = "min_cjk_lcs_matches"
        return {
            "phrase": phrase,
            "line": line,
            "candidate_cjk": cand,
            "phrase_cjk": phrase_cjk,
            "length_diff": length_diff,
            "max_cjk_length_diff": self._max_cjk_length_diff,
            "lcs": lcs,
            "min_cjk_lcs_matches": self._min_cjk_lcs_matches,
            "passed": length_ok and lcs_ok,
            "failed_on": failed_on,
            "summary": (
                (f"「{phrase}」vs「{line[:40]}…」" if len(line) > 40 else f"「{phrase}」vs「{line}」")
                + f" | CJK {cand!r}/{len(cand)} vs {phrase_cjk!r}/{len(phrase_cjk)}"
                + f" | Δlen={length_diff}(≤{self._max_cjk_length_diff})"
                + f" LCS={lcs}(≥{self._min_cjk_lcs_matches})"
                + (f" → 失败于 {failed_on}" if failed_on else " → OK")
            ),
        }

    def _pair_ok(self, phrase_cjk: str, line: str) -> bool:
        cand = _candidate_cjk_before_colon(line)
        if abs(len(cand) - len(phrase_cjk)) > self._max_cjk_length_diff:
            return False
        if _lcs_length(cand, phrase_cjk) < self._min_cjk_lcs_matches:
            return False
        return True

    def match(self, strings: Union[Sequence[str], Mapping[str, str]]) -> bool:
        texts = _normalize_strings(strings)
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

    def diagnose(self, strings: Union[Sequence[str], Mapping[str, str]]) -> dict[str, Any]:
        texts = _normalize_strings(strings)
        n = len(texts)
        phrase_cjk = self._phrase_cjk()
        base: dict[str, Any] = {
            "passed": False,
            "text_box_count": n,
            "strategy": "ColonCjkPhraseMatchStrategy",
            "params": {
                "max_cjk_length_diff": self._max_cjk_length_diff,
                "min_cjk_lcs_matches": self._min_cjk_lcs_matches,
            },
            "failure": None,
        }
        if n < 3:
            base["failure"] = {
                "code": "INSUFFICIENT_TEXT_BOXES",
                "message": f"至少需要 3 个 OCR 文本框，当前 {n} 个",
            }
            return base

        if self.match(strings):
            base["passed"] = True
            return base

        best_by_phrase: list[dict[str, Any]] = []
        for p_idx, phrase in enumerate(self._phrases):
            pc = phrase_cjk[p_idx]
            details = [self._pair_detail(phrase, pc, t) for t in texts]
            details.sort(
                key=lambda d: (
                    0 if d["passed"] else 1,
                    d["length_diff"],
                    -d["lcs"],
                )
            )
            best = details[0] if details else None
            if best is not None:
                best_by_phrase.append(best)

        failed_phrases = [b for b in best_by_phrase if not b["passed"]]
        detail = [b["summary"] for b in failed_phrases] or [
            b["summary"] for b in best_by_phrase
        ]
        primary = failed_phrases[0] if failed_phrases else (best_by_phrase[0] if best_by_phrase else None)
        if primary and primary.get("failed_on"):
            code = f"CJK_{str(primary['failed_on']).upper()}"
            msg = (
                f"短语「{primary['phrase']}」最佳候选仍不满足 "
                f"{primary['failed_on']}"
            )
        else:
            code = "NO_INJECTIVE_ASSIGNMENT"
            msg = "各短语单独可能接近，但无法分配到 3 个互不相同的文本框"

        base["failure"] = {
            "code": code,
            "message": msg,
            "detail": detail,
            "best_pair_per_phrase": best_by_phrase,
        }
        return base
