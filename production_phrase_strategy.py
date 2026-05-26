# -*- coding: utf-8 -*-
"""
Production / expiry phrase strategy (same three strings as ``hik_camera_ui``).

Colon-prefix CJK length + LCS match-percentage threshold vs three phrases (injective).

Year / shelf-life date rules live in ``date_check_config`` (not in this strategy).
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


def _cjk_match_percentage(cand: str, phrase_cjk: str) -> float:
    """Ordered match ratio: LCS(cand, phrase) / len(phrase), in [0, 1]."""
    if not phrase_cjk:
        return 1.0
    return _lcs_length(cand, phrase_cjk) / len(phrase_cjk)


class ColonCjkPhraseMatchStrategy:
    """
    Same three phrases; assign each phrase to a distinct string candidate.

    - If a colon (ASCII ``:`` or fullwidth U+FF1A) exists, compare only CJK taken from
      the substring *before* that colon; otherwise CJK are taken from the full line.
    - Let ``L_p`` / ``L_c`` be CJK lengths for phrase vs candidate region. Require
      ``abs(L_c - L_p) <= max_cjk_length_diff``.
    - Require ``LCS(cand, phrase) / len(phrase) >= min_match_percentage_limit`` (0~1).
    """

    __slots__ = (
        "_phrases",
        "_max_cjk_length_diff",
        "_min_match_percentage_limit",
    )

    def __init__(
        self,
        phrases: tuple[str, ...] | None = None,
        *,
        max_cjk_length_diff: int = 2,
        min_match_percentage_limit: float = 0.75,
    ) -> None:
        self._phrases: tuple[str, ...] = (
            tuple(phrases) if phrases is not None else DEFAULT_REQUIRED_PHRASES
        )
        if len(self._phrases) != 3:
            raise ValueError("ColonCjkPhraseMatchStrategy expects exactly 3 phrases")
        self._max_cjk_length_diff: int = int(max_cjk_length_diff)
        self._min_match_percentage_limit: float = float(min_match_percentage_limit)
        if self._max_cjk_length_diff < 0:
            raise ValueError("max_cjk_length_diff must be >= 0")
        if not (0.0 <= self._min_match_percentage_limit <= 1.0):
            raise ValueError("min_match_percentage_limit must be in [0, 1]")

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
        match_pct = _cjk_match_percentage(cand, phrase_cjk)
        length_ok = length_diff <= self._max_cjk_length_diff
        pct_ok = match_pct >= self._min_match_percentage_limit
        failed_on: Optional[str] = None
        if not length_ok:
            failed_on = "max_cjk_length_diff"
        elif not pct_ok:
            failed_on = "min_match_percentage_limit"
        limit = self._min_match_percentage_limit
        return {
            "phrase": phrase,
            "line": line,
            "candidate_cjk": cand,
            "phrase_cjk": phrase_cjk,
            "length_diff": length_diff,
            "max_cjk_length_diff": self._max_cjk_length_diff,
            "lcs": lcs,
            "match_percentage": round(match_pct, 4),
            "min_match_percentage_limit": limit,
            "passed": length_ok and pct_ok,
            "failed_on": failed_on,
            "summary": (
                (f"「{phrase}」vs「{line[:40]}…」" if len(line) > 40 else f"「{phrase}」vs「{line}」")
                + f" | CJK {cand!r}/{len(cand)} vs {phrase_cjk!r}/{len(phrase_cjk)}"
                + f" | Δlen={length_diff}(≤{self._max_cjk_length_diff})"
                + f" match%={match_pct:.2f}(≥{limit:.2f})"
                + (f" → 失败于 {failed_on}" if failed_on else " → OK")
            ),
        }

    def _pair_ok(self, phrase_cjk: str, line: str) -> bool:
        cand = _candidate_cjk_before_colon(line)
        if abs(len(cand) - len(phrase_cjk)) > self._max_cjk_length_diff:
            return False
        if _cjk_match_percentage(cand, phrase_cjk) < self._min_match_percentage_limit:
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
                "min_match_percentage_limit": self._min_match_percentage_limit,
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
                    -float(d.get("match_percentage", 0)),
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
