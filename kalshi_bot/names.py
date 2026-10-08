"""Player-name matching and Kalshi ticker date parsing.

Vendored from the private model repo so the bot has no dependency on it.
"""

from __future__ import annotations

import re

import pandas as pd

_DATE_RE = re.compile(r"-(\d{2})([A-Z]{3})(\d{2})")
_MONTHS = {m: i for i, m in enumerate(
    ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"], 1)}


def parse_event_date(event_ticker: str) -> pd.Timestamp | None:
    """Match date from a Kalshi event ticker, e.g. KXATPMATCH-26SEP23DUCSON -> 2026-09-23."""
    m = _DATE_RE.search(str(event_ticker or ""))
    if not m or m.group(2) not in _MONTHS:
        return None
    return pd.Timestamp(year=2000 + int(m.group(1)), month=_MONTHS[m.group(2)], day=int(m.group(3)))


def _normalize_for_matching(name: str) -> str:
    """Normalize a player name for fuzzy matching."""
    if not name:
        return ""
    n = name.lower().strip()
    # Remove suffixes
    for suffix in [" jr.", " jr", " sr.", " sr", " ii", " iii", " iv"]:
        if n.endswith(suffix):
            n = n[: -len(suffix)].strip()
    # Normalize apostrophes, hyphens, accents
    n = n.replace("'", "").replace("'", "").replace("-", " ")
    # Collapse multiple spaces
    return " ".join(n.split())


def match_player_name(odds_name: str, player_names: list[str]) -> str | None:
    """
    Match a Kalshi player name to one of ``player_names``.

    Exact match first, then normalized, reversed name order, and finally
    last-name matching with apostrophe/suffix handling. Returns None if no
    candidate matches.
    """
    if not odds_name:
        return None

    # Exact match
    if odds_name in player_names:
        return odds_name

    odds_norm = _normalize_for_matching(odds_name)
    odds_parts = odds_norm.split()
    odds_last = odds_parts[-1] if odds_parts else ""

    # Normalized exact match
    for name in player_names:
        if _normalize_for_matching(name) == odds_norm:
            return name

    # Reversed name match (handles Chinese/East Asian name order)
    # e.g. "Wu Yibing" vs "Yibing Wu"
    if len(odds_parts) == 2:
        odds_reversed = f"{odds_parts[1]} {odds_parts[0]}"
        for name in player_names:
            if _normalize_for_matching(name) == odds_reversed:
                return name

    # Last name match (normalized) — collect all candidates, disambiguate
    odds_first = odds_parts[0] if odds_parts else ""
    last_name_candidates = []
    for name in player_names:
        name_norm = _normalize_for_matching(name)
        name_parts = name_norm.split()
        name_last = name_parts[-1] if name_parts else ""
        if name_last == odds_last:
            last_name_candidates.append(name)

    if len(last_name_candidates) == 1:
        return last_name_candidates[0]
    elif len(last_name_candidates) > 1:
        # Disambiguate by first name similarity
        best = _pick_best_name_match(odds_first, last_name_candidates)
        if best:
            return best

    # First 3 chars of last name (handles transliteration) — require first name match
    if len(odds_last) >= 3:
        prefix_candidates = []
        for name in player_names:
            name_last = _normalize_for_matching(name).split()[-1]
            if len(name_last) >= 3 and name_last[:3] == odds_last[:3]:
                prefix_candidates.append(name)

        if prefix_candidates:
            best = _pick_best_name_match(odds_first, prefix_candidates)
            if best:
                return best

    return None


def _pick_best_name_match(target_first: str, candidates: list[str]) -> str | None:
    """
    Given a target first name and multiple candidates, pick the best match.
    Uses first-name prefix matching, then falls back to longest common prefix.
    """
    if not target_first or not candidates:
        return candidates[0] if candidates else None

    target_first = target_first.lower()

    # Exact first name match
    for name in candidates:
        name_first = _normalize_for_matching(name).split()[0] if name else ""
        if name_first == target_first:
            return name

    # First name starts with same prefix (>= 3 chars)
    if len(target_first) >= 3:
        for name in candidates:
            name_first = _normalize_for_matching(name).split()[0] if name else ""
            if name_first.startswith(target_first[:3]) or target_first.startswith(name_first[:3]):
                return name

    # No first-name signal — return None to avoid a wrong match
    return None
