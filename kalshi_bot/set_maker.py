"""Challenger set-winner maker, DRY RUN ONLY (logs the bids it would rest; places nothing).

Kalshi's set-winner markets (KXATPCHALLENGERSETWINNER, listed from Nov 2026) are priced here from
Kalshi's own match-winner market, which is as sharp as Pinnacle on Challengers (atp-edge
market-structure study, Nov 2026). Fair set probability: logit(set) = 0.80 * logit(match),
calibrated on 5,131 Apr-Nov 2026 best-of-3 matches (set 1 and set 2 alike). A bid is logged when
the best bid, or the best bid + 1c, sits at least SET_MAKER_MARGIN below fair after the maker fee.
Quoting stops SET_MAKER_STOP_HOURS before the match market's occurrence time.
"""

from __future__ import annotations

import math

import pandas as pd

SET_SERIES = "KXATPCHALLENGERSETWINNER"
MATCH_SERIES = "KXATPCHALLENGERMATCH"
SET_SLOPE = 0.80
SET_MAKER_MARGIN = 0.03
SET_MAKER_STOP_HOURS = 6.0
MAX_MATCH_SPREAD = 0.05
MAX_SET_SPREAD = 0.10
MAKER_FEE_RATE = 0.0175

SET_QUOTE_COLUMNS = ["logged_at", "mode", "policy", "market_ticker", "event_ticker", "set_no", "match_ticker",
                     "match_mid", "fair", "yes_bid", "yes_ask", "price", "edge", "occurrence"]


def maker_fee(p: float) -> float:
    return MAKER_FEE_RATE * p * (1 - p)


def fair_set_prob(match_prob: float) -> float:
    m = min(max(match_prob, 1e-6), 1 - 1e-6)
    return 1.0 / (1.0 + math.exp(-SET_SLOPE * math.log(m / (1 - m))))


def _num(v) -> float | None:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def _book(m: dict) -> tuple[float, float] | None:
    bid, ask = _num(m.get("yes_bid_dollars")), _num(m.get("yes_ask_dollars"))
    if bid is None or ask is None or not (0 < bid < ask < 1):
        return None
    return bid, ask


def _parts(set_ticker: str) -> tuple[str, int, str] | None:
    """KXATPCHALLENGERSETWINNER-<match>-<set>-<code> -> (match, set number, player code)."""
    bits = set_ticker.split("-")
    if len(bits) != 4 or bits[0] != SET_SERIES or not bits[2].isdigit():
        return None
    return bits[1], int(bits[2]), bits[3]


def plan_set_quotes(set_markets: list[dict], match_markets: list[dict], now,
                    margin: float = SET_MAKER_MARGIN, stop_hours: float = SET_MAKER_STOP_HOURS) -> list[dict]:
    now = pd.Timestamp(now)
    match_by = {m.get("ticker"): m for m in match_markets}
    rows = []
    for s in set_markets:
        parts = _parts(str(s.get("ticker", "")))
        if parts is None:
            continue
        match_code, set_no, code = parts
        mt = f"{MATCH_SERIES}-{match_code}-{code}"
        mm = match_by.get(mt)
        if mm is None:
            continue
        occ = pd.to_datetime(mm.get("occurrence_datetime") or s.get("occurrence_datetime"), utc=True, errors="coerce")
        if pd.isna(occ) or now >= occ - pd.Timedelta(hours=stop_hours):
            continue
        mb, sb = _book(mm), _book(s)
        if mb is None or sb is None or mb[1] - mb[0] > MAX_MATCH_SPREAD + 1e-9 or sb[1] - sb[0] > MAX_SET_SPREAD + 1e-9:
            continue
        match_mid = (mb[0] + mb[1]) / 2
        fair = fair_set_prob(match_mid)
        bid, ask = sb
        cands = [("join", round(bid, 2))]
        if round(bid + 0.01, 2) < ask - 1e-9:
            cands.append(("improve", round(bid + 0.01, 2)))
        for policy, price in cands:
            edge = fair - price - maker_fee(price)
            if edge < margin - 1e-9:
                continue
            rows.append({"logged_at": now, "mode": "dry_run", "policy": policy, "market_ticker": s["ticker"],
                         "event_ticker": s.get("event_ticker"), "set_no": set_no, "match_ticker": mt,
                         "match_mid": match_mid, "fair": fair, "yes_bid": bid, "yes_ask": ask, "price": price,
                         "edge": edge, "occurrence": occ})
    return rows


def run_set_maker_dry(client, now) -> list[dict]:
    """List open set-winner and Challenger match markets (public data) and plan quotes. Places nothing."""
    sets = client.list_markets(SET_SERIES, status="open")
    match = client.list_markets(MATCH_SERIES, status="open")
    return plan_set_quotes(sets, match, now)
