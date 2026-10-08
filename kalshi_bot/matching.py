"""Map a model play (player vs opponent) to the player's Kalshi YES market."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from kalshi_bot.names import parse_event_date
from kalshi_bot.names import match_player_name


@dataclass(frozen=True)
class MarketMatch:
    event_ticker: str
    market_ticker: str
    yes_ask: float
    yes_ask_size: float
    occurrence_time: pd.Timestamp | None
    level: str


def index_open_markets(markets_by_level: dict[str, list[dict]]) -> dict[str, tuple[str, list[dict]]]:
    index: dict[str, tuple[str, list[dict]]] = {}
    for level, markets in markets_by_level.items():
        for m in markets:
            ev = m.get("event_ticker", "")
            index.setdefault(ev, (level, []))[1].append(m)
    return index


def _to_float(v) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def find_market(player: str, opponent: str, pred_date: pd.Timestamp, index: dict) -> MarketMatch | None:
    pred_date = pd.Timestamp(pred_date).normalize()
    candidates = [player, opponent]
    for event_ticker, (level, markets) in index.items():
        if len(markets) != 2:
            continue
        ev_date = parse_event_date(event_ticker)
        if ev_date is None or not (pred_date - pd.Timedelta(days=1) <= ev_date <= pred_date + pd.Timedelta(days=3)):
            continue
        mapped = [match_player_name(m.get("yes_sub_title", ""), candidates) for m in markets]
        if set(mapped) != {player, opponent}:
            continue
        mk = markets[mapped.index(player)]
        occ = mk.get("occurrence_datetime")
        return MarketMatch(
            event_ticker=event_ticker,
            market_ticker=mk["ticker"],
            yes_ask=_to_float(mk.get("yes_ask_dollars")),
            yes_ask_size=_to_float(mk.get("yes_ask_size_fp")),
            occurrence_time=pd.Timestamp(occ) if occ else None,
            level=level,
        )
    return None
