"""Pure quote planner for the maker policy: decides which bids to post. No I/O."""

from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import dataclass

import pandas as pd

from kalshi_bot.config import (
    MAKER_MAX_ASK,
    MAKER_MAX_POSITION,
    MAKER_MAX_SPREAD,
    MAKER_MIN_BID,
    MAKER_ORDER_COUNT,
    MAKER_RETIRE_MOVE,
)
from kalshi_bot.matching import Book, EventBooks

_EPS = 1e-9  # float slack so boundary values (spread 0.10, move 0.10) compare as exact


@dataclass(frozen=True)
class Quote:
    event_ticker: str
    market_ticker: str
    role: str
    price: float
    count: int
    mid: float
    yes_bid: float
    yes_ask: float
    occurrence: pd.Timestamp


class MakerState:
    """First-seen mid per market and the set of retired events; persisted as JSON."""

    def __init__(self) -> None:
        self._first_mid: dict[str, dict[str, float]] = {}
        self._retired: dict[str, str] = {}

    def first_mid(self, event: str, market: str) -> float | None:
        return self._first_mid.get(event, {}).get(market)

    def set_first_mid(self, event: str, market: str, mid: float) -> None:
        self._first_mid.setdefault(event, {})[market] = float(mid)

    def is_retired(self, event: str) -> bool:
        return event in self._retired

    def retire(self, event: str, at: pd.Timestamp) -> None:
        self._retired[event] = pd.Timestamp(at).isoformat()

    def to_json(self) -> bytes:
        return json.dumps(
            {"first_mid": self._first_mid, "retired": self._retired}, sort_keys=True
        ).encode()

    @classmethod
    def from_json(cls, raw: bytes | str | None) -> "MakerState":
        state = cls()
        if not raw:
            return state
        data = json.loads(raw)
        state._first_mid = {
            ev: {mk: float(v) for mk, v in mids.items()} for ev, mids in data.get("first_mid", {}).items()
        }
        state._retired = {ev: str(at) for ev, at in data.get("retired", {}).items()}
        return state


def _book_extreme(book: Book) -> bool:
    bid, ask = book.yes_bid, book.yes_ask
    if math.isnan(bid) or math.isnan(ask):
        return True
    return bid < MAKER_MIN_BID - _EPS or ask > MAKER_MAX_ASK + _EPS


def _wide(book: Book) -> bool:
    return book.yes_ask - book.yes_bid > MAKER_MAX_SPREAD + _EPS


def plan_quotes(
    events: list[EventBooks],
    positions: dict[str, float],
    resting_budget: float,
    state: MakerState,
    now: pd.Timestamp,
    stop_hours: float,
) -> tuple[list[Quote], Counter]:
    """Bids to post this run, plus skip-reason counts.

    Event-level reasons (retired, no_start_time, too_close_to_start, retire_move)
    count once per event; market-level reasons (book_extreme, wide_spread,
    position_cap, budget) count once per market. Mutates `state`: records first
    mids and retires events.
    """
    skips: Counter = Counter()
    candidates: list[Quote] = []
    stop = pd.Timedelta(hours=stop_hours)

    for ev in events:
        if state.is_retired(ev.event_ticker):
            skips["retired"] += 1
            continue
        if ev.occurrence_time is None:
            skips["no_start_time"] += 1
            continue
        if now >= ev.occurrence_time - stop:
            skips["too_close_to_start"] += 1
            continue

        quotable: list[Book] = []
        for book in ev.books:
            if _book_extreme(book):
                skips["book_extreme"] += 1
            elif _wide(book):
                skips["wide_spread"] += 1
            else:
                quotable.append(book)

        retire = False
        for book in quotable:
            mid = (book.yes_bid + book.yes_ask) / 2
            first = state.first_mid(ev.event_ticker, book.market_ticker)
            if first is None:
                state.set_first_mid(ev.event_ticker, book.market_ticker, mid)
            elif abs(mid - first) > MAKER_RETIRE_MOVE + _EPS:
                retire = True
        if retire:
            state.retire(ev.event_ticker, now)
            skips["retire_move"] += 1
            continue

        for book in quotable:
            if positions.get(book.market_ticker, 0.0) + MAKER_ORDER_COUNT > MAKER_MAX_POSITION:
                skips["position_cap"] += 1
                continue
            candidates.append(Quote(
                event_ticker=ev.event_ticker,
                market_ticker=book.market_ticker,
                role=book.role,
                price=book.yes_bid,
                count=MAKER_ORDER_COUNT,
                mid=(book.yes_bid + book.yes_ask) / 2,
                yes_bid=book.yes_bid,
                yes_ask=book.yes_ask,
                occurrence=ev.occurrence_time,
            ))

    # Soonest match first, then ticker: the budget goes to the same quotes every run.
    candidates.sort(key=lambda q: (q.occurrence, q.market_ticker))
    quotes: list[Quote] = []
    spent = 0.0
    for q in candidates:
        cost = q.price * q.count
        if spent + cost > resting_budget + _EPS:
            skips["budget"] += 1
            continue
        spent += cost
        quotes.append(q)
    return quotes, skips
