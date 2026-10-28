"""Maker policy: a pure quote planner (`plan_quotes`) and its executor (`run_maker`)."""

from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import dataclass

import pandas as pd

from kalshi_bot.client import KalshiAPIError
from kalshi_bot.config import (
    KALSHI_MATCH_SERIES,
    MAKER_MAX_ASK,
    MAKER_MAX_POSITION,
    MAKER_MAX_SPREAD,
    MAKER_MIN_BID,
    MAKER_ORDER_COUNT,
    MAKER_ORDER_PREFIX,
    MAKER_ORDER_TTL_MIN,
    MAKER_RESTING_BUDGET,
    MAKER_RETIRE_MOVE,
    MAKER_STOP_HOURS,
)
from kalshi_bot.fields import get_position_fp
from kalshi_bot.guard import client_order_id
from kalshi_bot.matching import Book, EventBooks, find_event, index_open_markets

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


def _finite(book: Book) -> bool:
    return not (math.isnan(book.yes_bid) or math.isnan(book.yes_ask))


def _mid(book: Book) -> float:
    return (book.yes_bid + book.yes_ask) / 2


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
    mids (only for events that get a quote) and retires events.
    """
    skips: Counter = Counter()
    candidates: list[Quote] = []
    by_event: dict[str, EventBooks] = {}
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

        # Any finite book counts toward the retire rule, even one that is wide or
        # extreme: after news one side can blow out while the other still looks quotable.
        retire = False
        for book in ev.books:
            if _finite(book):
                first = state.first_mid(ev.event_ticker, book.market_ticker)
                if first is not None and abs(_mid(book) - first) > MAKER_RETIRE_MOVE + _EPS:
                    retire = True
        if retire:
            state.retire(ev.event_ticker, now)
            skips["retire_move"] += 1
            continue

        by_event[ev.event_ticker] = ev
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
                mid=_mid(book),
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

    # First mids are fixed the first time an event is actually quoted, for both
    # sides together; an event that yielded no accepted quote records nothing.
    for ticker in {q.event_ticker for q in quotes}:
        for book in by_event[ticker].books:
            if _finite(book) and state.first_mid(ticker, book.market_ticker) is None:
                state.set_first_mid(ticker, book.market_ticker, _mid(book))
    return quotes, skips


# ── Executor ─────────────────────────────────────────────────────────────────

QUOTE_COLUMNS = [
    "logged_at", "policy", "mode", "kalshi_env", "event_ticker", "market_ticker", "role",
    "yes_bid", "yes_ask", "mid", "occurrence", "price", "count", "status", "error",
    "order_id", "client_order_id", "expiration_time",
]
FILL_COLUMNS = [
    "fill_id", "fill_ts", "market_ticker", "event_ticker", "order_id", "client_order_id",
    "price", "count", "fee", "is_taker",
]
MAKER_MODES = ("live", "dry_run", "shadow")
_PLACE_STATUS = {"live": "placed", "dry_run": "dry_run", "shadow": "shadow_quote"}

# Maker orders expire MAKER_ORDER_TTL_MIN after placement, so their fills are at
# most that old when the next run starts. A week of lookback covers a long bot
# outage while keeping the fills read (and order lookups for non-maker fills) bounded.
_FILL_LOOKBACK = pd.Timedelta(days=7)


def _first(obj: dict, *keys):
    for k in keys:
        v = obj.get(k)
        if v is not None and v != "":
            return v
    return None


def _num(v) -> float | None:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _fill_ts(fill: dict) -> pd.Timestamp | None:
    v = _first(fill, "created_time", "ts")
    if v is None:
        return None
    try:
        if isinstance(v, (int, float)) or str(v).isdigit():
            return pd.Timestamp(int(v), unit="s", tz="UTC")
        ts = pd.Timestamp(v)
        return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
    except (TypeError, ValueError):
        return None


def _is_taker(v) -> bool:
    if isinstance(v, str):
        return v.strip().lower() == "true"
    return bool(v)


def _event_of(market_ticker: str) -> str:
    return market_ticker.rsplit("-", 1)[0]


def _error_text(e: Exception) -> str:
    return str(e) if isinstance(e, KalshiAPIError) else repr(e)


def _reconcile_fills(client, fills_seen: set[str], now: pd.Timestamp,
                     counts: Counter) -> list[dict]:
    """New fills of maker orders, deduped by fill id against (and added to) fills_seen.

    Kalshi's /portfolio/fills field names are unverified; each is read with a
    fallback: id fill_id|trade_id, time created_time|ts, ticker ticker|market_ticker,
    price yes_price_dollars|price (dollars), count count_fp|count, fee fee_cost|fees
    (total dollars for the fill). A fill is a maker fill when its order's
    client_order_id has MAKER_ORDER_PREFIX: taken from the fill when present,
    else looked up by order_id among that event's orders.
    """
    out: list[dict] = []
    orders_by_event: dict[str, dict[str, str]] = {}
    min_ts = int((now - _FILL_LOOKBACK).timestamp())
    for fill in client.get_fills(min_ts=min_ts):
        fill_id = _first(fill, "fill_id", "trade_id")
        market = _first(fill, "ticker", "market_ticker")
        if fill_id is None or market is None:
            counts["fill_unreadable"] += 1
            continue
        fill_id, market = str(fill_id), str(market)
        if fill_id in fills_seen:
            continue
        event = str(_first(fill, "event_ticker") or _event_of(market))
        order_id = fill.get("order_id")
        coid = fill.get("client_order_id")
        if not coid:
            if event not in orders_by_event:
                orders_by_event[event] = {
                    str(o.get("order_id")): str(o.get("client_order_id") or "")
                    for o in client.get_orders(event_ticker=event)
                }
            coid = orders_by_event[event].get(str(order_id))
            if coid is None:
                # Order not found: leave unseen so a later run can classify it.
                counts["fill_unclassified"] += 1
                continue
        fills_seen.add(fill_id)
        if not str(coid).startswith(MAKER_ORDER_PREFIX):
            continue
        taker = _is_taker(fill.get("is_taker"))
        if taker:
            counts["taker_fill"] += 1
        counts["fills"] += 1
        out.append({
            "fill_id": fill_id,
            "fill_ts": _fill_ts(fill),
            "market_ticker": market,
            "event_ticker": event,
            "order_id": order_id,
            "client_order_id": str(coid),
            "price": _num(_first(fill, "yes_price_dollars", "price")),
            "count": _num(_first(fill, "count_fp", "count")),
            "fee": _num(_first(fill, "fee_cost", "fees")),
            "is_taker": taker,
        })
    return out


def _cancel_stale(client, counts: Counter) -> dict[str, str]:
    """Cancel every resting maker order; returns {market_ticker: error} for failed cancels."""
    failed: dict[str, str] = {}
    for o in client.get_orders(status="resting"):
        if not str(o.get("client_order_id") or "").startswith(MAKER_ORDER_PREFIX):
            continue
        try:
            client.cancel_order(o["order_id"])
            counts["cancelled"] += 1
        except Exception as e:  # incl. "already filled / not found": unknown state, hands off
            counts["cancel_errors"] += 1
            failed[str(o.get("ticker"))] = _error_text(e)
    return failed


def _positions(client) -> dict[str, float]:
    """Contracts held per market; an unreadable position fails closed (never quoted)."""
    out: dict[str, float] = {}
    for p in client.get_positions():
        pos = get_position_fp(p)
        try:
            out[p["ticker"]] = abs(float(pos))
        except (TypeError, ValueError):
            out[p["ticker"]] = math.inf
    return out


def run_maker(mode: str, client, plays, state: MakerState, fills_seen: set[str],
              bankroll: float, now, kalshi_env: str, housekeeping: bool,
              ) -> tuple[list[dict], list[dict], Counter]:
    """One maker run: (quote rows, new fill rows, counts).

    housekeeping=True reconciles maker fills and cancels every resting maker
    order first, in any mode. Only mode "live" places orders; "dry_run" and
    "shadow" log the quotes they would post. Prints nothing.
    """
    if mode not in MAKER_MODES:
        raise ValueError(f"invalid maker mode: {mode!r}")
    if mode == "live" and not housekeeping:
        raise ValueError("live maker runs must cancel stale orders first (housekeeping=True)")
    now = pd.Timestamp(now)
    if now.tzinfo is None:
        now = now.tz_localize("UTC")
    counts: Counter = Counter()

    fills: list[dict] = []
    cancel_failed: dict[str, str] = {}
    if housekeeping:
        fills = _reconcile_fills(client, fills_seen, now, counts)
        cancel_failed = _cancel_stale(client, counts)

    index = index_open_markets({lvl: client.list_markets(series, status="open")
                                for lvl, series in KALSHI_MATCH_SERIES.items()})
    events: dict[str, EventBooks] = {}
    for play in plays:
        ev = find_event(play.player_name, play.opponent_name, play.pred_date, index)
        if ev is None:
            counts["no_event_books"] += 1
            continue
        events.setdefault(ev.event_ticker, ev)

    positions = _positions(client)
    quotes, skips = plan_quotes(list(events.values()), positions, MAKER_RESTING_BUDGET * bankroll,
                                state, now, MAKER_STOP_HOURS)
    counts.update(skips)

    attempt = str(int(now.timestamp()))
    expiration = int(now.timestamp()) + MAKER_ORDER_TTL_MIN * 60
    rows: list[dict] = []
    for q in quotes:
        coid = client_order_id(q.event_ticker, q.market_ticker, attempt, prefix=MAKER_ORDER_PREFIX)
        row = {
            "logged_at": now, "policy": "maker", "mode": mode, "kalshi_env": kalshi_env,
            "event_ticker": q.event_ticker, "market_ticker": q.market_ticker, "role": q.role,
            "yes_bid": q.yes_bid, "yes_ask": q.yes_ask, "mid": q.mid, "occurrence": q.occurrence,
            "price": q.price, "count": q.count, "status": None, "error": None,
            "order_id": None, "client_order_id": coid, "expiration_time": expiration,
        }
        if q.market_ticker in cancel_failed:
            row.update(status="cancel_failed", error=cancel_failed[q.market_ticker])
        elif mode != "live":
            row["status"] = _PLACE_STATUS[mode]
        else:
            try:
                resp = client.create_order(
                    q.market_ticker, q.price, q.count, coid,
                    time_in_force="good_till_canceled", expiration_time=expiration,
                    post_only=True, cancel_order_on_pause=True, stp="maker",
                )
                row.update(status="placed", order_id=resp.get("order_id"))
            except Exception as e:  # e.g. post-only cross: log, no retry, keep going
                row.update(status="error", error=_error_text(e))
        counts[row["status"]] += 1
        rows.append(row)
    return rows, fills, counts
