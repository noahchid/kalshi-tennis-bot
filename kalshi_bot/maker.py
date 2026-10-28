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
    """One env's maker state: first-seen mid per market, the retired events and the
    halt flag; persisted as JSON (see load_env_state / store_env_state for the file).

    The halt flag is set when a maker fill took liquidity (is_taker true). While it is
    set nothing is quoted; it is cleared only by hand, by deleting "halted" from the file.
    """

    def __init__(self) -> None:
        self._first_mid: dict[str, dict[str, float]] = {}
        self._retired: dict[str, str] = {}
        self._halted: str | None = None

    @property
    def halted(self) -> bool:
        return self._halted is not None

    @property
    def halted_at(self) -> pd.Timestamp | None:
        return None if self._halted is None else pd.Timestamp(self._halted)

    def halt(self, at: pd.Timestamp) -> None:
        if self._halted is None:
            self._halted = pd.Timestamp(at).isoformat()

    def first_mid(self, event: str, market: str) -> float | None:
        return self._first_mid.get(event, {}).get(market)

    def set_first_mid(self, event: str, market: str, mid: float) -> None:
        self._first_mid.setdefault(event, {})[market] = float(mid)

    def is_retired(self, event: str) -> bool:
        return event in self._retired

    def retire(self, event: str, at: pd.Timestamp) -> None:
        self._retired[event] = pd.Timestamp(at).isoformat()

    def _doc(self) -> dict:
        doc = {"first_mid": self._first_mid, "retired": self._retired}
        if self._halted is not None:
            doc["halted"] = self._halted
        return doc

    def to_json(self) -> bytes:
        return json.dumps(self._doc(), sort_keys=True).encode()

    @classmethod
    def _from_doc(cls, data: dict) -> "MakerState":
        state = cls()
        state._first_mid = {
            ev: {mk: float(v) for mk, v in mids.items()} for ev, mids in (data.get("first_mid") or {}).items()
        }
        state._retired = {ev: str(at) for ev, at in (data.get("retired") or {}).items()}
        halted = data.get("halted")
        state._halted = str(halted) if halted else None
        return state

    @classmethod
    def from_json(cls, raw: bytes | str | None) -> "MakerState":
        return cls._from_doc(json.loads(raw)) if raw else cls()


_STATE_KEYS = ("first_mid", "retired", "halted")


def _env_docs(raw: bytes | str | None) -> dict[str, dict]:
    """The state file as {env: state doc}. The pre-env file (one state at the top
    level, written before demo runs were tagged) is prod's state."""
    if not raw:
        return {}
    data = json.loads(raw)
    if any(k in data for k in _STATE_KEYS):
        return {"prod": data}
    return dict(data)


def load_env_state(raw: bytes | str | None, env: str) -> MakerState:
    """This env's MakerState from the state file (fresh when the env has none)."""
    doc = _env_docs(raw).get(env)
    return MakerState._from_doc(doc) if doc else MakerState()


def store_env_state(raw: bytes | str | None, env: str, state: MakerState) -> bytes:
    """The state file with this env's state replaced; other envs are kept as they are."""
    docs = _env_docs(raw)
    docs[env] = state._doc()
    return json.dumps(docs, sort_keys=True).encode()


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
    mids (only for events that get a quote) and retires events. A halted state
    plans nothing (reason "halted", counted once).
    """
    skips: Counter = Counter()
    if state.halted:  # a maker fill took liquidity: nothing until the flag is cleared by hand
        skips["halted"] += 1
        return [], skips
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
    "price", "count", "fee", "is_taker", "kalshi_env",
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


def _is_taker(v) -> bool | None:
    """True/False when the fill says so; None when the field is missing or unreadable."""
    if isinstance(v, bool):
        return v
    if isinstance(v, str) and v.strip().lower() in ("true", "false"):
        return v.strip().lower() == "true"
    if isinstance(v, (int, float)) and v in (0, 1):
        return bool(v)
    return None


def _unit_price(v) -> float | None:
    """A price in dollars strictly inside (0, 1); None otherwise (e.g. a value in cents)."""
    x = _num(v)
    return x if x is not None and math.isfinite(x) and 0 < x < 1 else None


def _event_of(market_ticker: str) -> str:
    return market_ticker.rsplit("-", 1)[0]


def _error_text(e: Exception) -> str:
    return str(e) if isinstance(e, KalshiAPIError) else repr(e)


def _reconcile_fills(client, fills_seen: set[str], now: pd.Timestamp, counts: Counter,
                     placed_orders: dict[str, str], kalshi_env: str) -> list[dict]:
    """New fills of maker orders, deduped by fill id against (and added to) fills_seen.

    Kalshi's /portfolio/fills field names are unverified; each is read with a
    fallback: id fill_id|trade_id, time created_time|ts, ticker ticker|market_ticker,
    price yes_price_dollars|price (dollars), count count_fp|count, fee fee_cost|fees
    (total dollars for the fill). A fill is a maker fill when its order is one the
    bot logged as placed (`placed_orders`: order_id -> client_order_id), else when
    its client_order_id has MAKER_ORDER_PREFIX: taken from the fill when present,
    else looked up by order_id among that event's orders (read once per event per run).

    A maker fill whose price is not strictly inside (0, 1) dollars counts as
    fill_unreadable and is not stored. A missing is_taker is stored as None
    (taker_unknown), never as False.
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
        coid = placed_orders.get(str(order_id)) if order_id is not None else None
        coid = coid or fill.get("client_order_id")
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
        price = _unit_price(_first(fill, "yes_price_dollars", "price"))
        if price is None:
            counts["fill_unreadable"] += 1
            continue
        taker = _is_taker(fill.get("is_taker"))
        if taker is None:
            counts["taker_unknown"] += 1
        elif taker:
            counts["taker_fill"] += 1
        counts["fills"] += 1
        out.append({
            "fill_id": fill_id,
            "fill_ts": _fill_ts(fill),
            "market_ticker": market,
            "event_ticker": event,
            "order_id": order_id,
            "client_order_id": str(coid),
            "price": price,
            "count": _num(_first(fill, "count_fp", "count")),
            "fee": _num(_first(fill, "fee_cost", "fees")),
            "is_taker": taker,
            "kalshi_env": kalshi_env,
        })
    return out


# Key in _cancel_stale's result for a failed cancel whose order has no ticker:
# its market is unknown, so no market may be quoted that run.
_ANY_MARKET = "*"


def _is_maker_order(o: dict) -> bool:
    return str(o.get("client_order_id") or "").startswith(MAKER_ORDER_PREFIX)


def _cancel_stale(client, counts: Counter) -> dict[str, str]:
    """Cancel every resting maker order; returns {market_ticker: error} for failed
    cancels (key _ANY_MARKET when the failed order carries no ticker)."""
    failed: dict[str, str] = {}
    for o in client.get_orders(status="resting"):
        if not _is_maker_order(o):
            continue
        try:
            client.cancel_order(o["order_id"])
            counts["cancelled"] += 1
        except Exception as e:  # incl. "already filled / not found": unknown state, hands off
            counts["cancel_errors"] += 1
            failed[str(o.get("ticker") or _ANY_MARKET)] = _error_text(e)
    return failed


def cancel_resting_maker_orders(client) -> Counter:
    """Cancel-only maker housekeeping (BETTING_MODE=off, or a run with no usable plays):
    cancels every resting atpedge-mk- order, reconciles nothing. Counts only."""
    counts: Counter = Counter()
    _cancel_stale(client, counts)
    return counts


def _has_expiry(o: dict) -> bool:
    return any(o.get(k) not in (None, "", 0) for k in ("expiration_time", "expiration_ts"))


def _cancel_unexpiring(client, rows: list[dict], now: pd.Timestamp, mode: str, kalshi_env: str,
                       counts: Counter) -> list[dict]:
    """Read back resting maker orders and cancel any without an expiry (an endpoint that
    ignored expiration_time would leave a plain GTC bid). One `error` row ("no_expiry")
    per such order."""
    by_order = {r["order_id"]: r for r in rows if r.get("order_id")}
    out = []
    for o in client.get_orders(status="resting"):
        if not _is_maker_order(o) or _has_expiry(o):
            continue
        counts["no_expiry"] += 1
        error = "no_expiry"
        try:
            client.cancel_order(o["order_id"])
            counts["cancelled"] += 1
        except Exception as e:
            counts["cancel_errors"] += 1
            error = f"no_expiry; cancel failed: {_error_text(e)}"
        base = by_order.get(o.get("order_id"))
        if base is None:
            market = str(o.get("ticker") or "")
            base = {c: None for c in QUOTE_COLUMNS}
            base.update(policy="maker", mode=mode, kalshi_env=kalshi_env, market_ticker=market or None,
                        event_ticker=_event_of(market) if market else None, order_id=o.get("order_id"),
                        client_order_id=o.get("client_order_id"))
        out.append(dict(base, logged_at=now, status="error", error=error))
    return out


def _positions(client) -> dict[str, float]:
    """Contracts held per market; an unreadable position fails closed (never quoted)."""
    out: dict[str, float] = {}
    for p in client.get_positions():
        pos = get_position_fp(p)
        try:
            n = abs(float(pos))
        except (TypeError, ValueError):
            n = math.inf
        out[p["ticker"]] = n if math.isfinite(n) else math.inf
    return out


def run_maker(mode: str, client, plays, state: MakerState, fills_seen: set[str],
              bankroll: float, now, kalshi_env: str, housekeeping: bool,
              placed_orders: dict[str, str] | None = None,
              ) -> tuple[list[dict], list[dict], Counter]:
    """One maker run: (quote rows, new fill rows, counts).

    housekeeping=True reconciles maker fills and cancels every resting maker
    order first, in any mode. Only mode "live" places orders; "dry_run" and
    "shadow" log the quotes they would post. A reconciled maker fill with
    is_taker true halts `state` (persisted), and a halted state quotes nothing.
    After placing, resting maker orders without an expiry are cancelled
    ("no_expiry" error rows). `placed_orders` (order_id -> client_order_id of the
    bot's own placed quotes) classifies fills before any order lookup. Prints nothing.
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
        fills = _reconcile_fills(client, fills_seen, now, counts, placed_orders or {}, kalshi_env)
        if any(f["is_taker"] is True for f in fills):
            state.halt(now)
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
        blocked = cancel_failed.get(q.market_ticker) or cancel_failed.get(_ANY_MARKET)
        if blocked:
            row.update(status="cancel_failed", error=blocked)
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
    if mode == "live" and any(r["status"] in ("placed", "error") for r in rows):
        rows += _cancel_unexpiring(client, rows, now, mode, kalshi_env, counts)
    return rows, fills, counts
