"""One bet per Kalshi event, ever.

Kalshi is checked directly (positions, resting orders, prior bot fills) so a
lost local bet log can never cause a double bet.
"""

from __future__ import annotations

import hashlib

import pandas as pd

from kalshi_bot.config import KALSHI_ENV, KALSHI_ORDER_PREFIX
from kalshi_bot.bet_log import PLACED_STATUSES
from kalshi_bot.fields import get_fill_count_fp, get_position_fp


def client_order_id(event_ticker: str, market_ticker: str, attempt: str) -> str:
    """Bot-prefixed id: a stable per-market hash plus a per-attempt suffix.

    Kalshi rejects any reused client_order_id (409 order_already_exists), even
    after an unfilled IOC, so each attempt needs a fresh id. Double-bet
    protection comes from duplicate_reason, not from id reuse.
    """
    digest = hashlib.sha1(f"{event_ticker}|{market_ticker}".encode()).hexdigest()[:12]
    return f"{KALSHI_ORDER_PREFIX}{digest}-{attempt}"


def duplicate_reason(event_ticker: str, bet_log: pd.DataFrame, client, env: str = KALSHI_ENV,
                     include_dry_run: bool = False) -> str | None:
    """Why this event must not be bet again, or None.

    include_dry_run: in dry_run mode, earlier dry_run rows count too, so a
    dry run logs each play once instead of on every scheduled run.
    """
    if not bet_log.empty:
        statuses = PLACED_STATUSES | ({"dry_run"} if include_dry_run else set())
        hit = bet_log[(bet_log["event_ticker"] == event_ticker) & bet_log["status"].isin(statuses)]
        if "kalshi_env" in hit.columns:
            hit = hit[hit["kalshi_env"] == env]
        if not hit.empty:
            return "bet_log"
    for p in client.get_positions(event_ticker=event_ticker):
        pos = get_position_fp(p)
        if pos is None:
            return "guard_unverifiable"
        if float(pos) != 0:
            return "kalshi_position"
    if client.get_orders(event_ticker=event_ticker, status="resting"):
        return "kalshi_resting_order"
    for o in client.get_orders(event_ticker=event_ticker):
        if str(o.get("client_order_id") or "").startswith(KALSHI_ORDER_PREFIX):
            fill_count = get_fill_count_fp(o)
            if fill_count is None:
                return "guard_unverifiable"
            if float(fill_count) > 0:
                return "kalshi_bot_order"
    return None
