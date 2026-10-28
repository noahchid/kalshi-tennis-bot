"""Bot bankroll inside a shared Kalshi account.

bankroll = start + realized PnL on bot tickers - cost of open bot positions,
capped at the cash on Kalshi's tennis exchange shard (the only cash tennis
orders can draw on), so the bot never counts money it can't spend.
Bot tickers are those with an order whose client_order_id has the bot prefix.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd

from kalshi_bot.config import KALSHI_ENV, KALSHI_ORDER_PREFIX, KALSHI_TENNIS_EXCHANGE_INDEX
from kalshi_bot.bet_log import PLACED_STATUSES
from kalshi_bot.fields import get_fill_count_fp, get_market_exposure_dollars, get_position_fp


@dataclass
class BankrollState:
    start: float
    realized_pnl: float
    open_cost: float
    cash: float
    bankroll: float

    def as_dict(self) -> dict:
        return asdict(self)


def _f(v) -> float:
    try:
        return float(v or 0)
    except (TypeError, ValueError):
        return 0.0


def compute_bankroll(start_bankroll, orders, settlements, positions, cash,
                     extra_bot_tickers=()) -> BankrollState:
    bot_tickers = {
        o["ticker"] for o in orders
        if str(o.get("client_order_id") or "").startswith(KALSHI_ORDER_PREFIX) and _f(get_fill_count_fp(o)) > 0
    } | set(extra_bot_tickers)
    realized = sum(
        _f(s.get("revenue")) / 100.0
        - _f(s.get("yes_total_cost_dollars")) - _f(s.get("no_total_cost_dollars"))
        - _f(s.get("fee_cost"))
        for s in settlements if s.get("ticker") in bot_tickers
    )
    open_cost = sum(
        get_market_exposure_dollars(p)
        for p in positions if p.get("ticker") in bot_tickers and _f(get_position_fp(p)) != 0
    )
    bankroll = max(0.0, min(start_bankroll + realized - open_cost, cash))
    return BankrollState(start_bankroll, round(realized, 2), round(open_cost, 2), cash, round(bankroll, 2))


def _logged_bot_tickers(bet_log: pd.DataFrame | None, env: str) -> set[str]:
    """Tickers the bet log shows the bot filled — a backup to Kalshi's order
    history, which the demo probe showed does not always list past orders."""
    if bet_log is None or bet_log.empty or "market_ticker" not in bet_log.columns:
        return set()
    rows = bet_log[bet_log["status"].isin(PLACED_STATUSES)]
    if "kalshi_env" in rows.columns:
        rows = rows[rows["kalshi_env"] == env]
    return set(rows["market_ticker"].dropna())


def fetch_bankroll(client, start_bankroll: float, bet_log: pd.DataFrame | None = None,
                   env: str = KALSHI_ENV, extra_bot_tickers=()) -> BankrollState:
    """extra_bot_tickers: further bot markets (the maker's filled markets), unioned with
    the bet log's filled taker markets as a backup to Kalshi's order history."""
    cash = client.get_balance(exchange_index=KALSHI_TENNIS_EXCHANGE_INDEX)
    start = start_bankroll if start_bankroll > 0 else cash
    return compute_bankroll(start, client.get_orders(), client.get_settlements(), client.get_positions(),
                            cash, extra_bot_tickers=_logged_bot_tickers(bet_log, env) | set(extra_bot_tickers))
