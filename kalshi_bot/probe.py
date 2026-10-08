"""One-off order-path probe against Kalshi's DEMO exchange.

Sends a 1-contract IOC buy at 1¢ on an open tennis market (which should not
fill), sends it again with the same client_order_id, then lists the resulting
orders and portfolio objects. The printed JSON shows the real endpoint
behaviour and field names the bettor, guard and bankroll depend on.

    python3 -m kalshi_bot.probe      # requires KALSHI_ENV=demo
"""

from __future__ import annotations

import json
import sys
import time

from kalshi_bot.config import KALSHI_ENV, KALSHI_MATCH_SERIES, KALSHI_TENNIS_EXCHANGE_INDEX
from kalshi_bot.client import KalshiAPIError, KalshiClient

PROBE_PRICE = 0.01
PROBE_COUNT = 1


def _attempt(client, ticker: str, coid: str) -> dict:
    try:
        return {"ok": True, "response": client.create_order(ticker, PROBE_PRICE, PROBE_COUNT, coid)}
    except KalshiAPIError as e:
        return {"ok": False, "status_code": e.status_code, "error": e.body}


def _field_names(objects: list[dict]) -> list[str]:
    return sorted(objects[0]) if objects else []


def run_probe(client, env: str = KALSHI_ENV, now_ts: int | None = None) -> dict:
    if env != "demo":
        raise ValueError(f"probe only runs against demo (KALSHI_ENV={env!r})")

    market = next(
        (m for series in KALSHI_MATCH_SERIES.values()
         for m in client.list_markets(series, status="open") if m.get("ticker")),
        None,
    )
    report: dict = {
        "env": env,
        "ticker": market["ticker"] if market else None,
        "market_exchange_index": market.get("exchange_index") if market else None,
        "tennis_shard_balance": client.get_balance(exchange_index=KALSHI_TENNIS_EXCHANGE_INDEX),
    }
    if market is None:
        return report

    coid = f"atpedge-probe-{int(time.time()) if now_ts is None else now_ts}"
    report["client_order_id"] = coid
    report["first_order"] = _attempt(client, market["ticker"], coid)
    report["repeat_order"] = _attempt(client, market["ticker"], coid)

    orders = client.get_orders(event_ticker=market["event_ticker"])
    probe_orders = [o for o in orders if o.get("client_order_id") == coid]
    report["probe_orders_listed"] = len(probe_orders)
    report["order_fields"] = _field_names(probe_orders or orders)
    report["probe_orders"] = probe_orders
    report["position_fields"] = _field_names(client.get_positions())
    report["settlement_fields"] = _field_names(client.get_settlements())
    return report


def main() -> int:
    try:
        report = run_probe(KalshiClient())
    except ValueError as e:
        print(f"✗ {e}")
        return 2
    print(json.dumps(report, indent=2, default=str))
    return 0 if report.get("first_order", {}).get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())
