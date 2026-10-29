"""One-off maker order-path probe against Kalshi's DEMO exchange.

Exercises every exchange behaviour the maker policy relies on, through the
bot's own client calls and helpers, and prints a JSON report:

1. a post-only, expiring bid far below the market rests, and comes back from
   GET /portfolio/orders?status=resting with its expiry (the bot's _has_expiry);
2. resting collateral is reserved out of the balance;
3. the bot's _cancel_stale cancels it and it leaves the resting list;
4. a post-only bid at the ask is rejected (never takes liquidity);
5. an order sent with a ~70 s expiry is gone from the resting list after it;
6. a 1-contract immediate-or-cancel buy at the ask fills, and the bot's
   _reconcile_fills reads the fill back (id, time, dollar price, count, fee,
   is_taker=True).

    python3 -m kalshi_bot.maker_probe      # requires KALSHI_ENV=demo
"""

from __future__ import annotations

import json
import math
import sys
import time
from collections import Counter

import pandas as pd

from kalshi_bot.client import KalshiAPIError, KalshiClient
from kalshi_bot.config import KALSHI_ENV, KALSHI_MATCH_SERIES, KALSHI_TENNIS_EXCHANGE_INDEX, MAKER_ORDER_PREFIX
from kalshi_bot.maker import _cancel_stale, _has_expiry, _reconcile_fills

REST_PRICE = 0.01
SHORT_TTL_S = 70
LONG_TTL_S = 14 * 60


def _f(v) -> float:
    try:
        x = float(v)
        return x if math.isfinite(x) else float("nan")
    except (TypeError, ValueError):
        return float("nan")


def _pick_market(client) -> dict | None:
    """An open tennis market with a buyable ask above the resting price."""
    best = None
    for series in KALSHI_MATCH_SERIES.values():
        for m in client.list_markets(series, status="open"):
            ask = _f(m.get("yes_ask_dollars"))
            if not (REST_PRICE + 0.01 < ask < 1):
                continue
            if best is None or ask < _f(best.get("yes_ask_dollars")):
                best = m
    return best


def _post_only(client, ticker: str, price: float, coid: str, ttl_s: int) -> dict:
    try:
        resp = client.create_order(ticker, price, 1, coid, time_in_force="good_till_canceled",
                                   expiration_time=int(time.time()) + ttl_s, post_only=True,
                                   cancel_order_on_pause=True, stp="maker")
        return {"ok": True, "response": resp}
    except KalshiAPIError as e:
        return {"ok": False, "status_code": e.status_code, "error": e.body}


def _resting(client, coid: str) -> list[dict]:
    return [o for o in client.get_orders(status="resting") if o.get("client_order_id") == coid]


VARIANTS = {  # which order-body field makes a resting order acceptable on this exchange?
    "bot_body": dict(time_in_force="good_till_canceled", ttl=True, post_only=True, cancel_order_on_pause=True, stp="maker"),
    "no_cancel_on_pause": dict(time_in_force="good_till_canceled", ttl=True, post_only=True, cancel_order_on_pause=None, stp="maker"),
    "no_post_only": dict(time_in_force="good_till_canceled", ttl=True, post_only=None, cancel_order_on_pause=True, stp="maker"),
    "no_expiry": dict(time_in_force="good_till_canceled", ttl=False, post_only=True, cancel_order_on_pause=True, stp="maker"),
    "stp_taker_at_cross": dict(time_in_force="good_till_canceled", ttl=True, post_only=True, cancel_order_on_pause=True, stp="taker_at_cross"),
    "plain_gtc": dict(time_in_force="good_till_canceled", ttl=False, post_only=None, cancel_order_on_pause=None, stp="taker_at_cross"),
}


def _variant(client, ticker: str, coid: str, v: dict) -> dict:
    try:
        resp = client.create_order(ticker, REST_PRICE, 1, coid, time_in_force=v["time_in_force"],
                                   expiration_time=int(time.time()) + LONG_TTL_S if v["ttl"] else None,
                                   post_only=v["post_only"], cancel_order_on_pause=v["cancel_order_on_pause"],
                                   stp=v["stp"])
        return {"ok": True, "order_id": resp.get("order_id")}
    except KalshiAPIError as e:
        return {"ok": False, "status_code": e.status_code, "error": e.body}


def run_variants(client, env: str = KALSHI_ENV) -> dict:
    """Try each order body on up to two markets; cancel whatever rests."""
    if env != "demo":
        raise ValueError(f"maker probe only runs against demo (KALSHI_ENV={env!r})")
    stamp = int(time.time())
    markets = []
    for series in KALSHI_MATCH_SERIES.values():
        for m in client.list_markets(series, status="open"):
            if REST_PRICE + 0.01 < _f(m.get("yes_ask_dollars")) < 1:
                markets.append(m)
    out: dict = {"env": env, "markets_tried": [], "results": {}}
    for m in markets[:2]:
        t = m["ticker"]
        out["markets_tried"].append({k: m.get(k) for k in ("ticker", "status", "yes_bid_dollars", "yes_ask_dollars",
                                                          "can_close_early", "market_type")})
        out["results"][t] = {name: _variant(client, t, f"{MAKER_ORDER_PREFIX}var-{name}-{stamp}", v)
                             for name, v in VARIANTS.items()}
    counts: Counter = Counter()
    out["cleanup_failed"] = _cancel_stale(client, counts)
    out["cleanup_counts"] = dict(counts)
    leftovers = [o for o in client.get_orders(status="resting") if str(o.get("client_order_id", "")).startswith(
        f"{MAKER_ORDER_PREFIX}var-")]
    out["leftover_resting"] = len(leftovers)
    out["sample_resting_order"] = None
    return out


def run_probe(client, env: str = KALSHI_ENV, sleep=time.sleep) -> dict:
    if env != "demo":
        raise ValueError(f"maker probe only runs against demo (KALSHI_ENV={env!r})")
    stamp = int(time.time())
    checks: dict[str, bool] = {}
    report: dict = {"env": env, "checks": checks}
    market = _pick_market(client)
    if market is None:
        report["error"] = "no open demo tennis market with an ask above 2c"
        return report
    ticker = market["ticker"]
    report.update(ticker=ticker, yes_bid=market.get("yes_bid_dollars"), yes_ask=market.get("yes_ask_dollars"))

    # 1-2. resting post-only bid with expiry, and collateral
    bal0 = client.get_balance(exchange_index=KALSHI_TENNIS_EXCHANGE_INDEX)
    coid = f"{MAKER_ORDER_PREFIX}probe-rest-{stamp}"
    report["rest_order"] = _post_only(client, ticker, REST_PRICE, coid, LONG_TTL_S)
    checks["rest_order_accepted"] = report["rest_order"]["ok"]
    if not report["rest_order"]["ok"]:
        err = str(report["rest_order"].get("error") or "")
        if "not_currently_allowed" in err:  # Kalshi geo-blocks some runner IPs: retry on another runner
            report["geo_blocked"] = True
            report["all_passed"] = False
            return report
        # a non-geo rejection: find which order-body field causes it, on this market
        report["variants"] = {name: _variant(client, ticker, f"{MAKER_ORDER_PREFIX}var-{name}-{stamp}", v)
                              for name, v in VARIANTS.items()}
        _cancel_stale(client, Counter())
    rest = _resting(client, coid)
    report["resting_order_fields"] = sorted(rest[0]) if rest else []
    report["resting_order"] = rest[0] if rest else None
    checks["rest_order_listed_as_resting"] = len(rest) == 1
    checks["rest_order_has_expiry"] = bool(rest) and _has_expiry(rest[0])
    bal1 = client.get_balance(exchange_index=KALSHI_TENNIS_EXCHANGE_INDEX)
    report["balance_before_after_rest"] = [bal0, bal1]
    checks["resting_collateral_reserved"] = bal1 < bal0

    # 3. the bot's own cancel path
    counts: Counter = Counter()
    failed = _cancel_stale(client, counts)
    report["cancel_counts"], report["cancel_failed"] = dict(counts), failed
    checks["cancel_stale_cancelled"] = counts["cancelled"] >= 1 and not failed
    checks["cancelled_order_gone"] = not _resting(client, coid)

    # 4. post-only at the ask must be rejected
    ask = _f(market.get("yes_ask_dollars"))
    cross = _post_only(client, ticker, ask, f"{MAKER_ORDER_PREFIX}probe-cross-{stamp}", LONG_TTL_S)
    report["crossing_order"] = cross
    fill_now = _f((cross.get("response") or {}).get("fill_count"))
    checks["post_only_cross_rejected"] = (not cross["ok"]) or fill_now == 0
    _cancel_stale(client, Counter())  # in case it was accepted and rests

    # 5. exchange-side expiry
    short = f"{MAKER_ORDER_PREFIX}probe-ttl-{stamp}"
    report["short_ttl_order"] = _post_only(client, ticker, REST_PRICE, short, SHORT_TTL_S)
    listed_before = len(_resting(client, short))
    sleep(SHORT_TTL_S + 30)
    listed_after = len(_resting(client, short))
    report["short_ttl_listed_before_after"] = [listed_before, listed_after]
    checks["expiry_honoured"] = listed_before == 1 and listed_after == 0
    _cancel_stale(client, Counter())

    # 6. a real fill, read back by the bot's parser
    t0 = pd.Timestamp.now(tz="UTC")
    ioc_coid = f"atpedge-probe-fill-{stamp}"
    try:
        ioc = client.create_order(ticker, ask, 1, ioc_coid)  # today's taker IOC path
        report["ioc_order"] = {"ok": True, "response": ioc}
    except KalshiAPIError as e:
        ioc = None
        report["ioc_order"] = {"ok": False, "status_code": e.status_code, "error": e.body}
    sleep(5)
    raw = [f for f in client.get_fills(min_ts=int(t0.timestamp()) - 60)
           if ioc and str(f.get("order_id")) == str(ioc.get("order_id"))]
    report["raw_fill_fields"] = sorted(raw[0]) if raw else []
    report["raw_fill"] = raw[0] if raw else None
    fcounts: Counter = Counter()
    parsed = _reconcile_fills(client, set(), pd.Timestamp.now(tz="UTC"), fcounts,
                              {str(ioc.get("order_id")): f"{MAKER_ORDER_PREFIX}probe"} if ioc else {}, env)
    parsed = [p for p in parsed if ioc and str(p["order_id"]) == str(ioc.get("order_id"))]
    report["parsed_fill"], report["parse_counts"] = (parsed[0] if parsed else None), dict(fcounts)
    p = parsed[0] if parsed else {}
    checks["fill_read_back"] = bool(parsed)
    checks["fill_price_in_dollars"] = p.get("price") is not None and 0 < p["price"] < 1
    checks["fill_count_read"] = p.get("count") == 1
    checks["fill_fee_read"] = p.get("fee") is not None and p["fee"] >= 0
    checks["fill_is_taker_read"] = p.get("is_taker") is True
    checks["fill_time_read"] = p.get("fill_ts") is not None
    report["all_passed"] = all(checks.values())
    return report


def main() -> int:
    try:
        if "--variants" in sys.argv:
            print(json.dumps(run_variants(KalshiClient()), indent=2, default=str))
            return 0
        report = run_probe(KalshiClient())
    except ValueError as e:
        print(f"✗ {e}")
        return 2
    print(json.dumps(report, indent=2, default=str))
    if report.get("geo_blocked"):
        return 3
    return 0 if report.get("all_passed") else 1


if __name__ == "__main__":
    sys.exit(main())
