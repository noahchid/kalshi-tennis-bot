"""In-memory Kalshi stand-in shared by bettor/guard/bankroll tests."""

from __future__ import annotations

import pandas as pd


def make_market(event_ticker, code, name, ask=0.50, ask_size=1000.0, result="",
                occurrence="2030-01-01T00:00:00Z", rules="", bid=None, bid_size=1000.0):
    if bid is None:
        bid = ask - 0.01
    return {
        "event_ticker": event_ticker,
        "ticker": f"{event_ticker}-{code}",
        "yes_sub_title": name,
        "title": f"{name} wins",
        "yes_ask_dollars": f"{ask:.4f}",
        "yes_ask_size_fp": f"{ask_size:.2f}",
        "yes_bid_dollars": f"{bid:.4f}",
        "yes_bid_size_fp": f"{bid_size:.2f}",
        "status": "active" if not result else "finalized",
        "result": result,
        "occurrence_datetime": occurrence,
        "close_time": occurrence,
        "rules_primary": rules,
    }


def make_event(series, date_code, p1, p2, ask1=0.50, ask2=0.52, result_winner=None,
               walkover=False, occurrence="2030-01-01T00:00:00Z", rules="", bid1=None, bid2=None):
    """Two markets for one match. date_code like '26SEP23'."""
    c1, c2 = p1.split()[-1][:3].upper(), p2.split()[-1][:3].upper()
    ev = f"{series}-{date_code}{c1}{c2}"
    if walkover:
        r1 = r2 = "scalar"
    elif result_winner is None:
        r1 = r2 = ""
    else:
        r1 = "yes" if result_winner == p1 else "no"
        r2 = "yes" if result_winner == p2 else "no"
    return [
        make_market(ev, c1, p1, ask1, result=r1, occurrence=occurrence, rules=rules, bid=bid1),
        make_market(ev, c2, p2, ask2, result=r2, occurrence=occurrence, rules=rules, bid=bid2),
    ]


class FakeKalshi:
    """In-memory Kalshi.

    IOC (taker) orders fill immediately at `fill_ratio`. Good-till-canceled
    (maker) orders rest until `cancel_order` or `fill_resting`. `reject_crossing`
    is a set of market tickers whose maker orders are rejected as post-only
    crosses (raising `crossing_exc` if set, else a KalshiAPIError).
    `fail_cancel` is a set of order ids whose cancel raises. Every create,
    cancel and fills read is appended to `calls` in order. Like Kalshi, the
    balance excludes the collateral of resting orders placed here
    (price x unfilled count), until they fill or are cancelled.
    """

    def __init__(self, markets=None, cash=500.0, positions=None, orders=None,
                 settlements=None, fill_ratio=1.0, fail_orders=False, reject_crossing=None,
                 fail_cancel=None, fills=None):
        self.markets = list(markets or [])
        self.cash = cash
        self.positions = list(positions or [])
        self.orders = list(orders or [])
        self.settlements = list(settlements or [])
        self.fill_ratio = fill_ratio
        self.fail_orders = fail_orders
        self.reject_crossing = set(reject_crossing or ())
        self.crossing_exc = None
        self.fail_cancel = set(fail_cancel or ())
        self.fills = list(fills or [])
        self.placed = []
        self.cancelled = []
        self.calls = []
        self._collateral_ids = set()  # resting orders created here hold collateral

    def list_markets(self, series_ticker, status="open", min_close_ts=None):
        return [m for m in self.markets if m["event_ticker"].startswith(series_ticker + "-")]

    def get_balance(self, exchange_index=None):
        reserved = sum(float(o["yes_price_dollars"]) * (o["count"] - float(o["fill_count_fp"]))
                       for o in self.orders
                       if o.get("order_id") in self._collateral_ids and o["status"] == "resting")
        return round(self.cash - reserved, 2)

    def get_positions(self, event_ticker=None):
        return [p for p in self.positions
                if event_ticker is None or p["ticker"].startswith(event_ticker + "-")]

    def get_orders(self, event_ticker=None, status=None):
        return [o for o in self.orders
                if (event_ticker is None or o["ticker"].startswith(event_ticker + "-"))
                and (status is None or o["status"] == status)]

    def get_settlements(self, min_ts=None):
        return list(self.settlements)

    def get_fills(self, min_ts=None, ticker=None):
        self.calls.append(("get_fills", min_ts, ticker))

        def _ts(f):
            if f.get("ts") is not None:
                return int(f["ts"])
            return int(pd.Timestamp(f["created_time"]).timestamp())
        return [f for f in self.fills
                if (min_ts is None or _ts(f) >= min_ts)
                and (ticker is None or (f.get("ticker") or f.get("market_ticker")) == ticker)]

    def create_order(self, ticker, price, count, client_order_id, *,
                     time_in_force="immediate_or_cancel", expiration_time=None, post_only=None,
                     cancel_order_on_pause=None, stp="taker_at_cross"):
        from kalshi_bot.client import KalshiAPIError
        self.calls.append(("create", client_order_id, ticker))
        if self.fail_orders:
            raise KalshiAPIError(400, '{"error":"rejected"}')
        if any(p["client_order_id"] == client_order_id for p in self.placed):
            raise KalshiAPIError(409, '{"error":{"code":"order_already_exists"}}')
        if time_in_force == "good_till_canceled" and ticker in self.reject_crossing:
            if self.crossing_exc is not None:
                raise self.crossing_exc("connection dropped")
            raise KalshiAPIError(400, '{"error":{"code":"invalid_order",'
                                      '"message":"post only order would cross"}}')
        order_id = f"o{len(self.placed) + 1}"
        self.placed.append({"ticker": ticker, "price": price, "count": count,
                            "client_order_id": client_order_id, "time_in_force": time_in_force,
                            "expiration_time": expiration_time, "post_only": post_only,
                            "cancel_order_on_pause": cancel_order_on_pause, "stp": stp})
        if time_in_force == "good_till_canceled":
            self.orders.append({"order_id": order_id, "ticker": ticker,
                                "client_order_id": client_order_id, "status": "resting",
                                "yes_price_dollars": f"{price:.4f}", "count": count,
                                "fill_count_fp": "0.00"})
            self._collateral_ids.add(order_id)
            return {"order_id": order_id, "client_order_id": client_order_id,
                    "status": "resting", "fill_count": "0.00", "remaining_count": f"{count:.2f}"}
        filled = int(count * self.fill_ratio)
        self.orders.append({"order_id": order_id, "ticker": ticker,
                            "client_order_id": client_order_id,
                            "status": "executed" if filled else "canceled",
                            "fill_count_fp": f"{filled:.2f}"})
        if filled:
            self._add_position(ticker, filled, filled * price)
        return {"order_id": order_id, "client_order_id": client_order_id,
                "fill_count": f"{filled:.2f}", "remaining_count": f"{count - filled:.2f}",
                "average_fill_price": f"{price:.4f}" if filled else None,
                "average_fee_paid": "0.0200" if filled else None}

    def cancel_order(self, order_id):
        from kalshi_bot.client import KalshiAPIError
        self.calls.append(("cancel", order_id))
        if order_id in self.fail_cancel:
            raise KalshiAPIError(404, '{"error":{"code":"not_found"}}')
        for o in self.orders:
            if o.get("order_id") == order_id and o["status"] == "resting":
                o["status"] = "canceled"
                self.cancelled.append(order_id)
                return dict(o)
        raise KalshiAPIError(404, '{"error":{"code":"not_found"}}')

    def fill_resting(self, order_id, n, ts, is_taker=False, fee_per_contract=0.02):
        """Fill `n` contracts of a resting maker order at its price (test helper)."""
        o = next(o for o in self.orders if o.get("order_id") == order_id)
        filled = float(o["fill_count_fp"]) + n
        o["fill_count_fp"] = f"{filled:.2f}"
        if filled >= o["count"]:
            o["status"] = "executed"
        price = float(o["yes_price_dollars"])
        self._add_position(o["ticker"], n, n * price)
        ts = pd.Timestamp(ts)
        self.fills.append({"fill_id": f"f{len(self.fills) + 1}", "trade_id": f"t{len(self.fills) + 1}",
                           "order_id": order_id, "ticker": o["ticker"], "side": "yes", "action": "buy",
                           "created_time": ts.strftime("%Y-%m-%dT%H:%M:%SZ"),
                           "yes_price_dollars": f"{price:.4f}", "count_fp": f"{n:.2f}",
                           "is_taker": is_taker, "fee_cost": f"{fee_per_contract * n:.4f}"})

    def _add_position(self, ticker, n, exposure):
        for p in self.positions:
            if p["ticker"] == ticker and p.get("position_fp") is not None:
                p["position_fp"] = f"{float(p['position_fp']) + n:.2f}"
                p["market_exposure_dollars"] = (
                    f"{float(p.get('market_exposure_dollars') or 0) + exposure:.2f}")
                return
        self.positions.append({"ticker": ticker, "position_fp": f"{n:.2f}",
                               "market_exposure_dollars": f"{exposure:.2f}"})
