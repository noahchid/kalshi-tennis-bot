"""In-memory Kalshi stand-in shared by bettor/guard/bankroll tests."""

from __future__ import annotations


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
    def __init__(self, markets=None, cash=500.0, positions=None, orders=None,
                 settlements=None, fill_ratio=1.0, fail_orders=False):
        self.markets = list(markets or [])
        self.cash = cash
        self.positions = list(positions or [])
        self.orders = list(orders or [])
        self.settlements = list(settlements or [])
        self.fill_ratio = fill_ratio
        self.fail_orders = fail_orders
        self.placed = []

    def list_markets(self, series_ticker, status="open", min_close_ts=None):
        return [m for m in self.markets if m["event_ticker"].startswith(series_ticker + "-")]

    def get_balance(self, exchange_index=None):
        return self.cash

    def get_positions(self, event_ticker=None):
        return [p for p in self.positions
                if event_ticker is None or p["ticker"].startswith(event_ticker + "-")]

    def get_orders(self, event_ticker=None, status=None):
        return [o for o in self.orders
                if (event_ticker is None or o["ticker"].startswith(event_ticker + "-"))
                and (status is None or o["status"] == status)]

    def get_settlements(self, min_ts=None):
        return list(self.settlements)

    def create_order(self, ticker, price, count, client_order_id):
        from kalshi_bot.client import KalshiAPIError
        if self.fail_orders:
            raise KalshiAPIError(400, '{"error":"rejected"}')
        if any(p["client_order_id"] == client_order_id for p in self.placed):
            raise KalshiAPIError(409, '{"error":{"code":"order_already_exists"}}')
        filled = int(count * self.fill_ratio)
        self.placed.append({"ticker": ticker, "price": price, "count": count,
                            "client_order_id": client_order_id})
        self.orders.append({"ticker": ticker, "client_order_id": client_order_id,
                            "status": "executed" if filled else "canceled",
                            "fill_count_fp": f"{filled:.2f}"})
        if filled:
            self.positions.append({"ticker": ticker, "position_fp": f"{filled:.2f}",
                                   "market_exposure_dollars": f"{filled * price:.2f}"})
        return {"order_id": f"o{len(self.placed)}", "client_order_id": client_order_id,
                "fill_count": f"{filled:.2f}", "remaining_count": f"{count - filled:.2f}",
                "average_fill_price": f"{price:.4f}" if filled else None,
                "average_fee_paid": "0.0200" if filled else None}
