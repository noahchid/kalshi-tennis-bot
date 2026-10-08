import pytest

from kalshi_bot.client import KalshiAPIError
from kalshi_bot.probe import run_probe
from tests.kalshi_fakes import FakeKalshi, make_event


def _fake(**kw):
    return FakeKalshi(markets=make_event("KXATPCHALLENGERMATCH", "26SEP26", "Keegan Smith", "Martin Damm"),
                      **kw)


def test_refuses_outside_demo():
    fake = _fake()
    with pytest.raises(ValueError):
        run_probe(fake, env="prod")
    assert fake.placed == []


def test_places_one_cent_order_then_repeats_same_id():
    fake = _fake()
    report = run_probe(fake, env="demo", now_ts=1234)
    assert [(o["price"], o["count"], o["client_order_id"]) for o in fake.placed] == [
        (0.01, 1, "atpedge-probe-1234")]
    assert report["first_order"]["ok"]
    assert report["repeat_order"]["status_code"] == 409
    assert "fill_count" in report["first_order"]["response"]
    assert report["ticker"].startswith("KXATPCHALLENGERMATCH-")


def test_records_rejection_of_repeat_id():
    class RejectRepeat(FakeKalshi):
        def create_order(self, ticker, price, count, client_order_id):
            if any(p["client_order_id"] == client_order_id for p in self.placed):
                raise KalshiAPIError(409, '{"error":"duplicate client_order_id"}')
            return super().create_order(ticker, price, count, client_order_id)

    report = run_probe(RejectRepeat(markets=_fake().markets), env="demo", now_ts=1)
    assert report["first_order"]["ok"]
    assert report["repeat_order"] == {"ok": False, "status_code": 409,
                                      "error": '{"error":"duplicate client_order_id"}'}


def test_records_order_and_portfolio_field_names():
    fake = _fake(settlements=[{"ticker": "X", "revenue": 0, "fee_cost": "0"}])
    report = run_probe(fake, env="demo", now_ts=1)
    assert report["probe_orders_listed"] == 1
    assert "fill_count_fp" in report["order_fields"]
    assert report["settlement_fields"] == ["fee_cost", "revenue", "ticker"]


def test_no_open_market_reports_without_ordering():
    fake = FakeKalshi(markets=[])
    report = run_probe(fake, env="demo")
    assert report["ticker"] is None and fake.placed == []
