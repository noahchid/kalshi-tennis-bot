import pytest

from kalshi_bot import maker_probe
from tests.kalshi_fakes import FakeKalshi, make_event


def test_maker_probe_refuses_prod():
    with pytest.raises(ValueError, match="demo"):
        maker_probe.run_probe(FakeKalshi(), env="prod")


def test_maker_probe_runs_against_fake_exchange():
    markets = make_event("KXATPMATCH", "30JAN01", "Carlos Alcaraz", "Jannik Sinner", ask1=0.40, ask2=0.62)
    report = maker_probe.run_probe(FakeKalshi(markets=markets, cash=100.0), env="demo", sleep=lambda s: None)
    assert report["ticker"] and set(report["checks"]) >= {
        "rest_order_accepted", "rest_order_listed_as_resting", "rest_order_has_expiry", "cancel_stale_cancelled",
        "post_only_cross_rejected", "expiry_honoured", "fill_read_back", "fill_is_taker_read"}
    assert report["checks"]["rest_order_accepted"] and report["checks"]["cancel_stale_cancelled"]
