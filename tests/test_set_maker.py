"""Challenger set-winner maker, dry run only: price set markets from the match market."""

import math

import pandas as pd
import pytest

from kalshi_bot import set_maker as SM

NOW = pd.Timestamp("2026-11-26T08:00:00Z")
OCC = "2026-11-26T22:00:00Z"


def _m(ticker, event, bid, ask, occ=OCC, status="active"):
    return {"ticker": ticker, "event_ticker": event, "yes_bid_dollars": f"{bid:.4f}", "yes_ask_dollars": f"{ask:.4f}",
            "occurrence_datetime": occ, "status": status}


def _books(set_bid=0.60, set_ask=0.63, match_bid=0.74, match_ask=0.76, setno=1, occ=OCC):
    match = [_m("KXATPCHALLENGERMATCH-26NOV26PHABOL-PHA", "KXATPCHALLENGERMATCH-26NOV26PHABOL", match_bid, match_ask, occ)]
    sets = [_m(f"KXATPCHALLENGERSETWINNER-26NOV26PHABOL-{setno}-PHA", f"KXATPCHALLENGERSETWINNER-26NOV26PHABOL-{setno}",
               set_bid, set_ask, occ)]
    return sets, match


def test_fair_from_match_is_the_calibrated_logit_map():
    f = SM.fair_set_prob(0.75)
    assert f == pytest.approx(1 / (1 + math.exp(-0.80 * math.log(0.75 / 0.25))))
    assert SM.fair_set_prob(0.5) == pytest.approx(0.5)
    assert SM.fair_set_prob(0.25) == pytest.approx(1 - f)


def test_quotes_when_the_bid_sits_well_below_fair():
    sets, match = _books(set_bid=0.60, set_ask=0.64)            # match mid 0.75 -> fair ~0.709
    rows = SM.plan_set_quotes(sets, match, NOW)
    assert [r["policy"] for r in rows] == ["join", "improve"]
    j = rows[0]
    assert j["price"] == 0.60 and j["fair"] == pytest.approx(SM.fair_set_prob(0.75))
    assert j["edge"] == pytest.approx(j["fair"] - 0.60 - SM.maker_fee(0.60))
    assert j["set_no"] == 1 and j["match_ticker"].endswith("-PHA")


def test_no_quote_when_the_bid_is_close_to_fair():
    sets, match = _books(set_bid=0.69, set_ask=0.72)
    assert SM.plan_set_quotes(sets, match, NOW) == []


def test_no_quote_after_the_stop_time():
    sets, match = _books(occ="2026-11-26T12:00:00Z")            # occurrence - 6 h = 06:00 < now
    assert SM.plan_set_quotes(sets, match, NOW, stop_hours=6.0) == []


@pytest.mark.parametrize("kw", [dict(match_bid=0.70, match_ask=0.80), dict(set_bid=0.0, set_ask=0.64),
                                dict(set_bid=0.50, set_ask=0.70)])
def test_no_quote_on_a_wide_or_empty_book(kw):
    sets, match = _books(**kw)
    assert SM.plan_set_quotes(sets, match, NOW) == []


def test_no_quote_without_the_match_market():
    sets, _ = _books()
    assert SM.plan_set_quotes(sets, [], NOW) == []


class _Client:
    def __init__(self, sets, match):
        self.by = {"KXATPCHALLENGERSETWINNER": sets, "KXATPCHALLENGERMATCH": match}
        self.orders = []

    def list_markets(self, series_ticker, status="open", min_close_ts=None):
        return self.by.get(series_ticker, [])

    def create_order(self, *a, **k):
        self.orders.append(a)
        raise AssertionError("the set maker must never place orders in dry run")


def test_dry_run_logs_rows_and_never_places_orders():
    sets, match = _books()
    c = _Client(sets, match)
    rows = SM.run_set_maker_dry(c, NOW)
    assert rows and all(r["mode"] == "dry_run" for r in rows) and c.orders == []
    assert set(SM.SET_QUOTE_COLUMNS) <= set(rows[0])
