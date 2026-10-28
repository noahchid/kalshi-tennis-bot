import json
import math

import pandas as pd
import pytest

from kalshi_bot.bettor import Play
from kalshi_bot.maker import FILL_COLUMNS, QUOTE_COLUMNS, MakerState, Quote, plan_quotes, run_maker
from kalshi_bot.matching import Book, EventBooks
from tests.kalshi_fakes import FakeKalshi, make_event

OCC = pd.Timestamp("2026-10-30T12:00:00Z")
NOW = OCC - pd.Timedelta(hours=10)


def _book(ticker, role, bid, ask):
    return Book(ticker, role, bid, ask, 20.0, 20.0)


def _ev(name="E1", pick=(0.44, 0.48), other=(0.50, 0.54), occ=OCC):
    return EventBooks(
        event_ticker=name, occurrence_time=occ, level="tour",
        books=(_book(f"{name}-P", "pick", *pick), _book(f"{name}-O", "other", *other)),
    )


def _plan(events, positions=None, budget=10.0, state=None, now=NOW, stop_hours=4.0):
    state = state if state is not None else MakerState()
    quotes, skips = plan_quotes(events, positions or {}, budget, state, now, stop_hours)
    return quotes, skips, state


def test_quotes_both_sides_at_best_bid_count_five():
    quotes, skips, _ = _plan([_ev()])
    assert [q.market_ticker for q in quotes] == ["E1-O", "E1-P"]
    by = {q.market_ticker: q for q in quotes}
    assert by["E1-P"].price == 0.44 and by["E1-O"].price == 0.50  # never improved
    assert all(q.count == 5 for q in quotes)
    assert by["E1-P"].role == "pick" and by["E1-O"].role == "other"
    assert by["E1-P"].event_ticker == "E1" and by["E1-P"].occurrence == OCC
    assert by["E1-P"].yes_bid == 0.44 and by["E1-P"].yes_ask == 0.48
    assert math.isclose(by["E1-P"].mid, 0.46)
    assert not skips
    assert isinstance(quotes[0], Quote)


def test_retired_event_skipped():
    state = MakerState()
    state.retire("E1", NOW)
    quotes, skips, _ = _plan([_ev()], state=state)
    assert not quotes and skips == {"retired": 1}


def test_no_start_time():
    quotes, skips, _ = _plan([_ev(occ=None)])
    assert not quotes and skips == {"no_start_time": 1}


def test_stop_boundary_is_inclusive_and_minute_before_quotes():
    boundary = OCC - pd.Timedelta(hours=4)
    quotes, skips, _ = _plan([_ev()], now=boundary)
    assert not quotes and skips == {"too_close_to_start": 1}
    quotes, skips, _ = _plan([_ev()], now=boundary - pd.Timedelta(minutes=1))
    assert len(quotes) == 2 and not skips


def test_stop_hours_six_moves_boundary():
    now = OCC - pd.Timedelta(hours=5)
    assert len(_plan([_ev()], now=now, stop_hours=4.0)[0]) == 2
    quotes, skips, _ = _plan([_ev()], now=now, stop_hours=6.0)
    assert not quotes and skips == {"too_close_to_start": 1}


def test_book_extreme_bounds():
    # bid 0.05 and ask 0.95 exactly are still quotable
    quotes, skips, _ = _plan([_ev(pick=(0.05, 0.10), other=(0.90, 0.95))])
    assert len(quotes) == 2 and not skips
    quotes, skips, _ = _plan([_ev(pick=(0.04, 0.09), other=(0.90, 0.96))])
    assert not quotes and skips == {"book_extreme": 2}


def test_book_extreme_is_per_market():
    quotes, skips, _ = _plan([_ev(pick=(0.04, 0.09), other=(0.90, 0.95))])
    assert [q.market_ticker for q in quotes] == ["E1-O"]
    assert skips == {"book_extreme": 1}


def test_nan_book_is_extreme():
    quotes, skips, _ = _plan([_ev(pick=(float("nan"), 0.48), other=(0.50, float("nan")))])
    assert not quotes and skips == {"book_extreme": 2}


def test_wide_spread_boundary():
    quotes, skips, _ = _plan([_ev(pick=(0.45, 0.55), other=(0.40, 0.51))])  # 0.10 ok, 0.11 skip
    assert [q.market_ticker for q in quotes] == ["E1-P"]
    assert skips == {"wide_spread": 1}


def test_retire_on_move_over_threshold_and_stays_retired():
    state = MakerState()
    _plan([_ev(pick=(0.44, 0.48), other=(0.50, 0.54))], state=state)  # first mids 0.46, 0.52
    assert state.first_mid("E1", "E1-P") == pytest.approx(0.46)
    # pick mid moves to 0.57 (+0.11)
    moved = _ev(pick=(0.55, 0.59), other=(0.50, 0.54))
    quotes, skips, _ = _plan([moved], state=state)
    assert not quotes and skips == {"retire_move": 1}
    assert state.is_retired("E1")
    # back at the original prices: still retired
    quotes, skips, _ = _plan([_ev()], state=state)
    assert not quotes and skips == {"retired": 1}


def test_keep_on_move_of_exactly_threshold():
    state = MakerState()
    _plan([_ev(pick=(0.44, 0.48), other=(0.50, 0.54))], state=state)
    kept = _ev(pick=(0.54, 0.58), other=(0.50, 0.54))  # mid 0.56, +0.10
    quotes, skips, _ = _plan([kept], state=state)
    assert len(quotes) == 2 and not skips
    assert not state.is_retired("E1")


def test_retire_on_other_side_move_too():
    state = MakerState()
    _plan([_ev()], state=state)
    moved = _ev(other=(0.38, 0.42))  # other mid 0.40 vs 0.52
    quotes, skips, _ = _plan([moved], state=state)
    assert not quotes and skips == {"retire_move": 1} and state.is_retired("E1")


def test_first_mids_recorded_for_both_sides_when_event_quoted_and_not_overwritten():
    state = MakerState()
    # pick side is extreme (not quoted) but finite: still recorded alongside the quoted side
    quotes, _, _ = _plan([_ev(pick=(0.04, 0.09))], state=state)
    assert [q.market_ticker for q in quotes] == ["E1-O"]
    assert state.first_mid("E1", "E1-P") == pytest.approx(0.065)
    assert state.first_mid("E1", "E1-O") == pytest.approx(0.52)
    _plan([_ev(pick=(0.06, 0.10), other=(0.51, 0.55))], state=state)
    assert state.first_mid("E1", "E1-P") == pytest.approx(0.065)
    assert state.first_mid("E1", "E1-O") == pytest.approx(0.52)


def test_no_first_mids_when_nothing_quoted():
    state = MakerState()
    _plan([_ev(pick=(0.04, 0.09), other=(0.90, 0.96))], state=state)  # both extreme
    assert state.first_mid("E1", "E1-P") is None and state.first_mid("E1", "E1-O") is None
    _plan([_ev()], positions={"E1-P": 6, "E1-O": 6}, state=state)  # both capped
    assert state.first_mid("E1", "E1-P") is None and state.first_mid("E1", "E1-O") is None


def test_event_fully_dropped_by_budget_records_no_first_mids():
    state = MakerState()
    quotes, skips, _ = _plan([_ev("A"), _ev("B")], budget=5.0, state=state)  # A fits, B dropped
    assert [q.event_ticker for q in quotes] == ["A", "A"] and skips == {"budget": 2}
    assert state.first_mid("A", "A-P") is not None and state.first_mid("A", "A-O") is not None
    assert state.first_mid("B", "B-P") is None and state.first_mid("B", "B-O") is None


def test_event_with_one_market_dropped_records_both_sides():
    state = MakerState()
    _plan([_ev()], positions={"E1-P": 6}, state=state)
    assert state.first_mid("E1", "E1-P") == pytest.approx(0.46)
    assert state.first_mid("E1", "E1-O") == pytest.approx(0.52)


def test_wide_side_with_large_move_retires_event_even_if_other_side_quotable():
    state = MakerState()
    _plan([_ev()], state=state)  # first mids 0.46, 0.52
    blown = _ev(pick=(0.50, 0.68), other=(0.51, 0.55))  # pick wide, mid 0.59 (+0.13); other +0.01
    quotes, skips, _ = _plan([blown], state=state)
    assert not quotes and skips["retire_move"] == 1 and state.is_retired("E1")
    quotes, skips, _ = _plan([_ev()], state=state)
    assert not quotes and skips == {"retired": 1}


def test_wide_side_with_026_move_retires():
    state = MakerState()
    _plan([_ev()], state=state)
    blown = _ev(pick=(0.58, 0.86), other=(0.50, 0.54))  # mid 0.72 vs 0.46 = +0.26
    quotes, skips, _ = _plan([blown], state=state)
    assert not quotes and skips["retire_move"] == 1 and state.is_retired("E1")


def test_extreme_side_with_large_move_retires():
    state = MakerState()
    _plan([_ev()], state=state)
    blown = _ev(pick=(0.02, 0.06), other=(0.50, 0.54))  # pick mid 0.04 vs 0.46
    quotes, skips, _ = _plan([blown], state=state)
    assert not quotes and skips["retire_move"] == 1 and state.is_retired("E1")


def test_position_cap_boundary():
    quotes, skips, _ = _plan([_ev()], positions={"E1-P": 5})
    assert len(quotes) == 2 and not skips
    quotes, skips, _ = _plan([_ev()], positions={"E1-P": 6})
    assert [q.market_ticker for q in quotes] == ["E1-O"]
    assert skips == {"position_cap": 1}


def test_budget_cumulative_and_deterministic_order():
    later = pd.Timestamp("2026-10-31T12:00:00Z")
    soon = _ev("B", occ=OCC)
    far = _ev("A", occ=later)
    # each quote costs price*5: B-O 2.50, B-P 2.20, A-O 2.50, A-P 2.20
    quotes, skips, _ = _plan([far, soon], budget=5.0)
    # sorted by (occurrence, ticker): B-O (2.50), B-P (2.20 -> 4.70), A-O (7.20 > 5) skipped, A-P (6.90) skipped
    assert [q.market_ticker for q in quotes] == ["B-O", "B-P"]
    assert skips == {"budget": 2}
    quotes2, _, _ = _plan([soon, far], budget=5.0)
    assert [q.market_ticker for q in quotes2] == ["B-O", "B-P"]


def test_budget_exact_fit_quotes():
    quotes, skips, _ = _plan([_ev()], budget=2.20 + 2.50)
    assert len(quotes) == 2 and not skips


def test_budget_ties_on_occurrence_break_by_ticker():
    quotes, skips, _ = _plan([_ev("Z"), _ev("A")], budget=5.0)
    assert [q.market_ticker for q in quotes] == ["A-O", "A-P"]
    assert skips == {"budget": 2}


def test_state_json_round_trip():
    state = MakerState()
    state.set_first_mid("E1", "E1-P", 0.46)
    state.set_first_mid("E1", "E1-O", 0.52)
    state.retire("E2", NOW)
    blob = state.to_json()
    assert isinstance(blob, bytes)
    back = MakerState.from_json(blob)
    assert back.first_mid("E1", "E1-P") == pytest.approx(0.46)
    assert back.first_mid("E1", "E1-O") == pytest.approx(0.52)
    assert back.first_mid("E1", "nope") is None
    assert back.is_retired("E2") and not back.is_retired("E1")
    assert back.to_json() == blob


def test_state_from_none_or_empty_is_fresh():
    assert not MakerState.from_json(None).is_retired("E1")
    assert MakerState.from_json(b"").first_mid("E1", "x") is None


# ── run_maker: executor (cancel stale, place, reconcile fills) ─────────────

RUN_NOW = pd.Timestamp("2026-10-29T00:00:00Z")
EV = "KXATPMATCH-26OCT29DUCSON"
DUC, SON = EV + "-DUC", EV + "-SON"


def _mplay(p="James Duckworth", o="Lorenzo Sonego"):
    return Play(pd.Timestamp("2026-10-29"), p, o, 0.60, "Some Open", "A", 0.50)


def _mfake(**kw):
    # DUC: bid 0.44 / ask 0.45; SON: bid 0.56 / ask 0.57; start 18h after RUN_NOW
    markets = make_event("KXATPMATCH", "26OCT29", "James Duckworth", "Lorenzo Sonego",
                         ask1=0.45, ask2=0.57, occurrence="2026-10-29T18:00:00Z")
    return FakeKalshi(markets=markets, cash=1000.0, **kw)


def _mrun(fake, mode="live", now=RUN_NOW, state=None, seen=None, bankroll=100.0,
          plays=None, housekeeping=True, placed_orders=None):
    state = state if state is not None else MakerState()
    seen = seen if seen is not None else set()
    kw = {} if placed_orders is None else {"placed_orders": placed_orders}
    quotes, fills, counts = run_maker(mode, fake, plays if plays is not None else [_mplay()],
                                      state, seen, bankroll, now, "demo", housekeeping, **kw)
    return quotes, fills, counts


def _creates(fake):
    return [c for c in fake.calls if c[0] == "create"]


def test_live_places_both_markets_with_exact_maker_body():
    fake = _mfake()
    quotes, fills, counts = _mrun(fake)
    assert not fills
    assert len(fake.placed) == 2
    ts = int(RUN_NOW.timestamp())
    by = {p["ticker"]: p for p in fake.placed}
    assert set(by) == {DUC, SON}
    for ticker, price in ((DUC, 0.44), (SON, 0.56)):
        p = by[ticker]
        assert p["price"] == pytest.approx(price) and p["count"] == 5
        assert p["time_in_force"] == "good_till_canceled"
        assert p["expiration_time"] == ts + 14 * 60
        assert p["post_only"] is True and p["cancel_order_on_pause"] is True
        assert p["stp"] == "maker"
        assert p["client_order_id"].startswith("atpedge-mk-")
        assert p["client_order_id"].endswith(f"-{ts}")
    assert by[DUC]["client_order_id"] != by[SON]["client_order_id"]
    assert len(quotes) == 2 and all(set(r) == set(QUOTE_COLUMNS) for r in quotes)
    assert {r["status"] for r in quotes} == {"placed"}
    row = next(r for r in quotes if r["market_ticker"] == DUC)
    assert row["policy"] == "maker" and row["mode"] == "live" and row["kalshi_env"] == "demo"
    assert row["event_ticker"] == EV and row["role"] == "pick"
    assert row["yes_bid"] == pytest.approx(0.44) and row["yes_ask"] == pytest.approx(0.45)
    assert row["order_id"] and row["client_order_id"] == by[DUC]["client_order_id"]
    assert row["expiration_time"] == ts + 14 * 60
    assert counts["placed"] == 2


def test_next_run_cancels_stale_maker_orders_before_placing():
    fake = _mfake()
    state, seen = MakerState(), set()
    first, _, _ = _mrun(fake, state=state, seen=seen)
    first_ids = {r["order_id"] for r in first}
    fake.calls.clear()
    later = RUN_NOW + pd.Timedelta(minutes=15)
    second, _, counts = _mrun(fake, now=later, state=state, seen=seen)
    kinds = [c[0] for c in fake.calls if c[0] in ("cancel", "create")]
    assert kinds == ["cancel", "cancel", "create", "create"]
    assert {c[1] for c in fake.calls if c[0] == "cancel"} == first_ids
    assert counts["cancelled"] == 2 and counts["placed"] == 2
    resting = fake.get_orders(status="resting")
    assert {o["order_id"] for o in resting} == {r["order_id"] for r in second}


def test_non_maker_resting_orders_are_not_cancelled():
    fake = _mfake(orders=[{"order_id": "manual1", "ticker": DUC, "status": "resting",
                           "client_order_id": "atpedge-abc-1", "fill_count_fp": "0.00"}])
    _mrun(fake)
    assert not [c for c in fake.calls if c[0] == "cancel"]


def test_fill_recorded_once_across_two_runs():
    fake = _mfake()
    state, seen = MakerState(), set()
    first, _, _ = _mrun(fake, state=state, seen=seen)
    duc = next(r for r in first if r["market_ticker"] == DUC)
    fake.fill_resting(duc["order_id"], 5, ts=RUN_NOW + pd.Timedelta(minutes=3))
    _, fills, counts = _mrun(fake, now=RUN_NOW + pd.Timedelta(minutes=15), state=state, seen=seen)
    assert len(fills) == 1 and set(fills[0]) == set(FILL_COLUMNS)
    f = fills[0]
    assert f["market_ticker"] == DUC and f["event_ticker"] == EV
    assert f["order_id"] == duc["order_id"] and f["client_order_id"] == duc["client_order_id"]
    assert f["price"] == pytest.approx(0.44) and f["count"] == pytest.approx(5)
    assert f["fee"] == pytest.approx(0.10)  # total for the fill, not per contract
    assert f["is_taker"] is False
    assert f["fill_ts"] == RUN_NOW + pd.Timedelta(minutes=3)
    assert f["fill_id"] in seen
    assert counts["fills"] == 1 and counts["taker_fill"] == 0
    _, fills3, counts3 = _mrun(fake, now=RUN_NOW + pd.Timedelta(minutes=30), state=state, seen=seen)
    assert fills3 == [] and counts3["fills"] == 0


def test_partial_fill_of_two_counts_toward_cap():
    fake = _mfake(positions=[{"ticker": DUC, "position_fp": "4.00"}])
    state, seen = MakerState(), set()
    first, _, _ = _mrun(fake, state=state, seen=seen)
    assert {r["market_ticker"] for r in first} == {DUC, SON}  # 4 + 5 <= 10
    duc = next(r for r in first if r["market_ticker"] == DUC)
    fake.fill_resting(duc["order_id"], 2, ts=RUN_NOW + pd.Timedelta(minutes=5))
    fake.calls.clear()
    second, fills, counts = _mrun(fake, now=RUN_NOW + pd.Timedelta(minutes=15), state=state, seen=seen)
    assert len(fills) == 1 and fills[0]["count"] == pytest.approx(2)
    # position 6 + 5 > 10: DUC is not quoted again, SON is
    assert [c[2] for c in _creates(fake)] == [SON]
    assert {r["market_ticker"] for r in second} == {SON}
    assert counts["position_cap"] == 1


def test_crossing_reject_logs_error_and_remaining_quotes_still_placed():
    fake = _mfake(reject_crossing={DUC})
    quotes, _, counts = _mrun(fake)
    assert [c[2] for c in _creates(fake)] == [DUC, SON]  # one attempt each, no retry
    by = {r["market_ticker"]: r for r in quotes}
    assert by[DUC]["status"] == "error" and "post" in by[DUC]["error"].lower()
    assert by[DUC]["order_id"] is None
    assert by[SON]["status"] == "placed"
    assert counts["error"] == 1 and counts["placed"] == 1


def test_unexpected_exception_on_place_logs_error_and_continues():
    fake = _mfake(reject_crossing={DUC})
    fake.crossing_exc = ConnectionError
    quotes, _, counts = _mrun(fake)
    by = {r["market_ticker"]: r for r in quotes}
    assert by[DUC]["status"] == "error" and by[SON]["status"] == "placed"


def test_cancel_failure_marks_market_and_places_nothing_there():
    fake = _mfake()
    state, seen = MakerState(), set()
    first, _, _ = _mrun(fake, state=state, seen=seen)
    duc = next(r for r in first if r["market_ticker"] == DUC)
    fake.fail_cancel = {duc["order_id"]}
    fake.calls.clear()
    second, _, counts = _mrun(fake, now=RUN_NOW + pd.Timedelta(minutes=15), state=state, seen=seen)
    assert [c[2] for c in _creates(fake)] == [SON]
    by = {r["market_ticker"]: r for r in second}
    assert by[DUC]["status"] == "cancel_failed" and by[DUC]["order_id"] is None
    assert by[DUC]["error"]
    assert by[SON]["status"] == "placed"
    assert counts["cancel_errors"] == 1 and counts["cancelled"] == 1


def test_dry_run_sends_nothing():
    fake = _mfake()
    quotes, _, counts = _mrun(fake, mode="dry_run")
    assert fake.placed == [] and not _creates(fake)
    assert [r["status"] for r in quotes] == ["dry_run", "dry_run"]
    assert all(r["order_id"] is None for r in quotes)
    assert counts["dry_run"] == 2


def test_shadow_sends_nothing_and_logs_shadow_quotes():
    fake = _mfake()
    quotes, _, counts = _mrun(fake, mode="shadow", housekeeping=False)
    assert fake.placed == [] and not _creates(fake)
    assert [r["status"] for r in quotes] == ["shadow_quote", "shadow_quote"]
    assert all(r["mode"] == "shadow" for r in quotes)
    # no housekeeping: no fills read, no orders listed or cancelled
    assert not [c for c in fake.calls if c[0] in ("get_fills", "cancel")]
    assert counts["shadow_quote"] == 2


def test_shadow_with_housekeeping_cancels_stale_and_reconciles():
    fake = _mfake()
    state, seen = MakerState(), set()
    first, _, _ = _mrun(fake, state=state, seen=seen)
    duc = next(r for r in first if r["market_ticker"] == DUC)
    fake.fill_resting(duc["order_id"], 1, ts=RUN_NOW + pd.Timedelta(minutes=2))
    fake.calls.clear()
    quotes, fills, counts = _mrun(fake, mode="shadow", now=RUN_NOW + pd.Timedelta(minutes=15),
                                  state=state, seen=seen)
    assert len(fills) == 1
    assert len([c for c in fake.calls if c[0] == "cancel"]) == 2
    assert not _creates(fake)
    assert not fake.get_orders(status="resting")
    assert {r["status"] for r in quotes} == {"shadow_quote"}


def test_is_taker_fill_counted():
    fake = _mfake()
    state, seen = MakerState(), set()
    first, _, _ = _mrun(fake, state=state, seen=seen)
    fake.fill_resting(first[0]["order_id"], 5, ts=RUN_NOW + pd.Timedelta(minutes=1), is_taker=True)
    _, fills, counts = _mrun(fake, now=RUN_NOW + pd.Timedelta(minutes=15), state=state, seen=seen)
    assert fills[0]["is_taker"] is True
    assert counts["taker_fill"] == 1


def test_non_maker_fills_ignored_and_fill_coid_used_directly():
    fake = _mfake(orders=[{"order_id": "t1", "ticker": DUC, "status": "executed",
                           "client_order_id": "atpedge-abc-1", "fill_count_fp": "3.00"}])
    fake.fills.append({"fill_id": "f-taker", "order_id": "t1", "ticker": DUC,
                       "created_time": "2026-10-28T23:00:00Z", "yes_price_dollars": "0.45",
                       "count_fp": "3.00", "is_taker": True})
    fake.fills.append({"fill_id": "f-mk", "order_id": "gone", "ticker": SON,
                       "client_order_id": "atpedge-mk-zzz-1",
                       "created_time": "2026-10-28T23:30:00Z", "yes_price_dollars": "0.56",
                       "count_fp": "1.00", "is_taker": False, "fee_cost": "0.01"})
    _, fills, counts = _mrun(fake, mode="dry_run")
    assert [f["fill_id"] for f in fills] == ["f-mk"]
    assert fills[0]["client_order_id"] == "atpedge-mk-zzz-1" and fills[0]["event_ticker"] == EV
    assert counts["taker_fill"] == 0


def test_fill_field_fallbacks():
    fake = _mfake(orders=[{"order_id": "m1", "ticker": SON, "status": "executed",
                           "client_order_id": "atpedge-mk-abc-1", "fill_count_fp": "2.00"}])
    fake.fills.append({"trade_id": "tr1", "order_id": "m1", "market_ticker": SON,
                       "ts": int((RUN_NOW - pd.Timedelta(hours=1)).timestamp()),
                       "price": "0.56", "count": 2, "is_taker": "false", "fees": "0.03"})
    _, fills, _ = _mrun(fake, mode="dry_run")
    f = fills[0]
    assert f["fill_id"] == "tr1" and f["market_ticker"] == SON
    assert f["fill_ts"] == RUN_NOW - pd.Timedelta(hours=1)
    assert f["price"] == pytest.approx(0.56) and f["count"] == pytest.approx(2)
    assert f["fee"] == pytest.approx(0.03) and f["is_taker"] is False


def test_unreadable_position_fails_closed():
    fake = _mfake(positions=[{"ticker": DUC}])
    quotes, _, counts = _mrun(fake)
    assert [r["market_ticker"] for r in quotes] == [SON]
    assert counts["position_cap"] == 1


def test_resting_budget_is_fraction_of_bankroll():
    fake = _mfake()
    # budget 0.40 * 10 = 4.00: DUC (2.20) fits, SON (2.80) would make 5.00
    quotes, _, counts = _mrun(fake, bankroll=10.0)
    assert [r["market_ticker"] for r in quotes] == [DUC]
    assert counts["budget"] == 1


def test_unmatched_play_counts_no_event_books():
    fake = _mfake()
    quotes, _, counts = _mrun(fake, plays=[_mplay("Nobody Atall", "Lorenzo Sonego")])
    assert quotes == [] and counts["no_event_books"] == 1 and not _creates(fake)


def test_one_market_event_counts_no_event_books():
    fake = _mfake()
    fake.markets = [m for m in fake.markets if m["ticker"] == DUC]
    quotes, _, counts = _mrun(fake)
    assert quotes == [] and counts["no_event_books"] == 1


def test_mode_guards():
    with pytest.raises(ValueError):
        _mrun(_mfake(), mode="off")
    with pytest.raises(ValueError):
        _mrun(_mfake(), mode="live", housekeeping=False)


# ── final-review fixes ───────────────────────────────────────────────────────

LATER = RUN_NOW + pd.Timedelta(minutes=15)


def _maker_fill(fill_id, ticker=SON, order_id="own1", **kw):
    f = {"fill_id": fill_id, "order_id": order_id, "ticker": ticker,
         "created_time": "2026-10-28T23:30:00Z", "yes_price_dollars": "0.56",
         "count_fp": "2.00", "is_taker": False, "fee_cost": "0.02"}
    f.update(kw)
    return {k: v for k, v in f.items() if v is not None}


# I1: fills are tagged with the env they came from
def test_fill_rows_carry_kalshi_env():
    assert "kalshi_env" in FILL_COLUMNS
    fake = _mfake()
    state, seen = MakerState(), set()
    first, _, _ = _mrun(fake, state=state, seen=seen)
    fake.fill_resting(first[0]["order_id"], 1, ts=RUN_NOW + pd.Timedelta(minutes=3))
    _, fills, _ = _mrun(fake, now=LATER, state=state, seen=seen)
    assert len(fills) == 1 and fills[0]["kalshi_env"] == "demo"
    assert set(fills[0]) == set(FILL_COLUMNS)


# I1: one MakerState per env in the same JSON; the pre-env shape is prod's
def test_env_states_are_separate_and_legacy_shape_is_prod():
    from kalshi_bot.maker import load_env_state, store_env_state
    legacy = b'{"first_mid": {"E1": {"E1-P": 0.46}}, "retired": {"E2": "2026-10-01T00:00:00+00:00"}}'
    prod = load_env_state(legacy, "prod")
    assert prod.first_mid("E1", "E1-P") == pytest.approx(0.46) and prod.is_retired("E2")
    demo = load_env_state(legacy, "demo")
    assert demo.first_mid("E1", "E1-P") is None and not demo.is_retired("E2")
    demo.set_first_mid("E1", "E1-P", 0.60)
    merged = store_env_state(legacy, "demo", demo)
    assert set(json.loads(merged)) == {"prod", "demo"}
    assert load_env_state(merged, "prod").first_mid("E1", "E1-P") == pytest.approx(0.46)
    assert load_env_state(merged, "prod").is_retired("E2")
    assert load_env_state(merged, "demo").first_mid("E1", "E1-P") == pytest.approx(0.60)
    assert load_env_state(None, "prod").first_mid("E1", "E1-P") is None
    assert store_env_state(merged, "demo", load_env_state(merged, "demo")) == merged


# I3: classification by the bot's own placed order ids first, then the fill's
# client_order_id, and only then the per-event order history
def test_fill_classified_by_own_placed_order_id_without_order_lookup():
    fake = _mfake()
    fake.fills.append(_maker_fill("f1"))
    _, fills, counts = _mrun(fake, mode="dry_run", placed_orders={"own1": "atpedge-mk-aaa-1"})
    assert [f["fill_id"] for f in fills] == ["f1"]
    assert fills[0]["client_order_id"] == "atpedge-mk-aaa-1"
    assert fake.order_queries == [] and counts["fill_unclassified"] == 0


def test_fill_coid_prefix_classifies_without_order_lookup():
    fake = _mfake()
    fake.fills.append(_maker_fill("f-taker", order_id="t9", client_order_id="atpedge-abc-1"))
    fake.fills.append(_maker_fill("f-mk", order_id="m9", client_order_id="atpedge-mk-zzz-1"))
    _, fills, _ = _mrun(fake, mode="dry_run")
    assert [f["fill_id"] for f in fills] == ["f-mk"] and fake.order_queries == []


def test_non_maker_fills_without_coid_look_up_orders_once_per_event():
    fake = _mfake(orders=[{"order_id": "t1", "ticker": DUC, "status": "executed",
                           "client_order_id": "atpedge-abc-1", "fill_count_fp": "3.00"}])
    for i in range(3):
        fake.fills.append(_maker_fill(f"ft{i}", ticker=DUC, order_id="t1"))
    _, fills, _ = _mrun(fake, mode="dry_run")
    assert fills == [] and fake.order_queries == [EV]


def test_missing_is_taker_is_stored_as_unknown_and_does_not_halt():
    fake = _mfake()
    fake.fills.append(_maker_fill("f1", client_order_id="atpedge-mk-zzz-1", is_taker=None))
    state = MakerState()
    _, fills, counts = _mrun(fake, mode="dry_run", state=state)
    assert fills[0]["is_taker"] is None
    assert counts["taker_unknown"] == 1 and counts["taker_fill"] == 0
    assert not state.halted


@pytest.mark.parametrize("price", ["45", "0", "1.00", "-0.2", "nan", None])
def test_fill_price_outside_unit_interval_is_stored_without_price(price):
    fake = _mfake()
    fake.fills.append(_maker_fill("f1", client_order_id="atpedge-mk-zzz-1", yes_price_dollars=price))
    _, fills, counts = _mrun(fake, mode="dry_run")
    assert [f["fill_id"] for f in fills] == ["f1"] and fills[0]["price"] is None
    assert counts["fill_unreadable"] == 1 and counts["fills"] == 1


def test_taker_fill_with_unreadable_price_still_halts():
    fake = _mfake()
    fake.fills.append(_maker_fill("f1", client_order_id="atpedge-mk-zzz-1", yes_price_dollars="45",
                                  is_taker=True))
    state = MakerState()
    quotes, fills, counts = _mrun(fake, state=state)
    assert state.halted and counts["taker_fill"] == 1 and counts["fill_unreadable"] == 1
    assert len(fills) == 1 and fills[0]["price"] is None and fills[0]["is_taker"] is True
    assert quotes == [] and not _creates(fake)


class _ReadbackDown(FakeKalshi):
    """The resting-orders read after placing (the second one this run) fails."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.resting_reads = 0

    def get_orders(self, event_ticker=None, status=None):
        if status == "resting":
            self.resting_reads += 1
            if self.resting_reads == 2:
                from kalshi_bot.client import KalshiAPIError
                raise KalshiAPIError(429, '{"error":"rate limited KXATPMATCH-26OCT29DUCSON"}')
        return super().get_orders(event_ticker=event_ticker, status=status)


def test_expiry_readback_failure_keeps_rows_and_state_and_is_counted():
    markets = make_event("KXATPMATCH", "26OCT29", "James Duckworth", "Lorenzo Sonego",
                         ask1=0.45, ask2=0.57, occurrence="2026-10-29T18:00:00Z")
    fake = _ReadbackDown(markets=markets, cash=1000.0)
    state = MakerState()
    quotes, _, counts = _mrun(fake, state=state)
    assert [r["status"] for r in quotes] == ["placed", "placed"]
    assert counts["expiry_check_failed"] == 1 and counts["placed"] == 2
    assert state.first_mid(EV, DUC) == pytest.approx(0.445)


def test_failed_cancel_of_unexpiring_order_counts_expiry_check_failed():
    fake = _mfake()
    fake.honor_expiry = False
    fake.fail_cancel = {"o1"}
    quotes, _, counts = _mrun(fake)
    assert counts["no_expiry"] == 2 and counts["expiry_check_failed"] == 1
    assert any(r["error"].startswith("no_expiry; cancel failed") for r in quotes if r["status"] == "error")


# I3: a confirmed taker fill halts the maker (persisted) until the flag is removed by hand
def test_taker_fill_halts_quoting_until_flag_removed_by_hand():
    fake = _mfake()
    state, seen = MakerState(), set()
    first, _, _ = _mrun(fake, state=state, seen=seen)
    fake.fill_resting(first[0]["order_id"], 5, ts=RUN_NOW + pd.Timedelta(minutes=1), is_taker=True)
    fake.calls.clear()
    quotes, fills, counts = _mrun(fake, now=LATER, state=state, seen=seen)
    assert fills[0]["is_taker"] is True and counts["taker_fill"] == 1
    assert quotes == [] and not _creates(fake) and counts["halted"] == 1
    assert len([c for c in fake.calls if c[0] == "cancel"]) == 1   # housekeeping still cancels
    assert state.halted and state.halted_at == LATER
    blob = state.to_json()
    back = MakerState.from_json(blob)
    assert back.halted and back.to_json() == blob
    quotes, _, counts = _mrun(fake, now=LATER + pd.Timedelta(minutes=15), state=back, seen=seen)
    assert quotes == [] and not _creates(fake) and counts["halted"] == 1
    doc = json.loads(blob)
    del doc["halted"]                                                # removed by hand
    cleared = MakerState.from_json(json.dumps(doc))
    assert not cleared.halted
    quotes, _, _ = _mrun(fake, now=LATER + pd.Timedelta(minutes=30), state=cleared, seen=seen)
    assert quotes and {r["status"] for r in quotes} == {"placed"}


def test_plan_quotes_places_nothing_while_halted():
    state = MakerState()
    state.halt(NOW)
    quotes, skips, _ = _plan([_ev("E1"), _ev("E2")], state=state)
    assert quotes == [] and skips == {"halted": 1}


# I6: an order resting without an expiry is cancelled at once and logged
def test_order_without_expiry_is_cancelled_and_logged_no_expiry():
    fake = _mfake()
    fake.honor_expiry = False
    quotes, _, counts = _mrun(fake)
    placed_ids = {r["order_id"] for r in quotes if r["status"] == "placed"}
    assert len(placed_ids) == 2 and not fake.get_orders(status="resting")
    errs = [r for r in quotes if r["status"] == "error"]
    assert {r["order_id"] for r in errs} == placed_ids
    assert all(r["error"] == "no_expiry" and set(r) == set(QUOTE_COLUMNS) for r in errs)
    assert counts["no_expiry"] == 2 and counts["placed"] == 2 and counts["cancelled"] == 2


@pytest.mark.parametrize("field", ["expiration_time", "expiration_ts"])
def test_orders_with_expiry_are_left_resting(field):
    fake = _mfake()
    fake.expiry_field = field
    quotes, _, counts = _mrun(fake)
    assert len(fake.get_orders(status="resting")) == 2 and counts["no_expiry"] == 0
    assert {r["status"] for r in quotes} == {"placed"}


# minors: non-finite positions and ticker-less cancel failures fail closed
@pytest.mark.parametrize("pos", ["nan", "inf", "-inf"])
def test_non_finite_position_fails_closed(pos):
    fake = _mfake(positions=[{"ticker": DUC, "position_fp": pos}])
    quotes, _, counts = _mrun(fake)
    assert [r["market_ticker"] for r in quotes] == [SON]
    assert counts["position_cap"] == 1


def test_cancel_failure_on_order_without_ticker_blocks_every_market():
    fake = _mfake(orders=[{"order_id": "x1", "client_order_id": "atpedge-mk-zzz-1",
                           "status": "resting", "fill_count_fp": "0.00"}], fail_cancel={"x1"})
    quotes, _, counts = _mrun(fake)
    assert not _creates(fake)
    assert len(quotes) == 2 and {r["status"] for r in quotes} == {"cancel_failed"}
    assert all(r["error"] for r in quotes) and counts["cancel_failed"] == 2
