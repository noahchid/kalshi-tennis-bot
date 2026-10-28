import math

import pandas as pd
import pytest

from kalshi_bot.maker import MakerState, Quote, plan_quotes
from kalshi_bot.matching import Book, EventBooks

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
