import pandas as pd
import pytest

from kalshi_bot.matching import find_event, find_market, index_open_markets
from tests.kalshi_fakes import make_event, make_market


def _index():
    return index_open_markets({
        "challenger": make_event("KXATPCHALLENGERMATCH", "26SEP23", "Keegan Smith", "Martin Damm",
                                 ask1=0.47, ask2=0.54, occurrence="2026-09-24T00:30:00Z"),
        "tour": make_event("KXATPMATCH", "26SEP23", "James Duckworth", "Lorenzo Sonego", ask1=0.45, ask2=0.57),
    })


def test_finds_yes_market_for_player_side():
    m = find_market("Lorenzo Sonego", "James Duckworth", pd.Timestamp("2026-09-23"), _index())
    assert m.market_ticker.endswith("-SON")
    assert m.yes_ask == 0.57
    assert m.level == "tour"


def test_occurrence_time_parsed():
    m = find_market("Keegan Smith", "Martin Damm", pd.Timestamp("2026-09-23"), _index())
    assert m.occurrence_time == pd.Timestamp("2026-09-24T00:30:00Z")
    assert m.level == "challenger"


def test_no_market_for_unlisted_match():
    assert find_market("Ann Aa", "Bob Bb", pd.Timestamp("2026-09-23"), _index()) is None


def test_date_window_excludes_old_rematch():
    assert find_market("Keegan Smith", "Martin Damm", pd.Timestamp("2026-09-10"), _index()) is None


def test_partial_name_match_rejected():
    """Only one player name maps correctly; the other doesn't. Should reject."""
    index = index_open_markets({
        "tour": make_event("KXATPMATCH", "26SEP23", "Lorenzo Sonego", "Lorenzo Musetti", ask1=0.45, ask2=0.57),
    })
    # Try to find market for Lorenzo Sonego vs James Duckworth
    # James Duckworth is not in this event, so mapping will be partial
    assert find_market("Lorenzo Sonego", "James Duckworth", pd.Timestamp("2026-09-23"), index) is None


def test_both_markets_collapsing_to_one_candidate_rejected():
    """Both markets map to the same candidate. Should reject."""
    # Build event with two markets that both map to "Jannik Sinner"
    markets = [
        make_market("KXATPMATCH-26SEP23SINSIN", "SIN", "Jannik Sinner", ask=0.45),
        make_market("KXATPMATCH-26SEP23SINSIN", "SI2", "J. Sinner", ask=0.57),
    ]
    index = index_open_markets({"tour": markets})
    # Try to find market for Jannik Sinner vs Carlos Alcaraz
    # Both markets collapse to Jannik Sinner, so set(mapped) = {Jannik Sinner} != {Jannik Sinner, Carlos Alcaraz}
    assert find_market("Jannik Sinner", "Carlos Alcaraz", pd.Timestamp("2026-09-23"), index) is None


def test_find_event_returns_both_books_with_roles():
    ev = find_event("Lorenzo Sonego", "James Duckworth", pd.Timestamp("2026-09-23"), _index())
    assert ev.level == "tour"
    assert ev.event_ticker.startswith("KXATPMATCH-26SEP23")
    pick, other = ev.books
    assert (pick.role, other.role) == ("pick", "other")
    assert pick.market_ticker.endswith("-SON")
    assert other.market_ticker.endswith("-DUC")


def test_find_event_roles_follow_the_play_not_listing_order():
    ev = find_event("James Duckworth", "Lorenzo Sonego", pd.Timestamp("2026-09-23"), _index())
    pick, other = ev.books
    assert pick.market_ticker.endswith("-DUC") and other.market_ticker.endswith("-SON")


def test_find_event_parses_bids_asks_and_sizes():
    index = index_open_markets({
        "tour": make_event("KXATPMATCH", "26SEP23", "James Duckworth", "Lorenzo Sonego",
                           ask1=0.45, ask2=0.57, bid1=0.42, bid2=0.53,
                           occurrence="2026-09-24T00:30:00Z"),
    })
    ev = find_event("Lorenzo Sonego", "James Duckworth", pd.Timestamp("2026-09-23"), index)
    pick, other = ev.books
    assert (pick.yes_bid, pick.yes_ask) == (0.53, 0.57)
    assert (other.yes_bid, other.yes_ask) == (0.42, 0.45)
    assert pick.yes_bid_size == 1000.0 and pick.yes_ask_size == 1000.0
    assert ev.occurrence_time == pd.Timestamp("2026-09-24T00:30:00Z")


def test_find_event_default_bid_is_ask_minus_one_cent():
    ev = find_event("Lorenzo Sonego", "James Duckworth", pd.Timestamp("2026-09-23"), _index())
    assert abs(ev.books[0].yes_bid - 0.56) < 1e-9


def test_find_event_unparseable_bid_is_nan():
    markets = make_event("KXATPMATCH", "26SEP23", "James Duckworth", "Lorenzo Sonego")
    markets[1]["yes_bid_dollars"] = ""
    ev = find_event("Lorenzo Sonego", "James Duckworth", pd.Timestamp("2026-09-23"),
                    index_open_markets({"tour": markets}))
    assert ev.books[0].yes_bid != ev.books[0].yes_bid  # NaN


def test_find_event_single_market_event_is_none():
    markets = make_event("KXATPMATCH", "26SEP23", "James Duckworth", "Lorenzo Sonego")[:1]
    index = index_open_markets({"tour": markets})
    assert find_event("James Duckworth", "Lorenzo Sonego", pd.Timestamp("2026-09-23"), index) is None


def test_find_event_unlisted_match_and_old_rematch_none():
    assert find_event("Ann Aa", "Bob Bb", pd.Timestamp("2026-09-23"), _index()) is None
    assert find_event("Keegan Smith", "Martin Damm", pd.Timestamp("2026-09-10"), _index()) is None


def test_find_event_names_mapping_to_one_market_is_none():
    markets = [
        make_market("KXATPMATCH-26SEP23SINSIN", "SIN", "Jannik Sinner", ask=0.45),
        make_market("KXATPMATCH-26SEP23SINSIN", "SI2", "J. Sinner", ask=0.57),
    ]
    index = index_open_markets({"tour": markets})
    assert find_event("Jannik Sinner", "Carlos Alcaraz", pd.Timestamp("2026-09-23"), index) is None


def test_find_event_partial_name_match_is_none():
    index = index_open_markets({
        "tour": make_event("KXATPMATCH", "26SEP23", "Lorenzo Sonego", "Lorenzo Musetti"),
    })
    assert find_event("Lorenzo Sonego", "James Duckworth", pd.Timestamp("2026-09-23"), index) is None


def _parity_cases():
    sin_dup = [
        make_market("KXATPMATCH-26SEP23SINSIN", "SIN", "Jannik Sinner", ask=0.45),
        make_market("KXATPMATCH-26SEP23SINSIN", "SI2", "J. Sinner", ask=0.57),
    ]
    one_market = make_event("KXATPMATCH", "26SEP23", "James Duckworth", "Lorenzo Sonego")[:1]
    partial = make_event("KXATPMATCH", "26SEP23", "Lorenzo Sonego", "Lorenzo Musetti")
    d = pd.Timestamp("2026-09-23")
    return [
        ("in_window_tour", "Lorenzo Sonego", "James Duckworth", d, _index()),
        ("in_window_challenger", "Keegan Smith", "Martin Damm", d, _index()),
        ("out_of_window_rematch", "Keegan Smith", "Martin Damm", pd.Timestamp("2026-09-10"), _index()),
        ("after_window", "Keegan Smith", "Martin Damm", pd.Timestamp("2026-09-30"), _index()),
        ("partial_name", "Lorenzo Sonego", "James Duckworth", d, index_open_markets({"tour": partial})),
        ("both_names_one_market", "Jannik Sinner", "Carlos Alcaraz", d, index_open_markets({"tour": sin_dup})),
        ("single_market_event", "James Duckworth", "Lorenzo Sonego", d, index_open_markets({"tour": one_market})),
        ("unlisted", "Ann Aa", "Bob Bb", d, _index()),
        # play lists the opponent's market first in the event: pick must still be the play's player
        ("opponent_listed_first", "Lorenzo Sonego", "James Duckworth", d, _index()),
        ("play_player_listed_first", "James Duckworth", "Lorenzo Sonego", d, _index()),
    ]


@pytest.mark.parametrize("case", _parity_cases(), ids=lambda c: c[0])
def test_find_event_agrees_with_find_market(case):
    _, player, opponent, date, index = case
    m = find_market(player, opponent, date, index)
    ev = find_event(player, opponent, date, index)
    if m is None:
        assert ev is None
        return
    assert ev is not None
    pick, other = ev.books
    assert pick.role == "pick" and other.role == "other"
    assert pick.market_ticker == m.market_ticker
    assert other.market_ticker != m.market_ticker
    assert ev.event_ticker == m.event_ticker
    assert ev.occurrence_time == m.occurrence_time
    assert ev.level == m.level
    assert pick.yes_ask == m.yes_ask and pick.yes_ask_size == m.yes_ask_size
