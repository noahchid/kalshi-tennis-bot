import pandas as pd

from kalshi_bot.matching import find_market, index_open_markets
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
