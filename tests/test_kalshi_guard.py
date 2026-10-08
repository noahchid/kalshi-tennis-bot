import pandas as pd

from kalshi_bot.bet_log import (
    append_bet_log, load_bet_log, mark_emailed, pending_email_bets,
)
from kalshi_bot.guard import client_order_id, duplicate_reason
from tests.kalshi_fakes import FakeKalshi

EV = "KXATPMATCH-26SEP23DUCSON"


def test_client_order_id_deterministic_and_prefixed():
    a = client_order_id(EV, EV + "-SON", "100")
    assert a == client_order_id(EV, EV + "-SON", "100")
    assert a != client_order_id(EV, EV + "-SON", "200")  # Kalshi rejects reused ids
    assert a[:-len("100")] == client_order_id(EV, EV + "-SON", "200")[:-len("200")]
    assert a.startswith("atpedge-") and len(a) <= 64
    assert a != client_order_id(EV, EV + "-DUC", "100")


def test_clean_event_passes():
    assert duplicate_reason(EV, load_bet_log_empty(), FakeKalshi()) is None


def load_bet_log_empty():
    return pd.DataFrame(columns=["event_ticker", "status"])


def test_bet_log_hit_blocks():
    log = pd.DataFrame([{"event_ticker": EV, "status": "filled"}])
    assert duplicate_reason(EV, log, FakeKalshi()) == "bet_log"


def test_dry_run_rows_do_not_block():
    log = pd.DataFrame([{"event_ticker": EV, "status": "dry_run"}])
    assert duplicate_reason(EV, log, FakeKalshi()) is None


def test_kalshi_position_blocks_even_if_log_lost():
    fake = FakeKalshi(positions=[{"ticker": EV + "-SON", "position_fp": "10.00"}])
    assert duplicate_reason(EV, load_bet_log_empty(), fake) == "kalshi_position"


def test_resting_order_blocks():
    fake = FakeKalshi(orders=[{"ticker": EV + "-SON", "status": "resting", "client_order_id": "x",
                               "fill_count_fp": "0.00"}])
    assert duplicate_reason(EV, load_bet_log_empty(), fake) == "kalshi_resting_order"


def test_prior_bot_fill_blocks_after_position_settled():
    fake = FakeKalshi(orders=[{"ticker": EV + "-SON", "status": "executed",
                               "client_order_id": "atpedge-0123456789abcdef", "fill_count_fp": "5.00"}])
    assert duplicate_reason(EV, load_bet_log_empty(), fake) == "kalshi_bot_order"


def test_demo_bet_log_row_does_not_block_prod():
    log = pd.DataFrame([{"event_ticker": EV, "status": "filled", "kalshi_env": "demo"}])
    assert duplicate_reason(EV, log, FakeKalshi(), env="prod") is None


def test_same_env_bet_log_row_blocks():
    log = pd.DataFrame([{"event_ticker": EV, "status": "filled", "kalshi_env": "prod"}])
    assert duplicate_reason(EV, log, FakeKalshi(), env="prod") == "bet_log"


def test_position_legacy_field_fallback_blocks():
    fake = FakeKalshi(positions=[{"ticker": EV + "-SON", "position": "10.00"}])
    assert duplicate_reason(EV, load_bet_log_empty(), fake) == "kalshi_position"


def test_position_neither_field_is_unverifiable():
    fake = FakeKalshi(positions=[{"ticker": EV + "-SON"}])
    assert duplicate_reason(EV, load_bet_log_empty(), fake) == "guard_unverifiable"


def test_fill_count_legacy_field_fallback_blocks():
    fake = FakeKalshi(orders=[{"ticker": EV + "-SON", "status": "executed",
                               "client_order_id": "atpedge-0123456789abcdef", "fill_count": "5.00"}])
    assert duplicate_reason(EV, load_bet_log_empty(), fake) == "kalshi_bot_order"


def test_fill_count_neither_field_is_unverifiable():
    fake = FakeKalshi(orders=[{"ticker": EV + "-SON", "status": "executed",
                               "client_order_id": "atpedge-0123456789abcdef"}])
    assert duplicate_reason(EV, load_bet_log_empty(), fake) == "guard_unverifiable"


def test_bet_log_roundtrip_and_email_marking(tmp_path):
    p = tmp_path / "bets.parquet"
    append_bet_log([{"event_ticker": EV, "status": "filled", "contracts": 5}], p)
    append_bet_log([{"event_ticker": "X", "status": "skipped", "skip_reason": "no_market"}], p)
    assert len(load_bet_log(p)) == 2
    assert list(pending_email_bets(p).event_ticker) == [EV]
    mark_emailed(p)
    assert pending_email_bets(p).empty


def test_dry_run_rows_block_only_in_dry_run_mode():
    log = pd.DataFrame([{"event_ticker": EV, "status": "dry_run", "kalshi_env": "prod"}])
    assert duplicate_reason(EV, log, FakeKalshi(), env="prod", include_dry_run=True) == "bet_log"
    assert duplicate_reason(EV, log, FakeKalshi(), env="prod") is None
