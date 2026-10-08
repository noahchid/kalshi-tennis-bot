import pytest

from kalshi_bot.bankroll import compute_bankroll, fetch_bankroll
from tests.kalshi_fakes import FakeKalshi

BOT = {"ticker": "EV1-A", "client_order_id": "atpedge-aaaaaaaaaaaaaaaa", "status": "executed", "fill_count_fp": "10.00"}
BOT_OPEN = {"ticker": "EV2-B", "client_order_id": "atpedge-bbbbbbbbbbbbbbbb", "status": "executed", "fill_count_fp": "5.00"}
MANUAL = {"ticker": "OTHER-X", "client_order_id": "", "status": "executed", "fill_count_fp": "100.00"}

SETTLED_WIN = {"ticker": "EV1-A", "revenue": 1000, "yes_total_cost_dollars": "5.00",
               "no_total_cost_dollars": "0", "fee_cost": "0.20"}
SETTLED_MANUAL = {"ticker": "OTHER-X", "revenue": 0, "yes_total_cost_dollars": "60.00",
                  "no_total_cost_dollars": "0", "fee_cost": "1.00"}


def test_only_bot_tickers_count():
    s = compute_bankroll(
        200.0, [BOT, BOT_OPEN, MANUAL], [SETTLED_WIN, SETTLED_MANUAL],
        [{"ticker": "EV2-B", "position_fp": "5.00", "market_exposure_dollars": "2.60"},
         {"ticker": "OTHER-X", "position_fp": "3.00", "market_exposure_dollars": "9.00"}],
        cash=1000.0,
    )
    assert s.realized_pnl == pytest.approx(10.00 - 5.00 - 0.20)
    assert s.open_cost == pytest.approx(2.60)
    assert s.bankroll == pytest.approx(200 + 4.80 - 2.60)


def test_capped_at_cash():
    s = compute_bankroll(500.0, [], [], [], cash=120.0)
    assert s.bankroll == 120.0


def test_never_negative():
    s = compute_bankroll(10.0, [BOT], [dict(SETTLED_WIN, revenue=0, yes_total_cost_dollars="50")], [], cash=100.0)
    assert s.bankroll == 0.0


def test_fetch_uses_cash_when_no_start():
    s = fetch_bankroll(FakeKalshi(cash=321.0), start_bankroll=0)
    assert s.start == 321.0 and s.bankroll == 321.0


def test_legacy_fill_count_field_counts_as_bot_ticker():
    bot_legacy = {"ticker": "EV3-C", "client_order_id": "atpedge-cccccccccccccccc",
                  "status": "executed", "fill_count": "10.00"}
    settled = {"ticker": "EV3-C", "revenue": 1000, "yes_total_cost_dollars": "5.00",
              "no_total_cost_dollars": "0", "fee_cost": "0.20"}
    s = compute_bankroll(200.0, [bot_legacy], [settled], [], cash=1000.0)
    assert s.realized_pnl == pytest.approx(10.00 - 5.00 - 0.20)


def test_legacy_position_and_market_exposure_fields_used():
    s = compute_bankroll(
        200.0, [BOT_OPEN], [],
        [{"ticker": "EV2-B", "position": "5.00", "market_exposure": "260"}],  # cents
        cash=1000.0,
    )
    assert s.open_cost == pytest.approx(2.60)
    assert s.bankroll == pytest.approx(200 - 2.60)


def test_fetch_reads_tennis_shard_balance():
    class ShardFake(FakeKalshi):
        def get_balance(self, exchange_index=None):
            self.asked_index = exchange_index
            return 40.0 if exchange_index == 3 else 900.0

    fake = ShardFake()
    s = fetch_bankroll(fake, start_bankroll=100.0)
    assert fake.asked_index == 3
    assert s.cash == 40.0 and s.bankroll == 40.0


def test_bot_tickers_also_come_from_bet_log():
    import pandas as pd
    log = pd.DataFrame([
        {"market_ticker": "EV9-Z", "status": "filled", "kalshi_env": "prod"},
        {"market_ticker": "EV8-Y", "status": "filled", "kalshi_env": "demo"},
    ])
    fake = FakeKalshi(cash=1000.0, settlements=[
        {"ticker": "EV9-Z", "revenue": 1000, "yes_total_cost_dollars": "4.00", "fee_cost": "0.10"},
        {"ticker": "EV8-Y", "revenue": 1000, "yes_total_cost_dollars": "4.00", "fee_cost": "0.10"},
    ])
    s = fetch_bankroll(fake, start_bankroll=100.0, bet_log=log, env="prod")
    assert s.realized_pnl == pytest.approx(5.90)
