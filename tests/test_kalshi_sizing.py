import pytest

from kalshi_bot.sizing import (
    all_in_cost, contracts_for, kalshi_fee, kelly_fraction, net_edge, target_stake,
)


def test_fee_rounds_up_to_cent():
    assert kalshi_fee(0.50) == pytest.approx(0.02)   # 1.75c -> 2c
    assert kalshi_fee(0.10) == pytest.approx(0.01)   # 0.63c -> 1c
    assert kalshi_fee(0.99) == pytest.approx(0.01)


def test_net_edge_includes_fee():
    assert net_edge(0.60, 0.50) == pytest.approx(0.08)


def test_kelly_binary_contract():
    # cost 0.52, p 0.60 -> (0.60-0.52)/(1-0.52)
    assert kelly_fraction(0.60, 0.52) == pytest.approx(0.08 / 0.48)
    assert kelly_fraction(0.40, 0.52) == 0.0
    assert kelly_fraction(0.60, 1.0) == 0.0


def test_target_stake_quarter_kelly_and_challenger_half():
    full = kelly_fraction(0.60, all_in_cost(0.50))
    atp = target_stake(1000, 0.60, 0.50, is_challenger=False)
    chal = target_stake(1000, 0.60, 0.50, is_challenger=True)
    assert atp == pytest.approx(1000 * 0.25 * full)
    assert chal == pytest.approx(atp / 2)


def test_target_stake_capped_at_5pct():
    assert target_stake(1000, 0.95, 0.30, is_challenger=False) == pytest.approx(50.0)


def test_contracts_floor_on_all_in_cost():
    assert contracts_for(10.0, 0.50) == 19   # 10 / 0.52
    assert contracts_for(0.40, 0.50) == 0


def test_contracts_zero_when_ask_zero():
    assert contracts_for(10.0, 0.0) == 0
