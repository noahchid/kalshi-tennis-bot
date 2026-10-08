"""Fee, edge and Kelly sizing for Kalshi binary contracts (pure functions).

A YES contract costs ``ask`` plus a per-contract fee and pays $1 if it wins.
For win probability p and all-in cost c, full Kelly is (p - c) / (1 - c).
Prices are expected in (0, 1).
"""

from __future__ import annotations

import math

from kalshi_bot.config import (
    KALSHI_BET_CAP,
    KALSHI_CHALLENGER_MULT,
    KALSHI_FEE_COEF,
    KALSHI_KELLY_FRACTION,
)


def kalshi_fee(price: float, coef: float = KALSHI_FEE_COEF) -> float:
    cents = 100 * coef * price * (1 - price)
    return math.ceil(round(cents, 6)) / 100


def all_in_cost(ask: float) -> float:
    return ask + kalshi_fee(ask)


def net_edge(model_prob: float, ask: float) -> float:
    return model_prob - all_in_cost(ask)


def kelly_fraction(model_prob: float, cost: float) -> float:
    if cost >= 1 or model_prob <= cost:
        return 0.0
    return (model_prob - cost) / (1 - cost)


def target_stake(bankroll: float, model_prob: float, ask: float, is_challenger: bool) -> float:
    mult = KALSHI_KELLY_FRACTION * (KALSHI_CHALLENGER_MULT if is_challenger else 1.0)
    frac = min(mult * kelly_fraction(model_prob, all_in_cost(ask)), KALSHI_BET_CAP)
    return max(bankroll, 0.0) * frac


def contracts_for(stake: float, ask: float) -> int:
    cost = all_in_cost(ask)
    if cost <= 0:
        return 0
    return int(math.floor(stake / cost + 1e-9))
