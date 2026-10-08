"""Shared helpers for reading Kalshi API fields with legacy fallbacks.

Kalshi's API has, at various points, used both a fixed-point field (e.g.
`position_fp`) and a plain-integer legacy field (e.g. `position`) for the
same value. These helpers read the fixed-point field first and fall back to
the legacy field, returning None only when *neither* is present so callers
can fail closed instead of silently treating a missing field as zero.
"""

from __future__ import annotations


def get_field(obj: dict, new_key: str, old_key: str):
    """Read `new_key`, falling back to `old_key`. None if neither is set."""
    value = obj.get(new_key)
    if value is not None:
        return value
    value = obj.get(old_key)
    if value is not None:
        return value
    return None


def get_position_fp(position: dict):
    """Return a position object's fixed-point size, or None if unverifiable."""
    return get_field(position, "position_fp", "position")


def get_fill_count_fp(order: dict):
    """Return an order object's fixed-point fill count, or None if unverifiable."""
    return get_field(order, "fill_count_fp", "fill_count")


def get_market_exposure_dollars(position: dict) -> float:
    """Return a position's market exposure in dollars.

    Falls back to the legacy `market_exposure` field, which is denominated
    in cents.
    """
    value = position.get("market_exposure_dollars")
    if value is not None:
        return float(value)
    value = position.get("market_exposure")
    if value is not None:
        return float(value) / 100.0
    return 0.0
