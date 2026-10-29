from datetime import datetime, timezone

from kalshi_bot.config import resolve_policy

SWITCH = "2026-11-02T14:00:00Z"


def _t(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def test_explicit_policy_always_wins():
    assert resolve_policy("taker", SWITCH, _t("2026-12-01T00:00:00Z")) == "taker"
    assert resolve_policy("MAKER", SWITCH, _t("2026-10-30T00:00:00Z")) == "maker"


def test_unset_policy_switches_to_maker_at_the_scheduled_time():
    assert resolve_policy("", SWITCH, _t("2026-11-02T13:59:59Z")) == "taker"
    assert resolve_policy("", SWITCH, _t("2026-11-02T14:00:00Z")) == "maker"
    assert resolve_policy("  ", SWITCH, _t("2026-11-03T00:00:00Z")) == "maker"


def test_unreadable_switch_time_fails_safe_to_taker():
    for bad in ("never", "2026-13-45", ""):
        assert resolve_policy("", bad, _t("2030-01-01T00:00:00Z")) == "taker"
