"""Bot configuration. Credentials and switches come only from the environment."""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path

# ── Kalshi API ───────────────────────────────────────────────────────────────
# Real money moves only when KALSHI_ENV=prod AND BETTING_MODE=live.
KALSHI_API_KEY_ID = os.environ.get("KALSHI_API_KEY_ID", "")
KALSHI_PRIVATE_KEY_PATH = os.environ.get("KALSHI_PRIVATE_KEY_PATH", "")
KALSHI_ENV = os.environ.get("KALSHI_ENV", "demo").lower()
KALSHI_BASE_URLS = {
    "prod": "https://api.elections.kalshi.com",
    "demo": "https://demo-api.kalshi.co",
}
KALSHI_BASE_URL = KALSHI_BASE_URLS.get(KALSHI_ENV, KALSHI_BASE_URLS["demo"])
KALSHI_API_PREFIX = "/trade-api/v2"
KALSHI_MATCH_SERIES = {"tour": "KXATPMATCH", "challenger": "KXATPCHALLENGERMATCH"}
KALSHI_FEE_COEF = 0.07
# Kalshi trades tennis on exchange shard 3 (shared with baseball/basketball).
# Orders only draw on cash preallocated to that shard, so the bot's cash cap is
# that shard's balance, not the whole account.
KALSHI_TENNIS_EXCHANGE_INDEX = 3

BETTING_MODE = os.environ.get("BETTING_MODE", "dry_run").lower()  # off | dry_run | live
def _float_env(name: str) -> float:
    """0 when unset or malformed — never raise, since a traceback would print the value."""
    try:
        return float(os.environ.get(name, "") or 0)
    except ValueError:
        return 0.0


KALSHI_START_BANKROLL = _float_env("KALSHI_START_BANKROLL")

# ── Sizing and filters ───────────────────────────────────────────────────────
KALSHI_KELLY_FRACTION = 0.25
KALSHI_CHALLENGER_MULT = 0.5
KALSHI_BET_CAP = 0.05          # of bankroll, per bet
KALSHI_RUN_EXPOSURE_CAP = 0.25 # of bankroll, per run
KALSHI_MIN_NET_EDGE = 0.03
KALSHI_START_BUFFER_MINUTES = 5  # never bet within this long of scheduled start
# In-play protection: skip a play if the Kalshi ask has diverged from the
# sportsbook market_prob captured when the prediction was made by more than this.
KALSHI_MAX_PRICE_DIVERGENCE = 0.15
KALSHI_ORDER_PREFIX = "atpedge-"

# ── Maker policy (post short-lived bids at the best bid on both markets) ─────
# Scheduled switch to the maker policy (Noah, 2026-10-29: record-only until then).
# An explicit BOT_POLICY (repo variable or manual input) always wins; when it is
# blank the bot is taker before MAKER_SWITCH_AT and maker from then on. An
# unreadable MAKER_SWITCH_AT (e.g. "never") keeps the taker.
DEFAULT_MAKER_SWITCH_AT = "2026-11-02T14:00:00Z"


def resolve_policy(explicit: str, switch_at: str, now: datetime) -> str:
    explicit = (explicit or "").strip().lower()
    if explicit:
        return explicit
    try:
        at = datetime.fromisoformat(str(switch_at).strip().replace("Z", "+00:00"))
    except ValueError:
        return "taker"
    if at.tzinfo is None:
        at = at.replace(tzinfo=timezone.utc)
    return "maker" if now >= at else "taker"


MAKER_SWITCH_AT = os.environ.get("MAKER_SWITCH_AT", "") or DEFAULT_MAKER_SWITCH_AT
BOT_POLICY = resolve_policy(os.environ.get("BOT_POLICY", ""), MAKER_SWITCH_AT,
                            datetime.now(timezone.utc))  # taker | maker
MAKER_STOP_HOURS = _float_env("MAKER_STOP_HOURS") or 4.0  # stop quoting this long before start
MAKER_RETIRE_MOVE = 0.10   # retire an event once either side's mid moves this far from its first mid
MAKER_MAX_SPREAD = 0.10
MAKER_MIN_BID = 0.05
MAKER_MAX_ASK = 0.95
MAKER_ORDER_COUNT = 5
MAKER_MAX_POSITION = 10
MAKER_RESTING_BUDGET = 0.40
MAKER_ORDER_TTL_MIN = 14
MAKER_ORDER_PREFIX = KALSHI_ORDER_PREFIX + "mk-"

# ── State (a dedicated private repo, via the GitHub contents API) ────────────
STATE_REPO = os.environ.get("STATE_REPO", "")
STATE_BRANCH = os.environ.get("STATE_BRANCH", "main")
BOT_STATE_TOKEN = os.environ.get("BOT_STATE_TOKEN", "")
PLAYS_MAX_AGE_HOURS = 30  # older plays.json → bet nothing (pipeline likely broken)

STATE_DIR = Path(os.environ.get("BOT_STATE_DIR", "state"))
KALSHI_BETS_PATH = STATE_DIR / "kalshi_bets.parquet"
KALSHI_BANKROLL_PATH = STATE_DIR / "kalshi_bankroll.json"
