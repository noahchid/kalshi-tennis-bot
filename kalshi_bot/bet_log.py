"""Append-only log of every bettor decision, plus the latest bankroll snapshot."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from kalshi_bot.config import KALSHI_BANKROLL_PATH, KALSHI_BETS_PATH

BET_COLUMNS = [
    "logged_at", "run_mode", "kalshi_env", "pred_date", "player_name", "opponent_name",
    "tournament", "tournament_level", "is_challenger", "event_ticker", "market_ticker",
    "model_prob", "yes_ask", "fee", "net_edge", "kelly", "bankroll", "stake_target",
    "contracts", "client_order_id", "status", "skip_reason", "order_id", "fill_count",
    "avg_fill_price", "avg_fee", "error", "emailed",
]
PLACED_STATUSES = {"filled", "partial"}
EMAIL_STATUSES = {"filled", "partial", "dry_run"}
# What the taker would have bet while BOT_POLICY=maker: logged, never sent, never emailed.
SHADOW_STATUSES = {"shadow"}


def load_bet_log(path: Path = KALSHI_BETS_PATH) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        return pd.DataFrame(columns=BET_COLUMNS)
    return pd.read_parquet(path)


def append_bet_log(rows: list[dict], path: Path = KALSHI_BETS_PATH) -> None:
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    new = pd.DataFrame(rows)
    for col in BET_COLUMNS:
        if col not in new.columns:
            new[col] = False if col == "emailed" else None
    new["emailed"] = new["emailed"].fillna(False).astype(bool)
    log = load_bet_log(path)
    out = pd.concat([log, new[BET_COLUMNS]], ignore_index=True) if not log.empty else new[BET_COLUMNS]
    out.astype({"emailed": bool}).to_parquet(path, index=False)


def pending_email_bets(path: Path = KALSHI_BETS_PATH) -> pd.DataFrame:
    log = load_bet_log(path)
    if log.empty:
        return log
    return log[log["status"].isin(EMAIL_STATUSES) & ~log["emailed"].fillna(False).astype(bool)]


def mark_emailed(path: Path = KALSHI_BETS_PATH) -> None:
    log = load_bet_log(path)
    if log.empty:
        return
    log.loc[log["status"].isin(EMAIL_STATUSES), "emailed"] = True
    log.astype({"emailed": bool}).to_parquet(path, index=False)


def save_bankroll_snapshot(state: dict, path: Path = KALSHI_BANKROLL_PATH) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2, default=str))


def load_bankroll_snapshot(path: Path = KALSHI_BANKROLL_PATH) -> dict | None:
    path = Path(path)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
