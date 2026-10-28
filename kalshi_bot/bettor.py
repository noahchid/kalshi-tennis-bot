"""Kalshi auto-bettor: bet each model play once, fractional-Kelly sized.

Run on a schedule:  python3 -m kalshi_bot.bettor [--mode ...]
Plays come from plays.json on the private state branch; the bet log and
bankroll are written back there. stdout carries counts only (public logs).
Modes: off (exit), dry_run (price + size + log, send nothing), live (send IOC
limit orders). Real money only with KALSHI_ENV=prod and BETTING_MODE=live.

BOT_POLICY picks who trades: taker (the IOC bettor above; the maker policy runs
in shadow, logging the bids it would post) or maker (kalshi_bot.maker posts the
bids; the taker runs in shadow mode, logging the bets it would place).
"""

from __future__ import annotations

import argparse
from collections import Counter
import io
import json
import math
import sys
from dataclasses import dataclass

import pandas as pd

from kalshi_bot.config import (
    BETTING_MODE,
    BOT_POLICY,
    BOT_STATE_TOKEN,
    KALSHI_BANKROLL_PATH,
    KALSHI_BETS_PATH,
    KALSHI_ENV,
    KALSHI_MATCH_SERIES,
    KALSHI_MAX_PRICE_DIVERGENCE,
    KALSHI_MIN_NET_EDGE,
    KALSHI_RUN_EXPOSURE_CAP,
    KALSHI_START_BANKROLL,
    KALSHI_START_BUFFER_MINUTES,
    PLAYS_MAX_AGE_HOURS,
    STATE_BRANCH,
    STATE_REPO,
)
from kalshi_bot.client import KalshiAPIError, KalshiClient
from kalshi_bot.bankroll import fetch_bankroll
from kalshi_bot.bet_log import SHADOW_STATUSES, append_bet_log, load_bet_log, save_bankroll_snapshot
from kalshi_bot.guard import client_order_id, duplicate_reason
from kalshi_bot.maker import FILL_COLUMNS, QUOTE_COLUMNS, MakerState, run_maker
from kalshi_bot.matching import find_market, index_open_markets
from kalshi_bot.state import StateError, StateStore
from kalshi_bot.sizing import (
    all_in_cost, contracts_for, kalshi_fee, kelly_fraction, net_edge, target_stake,
)


@dataclass(frozen=True)
class Play:
    pred_date: pd.Timestamp
    player_name: str
    opponent_name: str
    model_prob: float
    tournament: str
    tournament_level: str
    market_prob: float | None = None


def load_plays_doc(doc: dict) -> list[Play]:
    """Plays from the pipeline's plays.json (already one per match, has_edge only)."""
    plays = []
    for r in doc.get("plays", []):
        market_prob = r.get("market_prob")
        plays.append(Play(pd.Timestamp(r["pred_date"]), r["player_name"], r["opponent_name"],
                          float(r["model_prob"]), r.get("tournament", ""), str(r.get("tournament_level", "")),
                          float(market_prob) if market_prob is not None else None))
    return plays


def plays_age_hours(doc: dict, now: pd.Timestamp) -> float:
    generated = doc.get("generated_at")
    if not generated:
        return float("inf")
    generated = pd.Timestamp(generated)
    if generated.tzinfo is None:
        generated = generated.tz_localize("UTC")
    return (now - generated).total_seconds() / 3600


def summarize(rows: list[dict]) -> str:
    """Counts by status and skip reason — nothing identifying (logs are public)."""
    statuses = Counter(r["status"] for r in rows)
    reasons = Counter(r.get("skip_reason") for r in rows if r["status"] == "skipped")
    parts = []
    for status, n in sorted(statuses.items()):
        part = f"{status} {n}"
        if status == "skipped" and reasons:
            ordered = sorted(reasons.items(), key=lambda kv: (-kv[1], kv[0]))
            part += " (" + ", ".join(f"{k} {v}" for k, v in ordered) + ")"
        parts.append(part)
    return " · ".join(parts) if parts else "no plays"


def _base_row(play: Play, mode: str, kalshi_env: str, now: pd.Timestamp) -> dict:
    return {
        "logged_at": now, "run_mode": mode, "kalshi_env": kalshi_env,
        "pred_date": play.pred_date, "player_name": play.player_name,
        "opponent_name": play.opponent_name, "tournament": play.tournament,
        "tournament_level": play.tournament_level,
        "is_challenger": play.tournament_level == "C", "model_prob": play.model_prob,
    }


def _shadow_duplicate(event_ticker: str, log: pd.DataFrame, env: str) -> str | None:
    """Shadow mode logs each event once. It checks only its own earlier shadow rows,
    never Kalshi: the maker's atpedge-mk- orders would otherwise hide the event."""
    if log.empty:
        return None
    hit = log[(log["event_ticker"] == event_ticker) & log["status"].isin(SHADOW_STATUSES)]
    if "kalshi_env" in hit.columns:
        hit = hit[hit["kalshi_env"] == env]
    return None if hit.empty else "bet_log"


def run(mode, client, start_bankroll, plays, bets_path, bankroll_path, now=None, kalshi_env=KALSHI_ENV):
    """mode: live (send IOC orders), dry_run, or shadow (like dry_run, but logs status
    "shadow" and dedupes only against earlier shadow rows; used under BOT_POLICY=maker)."""
    if mode not in ("dry_run", "live", "shadow"):
        raise ValueError(f"invalid mode: {mode!r} (must be 'dry_run', 'live' or 'shadow')")

    now = pd.Timestamp.now(tz="UTC") if now is None else pd.Timestamp(now)
    if now.tzinfo is None:
        now = now.tz_localize("UTC")

    log = load_bet_log(bets_path)
    state = fetch_bankroll(client, start_bankroll, bet_log=log, env=kalshi_env)
    save_bankroll_snapshot(dict(state.as_dict(), as_of=str(now), kalshi_env=kalshi_env, mode=mode), bankroll_path)
    remaining = state.bankroll * KALSHI_RUN_EXPOSURE_CAP

    index = index_open_markets({lvl: client.list_markets(series, status="open")
                                for lvl, series in KALSHI_MATCH_SERIES.items()})

    rows, candidates = [], []
    had_error = False
    try:
        for play in plays:
            row = _base_row(play, mode, kalshi_env, now)
            match = find_market(play.player_name, play.opponent_name, play.pred_date, index)
            if match is None:
                rows.append(dict(row, status="skipped", skip_reason="no_market"))
                continue
            row.update(event_ticker=match.event_ticker, market_ticker=match.market_ticker,
                       yes_ask=match.yes_ask,
                       client_order_id=client_order_id(match.event_ticker, match.market_ticker,
                                                       str(int(now.timestamp()))))

            if match.occurrence_time is None:
                rows.append(dict(row, status="skipped", skip_reason="no_start_time"))
                continue
            occurrence_time = match.occurrence_time
            if occurrence_time.tzinfo is None:
                occurrence_time = occurrence_time.tz_localize("UTC")
            buffer = pd.Timedelta(minutes=KALSHI_START_BUFFER_MINUTES)
            if now >= occurrence_time - buffer:
                rows.append(dict(row, status="skipped", skip_reason="started"))
                continue

            # Kalshi reports 0 (or occasionally >=1) on an empty order book; that is
            # not a real, tradeable price, so screen it out BEFORE computing fee/edge
            # (kalshi_fee/net_edge blow up on NaN and are meaningless on 0 or >=1).
            if math.isnan(match.yes_ask) or not (0 < match.yes_ask < 1):
                rows.append(dict(row, status="skipped", skip_reason="no_ask"))
                continue

            if play.market_prob is None:
                rows.append(dict(row, status="skipped", skip_reason="no_market_prob"))
                continue
            if abs(match.yes_ask - play.market_prob) > KALSHI_MAX_PRICE_DIVERGENCE:
                rows.append(dict(row, status="skipped", skip_reason="price_divergence"))
                continue

            fee = kalshi_fee(match.yes_ask)
            edge = net_edge(play.model_prob, match.yes_ask)
            row.update(fee=fee, net_edge=edge)
            if edge < KALSHI_MIN_NET_EDGE:
                rows.append(dict(row, status="skipped", skip_reason="edge_below_min"))
                continue
            candidates.append((row, play, match))

        placed_events: set[str] = set()
        for row, play, match in sorted(candidates, key=lambda c: -c[0]["net_edge"]):
            try:
                if match.event_ticker in placed_events:
                    rows.append(dict(row, status="skipped", skip_reason="bet_log"))
                    continue
                if mode == "shadow":
                    reason = _shadow_duplicate(match.event_ticker, log, kalshi_env)
                else:
                    reason = duplicate_reason(match.event_ticker, log, client, env=kalshi_env,
                                              include_dry_run=(mode == "dry_run"))
                if reason:
                    rows.append(dict(row, status="skipped", skip_reason=reason))
                    continue
                cost = all_in_cost(match.yes_ask)
                stake = min(
                    target_stake(state.bankroll, play.model_prob, match.yes_ask, play.tournament_level == "C"),
                    remaining,
                )
                n = contracts_for(stake, match.yes_ask)
                if not math.isnan(match.yes_ask_size):
                    n = min(n, int(match.yes_ask_size))
                row.update(kelly=kelly_fraction(play.model_prob, cost), bankroll=state.bankroll,
                           stake_target=round(stake, 2), contracts=n)
                if n < 1:
                    rows.append(dict(row, status="skipped", skip_reason="below_min_stake"))
                    continue
                if mode in ("dry_run", "shadow"):
                    remaining -= n * cost
                    placed_events.add(match.event_ticker)
                    rows.append(dict(row, status=mode))
                    continue
                try:
                    resp = client.create_order(match.market_ticker, match.yes_ask, n, row["client_order_id"])
                except KalshiAPIError as e:
                    had_error = True
                    rows.append(dict(row, status="error", error=str(e)))
                    continue

                fill_raw = resp.get("fill_count")
                if fill_raw is None:
                    fill_raw = resp.get("fill_count_fp")
                fill_error = None
                if fill_raw is None:
                    filled = n
                    fill_error = "fill count missing from order response; assumed full fill"
                else:
                    filled = int(float(fill_raw))
                remaining -= filled * cost
                placed_events.add(match.event_ticker)
                status = "filled" if filled == n else "partial" if filled > 0 else "unfilled"
                out = dict(row, status=status, order_id=resp.get("order_id"), fill_count=filled,
                          avg_fill_price=float(resp["average_fill_price"]) if resp.get("average_fill_price") else None,
                          avg_fee=float(resp["average_fee_paid"]) if resp.get("average_fee_paid") else None)
                if fill_error:
                    out["error"] = fill_error
                rows.append(out)
            except Exception as e:
                had_error = True
                rows.append(dict(row, status="error", error=repr(e)))
    finally:
        # Skips are reported as counts only; persisting them every 15 minutes
        # would bloat the state branch.
        append_bet_log([r for r in rows if r["status"] != "skipped"], bets_path)

    return rows, had_error


BETS_FILE = "kalshi_bets.parquet"
BANKROLL_FILE = "kalshi_bankroll.json"
_BANKROLL_KEYS = ("start", "realized_pnl", "open_cost", "cash", "bankroll", "kalshi_env", "mode")


def _bankroll_numbers(raw: bytes | None) -> dict | None:
    if not raw:
        return None
    try:
        snap = json.loads(raw)
    except ValueError:
        return None
    return {k: snap.get(k) for k in _BANKROLL_KEYS}


def _push_bet_log(store, new_rows: pd.DataFrame, sha: str | None) -> None:
    """Push the bet log; on a write conflict, re-read the remote log, re-append this
    run's rows and push once more so fills recorded this run aren't lost."""
    try:
        store.put(BETS_FILE, KALSHI_BETS_PATH.read_bytes(), sha, "bot: bet log")
        return
    except StateError as e:
        if e.status_code not in (409, 422):
            raise
    remote_raw, remote_sha = store.get(BETS_FILE)
    if remote_raw is not None:
        KALSHI_BETS_PATH.write_bytes(remote_raw)
    elif KALSHI_BETS_PATH.exists():
        KALSHI_BETS_PATH.unlink()
    append_bet_log(new_rows.to_dict("records"), KALSHI_BETS_PATH)
    store.put(BETS_FILE, KALSHI_BETS_PATH.read_bytes(), remote_sha, "bot: bet log")


# ── Maker policy state (private state repo) ──────────────────────────────────

QUOTES_FILE = "kalshi_quotes.parquet"
MAKER_FILLS_FILE = "kalshi_maker_fills.parquet"
MAKER_STATE_FILE = "kalshi_maker_state.json"
_MAKER_STATUSES = ("placed", "dry_run", "shadow_quote", "error", "cancel_failed")
_MAKER_HOUSEKEEPING = ("taker_fill", "cancelled", "cancel_errors", "fill_unreadable", "fill_unclassified")


def _read_rows(raw: bytes | None, columns: list[str]) -> pd.DataFrame:
    if not raw:
        return pd.DataFrame(columns=columns)
    return pd.read_parquet(io.BytesIO(raw))


def _append_rows(raw: bytes | None, rows: list[dict], columns: list[str]) -> bytes:
    # Rebuilt from records rather than pd.concat: concat warns (FutureWarning) when a
    # column is all-null on one side, which is routine here (order_id under shadow).
    records = _read_rows(raw, columns).to_dict("records") + [{c: r.get(c) for c in columns} for r in rows]
    buf = io.BytesIO()
    pd.DataFrame.from_records(records, columns=columns).to_parquet(buf, index=False)
    return buf.getvalue()


def _push_appended(store, path: str, raw: bytes | None, sha: str | None, rows: list[dict],
                   columns: list[str], message: str) -> None:
    """Append rows to a parquet file in the state repo; on a write conflict, re-read
    the remote file, re-append and push once more (as _push_bet_log does)."""
    if not rows:
        return
    try:
        store.put(path, _append_rows(raw, rows, columns), sha, message)
        return
    except StateError as e:
        if e.status_code not in (409, 422):
            raise
    raw, sha = store.get(path)
    store.put(path, _append_rows(raw, rows, columns), sha, message)


def _push_replaced(store, path: str, data: bytes, raw: bytes | None, sha: str | None,
                   message: str) -> None:
    """Replace a file when its content changed; on a write conflict, re-read the sha
    and push once more (the bot is the file's only writer, so this run's content wins)."""
    if data == raw:
        return
    try:
        store.put(path, data, sha, message)
        return
    except StateError as e:
        if e.status_code not in (409, 422):
            raise
    _, sha = store.get(path)
    store.put(path, data, sha, message)


def _utc_date(ts):
    ts = pd.Timestamp(ts)
    return (ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")).date()


def _dedupe_shadow_quotes(rows: list[dict], existing_quotes: pd.DataFrame) -> list[dict]:
    """Keep a shadow_quote row only if its market has no shadow_quote row on the same
    UTC date (in the file or earlier in `rows`). Other statuses pass through."""
    seen = set()
    if not existing_quotes.empty:
        prior = existing_quotes[existing_quotes["status"] == "shadow_quote"]
        seen = {(str(m), _utc_date(t)) for m, t in zip(prior["market_ticker"], prior["logged_at"])
                if pd.notna(t)}
    out = []
    for r in rows:
        if r.get("status") == "shadow_quote":
            key = (r["market_ticker"], _utc_date(r["logged_at"]))
            if key in seen:
                continue
            seen.add(key)
        out.append(r)
    return out


def summarize_maker(mode: str, counts: Counter, logged: int) -> str:
    """Counts only (public logs): `maker (live): placed 6, skipped {too_close_to_start: 4}, fills 1`."""
    parts = [f"{s} {counts[s]}" for s in _MAKER_STATUSES if counts[s]] or ["quotes 0"]
    if mode == "shadow":
        parts[-1] += f" (logged {logged})"
    skips = {k: v for k, v in counts.items()
             if v and k not in _MAKER_STATUSES and k not in _MAKER_HOUSEKEEPING and k != "fills"}
    if skips:
        ordered = sorted(skips.items(), key=lambda kv: (-kv[1], kv[0]))
        parts.append("skipped {" + ", ".join(f"{k}: {v}" for k, v in ordered) + "}")
    parts.append(f"fills {counts['fills']}")
    parts += [f"{k} {counts[k]}" for k in _MAKER_HOUSEKEEPING if counts[k]]
    return f"maker ({mode}): " + ", ".join(parts)


def _maker_step(mode: str, client, plays: list[Play], store, now: pd.Timestamp,
                housekeeping: bool) -> None:
    """One maker run plus its state: quotes and fills appended, MakerState replaced.

    fills_seen is rebuilt from the persisted fills file every run (never stored on its
    own), so a run that dies after reading fills loses nothing: the next run re-reads them.
    """
    quotes_raw, quotes_sha = store.get(QUOTES_FILE)
    fills_raw, fills_sha = store.get(MAKER_FILLS_FILE)
    state_raw, state_sha = store.get(MAKER_STATE_FILE)
    state = MakerState.from_json(state_raw)
    fills_seen = set(_read_rows(fills_raw, FILL_COLUMNS)["fill_id"].dropna().astype(str))
    bankroll = fetch_bankroll(client, KALSHI_START_BANKROLL, bet_log=load_bet_log(KALSHI_BETS_PATH),
                              env=KALSHI_ENV)

    quote_rows, fill_rows, counts = run_maker(mode, client, plays, state, fills_seen, bankroll.bankroll,
                                              now, KALSHI_ENV, housekeeping)
    if mode == "shadow":
        quote_rows = _dedupe_shadow_quotes(quote_rows, _read_rows(quotes_raw, QUOTE_COLUMNS))
    print(f"  {summarize_maker(mode, counts, len(quote_rows))}")

    _push_appended(store, MAKER_FILLS_FILE, fills_raw, fills_sha, fill_rows, FILL_COLUMNS, "bot: maker fills")
    _push_appended(store, QUOTES_FILE, quotes_raw, quotes_sha, quote_rows, QUOTE_COLUMNS, "bot: maker quotes")
    _push_replaced(store, MAKER_STATE_FILE, state.to_json(), state_raw, state_sha, "bot: maker state")


def _redacted(e: Exception) -> str:
    status = getattr(e, "status_code", None)
    return type(e).__name__ + (f" (HTTP {status})" if status is not None else "")


def _taker_step(mode: str, client, plays: list[Play], store, now: pd.Timestamp, rows_before: int,
                bets_sha: str | None, bankroll_raw: bytes | None, bankroll_sha: str | None) -> bool:
    """The taker run (live/dry_run under BOT_POLICY=taker, shadow under maker) and its
    bet-log and bankroll pushes. Returns had_error."""
    rows, had_error = run(mode, client, KALSHI_START_BANKROLL, plays,
                          KALSHI_BETS_PATH, KALSHI_BANKROLL_PATH, now=now)
    print(f"  {'taker (shadow): ' if mode == 'shadow' else ''}{summarize(rows)}")

    new_rows = load_bet_log(KALSHI_BETS_PATH).iloc[rows_before:]
    if not new_rows.empty:
        _push_bet_log(store, new_rows, bets_sha)
    new_bankroll = KALSHI_BANKROLL_PATH.read_bytes()
    if _bankroll_numbers(new_bankroll) != _bankroll_numbers(bankroll_raw):
        store.put(BANKROLL_FILE, new_bankroll, bankroll_sha, "bot: bankroll")
    return had_error


def main(argv: list[str] | None = None, store=None, client=None, now=None) -> int:
    parser = argparse.ArgumentParser(description="Kalshi auto-bettor")
    parser.add_argument("--mode", choices=["off", "dry_run", "live"], default=BETTING_MODE)
    args = parser.parse_args(argv)

    print(f"Kalshi bettor — mode={args.mode} env={KALSHI_ENV} policy={BOT_POLICY}")
    if args.mode == "off":
        print("  BETTING_MODE=off; nothing to do")
        return 0
    if BOT_POLICY not in ("taker", "maker"):
        print("  ✗ BOT_POLICY must be 'taker' or 'maker'")
        return 2
    if args.mode == "live" and KALSHI_ENV == "prod" and KALSHI_START_BANKROLL <= 0:
        print("  ✗ KALSHI_START_BANKROLL must be set for live prod betting")
        return 2

    now = pd.Timestamp.now(tz="UTC") if now is None else pd.Timestamp(now)
    try:
        store = store or StateStore(STATE_REPO, STATE_BRANCH, BOT_STATE_TOKEN)
        store.check_branch()
        plays_raw, _ = store.get("plays.json")
        if plays_raw is None:
            print("  no plays.json yet; nothing to do")
            return 0
        doc = json.loads(plays_raw)
        age = plays_age_hours(doc, now)
        if age > PLAYS_MAX_AGE_HOURS:
            print(f"  plays are stale ({age:.0f}h old > {PLAYS_MAX_AGE_HOURS}h); betting nothing")
            return 0
        plays = load_plays_doc(doc)

        bets_raw, bets_sha = store.get(BETS_FILE)
        bankroll_raw, bankroll_sha = store.get(BANKROLL_FILE)
        KALSHI_BETS_PATH.parent.mkdir(parents=True, exist_ok=True)
        if bets_raw is not None:
            KALSHI_BETS_PATH.write_bytes(bets_raw)
        elif KALSHI_BETS_PATH.exists():
            KALSHI_BETS_PATH.unlink()
        rows_before = len(load_bet_log(KALSHI_BETS_PATH))

        # The policy in force trades; the other one runs in shadow and only logs.
        # Housekeeping (record maker fills, cancel leftover maker bids) runs whenever
        # betting is live, so a maker->taker flip still closes out the last window.
        client = client or KalshiClient()
        housekeeping = args.mode == "live"
        if BOT_POLICY == "taker":
            had_error = _taker_step(args.mode, client, plays, store, now,
                                    rows_before, bets_sha, bankroll_raw, bankroll_sha)
            _maker_step("shadow", client, plays, store, now, housekeeping)
        else:
            # The read-only shadow taker runs before the maker posts this run's bids:
            # Kalshi's balance excludes resting-order collateral, so sizing after them
            # would shrink the shadow stakes (and the pushed bankroll). Its failure is
            # recorded but must not stop the maker, which is the policy actually trading.
            try:
                had_error = _taker_step("shadow", client, plays, store, now,
                                        rows_before, bets_sha, bankroll_raw, bankroll_sha)
            except Exception as e:  # public logs: exception type and HTTP status only
                print(f"  ✗ taker (shadow) failed: {_redacted(e)}")
                had_error = True
            _maker_step(args.mode, client, plays, store, now, housekeeping)
    except Exception as e:  # public logs: exception type and HTTP status only
        print(f"  ✗ run failed: {_redacted(e)}")
        return 1
    return 1 if had_error else 0


if __name__ == "__main__":
    sys.exit(main())
