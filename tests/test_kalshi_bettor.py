import json

import pandas as pd
import pytest

from kalshi_bot.bet_log import load_bet_log
from kalshi_bot.bettor import Play, load_plays_doc, plays_age_hours, run, summarize
from tests.kalshi_fakes import FakeKalshi, make_event

NOW = pd.Timestamp("2026-09-23T12:00:00Z")


def _play(p, o, prob=0.62, level="D", market_prob=0.50):
    # market_prob=0.50 sits within KALSHI_MAX_PRICE_DIVERGENCE (0.15) of every
    # yes_ask used across this file's fake markets (0.45-0.57), so existing
    # tests keep their original behavior without needing a per-market value.
    return Play(pd.Timestamp("2026-09-23"), p, o, prob, "Chengdu Open", level, market_prob)


def _fake(**kw):
    markets = (make_event("KXATPMATCH", "26SEP23", "James Duckworth", "Lorenzo Sonego",
                          ask1=0.45, ask2=0.57, occurrence="2026-09-24T06:00:00Z")
               + make_event("KXATPCHALLENGERMATCH", "26SEP23", "Keegan Smith", "Martin Damm",
                            ask1=0.47, ask2=0.54, occurrence="2026-09-24T00:30:00Z"))
    return FakeKalshi(markets=markets, cash=1000.0, **kw)


def _run(fake, plays, tmp_path, mode="live", start=1000.0, now=NOW):
    return run(mode, fake, start, plays, tmp_path / "bets.parquet", tmp_path / "br.json", now=now, kalshi_env="demo")


def test_places_quarter_kelly_order(tmp_path):
    fake = _fake()
    rows, err = _run(fake, [_play("James Duckworth", "Lorenzo Sonego")], tmp_path)
    assert not err
    assert len(fake.placed) == 1
    order = fake.placed[0]
    assert order["ticker"].endswith("-DUC") and order["price"] == 0.45
    # p=.62, cost=.45+.02=.47 -> kelly=.15/.53=.283; quarter=.0708 -> capped at .05 -> $50 / .47 = 106
    assert order["count"] == 106
    assert rows[0]["status"] == "filled"
    assert json.loads((tmp_path / "br.json").read_text())["bankroll"] == 1000.0


def test_challenger_gets_half(tmp_path):
    fake = _fake()
    _run(fake, [_play("Keegan Smith", "Martin Damm", prob=0.55, level="C")], tmp_path)
    # cost=.47+.02=.49 kelly=.06/.51=.1176 ; quarter*half=.0147 -> $14.71 / .49 = 30
    assert fake.placed[0]["count"] == 30


def test_morning_and_night_runs_bet_once(tmp_path):
    fake = _fake()
    play = [_play("James Duckworth", "Lorenzo Sonego")]
    _run(fake, play, tmp_path)
    rows, _ = _run(fake, play, tmp_path)
    assert len(fake.placed) == 1
    assert rows[0]["status"] == "skipped" and rows[0]["skip_reason"] == "bet_log"


def test_lost_log_still_blocked_by_kalshi(tmp_path):
    fake = _fake()
    play = [_play("James Duckworth", "Lorenzo Sonego")]
    _run(fake, play, tmp_path)
    (tmp_path / "bets.parquet").unlink()
    rows, _ = _run(fake, play, tmp_path)
    assert len(fake.placed) == 1
    assert rows[0]["skip_reason"] == "kalshi_position"


def test_skips_when_edge_below_min_at_kalshi_price(tmp_path):
    fake = _fake()
    rows, _ = _run(fake, [_play("Lorenzo Sonego", "James Duckworth", prob=0.60)], tmp_path)  # .60 - .59 < .03
    assert fake.placed == []
    assert rows[0]["skip_reason"] == "edge_below_min"


def test_skips_started_match(tmp_path):
    fake = _fake()
    rows, _ = _run(fake, [_play("Keegan Smith", "Martin Damm", prob=0.60, level="C")], tmp_path,
                   now=pd.Timestamp("2026-09-24T00:28:00Z"))
    assert rows[0]["skip_reason"] == "started"


def test_no_market_logged(tmp_path):
    rows, _ = _run(_fake(), [_play("Ann Aa", "Bob Bb")], tmp_path)
    assert rows[0]["skip_reason"] == "no_market"


def test_dry_run_sends_nothing(tmp_path):
    fake = _fake()
    rows, _ = _run(fake, [_play("James Duckworth", "Lorenzo Sonego")], tmp_path, mode="dry_run")
    assert fake.placed == []
    assert rows[0]["status"] == "dry_run" and rows[0]["contracts"] == 106


def test_run_exposure_cap(tmp_path, monkeypatch):
    import kalshi_bot.bettor as bettor_mod
    monkeypatch.setattr(bettor_mod, "KALSHI_RUN_EXPOSURE_CAP", 0.06)  # $6 of a $100 bankroll
    fake = _fake()
    plays = [_play("James Duckworth", "Lorenzo Sonego", prob=0.95),
             _play("Keegan Smith", "Martin Damm", prob=0.95, level="C")]
    rows, _ = _run(fake, plays, tmp_path, start=100.0)
    # First bet: 5% cap = $5 -> 10 contracts @ .47 = $4.70. Second gets the $1.30
    # left -> 2 contracts @ .49 = $0.98. No dollar minimum, only >= 1 contract.
    assert [o["count"] for o in fake.placed] == [10, 2]
    assert sum(o["count"] * (o["price"] + 0.02) for o in fake.placed) <= 6.0 + 1e-9


def test_sub_one_contract_stake_skipped(tmp_path, monkeypatch):
    import kalshi_bot.bettor as bettor_mod
    monkeypatch.setattr(bettor_mod, "KALSHI_RUN_EXPOSURE_CAP", 0.004)  # $0.40 of $100
    rows, _ = _run(_fake(), [_play("James Duckworth", "Lorenzo Sonego", prob=0.95)], tmp_path, start=100.0)
    assert rows[0]["skip_reason"] == "below_min_stake"


def test_order_error_flags_run(tmp_path):
    rows, err = _run(_fake(fail_orders=True), [_play("James Duckworth", "Lorenzo Sonego")], tmp_path)
    assert err and rows[0]["status"] == "error"


def test_partial_fill(tmp_path):
    rows, _ = _run(_fake(fill_ratio=0.5), [_play("James Duckworth", "Lorenzo Sonego")], tmp_path)
    assert rows[0]["status"] == "partial" and rows[0]["fill_count"] == 53


def _doc(generated_at="2026-09-23T10:00:00Z", **rec):
    base = {"pred_date": "2026-09-23", "player_name": "James Duckworth", "opponent_name": "Lorenzo Sonego",
            "model_prob": 0.62, "market_prob": 0.45, "tournament": "Chengdu Open", "tournament_level": "D"}
    return {"generated_at": generated_at, "source_run": "1", "plays": [dict(base, **rec)]}


def test_load_plays_doc():
    plays = load_plays_doc(_doc())
    assert plays == [Play(pd.Timestamp("2026-09-23"), "James Duckworth", "Lorenzo Sonego", 0.62,
                          "Chengdu Open", "D", 0.45)]


def test_load_plays_doc_market_prob_missing_is_none():
    assert load_plays_doc(_doc(market_prob=None))[0].market_prob is None


def test_plays_age_hours():
    assert plays_age_hours(_doc(), NOW) == pytest.approx(2.0)
    assert plays_age_hours({"plays": []}, NOW) == float("inf")


def test_skip_rows_not_persisted(tmp_path):
    fake = _fake()
    _run(fake, [_play("James Duckworth", "Lorenzo Sonego"), _play("Ann Aa", "Bob Bb")], tmp_path)
    log = load_bet_log(tmp_path / "bets.parquet")
    assert list(log.status) == ["filled"]


def test_summarize_counts_only():
    rows = [{"status": "filled", "player_name": "James Duckworth"},
            {"status": "skipped", "skip_reason": "no_market", "player_name": "Ann Aa"},
            {"status": "skipped", "skip_reason": "no_market"},
            {"status": "skipped", "skip_reason": "started"}]
    out = summarize(rows)
    assert out == "filled 1 · skipped 3 (no_market 2, started 1)"
    assert "Duckworth" not in out


def test_skips_empty_book_ask(tmp_path):
    fake = _fake()
    # Overwrite Duckworth's market to have a zero (empty order book) ask.
    for m in fake.markets:
        if m["ticker"].endswith("-DUC"):
            m["yes_ask_dollars"] = "0.0000"
    rows, _ = _run(fake, [_play("James Duckworth", "Lorenzo Sonego")], tmp_path)
    assert fake.placed == []
    assert rows[0]["skip_reason"] == "no_ask"


def test_unparseable_ask_does_not_crash(tmp_path):
    fake = _fake()
    # Missing/unparseable yes_ask_dollars -> yes_ask is NaN. Must be caught as
    # no_ask BEFORE fee/net_edge (kalshi_fee(nan) raises) is ever computed.
    for m in fake.markets:
        if m["ticker"].endswith("-DUC"):
            m["yes_ask_dollars"] = None
    rows, _ = _run(fake, [_play("James Duckworth", "Lorenzo Sonego")], tmp_path)
    assert fake.placed == []
    assert rows[0]["skip_reason"] == "no_ask"


def test_skips_no_start_time(tmp_path):
    markets = make_event("KXATPMATCH", "26SEP23", "James Duckworth", "Lorenzo Sonego",
                         ask1=0.45, ask2=0.57, occurrence="")
    fake = FakeKalshi(markets=markets, cash=1000.0)
    rows, _ = _run(fake, [_play("James Duckworth", "Lorenzo Sonego")], tmp_path)
    assert rows[0]["skip_reason"] == "no_start_time"
    assert fake.placed == []


def test_naive_occurrence_treated_as_utc(tmp_path):
    # No trailing "Z" -> pandas parses this as a tz-naive timestamp. It must be
    # treated as UTC (not crash, not silently compare naive vs aware).
    markets = make_event("KXATPMATCH", "26SEP23", "James Duckworth", "Lorenzo Sonego",
                         ask1=0.45, ask2=0.57, occurrence="2026-09-24T06:00:00")
    fake = FakeKalshi(markets=markets, cash=1000.0)
    rows, _ = _run(fake, [_play("James Duckworth", "Lorenzo Sonego")], tmp_path,
                   now=pd.Timestamp("2026-09-24T05:56:00Z"))  # inside the 5-min buffer, in UTC
    assert rows[0]["skip_reason"] == "started"


def test_run_rejects_invalid_mode(tmp_path):
    fake = _fake()
    with pytest.raises(ValueError):
        _run(fake, [_play("James Duckworth", "Lorenzo Sonego")], tmp_path, mode="off")


def test_second_candidate_same_event_skipped(tmp_path):
    fake = _fake()
    plays = [_play("James Duckworth", "Lorenzo Sonego", prob=0.70),
             _play("James Duckworth", "Lorenzo Sonego", prob=0.62)]
    rows, _ = _run(fake, plays, tmp_path)
    assert len(fake.placed) == 1
    statuses = sorted(r["status"] for r in rows)
    assert statuses == ["filled", "skipped"]
    skipped = [r for r in rows if r["status"] == "skipped"]
    assert skipped[0]["skip_reason"] == "bet_log"


def test_fill_count_missing_assumes_full_fill(tmp_path):
    class NoFillCountKalshi(FakeKalshi):
        def create_order(self, ticker, price, count, client_order_id):
            self.placed.append({"ticker": ticker, "price": price, "count": count,
                                "client_order_id": client_order_id})
            return {"order_id": "o1", "client_order_id": client_order_id,
                    "average_fill_price": f"{price:.4f}", "average_fee_paid": "0.0200"}

    markets = (make_event("KXATPMATCH", "26SEP23", "James Duckworth", "Lorenzo Sonego",
                          ask1=0.45, ask2=0.57, occurrence="2026-09-24T06:00:00Z")
               + make_event("KXATPCHALLENGERMATCH", "26SEP23", "Keegan Smith", "Martin Damm",
                            ask1=0.47, ask2=0.54, occurrence="2026-09-24T00:30:00Z"))
    fake = NoFillCountKalshi(markets=markets, cash=1000.0)
    rows, err = _run(fake, [_play("James Duckworth", "Lorenzo Sonego")], tmp_path)
    assert not err
    assert rows[0]["status"] == "filled"
    assert rows[0]["fill_count"] == 106  # same count as test_places_quarter_kelly_order
    assert rows[0]["error"] == "fill count missing from order response; assumed full fill"
    assert json.loads((tmp_path / "br.json").read_text())["bankroll"] == 1000.0


def test_other_exception_in_candidate_loop_is_logged_not_raised(tmp_path):
    markets = (make_event("KXATPMATCH", "26SEP23", "James Duckworth", "Lorenzo Sonego",
                          ask1=0.45, ask2=0.57, occurrence="2026-09-24T06:00:00Z")
               + make_event("KXATPCHALLENGERMATCH", "26SEP23", "Keegan Smith", "Martin Damm",
                            ask1=0.47, ask2=0.54, occurrence="2026-09-24T00:30:00Z"))
    duck_event = markets[0]["event_ticker"]

    class RaisingPositionsKalshi(FakeKalshi):
        def get_positions(self, event_ticker=None):
            if event_ticker == duck_event:
                raise RuntimeError("boom")
            return super().get_positions(event_ticker=event_ticker)

    fake = RaisingPositionsKalshi(markets=markets, cash=1000.0)
    plays = [_play("James Duckworth", "Lorenzo Sonego"),
             _play("Keegan Smith", "Martin Damm", prob=0.55, level="C")]
    rows, err = _run(fake, plays, tmp_path)

    assert err
    by_player = {r["player_name"]: r for r in rows}
    assert by_player["James Duckworth"]["status"] == "error"
    assert "RuntimeError" in by_player["James Duckworth"]["error"]
    assert by_player["Keegan Smith"]["status"] == "filled"

    log = load_bet_log(tmp_path / "bets.parquet")
    assert len(log) == 2
    assert set(log["status"]) == {"error", "filled"}


def test_skips_missing_market_prob(tmp_path):
    fake = _fake()
    rows, _ = _run(fake, [_play("James Duckworth", "Lorenzo Sonego", market_prob=None)], tmp_path)
    assert fake.placed == []
    assert rows[0]["skip_reason"] == "no_market_prob"


def test_skips_price_divergence(tmp_path):
    fake = _fake()
    # yes_ask is 0.45; market_prob far enough away to exceed the 0.15 cap.
    rows, _ = _run(fake, [_play("James Duckworth", "Lorenzo Sonego", market_prob=0.90)], tmp_path)
    assert fake.placed == []
    assert rows[0]["skip_reason"] == "price_divergence"


def test_main_mode_off_returns_0_without_client(monkeypatch):
    import kalshi_bot.bettor as bettor_mod

    def boom():
        raise AssertionError("KalshiClient must not be constructed in --mode off")

    monkeypatch.setattr(bettor_mod, "KalshiClient", boom)

    assert bettor_mod.main(["--mode", "off"]) == 0


def test_main_live_prod_zero_bankroll_returns_2_without_client(monkeypatch):
    import kalshi_bot.bettor as bettor_mod

    def boom():
        raise AssertionError("KalshiClient must not be constructed when start bankroll guard trips")

    monkeypatch.setattr(bettor_mod, "KALSHI_ENV", "prod")
    monkeypatch.setattr(bettor_mod, "KALSHI_START_BANKROLL", 0)
    monkeypatch.setattr(bettor_mod, "KalshiClient", boom)

    assert bettor_mod.main(["--mode", "live"]) == 2


def test_unfilled_play_retries_next_run_with_new_order_id(tmp_path):
    fake = _fake(fill_ratio=0.0)
    play = [_play("James Duckworth", "Lorenzo Sonego")]
    rows1, err1 = _run(fake, play, tmp_path, now=NOW)
    rows2, err2 = _run(fake, play, tmp_path, now=NOW + pd.Timedelta(hours=12) - pd.Timedelta(hours=11))
    assert [r["status"] for r in rows1 + rows2] == ["unfilled", "unfilled"]
    assert not err1 and not err2
    assert fake.placed[0]["client_order_id"] != fake.placed[1]["client_order_id"]


class _FakeStore:
    def __init__(self, files=None):
        self.files = dict(files or {})  # path -> bytes
        self.puts = []

        self.fail_puts = 0  # fail this many puts with a 409 before succeeding
        self.fail_paths = {}  # path -> fail this many puts of that path with a 409
        self.conflict_writes = {}  # path -> bytes a concurrent writer lands at the conflict
        self.branch_ok = True

    def check_branch(self):
        from kalshi_bot.state import StateError
        if not self.branch_ok:
            raise StateError("state branch missing", 404)

    def get(self, path):
        data = self.files.get(path)
        return (data, f"sha-{path}") if data is not None else (None, None)

    def put(self, path, data, sha, message):
        from kalshi_bot.state import StateError
        if self.fail_puts:
            self.fail_puts -= 1
            raise StateError(f"writing {path} failed", 409)
        if self.fail_paths.get(path):
            self.fail_paths[path] -= 1
            if path in self.conflict_writes:
                self.files[path] = self.conflict_writes.pop(path)
            raise StateError(f"writing {path} failed", 409)
        self.puts.append((path, sha))
        self.files[path] = data
        return "new"


def _main(monkeypatch, tmp_path, store, client, mode="live", policy="taker", now=NOW):
    import kalshi_bot.bettor as bettor_mod
    monkeypatch.setattr(bettor_mod, "KALSHI_BETS_PATH", tmp_path / "kalshi_bets.parquet")
    monkeypatch.setattr(bettor_mod, "KALSHI_BANKROLL_PATH", tmp_path / "kalshi_bankroll.json")
    monkeypatch.setattr(bettor_mod, "KALSHI_START_BANKROLL", 1000.0)
    monkeypatch.setattr(bettor_mod, "BOT_POLICY", policy)
    return bettor_mod.main(["--mode", mode], store=store, client=client, now=now)


def _plays_bytes(**kw):
    doc = _doc(**kw)
    doc["plays"][0]["market_prob"] = 0.50
    return json.dumps(doc).encode()


def test_main_bets_and_pushes_state(monkeypatch, tmp_path, capsys):
    store, fake = _FakeStore({"plays.json": _plays_bytes()}), _fake()
    assert _main(monkeypatch, tmp_path, store, fake) == 0
    assert len(fake.placed) == 1
    # Taker state, plus the maker shadow's quotes and state (no maker fills yet).
    assert {p for p, _ in store.puts} == {"kalshi_bets.parquet", "kalshi_bankroll.json",
                                          "kalshi_quotes.parquet", "kalshi_maker_state.json"}
    out = capsys.readouterr().out
    assert "filled 1" in out and "Duckworth" not in out and "1000" not in out


def test_main_second_run_pushes_nothing_new(monkeypatch, tmp_path):
    store, fake = _FakeStore({"plays.json": _plays_bytes()}), _fake()
    _main(monkeypatch, tmp_path, store, fake)
    store.puts.clear()
    (tmp_path / "kalshi_bets.parquet").unlink()  # fresh runner: state comes from the store
    assert _main(monkeypatch, tmp_path, store, fake) == 0
    assert len(fake.placed) == 1
    # Only the skip row was produced, and skips aren't persisted, so the bet log
    # isn't rewritten. (The bankroll may change: the first bet is now open.)
    assert "kalshi_bets.parquet" not in [p for p, _ in store.puts]


def test_main_existing_bet_log_sha_used(monkeypatch, tmp_path):
    store, fake = _FakeStore({"plays.json": _plays_bytes()}), _fake()
    _main(monkeypatch, tmp_path, store, fake)
    store.puts.clear()
    store.files["plays.json"] = _plays_bytes(player_name="Keegan Smith", opponent_name="Martin Damm",
                                             tournament_level="C")
    _main(monkeypatch, tmp_path, store, fake)
    assert ("kalshi_bets.parquet", "sha-kalshi_bets.parquet") in store.puts


def test_main_stale_plays_bets_nothing(monkeypatch, tmp_path, capsys):
    store, fake = _FakeStore({"plays.json": _plays_bytes(generated_at="2026-09-21T00:00:00Z")}), _fake()
    assert _main(monkeypatch, tmp_path, store, fake) == 0
    assert fake.placed == [] and store.puts == []
    assert "stale" in capsys.readouterr().out


def test_main_no_plays_file(monkeypatch, tmp_path):
    store, fake = _FakeStore(), _fake()
    assert _main(monkeypatch, tmp_path, store, fake) == 0
    assert fake.placed == []


def test_main_error_output_redacted(monkeypatch, tmp_path, capsys):
    from kalshi_bot.client import KalshiAPIError

    class Broken(FakeKalshi):
        def get_balance(self, exchange_index=None):
            raise KalshiAPIError(401, '{"error":"secret detail about KXATPMATCH-26SEP23DUCSON"}')

    store = _FakeStore({"plays.json": _plays_bytes()})
    assert _main(monkeypatch, tmp_path, store, Broken()) == 1
    out = capsys.readouterr().out
    assert "401" in out and "KXATPMATCH" not in out and "secret detail" not in out


def test_main_missing_state_branch_fails_red(monkeypatch, tmp_path, capsys):
    store, fake = _FakeStore({"plays.json": _plays_bytes()}), _fake()
    store.branch_ok = False
    assert _main(monkeypatch, tmp_path, store, fake) == 1
    assert fake.placed == []
    assert "HTTP 404" in capsys.readouterr().out


def test_main_bet_log_push_retries_and_merges_remote(monkeypatch, tmp_path):
    store, fake = _FakeStore({"plays.json": _plays_bytes()}), _fake()
    remote_row = pd.DataFrame([{"event_ticker": "KXATPMATCH-26SEP20AAABBB", "status": "filled",
                                "kalshi_env": "demo", "emailed": False}])
    remote_path = tmp_path / "remote.parquet"
    remote_row.to_parquet(remote_path, index=False)
    store.files["kalshi_bets.parquet"] = remote_path.read_bytes()
    store.fail_puts = 1  # first bet-log push conflicts (e.g. a manual seed landed meanwhile)
    assert _main(monkeypatch, tmp_path, store, fake) == 0
    pushed = pd.read_parquet(__import__("io").BytesIO(store.files["kalshi_bets.parquet"]))
    assert set(pushed.status) == {"filled"} and len(pushed) == 2


def test_dry_run_does_not_relog_same_play_each_run(monkeypatch, tmp_path):
    store, fake = _FakeStore({"plays.json": _plays_bytes()}), _fake()
    _main(monkeypatch, tmp_path, store, fake, mode="dry_run")
    store.puts.clear()
    _main(monkeypatch, tmp_path, store, fake, mode="dry_run")
    assert "kalshi_bets.parquet" not in [p for p, _ in store.puts]


# ── run(mode="shadow") ───────────────────────────────────────────────────────

def test_shadow_mode_sizes_and_logs_without_sending_or_consulting_kalshi_guard(tmp_path):
    class NoGuardKalshi(FakeKalshi):
        def get_positions(self, event_ticker=None):
            if event_ticker is not None:
                raise AssertionError("shadow must not consult Kalshi positions per event")
            return super().get_positions()

        def get_orders(self, event_ticker=None, status=None):
            if event_ticker is not None:
                raise AssertionError("shadow must not consult Kalshi orders per event")
            return super().get_orders(status=status)

    markets = make_event("KXATPMATCH", "26SEP23", "James Duckworth", "Lorenzo Sonego",
                         ask1=0.45, ask2=0.57, occurrence="2026-09-24T06:00:00Z")
    fake = NoGuardKalshi(markets=markets, cash=1000.0)
    rows, err = _run(fake, [_play("James Duckworth", "Lorenzo Sonego")], tmp_path, mode="shadow")
    assert not err and fake.placed == [] and fake.calls == []
    assert rows[0]["status"] == "shadow" and rows[0]["run_mode"] == "shadow"
    assert rows[0]["contracts"] == 106  # sized exactly like the live taker
    assert list(load_bet_log(tmp_path / "bets.parquet").status) == ["shadow"]


def test_shadow_mode_logs_one_row_per_event_across_runs(tmp_path):
    fake = _fake()
    play = [_play("James Duckworth", "Lorenzo Sonego")]
    _run(fake, play, tmp_path, mode="shadow")
    rows, _ = _run(fake, play, tmp_path, mode="shadow", now=NOW + pd.Timedelta(minutes=15))
    assert rows[0]["status"] == "skipped" and rows[0]["skip_reason"] == "bet_log"
    assert list(load_bet_log(tmp_path / "bets.parquet").status) == ["shadow"]


def test_shadow_rows_do_not_block_live_taker_or_dry_run(tmp_path):
    fake = _fake()
    play = [_play("James Duckworth", "Lorenzo Sonego")]
    _run(fake, play, tmp_path, mode="shadow")
    dry, _ = _run(fake, play, tmp_path, mode="dry_run")
    live, _ = _run(fake, play, tmp_path, mode="live")
    assert dry[0]["status"] == "dry_run" and live[0]["status"] == "filled"
    assert len(fake.placed) == 1


def test_shadow_status_is_not_emailed():
    from kalshi_bot.bet_log import EMAIL_STATUSES, SHADOW_STATUSES
    assert SHADOW_STATUSES == {"shadow"}
    assert EMAIL_STATUSES == {"filled", "partial", "dry_run"}


# ── main: BOT_POLICY routing ─────────────────────────────────────────────────

def _read_parquet_file(store, path):
    import io
    return pd.read_parquet(io.BytesIO(store.files[path]))


# Today's (pre-maker) taker output for this fixture, captured from main at 8f20609.
_GOLDEN_TAKER_ORDER = {
    "ticker": "KXATPMATCH-26SEP23DUCSON-DUC", "price": 0.45, "count": 106,
    "client_order_id": "atpedge-61dc372729af-1790164800", "time_in_force": "immediate_or_cancel",
    "expiration_time": None, "post_only": None, "cancel_order_on_pause": None, "stp": "taker_at_cross",
}
_GOLDEN_TAKER_ROW = {
    "logged_at": pd.Timestamp("2026-09-23 12:00:00+0000", tz="UTC"), "run_mode": "live",
    "kalshi_env": "demo", "pred_date": pd.Timestamp("2026-09-23 00:00:00"),
    "player_name": "James Duckworth", "opponent_name": "Lorenzo Sonego", "tournament": "Chengdu Open",
    "tournament_level": "D", "is_challenger": False, "event_ticker": "KXATPMATCH-26SEP23DUCSON",
    "market_ticker": "KXATPMATCH-26SEP23DUCSON-DUC", "model_prob": 0.62, "yes_ask": 0.45, "fee": 0.02,
    "net_edge": 0.14999999999999997, "kelly": 0.28301886792452824, "bankroll": 1000.0,
    "stake_target": 50.0, "contracts": 106, "client_order_id": "atpedge-61dc372729af-1790164800",
    "status": "filled", "skip_reason": None, "order_id": "o1", "fill_count": 106,
    "avg_fill_price": 0.45, "avg_fee": 0.02, "error": None, "emailed": False,
}
_GOLDEN_BANKROLL = {"start": 1000.0, "realized_pnl": 0, "open_cost": 0, "cash": 1000.0,
                    "bankroll": 1000.0, "as_of": "2026-09-23 12:00:00+00:00",
                    "kalshi_env": "demo", "mode": "live"}


def test_taker_policy_unchanged(monkeypatch, tmp_path):
    store, fake = _FakeStore({"plays.json": _plays_bytes()}), _fake()
    assert _main(monkeypatch, tmp_path, store, fake, policy="taker") == 0
    # Exactly today's single IOC order and nothing else sent to Kalshi.
    assert fake.placed == [_GOLDEN_TAKER_ORDER]
    assert [c for c in fake.calls if c[0] in ("create", "cancel")] == [
        ("create", _GOLDEN_TAKER_ORDER["client_order_id"], _GOLDEN_TAKER_ORDER["ticker"])]
    log = _read_parquet_file(store, "kalshi_bets.parquet")
    assert log.to_dict("records") == [_GOLDEN_TAKER_ROW]
    assert json.loads(store.files["kalshi_bankroll.json"]) == _GOLDEN_BANKROLL
    # The maker shadow wrote shadow_quote rows only to the quotes file: DUC now
    # holds the taker's 106 contracts (position cap), so only SON would be quoted.
    quotes = _read_parquet_file(store, "kalshi_quotes.parquet")
    assert list(quotes.status) == ["shadow_quote"] and list(quotes["mode"]) == ["shadow"]
    assert list(quotes.market_ticker) == ["KXATPMATCH-26SEP23DUCSON-SON"]
    assert "kalshi_maker_fills.parquet" not in store.files


def test_taker_policy_maker_shadow_logs_once_per_market_per_utc_day(monkeypatch, tmp_path):
    store, fake = _FakeStore({"plays.json": _plays_bytes()}), _fake()
    _main(monkeypatch, tmp_path, store, fake, mode="dry_run")
    store.puts.clear()
    _main(monkeypatch, tmp_path, store, fake, mode="dry_run", now=NOW + pd.Timedelta(minutes=15))
    # Same UTC day, same quotes, same first mids: nothing new for the maker files.
    assert not {"kalshi_quotes.parquet", "kalshi_maker_state.json"} & {p for p, _ in store.puts}
    _main(monkeypatch, tmp_path, store, fake, mode="dry_run",
          now=pd.Timestamp("2026-09-24T00:30:00Z"))  # next UTC day, 5.5h before start
    quotes = _read_parquet_file(store, "kalshi_quotes.parquet")
    assert sorted(quotes.market_ticker.str[-3:]) == ["DUC", "DUC", "SON", "SON"]
    assert set(quotes.status) == {"shadow_quote"}
    assert sorted(quotes.logged_at.dt.date.astype(str)) == ["2026-09-23"] * 2 + ["2026-09-24"] * 2
    assert fake.placed == []


def test_maker_policy_places_quotes_and_shadows_taker_once_per_event(monkeypatch, tmp_path):
    store, fake = _FakeStore({"plays.json": _plays_bytes()}), _fake()
    assert _main(monkeypatch, tmp_path, store, fake, policy="maker") == 0
    assert _main(monkeypatch, tmp_path, store, fake, policy="maker",
                 now=NOW + pd.Timedelta(minutes=15)) == 0
    # No IOC order: every order is a post-only maker bid.
    assert fake.placed and all(p["time_in_force"] == "good_till_canceled" and p["post_only"] is True
                               and p["client_order_id"].startswith("atpedge-mk-") for p in fake.placed)
    assert len(fake.placed) == 4 and len(fake.cancelled) == 2
    quotes = _read_parquet_file(store, "kalshi_quotes.parquet")
    assert list(quotes.status) == ["placed"] * 4 and set(quotes["mode"]) == {"live"}
    # The taker is shadowed: one row per event across both runs. In run 1 the maker's
    # resting atpedge-mk- bids were already on this event, so the shadow didn't use the guard.
    log = _read_parquet_file(store, "kalshi_bets.parquet")
    assert list(log.status) == ["shadow"] and list(log.run_mode) == ["shadow"]
    assert log.contracts.iloc[0] == 106
    state = json.loads(store.files["kalshi_maker_state.json"])
    assert set(state["first_mid"]) == {"KXATPMATCH-26SEP23DUCSON"}


def test_maker_policy_fills_recorded_once_and_seen_rebuilt_from_store(monkeypatch, tmp_path):
    store, fake = _FakeStore({"plays.json": _plays_bytes()}), _fake()
    _main(monkeypatch, tmp_path, store, fake, policy="maker")
    duc_order = next(o["order_id"] for o in fake.orders if o["ticker"].endswith("-DUC"))
    fake.fill_resting(duc_order, 2, ts=NOW + pd.Timedelta(minutes=3))
    store.fail_paths["kalshi_maker_fills.parquet"] = 1  # the fills push conflicts once
    assert _main(monkeypatch, tmp_path, store, fake, policy="maker",
                 now=NOW + pd.Timedelta(minutes=15)) == 0
    fills = _read_parquet_file(store, "kalshi_maker_fills.parquet")
    assert len(fills) == 1 and fills["count"].iloc[0] == 2
    store.puts.clear()
    (tmp_path / "kalshi_bets.parquet").unlink()  # fresh runner: everything comes from the store
    assert _main(monkeypatch, tmp_path, store, fake, policy="maker",
                 now=NOW + pd.Timedelta(minutes=30)) == 0
    assert "kalshi_maker_fills.parquet" not in [p for p, _ in store.puts]
    assert len(_read_parquet_file(store, "kalshi_maker_fills.parquet")) == 1


@pytest.mark.parametrize("policy", ["taker", "maker"])
def test_betting_mode_off_does_nothing_under_either_policy(monkeypatch, tmp_path, policy):
    import kalshi_bot.bettor as bettor_mod

    class NoStore:
        def __getattr__(self, name):
            raise AssertionError("BETTING_MODE=off must not touch the state repo")

    def boom():
        raise AssertionError("BETTING_MODE=off must not construct a Kalshi client")

    monkeypatch.setattr(bettor_mod, "KalshiClient", boom)
    fake = _fake()
    assert _main(monkeypatch, tmp_path, NoStore(), None, mode="off", policy=policy) == 0
    assert fake.calls == []


def test_invalid_policy_returns_2_without_client(monkeypatch, tmp_path):
    import kalshi_bot.bettor as bettor_mod

    def boom():
        raise AssertionError("an unknown BOT_POLICY must not reach Kalshi")

    monkeypatch.setattr(bettor_mod, "KalshiClient", boom)
    store = _FakeStore({"plays.json": _plays_bytes()})
    assert _main(monkeypatch, tmp_path, store, None, policy="both") == 2
    assert store.puts == []


@pytest.mark.parametrize("policy", ["taker", "maker"])
def test_stdout_is_counts_only(monkeypatch, tmp_path, capsys, policy):
    store, fake = _FakeStore({"plays.json": _plays_bytes()}), _fake()
    _main(monkeypatch, tmp_path, store, fake, policy=policy)
    duc_order = next((o["order_id"] for o in fake.orders if o["ticker"].endswith("-DUC")
                      and o["status"] == "resting"), None)
    if duc_order:
        fake.fill_resting(duc_order, 2, ts=NOW + pd.Timedelta(minutes=3))
    _main(monkeypatch, tmp_path, store, fake, policy=policy, now=NOW + pd.Timedelta(minutes=15))
    out = capsys.readouterr().out
    assert "maker" in out
    if policy == "maker":
        assert "placed 2" in out and "fills 1" in out
    for secret in ("KXATP", "26SEP23", "DUCSON", "-DUC", "-SON", "atpedge", "Duckworth", "Sonego",
                   "James", "Lorenzo", "Chengdu", "0.44", "0.45", "0.56", "0.57", "1000"):
        assert secret not in out


@pytest.mark.parametrize("path", ["kalshi_quotes.parquet", "kalshi_maker_state.json"])
def test_maker_state_push_conflict_rereads_remote_and_retries_once(monkeypatch, tmp_path, path):
    import io
    store, fake = _FakeStore({"plays.json": _plays_bytes()}), _fake()
    if path == "kalshi_quotes.parquet":
        prior = pd.DataFrame([{"logged_at": NOW - pd.Timedelta(days=3), "policy": "maker",
                               "mode": "live", "market_ticker": "KXATPMATCH-26SEP20AAABBB-AAA",
                               "status": "placed"}])
        buf = io.BytesIO()
        prior.to_parquet(buf, index=False)
        store.conflict_writes[path] = buf.getvalue()
    else:
        store.conflict_writes[path] = b'{"first_mid": {}, "retired": {}}'
    store.fail_paths[path] = 1
    assert _main(monkeypatch, tmp_path, store, fake, policy="maker") == 0
    attempts = [sha for p, sha in store.puts if p == path]
    assert attempts == [f"sha-{path}"]  # one successful retry, on the re-read sha
    if path == "kalshi_quotes.parquet":
        merged = _read_parquet_file(store, path)
        assert list(merged.status) == ["placed"] * 3  # remote row kept, this run's two re-appended
        assert merged.market_ticker.iloc[0] == "KXATPMATCH-26SEP20AAABBB-AAA"
    else:
        assert set(json.loads(store.files[path])["first_mid"]) == {"KXATPMATCH-26SEP23DUCSON"}


def test_maker_error_row_does_not_fail_run(monkeypatch, tmp_path):
    store = _FakeStore({"plays.json": _plays_bytes()})
    fake = _fake(reject_crossing={"KXATPMATCH-26SEP23DUCSON-DUC"})
    assert _main(monkeypatch, tmp_path, store, fake, policy="maker") == 0
    quotes = _read_parquet_file(store, "kalshi_quotes.parquet")
    assert sorted(quotes.status) == ["error", "placed"]


def test_maker_shadow_exception_fails_run_after_taker_state_pushed(monkeypatch, tmp_path, capsys):
    class FillsDown(FakeKalshi):
        def get_fills(self, min_ts=None, ticker=None):
            raise RuntimeError("KXATPMATCH-26SEP23DUCSON down")

    markets = make_event("KXATPMATCH", "26SEP23", "James Duckworth", "Lorenzo Sonego",
                         ask1=0.45, ask2=0.57, occurrence="2026-09-24T06:00:00Z")
    store, fake = _FakeStore({"plays.json": _plays_bytes()}), FillsDown(markets=markets, cash=1000.0)
    assert _main(monkeypatch, tmp_path, store, fake, policy="taker") == 1
    assert len(fake.placed) == 1
    assert list(_read_parquet_file(store, "kalshi_bets.parquet").status) == ["filled"]
    out = capsys.readouterr().out
    assert "RuntimeError" in out and "KXATP" not in out


def test_dedupe_shadow_quotes_keeps_first_per_market_per_utc_day():
    from kalshi_bot.bettor import _dedupe_shadow_quotes
    existing = pd.DataFrame([
        {"logged_at": pd.Timestamp("2026-09-23T01:00:00Z"), "market_ticker": "M1", "status": "shadow_quote"},
        {"logged_at": pd.Timestamp("2026-09-23T01:00:00Z"), "market_ticker": "M2", "status": "placed"},
        {"logged_at": pd.Timestamp("2026-09-22T23:59:00Z"), "market_ticker": "M3", "status": "shadow_quote"},
    ])
    t = pd.Timestamp("2026-09-23T23:00:00Z")
    rows = [{"logged_at": t, "market_ticker": m, "status": s}
            for m, s in (("M1", "shadow_quote"), ("M2", "shadow_quote"), ("M3", "shadow_quote"),
                         ("M4", "shadow_quote"), ("M4", "shadow_quote"), ("M1", "cancel_failed"))]
    kept = _dedupe_shadow_quotes(rows, existing)
    assert [(r["market_ticker"], r["status"]) for r in kept] == [
        ("M2", "shadow_quote"), ("M3", "shadow_quote"), ("M4", "shadow_quote"), ("M1", "cancel_failed")]
    assert _dedupe_shadow_quotes(rows[:1], pd.DataFrame()) == rows[:1]


def test_workflow_passes_policy_env():
    from pathlib import Path
    wf = (Path(__file__).resolve().parents[1] / ".github" / "workflows" / "bot.yml").read_text()
    assert "BOT_POLICY: ${{ vars.BOT_POLICY || 'taker' }}" in wf
    assert "MAKER_STOP_HOURS: ${{ vars.MAKER_STOP_HOURS || '4' }}" in wf
