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
        self.puts.append((path, sha))
        self.files[path] = data
        return "new"


def _main(monkeypatch, tmp_path, store, client, mode="live"):
    import kalshi_bot.bettor as bettor_mod
    monkeypatch.setattr(bettor_mod, "KALSHI_BETS_PATH", tmp_path / "kalshi_bets.parquet")
    monkeypatch.setattr(bettor_mod, "KALSHI_BANKROLL_PATH", tmp_path / "kalshi_bankroll.json")
    monkeypatch.setattr(bettor_mod, "KALSHI_START_BANKROLL", 1000.0)
    return bettor_mod.main(["--mode", mode], store=store, client=client, now=NOW)


def _plays_bytes(**kw):
    doc = _doc(**kw)
    doc["plays"][0]["market_prob"] = 0.50
    return json.dumps(doc).encode()


def test_main_bets_and_pushes_state(monkeypatch, tmp_path, capsys):
    store, fake = _FakeStore({"plays.json": _plays_bytes()}), _fake()
    assert _main(monkeypatch, tmp_path, store, fake) == 0
    assert len(fake.placed) == 1
    assert {p for p, _ in store.puts} == {"kalshi_bets.parquet", "kalshi_bankroll.json"}
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
