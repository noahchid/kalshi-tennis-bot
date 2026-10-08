# kalshi-tennis-bot

Places Kalshi tennis match-winner bets from a list of model plays produced elsewhere.
This repo contains only the betting bot — no model, data or picks.

## How it runs

`.github/workflows/bot.yml` runs every 15 minutes:

1. Reads `plays.json` from a private state repo (`STATE_REPO`) using `BOT_STATE_TOKEN`.
   If the plays are more than 30 hours old it bets nothing.
2. For each play, finds the player's Kalshi YES market and bets when the edge at the
   ask, net of Kalshi's fee, is at least 3pp:
   - ¼ Kelly on the all-in cost; Challenger plays get half of that.
   - At most 5% of bankroll per bet and 25% per run, limited by the size at the ask.
   - Minimum one contract.
   - Immediate-or-cancel limit orders at the ask.
3. Skips a play when:
   - the match starts within 5 minutes;
   - the Kalshi price is more than 15pp from the sportsbook price the play was made at;
   - the bot already holds or has bet that match (checked against the bet log and Kalshi).
4. Writes the bet log and bankroll back to the private state repo.

Bankroll = `KALSHI_START_BANKROLL` + realized P&L on the bot's own orders − open
cost, capped at the cash on Kalshi's tennis exchange shard.

Logs show counts only (e.g. `filled 1 · skipped 9 (no_market 6, started 3)`).

## Configuration

| Repo variable | Values |
|---|---|
| `KALSHI_ENV` | `demo` (default) or `prod` |
| `BETTING_MODE` | `off`, `dry_run` (default), `live` |

Secrets: `KALSHI_KEY_ID`, `KALSHI_PRIVATE_KEY`, `KALSHI_DEMO_KEY_ID`,
`KALSHI_DEMO_PRIVATE_KEY`, `BOT_STATE_TOKEN`, `STATE_REPO` (`owner/repo`),
`KALSHI_START_BANKROLL` (dollars; required for live prod). The last two are secrets
rather than variables so they stay masked in the public logs.

Real money moves only with `KALSHI_ENV=prod` and `BETTING_MODE=live`. A manual
dispatch with `betting_mode=live` overrides `BETTING_MODE` for that run.

## Local tests

```bash
pip install -r requirements.txt
python -m pytest tests -q
```
