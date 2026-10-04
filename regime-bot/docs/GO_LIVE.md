# Going live: the questions, and who answers them

This bot only paper trades. No code path can send a real order: there is no exchange
client with order permissions, and no keys are used. Adding one is a separate, reviewed
change. That change must start with `regimebot preflight` exiting 0 on at least 30 days of
paper trading.

Each question below is answered by code, not by opinion. `regimebot preflight` runs every
check and prints PASS or FAIL. There is no override flag.

| Question | How it is answered | Preflight check |
|---|---|---|
| **Does paper match the backtest?** | `state/live/paper_summary.json` (written nightly from the journal) is compared with `backtest_summary.json`. It needs ≥ 30 days and ≥ 30 trades, a paper Sharpe of at least half the backtest's, and a hit rate within 10 points. | `paper_matches_backtest` |
| **Is every decision using filtered probabilities only?** | Property tests replace future candles with noise and require every earlier decision to stay byte-identical, at the filter, engine and whole-walk-forward levels. An AST test bans smoothing and Viterbi calls outside fitting. | `filtered_probabilities_only` |
| **Did the kill switch fire in testing?** | `regimebot drill-kill` fires it against a held position, checks that it flattens and blocks entries, resets, and records the evidence. The kill unit tests run too. | `kill_switch_fired_in_testing` |
| **Is any hard limit delegated to a model instead of code?** | Limits are frozen constants in `limits.py`, which reads no file, environment variable or model output (AST test). Playbooks can only tighten them. The research layer can't be imported by the trading path (AST test). | `no_limit_delegated_to_model` |
| **Does the refit keep state labels stable?** | At least 80% of walk-forward refits must be accepted, and the latest must be clean, with no label conflict. | `refit_labels_stable` |
| **What market would break this?** | The section below. Preflight checks that this file exists. Its content is for a human to review. | `breaking_markets_documented` |
| (gate) | `strategy.md` must say `Status: ACCEPTED`. | `strategy_accepted` |

## What market would break this?

- **A regime the training data never contained.** Four years of BTC (2018–2021) include a
  bear market, the March 2020 crash and a mania. They do not include an exchange collapse
  that takes the data feed down with it, or a sustained depeg of the quote currency (USDT).
  The filter will confidently pick whichever known state is least wrong.
- **Fast regime flips.** Hysteresis requires 3 candles above 0.70 plus a cooldown, so the
  bot lags every turn by at least 3 hours. A market that whipsaws faster than that is pure
  cost for it.
- **Liquidity holes.** Slippage is modeled at 2 bps. A thin weekend book or a liquidation
  cascade can cost 50–500 bps on a market exit. The backtest does not include that.

## WHAT COULD BLOW UP THIS ACCOUNT?

These are ranked by how badly each one could hurt.

1. **Gap through the stop.** Crypto trades 24/7 but still gaps on exchange halts and
   liquidation cascades. A stop is a market order once triggered; it caps intent, not loss.
   *Bound:* max position 20% of equity, and STRESS is capped at 0.25×, so a 30% gap costs at
   most about 6% of equity.
2. **The bot keeps trading on a dead or stale feed.** *Bound:* a feed more than 2 bars late
   flattens. The watchdog fires the kill switch if the loop stops heartbeating for 3
   minutes. Stop orders sit at the broker in live mode (paper simulates them).
3. **State-label swap after a refit.** If CALM_UP and CRASH traded places, the bot would
   trend-follow a crash. *Bound:* Hungarian matching keeps labels on refit, an inadmissible
   label raises `label_conflict` and freezes entries, and CRASH is always flat and sizes to
   zero.
4. **Overfitting through nightly tweaks.** Each nightly playbook change is another trial on
   the same history, so the gates become easier to pass by luck over time. *Bound:* every
   change must re-clear the same gates. This only partly helps; see the open item below.
5. **Model or calibration drift.** *Bound:* drift alarms freeze entries, sizing is zero for
   any label that hasn't passed calibration, and the daily report shows calibration.
6. **Key leak (after a live exchange is added).** *Bound:* trade-only keys with withdrawals
   off and an IP allowlist, kept only in `.env`, redacted from every log and alert. This bot
   never asks for passwords or 2FA codes.
7. **Operator error.** Resetting the kill switch needs a written confirmation. There is no
   flag that skips preflight.

**Open items before any real money:**
- A deflated-Sharpe adjustment for the number of nightly trials (item 4).
- Real-exchange order handling: partial fills, rejects, rate limits.
- A slippage model calibrated on real fills.

Until these are done, the answer to "go live?" is **no**, whatever preflight says.
