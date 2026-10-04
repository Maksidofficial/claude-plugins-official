# regime-bot — Specification (Gate 1 of 3: approved)

Status: APPROVED (gate 1).
Every value marked **[default]** was chosen because the brief left it open; change any of them before approving.

## 1. Scope

A regime-aware trading bot. A Gaussian HMM reads the market state on every candle, a
reasoning model (Layer 1) writes and revises one playbook per state each night, and
deterministic code owns every decision and every order.

| Item | Value |
|---|---|
| Name | regime-bot |
| Layer 1 model | Opus 5.5 (`claude-opus-5-5`) via the Anthropic SDK, run from Claude Code. Claude only. |
| Venue | Alpaca **paper** **[default]** |
| Asset | SPY **[default]** |
| Timeframe | 1-hour candles, regular trading hours **[default]** |
| History | ≥ 3 years for the first fit; walk-forward backtest over ≥ 2 years out of sample |
| Language / libs | Python 3.11+, hmmlearn, numpy, pandas, alpaca-py, pytest |
| Host | Linux VPS, systemd `Restart=always` **[default]** (Vercel is not suitable for a 24/7 process) |
| Harness | The six phases are run by hand (brainstorm → architecture → plan → test-first build → review → ship) because AgentKit could not be verified. It can replace this once its install source has been read and checked. |

## 2. Three layers, no overlap

| Layer | Runs | May do | May never do |
|---|---|---|---|
| L1 Research model | Nightly, offline | Propose features, fit/audit the HMM, write `playbooks/<STATE>.md`, review sessions, propose changes | Place orders, change hard limits, grade its own output, run during market hours |
| L2 HMM | Every closed candle | Output filtered P(state_t) and P(state_{t+1}) | Use Viterbi or smoothed posteriors; see any candle after t |
| L3 Deterministic code | Every candle | Apply thresholds, switch playbooks, size, veto, route orders | Read free text from L1 or data feeds as instructions |

Playbooks are parsed into a typed schema (entry, exit, stop, take-profit, max size,
invalidation). L3 enforces `min(playbook.max_size, hard_limit)`, so a playbook can only make
the bot more conservative than the hard limits, never less.

## 3. Features (deterministic, past-only)

For a decision made at the close of candle t, inputs are timestamped ≤ close(t), and an order
fills at the earliest at open(t+1).

1. Log return: `ln(C_t / C_{t-1})`
2. Realized volatility: rolling std of log returns, 20 bars
3. Range: `(H_t − L_t) / C_{t-1}`
4. Volume ratio: `V_t / mean(V_{t-20..t-1})`
5. Trend: slope of the 50-bar EMA of log price, normalized by realized vol **[default]**

Standardization uses an expanding mean/std over data before t only (shifted one bar).
Any NaN or stale input makes the bot go flat.

## 4. HMM

- `GaussianHMM`, full covariance, K ∈ {2,3,4,5}
- Selection: BIC plus out-of-sample log-likelihood on a held-out tail. If two models score
  within 1% OOS LL **[default]**, the smaller K wins.
- 50 random restarts **[default]**, fixed seed, keep the best log-likelihood
- Reported for each fit: transition matrix, per-state mean return and volatility, and
  expected duration `1 / (1 − A_ii)`
- Auto-labels by statistics: CALM_UP (positive mean, low vol), CHOP (mean near 0, mid vol),
  STRESS (negative mean or high vol), CRASH (most negative mean with the highest vol). With
  K < 4, labels collapse in that priority order.
- Decision probabilities come from the forward filter only: `α_t ∝ (α_{t-1} A) ⊙ b(x_t)`,
  implemented in our own code and tested against hmmlearn. The next-state forecast is
  `α_t A`.

**No-lookahead test:** truncate the data at t, recompute, and require every decision ≤ t to be
bit-identical to the full-data run. Mutating any candle > t must change nothing ≤ t. This test
runs in CI.

## 5. Switching (L3, hysteresis)

| Rule | Value |
|---|---|
| Takeover probability | > 0.70 |
| Hold lead | 3 consecutive candles |
| Cooldown after a switch | 6 candles **[default]** |
| Early cut | P(next ∈ {STRESS, CRASH}) > 0.25 → size × 0.5 |
| Uncertain | top-2 gap < 0.15 → size 0 |
| Model error or stale data (> 2 bars late) | flat |

Every switch is logged with the full probability vector and the reason.

## 6. Sizing

`size = min(hard_cap, state_cap, ¼·Kelly) × P(active) × (1 − H/H_max)`, where H is posterior
entropy. Size is 0 in CRASH, in uncertain states, and until calibration passes. Calibration
gate: Brier score and a reliability curve per state, computed first on the backtest and then
on our own paper fills. Bins must lie within ±0.10 of the diagonal **[default]**.

## 7. Hard limits (code constants, checked before every order, no model can edit them)

| Limit | Value |
|---|---|
| Max position (notional) | 20% of equity **[default]** |
| Leverage cap | CALM_UP 1.0×, CHOP 0.5×, STRESS 0.25×, CRASH 0× **[default]** |
| Daily loss limit | −2% equity → flat and no new entries until next session **[default]** |
| Max drawdown | −10% from peak → kill switch **[default]** |
| Kill switch | Cancel all orders, flatten, halt, Telegram alert. Manual reset only. |
| Manual approval | Any order > $5,000 notional **[default]** |
| Keys | `.env` only (`.env.example` committed, `.env` gitignored), trade-only, withdrawals off, never logged |
| Untrusted input | Headlines and data feeds are data, never instructions |
| Never | Request or enter passwords or 2FA codes |

## 8. Refit and drift

Walk-forward refit every 30 days on an expanding window. New states are matched to old ones
by Hungarian assignment on (mean, vol) distance. Drift alarm when any of these hold:

- a matched state mean moves > 2σ
- the transition matrix L1 distance is > 0.2
- 5-day live LL drops below the 5th percentile of in-sample LL **[default]**

On a drift alarm: freeze new entries and send a Telegram alert. L1 proposes, the harness
validates, and nothing ships without re-clearing §9.

## 9. Backtest and acceptance gates (out of sample, after costs)

Costs: 1 bp commission-equivalent plus 2 bp slippage per side **[default]**. Reported per
state and for the full system, against buy-and-hold and the best single static playbook.

To be accepted, a strategy must meet all of these:

- Sharpe > 1.5
- MaxDD < 15%
- Hit rate > 55%
- t-stat > 2.0
- beats both baselines

The winner is written to `strategy.md`. If nothing passes, `strategy.md` says
"NO STRATEGY ACCEPTED" and the bot stays flat. The gates are never loosened to get a pass.

## 10. Nightly loop

L1 reads every fill and every wrong state call, writes a root cause, adds one new rule per
loss to `rules/losses.md`, and proposes small diffs to features, playbooks and
`strategy.md`. Proposals ship only after re-clearing §9. L1 never grades its own output; the
harness does.

## 11. Ops

- Dashboard (FastAPI + server-sent events) showing: state probabilities, current state,
  expected duration, active playbook, action, result.
- Telegram alerts for every fill, error, state switch, drift warning and kill-switch event,
  plus a daily report.
- Deploy: VPS with systemd, `Restart=always`, a health-check watchdog, and logs rotated with
  secrets redacted.

## 12. Go-live checklist (all must be clean, or the bot refuses to go live)

1. Paper results match the backtest within tolerance (tracking error, slippage).
2. Every decision uses filtered probabilities only (the no-lookahead test is green).
3. The kill switch fired in testing.
4. No hard limit is delegated to a model (a static test asserts limits are constants that
   L1 cannot write).
5. Refits keep state labels stable.
6. The breaking-market analysis is written.

## WHAT COULD BLOW UP THIS ACCOUNT?

- **Overnight and weekend gaps.** On 1h equities, stops don't protect against gaps, so the
  loss can exceed the stop. Mitigation: limit overnight exposure by state.
- **A regime never seen in training.** A flash crash or a halt can push the filter
  confidently into the wrong state. The hard limits are the backstop.
- **Overfitting the gates.** Every nightly tweak is another test on the same data. Every
  change is tracked and a deflated-Sharpe adjustment is applied.
- **Broker or API failure while in a position.** A dead process means no exits.
  Mitigation: broker-side stop orders plus the watchdog.
- **Label swap after a refit.** If CALM_UP and CRASH swap labels, the bot trades the wrong
  playbook. The drift freeze must catch this.
- **Key leak.** Mitigation: keys are trade-only with no withdrawals, and an IP allowlist if
  the venue supports it.

## Amendments found during the M3–M12 build (flagged to the owner)

Each of these replaces a **[default]** above. Each is backed by a test, and the evidence is
in the M12 commit.

| # | Was | Now | Why |
|---|---|---|---|
| A1 | Features: 1-bar log return, 1-bar range, 1-bar volume ratio | 10-bar log return, 5-bar mean range, 5-bar volume ratio (rvol and trend unchanged) | With 1-bar inputs the HMM split states by the sign of each bar (expected durations of 1.8–2.5 bars), not by regime. With smoothed inputs, durations are 15–70 bars. |
| A2 | Mean-shift drift: 2σ Mahalanobis | Any feature of a matched state moves more than 1.0 training std | Fitted state covariances are near-singular, so Mahalanobis flagged 0.45σ moves as 5.6σ. |
| A3 | Live LL alarm: below the in-sample 5th percentile | Below the worst 35-bar window seen in training | A 5% threshold checked on every bar makes a false alarm near-certain; one fired on day 1. |
| A4 | Labels from volatility rank only | Also: any state with vol > 2.5× the calmest state is at least STRESS. A label conflict means the label is inadmissible for the state's statistics. | A 5-state fit labeled a high-vol state CHOP (mean reversion at 0.5×). Near-tied calm states flipped CALM_UP/CHOP between refits, raising false alarms. |
| A5 | Refit: fit from scratch | Warm-start EM from the current model (or the pending candidate). A random restart must beat it by more than 1% LL. K is fixed at refits. | Cold refits reshuffled rarely visited states every month. |
| A6 | — | The transition matrix is floored at 1e-6, and the filter runs in log space | An exact-zero transition made the filter's normalizer underflow and crash on a crash-like bar. |
| A7 | Calibration "per state" | Calibration is scored on the observable next-bar return forecast: the PIT reliability of the next-state-weighted mixture must be within ±0.10 at the deciles, plus Brier on P(up). It uses OOS history only, and sizing is zero for a label until that label passes. | True states are never observed. |
| A8 | Beats a baseline | Higher Sharpe after costs, with total return reported alongside. Plus a minimum of 30 trades. | At a 20% cap, raw return can never beat SPY buy-and-hold. With few trades, the statistics are meaningless. |
| A9 | — | Backtest-only stand-ins for humans: approvals are granted, and a drift freeze lifts on a clean refit or when two consecutive refits agree. Live, a human does both. | The backtest needs a deterministic rule. |
| A10 | Kelly | Per label, from that label's closed OOS trades (≥ 20 trades). The prior until then is 0.4, which quarter-Kelly makes 10%. | The data must come from past trades only. |
| A11 | Venue: Alpaca paper, SPY, 1h regular trading hours | BTC/USDT 1h from Binance's keyless public market-data host (`data-api.binance.vision`). Paper trading runs on our own SimBroker fed with live public candles. 24/7 session: no overnight cap, daily loss resets at UTC midnight, Sharpe annualized over 365 days. Quantity step 1e-5 BTC. Costs 10 bps commission + 2 bps slippage per side. | Owner asked for a source usable without keys. Equity sources (Yahoo, Stooq) are rate-limited from this environment, and Alpaca needs keys. A self-run paper broker also has no ability to move money. |
