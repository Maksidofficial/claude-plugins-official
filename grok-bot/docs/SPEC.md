# grok-bot — Specification (Gate 1 of 3: approved)

Status: APPROVED (gate 1).
Every value marked **[default]** was chosen because the brief left it open; change any of them before approving.

## 1. Scope

A regime-aware trading bot. A Gaussian HMM reads the market state on every candle, a
reasoning model (Layer 1) writes and revises one playbook per state each night, and
deterministic code owns every decision and every order.

| Item | Value |
|---|---|
| Name | grok-bot |
| Layer 1 model | Opus 5.5 (`claude-opus-5-5`), called behind a `ResearchModel` interface so another provider (e.g. Grok) can be swapped in without touching Layers 2–3 **[default]** |
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
