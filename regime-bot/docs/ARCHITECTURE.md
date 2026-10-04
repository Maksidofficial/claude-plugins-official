# regime-bot — Architecture (Gate 2 of 3: approved)

Spec: `docs/SPEC.md` (approved). Status: APPROVED (gate 2).

## 1. Repository layout

```
regime-bot/
├── pyproject.toml            # pinned deps, pytest/ruff/mypy config
├── .env.example              # ALPACA_KEY_ID, ALPACA_SECRET, TELEGRAM_TOKEN, TELEGRAM_CHAT_ID, ANTHROPIC_API_KEY
├── .gitignore                # .env, data/, state/, logs/
├── config/
│   └── params.toml           # tunable thresholds (switching, refit, costs). Not limits.
├── regimebot/
│   ├── limits.py             # L3  HARD LIMITS: frozen constants, no I/O, no imports from L1
│   ├── clock.py              # L3  single source of "now"; Candle.closed_at contract
│   ├── data/
│   │   ├── bars.py           # Candle type, Alpaca fetch, local parquet cache, staleness check
│   │   └── features.py       # L3  5 features + expanding standardizer (past-only)
│   ├── hmm/
│   │   ├── fit.py            # L2  K selection (BIC + OOS LL), restarts, seed
│   │   ├── filter.py         # L2  own forward filter: step(alpha, x) -> alpha', next_probs
│   │   ├── labels.py         # L2  auto-labelling + Hungarian matching across refits
│   │   └── drift.py          # L2  drift metrics -> DriftReport
│   ├── decide/
│   │   ├── switcher.py       # L3  hysteresis state machine (pure)
│   │   ├── sizing.py         # L3  ¼-Kelly × P × entropy factor (pure)
│   │   ├── playbook.py       # L3  parse playbooks/*.md -> typed Playbook; reject on error
│   │   └── risk.py           # L3  pre-trade veto chain + kill switch
│   ├── exec/
│   │   ├── broker.py         # Broker protocol; AlpacaBroker, SimBroker (backtest)
│   │   └── approvals.py      # manual approval queue (> $ threshold) via Telegram
│   ├── engine.py             # L3  one function: on_candle(state, candle) -> (state', actions)
│   ├── backtest/
│   │   ├── walkforward.py    # drives engine.on_candle with SimBroker + scheduled refits
│   │   ├── costs.py          # commission + slippage model
│   │   ├── metrics.py        # Sharpe, MaxDD, hit rate, t-stat, per-state P&L
│   │   ├── gates.py          # acceptance gates -> GateResult (pass/fail per gate)
│   │   └── calibration.py    # Brier + reliability per state
│   ├── research/             # L1 — offline only, never imported by engine/exec
│   │   ├── model.py          # Claude Opus 5.5 client (Anthropic SDK)
│   │   ├── nightly.py        # gather session -> prompt -> proposals/ (files only)
│   │   └── validate.py       # harness: apply proposal on a branch, rerun gates, accept/reject
│   ├── ops/
│   │   ├── alerts.py         # Telegram sender (redacts secrets)
│   │   ├── report.py         # daily report
│   │   ├── journal.py        # append-only JSONL decision log (state/journal/*.jsonl)
│   │   └── dashboard.py      # FastAPI + SSE, read-only over journal
│   └── live.py               # live loop: wait for candle close -> engine -> broker
├── playbooks/                # CALM_UP.md, CHOP.md, STRESS.md, CRASH.md (L1 writes, L3 parses)
├── rules/losses.md           # one rule per loss (L1 appends, human reviews)
├── strategy.md               # accepted strategy or "NO STRATEGY ACCEPTED"
├── deploy/
│   ├── regimebot.service       # systemd, Restart=always
│   ├── regimebot-nightly.timer # nightly research + report
│   └── watchdog.sh           # heartbeat check -> kill switch + alert
└── tests/
```

## 2. Layer boundaries, enforced in code and tests

| Boundary | Enforcement |
|---|---|
| L1 never touches the trading path | `tests/test_imports.py` uses an AST scan to assert that nothing under `engine`, `decide`, `exec`, `live` imports `regimebot.research`. |
| L1 output is files only | Research writes to `proposals/`. Only `validate.py` can promote a proposal, and only after the gates pass. |
| Hard limits are not writable | `limits.py` holds module-level `Final` constants and a frozen dataclass. No loader reads them from config or env. A test asserts that `limits.py` has no `open`, `os.environ` or `toml` usage. |
| Playbooks can only tighten | `risk.py` computes `effective = min(playbook.max_size, limits.MAX_POSITION_PCT, limits.STATE_LEVERAGE[state])`. |
| Text is never treated as instructions | Playbook parsing is strict schema (numbers and enums). Unparseable → that state goes flat. News is never used as input. |

## 3. The decision path (one candle)

```
candle closes at t
  │
  ▼
bars.validate(candle)          stale / gap / NaN?  ──yes──► FLAT + alert
  │
  ▼
features.update(x_t)           uses buffer ≤ t, standardizer fit on ≤ t-1
  │
  ▼
filter.step(alpha_{t-1}, x_t)  -> alpha_t (P state now), alpha_t·A (P state next)
  │                             exception? ──► FLAT + alert
  ▼
switcher.step(...)             hysteresis: >0.70, 3 bars, cooldown, uncertainty gap
  │                             -> active_state, reason
  ▼
sizing.target(...)             ¼-Kelly × P(active) × entropy factor; early-cut ×0.5
  │
  ▼
playbook rules                 entry/exit/stop/TP for active_state -> proposed order
  │
  ▼
risk.check(order, account)     max pos, daily loss, maxDD, leverage, kill, approval
  │                             any veto ──► drop order, log reason
  ▼
broker.submit(order)           earliest fill: open(t+1); bracket stop placed at broker
  │
  ▼
journal.append(everything)     probs, state, reason, order, veto, fill
```

`engine.on_candle` is a pure function of `(EngineState, Candle) -> (EngineState, list[Action])`.
Live and backtest call the same function. Only the broker differs (Alpaca vs SimBroker), and
that is what lets paper trading be compared to the backtest line for line.

## 4. No-lookahead design

- **Candle contract:** `Candle.closed_at` is mandatory, and the engine refuses a candle whose
  `closed_at` is later than `clock.now()`.
- **Structural guarantee:** the filter is incremental (`step`). There is no API that takes a
  full series and returns decisions, so smoothing has nowhere to happen. `hmmlearn.predict`
  and `predict_proba` are banned from `regimebot/` outside `fit.py` (AST test).
- **Fills:** SimBroker fills market orders at `open(t+1)` plus slippage, never at `close(t)`.
- **Refits:** a refit at t uses data ≤ t, and the new model takes effect from candle t+1.
- **Proof test:** run the engine on N candles, then on candles 1..k with k+1..N replaced by
  random noise. Every action ≤ k must be identical. Property-tested over many k with
  Hypothesis.

## 5. State and persistence

| Data | Store | Why |
|---|---|---|
| Bars cache | parquet in `data/` | Reproducible backtests |
| Model artifacts | `state/models/<date>.pkl` + JSON summary + sha256 | Rollback to any prior model |
| Engine state | `state/engine.json`, written atomically (tmp + rename) every candle | Restart resumes alpha, cooldown, positions |
| Decisions | append-only JSONL | Audit, dashboard, nightly review |
| Kill flag | `state/KILLED` file | Survives restarts; only a human deletes it |

On startup the bot reconciles positions with Alpaca. On mismatch it goes flat and sends an
alert.

## 6. Processes (VPS)

| Unit | Runs | Restart |
|---|---|---|
| `regimebot.service` | `live.py` + dashboard (localhost only, reached over an SSH tunnel) | `Restart=always`, `RestartSec=10` |
| `regimebot-nightly.timer` | after close: report → research → validate | oneshot |
| `regimebot-watchdog.timer` | every minute: heartbeat older than 3 min during RTH → kill + alert | oneshot |

Broker-side bracket stops mean a dead process still has protective stops in place.

## 7. Refit flow

`fit.py` (expanding window) → `labels.match` (Hungarian on mean/vol) → `drift.compare(old, new)`.

- If no drift: the new model is promoted at the next candle.
- If drift: the old model is kept, new entries are frozen, an alert is sent, and a human
  decides.

Every model swap is journaled with both summaries.

## 8. Nightly L1 flow

`nightly.py` collects the journal, fills, wrong-state calls and the current playbooks, then
calls Opus 5.5. Output goes to `proposals/<date>/` as diffs plus a rationale.
`validate.py` then:

1. applies the diffs to a temp worktree
2. re-runs the full walk-forward
3. runs `gates.py`

If every gate passes, the proposal is promoted (git commit, tagged). If not, it is rejected
with the reason logged. L1 never sees the gate code's verdict before it writes its proposal,
and never grades its own output.

Model: `claude-opus-5-5` (Claude only) via the Anthropic SDK. If the API is unavailable, the night is skipped and trading is
unaffected.

## 9. Rollback

- **Code:** every module ships behind its own PR and tag, so rollback is `git revert` plus a
  service restart.
- **Model:** `state/models/` keeps every artifact; `regimebot rollback-model <date>` loads one.
- **Playbooks and strategy:** git-tracked, and every promotion is a tagged commit.
- **Live:** the kill switch always flattens first.

## 10. External dependencies

| Library | Purpose |
|---|---|
| `hmmlearn` | Fitting only |
| `numpy`, `pandas`, `scipy` | Hungarian matching, stats |
| `alpaca-py` | Market data and broker |
| `anthropic` | L1 only |
| `fastapi`, `uvicorn` | Dashboard |
| `httpx` | Telegram |
| `pytest`, `hypothesis`, `ruff`, `mypy` | Dev |

Versions are pinned, and no package is installed from an unverified source.

## 11. Open risks in this architecture

- One-process design: a hung (not crashed) process is caught only by the watchdog,
  within 3 minutes.
- Alpaca paper fills are optimistic, so paper-vs-backtest agreement may overstate live
  performance. Live slippage is tracked separately.
- The 1h RTH session has a 17.5h overnight gap. Overnight risk caps belong in playbooks and
  in `limits.MAX_OVERNIGHT_PCT`.
