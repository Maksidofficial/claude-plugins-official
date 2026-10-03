# regime-bot — Build Plan (Gate 3 of 3: approved)

Spec (approved): `docs/SPEC.md`. Architecture (approved): `docs/ARCHITECTURE.md`.
Built in Claude Code; the Layer 1 model is Claude Opus 5.5 only.

## How every module ships

1. **Red:** write the tests from the acceptance list below. Run them and show them failing.
2. **Green:** write the minimal code that makes them pass.
3. **Review against the spec:** a checklist of the module's spec clauses, run
   `/code-review`, then `ruff` + `mypy --strict` + full `pytest`.
4. **Ship:** one commit per module, tagged `m<N>-<name>`. Rollback is `git revert` of that
   commit; later modules depend only on tagged ones.

CI (`.github/workflows/regime-bot.yml`) runs ruff, mypy and pytest on every push touching
`regime-bot/`. Unit tests use synthetic data, with no keys and no network.

## Modules (build order)

| # | Module | Failing tests written first (acceptance) |
|---|---|---|
| M0 | Scaffold: `pyproject.toml`, `.env.example`, `.gitignore`, CI | `.env` is gitignored. Test that `.env.example` has keys with empty values. A log-redaction test proves no secret value appears in a log line. |
| M1 | `limits.py` | Constants match SPEC §7. AST test: no `open`/`os.environ`/`toml`/config imports. Values cannot be reassigned at runtime (frozen dataclass raises). |
| M2 | `clock.py`, `data/bars.py` | Candle with `closed_at > now` is rejected. Staleness > 2 bars → `Stale`. Gaps and NaNs are detected. Cache round-trip is identical. |
| M3 | `data/features.py` | All 5 features are hand-computed on a toy series. Standardizer at t uses only ≤ t-1. Truncation test: feature(t) is identical when future rows are deleted. |
| M4 | `hmm/fit.py` | On data simulated from a known 3-state HMM, it selects K=3 and recovers the means within tolerance. Same seed gives an identical model. The tie rule picks the smaller K. Duration = 1/(1−A_ii). |
| M5 | `hmm/filter.py` | `step` matches hmmlearn's forward pass to 1e-10. Next-state = αA. **No-lookahead property test** (Hypothesis): replacing candles > k with noise leaves every output ≤ k bit-identical. AST ban on `predict`/`predict_proba`/`decode` outside `fit.py`. |
| M6 | `hmm/labels.py`, `hmm/drift.py` | Stats → CALM_UP/CHOP/STRESS/CRASH. Permuted refit → Hungarian restores labels. Each drift trigger fires on a crafted case and is silent on an identical refit. |
| M7 | `decide/switcher.py` | Table-driven cases for each of the 6 rules: 0.69 doesn't switch; 0.71×2 bars doesn't; 0.71×3 does; cooldown blocks; gap < 0.15 → uncertain; error → FLAT. No flip-flop on an oscillating input. Every switch carries probs + reason. |
| M8 | `decide/sizing.py` | ¼-Kelly cap. Monotone decreasing in entropy. 0 in CRASH/uncertain/uncalibrated. Early cut ×0.5 at P(next stress) > 0.25. |
| M9 | `decide/playbook.py` + seed `playbooks/*.md` | Valid file parses to a typed object. Missing field / non-numeric / prompt-like text → reject → state goes FLAT. `max_size` above the hard limit is clamped. |
| M10 | `decide/risk.py` | Each veto blocks: max position, daily loss, max DD → kill, state leverage, approval > $5k. Kill writes `KILLED`, flattens, survives restart, and clears only manually. |
| M11 | `engine.py` | Pure function: same inputs → same outputs. End-to-end on a synthetic regime series: switches happen, vetoes are logged. Engine state JSON round-trip. |
| M12 | `backtest/*` + `exec/broker.SimBroker` | Fill at open(t+1) + slippage, never close(t). Costs are applied per side. Metrics match a hand-checked example. Gates return pass/fail per gate. Brier/reliability are correct on known data. Buy-and-hold and static baselines are computed. |
| M13 | `exec/broker.AlpacaBroker`, `exec/approvals.py`, `live.py` | Against a mocked Alpaca: bracket stop is always attached; reconciliation mismatch → FLAT. Keys are read only from env. Live loop calls the same `engine.on_candle`. |
| M14 | `ops/journal`, `alerts`, `report`, `dashboard` | Journal is append-only. An alert is fired for fill/error/switch/drift/kill. Report has all SPEC §11 fields. Dashboard SSE streams the latest decision and is read-only. |
| M15 | `research/nightly.py`, `research/validate.py` | With a mocked Opus 5.5 client: output lands only in `proposals/`. Validate rejects a proposal that fails any gate and promotes only on all-pass. Import test: trading path never imports `research`. API down → night skipped, no crash. |
| M16 | `deploy/` | systemd unit lint. Watchdog: stale heartbeat → kill + alert (simulated). |
| M17 | Go-live checklist script | `regimebot preflight` runs every SPEC §12 check and exits non-zero on any failure. |

## Milestones needing you

| When | What I need |
|---|---|
| After M12 | Alpaca **paper** keys in `.env` on the machine running it (not in chat) to pull 3+ years of SPY 1h bars and run the real walk-forward. You see the gate report. If nothing passes, the bot stays flat. |
| After M14 | Telegram bot token + chat ID in `.env`. |
| After M15 | `ANTHROPIC_API_KEY` in `.env` for the nightly Opus 5.5 review. |
| After M16 | VPS access (or you run the deploy steps I write). |
| Before live | `preflight` all green plus your explicit approval. Paper first. |

## Where it lives

The build continues in `regime-bot/` on this branch. I recommend moving it to its own repo
before M13; it does not belong in the plugin marketplace.

## Estimated size

About 2.5k lines of code + 2k lines of tests. M0–M11 need no external access and can be
built entirely here.
