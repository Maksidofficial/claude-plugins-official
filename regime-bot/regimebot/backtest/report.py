"""Write strategy.md from a walk-forward result. The harness writes it, never the model."""

from __future__ import annotations

from pathlib import Path

from regimebot.backtest.gates import GateResult
from regimebot.backtest.walkforward import WFResult
from regimebot.hmm.labels import state_stats


def _f(x: float) -> str:
    return f"{x:.4f}"


def render(res: WFResult, gate: GateResult) -> str:
    m = res.metrics
    head = "# strategy.md\n\n"
    if gate.passed:
        head += "## Status: ACCEPTED\n\nCleared every out-of-sample gate after costs.\n"
    else:
        failed = ", ".join(r.name for r in gate.rows if not r.ok)
        head += (
            "## Status: NO STRATEGY ACCEPTED\n\n"
            f"Failed gates: {failed}. The bot stays flat until a change re-clears every gate.\n"
        )
    lines = [head, "## Gates (out of sample, after costs)\n", "| gate | value | needs | ok |",
             "|---|---|---|---|"]
    lines += [f"| {r.name} | {_f(r.value)} | {r.threshold} | {'yes' if r.ok else 'NO'} |"
              for r in gate.rows]
    lines += ["", "## System vs baselines\n",
              "| | sharpe | max_drawdown | hit_rate | t_stat | total_return | trades |",
              "|---|---|---|---|---|---|---|"]
    for name, x in [("system", m), *res.baselines.items()]:
        lines.append(f"| {name} | {_f(x.sharpe)} | {_f(x.max_dd)} | {_f(x.hit_rate)} | "
                     f"{_f(x.t_stat)} | {_f(x.total_return)} | {x.n_trades} |")
    lines += ["", "## Per state\n", "| state | trades | win_rate | pnl | avg_ret | largest_loss |",
              "|---|---|---|---|---|---|"]
    for s, d in res.per_state.items():
        lines.append(f"| {s} | {int(d['trades'])} | {_f(d['win_rate'])} | {d['pnl']:.2f} | "
                     f"{_f(d['avg_ret'])} | {d['largest_loss']:.2f} |")
    lines += ["", "## Time in state\n"]
    lines += [f"- {k}: {v:.1%}" for k, v in sorted(res.time_in_state.items())]
    lines += ["", "## Calibration (PIT reliability, out of sample)\n",
              "| state | n | max_dev | brier | brier_climatology | calibrated |",
              "|---|---|---|---|---|---|"]
    for s, d in res.calibration.per_label.items():
        lines.append(f"| {s} | {int(d['n'])} | {_f(d['max_dev'])} | {_f(d['brier'])} | "
                     f"{_f(d['brier_climatology'])} | {res.calibration.calibrated[s]} |")
    fm = res.final_model
    lines += ["", f"## Model (final, K={fm.k})\n", "| state | label | mean_ret | vol | duration |",
              "|---|---|---|---|---|"]
    for i, st in enumerate(state_stats(fm)):
        lines.append(f"| {i} | {fm.labels[i]} | {st.mean_ret:.6f} | {st.vol:.5f} | "
                     f"{st.duration:.1f} |")
    lines += ["", "Transition matrix:", "", "```"]
    lines += ["  ".join(f"{p:.3f}" for p in row) for row in fm.transmat]
    lines += ["```", "", "K selection:", ""]
    lines += [f"- K={s.k}: OOS LL {s.oos_ll:.1f}, BIC {s.bic:.1f}" for s in res.selection.scores]
    accepted = sum(s["accepted"] for s in res.swaps)
    lines += ["", f"## Refits: {len(res.swaps)} ({accepted} accepted)\n"]
    lines += [f"- {s['at']}: accepted={s['accepted']} alarms={s['alarms']}" for s in res.swaps]
    lines += ["", "## Notes\n"] + [f"- {n}" for n in res.notes]
    return "\n".join(lines) + "\n"


def write_strategy(path: Path, res: WFResult, gate: GateResult) -> None:
    path.write_text(render(res, gate))
