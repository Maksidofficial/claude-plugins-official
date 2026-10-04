"""Acceptance gates (SPEC §9). Out of sample, after costs. Never loosened to get a pass."""

from __future__ import annotations

from dataclasses import dataclass

from regimebot.backtest.metrics import Metrics


@dataclass(frozen=True)
class Gates:
    sharpe: float = 1.5
    max_dd: float = 0.15
    hit_rate: float = 0.55
    t_stat: float = 2.0
    min_trades: int = 30  # below this none of the statistics mean anything


@dataclass(frozen=True)
class GateRow:
    name: str
    value: float
    threshold: str
    ok: bool


@dataclass(frozen=True)
class GateResult:
    passed: bool
    rows: list[GateRow]


def evaluate(system: Metrics, baselines: dict[str, Metrics], g: Gates) -> GateResult:
    """A baseline is beaten when the system's Sharpe after costs is strictly higher."""
    rows = [
        GateRow("sharpe", system.sharpe, f"> {g.sharpe}", system.sharpe > g.sharpe),
        GateRow("max_drawdown", system.max_dd, f"< {g.max_dd}", system.max_dd < g.max_dd),
        GateRow("hit_rate", system.hit_rate, f"> {g.hit_rate}", system.hit_rate > g.hit_rate),
        GateRow("t_stat", system.t_stat, f"> {g.t_stat}", system.t_stat > g.t_stat),
        GateRow("min_trades", system.n_trades, f">= {g.min_trades}",
                system.n_trades >= g.min_trades),
    ]
    for name, b in baselines.items():
        label = "static" if name.startswith("static") else name
        ok = system.sharpe > b.sharpe
        rows.append(GateRow(f"beats_{label}", system.sharpe, f"> {b.sharpe:.3f} ({name})", ok))
    return GateResult(all(r.ok for r in rows), rows)
