"""Historical simulation and performance statistics."""

from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd

from .broker import Trade
from .config import Config, timeframe_ms
from .engine import Engine
from .strategy import Bar, Signal


@dataclass
class BacktestResult:
    trades: list[Trade]
    signals: list[Signal]
    equity: pd.Series
    stats: dict

    def trades_frame(self) -> pd.DataFrame:
        return pd.DataFrame([t.__dict__ for t in self.trades])


def iter_bars(df: pd.DataFrame):
    has_liq = "liq_long" in df.columns and "liq_short" in df.columns
    for ts, row in zip(df.index, df.itertuples(index=False)):
        liq_l = liq_s = None
        if has_liq and not (pd.isna(row.liq_long) or pd.isna(row.liq_short)):
            liq_l, liq_s = float(row.liq_long), float(row.liq_short)
        yield Bar(ts, float(row.open), float(row.high), float(row.low), float(row.close), float(row.volume), liq_l, liq_s)


def run_backtest(df: pd.DataFrame, cfg: Config, initial_equity: float = 10_000.0) -> BacktestResult:
    engine = Engine(cfg, initial_equity, fill_on_close=False)
    trades: list[Trade] = []
    signals: list[Signal] = []
    curve = []
    last = None
    for bar in iter_bars(df):
        res = engine.step(bar)
        trades.extend(res.trades)
        if res.signal:
            signals.append(res.signal)
        curve.append(engine.broker.mark(bar.close))
        last = bar
    if last is not None:
        final = engine.finish(last)
        if final:
            trades.append(final)
            curve[-1] = engine.broker.equity
    equity = pd.Series(curve, index=df.index[: len(curve)], name="equity")
    return BacktestResult(trades, signals, equity, compute_stats(trades, equity, initial_equity, cfg.timeframe))


def compute_stats(trades: list[Trade], equity: pd.Series, initial: float, timeframe: str) -> dict:
    stats: dict = {"trades": len(trades), "initial_equity": initial}
    final = float(equity.iloc[-1]) if len(equity) else initial
    stats["final_equity"] = final
    stats["total_return_pct"] = (final / initial - 1) * 100
    if len(equity):
        peak = equity.cummax()
        stats["max_drawdown_pct"] = float(((equity / peak) - 1).min() * 100)
        rets = equity.pct_change().dropna()
        bars_per_year = 365 * 86_400_000 / timeframe_ms(timeframe)
        std = rets.std()
        stats["sharpe"] = float(rets.mean() / std * math.sqrt(bars_per_year)) if std and std > 0 else 0.0
    if not trades:
        return stats
    wins = [t for t in trades if t.pnl > 0]
    losses = [t for t in trades if t.pnl <= 0]
    gross_win = sum(t.pnl for t in wins)
    gross_loss = -sum(t.pnl for t in losses)
    stats.update(
        win_rate_pct=len(wins) / len(trades) * 100,
        profit_factor=gross_win / gross_loss if gross_loss > 0 else math.inf,
        avg_r=sum(t.r_multiple for t in trades) / len(trades),
        avg_bars_held=sum(t.bars_held for t in trades) / len(trades),
        fees_paid=sum(t.fees for t in trades),
        longs=sum(t.side == "long" for t in trades),
        shorts=sum(t.side == "short" for t in trades),
        exit_reasons=pd.Series([t.exit_reason for t in trades]).value_counts().to_dict(),
    )
    return stats


def format_stats(stats: dict) -> str:
    lines = []
    for k, v in stats.items():
        if isinstance(v, float):
            v = f"{v:,.2f}"
        lines.append(f"  {k:<18} {v}")
    return "\n".join(lines)
