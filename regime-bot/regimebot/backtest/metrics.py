from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from regimebot.exec.broker import Fill

TRADING_DAYS = 252


@dataclass(frozen=True)
class Trade:
    entry_ts: str
    exit_ts: str
    state: str | None
    entry_price: float
    exit_price: float
    qty: float
    pnl: float  # after costs
    ret: float  # pnl / entry notional


@dataclass(frozen=True)
class Metrics:
    sharpe: float
    max_dd: float
    hit_rate: float
    t_stat: float
    total_return: float
    n_trades: int


def trades_from_fills(fills: list[Fill]) -> list[Trade]:
    """A trade runs from flat to flat. Adds and partial exits belong to the same trade."""
    out: list[Trade] = []
    pos = 0.0
    bought = bought_qty = sold = sold_qty = commission = 0.0
    entry_ts, state = "", None
    for f in fills:
        if pos == 0 and f.qty > 0:
            entry_ts, state = f.ts, f.state
            bought = bought_qty = sold = sold_qty = commission = 0.0
        if f.qty > 0:
            bought += f.qty * f.price
            bought_qty += f.qty
        else:
            sold += -f.qty * f.price
            sold_qty += -f.qty
        commission += f.commission
        pos += f.qty
        if abs(pos) < 1e-9 and bought_qty > 0:
            pnl = sold - bought - commission
            out.append(
                Trade(
                    entry_ts=entry_ts,
                    exit_ts=f.ts,
                    state=state,
                    entry_price=bought / bought_qty,
                    exit_price=sold / sold_qty,
                    qty=bought_qty,
                    pnl=pnl,
                    ret=pnl / bought,
                )
            )
            pos = 0.0
    return out


def daily_returns(equity: pd.Series) -> pd.Series:
    idx = pd.DatetimeIndex(equity.index)
    days = equity.groupby(idx.tz_convert("America/New_York").date if idx.tz else idx.date).last()
    return days.pct_change().dropna()


def max_drawdown(equity: pd.Series) -> float:
    e = equity.to_numpy(dtype=float)
    peak = np.maximum.accumulate(e)
    return float(np.max(1 - e / peak)) if len(e) else 0.0


def summarize(equity: pd.Series, trades: list[Trade]) -> Metrics:
    r = daily_returns(equity)
    sd = float(r.std(ddof=1)) if len(r) > 1 else 0.0
    mean = float(r.mean()) if len(r) else 0.0
    sharpe = mean / sd * math.sqrt(TRADING_DAYS) if sd > 0 else 0.0
    t = mean / (sd / math.sqrt(len(r))) if sd > 0 else 0.0
    wins = sum(1 for x in trades if x.pnl > 0)
    return Metrics(
        sharpe=sharpe,
        max_dd=max_drawdown(equity),
        hit_rate=wins / len(trades) if trades else 0.0,
        t_stat=t,
        total_return=float(equity.iloc[-1] / equity.iloc[0] - 1) if len(equity) else 0.0,
        n_trades=len(trades),
    )
