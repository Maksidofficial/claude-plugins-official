"""Baselines the system must beat after the same costs."""

from __future__ import annotations

import math

import pandas as pd

from regimebot.backtest.costs import CostModel
from regimebot.backtest.metrics import Metrics, Trade, summarize, trades_from_fills
from regimebot.data.bars import frame_to_candles
from regimebot.data.features import FeatureEngine
from regimebot.decide.playbook import Playbook
from regimebot.engine import STRETCH_WINDOW, TREND, OrderAction, entry_signal, stretch
from regimebot.exec.broker import Fill, SimBroker


def buy_and_hold(bars: pd.DataFrame, cash: float, costs: CostModel) -> tuple[pd.Series, Metrics]:
    first = float(bars["open"].iloc[0])
    px = costs.fill_price(first, 1)
    qty = cash / (px * (1 + costs.commission_bps / 1e4))
    rest = cash - qty * px - costs.commission(px, qty)
    eq = rest + qty * bars["close"].astype(float)
    last = float(bars["close"].iloc[-1])
    out_px = costs.fill_price(last, -1)
    final = rest + qty * out_px - costs.commission(out_px, qty)
    eq.iloc[-1] = final
    eq.index = pd.DatetimeIndex(bars["closed_at"])
    t = Trade(str(bars["opened_at"].iloc[0]), str(bars["closed_at"].iloc[-1]), None,
              px, out_px, qty, final - cash, final / cash - 1)
    return eq, summarize(eq, [t])


def static_playbook(
    bars: pd.DataFrame, pb: Playbook, cash: float, costs: CostModel
) -> tuple[pd.Series, Metrics, list[Fill]]:
    """Run one playbook as if its regime were always active, at its max size, no HMM."""
    broker = SimBroker(cash, costs)
    fe = FeatureEngine()
    log_closes: list[float] = []
    held = 0
    eq = []
    for c in frame_to_candles(bars):
        broker.on_bar(c)
        x = fe.update(c)
        log_closes = (log_closes + [math.log(c.close)])[-STRETCH_WINDOW:]
        held = held + 1 if broker.qty > 0 else 0
        if x is not None:
            sig = {"trend": float(x[TREND]), "stretch": stretch(log_closes)}
            if broker.qty > 0:
                out = (
                    (pb.style == "trend_following" and sig["trend"] < pb.exit_threshold)
                    or (pb.style == "mean_reversion" and sig["stretch"] > pb.exit_threshold)
                    or held >= pb.max_hold_bars
                )
                if out:
                    broker.submit([OrderAction(-broker.qty, c.close, None, None, "exit")],
                                  c.closed_at.isoformat(), pb.state)
            elif entry_signal(pb, sig):
                qty = math.floor(pb.max_size * broker.equity(c.close) / c.close)
                if qty > 0:
                    broker.submit(
                        [OrderAction(qty, c.close, c.close * (1 - pb.stop_pct),
                                     c.close * (1 + pb.take_profit_pct), "entry")],
                        c.closed_at.isoformat(), pb.state,
                    )
        eq.append(broker.equity(c.close))
    series = pd.Series(eq, index=pd.DatetimeIndex(bars["closed_at"]))
    return series, summarize(series, trades_from_fills(broker.fills)), broker.fills
