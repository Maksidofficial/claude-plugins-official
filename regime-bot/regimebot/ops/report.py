"""Daily report from the live journal: what the bot believed, did, and earned."""

from __future__ import annotations

import math
from collections import Counter
from typing import Any

import numpy as np
from scipy.stats import norm

from regimebot.backtest.calibration import calibration_report, pit
from regimebot.backtest.metrics import trades_from_fills
from regimebot.exec.broker import Fill
from regimebot.hmm.fit import RegimeModel
from regimebot.hmm.labels import state_stats


def daily_report(records: list[dict[str, Any]], model: RegimeModel,
                 day: str | None = None) -> dict[str, Any]:
    """``day`` (YYYY-MM-DD, UTC) limits activity stats to that day; state is always latest."""
    candles = [r for r in records if r.get("kind") == "candle"]
    scope = [r for r in candles if day is None or str(r["ts"]).startswith(day)]
    active = [r["result"]["decision"].get("active") or "NONE" for r in scope]
    n = max(len(active), 1)
    fills = [Fill(**f) for r in scope for f in r.get("fills", [])]
    trades = trades_from_fills(fills)
    pnl: dict[str, float] = {}
    for t in trades:
        pnl[t.state or "NONE"] = pnl.get(t.state or "NONE", 0.0) + t.pnl

    st = state_stats(model)
    mus = np.array([s.mean_ret for s in st])
    sds = np.array([s.vol for s in st])
    rows = []
    for a, b in zip(candles, candles[1:], strict=False):
        d = a["result"]["decision"]
        if d.get("state_next") and d.get("active") and "close" in a and "close" in b:
            nps = np.array(d["state_next"])
            if len(nps) != len(mus):
                continue
            r = math.log(b["close"] / a["close"])
            p_up = 1.0 - float(np.sum(nps * norm.cdf(-mus / sds)))
            rows.append((d["active"], pit(r, nps, mus, sds), p_up, int(r > 0)))
    cal = calibration_report(rows, min_n=200, tol=0.10)

    last = candles[-1] if candles else {}
    return {
        "day": day,
        "current_state": (last.get("result", {}).get("decision", {}).get("active")
                          if last else None),
        "time_in_state": {k: v / n for k, v in Counter(active).items()},
        "trades": len(trades),
        "pnl_per_state": pnl,
        "win_rate": (sum(t.pnl > 0 for t in trades) / len(trades)) if trades else None,
        "largest_loss": min((t.pnl for t in trades), default=0.0),
        "calibration": {k: {"n": v["n"], "max_dev": v["max_dev"], "brier": v["brier"],
                            "calibrated": cal.calibrated[k]} for k, v in cal.per_label.items()},
        "equity": last.get("equity"),
        "position": last.get("position_after"),
    }


def render_report(rep: dict[str, Any]) -> str:
    lines = [f"Daily report {rep['day'] or '(all time)'}",
             f"Current state: {rep['current_state']}",
             f"Equity: {rep['equity']}  Position: {rep['position']}",
             "Time in state: " + ", ".join(f"{k} {v:.0%}"
                                           for k, v in sorted(rep["time_in_state"].items())),
             f"Trades: {rep['trades']}  Win rate: "
             + ("n/a" if rep["win_rate"] is None else f"{rep['win_rate']:.0%}"),
             "P&L per state: "
             + (", ".join(f"{k} {v:+.2f}" for k, v in rep["pnl_per_state"].items()) or "none"),
             f"Largest loss: {rep['largest_loss']:.2f}",
             "Calibration: " + (", ".join(
                 f"{k} dev {v['max_dev']:.3f} brier {v['brier']:.3f} "
                 f"{'ok' if v['calibrated'] else 'NOT calibrated'} (n={int(v['n'])})"
                 for k, v in rep["calibration"].items()) or "not enough data")]
    return "\n".join(lines)
