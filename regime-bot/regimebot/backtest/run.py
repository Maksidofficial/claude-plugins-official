"""Run the walk-forward on cached public data and write strategy.md.

    python -m regimebot.backtest.run --symbol BTCUSDT --oos 2022-01-01 --out strategy.md
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from regimebot.backtest.costs import CostModel
from regimebot.backtest.gates import GateResult, Gates, evaluate
from regimebot.backtest.report import write_strategy
from regimebot.backtest.walkforward import WFConfig, WFResult, run_walkforward
from regimebot.data.bars import load_cache, save_cache
from regimebot.data.binance import fetch_klines

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "data" / "BTCUSDT_1h.parquet"
STATE = ROOT / "state"


def config_for(oos: str = "2022-01-01", restarts_initial: int = 10,
               restarts_refit: int = 3) -> WFConfig:
    """The one BTCUSDT 1h configuration both the backtest and nightly validation use."""
    return WFConfig(
        oos_start=datetime.fromisoformat(oos).replace(tzinfo=UTC),
        state_dir=STATE / "backtest",
        restarts_initial=restarts_initial,
        restarts_refit=restarts_refit,
        costs=CostModel(commission_bps=10.0, slippage_bps=2.0),
        qty_step=1e-5,
        session_close=None,
        tz="UTC",
        days_per_year=365,
    )


def load_bars(end: str | None = None) -> pd.DataFrame:
    bars = load_cache(CACHE)
    if end:
        bars = bars[bars["closed_at"] <= datetime.fromisoformat(end).replace(tzinfo=UTC)]
    return bars.reset_index(drop=True)


def save_outputs(res: WFResult, gate: GateResult, out: Path) -> dict[str, object]:
    """strategy.md, the latest walk-forward model for live, live params, and a summary."""
    write_strategy(out, res, gate)
    (STATE / "models").mkdir(parents=True, exist_ok=True)
    (STATE / "models" / "current.json").write_text(json.dumps(res.final_model.to_dict()))
    by_state: dict[str, list[float]] = {}
    for t in res.trades:
        by_state.setdefault(t.state or "NONE", []).append(t.ret)
    kelly = {}
    for k, r in by_state.items():
        a = np.array(r)
        kelly[k] = float(np.clip(a.mean() / a.var(ddof=1), 0, 10)) if len(a) > 2 else 0.4
    params = {
        "symbol": "BTCUSDT",
        "initial_cash": 100_000.0,
        # a label sizes above zero live only if it passed OOS calibration and the gates
        "calibrated": {k: bool(v and gate.passed) for k, v in res.calibration.calibrated.items()},
        "kelly": kelly,
        "frozen": False,
    }
    (STATE / "live_params.json").write_text(json.dumps(params, indent=1))
    summary: dict[str, object] = {
        "passed": gate.passed,
        "failed": [r.name for r in gate.rows if not r.ok],
        "metrics": res.metrics.__dict__,
        "baselines": {k: v.__dict__ for k, v in res.baselines.items()},
        "swaps": [{"at": str(s["at"]), "accepted": s["accepted"], "alarms": s["alarms"]}
                  for s in res.swaps],
        "calibration": res.calibration.per_label,
        "per_state": res.per_state,
        "time_in_state": res.time_in_state,
        "notes": res.notes,
    }
    (STATE / "live").mkdir(parents=True, exist_ok=True)
    (STATE / "live" / "backtest_summary.json").write_text(json.dumps(summary, indent=1,
                                                                     default=str))
    return summary


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--start", default="2018-01-01")
    ap.add_argument("--oos", default="2022-01-01")
    ap.add_argument("--end", default=None, help="last candle used (default: all cached)")
    ap.add_argument("--refresh", action="store_true", help="re-download public data")
    ap.add_argument("--restarts-initial", type=int, default=10)
    ap.add_argument("--restarts-refit", type=int, default=3)
    ap.add_argument("--out", default=str(ROOT / "strategy.md"))
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    if a.refresh or not CACHE.exists():
        start = datetime.fromisoformat(a.start).replace(tzinfo=UTC)
        save_cache(fetch_klines(a.symbol, "1h", start), CACHE)
    bars = load_bars(a.end)
    cfg = config_for(a.oos, a.restarts_initial, a.restarts_refit)
    (cfg.state_dir / "KILLED").unlink(missing_ok=True)  # each backtest starts un-killed
    t0 = time.time()
    res = run_walkforward(bars, cfg, ROOT / "playbooks")
    gate = evaluate(res.metrics, res.baselines, Gates())
    summary = save_outputs(res, gate, Path(a.out))
    summary["seconds"] = round(time.time() - t0)
    summary["bars"] = len(bars)
    print(json.dumps({k: summary[k] for k in ("passed", "failed", "metrics", "baselines",
                                              "seconds", "bars")}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
