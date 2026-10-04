"""Run the walk-forward on cached public data and write strategy.md.

    python -m regimebot.backtest.run --symbol BTCUSDT --oos 2022-01-01 --out strategy.md
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import UTC, datetime
from pathlib import Path

from regimebot.backtest.costs import CostModel
from regimebot.backtest.gates import Gates, evaluate
from regimebot.backtest.report import write_strategy
from regimebot.backtest.walkforward import WFConfig, run_walkforward
from regimebot.data.bars import load_cache, save_cache
from regimebot.data.binance import fetch_klines

ROOT = Path(__file__).resolve().parents[2]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--interval", default="1h")
    ap.add_argument("--start", default="2018-01-01")
    ap.add_argument("--oos", default="2022-01-01")
    ap.add_argument("--end", default=None, help="last candle used (default: all cached)")
    ap.add_argument("--refresh", action="store_true", help="re-download public data")
    ap.add_argument("--restarts-initial", type=int, default=10)
    ap.add_argument("--restarts-refit", type=int, default=3)
    ap.add_argument("--out", default=str(ROOT / "strategy.md"))
    a = ap.parse_args(argv)

    cache = ROOT / "data" / f"{a.symbol}_{a.interval}.parquet"
    if a.refresh or not cache.exists():
        start = datetime.fromisoformat(a.start).replace(tzinfo=UTC)
        save_cache(fetch_klines(a.symbol, a.interval, start), cache)
    bars = load_cache(cache)
    if a.end:
        bars = bars[bars["closed_at"] <= datetime.fromisoformat(a.end).replace(tzinfo=UTC)]
    bars = bars.reset_index(drop=True)

    cfg = WFConfig(
        oos_start=datetime.fromisoformat(a.oos).replace(tzinfo=UTC),
        state_dir=ROOT / "state" / "backtest",
        restarts_initial=a.restarts_initial,
        restarts_refit=a.restarts_refit,
        costs=CostModel(commission_bps=10.0, slippage_bps=2.0),
        qty_step=1e-5,
        session_close=None,
        tz="UTC",
        days_per_year=365,
    )
    kill = cfg.state_dir / "KILLED"
    kill.unlink(missing_ok=True)  # each backtest starts un-killed; live never does this
    t0 = time.time()
    res = run_walkforward(bars, cfg, ROOT / "playbooks")
    gate = evaluate(res.metrics, res.baselines, Gates())
    write_strategy(Path(a.out), res, gate)
    summary = {
        "passed": gate.passed,
        "failed": [r.name for r in gate.rows if not r.ok],
        "metrics": res.metrics.__dict__,
        "baselines": {k: v.__dict__ for k, v in res.baselines.items()},
        "seconds": round(time.time() - t0),
        "bars": len(bars),
    }
    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
