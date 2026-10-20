"""Live loop: closed klines via REST + liquidations via websocket.

Modes:
  paper    simulated fills (default, no keys)
  testnet  real orders on the Binance futures testnet
  live     real orders with real money
"""

from __future__ import annotations

import asyncio
import logging
import time

import pandas as pd

from .backtest import iter_bars
from .broker import CcxtBroker, position_size
from .config import Config, timeframe_ms
from .data import LiquidationStream, fetch_binance_klines
from .engine import Engine
from .strategy import Bar

log = logging.getLogger(__name__)


async def run_live(cfg: Config, mode: str = "paper", equity: float = 10_000.0, record_path: str | None = None) -> None:
    tf_ms = timeframe_ms(cfg.timeframe)
    stream = LiquidationStream(cfg.symbol, cfg.timeframe, record_path)
    stream_task = asyncio.create_task(stream.run())

    engine = Engine(cfg, equity, fill_on_close=True)
    exchange = CcxtBroker(cfg.symbol, cfg.risk, testnet=(mode == "testnet")) if mode in ("testnet", "live") else None

    warm = await asyncio.to_thread(fetch_binance_klines, cfg.symbol, cfg.timeframe, limit=1000)
    for bar in iter_bars(warm):
        engine.strategy.on_bar(bar)  # history only: build levels/ATR, never trade it
    engine.index = engine.strategy.index
    last_open = int(warm.index[-1].timestamp() * 1000) if len(warm) else 0
    log.info("warmed up on %d bars; %d high / %d low pools mapped (mode=%s)",
             len(warm), len(engine.strategy.levels.highs), len(engine.strategy.levels.lows), mode)

    try:
        while True:
            now = int(time.time() * 1000)
            await asyncio.sleep(((now // tf_ms + 1) * tf_ms - now) / 1000 + 2)
            try:
                recent = await asyncio.to_thread(fetch_binance_klines, cfg.symbol, cfg.timeframe, limit=5)
            except Exception as e:
                log.warning("kline fetch failed: %s", e)
                continue
            for ts, row in recent.iterrows():
                open_ms = int(ts.timestamp() * 1000)
                if open_ms <= last_open:
                    continue
                last_open = open_ms
                liq_l, liq_s = stream.take(open_ms)
                bar = Bar(ts, row.open, row.high, row.low, row.close, row.volume, liq_l, liq_s)
                if exchange is None:
                    _paper_step(engine, bar)
                else:
                    _exchange_step(engine, exchange, bar)
    finally:
        stream_task.cancel()


def _paper_step(engine: Engine, bar: Bar) -> None:
    res = engine.step(bar)
    if res.signal:
        log.info("SIGNAL %s", res.signal.describe())
    if res.skipped:
        log.info("  skipped: %s", res.skipped)
    for t in res.trades:
        log.info("EXIT %s %s @ %.2f (%s) pnl=%.2f R=%.2f", t.side, t.qty, t.exit, t.exit_reason, t.pnl, t.r_multiple)
    if res.opened:
        p = engine.broker.position
        log.info("ENTRY %s qty=%.6f @ %.2f SL %.2f TP %.2f", p.side, p.qty, p.entry, p.stop, p.target)
    log.info("bar %s close=%.2f equity=%.2f", pd.Timestamp(bar.time).isoformat(), bar.close, engine.broker.mark(bar.close))


def _exchange_step(engine: Engine, ex: CcxtBroker, bar: Bar) -> None:
    """Exchange-side SL/TP manage exits; the bot only opens and flips."""
    sig = engine.strategy.on_bar(bar)
    engine.index = engine.strategy.index
    if sig is None:
        return
    log.info("SIGNAL %s", sig.describe())
    current = ex.current_side()
    if current == sig.side:
        log.info("  skipped: already %s", current)
        return
    if current is not None:
        if not engine.cfg.trade.flip_on_opposite:
            log.info("  skipped: in opposite position, flipping disabled")
            return
        log.info("  flipping out of %s", current)
        ex.flatten()
    qty = position_size(ex.equity, bar.close, sig.stop, engine.cfg.risk)
    try:
        ex.open(sig.side, qty, sig.stop, sig.target)
    except Exception as e:
        log.error("order failed: %s", e)
        ex.flatten()


async def record_liquidations(symbol: str, timeframe: str, path: str) -> None:
    stream = LiquidationStream(symbol, timeframe, path)
    log.info("recording %s forced orders to %s (Ctrl-C to stop)", symbol, path)
    await stream.run()
