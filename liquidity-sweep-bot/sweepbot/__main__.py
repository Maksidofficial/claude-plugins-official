"""Command line entry point: ``python -m sweepbot <command>``."""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys

from .backtest import format_stats, run_backtest
from .config import Config


def _load_config(path: str | None, symbol: str | None, timeframe: str | None) -> Config:
    cfg = Config.load(path) if path else Config()
    if symbol:
        cfg.symbol = symbol.upper()
    if timeframe:
        cfg.timeframe = timeframe
    return cfg


def cmd_backtest(args) -> int:
    from . import data

    cfg = _load_config(args.config, args.symbol, args.timeframe)
    if args.demo:
        from .synthetic import make_sweep_series

        df = make_sweep_series()
        cfg.timeframe = "15m"
    elif args.csv:
        df = data.load_ohlcv_csv(args.csv)
    else:
        if not args.start:
            print("backtest needs --csv, --demo, or --start (to download from Binance)", file=sys.stderr)
            return 2
        if args.source == "archive":
            df = data.fetch_binance_archive(cfg.symbol, cfg.timeframe, args.start, args.end)
        else:
            df = data.fetch_binance_klines(cfg.symbol, cfg.timeframe, args.start, args.end)
        if args.save:
            df.to_csv(args.save, index_label="time")
    if args.liquidations:
        df = data.merge_liquidations(df, data.load_liquidations_csv(args.liquidations, cfg.timeframe))

    res = run_backtest(df, cfg, args.equity)
    print(f"{cfg.symbol} {cfg.timeframe}  {df.index[0]} -> {df.index[-1]}  ({len(df)} bars, {len(res.signals)} signals)")
    print(format_stats(res.stats))
    if args.verbose:
        for s in res.signals:
            print(f"  {s.time}  {s.describe()}")
    if args.trades:
        res.trades_frame().to_csv(args.trades, index=False)
        print(f"trades written to {args.trades}")
    return 0


def cmd_live(args) -> int:
    from .live import run_live

    if args.mode == "live" and not args.i_understand_the_risks:
        print("Refusing to trade real money without --i-understand-the-risks", file=sys.stderr)
        return 2
    cfg = _load_config(args.config, args.symbol, args.timeframe)
    try:
        asyncio.run(run_live(cfg, args.mode, args.equity, args.record))
    except KeyboardInterrupt:
        pass
    return 0


def cmd_record(args) -> int:
    from .live import record_liquidations

    try:
        asyncio.run(record_liquidations(args.symbol.upper(), args.timeframe, args.out))
    except KeyboardInterrupt:
        pass
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="sweepbot", description="Liquidity-sweep + liquidation-cascade reversal bot")
    sub = p.add_subparsers(dest="command", required=True)

    def common(sp):
        sp.add_argument("--config", help="TOML config file (see config.example.toml)")
        sp.add_argument("--symbol", help="e.g. BTCUSDT")
        sp.add_argument("--timeframe", help="e.g. 5m, 15m, 1h")
        sp.add_argument("--equity", type=float, default=10_000.0)
        sp.add_argument("-v", "--verbose", action="store_true", help="print every signal / debug logging")

    bt = sub.add_parser("backtest", help="simulate on historical data")
    common(bt)
    bt.add_argument("--csv", help="OHLCV CSV (time,open,high,low,close,volume[,liq_long,liq_short])")
    bt.add_argument("--demo", action="store_true", help="use built-in synthetic data")
    bt.add_argument("--start", help="download Binance futures klines from this date")
    bt.add_argument("--end", help="... up to this date (default: now)")
    bt.add_argument("--source", choices=["archive", "api"], default="archive",
                    help="data.binance.vision bulk archive (default) or the fapi REST endpoint")
    bt.add_argument("--save", help="save downloaded klines to this CSV")
    bt.add_argument("--liquidations", help="raw forced-order CSV (from `record`) to merge in")
    bt.add_argument("--trades", help="write the trade list to this CSV")
    bt.set_defaults(func=cmd_backtest)

    lv = sub.add_parser("live", help="run on live Binance futures data")
    common(lv)
    lv.add_argument("--mode", choices=["paper", "testnet", "live"], default="paper")
    lv.add_argument("--record", help="also append forced orders to this CSV")
    lv.add_argument("--i-understand-the-risks", action="store_true")
    lv.set_defaults(func=cmd_live)

    rc = sub.add_parser("record", help="record Binance forced orders for future backtests")
    rc.add_argument("--symbol", default="BTCUSDT")
    rc.add_argument("--timeframe", default="15m")
    rc.add_argument("--out", default="liquidations.csv")
    rc.add_argument("-v", "--verbose", action="store_true")
    rc.set_defaults(func=cmd_record)

    args = p.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
