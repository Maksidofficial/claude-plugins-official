import tempfile
import unittest
from pathlib import Path

import pandas as pd

from sweepbot.backtest import iter_bars, run_backtest
from sweepbot.broker import PaperBroker, position_size
from sweepbot.cascade import CascadeDetector
from sweepbot.config import CascadeConfig, Config, LevelConfig, RiskConfig
from sweepbot.data import load_liquidations_csv, merge_liquidations
from sweepbot.levels import LiquidityMap
from sweepbot.strategy import Bar, SweepStrategy
from sweepbot.synthetic import make_sweep_series


def demo_config(**trade) -> Config:
    cfg = Config(timeframe="15m")
    cfg.levels.min_touches = 1
    cfg.cascade.mode = "liquidations"
    cfg.trade.min_rr = 1.5
    for k, v in trade.items():
        setattr(cfg.trade, k, v)
    return cfg


class ConfigTests(unittest.TestCase):
    def test_load_toml_overrides_nested_values(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "c.toml"
            p.write_text('symbol = "ETHUSDT"\n[trade]\nmin_rr = 3.0\n')
            cfg = Config.load(p)
        self.assertEqual(cfg.symbol, "ETHUSDT")
        self.assertEqual(cfg.trade.min_rr, 3.0)
        self.assertEqual(cfg.trade.stop_buffer_atr, Config().trade.stop_buffer_atr)

    def test_unknown_key_rejected(self):
        with self.assertRaises(ValueError):
            Config.from_dict({"trade": {"min_rrr": 2}})

    def test_example_config_loads(self):
        Config.load(Path(__file__).parent.parent / "config.example.toml")


class LevelTests(unittest.TestCase):
    def test_pivot_confirmed_only_after_right_bars(self):
        m = LiquidityMap(LevelConfig(pivot_left=2, pivot_right=2))
        highs = [1, 2, 5, 3, 2, 1]
        for i, h in enumerate(highs[:4]):
            m.update(i, h, h - 0.5, atr=1.0)
        self.assertEqual(m.highs, [])  # bar 4 hasn't closed yet
        m.update(4, highs[4], highs[4] - 0.5, atr=1.0)
        self.assertEqual([p.price for p in m.highs], [5])
        self.assertEqual(m.highs[0].confirmed, 4)

    def test_equal_highs_merge_into_one_pool(self):
        m = LiquidityMap(LevelConfig(pivot_left=1, pivot_right=1, equal_tolerance_atr=0.5))
        for i, h in enumerate([1, 5, 1, 5.2, 1]):
            m.update(i, h, 0, atr=1.0)
        self.assertEqual(len(m.highs), 1)
        self.assertEqual(m.highs[0].touches, 2)
        self.assertEqual(m.highs[0].price, 5.2)

    def test_take_swept_removes_pools(self):
        m = LiquidityMap(LevelConfig(pivot_left=1, pivot_right=1))
        for i, (h, lo) in enumerate([(2, 1), (5, 0.5), (2, 1)]):
            m.update(i, h, lo, atr=1.0)
        highs, lows = m.take_swept(high=6, low=0.6)
        self.assertEqual([p.price for p in highs], [5])
        self.assertEqual(lows, [])
        self.assertEqual(m.highs, [])
        self.assertEqual(len(m.lows), 1)


class CascadeTests(unittest.TestCase):
    def _warm(self, det, n=40):
        for k in range(n):
            det.update(100, 1.0, 1.0, 1000 + k % 5, 1000 + k % 3)

    def test_sell_liquidations_flag_long_cascade(self):
        det = CascadeDetector(CascadeConfig(mode="liquidations", window_bars=1))
        self._warm(det)
        r = det.update(100, 1.0, 1.0, liq_long=1e6, liq_short=1000)
        self.assertTrue(r.long_cascade)
        self.assertFalse(r.short_cascade)

    def test_proxy_needs_volume_and_range(self):
        det = CascadeDetector(CascadeConfig(mode="proxy"))
        for k in range(40):
            det.update(100 + k % 7, 1.0, 1.0, None, None)
        self.assertFalse(det.update(1000, 0.5, 1.0, None, None).long_cascade)
        self.assertTrue(det.update(1000, 2.0, 1.0, None, None).long_cascade)

    def test_auto_falls_back_to_proxy_without_liquidations(self):
        det = CascadeDetector(CascadeConfig(mode="auto"))
        for k in range(40):
            det.update(100 + k % 7, 1.0, 1.0, None, None)
        self.assertEqual(det.update(100, 1.0, 1.0, None, None).source, "proxy")

    def test_liquidations_mode_without_data_never_fires(self):
        det = CascadeDetector(CascadeConfig(mode="liquidations"))
        for _ in range(40):
            r = det.update(1e9, 100.0, 1.0, None, None)
        self.assertFalse(r.long_cascade or r.short_cascade)


class StrategyTests(unittest.TestCase):
    def test_signals_trade_toward_opposite_liquidity(self):
        cfg = demo_config()
        strat = SweepStrategy(cfg)
        signals = [s for b in iter_bars(make_sweep_series()) if (s := strat.on_bar(b))]
        self.assertGreater(len(signals), 5)
        for s in signals:
            if s.side == "long":
                self.assertLess(s.stop, s.swept_level)
                self.assertGreater(s.target, s.ref_price)
            else:
                self.assertGreater(s.stop, s.swept_level)
                self.assertLess(s.target, s.ref_price)
            self.assertGreaterEqual(s.rr, cfg.trade.min_rr - 1e-9)

    def test_no_lookahead(self):
        df = make_sweep_series()
        cut = len(df) // 2

        def sigs(frame):
            strat = SweepStrategy(demo_config())
            return [(s.time, s.side, round(s.stop, 6)) for b in iter_bars(frame) if (s := strat.on_bar(b))]

        full, prefix = sigs(df), sigs(df.iloc[:cut])
        self.assertTrue(prefix)
        self.assertEqual(full[: len(prefix)], prefix)

    def test_breakout_without_close_back_is_not_a_signal(self):
        cfg = demo_config()
        cfg.cascade.mode = "off"
        cfg.levels.pivot_left = cfg.levels.pivot_right = 2
        cfg.atr_period = 3
        strat = SweepStrategy(cfg)
        t = 0
        for h in [10, 11, 15, 11, 10, 11, 12, 11]:
            strat.on_bar(Bar(t, h - 0.5, h, h - 1, h - 0.5, 1)); t += 1
        breakout = strat.on_bar(Bar(t, 12, 16, 11.5, 15.8, 1))  # closes above the 15 high
        self.assertIsNone(breakout)


class BrokerTests(unittest.TestCase):
    def setUp(self):
        self.risk = RiskConfig(fee_rate=0.0, slippage_bps=0.0)

    def test_stop_assumed_first_when_bar_hits_both(self):
        b = PaperBroker(10_000, self.risk)
        b.open("long", 1.0, 100, stop=95, target=110, time=0, index=0)
        t = b.on_bar(Bar(1, 100, 111, 94, 105, 1), index=1, max_bars=0)
        self.assertEqual(t.exit_reason, "stop")
        self.assertAlmostEqual(t.pnl, -5)
        self.assertAlmostEqual(t.r_multiple, -1)

    def test_gap_through_stop_fills_at_open(self):
        b = PaperBroker(10_000, self.risk)
        b.open("short", 2.0, 100, stop=105, target=90, time=0, index=0)
        t = b.on_bar(Bar(1, 108, 109, 107, 108, 1), index=1, max_bars=0)
        self.assertEqual(t.exit_reason, "stop (gap)")
        self.assertAlmostEqual(t.pnl, -16)

    def test_fees_reduce_equity(self):
        b = PaperBroker(10_000, RiskConfig(fee_rate=0.001, slippage_bps=0))
        b.open("long", 1.0, 100, 90, 120, 0, 0)
        t = b.on_bar(Bar(1, 100, 121, 99, 120, 1), index=1, max_bars=0)
        self.assertAlmostEqual(t.pnl, 20 - 0.1 - 0.12)
        self.assertAlmostEqual(b.equity, 10_000 + t.pnl)

    def test_position_size_risk_and_leverage_cap(self):
        risk = RiskConfig(risk_per_trade=0.01, max_leverage=2.0)
        self.assertAlmostEqual(position_size(10_000, 100, 90, risk), 10)  # $100 risk / $10
        self.assertAlmostEqual(position_size(10_000, 100, 99.9, risk), 200)  # capped at 2x


class DataTests(unittest.TestCase):
    def test_liquidation_csv_aggregation_and_merge(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "liq.csv"
            p.write_text("time,side,price,qty\n1700000100000,SELL,100,2\n1700000160000,BUY,100,1\n1700001000000,SELL,10,1\n")
            liq = load_liquidations_csv(p, "15m")
        self.assertEqual(liq.iloc[0].liq_long, 200)
        self.assertEqual(liq.iloc[0].liq_short, 100)
        idx = pd.date_range(liq.index[0], periods=4, freq="15min")
        ohlcv = pd.DataFrame({c: 1.0 for c in ("open", "high", "low", "close", "volume")}, index=idx)
        merged = merge_liquidations(ohlcv, liq)
        self.assertEqual(merged.liq_long.iloc[1], 10)
        self.assertTrue(pd.isna(merged.liq_long.iloc[3]))  # outside recorded span


class BacktestTests(unittest.TestCase):
    def test_backtest_runs_and_accounts_consistently(self):
        res = run_backtest(make_sweep_series(), demo_config(), 10_000)
        self.assertGreater(res.stats["trades"], 0)
        self.assertAlmostEqual(res.stats["final_equity"], 10_000 + sum(t.pnl for t in res.trades), places=6)
        self.assertEqual(len(res.equity), len(make_sweep_series()))

    def test_one_position_at_a_time(self):
        res = run_backtest(make_sweep_series(), demo_config(flip_on_opposite=False), 10_000)
        for a, b in zip(res.trades, res.trades[1:]):
            self.assertLessEqual(a.exit_time, b.entry_time)


if __name__ == "__main__":
    unittest.main()
