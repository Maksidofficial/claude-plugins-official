# Liquidity Sweep Bot

A crypto-futures bot that trades **from one liquidity pool to the other**. It
waits for price to sweep the stops resting beyond a swing high or low, checks
that the sweep set off a liquidation cascade, then trades the reversal and
targets the liquidity on the opposite side of the range.

> **Warning:** this is research software. It does not have a proven edge (see
> [Results](#results-so-far)). Paper trading is the default. Only trade money
> you can afford to lose, and use the Binance testnet before anything else.

## How it works

```
   buy-side liquidity  ─────────────── equal highs ──────── ◄── TARGET (short)
                                ╱╲    ╱╲
                               ╱  ╲  ╱  ╲      ╱╲
          ...range...         ╱    ╲╱    ╲    ╱  ╲
                                          ╲  ╱    ╲
   sell-side liquidity ─────────────────── ╲╱ ──── equal lows
                                            │  ◄── wick sweeps the lows,
                                            ▼      longs get liquidated,
                                          STOP     close back inside → LONG
```

1. **Map liquidity** (`levels.py`). Swing highs and lows become pools. A
   pivot is only known after `pivot_right` more bars close, so nothing looks
   ahead. Swings within `equal_tolerance_atr` of each other merge into one
   "equal highs/lows" pool. Its `touches` count goes up, because more stops
   sit there.
2. **Detect the sweep** (`strategy.py`). A bar's wick runs through a pool by
   between `min_penetration_atr` and `max_penetration_atr`, then the bar
   *closes back inside*. A close beyond the level counts as a breakout, so
   the pool is consumed and no trade is taken.
3. **Confirm the cascade** (`cascade.py`).
   - **`liquidations`**: real forced orders from Binance's `forceOrder`
     stream. A low sweep needs a spike in **long** liquidations (forced
     SELLs). A high sweep needs a spike in **short** liquidations (forced
     BUYs). The spike is measured as a z-score against a rolling window.
   - **`proxy`**: volume z-score plus range expansion (bar range divided by
     ATR). Used when no liquidation data exists, such as historical backtests.
   - **`auto`** (default): uses liquidations once enough history has
     streamed in, otherwise falls back to the proxy.
4. **Trade the other way.**
   - Entry: market, at the sweep bar's close (live) or the next open
     (backtest).
   - Stop: beyond the sweep wick plus `stop_buffer_atr`.
   - Target: the nearest **opposing pool** that pays at least `min_rr`.
   - If an opposite sweep and cascade fire mid-trade, the bot **flips**
     (`flip_on_opposite`). That is the "sweep to sweep" loop.
5. **Risk** (`broker.py`, `engine.py`).
   - Fixed-fractional sizing (1% per trade) with a leverage cap.
   - One position at a time, and a cooldown after each exit.
   - A daily loss limit.
   - Fees and slippage. If a bar touches both the stop and the target, the
     stop is assumed to fill first.

## Quick start

```bash
cd liquidity-sweep-bot
pip install -r requirements.txt        # numpy + pandas are the only hard deps

# Smoke test on synthetic data (setups are planted -- this only checks the plumbing)
python -m sweepbot backtest --demo -v

# Real history from Binance's public archive (data.binance.vision, no key)
python -m sweepbot backtest --symbol BTCUSDT --timeframe 1h \
    --start 2025-10-01 --end 2026-10-15 --save btc1h.csv --trades trades.csv

# Re-run from the saved CSV with your own settings
cp config.example.toml config.toml
python -m sweepbot backtest --csv btc1h.csv --config config.toml
```

### Collect liquidation data, then backtest with it

Binance no longer serves historical liquidations, so record them yourself:

```bash
python -m sweepbot record --symbol BTCUSDT --out btc_liqs.csv      # leave running
# later:
python -m sweepbot backtest --csv btc1h.csv --liquidations btc_liqs.csv
```

Bars inside the recorded span get real liquidation totals. Bars outside it
get none, and `auto` mode falls back to the proxy for those bars. Binance
pushes at most one forced order per symbol per second, so the totals are a
lower bound. That's fine for spotting spikes.

### Live

```bash
python -m sweepbot live --mode paper  --config config.toml   # simulated fills
export BINANCE_API_KEY=... BINANCE_API_SECRET=...
python -m sweepbot live --mode testnet --config config.toml  # testnet orders via ccxt
python -m sweepbot live --mode live --i-understand-the-risks --config config.toml
```

In `testnet` and `live` modes, each entry places reduce-only `STOP_MARKET` and
`TAKE_PROFIT_MARKET` orders on the exchange, so the position stays protected if
the bot stops. The Binance API is geo-restricted in some regions (HTTP 451),
including the US.

## Results so far

**Short version: no edge found.** All runs are BTCUSDT perpetual with 0.04%
taker fees and 2 bps slippage. They use **proxy** cascade detection, because
the archive has no liquidation data.

### Current defaults, Jan 2020 to Oct 2026

| Timeframe | Trades | Return | Max DD | Profit factor |
|---|---|---|---|---|
| 1h | 320 | −49% | −55% | 0.70 |
| 4h | 68 | −23% | −26% | 0.49 |
| 1d | 8 | −4% | −5% | 0.26 |

### Parameter scan

For each of 1h, 4h and 1d, 64 configurations were tuned on 2020–2023 and
tested on 2024–2026. The scan varied pivot size, minimum pool touches, stop
buffer, `min_rr`, the volume threshold and the time stop.

- **Not one of the 192 configurations was profitable in-sample.** The best
  was a 4h setup that broke even on 2020–2023, then lost 17%
  out-of-sample (profit factor 0.50).
- On 1h and 4h, wider stops and longer time stops lose less, but nothing
  crosses into profit.
- 1d produces only about 10–25 trades over six years, too few to judge either
  way.

An earlier one-year test (Oct 2025 to Oct 2026) showed 1h at +8% with profit
factor 1.24. The six-year run shows that was luck: the defaults had been
picked on that same year.

### What's still untested

The core idea, a confirmed *liquidation* cascade, could not be tested on
history. The volume and range proxy is clearly not enough on its own. Record
real liquidations for a few weeks, then backtest with `--liquidations`, before
you trust any of this.

## Layout

```
sweepbot/
  config.py     dataclass config + TOML loading
  levels.py     swing-high/low liquidity pools, equal-level merging
  cascade.py    liquidation-spike and proxy cascade detection
  strategy.py   sweep + cascade -> signal with stop and opposite-pool target
  broker.py     PaperBroker (fees/slippage/gaps) and CcxtBroker (Binance USDⓈ-M)
  engine.py     entries, flips, cooldown, daily loss limit
  backtest.py   simulation + stats
  data.py       CSV, Binance archive/REST klines, liquidation websocket
  live.py       live paper/testnet/live loop, liquidation recorder
  synthetic.py  planted-sweep data for tests/demo
tests/          python -m unittest discover -s tests
```
