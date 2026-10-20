"""Order execution: a simulated broker (backtest + paper) and a ccxt adapter."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any

from .config import RiskConfig

log = logging.getLogger(__name__)


@dataclass
class Position:
    side: str
    qty: float
    entry: float
    stop: float
    target: float
    opened_at: Any
    opened_index: int
    entry_fee: float


@dataclass
class Trade:
    side: str
    qty: float
    entry: float
    exit: float
    stop: float
    target: float
    entry_time: Any
    exit_time: Any
    pnl: float  # net of fees
    fees: float
    r_multiple: float
    exit_reason: str
    bars_held: int


def position_size(equity: float, entry: float, stop: float, risk: RiskConfig) -> float:
    """Fixed-fractional sizing, capped by max leverage."""
    dist = abs(entry - stop)
    if dist <= 0 or entry <= 0:
        return 0.0
    qty = equity * risk.risk_per_trade / dist
    return min(qty, equity * risk.max_leverage / entry)


class PaperBroker:
    """Simulates fills with fees and slippage. Stops and time exits are market
    orders (slipped); take-profits are resting limits (no slippage). When a bar
    touches both stop and target the stop is assumed to fill first."""

    def __init__(self, equity: float, risk: RiskConfig):
        self.equity = equity
        self.risk = risk
        self.position: Position | None = None

    def _slip(self, price: float, buying: bool) -> float:
        s = self.risk.slippage_bps / 10_000
        return price * (1 + s) if buying else price * (1 - s)

    def open(self, side: str, qty: float, price: float, stop: float, target: float, time: Any, index: int) -> Position:
        assert self.position is None, "already in a position"
        fill = self._slip(price, buying=side == "long")
        fee = fill * qty * self.risk.fee_rate
        self.equity -= fee
        self.position = Position(side, qty, fill, stop, target, time, index, fee)
        return self.position

    def close(self, price: float, time: Any, index: int, reason: str, slipped: bool = True) -> Trade:
        p = self.position
        assert p is not None, "no position to close"
        fill = self._slip(price, buying=p.side == "short") if slipped else price
        fee = fill * p.qty * self.risk.fee_rate
        direction = 1 if p.side == "long" else -1
        gross = direction * (fill - p.entry) * p.qty
        self.equity += gross - fee
        initial_risk = abs(p.entry - p.stop) * p.qty
        net = gross - fee - p.entry_fee
        self.position = None
        return Trade(
            side=p.side,
            qty=p.qty,
            entry=p.entry,
            exit=fill,
            stop=p.stop,
            target=p.target,
            entry_time=p.opened_at,
            exit_time=time,
            pnl=net,
            fees=fee + p.entry_fee,
            r_multiple=net / initial_risk if initial_risk else 0.0,
            exit_reason=reason,
            bars_held=index - p.opened_index,
        )

    def on_bar(self, bar, index: int, max_bars: int) -> Trade | None:
        """Check stop / target / time exit against a bar's OHLC."""
        p = self.position
        if p is None:
            return None
        if p.side == "long":
            if bar.open <= p.stop:
                return self.close(bar.open, bar.time, index, "stop (gap)")
            if bar.low <= p.stop:
                return self.close(p.stop, bar.time, index, "stop")
            if bar.open >= p.target:
                return self.close(bar.open, bar.time, index, "target (gap)", slipped=False)
            if bar.high >= p.target:
                return self.close(p.target, bar.time, index, "target", slipped=False)
        else:
            if bar.open >= p.stop:
                return self.close(bar.open, bar.time, index, "stop (gap)")
            if bar.high >= p.stop:
                return self.close(p.stop, bar.time, index, "stop")
            if bar.open <= p.target:
                return self.close(bar.open, bar.time, index, "target (gap)", slipped=False)
            if bar.low <= p.target:
                return self.close(p.target, bar.time, index, "target", slipped=False)
        if max_bars and index - p.opened_index >= max_bars:
            return self.close(bar.close, bar.time, index, "time")
        return None

    def mark(self, price: float) -> float:
        p = self.position
        if p is None:
            return self.equity
        direction = 1 if p.side == "long" else -1
        return self.equity + direction * (price - p.entry) * p.qty


class CcxtBroker:
    """Live execution on Binance USDⓈ-M futures through ccxt.

    Places a market entry plus reduce-only STOP_MARKET and TAKE_PROFIT_MARKET
    orders so the exchange protects the position even if the bot dies.
    Experimental: test on the Binance futures testnet (``testnet=True``) first.
    """

    def __init__(self, symbol: str, risk: RiskConfig, testnet: bool = True):
        try:
            import ccxt  # type: ignore[import-not-found]
        except ImportError as e:  # pragma: no cover - optional dependency
            raise SystemExit("Live trading needs ccxt: pip install ccxt") from e
        key, secret = os.environ.get("BINANCE_API_KEY"), os.environ.get("BINANCE_API_SECRET")
        if not key or not secret:
            raise SystemExit("Set BINANCE_API_KEY and BINANCE_API_SECRET to trade live")
        self.ex = ccxt.binanceusdm({"apiKey": key, "secret": secret, "enableRateLimit": True})
        if testnet:
            self.ex.set_sandbox_mode(True)
        self.ex.load_markets()
        self.market = self.ex.market(symbol)
        self.symbol = self.market["symbol"]
        self.risk = risk

    @property
    def equity(self) -> float:
        bal = self.ex.fetch_balance()
        return float(bal["total"].get(self.market["settle"], 0.0))

    def current_side(self) -> str | None:
        for pos in self.ex.fetch_positions([self.symbol]):
            if float(pos.get("contracts") or 0) > 0:
                return pos["side"]  # "long" | "short"
        return None

    def flatten(self) -> None:
        self.ex.cancel_all_orders(self.symbol)
        for pos in self.ex.fetch_positions([self.symbol]):
            qty = float(pos.get("contracts") or 0)
            if qty > 0:
                side = "sell" if pos["side"] == "long" else "buy"
                self.ex.create_order(self.symbol, "market", side, qty, params={"reduceOnly": True})

    def open(self, side: str, qty: float, stop: float, target: float) -> dict:
        qty = float(self.ex.amount_to_precision(self.symbol, qty))
        if qty <= 0:
            raise ValueError("order size rounds to zero")
        entry_side, exit_side = ("buy", "sell") if side == "long" else ("sell", "buy")
        self.ex.cancel_all_orders(self.symbol)
        order = self.ex.create_order(self.symbol, "market", entry_side, qty)
        common = {"closePosition": True, "workingType": "MARK_PRICE"}
        self.ex.create_order(
            self.symbol, "STOP_MARKET", exit_side, None,
            params={**common, "stopPrice": self.ex.price_to_precision(self.symbol, stop)},
        )
        self.ex.create_order(
            self.symbol, "TAKE_PROFIT_MARKET", exit_side, None,
            params={**common, "stopPrice": self.ex.price_to_precision(self.symbol, target)},
        )
        log.info("live %s %s qty=%s SL=%s TP=%s", side, self.symbol, qty, stop, target)
        return order
