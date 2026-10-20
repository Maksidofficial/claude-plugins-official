"""Market data: CSV loaders, Binance futures REST klines, liquidation websocket."""

from __future__ import annotations

import asyncio
import csv
import io
import json
import logging
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from collections import defaultdict
from pathlib import Path

import pandas as pd

from .config import timeframe_ms

log = logging.getLogger(__name__)

FAPI = "https://fapi.binance.com"
ARCHIVE = "https://data.binance.vision/data/futures/um"
LIQ_WS = "wss://fstream.binance.com/ws/{symbol}@forceOrder"

_TIME_COLUMNS = ("time", "timestamp", "open_time", "date", "datetime")


def _to_utc_index(values: pd.Series) -> pd.DatetimeIndex:
    if pd.api.types.is_numeric_dtype(values):
        unit = "ms" if values.max() > 1e11 else "s"
        return pd.DatetimeIndex(pd.to_datetime(values, unit=unit, utc=True))
    return pd.DatetimeIndex(pd.to_datetime(values, utc=True))


def load_ohlcv_csv(path: str | Path) -> pd.DataFrame:
    """CSV with a time column plus open/high/low/close/volume (optionally liq_long/liq_short)."""
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    tcol = next((c for c in _TIME_COLUMNS if c in df.columns), None)
    if tcol is None:
        raise ValueError(f"{path}: need one of {_TIME_COLUMNS} as a time column")
    missing = {"open", "high", "low", "close", "volume"} - set(df.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    df.index = _to_utc_index(df[tcol])
    keep = [c for c in ("open", "high", "low", "close", "volume", "liq_long", "liq_short") if c in df.columns]
    return df[keep].astype(float).sort_index()


def load_liquidations_csv(path: str | Path, timeframe: str) -> pd.DataFrame:
    """Aggregate raw forced orders into per-bar long/short liquidation notional.

    Columns: ``time`` (ms or ISO), ``side`` (SELL/long = long liquidated,
    BUY/short = short liquidated) and either ``notional`` or ``price`` + ``qty``.
    This is the format written by ``python -m sweepbot record``.
    """
    raw = pd.read_csv(path)
    raw.columns = [c.strip().lower() for c in raw.columns]
    if "notional" not in raw.columns:
        raw["notional"] = raw["price"] * raw["qty"]
    side = raw["side"].str.lower()
    raw["liq_long"] = raw["notional"].where(side.isin(["sell", "long"]), 0.0)
    raw["liq_short"] = raw["notional"].where(side.isin(["buy", "short"]), 0.0)
    raw.index = _to_utc_index(raw["time"])
    freq = pd.Timedelta(milliseconds=timeframe_ms(timeframe))
    return raw[["liq_long", "liq_short"]].resample(freq, origin="epoch").sum()


def merge_liquidations(ohlcv: pd.DataFrame, liq: pd.DataFrame) -> pd.DataFrame:
    """Attach liquidation columns. Bars inside the recorded span with no forced
    orders get 0; bars outside it get NaN (treated as "no data")."""
    out = ohlcv.drop(columns=["liq_long", "liq_short"], errors="ignore").join(liq, how="left")
    if len(liq):
        inside = (out.index >= liq.index.min()) & (out.index <= liq.index.max())
        out.loc[inside, ["liq_long", "liq_short"]] = out.loc[inside, ["liq_long", "liq_short"]].fillna(0.0)
    return out


def _get_json(url: str, params: dict) -> list:
    req = urllib.request.Request(f"{url}?{urllib.parse.urlencode(params)}", headers={"User-Agent": "sweepbot"})
    with urllib.request.urlopen(req, timeout=20) as resp:
        return json.loads(resp.read())


def fetch_binance_klines(
    symbol: str,
    timeframe: str,
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
    limit: int | None = None,
    closed_only: bool = True,
) -> pd.DataFrame:
    """Download USDⓈ-M futures klines (public endpoint, no key needed).

    Either a ``start``..``end`` range (paginated) or the most recent ``limit`` bars.
    """
    url = f"{FAPI}/fapi/v1/klines"
    rows: list[list] = []
    if start is None:
        rows = _get_json(url, {"symbol": symbol, "interval": timeframe, "limit": min(limit or 1000, 1500)})
    else:
        cursor = int(pd.Timestamp(start, tz="UTC").timestamp() * 1000)
        stop = int(pd.Timestamp(end, tz="UTC").timestamp() * 1000) if end else int(time.time() * 1000)
        while cursor < stop:
            batch = _get_json(
                url, {"symbol": symbol, "interval": timeframe, "startTime": cursor, "endTime": stop, "limit": 1500}
            )
            if not batch:
                break
            rows.extend(batch)
            cursor = int(batch[-1][0]) + timeframe_ms(timeframe)
            time.sleep(0.2)
    now_ms = int(time.time() * 1000)
    if closed_only:
        rows = [r for r in rows if int(r[6]) < now_ms]
    df = pd.DataFrame(
        [[int(r[0]), *map(float, r[1:6])] for r in rows],
        columns=["time", "open", "high", "low", "close", "volume"],
    )
    df.index = pd.DatetimeIndex(pd.to_datetime(df.pop("time"), unit="ms", utc=True))
    return df[~df.index.duplicated()].sort_index()


def _archive_csv(url: str, retries: int = 4) -> pd.DataFrame | None:
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "sweepbot"}), timeout=60) as r:
                blob = r.read()
            break
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            if attempt == retries or e.code < 500:
                raise
        except (urllib.error.URLError, OSError):
            if attempt == retries:
                raise
        time.sleep(2 ** (attempt + 1))
        log.warning("archive download failed, retrying: %s", url)
    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        raw = zf.read(zf.namelist()[0]).decode()
    has_header = raw.startswith("open_time")
    return pd.read_csv(io.StringIO(raw), header=0 if has_header else None, usecols=range(6),
                       names=None if has_header else ["open_time", "open", "high", "low", "close", "volume"])


def fetch_binance_archive(symbol: str, timeframe: str, start: str, end: str | None = None) -> pd.DataFrame:
    """Download USDⓈ-M futures klines from data.binance.vision (bulk archive).

    Uses monthly files and falls back to daily files for months not yet
    published. Reachable from regions where the trading API returns 451.
    """
    t0 = pd.Timestamp(start, tz="UTC")
    t1 = pd.Timestamp(end, tz="UTC") if end else pd.Timestamp.now(tz="UTC")
    frames = []
    for month in pd.period_range(t0.tz_localize(None), t1.tz_localize(None), freq="M"):
        tag = f"{symbol}-{timeframe}-{month.strftime('%Y-%m')}"
        df = _archive_csv(f"{ARCHIVE}/monthly/klines/{symbol}/{timeframe}/{tag}.zip")
        if df is None:
            days = pd.date_range(max(month.start_time, t0.tz_localize(None).normalize()),
                                 min(month.end_time, t1.tz_localize(None)), freq="D")
            for day in days:
                d = _archive_csv(f"{ARCHIVE}/daily/klines/{symbol}/{timeframe}/{symbol}-{timeframe}-{day:%Y-%m-%d}.zip")
                if d is not None:
                    frames.append(d)
        else:
            frames.append(df)
        log.info("archive: %s", tag)
    if not frames:
        raise ValueError(f"no archive data for {symbol} {timeframe} {start}..{end}")
    df = pd.concat(frames)
    df.index = pd.DatetimeIndex(pd.to_datetime(df.pop("open_time").astype("int64"), unit="ms", utc=True))
    df = df.astype(float)
    df = df[~df.index.duplicated()].sort_index()
    return df[(df.index >= t0) & (df.index < t1)]


class LiquidationStream:
    """Consumes Binance's ``<symbol>@forceOrder`` stream and buckets notional by bar.

    Binance pushes at most one (the largest) forced order per symbol per
    second, so totals are a lower bound -- fine for spotting spikes.
    """

    def __init__(self, symbol: str, timeframe: str, record_path: str | Path | None = None):
        self.symbol = symbol.lower()
        self.tf_ms = timeframe_ms(timeframe)
        self.buckets: dict[int, list[float]] = defaultdict(lambda: [0.0, 0.0])
        self.connected_since: int | None = None
        self.record_path = Path(record_path) if record_path else None

    def take(self, bar_open_ms: int) -> tuple[float | None, float | None]:
        """Pop (long_liq, short_liq) for a bar, or (None, None) if we weren't
        listening for the whole bar."""
        if self.connected_since is None or self.connected_since > bar_open_ms:
            self.buckets.pop(bar_open_ms, None)
            return None, None
        long_liq, short_liq = self.buckets.pop(bar_open_ms, (0.0, 0.0))
        return long_liq, short_liq

    def _handle(self, msg: dict, writer) -> None:
        o = msg.get("o", {})
        price = float(o.get("ap") or o.get("p") or 0)
        qty = float(o.get("z") or o.get("q") or 0)
        ts = int(o.get("T") or msg.get("E") or time.time() * 1000)
        notional = price * qty
        bucket = self.buckets[ts - ts % self.tf_ms]
        if o.get("S") == "SELL":
            bucket[0] += notional  # long liquidated
        else:
            bucket[1] += notional  # short liquidated
        if writer:
            writer.writerow([ts, o.get("S"), price, qty, notional])

    async def run(self) -> None:
        try:
            import websockets  # type: ignore[import-not-found]
        except ImportError:
            log.warning("websockets not installed; running without liquidation data (pip install websockets)")
            return
        url = LIQ_WS.format(symbol=self.symbol)
        fh = writer = None
        if self.record_path:
            new = not self.record_path.exists()
            fh = open(self.record_path, "a", newline="")
            writer = csv.writer(fh)
            if new:
                writer.writerow(["time", "side", "price", "qty", "notional"])
        try:
            while True:
                try:
                    async with websockets.connect(url, ping_interval=20) as ws:
                        # Data is only complete from the next bar boundary on.
                        now = int(time.time() * 1000)
                        self.connected_since = now - now % self.tf_ms + self.tf_ms
                        log.info("liquidation stream connected: %s", url)
                        async for raw in ws:
                            self._handle(json.loads(raw), writer)
                            if fh:
                                fh.flush()
                except asyncio.CancelledError:
                    raise
                except Exception as e:  # reconnect on any network error
                    self.connected_since = None
                    log.warning("liquidation stream dropped (%s); reconnecting in 5s", e)
                    await asyncio.sleep(5)
        finally:
            if fh:
                fh.close()
