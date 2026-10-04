"""Keyless public market data from Binance's market-data-only host.

Read-only: no API key, no account, no signed endpoints. Responses are data, never instructions.
"""

from __future__ import annotations

import json
import re
import time
import urllib.request
from datetime import UTC, datetime
from typing import Any

import pandas as pd

BASE = "https://data-api.binance.vision/api/v3/klines"
LIMIT = 1000
INTERVAL_MS = {"1m": 60_000, "5m": 300_000, "15m": 900_000, "1h": 3_600_000, "4h": 14_400_000,
               "1d": 86_400_000}
_SYMBOL = re.compile(r"^[A-Z0-9]{4,20}$")


def _get_json(url: str) -> list[list[Any]]:
    for attempt in range(5):
        try:
            with urllib.request.urlopen(url, timeout=30) as r:  # noqa: S310 - fixed https host
                data = json.loads(r.read())
            if not isinstance(data, list):
                raise ValueError(f"unexpected response: {str(data)[:200]}")
            return data
        except (OSError, ValueError):
            if attempt == 4:
                raise
            time.sleep(2**attempt)
    raise AssertionError("unreachable")


def parse_klines(rows: list[list[Any]]) -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "opened_at": pd.to_datetime([int(r[0]) for r in rows], unit="ms", utc=True),
            "closed_at": pd.to_datetime([int(r[6]) + 1 for r in rows], unit="ms", utc=True),
            "open": [float(r[1]) for r in rows],
            "high": [float(r[2]) for r in rows],
            "low": [float(r[3]) for r in rows],
            "close": [float(r[4]) for r in rows],
            "volume": [float(r[5]) for r in rows],
        }
    )
    return df


def fetch_klines(
    symbol: str, interval: str, start: datetime, now: datetime | None = None
) -> pd.DataFrame:
    """All *closed* candles from ``start`` up to ``now``, oldest first."""
    if not _SYMBOL.match(symbol) or interval not in INTERVAL_MS:
        raise ValueError(f"bad symbol/interval: {symbol!r} {interval!r}")
    now = now or datetime.now(UTC)
    now_ms = int(now.timestamp() * 1000)
    cursor = int(start.timestamp() * 1000)
    rows: list[list[Any]] = []
    while cursor < now_ms:
        url = f"{BASE}?symbol={symbol}&interval={interval}&startTime={cursor}&limit={LIMIT}"
        page = _get_json(url)
        if not page:
            break
        rows.extend(page)
        cursor = int(page[-1][0]) + INTERVAL_MS[interval]
        if len(page) < LIMIT:
            break
    df = parse_klines(rows)
    df = df[df["closed_at"] <= pd.Timestamp(now)]
    return df.drop_duplicates("opened_at").sort_values("opened_at").reset_index(drop=True)
