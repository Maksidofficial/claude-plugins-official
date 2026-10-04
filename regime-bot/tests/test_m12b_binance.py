from datetime import UTC, datetime
from typing import Any

import pandas as pd
import pytest

from regimebot.data import binance
from regimebot.data.binance import fetch_klines, parse_klines

H_MS = 3_600_000
T0 = int(datetime(2024, 1, 1, tzinfo=UTC).timestamp() * 1000)


def row(i: int, close: float = 100.0) -> list[Any]:
    o = T0 + i * H_MS
    c = f"{close}"
    return [o, "100.0", "101.0", "99.0", c, "12.5", o + H_MS - 1, "1250.0", 10, "6", "600", "0"]


def test_parse_klines() -> None:
    df = parse_klines([row(0), row(1, 102.5)])
    assert list(df.columns) == ["opened_at", "closed_at", "open", "high", "low", "close", "volume"]
    assert df["opened_at"].iloc[0] == pd.Timestamp("2024-01-01T00:00:00Z")
    assert df["closed_at"].iloc[0] == pd.Timestamp("2024-01-01T01:00:00Z")
    assert df["close"].iloc[1] == 102.5 and df["volume"].iloc[0] == 12.5


def test_fetch_paginates_and_drops_unclosed(monkeypatch: pytest.MonkeyPatch) -> None:
    rows = [row(i, 100 + i) for i in range(2500)]
    calls: list[str] = []

    def fake_get(url: str) -> list[list[Any]]:
        calls.append(url)
        q = dict(p.split("=") for p in url.split("?")[1].split("&"))
        start, limit = int(q["startTime"]), int(q["limit"])
        return [r for r in rows if r[0] >= start][:limit]

    monkeypatch.setattr(binance, "_get_json", fake_get)
    now = datetime.fromtimestamp((T0 + 2400 * H_MS + 1800_000) / 1000, UTC)  # bar 2400 still open
    df = fetch_klines("BTCUSDT", "1h", datetime.fromtimestamp(T0 / 1000, UTC), now=now)
    assert len(df) == 2400 and len(calls) == 3
    assert df["closed_at"].max() <= pd.Timestamp(now)
    assert df["opened_at"].is_monotonic_increasing and not df["opened_at"].duplicated().any()


def test_only_public_host(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[str] = []
    monkeypatch.setattr(binance, "_get_json", lambda u: seen.append(u) or [])
    fetch_klines("BTCUSDT", "1h", datetime(2024, 1, 1, tzinfo=UTC),
                 now=datetime(2024, 1, 2, tzinfo=UTC))
    host = "https://data-api.binance.vision/api/v3/klines?"
    assert seen and all(u.startswith(host) for u in seen)
    assert all("signature" not in u and "apiKey" not in u for u in seen)


def test_rejects_bad_symbol() -> None:
    with pytest.raises(ValueError):
        fetch_klines("BTC&evil=1", "1h", datetime(2024, 1, 1, tzinfo=UTC))
