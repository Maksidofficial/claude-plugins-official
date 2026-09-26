# TradingView

Connects Claude Code to the community [TradingView MCP server](https://github.com/atilaahmettaner/tradingview-mcp) for market data and technical analysis across stocks, crypto, and forex.

## Requirements

- [`uv`](https://docs.astral.sh/uv/) installed, so `uvx` is on your `PATH`. The server package (`tradingview-mcp-server`) is fetched from PyPI on first run.
- No API key is needed for core features. News and sentiment tools optionally use a Marketaux API key; see the upstream README.

## Tools

- **Technical analysis:** indicators, Bollinger Band analysis, candlestick patterns, multi-timeframe analysis
- **Screeners:** stock and crypto screeners, scan by signal, exchange-specific scanners (Binance, KuCoin, Bybit, NASDAQ, NYSE, and more)
- **Market data:** prices and market snapshots
- **Backtesting:** built-in strategies (RSI, MACD, EMA cross, Supertrend, and others), strategy comparison, walk-forward testing
- **Sentiment/news:** market sentiment and financial news

## Install

```
/plugin install tradingview@claude-plugins-official
```

This is an unofficial, community-maintained server and is not affiliated with TradingView. Output is for information only and is not financial advice.
