# Deploying regime-bot 24/7

## Where to run it

| Option | Verdict |
|---|---|
| **VPS (recommended)** | Any 1–2 vCPU Linux box with 2 GB RAM. systemd restarts the bot if it crashes, and timers run the watchdog and the nightly review. |
| **Mac Mini** | Works if it never sleeps. Use `launchd` with `KeepAlive=true` in place of systemd. Map the units below to plists. A home network is a single point of failure, but the stale-feed flatten and the watchdog still protect you. |
| **Vercel** | **Not suitable.** Serverless functions are short-lived and stateless. The bot is a long-running process that keeps state between candles. |

## VPS walkthrough (Ubuntu/Debian)

```bash
# 1. copy the project to the server, then as root:
cd regime-bot && sudo ./deploy/install.sh

# 2. secrets (optional): Telegram alerts and the nightly Opus 5.5 review
sudo -u regimebot nano /opt/regime-bot/.env       # TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID,
                                                  # ANTHROPIC_API_KEY. Market data needs no key.

# 3. backtest: downloads public BTCUSDT 1h data, writes strategy.md, the model, live params
cd /opt/regime-bot && sudo -u regimebot .venv/bin/python -m regimebot.backtest.run

# 4. prove the kill switch, then start everything
sudo -u regimebot .venv/bin/python -m regimebot.cli drill-kill
sudo systemctl enable --now regimebot regimebot-dashboard \
    regimebot-watchdog.timer regimebot-nightly.timer

# 5. watch it
journalctl -u regimebot -f
ssh -L 8765:127.0.0.1:8765 you@server    # then open http://localhost:8765
```

| Unit | What it does |
|---|---|
| `regimebot.service` | The live paper loop. `Restart=always`, non-root, sandboxed. |
| `regimebot-dashboard.service` | Read-only dashboard on 127.0.0.1. It refuses to bind anywhere else. |
| `regimebot-watchdog.timer` | Runs every minute. Fires the kill switch if the heartbeat is more than 3 minutes old. |
| `regimebot-nightly.timer` | Runs at 00:20 UTC. Sends the daily report, runs the Opus 5.5 review, and promotes a proposal only if it clears every gate. |

## Alerts (Telegram)

Create a bot with @BotFather and put the token and your chat id in `.env`. You get alerts
for every fill, error, state switch, approval request, stale feed, drift warning and
kill-switch event, plus the daily report. Without a token, alerts still go to
`state/live/alerts.jsonl`.

## Operating

```bash
python -m regimebot.cli report                     # today's report
python -m regimebot.cli approve <id>               # grant a pending large-order approval
python -m regimebot.cli kill --reason "..."        # flatten and halt
python -m regimebot.cli kill-reset --confirm "..." # only after reviewing positions
python -m regimebot.cli preflight                  # go-live gate (see GO_LIVE.md)
```
