#!/usr/bin/env bash
# Install regime-bot on a fresh Ubuntu/Debian VPS as root. Paper trading only.
set -euo pipefail
APP=/opt/regime-bot
id regimebot >/dev/null 2>&1 || useradd --system --home "$APP" --shell /usr/sbin/nologin regimebot
mkdir -p "$APP"
rsync -a --exclude .venv --exclude state --exclude data ./ "$APP/"
python3 -m venv "$APP/.venv"
"$APP/.venv/bin/pip" install -q -e "$APP[research]"
mkdir -p "$APP"/{state,data,rules,proposals}
[ -f "$APP/.env" ] || cp "$APP/.env.example" "$APP/.env"
chmod 600 "$APP/.env"
chown -R regimebot:regimebot "$APP"
install -m 644 "$APP"/deploy/*.service "$APP"/deploy/*.timer /etc/systemd/system/
systemctl daemon-reload
echo "Next: edit $APP/.env (Telegram, Anthropic), run the backtest as regimebot, then:"
echo "  systemctl enable --now regimebot regimebot-dashboard regimebot-watchdog.timer regimebot-nightly.timer"
