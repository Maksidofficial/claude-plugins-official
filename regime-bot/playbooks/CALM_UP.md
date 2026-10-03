# CALM_UP: trend following

Seed playbook from the approved spec. The nightly review revises it; a change ships only
after it re-clears the backtest gates.

Enter long when the normalized 50-bar EMA slope is positive enough; leave when it turns.
Invalidated if the filtered probability of CALM_UP drops below 0.5.

```playbook
state: CALM_UP
style: trend_following
entry_threshold: 0.02
exit_threshold: -0.02
stop_pct: 0.015
take_profit_pct: 0.03
max_size: 0.20
invalidate_below_prob: 0.5
max_hold_bars: 70
```
