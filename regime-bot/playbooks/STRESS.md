# STRESS: reduced size

Seed playbook. Only strong trends, at a quarter of normal size, tight stop.
Invalidated if the filtered probability of STRESS drops below 0.6.

```playbook
state: STRESS
style: trend_following
entry_threshold: 0.05
exit_threshold: 0.0
stop_pct: 0.01
take_profit_pct: 0.02
max_size: 0.05
invalidate_below_prob: 0.6
max_hold_bars: 14
```
