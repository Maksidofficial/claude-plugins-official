# CHOP: mean reversion

Seed playbook. Buy stretched dips (close more than 1.5 volatility units below its 20-bar
mean), exit back at the mean. Long only. Small size, short holds.
Invalidated if the filtered probability of CHOP drops below 0.5.

```playbook
state: CHOP
style: mean_reversion
entry_threshold: 1.5
exit_threshold: 0.0
stop_pct: 0.01
take_profit_pct: 0.01
max_size: 0.10
invalidate_below_prob: 0.5
max_hold_bars: 14
```
