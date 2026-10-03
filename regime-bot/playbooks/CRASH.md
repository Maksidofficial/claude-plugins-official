# CRASH: flat

No positions. The parser rejects any CRASH playbook that is not flat with max_size 0.

```playbook
state: CRASH
style: flat
entry_threshold: 0
exit_threshold: 0
stop_pct: 0.01
take_profit_pct: 0.01
max_size: 0
invalidate_below_prob: 1
max_hold_bars: 1
```
