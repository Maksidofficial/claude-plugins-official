# strategy.md

## Status: NO STRATEGY ACCEPTED

Failed gates: sharpe, hit_rate, t_stat, beats_buy_and_hold. The bot stays flat until a change re-clears every gate.

## Gates (out of sample, after costs)

| gate | value | needs | ok |
|---|---|---|---|
| sharpe | -0.6303 | > 1.5 | NO |
| max_drawdown | 0.0083 | < 0.15 | yes |
| hit_rate | 0.4375 | > 0.55 | NO |
| t_stat | -1.3749 | > 2.0 | NO |
| min_trades | 64.0000 | >= 30 | yes |
| beats_buy_and_hold | -0.6303 | > 0.493 (buy_and_hold) | NO |
| beats_static | -0.6303 | > -1.705 (static_CALM_UP) | yes |

## System vs baselines

| | sharpe | max_drawdown | hit_rate | t_stat | total_return | trades |
|---|---|---|---|---|---|---|
| system | -0.6303 | 0.0083 | 0.4375 | -1.3749 | -0.0059 | 64 |
| buy_and_hold | 0.4932 | 0.6738 | 1.0000 | 1.0758 | 0.8144 | 1 |
| static_CALM_UP | -1.7053 | 0.4223 | 0.2721 | -3.7201 | -0.4184 | 1573 |

## Per state

| state | trades | win_rate | pnl | avg_ret | largest_loss |
|---|---|---|---|---|---|
| CALM_UP | 20 | 0.4000 | -479.59 | -0.0006 | -168.72 |
| CHOP | 21 | 0.5714 | 80.08 | -0.0004 | -123.43 |
| STRESS | 23 | 0.3478 | -190.48 | -0.0035 | -31.03 |

## Time in state

- CALM_UP: 28.0%
- CHOP: 58.1%
- CRASH: 4.0%
- STRESS: 9.8%

## Calibration (PIT reliability, out of sample)

| state | n | max_dev | brier | brier_climatology | calibrated |
|---|---|---|---|---|---|
| CALM_UP | 11671 | 0.0694 | 0.2501 | 0.2500 | True |
| STRESS | 4093 | 0.1016 | 0.2499 | 0.2500 | False |
| CHOP | 24235 | 0.0606 | 0.2500 | 0.2499 | True |
| CRASH | 1687 | 0.1459 | 0.2501 | 0.2499 | False |

## Model (final, K=5)

| state | label | mean_ret | vol | duration |
|---|---|---|---|---|
| 0 | CHOP | 0.000018 | 0.00235 | 23.7 |
| 1 | CHOP | -0.000091 | 0.00463 | 7.4 |
| 2 | CRASH | -0.000581 | 0.01472 | 12.9 |
| 3 | CALM_UP | 0.000103 | 0.00469 | 12.6 |
| 4 | STRESS | 0.000273 | 0.00854 | 19.1 |

Transition matrix:

```
0.958  0.041  0.001  0.000  0.000
0.021  0.865  0.016  0.081  0.018
0.000  0.005  0.922  0.000  0.072
0.025  0.049  0.002  0.921  0.005
0.000  0.003  0.015  0.034  0.948
```

K selection:

- K=2: OOS LL -30774.6, BIC 241944.5
- K=3: OOS LL -27593.4, BIC 209135.5
- K=4: OOS LL -25651.7, BIC 188904.8
- K=5: OOS LL -23929.8, BIC 174467.8

## Refits: 58 (57 accepted)

- 2022-01-01 00:00:00+00:00: accepted=True alarms=[]
- 2022-01-31 00:00:00+00:00: accepted=True alarms=[]
- 2022-03-02 00:00:00+00:00: accepted=True alarms=[]
- 2022-04-01 00:00:00+00:00: accepted=True alarms=[]
- 2022-05-01 00:00:00+00:00: accepted=True alarms=[]
- 2022-05-31 00:00:00+00:00: accepted=False alarms=['mean_shift']
- 2022-06-30 00:00:00+00:00: accepted=True alarms=['mean_shift']
- 2022-07-30 00:00:00+00:00: accepted=True alarms=[]
- 2022-08-29 00:00:00+00:00: accepted=True alarms=[]
- 2022-09-28 00:00:00+00:00: accepted=True alarms=[]
- 2022-10-28 00:00:00+00:00: accepted=True alarms=[]
- 2022-11-27 00:00:00+00:00: accepted=True alarms=[]
- 2022-12-27 00:00:00+00:00: accepted=True alarms=[]
- 2023-01-26 00:00:00+00:00: accepted=True alarms=[]
- 2023-02-25 00:00:00+00:00: accepted=True alarms=[]
- 2023-03-27 00:00:00+00:00: accepted=True alarms=[]
- 2023-04-26 00:00:00+00:00: accepted=True alarms=[]
- 2023-05-26 00:00:00+00:00: accepted=True alarms=[]
- 2023-06-25 00:00:00+00:00: accepted=True alarms=[]
- 2023-07-25 00:00:00+00:00: accepted=True alarms=[]
- 2023-08-24 00:00:00+00:00: accepted=True alarms=[]
- 2023-09-23 00:00:00+00:00: accepted=True alarms=[]
- 2023-10-23 00:00:00+00:00: accepted=True alarms=[]
- 2023-11-22 00:00:00+00:00: accepted=True alarms=[]
- 2023-12-22 00:00:00+00:00: accepted=True alarms=[]
- 2024-01-21 00:00:00+00:00: accepted=True alarms=[]
- 2024-02-20 00:00:00+00:00: accepted=True alarms=[]
- 2024-03-21 00:00:00+00:00: accepted=True alarms=[]
- 2024-04-20 00:00:00+00:00: accepted=True alarms=[]
- 2024-05-20 00:00:00+00:00: accepted=True alarms=[]
- 2024-06-19 00:00:00+00:00: accepted=True alarms=[]
- 2024-07-19 00:00:00+00:00: accepted=True alarms=[]
- 2024-08-18 00:00:00+00:00: accepted=True alarms=[]
- 2024-09-17 00:00:00+00:00: accepted=True alarms=[]
- 2024-10-17 00:00:00+00:00: accepted=True alarms=[]
- 2024-11-16 00:00:00+00:00: accepted=True alarms=[]
- 2024-12-16 00:00:00+00:00: accepted=True alarms=[]
- 2025-01-15 00:00:00+00:00: accepted=True alarms=[]
- 2025-02-14 00:00:00+00:00: accepted=True alarms=[]
- 2025-03-16 00:00:00+00:00: accepted=True alarms=[]
- 2025-04-15 00:00:00+00:00: accepted=True alarms=[]
- 2025-05-15 00:00:00+00:00: accepted=True alarms=[]
- 2025-06-14 00:00:00+00:00: accepted=True alarms=[]
- 2025-07-14 00:00:00+00:00: accepted=True alarms=[]
- 2025-08-13 00:00:00+00:00: accepted=True alarms=[]
- 2025-09-12 00:00:00+00:00: accepted=True alarms=[]
- 2025-10-12 00:00:00+00:00: accepted=True alarms=[]
- 2025-11-11 00:00:00+00:00: accepted=True alarms=[]
- 2025-12-11 00:00:00+00:00: accepted=True alarms=[]
- 2026-01-10 00:00:00+00:00: accepted=True alarms=[]
- 2026-02-09 00:00:00+00:00: accepted=True alarms=[]
- 2026-03-11 00:00:00+00:00: accepted=True alarms=[]
- 2026-04-10 00:00:00+00:00: accepted=True alarms=[]
- 2026-05-10 00:00:00+00:00: accepted=True alarms=[]
- 2026-06-09 00:00:00+00:00: accepted=True alarms=[]
- 2026-07-09 00:00:00+00:00: accepted=True alarms=[]
- 2026-08-08 00:00:00+00:00: accepted=True alarms=[]
- 2026-09-07 00:00:00+00:00: accepted=True alarms=[]

## Notes

- manual approvals are simulated as granted in the backtest
- 2022-05-12T15:00:00+00:00: live LL alarm, entries frozen
- 2022-05-31T00:00:00+00:00: drift ['mean_shift'], entries frozen
- 2022-06-30T00:00:00+00:00: drift ['mean_shift'] persisted across two refits; accepted (simulated review)
- 2022-07-08T19:00:00+00:00: live LL alarm, entries frozen
- 2022-11-09T11:00:00+00:00: live LL alarm, entries frozen
- 2023-03-04T11:00:00+00:00: live LL alarm, entries frozen
- 2023-07-16T04:00:00+00:00: live LL alarm, entries frozen
- 2023-08-18T22:00:00+00:00: live LL alarm, entries frozen
- 2023-08-30T10:00:00+00:00: live LL alarm, entries frozen
- 2024-03-06T12:00:00+00:00: live LL alarm, entries frozen
- 2024-04-14T19:00:00+00:00: live LL alarm, entries frozen
- 2024-08-05T15:00:00+00:00: live LL alarm, entries frozen
- 2025-01-28T09:00:00+00:00: live LL alarm, entries frozen
- 2025-03-02T22:00:00+00:00: live LL alarm, entries frozen
- 2025-10-11T02:00:00+00:00: live LL alarm, entries frozen
- 2026-01-30T19:00:00+00:00: live LL alarm, entries frozen
- 2026-06-25T20:00:00+00:00: live LL alarm, entries frozen
- 2026-08-20T10:00:00+00:00: live LL alarm, entries frozen
