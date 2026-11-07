# Altitude control after thrust loss

Completed: a 90-episode study in a simple 1D simulator. The main result was a
tradeoff: estimating the lost thrust reduced the initial drop, but did not
always help the drone settle sooner.

## What we tested

Could an explicit estimate of thrust effectiveness improve on PID alone?
We compared PID, PID with an effectiveness estimator, and an oracle told the
true effectiveness. The drone hovered at 1.5 m before an unannounced change.

The grid covered healthy thrust, 10% loss and 25% loss, with two noise levels
and five paired noise/onset seeds for each of the three methods. Attitude was
assumed perfectly stabilized; this was collective thrust loss, not a damaged
individual rotor. Only the oracle received fault truth.

## Results

All 90 episodes met the recovery criterion, with no boundary failures. Below are means over the
five seeds with altitude-noise standard deviation 0.01 m:

| Thrust loss | Controller | Recovery time (s) | Peak altitude loss (m) |
|---|---|---:|---:|
| 10% | PID | 3.04 | 0.188 |
| 10% | Adaptive PID | 1.71 | 0.128 |
| 25% | PID | 4.46 | 0.582 |
| 25% | Adaptive PID | 5.26 | 0.362 |

Recovery required altitude error within 0.10 m and speed within 0.15 m/s for
one second. Time is measured to the start of that hold, with confirmation
required within eight seconds.

## What we learned

Adaptation helped both drop and recovery time at 10% loss. At 25%, it limited
the drop but overshot and settled later than PID. The initial dip and recovery
time therefore tell different parts of the story.

These are prototype gains in a known-mass vertical model, with no external
force or actuator lag. They do not establish a better full-flight controller.
The five seeds vary evaluation conditions, not independently trained policies.

[Saved report](data/starter/RESULTS.md) ·
[Per-episode data](data/starter/summary.csv) ·
[Setup, commands and metrics](GUIDE.md)
