# First baseline experiment

These are generated **one-dimensional model checks**, not Isaac Sim, learned-policy,
motor-failure, or sim-to-real results. Attitude is assumed perfectly stabilized.
PID and the explicit estimator receive noisy measurements and issued commands.
The oracle is intentionally privileged and is only a diagnostic reference.

| Effectiveness | Altitude noise (m) | Method | Recovered | Failed | Capped recovery (s) | Peak loss (m) |
|---:|---:|---|---:|---:|---:|---:|
| 1.00 | 0.000 | pid | 5/5 | 0/5 | 0.000 | 0.000 |
| 1.00 | 0.000 | adaptive | 5/5 | 0/5 | 0.000 | 0.000 |
| 1.00 | 0.000 | oracle | 5/5 | 0/5 | 0.000 | 0.000 |
| 1.00 | 0.010 | pid | 5/5 | 0/5 | 0.000 | 0.005 |
| 1.00 | 0.010 | adaptive | 5/5 | 0/5 | 0.000 | 0.006 |
| 1.00 | 0.010 | oracle | 5/5 | 0/5 | 0.000 | 0.005 |
| 0.90 | 0.000 | pid | 5/5 | 0/5 | 3.020 | 0.189 |
| 0.90 | 0.000 | adaptive | 5/5 | 0/5 | 1.700 | 0.130 |
| 0.90 | 0.000 | oracle | 5/5 | 0/5 | 0.000 | 0.000 |
| 0.90 | 0.010 | pid | 5/5 | 0/5 | 3.040 | 0.188 |
| 0.90 | 0.010 | adaptive | 5/5 | 0/5 | 1.708 | 0.128 |
| 0.90 | 0.010 | oracle | 5/5 | 0/5 | 0.000 | 0.005 |
| 0.75 | 0.000 | pid | 5/5 | 0/5 | 4.440 | 0.583 |
| 0.75 | 0.000 | adaptive | 5/5 | 0/5 | 5.216 | 0.364 |
| 0.75 | 0.000 | oracle | 5/5 | 0/5 | 0.000 | 0.000 |
| 0.75 | 0.010 | pid | 5/5 | 0/5 | 4.464 | 0.582 |
| 0.75 | 0.010 | adaptive | 5/5 | 0/5 | 5.260 | 0.362 |
| 0.75 | 0.010 | oracle | 5/5 | 0/5 | 0.000 | 0.005 |

## How to read this

- Effectiveness 1.0 is a no-fault control; its fault time is only an evaluation anchor.
- Recovery requires true altitude error <= 0.10 m and speed <= 0.15 m/s for 1 second.
- Recovery time is the beginning of that sustained interval; confirmation is logged separately.
- Nonrecoveries receive the common horizon (normally 8 s) in capped recovery time.
- Any physical boundary failure disqualifies recovery, even after an earlier settled interval.
- Post-fault RMSE covers only the recorded trace; a failure shortens it. Never compare it alone.
- Seeds here vary evaluation noise and onset, **not independently trained policies**.
- Gains and estimator settings are prototype defaults, not a tuned final benchmark.

## Files

- `manifest.json`: exact scenarios, configuration, controller settings, runtime and provenance.
- `summary.csv`: one row per method/scenario, retaining nonrecoveries and failures.
- `trajectories.csv`: true state, measurements, issued commands and hidden diagnostic state.
  A row at t stores state/measurement at t and actuation over [t,t+dt].
  The terminal row has no measurement or command because no further action is taken.

## Next check

Run `python3 -m unittest discover -s tests -v`, inspect the altitude and command traces,
then follow `docs/FIRST_WEEK.md`. Add learned residuals after these baseline checks pass.
