# Can the estimator separate motor damage from an external force?

Completed: 40 MuJoCo episodes on 4 November 2026. Estimating motor health and
force together was less accurate than the simpler motor-only estimate in the
reported fault conditions. Adding an unknown quantity did not automatically
improve the fit.

## What we tested

The same PID flew a simulated Skydio X2 while two estimators watched its motion:
one fitted motor effectiveness alone; the other fitted effectiveness and an
external vertical force. Neither estimator changed the flight commands.

There were four conditions: healthy, 20% loss on motor 3 from 6 s, a -0.8 N
force during 10–12 s, and both together. Each ran with steady hover and a small
altitude probe, over five paired initial-state/noise seeds: 40 flights of 20 s.
The model included full rigid-body dynamics and motor lag. The candidate motor,
mass and nominal actuator model were known; fault severity/onset were hidden.

## Results

All 40 flights completed without physical failure or incomplete execution.
For the probing flights, mean motor-effectiveness RMSE across five seeds was:

| Condition | Motor-only estimate | Joint motor/force estimate |
|---|---:|---:|
| Motor loss | 0.050 | 0.273 |
| External force | 0.066 | 0.440 |
| Both | 0.073 | 0.552 |

Lower is better. These are estimation errors, not position errors or controller
scores. The joint fit passed its input-variation check throughout the scored
interval in these probing cases, but that did not make its estimates accurate.

## What we learned

The joint baseline was unstable near changes, despite having enough input
variation to attempt a fit. Its windowing, model assumptions and transition
handling need investigation before adding learned selective updates.

Because both estimators watched the same PID flight, this experiment shows no
control benefit. The absence of crashes applies to this mild test grid. An
earlier 40-episode identification pilot used a separate 1D model; its numbers
should not be pooled with these MuJoCo results.

[Saved report](data/mujoco-development-2026-11-04/RESULTS.md) ·
[Per-episode data](data/mujoco-development-2026-11-04/episodes.csv) ·
[Setup, commands and assumptions](GUIDE.md)
