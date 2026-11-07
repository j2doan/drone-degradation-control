# Motor-effectiveness / external-force identification pilot

Matrix status: `complete`. Backend: `mujoco_x2`. Episodes: 40; failures: 0; incomplete: 0.

These are identification diagnostics. Both estimators observe the same PID flight in shadow;
neither estimate changes the controller. Tracking curves therefore do **not** establish a
control improvement. Portable vertical-model results are not Isaac flight validation. MuJoCo X2 results use full rigid-body physics and remain distinct from Isaac/ARL results.

Isaac estimates require a command-only nominal RPM-state replay to pass its per-step
audit against the installed actuator. A model mismatch makes the run incomplete;
absent outputs remain unassessed, never zero estimation error. Logged post-actuator
nominal/realized thrust validates the model and physical hook; it is not an estimator input.

Assumptions: one known candidate motor, known mass, known healthy initial actuator state
and white-box nominal actuator coefficients (an explicit simulation-calibration assumption),
world-vertical additive force. Attitude is stabilized only in the portable model; MuJoCo uses full rigid-body dynamics. The unknown fault
severity/onset and actual external force are available only to scoring, not estimator inputs.

## Condition-level description

Values below are arithmetic means of per-episode metrics across the declared paired nuisance seeds, not confidence intervals. Bracketed counts disclose missing assessments. Gate coverage means that the joint fit passes the excitation test; it does not guarantee an accurate estimate.

| Motion | Condition | Episodes | Failed / incomplete | Mean motor-only eta RMSE | Mean joint eta RMSE | Mean joint gate coverage |
|---|---|---:|---:|---:|---:|---:|
| steady | healthy | 5 | 0 / 0 | 0.002 | 0.000 | 0.000 |
| steady | motor | 5 | 0 / 0 | 0.050 | 0.361 | 0.986 |
| steady | disturbance | 5 | 0 / 0 | 0.066 | 1.495 | 0.466 |
| steady | combined | 5 | 0 / 0 | 0.072 | 1.267 | 0.986 |
| probe | healthy | 5 | 0 / 0 | 0.002 | 0.101 | 1.000 |
| probe | motor | 5 | 0 / 0 | 0.050 | 0.273 | 1.000 |
| probe | disturbance | 5 | 0 / 0 | 0.066 | 0.440 | 1.000 |
| probe | combined | 5 | 0 / 0 | 0.073 | 0.552 | 1.000 |

## Per-episode records

| Motion | Seed | Condition | Status | Tracking RMSE (m, observed) | Motor-only eta RMSE | Joint eta RMSE | Joint identified coverage |
|---|---:|---|---|---:|---:|---:|---:|
| steady | 0 | healthy | complete | 0.001 | 0.002 | 0.000 | 0.000 |
| steady | 0 | motor | complete | 0.109 | 0.050 | 0.356 | 0.986 |
| steady | 0 | disturbance | complete | 0.104 | 0.066 | 1.470 | 0.466 |
| steady | 0 | combined | complete | 0.152 | 0.072 | 1.285 | 0.986 |
| steady | 1 | healthy | complete | 0.001 | 0.002 | 0.000 | 0.000 |
| steady | 1 | motor | complete | 0.109 | 0.050 | 0.389 | 0.986 |
| steady | 1 | disturbance | complete | 0.104 | 0.066 | 1.513 | 0.466 |
| steady | 1 | combined | complete | 0.152 | 0.072 | 1.246 | 0.986 |
| steady | 2 | healthy | complete | 0.001 | 0.002 | 0.000 | 0.000 |
| steady | 2 | motor | complete | 0.109 | 0.050 | 0.365 | 0.986 |
| steady | 2 | disturbance | complete | 0.104 | 0.066 | 1.540 | 0.466 |
| steady | 2 | combined | complete | 0.152 | 0.073 | 1.254 | 0.986 |
| steady | 3 | healthy | complete | 0.002 | 0.002 | 0.000 | 0.000 |
| steady | 3 | motor | complete | 0.109 | 0.050 | 0.334 | 0.986 |
| steady | 3 | disturbance | complete | 0.104 | 0.066 | 1.479 | 0.466 |
| steady | 3 | combined | complete | 0.152 | 0.072 | 1.270 | 0.986 |
| steady | 4 | healthy | complete | 0.002 | 0.002 | 0.000 | 0.000 |
| steady | 4 | motor | complete | 0.109 | 0.050 | 0.362 | 0.986 |
| steady | 4 | disturbance | complete | 0.104 | 0.066 | 1.471 | 0.466 |
| steady | 4 | combined | complete | 0.152 | 0.072 | 1.278 | 0.986 |
| probe | 0 | healthy | complete | 0.045 | 0.002 | 0.127 | 1.000 |
| probe | 0 | motor | complete | 0.124 | 0.050 | 0.274 | 1.000 |
| probe | 0 | disturbance | complete | 0.117 | 0.065 | 0.444 | 1.000 |
| probe | 0 | combined | complete | 0.176 | 0.072 | 0.582 | 1.000 |
| probe | 1 | healthy | complete | 0.045 | 0.002 | 0.091 | 1.000 |
| probe | 1 | motor | complete | 0.124 | 0.050 | 0.262 | 1.000 |
| probe | 1 | disturbance | complete | 0.117 | 0.066 | 0.453 | 1.000 |
| probe | 1 | combined | complete | 0.176 | 0.073 | 0.521 | 1.000 |
| probe | 2 | healthy | complete | 0.045 | 0.002 | 0.134 | 1.000 |
| probe | 2 | motor | complete | 0.124 | 0.050 | 0.274 | 1.000 |
| probe | 2 | disturbance | complete | 0.117 | 0.066 | 0.427 | 1.000 |
| probe | 2 | combined | complete | 0.176 | 0.073 | 0.547 | 1.000 |
| probe | 3 | healthy | complete | 0.045 | 0.002 | 0.077 | 1.000 |
| probe | 3 | motor | complete | 0.124 | 0.050 | 0.265 | 1.000 |
| probe | 3 | disturbance | complete | 0.117 | 0.066 | 0.441 | 1.000 |
| probe | 3 | combined | complete | 0.176 | 0.073 | 0.546 | 1.000 |
| probe | 4 | healthy | complete | 0.045 | 0.002 | 0.079 | 1.000 |
| probe | 4 | motor | complete | 0.124 | 0.050 | 0.289 | 1.000 |
| probe | 4 | disturbance | complete | 0.117 | 0.066 | 0.432 | 1.000 |
| probe | 4 | combined | complete | 0.176 | 0.073 | 0.565 | 1.000 |

## Read before comparing

- Eta RMSE covers all retained estimates after the common first scheduled event anchor, including unidentified periods. Missing estimates are unassessed; the two estimate-coverage fields disclose missing time. Identification/gate coverage measures observed post-anchor time passing the excitation test, not estimator accuracy. A held initial estimate is not proof of identification.
- Observed RMSE and estimation errors are conditional on recorded time. Incomplete/failed traces cannot win a comparison by ending early. `penalized_tracking_mae_m` uses the common horizon, clips recorded absolute error at 1 m, and assigns 1 m to unobserved time. Report failure counts alongside it.
- In disturbance-only episodes, a false motor alarm requires absolute motor-health error greater than 0.05 for at least 0.20 consecutive seconds during the force interval. Bias after the force ends is reported separately; unavailable intervals are unassessed.
- Healthy controls use the same scheduled anchors; no physical fault or gust is asserted for them. A sustained external force has no post-force window.
- No recovery time is claimed for the moving probe reference. Tracking uses the actual prescribed reference, not a fixed hover target.
- Constant command can make motor effectiveness and additive force unidentifiable from vertical motion. Probe motion tests one source of information; it does not guarantee separation under arbitrary disturbances or model error.
- Seeds vary paired initial-state/sensor-noise draws, not independently trained policies. No learned selector, PPO training, novelty claim, or statistical significance is established here.

## Files

- `manifest.json`: declared configurations, provenance and backend limitations.
- `episodes.csv`: all episode summaries, including failure and identification coverage.
- `<motion>_seed<seed>/<condition>/trajectory.csv`: causal state, commands, estimates and separate hidden scoring labels.
- `diagnostic_<motion>_seed<seed>.png`: first predeclared seed for each motion, if plotting was requested.

Next decision: determine whether motor-only adaptation mistakes an external force for damage, whether the joint estimator has enough information, and whether selective updates are worth adding. A favorable shadow estimate is not yet a favorable controller.
