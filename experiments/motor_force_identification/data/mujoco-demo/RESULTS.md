# Motor-effectiveness / external-force identification pilot

Matrix status: `complete`. Backend: `mujoco_x2`. Episodes: 1; failures: 0; incomplete: 0.

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
| probe | combined | 1 | 0 / 0 | 0.072 | 0.582 | 1.000 |

## Per-episode records

| Motion | Seed | Condition | Status | Tracking RMSE (m, observed) | Motor-only eta RMSE | Joint eta RMSE | Joint identified coverage |
|---|---:|---|---|---:|---:|---:|---:|
| probe | 0 | combined | complete | 0.176 | 0.072 | 0.582 | 1.000 |

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
