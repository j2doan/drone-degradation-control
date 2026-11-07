"""Failure-aware scoring and figures for the motor/force identification pilot.

Only scoring and plotting use hidden effectiveness/force labels. The estimators
run in shadow: these figures cannot demonstrate a control improvement. Optional
Matplotlib is imported only by :func:`plot_matrix`.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

from .config import CONDITIONS, MOTIONS, PilotConfig

MOTOR_ERROR_THRESHOLD = 0.05
FALSE_ALARM_HOLD_S = 0.20
MISSING_TRACE_ERROR_PENALTY_M = 1.0


def _number(value):
    if value in (None, ""):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _truth(value):
    return value is True or str(value).lower() in ("true", "1", "1.0")


def _weighted(values, weights, square=False):
    pairs = [(float(v), w) for v, w in zip(values, weights) if v is not None and w > 0]
    duration = sum(w for _, w in pairs)
    if not duration:
        return None
    average = sum((v*v if square else v)*w for v, w in pairs)/duration
    return math.sqrt(average) if square else average


def _window_weights(times, start, end):
    return [max(0.0, min(b, end)-max(a, start)) for a, b in zip(times, times[1:])]


def _errors(rows, estimate, truth):
    result = []
    for row in rows[:-1]:
        a, b = _number(row.get(estimate)), _number(row.get(truth))
        result.append(None if a is None or b is None else a-b)
    return result


def _alarm(errors, weights):
    """Alarm needs >=0.2 consecutive recorded seconds, not just one sample."""
    run = longest = 0.0
    for error, weight in zip(errors, weights):
        if weight <= 0 or error is None or abs(error) <= MOTOR_ERROR_THRESHOLD:
            run = 0.0
        else:
            run += weight
            longest = max(longest, run)
    if not any(error is not None and w > 0 for error, w in zip(errors, weights)):
        return None
    return longest + 1e-10 >= FALSE_ALARM_HOLD_S


def summarize(rows: list[dict], config: PilotConfig, status: str = "complete") -> dict:
    """Score one trace, retaining unidentified estimates and failed episodes.

    All post-event eta errors include the retained estimate even while the joint
    fit is unidentified. Identification coverage is reported separately. Error
    metrics from an early-ending trace are conditional on observed time. The
    fixed-horizon tracking score clips recorded absolute error to 1 m and assigns
    1 m to unobserved time; failures must still be reported as failures.

    Velocity is part of the logged state but no recovery-time claim is made for
    the moving sinusoidal reference. The same scheduled event anchors apply to
    healthy controls, without asserting that they experienced a fault or gust.
    """
    if len(rows) < 2:
        raise ValueError("At least initial and terminal state rows are required")
    times = [_number(row.get("t_s")) for row in rows]
    if any(t is None for t in times) or abs(times[0]) > 1e-8:
        raise ValueError("Trace must begin at t_s=0 with finite timestamps")
    if any(b <= a or b-a > config.dt_s*1.001 for a, b in zip(times, times[1:])):
        raise ValueError("Trace must have increasing, contiguous control-step timestamps")
    if times[-1] > config.duration_s + config.dt_s*0.01:
        raise ValueError("Trace extends past the declared common horizon")
    for row in rows:
        for key in ("z_m", "vz_m_s", "target_z_m"):
            if _number(row.get(key)) is None:
                raise ValueError(f"Missing or nonfinite state field: {key}")
    failed = status == "failed" or any(_truth(row.get("failed", False)) for row in rows)
    trace_complete = times[-1] >= config.duration_s - config.dt_s*0.01
    if status == "complete" and not trace_complete:
        status = "incomplete"
    if failed:
        status = "failed"
    weights = _window_weights(times, 0.0, config.duration_s)
    observed_s = sum(weights)
    tracking = [float(row["z_m"])-float(row["target_z_m"]) for row in rows[:-1]]
    absolute_integral = sum(min(abs(e), MISSING_TRACE_ERROR_PENALTY_M)*w for e, w in zip(tracking, weights))
    penalized = (absolute_integral + (config.duration_s-observed_s)*MISSING_TRACE_ERROR_PENALTY_M)/config.duration_s
    anchor = min(config.motor_onset_s, config.force_onset_s)
    post_weights = _window_weights(times, anchor, config.duration_s)
    eta_motor = _errors(rows, "motor_only_eta", "eta_true")
    eta_joint = _errors(rows, "joint_eta", "eta_true")
    force_errors = _errors(rows, "joint_force_z_n", "force_true_z_n")
    identified = [_truth(row.get("joint_identified", False)) for row in rows[:-1]]
    post_duration = sum(post_weights)
    has_joint_estimates = any(error is not None for error in eta_joint)
    coverage = (sum(w for okay, error, w in zip(identified, eta_joint, post_weights)
                    if okay and error is not None)/post_duration
                if post_duration and has_joint_estimates else None)
    motor_coverage = (sum(w for error, w in zip(eta_motor, post_weights) if error is not None)/post_duration
                      if post_duration else None)
    joint_coverage = (sum(w for error, w in zip(eta_joint, post_weights) if error is not None)/post_duration
                      if post_duration else None)
    end_force = config.duration_s if config.force_duration_s is None else config.force_onset_s + config.force_duration_s
    force_weights = _window_weights(times, config.force_onset_s, end_force)
    after_force = None if config.force_duration_s is None else _window_weights(times, end_force, config.duration_s)
    wind_only = config.condition == "disturbance"
    result = {
        "condition": config.condition, "motion": config.motion, "seed": config.seed,
        "status": status, "failed": failed, "trace_complete": trace_complete,
        "observed_s": observed_s, "common_horizon_s": config.duration_s,
        "observed_tracking_rmse_m": _weighted(tracking, weights, square=True),
        "penalized_tracking_mae_m": penalized,
        "motor_only_eta_rmse_post_event": _weighted(eta_motor, post_weights, square=True),
        "joint_eta_rmse_post_event": _weighted(eta_joint, post_weights, square=True),
        "joint_identified_coverage_post_event": coverage,
        "motor_only_estimate_coverage_post_event": motor_coverage,
        "joint_estimate_coverage_post_event": joint_coverage,
        "joint_force_rmse_n_post_event": _weighted(force_errors, post_weights, square=True),
        "post_force_anchor_tracking_rmse_m": None if after_force is None else _weighted(tracking, after_force, square=True),
        "motor_only_eta_bias_during_force": _weighted(eta_motor, force_weights) if wind_only else None,
        "joint_eta_bias_during_force": _weighted(eta_joint, force_weights) if wind_only else None,
        "motor_only_false_motor_alarm_during_force": _alarm(eta_motor, force_weights) if wind_only else None,
        "joint_false_motor_alarm_during_force": _alarm(eta_joint, force_weights) if wind_only else None,
        "motor_only_eta_bias_after_force": _weighted(eta_motor, after_force) if wind_only and after_force is not None else None,
        "joint_eta_bias_after_force": _weighted(eta_joint, after_force) if wind_only and after_force is not None else None,
        "scoring_anchor_s": anchor,
        "motor_error_threshold": MOTOR_ERROR_THRESHOLD,
        "false_alarm_hold_s": FALSE_ALARM_HOLD_S,
        "missing_trace_error_penalty_m": MISSING_TRACE_ERROR_PENALTY_M,
    }
    return result


def _format(value, digits=3):
    number = _number(value)
    return "unassessed" if number is None else f"{number:.{digits}f}"


def _group_mean(rows, key):
    values = [_number(row.get(key)) for row in rows]
    valid = [value for value in values if value is not None]
    if not valid:
        return "unassessed"
    label = _format(sum(valid)/len(valid))
    return label if len(valid) == len(rows) else f"{label} [{len(valid)}/{len(rows)} assessed]"


def write_report(output: Path, summaries: list[dict], manifest: dict) -> None:
    """Write a descriptive report without implying estimator feedback control."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    backend = manifest.get("backend", "unspecified")
    failed = sum(_truth(row.get("failed")) for row in summaries)
    incomplete = sum(str(row.get("status")) not in ("complete", "failed") for row in summaries)
    lines = [
        "# Motor-effectiveness / external-force identification pilot", "",
        f"Matrix status: `{manifest.get('status', 'unspecified')}`. Backend: `{backend}`. Episodes: {len(summaries)}; failures: {failed}; incomplete: {incomplete}.", "",
        "These are identification diagnostics. Both estimators observe the same PID flight in shadow;",
        "neither estimate changes the controller. Tracking curves therefore do **not** establish a",
        "control improvement. Portable vertical-model results are not Isaac flight validation. MuJoCo X2 results use full rigid-body physics and remain distinct from Isaac/ARL results.", "",
        "Isaac estimates require a command-only nominal RPM-state replay to pass its per-step",
        "audit against the installed actuator. A model mismatch makes the run incomplete;",
        "absent outputs remain unassessed, never zero estimation error. Logged post-actuator",
        "nominal/realized thrust validates the model and physical hook; it is not an estimator input.", "",
        "Assumptions: one known candidate motor, known mass, known healthy initial actuator state",
        "and white-box nominal actuator coefficients (an explicit simulation-calibration assumption),",
        "world-vertical additive force. MuJoCo uses full rigid-body dynamics. The unknown fault",
        "severity/onset and actual external force are available only to scoring, not estimator inputs.", "",
        "## Condition-level description", "",
        "Values below are arithmetic means of per-episode metrics across the declared paired nuisance seeds, not confidence intervals. Bracketed counts disclose missing assessments. Gate coverage means that the joint fit passes the excitation test; it does not guarantee an accurate estimate.", "",
        "| Motion | Condition | Episodes | Failed / incomplete | Mean motor-only eta RMSE | Mean joint eta RMSE | Mean joint gate coverage |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for motion in manifest.get("motions", MOTIONS):
        for condition in CONDITIONS:
            group = [row for row in summaries if row.get("motion") == motion and row.get("condition") == condition]
            if not group:
                continue
            failed_count = sum(_truth(row.get("failed")) for row in group)
            incomplete_count = sum(str(row.get("status")) not in ("complete", "failed") for row in group)
            lines.append("| " + " | ".join([
                motion, condition, str(len(group)), f"{failed_count} / {incomplete_count}",
                _group_mean(group, "motor_only_eta_rmse_post_event"),
                _group_mean(group, "joint_eta_rmse_post_event"),
                _group_mean(group, "joint_identified_coverage_post_event"),
            ]) + " |")
    lines += [
        "", "## Per-episode records", "",
        "| Motion | Seed | Condition | Status | Tracking RMSE (m, observed) | Motor-only eta RMSE | Joint eta RMSE | Joint identified coverage |",
        "|---|---:|---|---|---:|---:|---:|---:|",
    ]
    for row in summaries:
        lines.append("| " + " | ".join([
            str(row.get("motion")), str(row.get("seed")), str(row.get("condition")), str(row.get("status")),
            _format(row.get("observed_tracking_rmse_m")), _format(row.get("motor_only_eta_rmse_post_event")),
            _format(row.get("joint_eta_rmse_post_event")), _format(row.get("joint_identified_coverage_post_event")),
        ]) + " |")
    lines += ["", "## Read before comparing", "",
        "- Eta RMSE covers all retained estimates after the common first scheduled event anchor, including unidentified periods. Missing estimates are unassessed; the two estimate-coverage fields disclose missing time. Identification/gate coverage measures observed post-anchor time passing the excitation test, not estimator accuracy. A held initial estimate is not proof of identification.",
        "- Observed RMSE and estimation errors are conditional on recorded time. Incomplete/failed traces cannot win a comparison by ending early. `penalized_tracking_mae_m` uses the common horizon, clips recorded absolute error at 1 m, and assigns 1 m to unobserved time. Report failure counts alongside it.",
        "- In disturbance-only episodes, a false motor alarm requires absolute motor-health error greater than 0.05 for at least 0.20 consecutive seconds during the force interval. Bias after the force ends is reported separately; unavailable intervals are unassessed.",
        "- Healthy controls use the same scheduled anchors; no physical fault or gust is asserted for them. A sustained external force has no post-force window.",
        "- No recovery time is claimed for the moving probe reference. Tracking uses the actual prescribed reference, not a fixed hover target.",
        "- Constant command can make motor effectiveness and additive force unidentifiable from vertical motion. Probe motion tests one source of information; it does not guarantee separation under arbitrary disturbances or model error.",
        "- Seeds vary paired initial-state/sensor-noise draws, not independently trained policies. No learned selector, PPO training, novelty claim, or statistical significance is established here.",
        "", "## Files", "",
        "- `manifest.json`: declared configurations, provenance and backend limitations.",
        "- `episodes.csv`: all episode summaries, including failure and identification coverage.",
        "- `<motion>_seed<seed>/<condition>/trajectory.csv`: causal state, commands, estimates and separate hidden scoring labels.",
        "- `diagnostic_<motion>_seed<seed>.png`: first predeclared seed for each motion, if plotting was requested.",
        "", "Next decision: determine whether motor-only adaptation mistakes an external force for damage, whether the joint estimator has enough information, and whether selective updates are worth adding. A favorable shadow estimate is not yet a favorable controller.", "",
    ]
    (output / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")


def _load_rows(path: Path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _segments(times, active):
    """Merge contiguous marked physics intervals for readable plot shading."""
    start = None
    for index, marked in enumerate(active[:-1]):
        if marked and start is None:
            start = times[index]
        if start is not None and (not marked or index == len(times)-2):
            end = times[index] if not marked else times[index+1]
            yield start, end
            start = None


def plot_matrix(output: Path) -> list[Path]:
    """Plot every condition for the first declared nuisance seed of each motion."""
    output = Path(output)
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    seeds = manifest.get("seeds")
    if not seeds:
        raise ValueError("Matrix manifest must declare seed order for representative plots")
    motions = manifest.get("motions", list(MOTIONS))
    seed = int(seeds[0])
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    paths = []
    for motion in motions:
        if motion not in MOTIONS:
            raise ValueError(f"Unknown motion {motion!r}")
        fig, axes = plt.subplots(3, 4, figsize=(16, 9), sharex=True, sharey="row", constrained_layout=True)
        for column, condition in enumerate(CONDITIONS):
            directory = output / f"{motion}_seed{seed}" / condition
            if not (directory / 'trajectory.csv').is_file():
                for axis in axes[:, column]:
                    axis.text(.5,.5,'Not collected',ha='center',va='center',transform=axis.transAxes)
                    axis.set_title(condition)
                continue
            rows = _load_rows(directory / "trajectory.csv")
            times = [float(row["t_s"]) for row in rows]
            eta_ax, force_ax, tracking_ax = axes[:, column]
            for key, label, color, style in (
                ("eta_true", "True motor effectiveness", "black", "--"),
                ("motor_only_eta", "Motor-only estimate", "#d55e00", "-"),
                ("joint_eta", "Joint motor estimate (incl. held values)", "#0072b2", "-"),
            ):
                values = [_number(row.get(key)) for row in rows]
                if any(v is not None for v in values):
                    eta_ax.plot(times, values, label=label, color=color, linestyle=style, linewidth=1.4)
            for key, label, color, style in (
                ("force_true_z_n", "Applied force (scoring truth)", "black", "--"),
                ("joint_force_z_n", "Joint force estimate", "#0072b2", "-"),
            ):
                values = [_number(row.get(key)) for row in rows]
                if any(v is not None for v in values):
                    force_ax.plot(times, values, label=label, color=color, linestyle=style, linewidth=1.4)
            tracking_ax.plot(times, [float(row["z_m"])-float(row["target_z_m"]) for row in rows], color="#009e73", label="Height − prescribed reference")
            tracking_ax.axhline(0, color="gray", linewidth=0.6)
            identified = [_truth(row.get("joint_identified", False)) for row in rows]
            has_estimates = any(_number(row.get("joint_eta")) is not None for row in rows)
            coverage = sum(identified)/len(identified) if identified else 0.0
            assessment = f"{coverage:.0%} of trace" if has_estimates else "unassessed"
            eta_ax.set_title(f"{condition}\nExcitation gate: {assessment}", fontsize=10)
            for start, end in _segments(times, [not okay for okay in identified]):
                for axis in (eta_ax, force_ax):
                    axis.axvspan(start, end, color="gray", alpha=0.12, linewidth=0)
            for start, end in _segments(times, [abs(_number(row.get("force_true_z_n")) or 0) > 0 for row in rows]):
                for axis in axes[:, column]:
                    axis.axvspan(start, end, color="#e69f00", alpha=0.09, linewidth=0)
            etas = [_number(row.get("eta_true")) for row in rows]
            for index in range(1, len(etas)):
                if etas[index] is not None and etas[index-1] is not None and abs(etas[index]-etas[index-1]) > 1e-9:
                    for axis in axes[:, column]:
                        axis.axvline(times[index], color="#7b3294", linestyle=":", linewidth=0.9)
            tracking_ax.set_xlabel("Time (s)")
            for axis in axes[:, column]:
                axis.grid(alpha=0.2)
                axis.tick_params(labelsize=9)
            if column == 0:
                eta_ax.set_ylabel("Motor effectiveness")
                force_ax.set_ylabel("Vertical force (N)")
                tracking_ax.set_ylabel("Tracking error (m)")
        handles, labels = [], []
        for axis in axes[:, 0]:
            new_handles, new_labels = axis.get_legend_handles_labels()
            handles.extend(new_handles)
            labels.extend(new_labels)
        handles.extend([
            Line2D([0], [0], color="#7b3294", linestyle=":", linewidth=0.9),
            Patch(facecolor="#e69f00", alpha=0.20), Patch(facecolor="gray", alpha=0.20),
        ])
        labels.extend(["Motor change (scoring truth)", "External force active (truth)", "Joint excitation gate fails / unassessed"])
        fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.07), ncol=4, fontsize=8)
        fig.suptitle(
            f"Identification diagnostic: {motion}, first declared seed {seed} · {manifest.get('backend', 'unspecified')}\n"
            "Known candidate motor / mass / nominal actuator model; vertical force only. Shadow estimates do not change PID.", fontsize=12)
        path = output / f"diagnostic_{motion}_seed{seed}.png"
        fig.savefig(path, dpi=160, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Completed matrix output directory")
    parser.add_argument("--plot", action="store_true", help="Write the diagnostic matrices with optional Matplotlib")
    args = parser.parse_args()
    if args.plot:
        for path in plot_matrix(args.input):
            print(path)
    else:
        print((args.input / "RESULTS.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
