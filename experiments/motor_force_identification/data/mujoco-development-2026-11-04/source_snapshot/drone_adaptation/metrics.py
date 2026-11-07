"""Recovery metrics computed from true state, never from noisy observations."""

from __future__ import annotations

import math
from typing import Any

from .core import ExperimentConfig


def summarize(
    rows: list[dict[str, Any]],
    config: ExperimentConfig,
    method: str,
    seed: int,
    *,
    recovery_horizon_s: float = 8.0,
    position_tolerance_m: float = 0.10,
    velocity_tolerance_m_s: float = 0.15,
    hold_s: float = 1.0,
) -> dict[str, Any]:
    """Keep nonrecoveries in the headline metrics.

    Recovery time is the beginning of the first in-tolerance interval lasting
    ``hold_s``. Confirmation time is reported separately. A crash disqualifies
    recovery even if an earlier interval happened to pass the tolerance test.
    RMSE covers the available post-fault trace and is explicitly conditional on
    trace survival; use failure/recovery rate and capped time for comparisons.
    """
    if not rows:
        raise ValueError("Cannot summarize an empty trajectory")
    if recovery_horizon_s <= 0 or hold_s <= 0:
        raise ValueError("Recovery horizon and hold duration must be positive")
    if position_tolerance_m <= 0 or velocity_tolerance_m_s <= 0:
        raise ValueError("Recovery tolerances must be positive")

    horizon = min(recovery_horizon_s, config.duration_s - config.fault_time_s)
    deadline = config.fault_time_s + horizon
    failed = any(
        row["z_m"] <= config.min_altitude_m
        or row["z_m"] >= config.max_altitude_m
        for row in rows
    )
    post_fault = [
        row for row in rows if row["time_s"] >= config.fault_time_s - 1e-9
    ]
    recovery_time = None
    confirmation_time = None
    candidate_start = None
    if not failed:
        for row in post_fault:
            if row["time_s"] > deadline + 1e-9:
                break
            inside = (
                abs(row["z_m"] - config.target_m) <= position_tolerance_m
                and abs(row["vz_m_s"]) <= velocity_tolerance_m_s
            )
            if not inside:
                candidate_start = None
                continue
            if candidate_start is None:
                candidate_start = row["time_s"]
            if row["time_s"] - candidate_start >= hold_s - 1e-9:
                recovery_time = max(0.0, candidate_start - config.fault_time_s)
                confirmation_time = max(0.0, row["time_s"] - config.fault_time_s)
                break

    rms = (
        math.sqrt(
            sum((row["z_m"] - config.target_m) ** 2 for row in post_fault)
            / len(post_fault)
        )
        if post_fault
        else None
    )
    commands = [row["command_n"] for row in post_fault if row["command_n"] is not None]
    saturation_fraction = (
        sum(command <= 1e-9 or command >= config.max_thrust_n - 1e-9 for command in commands)
        / len(commands)
        if commands
        else None
    )
    estimate_rows = [row for row in rows if row.get("eta_estimate") is not None]
    return {
        "method": method,
        "seed": seed,
        "eta_after": config.eta_after,
        "noise_z_m": config.noise_z_m,
        "noise_vz_m_s": config.noise_vz_m_s,
        "fault_time_s": config.fault_time_s,
        "recovered": recovery_time is not None,
        "failed": failed,
        "recovery_time_s": recovery_time,
        "recovery_confirmed_time_s": confirmation_time,
        "capped_recovery_time_s": recovery_time if recovery_time is not None else horizon,
        "recovery_horizon_s": horizon,
        "peak_altitude_loss_m": max(
            [0.0] + [config.target_m - row["z_m"] for row in post_fault]
        ),
        "post_fault_rmse_m": rms,
        "post_fault_rmse_is_conditional": failed,
        "post_fault_trace_duration_s": max(
            0.0, rows[-1]["time_s"] - config.fault_time_s
        ),
        "saturation_fraction": saturation_fraction,
        "final_eta_estimate": estimate_rows[-1]["eta_estimate"] if estimate_rows else None,
    }
