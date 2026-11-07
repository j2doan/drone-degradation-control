"""Causal, seeded episode loop shared by the baseline experiments."""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Any

from .core import (
    AdaptiveController,
    AltitudePlant,
    ExperimentConfig,
    OracleController,
    PIDController,
)
from .metrics import summarize


METHODS = ("pid", "adaptive", "oracle")


@dataclass
class EpisodeResult:
    rows: list[dict[str, Any]]
    summary: dict[str, Any]


def run_episode(config: ExperimentConfig, method: str, seed: int = 0) -> EpisodeResult:
    """Run one trajectory without exposing hidden effectiveness to real baselines.

    The same seed yields the same *measurement noise sequence* for every method.
    Methods share the scenario, not their subsequent states or commands.
    ``oracle`` alone is intentionally given the current true effectiveness.
    """
    if method not in METHODS:
        raise ValueError(f"Unknown method {method!r}; choose from {METHODS}")
    plant = AltitudePlant(config)
    known = {
        "target_m": config.target_m,
        "mass_kg": config.mass_kg,
        "gravity_m_s2": config.gravity_m_s2,
        "max_thrust_n": config.max_thrust_n,
    }
    if method == "pid":
        controller = PIDController(**known)
    elif method == "adaptive":
        controller = AdaptiveController(**known)
    else:
        controller = OracleController(
            **known, true_eta=lambda: plant.eta_at(plant.state.time_s)
        )

    noise_rng = random.Random(seed)
    rows: list[dict[str, Any]] = []
    for _ in range(config.num_steps):
        before = plant.state
        observation = plant.observe(
            noise_z=noise_rng.gauss(0.0, config.noise_z_m),
            noise_vz=noise_rng.gauss(0.0, config.noise_vz_m_s),
        )
        command = controller.act(observation, config.dt_s)
        after = plant.step(command)
        # State/observation are at t; actuation columns describe [t, t + dt].
        rows.append(
            {
                "time_s": before.time_s,
                "z_m": before.z_m,
                "vz_m_s": before.vz_m_s,
                "observed_z_m": observation.z_m,
                "observed_vz_m_s": observation.vz_m_s,
                "command_n": plant.last_command_n,
                "applied_thrust_n": plant.last_applied_thrust_n,
                "eta_true": plant.eta_at(before.time_s),
                "eta_estimate": controller.eta_estimate,
                "integral_state": controller.integral_state,
            }
        )
        if after.z_m <= config.min_altitude_m or after.z_m >= config.max_altitude_m:
            break

    final = plant.state
    rows.append(
        {
            "time_s": final.time_s,
            "z_m": final.z_m,
            "vz_m_s": final.vz_m_s,
            "observed_z_m": None,
            "observed_vz_m_s": None,
            "command_n": None,
            "applied_thrust_n": None,
            "eta_true": plant.eta_at(final.time_s),
            "eta_estimate": controller.eta_estimate,
            "integral_state": controller.integral_state,
        }
    )
    return EpisodeResult(rows, summarize(rows, config, method, seed))
