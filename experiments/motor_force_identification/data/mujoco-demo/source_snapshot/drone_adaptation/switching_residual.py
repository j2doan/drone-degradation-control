"""Evaluate existing frozen residual policies on hidden effectiveness schedules.

The original training environment and its checkpoint hashes remain unchanged.
This adapter changes only plant dynamics: its observation layout, noise stream,
PID state, action scaling, reward and episode rules are inherited unchanged.
"""

from typing import Any

from .residual_env import ResidualHoverEnv
from .switching import EffectivenessSchedule, SwitchingAltitudePlant


class SwitchingResidualEnv(ResidualHoverEnv):
    """Keep policy context across internal changes; reset only between episodes.

    The schedule belongs to the plant, not the policy observation or step info.
    A fresh switching plant starts at the same state as the original plant, so
    the observation already prepared by ``super().reset`` remains correct.
    Measuring or preparing the PID again would change the seeded experiment.
    """

    def __init__(self, schedule: EffectivenessSchedule, **kwargs: Any):
        if not isinstance(schedule, EffectivenessSchedule):
            raise ValueError("schedule must be an EffectivenessSchedule")
        self.schedule = schedule
        kwargs.setdefault("randomize", False)
        super().__init__(**kwargs)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        observation, info = super().reset(seed=seed, options=options)
        self.plant = SwitchingAltitudePlant(self.scenario, self.schedule)
        return observation, info
