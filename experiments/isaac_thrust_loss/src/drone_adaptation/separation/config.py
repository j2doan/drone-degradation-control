"""Shared, simulator-independent specification of the four-condition pilot."""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import math

CONDITIONS = ('healthy', 'motor', 'disturbance', 'combined')
MOTIONS = ('steady', 'probe')


@dataclass(frozen=True)
class PilotConfig:
    condition: str = 'healthy'
    motion: str = 'probe'
    seed: int = 0
    dt_s: float = 0.01
    duration_s: float = 20.0
    motor: int = 3
    motor_loss: float = 0.20
    motor_onset_s: float = 6.0
    motor_duration_s: float | None = None
    force_z_n: float = -0.8
    force_onset_s: float = 10.0
    force_duration_s: float | None = 2.0
    target_z_m: float = 1.0
    probe_onset_s: float = 4.0
    probe_amplitude_m: float = 0.06
    probe_frequency_hz: float = 0.7
    velocity_noise_std_m_s: float = 0.01
    initial_z_jitter_m: float = 0.01
    max_thrust_n: float = 6.0
    history_s: float = 3.0

    def __post_init__(self):
        if self.condition not in CONDITIONS or self.motion not in MOTIONS:
            raise ValueError('Unknown condition or reference motion')
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or not 0 <= self.seed < 2**32:
            raise ValueError('seed must be an integer in [0, 2**32)')
        if isinstance(self.motor, bool) or not isinstance(self.motor, int) or not 0 <= self.motor < 4:
            raise ValueError('motor must be an integer in [0, 3]')
        for name in ('dt_s', 'duration_s', 'target_z_m', 'probe_frequency_hz', 'max_thrust_n', 'history_s'):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f'{name} must be finite and positive')
        if self.dt_s != 0.01:
            raise ValueError('The first pilot fixes the physics/control interval at 0.01 s')
        for name in ('motor_onset_s', 'force_onset_s', 'probe_onset_s', 'probe_amplitude_m',
                     'velocity_noise_std_m_s', 'initial_z_jitter_m'):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f'{name} must be finite and nonnegative')
        if not math.isfinite(self.force_z_n) or not math.isfinite(self.motor_loss) or not 0 <= self.motor_loss < 1:
            raise ValueError('force must be finite and motor_loss must be in [0, 1)')
        if self.target_z_m - self.probe_amplitude_m - self.initial_z_jitter_m <= 0.3:
            raise ValueError('Reference and initial-state jitter must remain above the 0.3 m failure height')
        for stem in ('motor', 'force'):
            onset = getattr(self, stem + '_onset_s')
            duration = getattr(self, stem + '_duration_s')
            if onset >= self.duration_s:
                raise ValueError(f'{stem} onset must precede the common episode end')
            if duration is not None and (not math.isfinite(duration) or duration <= 0 or onset + duration >= self.duration_s):
                raise ValueError(f'{stem} duration must be positive and end before the episode does')
        if self.probe_onset_s >= self.motor_onset_s or self.probe_onset_s >= self.force_onset_s:
            raise ValueError('Probe must start before either event, independent of the condition')
        if not 0.3 <= self.history_s <= self.duration_s:
            raise ValueError('history must hold at least three 0.1 s integration windows and fit inside the episode')
        for name in ('duration_s', 'motor_onset_s', 'force_onset_s', 'probe_onset_s', 'motor_duration_s', 'force_duration_s'):
            value = getattr(self, name)
            if value is not None and abs(value / self.dt_s - round(value / self.dt_s)) > 1e-7:
                raise ValueError(f'{name} must be on the 0.01 s grid')

    def effectiveness(self, t_s: float) -> tuple[float, ...]:
        active = self.condition in ('motor', 'combined') and t_s >= self.motor_onset_s and (
            self.motor_duration_s is None or t_s < self.motor_onset_s + self.motor_duration_s)
        return tuple(1.0-self.motor_loss if active and i == self.motor else 1.0 for i in range(4))

    def force(self, t_s: float) -> tuple[float, float, float]:
        active = self.condition in ('disturbance', 'combined') and t_s >= self.force_onset_s and (
            self.force_duration_s is None or t_s < self.force_onset_s + self.force_duration_s)
        return (0.0, 0.0, self.force_z_n if active else 0.0)

    def reference(self, t_s: float) -> tuple[float, float]:
        """Height and height rate. Probe envelope rises smoothly in one second."""
        elapsed = t_s - self.probe_onset_s
        if self.motion == 'steady' or elapsed <= 0:
            return self.target_z_m, 0.0
        phase = 2*math.pi*self.probe_frequency_hz*elapsed
        s = min(elapsed, 1.0)
        envelope = 3*s*s - 2*s*s*s
        envelope_rate = 6*s*(1-s) if elapsed < 1.0 else 0.0
        offset = self.probe_amplitude_m*envelope*math.sin(phase)
        speed = self.probe_amplitude_m*(envelope_rate*math.sin(phase)
                    + envelope*2*math.pi*self.probe_frequency_hz*math.cos(phase))
        return self.target_z_m + offset, speed

    def to_dict(self):
        return asdict(self)


def matrix(base: PilotConfig, seeds=(0,), motions=MOTIONS):
    """Pair every condition with exactly the same motion and nuisance seed."""
    seeds = tuple(seeds)
    motions = tuple(motions)
    if len(set(seeds)) != len(seeds) or len(set(motions)) != len(motions) or not seeds or not motions:
        raise ValueError('seeds and motions must be nonempty and unique')
    return [replace(base, condition=c, seed=s, motion=m) for m in motions for s in seeds for c in CONDITIONS]
