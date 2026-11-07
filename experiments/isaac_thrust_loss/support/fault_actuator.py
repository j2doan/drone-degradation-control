"""Source-based, not Isaac-runtime-validated, plant-side motor-loss adapter.

Official Isaac Lab source documents this path:

``Thruster.compute -> returned MultiRotorActions.thrusts ->
Multirotor._thrust_target_sim -> allocation_matrix -> body force AND torque``.

The adapter wraps an existing thruster instance after its nominal motor lag and
clipping.  It scales the returned thrust once and updates ``applied_thrust`` for
logging, leaving the nominal lag state, command and allocation matrix intact.
The multiplier models thrust effectiveness, not RPM loss.  Equal scaling of the
rotor's lift and reaction torque is an explicit model assumption.

Source references (these do NOT verify the repository's recorded 49139e1 pin):
https://isaac-sim.github.io/IsaacLab/develop/_modules/isaaclab_contrib/actuators/thruster.html
https://isaac-sim.github.io/IsaacLab/release/3.0.0/_modules/isaaclab_contrib/assets/multirotor/multirotor.html

The runner must verify its installed multirotor consumes returned thrusts before
computing the wrench.  Shape/type guards cannot prove that downstream behavior;
``source_info`` records the actual compute method's provenance for the run log.
Only a single four-motor environment is supported.  Torch is imported lazily.
"""

from __future__ import annotations

import functools
import hashlib
import inspect
import math
from typing import Any

if __package__:
    from .faults import StepMotorFault
else:  # Existing standalone flight scripts retain their original import path.
    from faults import StepMotorFault


_OWNER_ATTRIBUTE = "_capstone_motor_fault_adapter"


def _same_callable(left, right) -> bool:
    """Compare repeated bound-method lookups without relying on object identity."""
    return (getattr(left, "__self__", None) is getattr(right, "__self__", None)
            and getattr(left, "__func__", left) is getattr(right, "__func__", right))


def _source_info(method) -> dict[str, Any]:
    info = {
        "qualified_name": getattr(method, "__qualname__", type(method).__name__),
        "module": getattr(method, "__module__", None),
        "source_file": None,
        "source_start_line": None,
        "source_sha256": None,
        "inspection_error": None,
    }
    try:
        info["source_file"] = inspect.getsourcefile(method) or inspect.getfile(method)
        lines, start = inspect.getsourcelines(method)
        info["source_start_line"] = start
        info["source_sha256"] = hashlib.sha256("".join(lines).encode("utf-8")).hexdigest()
    except (OSError, TypeError) as exc:
        info["inspection_error"] = f"{type(exc).__name__}: {exc}"
    return info


class MotorFaultAdapter:
    """Wrap a single actuator instance's compute method, keeping its class.

    Usage::

        adapter = MotorFaultAdapter(robot.actuators["thrusters"], fault).install()
        adapter.set_time(t)  # physics interval begins at t; no event preview
        robot.write_data_to_sim()
        # nominal_thrust/realized_thrust are now detached snapshots, shape (1,4)
        adapter.close()

    Installation is idempotent for this adapter, and a second adapter is
    rejected.  ``close`` is idempotent and restores the original instance lookup
    exactly.  An intervening unknown compute wrapper causes a clear error;
    closing never silently overwrites it.  If an output contract check fails,
    abort that episode: the original actuator may already have advanced its lag
    state, but no validated scaled action was returned for physics application.
    """

    def __init__(self, actuator, fault: StepMotorFault):
        if not isinstance(fault, StepMotorFault):
            raise ValueError("fault must be a StepMotorFault")
        original = getattr(actuator, "compute", None)
        if not callable(original):
            raise ValueError("actuator must expose a callable compute(control_action)")
        self.actuator = actuator
        self._fault = fault
        self._original_compute = original
        self._source_info = _source_info(original)
        self._time_s = 0.0
        self._installed = False
        self._wrapped_compute = None
        self._had_instance_compute = "compute" in getattr(actuator, "__dict__", {})
        self._original_instance_compute = getattr(actuator, "__dict__", {}).get("compute")
        self.nominal_thrust = None
        self.realized_thrust = None
        self.last_requested = None
        self.calls = 0

    @property
    def source_info(self) -> dict[str, Any]:
        """Actual pre-hook compute provenance; not evidence of downstream ABI."""
        return dict(self._source_info)

    @property
    def installed(self) -> bool:
        return self._installed

    @property
    def time_s(self) -> float:
        return self._time_s

    @property
    def fault(self) -> StepMotorFault:
        return self._fault

    def set_fault(self, fault: StepMotorFault) -> None:
        """Arm a new immutable specification without resetting nominal motor lag.

        The runner can begin with effectiveness one and arm the actual fault
        after a measured settling condition.  Its onset is still an absolute
        simulation time.  This method does not advance time or invoke compute.
        """
        if not isinstance(fault, StepMotorFault):
            raise ValueError("fault must be a StepMotorFault")
        self._fault = fault

    def set_time(self, t: float) -> None:
        if isinstance(t, bool):
            raise ValueError("time must be a finite nonnegative number")
        try:
            t = float(t)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("time must be a finite nonnegative number") from exc
        if not math.isfinite(t) or t < 0:
            raise ValueError("time must be a finite nonnegative number")
        self._time_s = t

    def install(self) -> MotorFaultAdapter:
        owner = getattr(self.actuator, _OWNER_ATTRIBUTE, None)
        if self._installed:
            if owner is not self or not _same_callable(self.actuator.compute, self._wrapped_compute):
                raise RuntimeError("actuator compute changed after fault adapter installation")
            return self
        if owner is not None:
            raise RuntimeError("an actuator may have only one motor fault adapter")
        if not _same_callable(self.actuator.compute, self._original_compute):
            raise RuntimeError("actuator compute changed before fault adapter installation")

        @functools.wraps(self._original_compute)
        def wrapped_compute(control_action):
            return self._compute(control_action)

        self._wrapped_compute = wrapped_compute
        setattr(self.actuator, _OWNER_ATTRIBUTE, self)
        try:
            self.actuator.compute = wrapped_compute
        except Exception:
            delattr(self.actuator, _OWNER_ATTRIBUTE)
            self._wrapped_compute = None
            raise
        self._installed = True
        return self

    def close(self) -> None:
        if not self._installed:
            return
        if (getattr(self.actuator, _OWNER_ATTRIBUTE, None) is not self
                or not _same_callable(self.actuator.compute, self._wrapped_compute)):
            raise RuntimeError("actuator compute changed; refusing to overwrite another wrapper")
        if self._had_instance_compute:
            self.actuator.compute = self._original_instance_compute
        else:
            delattr(self.actuator, "compute")
        delattr(self.actuator, _OWNER_ATTRIBUTE)
        self._installed = False

    @staticmethod
    def _require_thrust_tensor(value, label: str):
        import torch

        if not isinstance(value, torch.Tensor):
            raise RuntimeError(f"{label} must be a torch.Tensor; unsupported actuator interface")
        if tuple(value.shape) != (1, 4):
            raise RuntimeError(f"{label} must have shape (1, 4), got {tuple(value.shape)}")
        if not value.is_floating_point() or not bool(torch.isfinite(value).all().item()):
            raise RuntimeError(f"{label} must contain finite floating-point thrusts")
        return value

    def _compute(self, control_action):
        import torch

        if (not self._installed or getattr(self.actuator, _OWNER_ATTRIBUTE, None) is not self
                or not _same_callable(self.actuator.compute, self._wrapped_compute)):
            raise RuntimeError("fault adapter is no longer the active compute wrapper")
        requested = self._require_thrust_tensor(
            getattr(control_action, "thrusts", None), "control_action.thrusts")
        requested_snapshot = requested.detach().clone()
        output = self._original_compute(control_action)
        nominal = self._require_thrust_tensor(getattr(output, "thrusts", None),
                                               "compute output.thrusts")
        applied = self._require_thrust_tensor(getattr(self.actuator, "applied_thrust", None),
                                              "actuator.applied_thrust")
        if not torch.equal(applied, nominal):
            raise RuntimeError("compute output and actuator.applied_thrust disagree; unsupported interface")
        if bool((nominal < 0).any().item()):
            raise RuntimeError("compute output.thrusts must be nonnegative for this motor-loss model")
        factors = torch.tensor(self.fault.factors_at(self._time_s),
                               device=nominal.device, dtype=nominal.dtype).unsqueeze(0)
        # Multiplication allocates new storage: never scale a motor-lag alias.
        realized = nominal * factors
        self._require_thrust_tensor(realized, "realized thrust")
        output.thrusts = realized
        self.actuator.applied_thrust = realized
        self.last_requested = requested_snapshot
        self.nominal_thrust = nominal.detach().clone()
        self.realized_thrust = realized.detach().clone()
        self.calls += 1
        return output
