"""Load the Skydio X2 and work out the numbers the controller and the scoring need.

The X2 model (from MuJoCo Menagerie, Apache-2.0) is bundled in ../assets/skydio_x2 and used as published, plus one
forward-looking camera ("front") attached to the body. A camera has no mass or collision, so it never changes the
physics.
"""

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

from . import config

EXPERIMENT_DIR = Path(__file__).resolve().parents[1]  # the camera_effect folder, wherever it is
BUNDLED_MODEL = EXPERIMENT_DIR / "assets"  # holds skydio_x2/


def find_scene() -> Path:
    """Path to skydio_x2/scene.xml: the copy bundled in assets/ (the default), or the one in $MENAGERIE_DIR if that
    environment variable is set (to use a different MuJoCo Menagerie checkout)."""
    candidates = []
    if os.environ.get("MENAGERIE_DIR"):
        candidates.append(Path(os.environ["MENAGERIE_DIR"]))
    candidates.append(BUNDLED_MODEL)
    for folder in candidates:
        scene = folder.expanduser().resolve() / "skydio_x2" / "scene.xml"
        if scene.is_file():
            return scene
    raise FileNotFoundError("Skydio X2 model not found. Tried: " + ", ".join(map(str, candidates))
                            + ". Restore assets/skydio_x2 (from https://github.com/google-deepmind/mujoco_menagerie)"
                            " or set MENAGERIE_DIR to a Menagerie checkout.")


def model_file_hashes(scene: Path) -> dict:
    """sha256 (first 12 hex) of every file of the X2 model, so a run records exactly which model it used."""
    folder = scene.parent
    return {str(p.relative_to(folder)): hashlib.sha256(p.read_bytes()).hexdigest()[:12]
            for p in sorted(folder.rglob("*")) if p.is_file()}


def build_model(scene: Path | None = None) -> mujoco.MjModel:
    """Compile the X2 scene with the front camera added."""
    spec = mujoco.MjSpec.from_file(str(scene or find_scene()))
    body = spec.body("x2")
    # Camera axes in the body frame: image x = body -y (right), image y = body z (up); MuJoCo cameras look along -z,
    # which is then body +x (forward).
    rot = np.array([[0.0, 0.0, -1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])  # columns = camera x, y, z
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, rot.flatten())
    body.add_camera(name="front", pos=list(config.CAMERA_POS), quat=list(quat), fovy=config.CAMERA_FOVY)
    return spec.compile()


@dataclass
class DroneFacts:
    mass: float  # kg
    inertia: np.ndarray  # (3, 3) kg m^2, body frame, about the centre of mass
    com: np.ndarray  # (3,) centre of mass in the body frame
    allocation: np.ndarray  # (4, 4) rows Fz, Tx, Ty, Tz; columns motors 1-4
    hover_thrust: np.ndarray  # (4,) N per motor for a level hover
    actuator_names: list
    native_range: np.ndarray  # (4, 2) N


def drone_facts(model: mujoco.MjModel) -> DroneFacts:
    """Mass, inertia and the motor allocation matrix, read from the compiled model (nothing hand-entered)."""
    body = model.body("x2").id
    mass = float(model.body_subtreemass[body])
    com = model.body_ipos[body].copy()
    rot = np.zeros(9)
    mujoco.mju_quat2Mat(rot, model.body_iquat[body])
    rot = rot.reshape(3, 3)
    inertia = rot @ np.diag(model.body_inertia[body]) @ rot.T

    allocation = np.zeros((4, model.nu))
    for i in range(model.nu):
        site = model.actuator_trnid[i, 0]
        r = model.site_pos[site] - com  # thrust point relative to the centre of mass
        gear = model.actuator_gear[i]  # force along body z, and a reaction torque about body z
        allocation[:, i] = [gear[2], r[1] * gear[2], -r[0] * gear[2], gear[5]]
    hover = np.linalg.solve(allocation, [mass * config.GRAVITY, 0.0, 0.0, 0.0])
    return DroneFacts(mass, inertia, com, allocation, hover,
                      [model.actuator(i).name for i in range(model.nu)], model.actuator_ctrlrange.copy())


def hover_feasible(facts: DroneFacts, motor: int, loss: float, cap: float = config.MAX_THRUST) -> bool:
    """Can the damaged drone still hold a level hover? (motor 1-4, loss 0-1)

    A level hover needs zero torque, which with four motors has exactly one solution: facts.hover_thrust. The damaged
    motor can deliver at most cap * (1 - loss). If that is not enough, no controller can hold a level hover, so a
    crash is the airframe's limit rather than the controller's fault.
    """
    limits = np.full(4, cap)
    limits[motor - 1] *= 1.0 - loss
    return bool(np.all(facts.hover_thrust <= limits + 1e-9))


def max_survivable_loss(facts: DroneFacts, motor: int, cap: float = config.MAX_THRUST) -> float:
    """Largest single-motor loss that still allows a level hover."""
    return float(1.0 - facts.hover_thrust[motor - 1] / cap)
