# Drone Degradation Control Adaptation through Proximal Policy Optimization

Thrust-loss study on the ARL Robot 1 quadrotor in Isaac Lab. `drone/` flies the simulated quadrotor from point A
to point B with a PID controller, optionally knocks out part of one motor's thrust mid-flight, logs the flight (true
state + IMU), and scores and plots it.

## The study
Question: when one motor suddenly loses thrust, can PPO keep the drone flying safely from A to B? A crash is an
automatic fail.

| Drone | Controller | PPO | Status |
|---|---|---|---|
| **D0** (baseline) | PID | none | **this folder, works** |
| D1 | PID | residual PPO (adds a correction to the PID) | not built yet |
| D2 | PID | residual PPO + memory of recent states | not built yet |
| D3 | none | PPO from scratch | not built yet |

All drones fly the same faults: one random motor loses 10-50% of its thrust at a random moment on the way to B.
Every fault comes from a seed, so the same seed gives every drone the identical fault. `rl/` is reserved for the
D1-D3 training code.

## Requirements
- Isaac Lab installed with Isaac Sim (via its `uv` environment), on the **`develop`** branch, on a machine with an
  NVIDIA GPU (Windows or native Linux; WSL does not work for Isaac Sim).
- **Verified on Isaac Lab `develop` commit `49139e1`** (version 3.0.0, Windows 11, RTX 5070 laptop GPU). Newer `develop`
  commits may rename APIs; if something breaks, please report the error together with `git log -1` of your IsaacLab.
- Nothing else to install: pandas, matplotlib and torch come with Isaac Lab's environment.
- **No Isaac Sim?** `controllers/`, `faults/` and `analysis/` never import Isaac. You can work on them, and replot or
  rescore any saved CSV, with plain Python + torch/pandas/matplotlib (see "Without Isaac Sim" below).

## Setup (one time)
1. Open `run.sh` (Git Bash / Linux) or `run.ps1` (Windows PowerShell) and set the Isaac Lab path at the top
   (marked `EDIT THIS`). Or skip editing and set an `ISAACLAB_DIR` environment variable instead.
2. That is all. No other path in this folder is machine-specific.

## How to run
Use `./run.sh` in Git Bash / Linux, or `.\run.ps1` in PowerShell (`./run.sh` does not work in PowerShell):

```bash
cd PATH_TO_THIS_DRONE_DIRECTORY

# PID, no fault, opens the viewer
./run.sh 03_collect_data.py --name pid_healthy

# PID, random fault, example run
./run.sh 03_collect_data.py --fault random --fault_seed 67 --name test_run

# PID, headless, fixed fault
./run.sh 03_collect_data.py --headless --fault fixed --fault_motor 2 --fault_loss 0.3 --fault_onset 4 --name m2_30
```

Each run of `03_collect_data.py` writes to `data/`:
- `<name>.csv`: one row per physics step (100 Hz), columns listed below
- `<name>.png`: one figure, 3 panels over time, with a one-line run summary on top (see "How to read an output figure")
- `<name>.json`: the settings, the fault and the metrics

The first launch takes about a minute; headless runs take about 30 s.

| Step | Command (after `./run.sh`) | What it does | Status |
|---|---|---|---|
| 1 | `01_inspect_env.py --headless` | Prints thrusters, mass, inertia, allocation matrix, thruster params, state at spawn | works |
| 2 | `02_fly_a_to_b.py` | A to B flight, prints progress and the moment the fault hits. Saves nothing | works |
| 3 | `03_collect_data.py --name run1` | Same flight, writes `data/run1.csv`, `.png`, `.json` | works |

Options for steps 2 and 3:

| Flag | Meaning | Default |
|---|---|---|
| `--name run1` | Output file name (step 3 only). Reusing a name overwrites. | `baseline_pid` |
| `--controller pid\|lee` | Our PID, or Isaac Lab's Lee controller (kept as a regression check) | `pid` |
| `--fault none\|random\|fixed` | No fault; a seeded random fault; or the fault given by the next three flags | `none` |
| `--fault_seed N` | Seed for `--fault random`. Same seed = same motor, loss and onset time | 0 |
| `--fault_motor`, `--fault_loss`, `--fault_onset` | Motor 0-3, lost fraction, start time [s], for `--fault fixed` | 0, 0.25, 4.0 |
| `--hold 10` | Seconds to hover at B after arriving | 3 |
| `--start X Y Z` / `--goal X Y Z` | Points A and B in meters (Z is height) | 0 0 1 / 3 0 1 |
| `--speed 0.5` | Speed of the moving target point, m/s | 0.5 |
| `--max_steps N` | Stop after N physics steps (0.01 s each) | no limit |
| `--headless` | No window | window opens |

Random faults: one motor (0-3) loses 10-50% of its thrust, starting while the drone is still travelling: between 3 s
(after the start-up sag settles) and 1 s before it reaches B (3-5 s on the default route), never during the hover.
A faulted run always lasts **at least 10 s after the fault starts** (default route: about 13-15 s); healthy runs last
travel + `--hold`. The fault depends only on the seed and the route (`--start`, `--goal`, `--speed`), not on `--hold`,
so keep the route the same when comparing drones. Ranges are constants at the top of `faults/thrust_loss.py`.

<details><summary>Running without the launcher</summary>

The launchers just do this: go to your IsaacLab folder and run
`uv run --extra isaacsim python "<path-to>/drone/03_collect_data.py" --name run1`.
</details>

## Without Isaac Sim
```bash
python analysis/plots.py data/pid_s1.csv  # redo the plot (start/goal/controller read from data/pid_s1.json)
python analysis/metrics.py data/pid_s1.csv  # print the metrics
python controllers/test_pid.py  # PID sanity tests on a toy drone model
python faults/test_faults.py  # fault sampler tests
```

## Sample baseline run
Seven D0 flights on the verified machine: the healthy drone and fault seeds 0-5. To reproduce them:
```bash
# the healthy flight
./run.sh 03_collect_data.py --headless --name pid_healthy

# seed N (0-5)
./run.sh 03_collect_data.py --headless --fault random --fault_seed N --name pid_sN
```

### How to read it
Each run writes three files to `data/`:
1. **`<name>.json`** first: the settings, the fault and every
   [metric](#metrics-analysismetricspy-tolerances-are-constants-at-the-top).
2. **`<name>.png`**: what happened in that flight, over time (see
   [How to read an output figure](#how-to-read-an-output-figure)).
3. **`<name>.csv`**: the full log, one row per physics step, for your own analysis.

Seed "none" is the healthy reference; every other seed loses part of one motor's thrust. The table below is the
seven `<name>.json` files in short.

| Seed | Fault | Final distance to B | Max drop below target | Recovery | Crashed |
|---|---|---|---|---|---|
| none | - | 0.023 m | - | - | no |
| 0 | motor 3, -21% at 3.08 s | 0.024 m | 0.29 m | 4.6 s | no |
| 1 | motor 1, -48% at 3.29 s | - | 0.75 m | - | **yes**, 4.45 s |
| 2 | motor 3, -22% at 4.63 s | 0.010 m | 0.45 m | 4.9 s | no |
| 3 | motor 3, -19% at 4.60 s | 0.012 m | 0.40 m | 4.6 s | no |
| 4 | motor 2, -30% at 4.95 s | - | 0.73 m | - | **yes**, 7.05 s (skimmed the ground, then flew on to B) |
| 5 | motor 2, -42% at 4.03 s | - | 0.75 m | - | **yes**, 5.36 s |

"Recovery" is the time from the fault until the height is back near the target. The PID survives losses around 20%
and crashes at 30% and above (seeds 1, 4, 5). For reference, Isaac Lab's Lee controller ends 0.007 m from B on the
healthy drone.

## How to read an output figure

- **Title:** controller, fault, and the outcome in one line.
- **Distance to B:** did the drone get there, and when? The light band is "arrived" (within 0.10 m).
- **Altitude** vs the target altitude (gray dashed). The shaded band is the crash zone (below 0.30 m).
- **Motor thrust:** the faulty motor in orange, *asked by the controller* (dashed) vs *actually delivered* (solid); the
  other motors in gray. The gap between the orange lines is the fault; the climbing dashed line is the controller
  fighting it.
- Red dashed vertical line = the fault starts. Red solid vertical line = crash; every line stops there.

### Can these numbers still be reproduced?
Only with this version's code, and the code will keep changing. Changing a constant (a PID gain, the motor cap, the
fault ranges) gives the same seed a different fault or outcome, and the table above no longer applies. So treat these
numbers as a record of this version, not as what later code must produce.

## About the PID baseline: FAQ

1. **What does the PID control, and how?**
It is a cascade: position error -> desired tilt + thrust -> body torques -> 4 motor thrusts. It has integral terms on
height and on attitude (tilt and heading), with anti-windup limits. The gains are constants at the top of
`controllers/pid.py` (`PIDGains`).

2. **Why an integral term on attitude?**
Without it, a weak motor leaves the drone settled level but 0.2-0.4 m away from B: the controller needs a leftover
error to keep pushing the weak motor harder. Real flight controllers (e.g. PX4) use attitude integrators for the same
reason.

3. **How much thrust loss does the PID survive?**
In the [sample baseline run](#sample-baseline-run) it survives losses around 20% and crashes at 30% and above.

## Files
- `run.sh` / `run.ps1`: launchers; the Isaac Lab path is set at the top of each.
- `common.py`: the only file that talks to Isaac Sim. Builds the world (sim, ground, drone at A, IMU, controller,
  fault) and runs the flight loop. Steps 1 to 3 share it.
- `controllers/pid.py`: cascaded PID (position -> desired tilt + thrust -> body torques), pure PyTorch, batched over
  many drones. Gains are constants at the top (`PIDGains`). `test_pid.py` tests it on a toy drone model.
- `faults/thrust_loss.py`: `FaultSpec`, the seeded `sample_fault`, `run_duration`, and `apply` (scales the faulty
  motor's command).
- `analysis/`: `metrics.py`, `plots.py`, `route.py` (progress along / deviation from the A->B line).
- **Viewer lines** (when not `--headless`): green = ideal straight path A to B, red = path the drone actually flew.
- `data/`: run outputs (not tracked by git). CSV columns: `step, t, pos_xyz, quat_xyzw,
  lin_vel_xyz, ang_vel_xyz, setpoint_xyz, thrust_cmd_0..3` (controller's command), `thrust_applied_0..3` (after the
  fault), `fault_active, fault_motor, fault_loss`, `imu_lin_acc_xyz, imu_ang_vel_xyz`.
- `rl/`: reserved for the D1-D3 training code.

## Metrics (`analysis/metrics.py`, tolerances are constants at the top)
- `final_dist_to_B`: where the drone ended up; **blank (NaN) for a crashed flight**, since a crash is a fail
- `path_length` (distance actually flown), `max_deviation` / `mean_deviation` (from the A->B line, in any direction),
  `max_sideways_drift` (horizontal drift off the A->B line, ignoring height)
- `time_to_arrive`: first time within 0.10 m of B (NaN if never)
- `smoothness`: mean step-to-step change of the commanded thrusts (lower = smoother)
- `fault_onset_t`, `max_alt_loss_after_fault` (deepest drop below the target altitude after the fault)
- `recovery_time`: from onset until the height is back within 0.05 m of the target altitude for 0.5 s (NaN if never)
- `crashed` / `crash_t`: height below 0.30 m after the first 0.5 s, or tilted more than 60 deg. Touching the ground
  counts, even if the drone flies on afterwards.

## Things worth knowing
- **How the flight works:** a target point moves along the line A to B at `--speed`, then stops at B. The controller
  turns "where am I vs. the target" into a force and torques; a pseudoinverse of the allocation matrix converts that
  into 4 propeller thrusts; the fault then scales the faulty motor's thrust before it reaches the drone.
- **Initial sag:** the drone spawns with the propellers not yet at hover thrust, so it dips from 1.0 m to about
  0.45 m in the first second. Faults start after 3 s for this reason.
- **Ground height:** the logged height is the drone body's reference point, which sits 0.25 m above the ground when the
  drone rests on it. A height that flattens at 0.25 m means the drone is on the ground; the crash rule uses 0.30 m.
- **IMU:** Isaac Lab's built-in IMU on the drone body, no noise. Acceleration is "proper" acceleration: +9.81 m/s^2 on z
  at hover, 0 in free fall. The first row reads 0 because the sensor has not updated yet. The controller flies on the
  true state, not the IMU.
- **PID overshoot at B:** the target point stops abruptly at B, so the PID overshoots B by up to ~0.2 m before settling.
- **`launch_simulation()` exits the process** when its `with` block ends. Do all saving and plotting inside it.
- The long run of "Failed to find rigid body ... /Looks/..." warnings at shutdown is harmless.
