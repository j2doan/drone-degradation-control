# Drone Degradation Control Adaptation through Proximal Policy Optimization

Proof of concept: `drone/` flies a simulated ARL Robot 1 quadrotor in Isaac Lab from point A to point B and logs
the data. The thrust-loss faults, the PID baseline and the PPO controllers are not built yet.

## Requirements
- Isaac Lab installed with Isaac Sim (via its `uv` environment), on the **`develop`** branch, on a machine with an
  NVIDIA GPU (Windows or native Linux; WSL does not work for Isaac Sim).
- **Verified on Isaac Lab `develop` commit `49139e1`** (version 3.0.0, Windows 11, RTX 5070 laptop GPU). Newer `develop`
  commits may rename APIs; if something breaks, please report the error together with `git log -1` of your IsaacLab.
- Nothing else to install: pandas, matplotlib and torch come with Isaac Lab's environment.

## Setup (one time)
1. Open `run.sh` (Git Bash / Linux) or `run.ps1` (Windows PowerShell) and set the Isaac Lab path at the top
   (marked `EDIT THIS`). Or skip editing and set an `ISAACLAB_DIR` environment variable instead.
2. That is all. No other path in this folder is machine-specific.

## How to run
Run from a terminal, from anywhere. Use `./run.sh` in Git Bash / Linux, or `.\run.ps1` in PowerShell:

```bash
cd <path-to>/this/drone/folder
./run.sh 03_collect_data.py --name run1 --hold 10
```

This opens the Isaac Sim viewer, flies the drone from A to B, and saves `data/run1.csv` and `data/run1.png`.
The first launch takes about a minute. Add `--headless` to skip the window (faster).

| Step | Command (after `./run.sh`) | What it does | Status |
|---|---|---|---|
| 1 | `01_inspect_env.py --headless` | Prints thrusters, mass, allocation matrix, thruster params, state fields | works |
| 2 | `02_fly_a_to_b.py` | Flies A=(0,0,1) to B=(3,0,1) at 0.5 m/s with the Lee controller, hovers 3 s. Prints only, saves nothing | works |
| 3 | `03_collect_data.py --name run1` | Same flight, writes `data/run1.csv` + `.png` | works |

Options for steps 2 and 3:

| Flag | Meaning | Default |
|---|---|---|
| `--name run1` | Output file name (step 3 only). Reusing a name overwrites. | `baseline_lee` |
| `--hold 10` | Seconds to hover at B after arriving | 3 |
| `--start X Y Z` | Point A in meters (Z is height) | 0 0 1 |
| `--goal X Y Z` | Point B in meters | 3 0 1 |
| `--speed 0.5` | Speed of the moving target point, m/s | 0.5 |
| `--max_steps N` | Stop after N physics steps (0.01 s each) | no limit |
| `--headless` | No window | window opens |

<details><summary>Running without the launcher</summary>

The launchers just do this: go to your IsaacLab folder and run
`uv run --extra isaacsim python "<path-to>/drone/03_collect_data.py" --name run1`.
</details>

## Expected result
A good run ends about 0.01 m from B (0.007 m on the verified machine); the minimum height varies by run,
about 0.4 to 0.5 m. These numbers come from this version's code; later versions change the controller and
the flight, so they may not reproduce them.

## How to read an output figure

`data/<name>.png` has three panels over time:
- **Position:** the drone's x, y and z (solid) vs the moving setpoint (dashed).
- **Distance to B:** did the drone get there, and when?
- **Thrust command:** what the controller asks of each of the 4 propellers, in newtons.

## Files
- `run.sh` / `run.ps1`: launchers; the Isaac Lab path is set at the top of each.
- `common.py`: builds the world (sim, ground, drone at A, Lee controller) and the flight loop. Steps 1 to 3 share it.
- **Viewer lines** (when not `--headless`): green = ideal straight path A to B, red = path the drone actually flew.
- `data/`: CSVs and plots. One row per physics step (100 Hz). Columns: `step, t, pos_xyz, quat_xyzw, lin_vel_xyz,
  ang_vel_xyz, setpoint_xyz, thrust_cmd_0..3`. Load with `pandas.read_csv`.
- `rl/`: reserved for the learning-based controllers.

## Things worth knowing
- **How the scripted flight works:** a setpoint moves along the line A to B at `--speed`. The Lee controller
  (classical, not learned) turns "where am I vs. the setpoint" into a force/torque; a pseudoinverse of the
  allocation matrix converts that into 4 propeller thrusts.
- **Initial sag:** the drone spawns with the propellers already at a default speed but not at hover thrust,
  so it dips from 1.0 m to about 0.5 m in the first second before the controller recovers. Spawn higher, or
  watch `min height` in step 3's output, if the ground gets close.
- **Data is perfect state, not sensors:** position, velocity, orientation and angular velocity come straight from the
  simulator. No camera or IMU is used in this version.
- **Degradation hook:** the thruster model (`robot.cfg.actuators["thrusters"]`) has `thrust_const_range` and
  spin-up/spin-down time constants. Scaling one propeller's thrust is the natural way to model a faulty motor.
- **`launch_simulation()` exits the process** when its `with` block ends. Do all saving and plotting inside it.
- The long run of "Failed to find rigid body ... /Looks/..." warnings at shutdown is harmless.
