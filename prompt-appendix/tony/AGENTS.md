# Instructions for coding assistants: `experiments/`

This folder holds small, self-contained drone-control experiments on a simulated Skydio X2 quadrotor (MuJoCo) and a
1D hover model. Each subfolder is one experiment with its own code, notebook, README and data. Work only inside
this folder; nothing here depends on anything outside it.

Read [PLAN.md](PLAN.md) for the week's log, the key results and the proposed next steps.

## Folder map
| Folder | Question | Main files | Run |
|---|---|---|---|
| `camera_effect/` | Does a downward camera help the drone recover from a single-motor thrust loss? | `sensing.py` (camera, image processing, Kalman filter), `experiment.py`, `x2/` (simulation and PID), `assets/skydio_x2/` | `python experiment.py` (about 5 min, 140 flights) |
| `ppo_1d/` | Can the simplest PPO learn a thrust correction for a sudden total-thrust loss? | `sim1d.py` (1D physics, PID, Gymnasium env), `train.py` | `python train.py` (about 2 min, 3 seeds) |
| `wind_pid/` | PD vs PID in a steady wind, and which integral settings work in 0 / 2 / 5 N | `sim.py`, `experiment.py`, `assets/skydio_x2/` | `python experiment.py` (about 1 min, 543 flights) |

Each folder has `<name>.ipynb`, the full write-up (question, setup, run, data, analysis, results, conclusion), and
a `README.md` with the same core sections in every experiment: a summary (question, task or flights), notebook,
dependencies, resumable, code, data, result. `python experiment.py --status` shows progress in `camera_effect`
and `wind_pid`; `ppo_1d/train.py` prints its progress as it runs.

## Environment setup
*The user asked: "How do I set up the environment to run these?"* The answer, which is what these experiments
were run with:

1. **Python 3.13** in a virtual environment, on CPU (an Apple M1 is enough; no GPU is used anywhere).
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```
2. **Simulation and analysis** (all experiments):
   ```bash
   python -m pip install mujoco numpy pandas matplotlib jupyter
   ```
   Verified versions: MuJoCo 3.16.0, numpy 2.5.4, pandas 3.1.0, matplotlib 3.11.2.
3. **PPO** (`ppo_1d` only). torch was already installed. `gymnasium` and `stable-baselines3` were not; the user
   chose to install Stable-Baselines3 instead of writing PPO by hand:
   ```bash
   python -m pip install torch gymnasium==1.4.0 stable-baselines3==2.9.2
   ```
   Verified with torch 2.15.0.
4. **The drone model.** The Skydio X2 from MuJoCo Menagerie (Apache-2.0, `LICENSE` included) is bundled in each
   X2 experiment's `assets/skydio_x2/`. Nothing has to be downloaded. To use another Menagerie checkout, set
   `MENAGERIE_DIR`.
5. **Notebooks.** Open them with Jupyter or VS Code. The first cell finds the notebook's own folder, so start the
   kernel there (both tools do this by default).

## How the experiments were built
All three were designed in conversation with the user before any code was written. The user asks for clarifying
questions first, and the options and answers below are what the designs rest on.

### `camera_effect`: the camera as an extra sensor (Nov 5, 2026)
- **User:** can the camera be treated as an additional sensor, so we can test whether having it changes the
  drone's recovery?
- **Discussion:** a camera can only help a drone that doesn't already know its state. If the PID reads the
  simulator's true position and velocity, no sensor can add anything. So the test needs a realistic sensor setup.
  Three conditions were agreed:
  - `perfect`: the true state, as the best case
  - `no_camera`: IMU + GPS (5 Hz, σ 0.3 m horizontal / 0.4 m vertical) + barometer (20 Hz, σ 0.2 m), fused by a
    Kalman filter
  - `camera`: the same filter, plus camera readings
- **Which camera:** a forward camera mostly shows tilt, which the gyro already measures well. A **downward camera**
  over the checkered floor can measure height and ground velocity. The user chose the downward camera: 160 × 160 px,
  90° field of view, 20 Hz.
- **Expectation stated up front:** the IMU senses a fault within hundredths of a second, so the camera would likely
  help accuracy more than recovery. Either outcome was worth reporting.
- **Flights:** 4 random curved routes (seeds 1-4) × 5 losses on motor 1 (0 / 12.5 / 25 / 37.5 / 50 %), with the
  fault switching on halfway along the route. The two sensor conditions fly 3 paired noise seeds each: 140 flights.
  The fault is fixed and only the sensor noise is random, so any difference comes from the sensors.
- **Problems found while building, and fixed:**
  - the height formula was inverted at first
  - the Fourier peak picked a low-frequency gradient above 2 m, so bins within radius 5 are masked
  - the drone's reflection pinned the phase correlation at zero shift, so reflections and shadows are off in the
    camera render
  - the model's spotlight blacked out the floor above 2 m, so it is switched off
  - the checkerboard repeats every square, so the velocity peak is searched only near the filter's prediction
  - none of these affects the physics
- **Follow-ups from the user:**
  - "How does the drone read the camera?", now a notebook section with live demos of height and velocity
  - "What does the true state mean?"
  - renamed `camera_ablation` → `camera_effect`; all 140 flights re-flown byte-identical

### `ppo_1d`: the simplest PPO (Nov 5, 2026)
- **User:** is it possible to start PPO now, or would it take too long?
- **Options discussed:**

  | Version | Size | Estimated time |
  |---|---|---|
  | 3D residual PPO: PPO adds corrections to the PID's 4 motor commands | 29 observations, 4 actions, about 12,000 parameters, 2 M decisions | 1-2 days to a trustworthy result |
  | 1D: up/down only, so the 4 motors act as one thrust and the fault is a total-thrust drop | 5 observations, 1 action, about 1,300 policy weights | about half a day |
  | 1D cut down to fit in about an hour | hover only, no sensor noise | about 1 hour |

  The user chose the one-hour 1D version.
- **As built:**
  - X2 numbers: 1.325 kg, 24 N total cap, 0.125 s motor lag, 100 Hz
  - hover at 1.5 m; total thrust drops by a random 0-50 % at a random time between 2 and 5 s (20 % of episodes
    healthy)
  - PPO adds a correction of up to ±6 N to a PID, decides at 50 Hz, and never sees the fault
  - observations: height error, vertical speed, accelerometer, the PID's command, its own last correction
  - network: MLP [32, 32] tanh, 2,563 trained parameters (policy 1,281, log_std 1, value 1,281)
  - training: 3 seeds × 300,000 decisions, 8 parallel environments, a checkpoint every 50,000
  - test: fixed onset at 3 s, losses 0-50 % plus a 2.5 % sweep
  - compared with the gentle PID, a stiffer PID (kp 5, kd 3.5, ki 2) and an oracle that is told the loss
- **Problems found while building, and fixed:**
  - the PID's integrator limit only allowed +2 N, so it was widened to the full thrust headroom
  - the stiff PID was added so PPO isn't only compared with a weak baseline
  - the first reward (squared height error) made PPO hover 0.27 m high as a hedge, so the reward became linear:
    `1 − 2|e| − 0.1 vz² − 0.01 a²`, −50 on a crash
  - attempt 1's data was not kept and can't be recreated with the current code; the notebook skips that
    comparison
- **Follow-ups:**
  - "Do the faults come randomly?" Random in training, fixed in testing.
  - "What parameters does PPO train?" Now notebook section 2b.

### `wind_pid`: PD vs PID in wind (re-created Nov 5, 2026)
- **User:** an earlier PD vs PID wind experiment wasn't stored. Re-create it with the same mechanisms,
  self-contained, with no mention of other folders.
- **Design:**
  - PD with kp 4 s⁻² and kd 3; always-on PID with ki 2; PID that integrates only within
    0.5 m of the waypoint
  - a 1.5 m square at 1.5 m height, 4 s per waypoint; constant wind of 0 / 2 / 5 N along +x
  - error = distance to each waypoint at the end of its 4 s
  - sweep: ki (6 values) × zone (6) × limit (5) × 3 winds
- These defaults were used as set, without tuning. A crashed flight is scored as failed (inf),
  not by the waypoints it reached before crashing.

## Working rules
- **Self-contained folders.** An experiment imports only from its own folder and bundles any model files it needs.
  Code finds its folder with `Path(__file__).resolve().parent` and notebooks with their own location, never through
  a fixed path. Check this by copying the folder somewhere else, running it from scratch and comparing the data
  (`cmp` / `diff -r`). On the original machine every experiment reproduced byte-identically this way; on other
  machines the reported numbers reproduce, but the files have not been compared byte for byte.
- **Resumable runs.** The user may close the laptop mid-run.
  - Each flight is written to `<id>.tmp/` and renamed when complete.
  - A `flight.json` records the settings and a code fingerprint. A finished flight is skipped, a half-written one
    is redone, and a flight is redone if the code changed.
  - Training saves checkpoints and continues from the newest one.
  - `manifest.json` appends one entry per session with real timestamps and library versions.
- **Data stays out of git.** Everything goes in `<experiment>/data/`, which is git-ignored. Notebooks keep their
  executed outputs so they still display without the data. The README says how long regeneration takes.
- **Deterministic, paired comparisons.** Use fixed seeds. Fly every condition or controller on the same routes,
  faults and noise draws, so differences come from the thing being tested.
- **No fault truth for learned policies.** Only an explicit oracle baseline is told the fault.
- **Report only what was observed.** Every number in a notebook or README must come from an executed output.
  Re-check the text after re-running. Say so when a result is weaker than hoped, or when a run can't be recreated.
- **Dates.** A notebook header carries the date the user gives. Timestamps the code writes are always real.
- **Notebook layout:**
  - question, then what is being compared, then setup
  - a run cell (resumable), then the data, then the analysis with plots
  - results, a conclusion, limitations, and the run record from `manifest.json`
- **New experiments:**
  - ask the user clarifying questions first (scope, conditions, dates)
  - create a new subfolder following the layout above
  - give it a README with the same sections as the others
- **Commits:** no AI co-author or "Generated with" lines in commit messages or pull requests. Don't commit `data/`.
  Ask before pushing.
