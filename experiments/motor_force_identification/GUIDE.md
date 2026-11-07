# Running the identification pilot

This folder includes its own source, X2 model assets, configuration and data.
Copy the whole folder and run from inside it:

```bash
python -m pip install -r requirements.txt
python run.py --help
python run.py --seeds 0 --motions steady --output data/new-check
python run.py --output data/new-matrix --plot
```

The default backend is MuJoCo.
`--view` selects a single episode; on macOS run the same command with `mjpython`.
Use `--conditions combined --motions probe --seeds 0 --view` to select one.
Omitting `--output` creates a fresh directory under `data/`.

The four conditions share one PID flight setup: healthy, motor loss, external
force and both. Motor 3 loses 20% effectiveness at 6 seconds; a -0.8 N vertical
force acts from 10 to 12 seconds. Episodes last 20 seconds. Probe motion starts
at 4 seconds with a 0.06 m, 0.7 Hz altitude sinusoid. Limits are 6 N per motor,
with 0.125 s nominal motor lag and effectiveness applied after the lag.

The estimator knows the candidate motor, mass and nominal actuator model. It
uses previous-interval command-model thrust and noisy COM vertical velocity,
never the true fault or realized damaged thrust. Motor-only and joint estimates
watch the same flight. The joint fit uses a three-second history of 0.1 s windows;
its gate indicates regressor excitation, not estimate accuracy.

Full recorded results, manifests, traces and original source snapshots are in
`data/mujoco-development-2026-11-04/`. Paths inside those saved receipts are
historical provenance, not dependencies. The model's source revision and
license are included under `assets/`. See [the summary](README.md) for results.
