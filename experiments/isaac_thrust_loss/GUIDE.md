# Running the Isaac experiment

Keep this entire folder together. Its flight code, controller, fault helpers
and saved examples are local. A supported NVIDIA machine with Isaac Lab and
Isaac Sim is still required; those are installed software dependencies.

Set `ISAACLAB_DIR` to that installation, or edit the installation setting in
`run.sh` / `run.ps1`. The launchers change into the SDK directory while running
this folder's scripts, so use absolute output paths where an output is requested.

From this experiment folder on Linux/Git Bash:

```bash
./run.sh 03_collect_data.py --name pid_healthy
./run.sh 03_collect_data.py --fault fixed --fault_motor 2 --fault_loss 0.3 --fault_onset 4 --name new_fault
```

On Windows PowerShell, use `run.ps1` with the same arguments. For local
configuration and controller checks, install the packages in `requirements.txt`
in a virtual environment. These checks are not GPU flight validation.

`python 04_separation_pilot.py --check-config` validates the separate diagnostic
without launching Isaac. Historical PID examples are in `data/`. New collector
names must be unique because this older collector can overwrite matching names.

See [README](README.md) for the experiment's actual results and limitations.
