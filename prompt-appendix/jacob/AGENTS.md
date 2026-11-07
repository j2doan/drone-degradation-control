# DSC180A Capstone Project

## Restrictions

You have permission to access the following directories:
- This current one.
- `/d/isaac/` (contains `IsaacLab/` and `isaacsim/`).

- The experimental repo outside of this directory is used to share code with teammates in an informal manner. Treat that repo as read-only. Never push or pull, never
  edit teammates' work there.

Approved directories will be added accordingly if more access is required.

This is a **personal workshop / testing environment**. Do not do anything GitHub-related here (no git init, commits,
pushes). The user copies `drone/` to other repos by hand.

## How to work with me

- **Think simple.** Simplify the task as much as possible and get something, anything, to work. Complexity comes later.
- **Ask before installing.** Read-only commands and running scripts are fine. Ask before installing packages, cloning repos, or launching long training runs.
- **Report faithfully.** If something fails or was skipped, say so with the output.
- **Explain for a data-science background.** The user is strong in data science, newer to robotics/control. Explain
  control and Isaac concepts in plain terms (analogies welcome) when they come up.
- The user writes their own reports and participation answers; review and correct them, do not ghost-write unless asked.
- **Navigate code with graphify, not shell text tools.** Do not use `grep` / `rg` / `cat` / `sed` / `awk` / `head` /
  `tail` / `find` (or the Grep tool) to explore the workspace: it wastes compute. Use the graph (see "Code graph" below)
  to find where things live, then the Read tool on just those files/lines (Read is still needed before editing).

## Overview and context

Drones are expensive to test. That is why IsaacSim and PegasusSimulator exist: they have enough detail to render a reasonable amount of physics and graphics, but are not as computationally expensive as modern game engines.

This capstone uses IsaacSim and IsaacLab (PegasusSimulator is not installed) to:
1. simulate drones,
2. collect sensory information (IMU, camera, GPS, pressure sensor, ...),
3. use that information to train PPO reinforcement-learning algorithms that help the drone stabilize and fly from point A to point B.

A second team works on the environment side (obstacles, LiDAR, navigation); their architecture is "residual RL" (a
classical controller's command plus an RL correction). This team does **internal degradation**.

## Goal

Apply internal degradation to a drone and use an algorithm to make the drone fly from point A to point B safely.

Example: one of the drone's fans generates too much friction and is faulty. Can the controller still work despite the faulty behavior?

## Deliverables

Speed, stability, etc. should be good. **Crashing the drone is an automatic fail.**

## Capstone schedule

1. **First quarter:** get used to IsaacSim and explore only simple models (think a random forest or a basic PPO / RL algorithm).
2. **Second quarter:** explore more complex models.

## Environment

| Item | Value |
|---|---|
| Device | Legion Pro 5 16IAX10 |
| OS | Windows 11 Pro 64-bit |
| CPU | Intel Core Ultra 7 255HX (2.40 GHz, 20-core) |
| RAM | 64 GB DDR5 (63.4 GB usable) |
| GPU 0 | Intel integrated graphics (128 MB shared) |
| GPU 1 | NVIDIA GeForce RTX 5070 Laptop (8 GB VRAM), **the one that must be used** |
| Storage | C: 1 TB, D: 2 TB |

- Shell: Git Bash (PowerShell also works with `drone/run.ps1`; `./run.sh` does NOT work in PowerShell). The standalone IsaacSim GUI is the `isaacsim` function in `.bashrc` (`cd /d/isaac/isaacsim && ./isaac-sim.bat`); the drone scripts do not need it.
- IsaacLab lives in `/d/isaac/IsaacLab`, **`develop` branch, commit `49139e1` (3.0.0)**, so many names differ from the online docs. Run with `uv run --extra isaacsim ...` from that directory (the `--extra isaacsim` installs Isaac Sim as a Python package in Isaac Lab's venv).
- IsaacSim runs on **native Windows, not WSL** (in WSL, Vulkan refused the NVIDIA GPU; WSL is unsupported). Teammates without an NVIDIA RTX GPU (e.g. Macs) cannot run Isaac Sim; they use MuJoCo.
