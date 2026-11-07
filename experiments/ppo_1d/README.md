## Simplest PPO: 1D thrust correction
- **Task:** a 1D (up/down only) drone with X2 numbers hovers at 1.5 m, and total thrust drops by 0-50 % at a random
  time. PPO learns a correction of up to ±6 N added to a PID, from 5 inputs. It's compared with the PID, a stiffer
  PID and an oracle. 3 training seeds × 300,000 decisions, about 30 s each.
- **Notebook:** `ppo_1d.ipynb` covers the question, the setup, training, learning curves, the test, plots, the
  results and the conclusion.

- **Dependencies:**
  - **Python packages:** `numpy`, `pandas`, `matplotlib`, `torch`, `gymnasium` and `stable-baselines3`, plus
    `jupyter`/`ipykernel` for the notebook. Python 3.13, numpy 2.5.4, torch 2.15.0,
    gymnasium 1.4.0, stable-baselines3 2.9.2, macOS on CPU. No GPU needed. In a fresh environment:

    ```bash
    python -m pip install numpy pandas matplotlib torch gymnasium==1.4.0 stable-baselines3==2.9.2 jupyter
    ```
- **Resumable:** training saves a checkpoint every 50,000 decisions. If the computer shuts down, run `train.py` (or
  the notebook's training cell) again: it continues from the newest checkpoint and skips finished seeds.
- **Code:**
  - `sim1d.py`: physics, controllers, Gymnasium environment
  - `train.py`: training with checkpoints, then evaluation. Run `python train.py` from this folder (or give its full
    path from anywhere); `data/` is always written next to it.
- **Data:** `data/`
  - `seed*/`: models, checkpoints, training monitor logs
  - `eval/`: per-episode CSVs and `summary.csv`
  - `attempt1_squared_reward/`: the first reward version, kept for comparison. It **cannot** be recreated by the
    current code, since the reward has changed, so keep a local backup if you need it.
  - `manifest.json`
  - **Not stored in git**; it stays on the machine that ran the experiment. The
    notebook keeps its saved outputs, so it still displays without the data. Without it, running `train.py`
    retrains and re-evaluates in about 2 minutes. Retraining seed 0 from scratch in a copy of this folder gave
    bit-identical results on the same machine and library versions; another machine or other versions may give
    slightly different policies. `attempt1_squared_reward/` (above) can't be recreated at all.
- **Result:**
  - PPO reacts within 0.02 s of the fault, using the accelerometer.
  - Drop at 25 % loss: 0.02-0.04 m, against 0.64 m for the stiff PID (the gentle PID crashes).
  - Survives up to 40 % loss, against 35 % for the stiff PID and 45 % for the oracle.
  - Hovers 2-4 cm above the target when healthy.
