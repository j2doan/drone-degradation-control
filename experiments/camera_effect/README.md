## Does a downward camera improve recovery from a motor fault?
- **Flights:** 4 routes × 5 losses on motor 1, flown under three sensing conditions:
  - `perfect`: the PID reads the true state
  - `no_camera`: Kalman estimate from IMU + GPS + barometer
  - `camera`: the same, plus a downward camera giving height and ground velocity

  Both sensor conditions fly 3 paired noise seeds, so 140 flights in all, about 5 minutes.
- **Notebook:** `camera_effect.ipynb` covers the question, the camera sensor and its accuracy check, the run, the
  data, the analysis, the results and the conclusion.

- **Dependencies:**
  - **Python packages:** `mujoco`, `numpy`, `pandas`, `matplotlib`, plus `jupyter`/`ipykernel` for the notebook.
    Verified with Python 3.13, MuJoCo 3.16.0, numpy 2.5.4, pandas 3.1.0, matplotlib 3.11.2, macOS on CPU. No GPU
    needed. In a fresh environment:

    ```bash
    python -m pip install mujoco numpy pandas matplotlib jupyter
    ```
  - **Model files:** the Skydio X2 from [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) is
    bundled in `assets/skydio_x2/` (2.3 MB, Apache-2.0; its `LICENSE` is included). To use another Menagerie
    checkout, set `MENAGERIE_DIR` to its folder.
- **Resumable:** if the computer shuts down mid-run, run the same command (or the notebook's run cell) again.
  Finished flights are skipped and a half-written flight is flown again.
- **Code:**
  - `sensing.py`: downward camera, image processing, Kalman filter
  - `experiment.py`: flights and resumable runner. Run `python experiment.py` from this folder (or give its full
    path from anywhere); add `--status` to see progress.
  - `x2/`: the simulation code
  - `assets/skydio_x2/`: the bundled X2 model
- **Data:** `data/`, about 90 MB.
  - `summary.csv` and `manifest.json`
  - one folder per flight with `flight.csv`, `flight.json`, and `frames.npz` (camera seed 0 only)
  - **Not stored in git**; it stays on the machine that ran the experiment. The
    notebook keeps its saved outputs, so it still displays without the data. Running `experiment.py` recreates it
    bit-identically in about 5 minutes.
- **Result:**
  - No change in crashes or in the average drop or recovery.
  - About 5× less run-to-run scatter in the lowest point, close to the perfect-state result.
  - Clearly better accuracy: final distance to B 0.33 → 0.11 m, height estimate 5.0 → 1.0 cm.
