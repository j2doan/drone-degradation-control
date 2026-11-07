## PD vs PID in a steady wind

- **Question:** in a steady wind, does a PD controller hold its waypoints? What does the integral term (PID) fix and
  what does it cost? Is any integral setting good in 0, 2 and 5 N of wind at once?
- **Flights:** a Skydio X2 flies a 1.5 m square of waypoints at 1.5 m height, 4 s per waypoint, in a constant wind
  of 0, 2 or 5 N along +x. The waypoint error is the distance at the end of each 4 s, averaged over the 4 waypoints.
  - PD (kp 4, kd 3): 3 flights
  - sweep of the integral: ki (6 values) × integrate only within a distance of the waypoint (6) × limit (5), at
    each wind: 540 flights. The always-on PID (ki 2) and the near-target PID (ki 2, within 0.5 m) are part of it.

  543 flights in all, about 1 minute.
- **Notebook:** `wind_pid.ipynb` covers the question, the setup, the run, the data, the
  analysis (PD, PID, near-target PID, the sweep, a 0-8 N wind check), the results and
  the conclusion.

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
  Finished flights are skipped, a half-written flight is flown again, and a flight is flown again if `sim.py` or
  `experiment.py` changed.
- **Code:**
  - `sim.py`: model loading, the position controller with optional integral (PD / PID / near-target PID), wind, one
    flight
  - `experiment.py`: the flight list, the resumable runner and `summary.csv`. Run `python experiment.py` from this
    folder (or give its full path from anywhere); add `--status` to see progress, `--force` to fly everything again.
  - `assets/skydio_x2/`: the bundled X2 model
- **Data:** `data/`, about 23 MB.
  - `summary.csv` (one row per flight: settings, error at each waypoint, mean, crashed) and `manifest.json`
  - `flights/<id>/`: `flight.csv` (trajectory at 20 Hz, including the integral term's push) and `flight.json`
  - **Not stored in git**; it stays on the machine that ran the experiment. The notebook keeps its saved outputs, so
    it still displays without the data. Running `experiment.py` recreates it in about 1 minute.
- **Result:**
  - PD: 0.5 cm in calm air, but a steady offset in wind exactly as `F / (m kp)` predicts: 37.8 cm at 2 N, 94.6 cm
    at 5 N.
  - Always-on PID (ki 2): the offset is gone (14.6 cm at 2 N, 15.4 cm at 5 N), but calm air rises to 13.6 cm
    from overshoot, because the integral builds during each move.
  - Integrating only within 0.5 m: calm 3.1 cm and 2 N 6.2 cm, but at 5 N the drone never gets within 0.5 m, so
    the integral never builds (94.3 cm).
  - Best all-round setting (ki 4, within 1.0 m, limit 4): 3.2 / 4.5 / 3.8 cm at 0 / 2 / 5 N. It only works because
    1.0 m just covers the 0.94 m offset at 5 N; at 6 N it jumps to 113 cm. No setting is best in every wind.
