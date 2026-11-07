# Running this experiment

Copy this whole folder, then run commands from inside it. Its implementation is
in `src/drone_adaptation/`; it does not import code from another experiment.
The numerical comparison uses the Python standard library. Install plotting
packages only if figures are needed:

```bash
python -m pip install -r requirements.txt
python run.py --help
python run.py --output data/new-run
python plot.py --input data/new-run
```

Omit `--output` to create a fresh timestamped folder under `data/`. Existing
outputs are refused unless the underlying baseline explicitly supports and is
given an overwrite flag. Saved historical reports and source hashes describe
the original code and are left unchanged.

The notebook is `notebook.ipynb`. It uses the local `src` and `data/starter`
files. It additionally needs `ipykernel`, `nbformat` and `nbclient` if executed
programmatically.

See [the summary](README.md) for the study status and interpretation of results.
