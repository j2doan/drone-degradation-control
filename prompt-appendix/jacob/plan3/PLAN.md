# Week 6 Goals

## Summary
After week 5 the D0 pipeline was complete in Isaac Sim. Week 6 made it run in MuJoCo as well (for teammates without
an NVIDIA GPU), then polished and verified everything for the checkpoint: comments, READMEs, lint, milestone versions
that each run on their own, and a sample baseline run kept outside `drone/`.

## Where we were
- D0 (PID baseline) works in Isaac Sim: random curved paths, seeded thrust-loss faults, IMU with noise, optional
  camera, batch outputs with a receipt, and a D1-D3 slot.
- A teammate had prototyped a MuJoCo version of the pipeline in our shared scratch repository, but it imported
  `drone/` through a hard-coded path to another folder and flew a different drone.

## Prompts and decisions (in order)

### 1. MuJoCo support
> "It seems my teammate helped make my code MuJoCo accessible. Can you verify this, and compare it to the `drone`
> folder here?"

> "I do not like their dependency on a hard-coded path to my drone, but I do want to adopt their implementation in
> the sense that it apparently works. I want you to work on the `drone` directory here. Do not edit any of their
> work."

> "Make sure that you ONLY edit inside `drone`. Don't use venvs from outside it; if you need a venv, create it in
> `drone`. Remember, `drone` is the repo."

Decision: build MuJoCo support inside `drone/` itself (`--sim isaac|mujoco` on the same scripts), with a MuJoCo copy
of the same ARL Robot 1 drone, instead of depending on the teammate's code. A teammate then ran it on their machine:
"my teammate just ran it. it works!"

### 2. Clean comments and README, re-run the baseline
> "Make sure that in this `drone`, the README.md is updated, and all comments are good: no AI / agent / plan
> references; clean, independent, and polished. I understand the seeds might get ruined; you can rerun `run1` if
> needed. Create this latest `drone` folder as another backup. Then update AGENTS.md with the comment styling we have
> discussed so far."

### 3. Verify every milestone version; no data inside `drone/`; lint
> "Double check all of the backups again. Make sure that each backup WORKS. If a backup has a data folder, empty it
> and add a .gitkeep. I do not think we should add the sample data to any `drone`; instead, add a sample output
> outside of drone (the repository will have `drone/`, `paper/`, `experiments/`, `sample_run/`, `README.md`). When
> checking that each backup works, do NOT depend on the backup folder name: they will all be committed as `drone`.
> For the comments: no odd spacing or aligned columns of spaces. Finally, the code should pass lint and ruff checks,
> since the serious repo has those. Do not install anything into this repo."

Decisions (from the AI's questions): inline comments follow ruff (2 spaces), line length 88, the never-run PPO
templates are removed from the early versions, and `sample_run/` holds the final Isaac and MuJoCo baseline batches.

### 4. README consistency and the sample baseline run
> "Make sure that the latest backup is identical to the current `drone`. Update and clean up the README.md for every
> backup; the title should be 'Drone Degradation Control Adaptation through Proximal Policy Optimization'. The D0
> baseline section is sorted by fault size instead of by seed; rename it 'Sample baseline run' and show how to READ
> the sample. Include TWO images, one healthy run and one crashed run. Since we will keep editing, say that this sample
> may not be reproducible anymore, and explain the hash and constant-edit issue."

Follow-ups: turn "About the PID baseline" into an FAQ; make every version's README follow the same format; add a
third example image (a recovery).

### 5. Final check
> "Just to make sure one more time: all the backups can run independently on their own, and successfully?"

> "Organize `drone-backups` so that each backup is named `drone`, with the backup name as its parent folder, so I can
> be sure all of it runs independently as `drone`."

## Outcome (for reference)
- `--sim mujoco` works on all three scripts and agrees with Isaac on all 11 seeds of the sample run (same 4 crashes).
- MuJoCo runs repeat bit for bit.
- Five milestone versions, each in `<version>/drone/`, pass ruff and their own self-tests.
- Each version flies in Isaac from its own folder.
- `sample_run/` (Isaac and MuJoCo batches) sits next to the latest version, and its code hashes match that version.
