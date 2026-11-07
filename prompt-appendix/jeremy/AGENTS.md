# AGENTS.md: instructions for AI assistants working on this paper

Read `PLAN.md` first (what we are doing and what is decided), then `RESEARCH_OVERVIEW.md` (teammate summary), then the
`drone/` docs (`README.md`, `EXPERIMENT_DESIGN.md`) for how the code runs. If these disagree with each other, say so and
ask which is current. Do not silently pick one.

## 1. The project in four lines

- Simulated quadrotor (ARL Robot 1, Isaac Sim / Isaac Lab; MuJoCo backend also exists) flies a seeded curved path from A to B.
- One random motor suddenly loses 10 to 50% of its thrust once per flight, at a random time while travelling. A crash is an automatic fail.
- Four drones on identical flights: D0 PID, D1 PID + residual PPO, D2 D1 + history window, D3 PPO alone.
- Questions: when does residual PPO beat the PID, does a history window help, and how much history is enough.

## 2. Ground rules

**Claims**
- Do not claim something works unless it was run. Mark unrun code and unrun experiments as such.
- Never invent results, numbers, citations, DOIs, author names or page ranges. If a detail is unknown, leave a visible placeholder
  such as `[RESULT]` or `[VERIFY]`.
- Do not write "nobody has tested history", "PPO is our novelty", "continual/online learning", or "low compute cost". See `PLAN.md`
  section 2 for the claims we can and cannot make. Kim et al. (arXiv 2505.08223) already use history over a PID.
- Use hedges that match the evidence: `run1` has 11 flights where loss, motor, onset and path vary together, so call it a lead, not a threshold.
- Say what the policies see: onboard state, noisy IMU and their own last thrust commands. Not the fault, not the camera, not LiDAR.
- State the limitation that velocity and orientation come from simulator ground truth.

**Experiment integrity**
- All four drones use the same scenario, seeds, fault timing and metrics. Change the benchmark spec only by team agreement and record it
  in `EXPERIMENT_DESIGN.md`.
- Never feed the fault columns (`fault_active`, `fault_motor`, `fault_loss`) or `thrust_applied_*` to a policy. Together with
  `thrust_cmd_*`, `thrust_applied_*` reveals the fault and turns the policy into an oracle.
- The camera is logged only. Do not add it to any observation without a team decision.
- The training reward is not an evaluation metric. Compare drones on the same fixed test seeds.
- Crash rate is the first ranking criterion. Final distance to B is blank for a crashed flight.

**Working with the code**
- Ask before installing packages or starting long training runs.
- Keep `drone/` free of machine-specific paths; use a named constant at the top of the file if unavoidable.
- Do not commit generated data (`data/*` except the shipped `data/run1/`, and `rl/runs/`).
- `launch_simulation()` exits the whole Python process when its `with` block ends. Do all saving and plotting inside the block.
- Most Isaac Lab modules can only be imported after the simulator starts, so imports go inside functions.
- Keep scripts small and comment the physics and sensor choices that are not obvious.
- A flight must be reproducible: same seed gives the same flight (GPU physics may differ in the last digits).

## 3. Facts to get right

| Item | Value |
|---|---|
| Logging rate | 100 Hz (0.01 s physics step); camera 10 Hz only with `--camera` |
| Policy rate | **Undecided.** Do not state one. Isaac Lab's env step is 0.1 s; the log is 100 Hz |
| Policy observation | 26 values: target position relative (3), target velocity relative (3), velocity (3), orientation (4), angular velocity (3), noisy IMU accel (3) and gyro (3), last thrust command (4) |
| D1/D2 output | correction up to +/-1 N per motor added to the PID thrust, then capped to 0 to 6 N |
| D3 output | the 4 motor thrusts directly |
| Fault | one motor, 10 to 50% loss, applied to motor output after the motor model; feasibility flag records whether hover was physically possible |
| Crash rule | height below 0.30 m after the first 0.5 s, or tilt above 60 degrees |
| Recovery | within 0.05 m of target altitude and stays 0.5 s |
| Arrival | within 0.10 m of B |
| Drone naming | D0 to D3 (older notes: arms A to D, with arm B = D3 and arm C = D1) |
| IMU | proper acceleration (about +9.81 m/s^2 on z at hover); first row reads 0 |
| Healthy baseline | `data/run1`: D0 recovered up to 30% loss, crashed from 42% (n = 11) |

## 4. Writing the paper

**Format**
- The paper is in LaTeX. Bibliography lives in `reference.bib`; keys are `mueller2014`, `sun2021`, `sharma2021`, `chen2025`, `kim2025`.
- Reference list style: numbered, author-year entries (Last, First, ... Year. "Title." Venue volume(issue): pages. [Link]),
  as in the course template. Always include the reference number.
- Tables: use only `\toprule`, `\midrule`, `\bottomrule` (booktabs-style rules) and `tabular`. No other external packages.
  If the template raises "not in outer par mode" or a floatrow error, the cause is usually a `table` nested inside another float or box.
  Ask for the exact error and the surrounding lines before changing the template.
- Summaries for teammates go in Markdown. Avoid em dashes and filler.

**Citations**
- Verified against the source or its listing: Chen et al. 2025, Kim et al. 2025, Sharma et al. 2021, Sun et al. 2021 (details from the reference
  list of Chen et al.), Mueller and D'Andrea 2014.
- Treat as unverified until checked: Silver et al. 2018, Johannink et al. 2019, Neural-Fly (O'Connell et al. 2022), Mnih et al. 2015,
  the Air Learning author list, the meta-RL ICAS 2022 paper, arXiv 2509.07707, and anything marked `[VERIFY]`.
- Attribute claims to the paper that actually makes them. Example: the IMU spin-rate limit comes from Ke et al. (T-RO 2023) via
  Chen et al., not from Mueller and D'Andrea.
- Quotes: keep to under 15 words and one per source; paraphrase otherwise.

**Positioning language**
- Good: "a controlled comparison of PID, residual PPO, residual PPO with history, and PPO alone under sudden partial thrust loss in
  one motor, on a fixed seeded test set."
- Good: "complementary to vision-based state estimation (Sun et al.), which we sidestep by using simulator state."
- Avoid: superlatives, "first", "novel" without a search that supports it.

## 5. When to stop and ask

- A metric tolerance, crash rule, severity range, seed count, or the history-length sweep would change. These are team decisions.
- A result contradicts `run1` or the README.
- You are about to use a source you could not open or verify.
- A request would feed privileged information (fault, applied thrust, clean IMU, camera) to a policy.
- Training is expected to take long, or a package must be installed.

## 6. Glossary

- **Residual RL:** the policy outputs a correction added to a classical controller's command instead of the full command.
- **History window:** the last k observations given to the policy, plus a flag marking which slots hold real readings.
- **Oracle:** a controller told the true fault. An upper bound, not in the current design (see open decision 5 in `PLAN.md`).
- **Seed:** fixes one flight (path, fault, IMU noise). Seed 0 is always healthy.
- **Feasibility:** whether the damaged drone could physically hover with the motor limits. A crash in an infeasible case is not the controller's fault.
- **Headless:** run the simulator without a window.
