# PLAN: paper on partial thrust loss, residual PPO and history

## 1. What the paper is about

A simulated quadrotor (ARL Robot 1, Isaac Sim / Isaac Lab, plus a MuJoCo backend) flies a seeded random curved path from
A to B. Partway through, one randomly chosen motor suddenly loses 10 to 50% of its thrust. Four controllers fly the same
flights. A crash is an automatic fail.

**Research questions**
1. Under what fault conditions (size, motor, onset time, path) does a residual PPO correction beat the PID alone?
   "Beat" = crash rate first, then final distance to B, then recovery time.
2. Does giving the residual policy a window of past states improve on the same policy with only the current state, and when?
3. How much history is enough before performance levels off? **[OPEN]** this needs one policy per window length
   (for example 0, 2, 5, 10, 25, 50 steps at 100 Hz), a large experiment. Confirm it or soften it to "does a history window help?"
4. Side comparison: can PPO alone (no PID) fly under these faults? If it cannot hover on the healthy drone within its
   training budget, report that as a result and stop.

**Drones** (older notes call them arms A to D)

| Drone | Arm | Controller | Status |
|---|---|---|---|
| D0 | A | PID | **[DONE]**, flown in `data/run1` |
| D1 | C | PID + residual PPO (up to +/-1 N per motor, then capped 0 to 6 N) | slot ready, no policy |
| D2 | D | D1 + last k observations + a flag for which history slots are real | **[NOT BUILT]** |
| D3 | B | PPO alone, outputs the 4 thrusts | slot ready, no policy |

## 2. Positioning (what we can and cannot claim)

**Cannot claim**
- "Nobody uses history." Kim et al. already condition an RL policy on sensor/command history over a PID.
- "PPO is what is new." Chen et al. also use PPO; Sharma et al. use SAC, which is only a weak contrast.
- "Little work uses onboard-only data." Several of the cited papers do.
- Online or continual learning. Our policies are trained offline and fixed at test time.
- A low compute cost, until it is measured.

**Can claim (defensible)**
- A *controlled comparison* on a fixed seeded test set: PID alone, residual, residual with history, and PPO alone, under
  *sudden partial* thrust loss in *one motor* during *path following*, scored by crash rate by fault size.
- A history-length study for the residual policy (if question 3 is kept).
- Onboard data only: the policy sees state and noisy IMU and its own last thrusts, never the fault, the camera or LiDAR. The camera is
  logged for future work (and this separates us from the other sub-team).
- Motivating lead from our data: in `data/run1` (11 flights) the PID recovered from every loss up to 30% and crashed from 42%
  up; three of those four crashes were physically survivable. Caveat: loss, motor, onset and path varied together, so this needs a
  controlled sweep before it is stated as a threshold.

**Limitations to state**: velocity and orientation are simulator ground truth; simulation only; one drone and one fault type; a
hand-tuned PID baseline; `run1` has 11 flights. The vision-based work (Sun et al.) solves state estimation, which we sidestep, so
the two are complementary.

## 3. Experimental protocol

- **Faults:** one motor, 10 to 50% loss, once per flight, onset between 3 s and 1 s before reaching B (never during the hover). Applied
  to the motor *output* after Isaac's motor model. A feasibility flag records whether the damaged drone could hover at all.
- **Seeds:** a seed fixes path, fault and IMU noise. Seed 0 is healthy. All drones fly identical flights for the same seeds.
- **Policy observation (26 values):** target position relative to the drone (3), target velocity relative to the drone (3), velocity (3),
  orientation (4), angular velocity (3), noisy IMU accelerometer (3) and gyro (3), last thrust command per motor (4).
  **Never** the fault columns, `thrust_applied_*` (with `thrust_cmd_*` it reveals the fault), the clean IMU, or the camera.
- **Metrics:** `crashed` (height below 0.30 m after 0.5 s, or tilt above 60 degrees), `final_dist_to_B` (blank if crashed),
  `time_to_arrive` (within 0.10 m), `max_alt_loss_after_fault`, `recovery_time` (within 0.05 m of target altitude for 0.5 s),
  deviation and sideways drift, `path_length`, `smoothness`. Aggregate: crash rate by fault size, same test seeds. Training reward is
  not an evaluation metric.
- **Ablation:** variants with one input group removed (no IMU, no last thrust, no history), same fixed test seeds, 3 to 5 training seeds
  per variant, mean with a confidence interval. AIC/BIC does not apply to PPO policies.
- **Logging rate:** 100 Hz (0.01 s physics step). Camera 10 Hz only with `--camera`.

## 4. Paper outline and status

| Section | Content | Status |
|---|---|---|
| Abstract | Motivation, gap, setup, four controllers, metrics, ablations, results placeholder | **[DRAFTED]** (`RESEARCH_OVERVIEW.md` section 11); results missing |
| Introduction / related work | Complete-failure work, vision-based estimation, SAC, PPO, history adaptation | **[DRAFTED]**; fix two phrasings (section 6) |
| Data collection | Seeded batches, fault, logged columns, closed loop | **[DRAFTED]** and corrected; reword "will observe" until D1 is trained |
| Observation table (26 values) | LaTeX `tabular`, no external packages | **[DONE]**; floatrow error unresolved (section 6) |
| Methods | The four drones, PID, residual wrapper, history window, training protocol | partly: D0 done, D1 to D3 **[NOT BUILT]** |
| Evaluation | Metrics, same-seed protocol, ablations | **[DRAFTED]** in `RESEARCH_OVERVIEW.md` |
| Results | Crash rate by fault size, recovery, history effect | **[NOT BUILT]** needs the runs below |
| Limitations / future work | Ground-truth state, camera, multi-motor faults | **[DRAFTED]** (bullet list only) |
| References | `reference.bib`, five entries | **[DONE]**; add more as sections grow |

## 5. Work plan

**Before writing results**
1. Controlled D0 sweep: fixed path and onset, vary loss and motor, many seeds. Report crash rate with a confidence interval.
   This pins the 30 to 42% band instead of leaving it as a lead from 11 flights.
2. Run it with the existing `03_collect_data.py --fault fixed` flags (motor, loss, onset) or a small loop around them; keep one results table format.
3. Residual wrapper (D1): PID thrust + bounded policy output, then cap. D0 is D1 with the correction set to zero.
4. Observation builder per drone; check against `drones.py` that it matches the 26-value spec.
5. Training protocol: mix of severities plus no-fault episodes, fault at a random time. Resolve the policy control rate.
6. Smoke-test `rl/train.py` for a couple of iterations to measure training speed before committing to a sweep.
7. D3 time-boxed. D2 and the window sweep only after D1 works.

**Writing**
- Keep the abstract's results sentence empty until numbers exist.
- Update `EXPERIMENT_DESIGN.md` so its "open decisions" match what the code now does (random motor, random 10 to 50% loss, random onset,
  drone naming D0 to D3) and record whatever is decided on seeds.

## 6. Open decisions and known issues

**Open decisions**
1. Commit to the history-length sweep, or soften question 3?
2. Test seeds per fault size, and training seeds per variant.
3. Policy control rate: every physics step (100 Hz) or slower (Isaac Lab's env step is 0.1 s)? State whichever is chosen.
4. Freeze metric tolerances.
5. Keep an oracle drone (told the fault) as an upper bound?
6. Pick one naming scheme (D0 to D3 or arms A to D).

**Known issues**
- LaTeX: the observation table triggers "not in outer par mode" then a floatrow error. Cause not yet identified; most likely a nested
  `table` inside a template wrapper or box. Fix by removing the inner `table` or moving it out of the box.
- Intro sentence 1: Mueller and D'Andrea support giving up yaw and spinning, but not the IMU-limit clause. Chen et al. attribute that
  to Ke et al. (T-RO 2023); cite that or Chen et al. for it.
- Intro sentence on Chen et al.: say the policy "sits beneath a PID-based controller" rather than "runs on" one.
- Sun et al. first name (Sihao) is from memory; the paper's reference list shows only the initial.
- The control-rate and "will observe" wording depend on D1 and D2 existing.

## 7. References: what has been checked

- **Read (full or abstract plus summary):** Chen et al. (2503.02649), Kim et al. (2505.08223), Sharma et al. (2109.10488),
  Sun et al. (RA-L 2021), Mueller and D'Andrea (ICRA 2014; listing checked).
- **Seen in search or in others' reference lists only:** the meta-RL ICAS 2022 paper, arXiv 2509.07707, Ke et al. T-RO 2023,
  Nan et al. RA-L 2022, Liu et al. T-II 2024, Jiang et al. 2023, Zhang et al. ICRA 2023.
- **From memory, [VERIFY] before citing:** Silver et al. 2018, Johannink et al. 2019, Neural-Fly (O'Connell et al. 2022),
  Mnih et al. 2015, Air Learning author list.
- `reference.bib` keys: `mueller2014`, `sun2021`, `sharma2021`, `chen2025`, `kim2025`.
- Repos: `HITSZcjh/uav_ftc` (promised in Chen et al.; confirm it exists), `uzh-rpg/fault_tolerant_control` (not opened), OmniDrones,
  `isaac_drone_racer`, AirGym (not inspected).
