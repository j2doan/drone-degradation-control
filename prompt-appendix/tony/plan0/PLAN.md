# Plan: drone-control experiments

## This week's log
**Thu Nov 5, 2026**
- **Camera as a sensor: discussed.** Agreed that a camera can only matter if the controller doesn't read the true
  state. Designed three sensing conditions (perfect / IMU + GPS + barometer / the same plus a camera), and the user
  chose a downward camera over a forward one.
- **PPO: scoped down.** Compared three versions: 3D residual PPO (about 1-2 days), 1D (about half a day) and 1D
  hover-only (about 1 hour). The user chose the one-hour 1D version and to install Stable-Baselines3.
- **`camera_effect` built and run.** Wrote the camera image processing (height from the checker frequency, velocity
  from phase correlation) and a Kalman filter. Fixed five vision problems found along the way: an inverted height
  formula, a low-frequency Fourier peak, reflections, the spotlight and checkerboard aliasing. Flew 140 flights.
- **`ppo_1d` built and run.**
  - widened the PID's integrator limit (it could only add 2 N) and added a stiffer PID baseline
  - the first, squared-error reward made PPO hover 0.27 m too high, so the reward was switched to linear; attempt 1's
    data was not kept
  - trained 3 seeds × 300,000 decisions, about 30 s each
- **`wind_pid` re-created and run.** PD vs PID in a steady 0 / 2 / 5 N wind on a 1.5 m square: 543 flights (3 PD
  and a 540-flight sweep of integral gain, zone and limit), about 1 minute.
- **Answered follow-up questions:** are the faults random, how the drone reads the camera, what "true state" means,
  and which parameters PPO trains. The camera-reading walkthrough and the parameter count were added to the
  notebooks.
- **Renamed** `camera_ablation` → `camera_effect`; all 140 flights re-flew byte-identical.
- **Made all three folders self-contained.** The simulation code and the X2 model are bundled, notebooks find their own
  folder, and each README lists its own dependencies. Copies moved elsewhere and run from scratch reproduced the
  data exactly; PPO seed 0 retrained bit-identically.
- **Data kept out of git**, with notebooks keeping their outputs.

## Key results
### Camera effect (140 flights)
**The camera makes the drone more accurate and its recovery more predictable, but not stronger.**

| | no camera | camera |
|---|---|---|
| crashes | all 50 % flights, none at 0-37.5 % | the same |
| largest drop, paired (48 cases) | - | −0.007 m on average; better in 24 of 48 (a coin flip) |
| scatter of the lowest point across noise seeds | 0.073 m | **0.014 m** (5× less) |
| height estimate error | 5.0 cm | **1.0 cm** |
| horizontal estimate error | 0.19 m | **0.09 m** |
| final distance to B | 0.33 m | **0.11 m** (better in 47 of 48) |

The fall after a fault is set by the thrust left and the PID's response, and the accelerometer already reports the
sag within milliseconds. The camera removes noise-driven luck: near the crash line, one no-camera seed ended 1 cm
above it.

### 1D PPO (3 seeds, about 30 s of training each)
**Even a 1,281-weight policy learns to notice a thrust loss from the accelerometer and compensate at once.**

| Total loss | Gentle PID | Stiff PID | PID + PPO | Oracle |
|---|---|---|---|---|
| 12.5 % | drops 0.71 m | 0.27 m | **≤ 0.005 m** | 0.06 m |
| 25 % | crashes | 0.64 m | **0.02-0.04 m** | 0.12 m |
| 37.5 % | crashes | crashes | **survives** (0.82-0.86 m) | 0.18 m |
| highest survivable loss (sweep) | 17.5 % | 35 % | **40 %** | 45 % |

- PPO reacts within one decision (0.02 s) of the fault.
- At 37.5 % it is limited by its own ±6 N correction range; about +7.8 N is needed.
- Cost: it hovers 1.7-4.3 cm high when nothing is wrong.
- The three seeds agree within 0.04 m.

## Next steps (proposed)
Ordered by value for effort. Each one reuses the self-contained, resumable layout and gets its own folder,
notebook and README.

1. **Finish the 1D PPO study** (`ppo_1d`, about 1-2 hours, mostly compute-free)
   - Widen the correction range from ±6 N to about ±10 N, to see whether it closes the gap to the oracle at 37.5 %.
   - Add accelerometer noise and bias. The fast reaction relies on the accelerometer, so does it survive noise?
   - Test onset times outside the 2-5 s training range, and gradual or intermittent losses.
   - *Done when:* the table above is re-run for each change and the healthy-hover offset is reported alongside.
2. **Spinning hover with a non-learning controller** (new folder, about half a day)
   - At a fixed heading the X2 can't hover past 45.8 % single-motor loss. If it gives up heading control and spins
     slowly, the motors can run at about 3.0 / 3.5 / 3.0 / 3.5 N at 50 % loss, all within the cap.
   - Test a PID variant that drops yaw control. If it survives 50 %, a learned policy has a real target to beat.
   - *Done when:* the highest survivable loss is known for heading-hold vs spinning, on the same routes and losses.
3. **3D residual PPO on the X2** (new folder, about 1-2 days)
   - PPO adds corrections to the PID's 4 motor commands: 26 observations (as in the report's Table 1), about 12,000 parameters, about 2 M
     decisions per seed, about 10 min per run on the M1.
   - Train on random routes (seeds ≥ 1000), all four motors and losses of 0-50 %. Test on routes 1-4 × 5 losses,
     paired against the PID alone.
   - Carry over what worked in 1D: a stiff-PID baseline, a reward that doesn't reward hedging, 3 seeds, and
     checkpoints.
   - *Done when:* drop, lowest point and survival are compared with the PID on all 20 test flights, and on all 4
     motors for generalisation.
4. **Camera under harder conditions** (`camera_effect` follow-up, about half a day)
   - The camera's advantage came from an ideal checkerboard and known attitude. Next:
     - a fault on other motors and during climbs or descents
     - camera latency and noise
     - a less regular floor texture
     - better GPS, to see how fast the camera's advantage shrinks
   - Longer term: let a learned policy use the camera-filter estimate. This experiment kept the PID fixed, so it
     couldn't show whether better knowledge can become better recovery.
5. **Learned correction for wind** (builds on `wind_pid`, about half a day)
   - No fixed integral setting was best in calm, 2 N and 5 N at once: the best all-round setting fails above 5.3 N.
   - Test a residual PPO, or a wind estimate fed forward, across 0-8 N. Then combine wind with a motor fault.

**Decision point after steps 1-2:**
- If spinning survives 50 % and the 1D PPO holds up with sensor noise, go straight to step 3 and let its action
  range allow giving up heading control.
- If not, keep step 3 to losses of 37.5 % and below, and focus it on drop and near-misses.
