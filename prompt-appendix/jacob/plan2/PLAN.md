# Week 5 Goals

## Summary
Week 5 finishes the D0 pipeline so that only the D1-D3 policies are missing:
- random curved paths instead of a straight line
- more sensors: IMU noise, and a camera on advisor advice
- organized batch outputs (one folder per run, seed and drone)
- MuJoCo compatibility
- a review of teammates' work

---

## Prompt

Continue building the workflow.

### Goals
1. **Implement randomized paths from A to B.**
   - One critique of our methodology was that the drone path is a straight line. What if the drone memorizes this
     line? We need more curved / randomized paths in all 6 degrees / directions.
2. **Implement more sensors.**
   - Currently, we collect IMU sensory data (with no noise).
   - We could add GPS or a barometer for extra information.
   - But one thing we forgot is the RGB camera sensor.
     - My advisors highly recommended IMU and camera as two kinds of sensory data that we can collect and use for RL
       (when we implement it).
3. **Reorganize the output.**
   - Currently, we just put 3 files into the data folder.
   - This is unorganized, and when we run many batches, it turns into an unorganized nightmare.
   - Every command should create an output folder named after the run name, e.g. `run1`.
   - I should also be able to set the number of runs in the batch, let's say 5.
   - So now (assuming D0 (baseline), D1, D2, and D3 are set up), it should look like this:

```
run1/
  seed1/
    d0/   csv, png, json
    d1/   csv, png, json
    d2/   ...
    d3/   ...
  seed2/
    d0/   ...
    ...
  seed5/
    d0/   ...
    ...
```

This week, I want to set everything up so that all we need to do on our end is investigate GitHub repos to implement
the PPOs (or, if that goes poorly, try building a PPO from scratch, but we will cross that bridge another time).

Basically, I want D0 and the pipeline COMPLETELY set up, or at least 99% working, ready for D1, D2, and D3.

### Questions
1. I am not sure about the reward, if that is a part of D1, D2, and D3. If so, then we can pause that for now.
2. We need to make sure that this is MuJoCo-accessible too (teammates still need to test this).
3. Am I forgetting anything else so far?
4. If I grant you access to another repo outside of this directory, would you be able to review my teammates' work?
   - For context, there are 3 directories:
     - a. This `isaacsim-test` directory: my own workshop, with my personal notes and experiments.
     - b. The team's shared scratch repository, where teammates send each other collaborative or AI-generated code.
       It may not work, but we send our work in separate folders, and we can pull each other's work and experiment
       with it. Every week, I copy the `drone` directory from here into that repo by hand. I do not work in that repo
       directly.
     - c. `drone-degradation-control`, the serious capstone repo. Nothing gets pushed there except fully completed
       checkpoints or finished products.
   - We have several layers of protection and keep our commit history clean. We could use branches, but that would
     make an unclean commit history.
   - If I give you access to the shared repository, can you review my teammates' code? I don't think they understand
     what they wrote, but I need to know so I can integrate their work if possible, and get this checkpoint ready.

My goal is to finish this pipeline (aside from D1, D2, D3) and push it to the serious repo.

---

## Notes added after the prompt
*(Not part of the prompt: a summary added later of design ideas from a teammate's 1-D practice environment. These
are ideas for a later week, not code to copy.)*

- Limit the size of PPO's correction, so it can nudge the PID but never wreck it.
- Never show the policy the true fault. It has to work it out from how the drone behaves, like a real drone.
- Mark which memory slots are real. At the start of a flight the memory window is mostly empty, so a flag says which
  entries hold actual readings.
- Reward: a small penalty every step for being off course, plus a big penalty for crashing.
- Stop windup when the motors max out: if PID plus correction hits the motor limit, the PID's integral stops building
  up.
