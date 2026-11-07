# Week 4 Goals

## Summary
Week 4 turns the week-3 pipeline into the D0 baseline:
- replace Isaac Lab's Lee controller with a PID
- add a fault ("internal degradation") folder with a command-line flag
- log real sensor data
- redesign the output plots

The D1-D3 PPO drones are planned but not built this week.

---

## Prompt

### Currently we have established the following
1. A basic pipeline that sets up test runs and collects "sensory info".
2. A long-term plan / workflow for the first quarter.

I talked with my PI and he approved our plan. So this week we will expand the workflow.

### Basic recap
1. We will be training a PPO residual on top of a controller, not completely replacing it (that would be extremely
   hard and long to train to get deliverable results).
2. The controller and degradation should remain consistent across all 4 models, unless we find something else (see
   4c).
3. Our original plan for sudden thrust loss was 3 set intervals. However, I was advised to train the models with
   randomized thrust losses, so the model doesn't learn the pattern of readjusting at a certain interval from A to B.
   The thrust loss % is also randomized, so the timing and intensity are both drawn from a random distribution.
   - 3a. It is unknown, however, whether we should design our test experiment with randomized losses or set-interval
     losses. If so, then all drones should use the same seed.
4. Here is our plan:

| Drone | Controller | Degradation | PPO |
|-------|------------|-------------|-----|
| D0 (baseline) | PID | randomized sudden thrust loss | NONE |
| D1 | PID | randomized sudden thrust loss | plain PPO |
| D2 | PID | randomized sudden thrust loss | residual PPO |
| D3 | PID | randomized sudden thrust loss | residual PPO + history |

   - 4b. Right now, we do not need to worry about training D1, D2, or D3. In fact, we may decide against those PPOs.
     After all, we need to find some existing PPOs on GitHub repos to download and then apply. HOWEVER, that is for
     another week.
   - 4c. However, if we do manage to find a PPO without a controller, we can use that too. For example:

| Drone | Controller | Degradation | PPO |
|-------|------------|-------------|-----|
| D4 | None | randomized sudden thrust loss | PPO? |

5. EVALUATION METRICS: path distance, reward, smoothness, failure (did it crash, T/F?), time. We have decided to
   measure multiple evaluation metrics.
6. Currently, our collected data at each time step does NOT capture real sensory data, only position, angular
   velocity, etc. We need to find a way to measure actual sensory info, so our models can train on that.

### Goals for this week
1. Replace the controller with PID instead of Lee.
2. Add a folder that controls the fault / "internal degradation", similar to how the rl folder works. I should be
   able to pass a flag to apply it on runs from the bash command line.
3. Make the drone collect actual sensory data. I am not sure how this would work. Getting the current information is
   a good start; however, we need to move on.
4. Revamp the output images. I ONLY want to see the following plots:
   - a plot showing time vs. distance from the drone's position to B
   - a plot showing distance from the current drone position to B vs. Euclidean distance from the drone's position
     (x, y, z) to the expected route. Vertical lines should also be shown when a thrust degradation is applied.

   If you have any other plot recommendations, please let me know.
5. Focus on building ONLY the baseline model. To be fair, it is basically done. It is similar to what we have right
   now, except the Lee controller is replaced with PID, and it needs to be simulated with faults.

**Ultimately, please review this week's current plans. Ask me questions before I give approval for implementing. If
anything seems unclear, contradictory, or flawed in my experimental design, please let me know. You may also search
GitHub for appropriate repos to download; ask me first beforehand.**

---

## The AI's questions before approval
*(Written by the AI in response to the prompt above.)*

- **D1:** what does "PID + plain PPO" mean, and how does it differ from D2? Should D1 be the no-controller drone
  instead of D4?
- **GitHub:** did you mean PPO code (you already have it) or pretrained models? Do you still want me to search?
- **Fault distribution:** what onset window, severity range (e.g. 0 to 50%), random or fixed motor, and can a fault
  happen during hover at B?
- **Test protocol:** OK to use a fixed seeded test set plus a few hand-picked faults?
- **Sensors:** OK to log sensors this week while the PID flies on true state? Which sensors: IMU only, or also
  simulated GPS and barometer with noise?
- **Plot 2:** OK to switch the x-axis to progress along the route? Does "expected route" mean the A→B line or the
  moving target point?
- **PID:** do your teammates have a Python PID to port, or should I write a cascaded one from scratch?
- **Goal 5:** how was the sentence meant to end?

## My answers
1. I am not sure. How would you design D1, D2, and D3? This was the plan from last week, but now that we have decided
   to run PPOs only as residual add-ons to the controller, things might change. As a matter of fact, we could explore
   more GitHub repos to find PPOs to test out. However, this should be decided later; it is not the goal for this
   week.
2. We could do a mix of code and pre-trained models. Those can be D1, D2, or D3. In fact, please help me design a
   table suitable for this.
3. For training, I would probably use randomized faults. It can be any motor, severity should be from 1-50%, and
   onset time should be between 10% and 90% of the way to B. The test sets should be fixed somehow. Actually, I am not
   sure how this would work for both training and testing. What would you recommend overall? Let us discuss this
   further.
4. Sure, but let's discuss it further together with 3. I want to get a clear idea of the experimental design.
5. What if we track both the sensors and the current "sensory info" that we have right now? Track as many sensors as
   you can.
6. Yes. "Expected route" means the red A→B line.
7. I added their separate work in `sim` and `drone-memory-control`. Feel free to look at what they did there. They
   claimed to use PID. If not, I guess we can stick with Lee? Talk with me more about this, though.
8. I finished the sentence.
9. As for adding the "altitude vs. time with the fault line" and "the 4 motor thrusts vs. time with the fault line"
   plots: pick the best 3 that would convey the most information for now. We can expand / go more in depth later.

Review my answers, explain the parts I asked about, and ask any more follow-up questions.
