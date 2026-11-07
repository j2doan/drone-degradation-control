# Week 3 Goals

## Summary
First drone week: get *something* flying from A to B in Isaac Lab, log the flight data, and set up the folder that the
rest of the project builds on.

## Where we were
1. Isaac Sim works on native Windows (WSL failed: Vulkan would not use the NVIDIA GPU).
2. Week 2: Isaac Lab's cartpole tasks trained with camera sensors, as proof that sensor data can be captured.
3. The PI approved the overall direction: simulate a drone, collect its data, and later use RL (PPO) to help it fly
   safely despite internal degradation.

## Goals for this week
1. Keep it simple: get a quadrotor to fly from point A to point B in Isaac Lab, with any working controller.
   - Use Isaac Lab's ARL Robot 1 quadrotor and its built-in Lee controller (no custom controller yet).
   - A = (0, 0, 1), B = (3, 0, 1): a straight line at 1 m height, then hover at B.
2. Split the work into 3 small scripts, each building on the one before:
   - `01_inspect_env.py`: print what the drone is (thrusters, mass, allocation matrix, thruster parameters, state).
   - `02_fly_a_to_b.py`: fly A to B and print progress; save nothing.
   - `03_collect_data.py`: the same flight, but log every physics step to a CSV and save a plot.
3. Collect data: one row per physics step (100 Hz) with position, orientation, velocities, the moving setpoint and the
   4 thrust commands, so it can be loaded with pandas.
4. Show the flight in the viewer: a green line for the ideal A-to-B path and a red line for the path actually flown.
5. Make it runnable from anywhere: launchers (`run.sh` for Git Bash, `run.ps1` for PowerShell) with the Isaac Lab path
   set once at the top, and no other machine-specific paths.
6. Reserve an `rl/` folder for the learning-based controllers later.

## Out of scope
Faults, PID, sensors (IMU, camera), metrics, curved paths, PPO training.

## Outcome (for reference)
The healthy flight reaches B about 0.007 m off. Two Isaac lessons were learned the hard way: `launch_simulation()`
ends the whole Python process when its block exits, so everything must be saved inside it; and most Isaac Lab modules
can only be imported after the simulator has started.
