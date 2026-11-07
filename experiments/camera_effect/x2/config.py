"""Every fixed setting of the study, in one place.

Units: metres, seconds, newtons, radians (degrees only where the name says so). World z points up.
Motors are numbered 1-4 as in the Skydio X2 model (actuators thrust1..thrust4); code indexes them 0-3.
"""

# ---------------------------------------------------------------- simulation
SIM_DT = 0.01  # s, physics step of the Skydio X2 model (100 Hz); the controller runs every step
GRAVITY = 9.81  # m/s^2

# ---------------------------------------------------------------- motors
MAX_THRUST = 6.0  # N, per-motor command cap (the X2 model itself allows 0-13 N)
MOTOR_TAU = 0.125  # s, first-order motor lag (same for spin-up and spin-down)

# ---------------------------------------------------------------- flight protocol
SPEED = 0.5  # m/s, the target point moves along the route at this speed
HOLD_T = 3.0  # s, hover at B after the target point arrives
POST_FAULT_T = 10.0  # s, every faulted flight continues at least this long after the fault starts
FAULT_MOTOR = 1  # which motor loses thrust (1-4), the same in every flight
FAULT_AT_FRACTION = 0.5  # the fault starts when the target point is this far along the route (0.5 = halfway)
LOSS_LEVELS = (0.0, 0.125, 0.25, 0.375, 0.50)  # fraction of thrust lost

# Straight route A -> B
STRAIGHT_A = (0.0, 0.0, 1.5)
STRAIGHT_B = (5.0, 0.0, 1.5)

# Random curved routes: seeds 1-4 give paths 1-4
RANDOM_PATH_SEEDS = (1, 2, 3, 4)
AREA_HALF = 3.0  # m, x and y stay within +-AREA_HALF (a 6 x 6 m area)
Z_RANGE = (1.0, 2.5)  # m, the whole route stays in this height band
LENGTH_RANGE = (3.0, 8.0)  # m, route length
WAYPOINTS = (1, 3)  # number of intermediate waypoints (inclusive range)
WAYPOINT_SPREAD = 1.0  # m, sideways push of a waypoint off the straight A-B line
WAYPOINT_DZ = 0.5  # m, height push of a waypoint off the A-B line
MIN_AB_DIST = 2.0  # m
MIN_TURN_RADIUS = 0.5  # m, no tighter turns than this

# ---------------------------------------------------------------- scoring
CRASH_MIN_Z = 0.30  # m, body origin (at the X2's feet) below this = crash (0.70 m of fall room from 1.0 m)
CRASH_GRACE_T = 0.5  # s, the height rule is ignored right after the start
CRASH_MAX_TILT_DEG = 60.0  # body z axis more than this from vertical = crash
ARRIVE_TOL = 0.10  # m, "arrived" = within this distance of B
RECOVER_TOL = 0.05  # m, altitude band around the target altitude
RECOVER_HOLD = 0.5  # s, must stay inside that band this long to count as recovered

# ---------------------------------------------------------------- sensors
IMU_ACCEL_NOISE = 0.05  # m/s^2, white noise per sample
IMU_GYRO_NOISE = 0.005  # rad/s, white noise per sample
IMU_ACCEL_BIAS = 0.02  # m/s^2, spread of a constant per-flight bias
IMU_GYRO_BIAS = 0.002  # rad/s, spread of a constant per-flight bias

CAMERA_RES = 128  # px, square frames
CAMERA_PERIOD = 0.1  # s, 10 Hz
CAMERA_POS = (0.15, 0.0, 0.05)  # m, in the drone body frame (forward of the body, looking along +x)
CAMERA_FOVY = 80.0  # deg
