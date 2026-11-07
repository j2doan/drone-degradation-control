# PID flight after one motor loses thrust

Status: earlier Isaac flight results, using ARL Robot 1 on a straight A→B route.
This folder includes the original controller, scripts and three saved examples.

## What we tested

Could the PID keep flying after one motor suddenly became weaker? The route
went from (0,0,1) to (3,0,1) m. A seed selected the motor, thrust loss and onset.
These are PID results; the proposed PPO and history comparisons were not run.

## What happened

The earlier notes report one healthy flight and six fault seeds:

| Condition | Outcome |
|---|---|
| Healthy | Reached B, 0.023 m final error |
| 19–22% motor loss, three seeds | All recovered; 0.29–0.45 m peak drop and 4.6–4.9 s altitude recovery |
| 30–48% motor loss, three seeds | All counted as crashes; the 30% case touched the ground before flying on |

The saved examples are [healthy](data/pid_healthy.json),
[21% loss](data/pid_s0.json), and [48% loss](data/pid_s1.json), with CSVs and plots
beside them. The other four fault seeds are recorded in the earlier notes,
not supplied as saved traces here.

## Takeaway

The PID handled the milder sampled faults but lost too much altitude in the
stronger cases. This small sample varies motor and onset as well as severity,
so it does not establish a general failure threshold. Recovery here measures
altitude only. The logged `thrust_applied` is a fault-scaled actuator target,
not measured thrust after motor lag.

[Run instructions](GUIDE.md) · [Example flight](data/pid_s0.png)

The related [healthy Lee baseline](data/reference_lee_baseline/README.md) is
retained separately as background. It is not a completed paired-fault result.
