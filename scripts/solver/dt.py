from pathlib import Path
import sys

# ran it for 1.6 sec which in the case of last dt iteration was 3200 timesteps

sys.path.append(str(Path(__file__).resolve().parents[1]))

from run_case import run_case

# Fixed total time from coupling.yaml
TOTAL_TIME = 2.0

# Desired time-step sizes
dt_values = [0.05, 0.01, 0.001]


for dt in dt_values:
    n_steps = int(TOTAL_TIME / dt)   # e.g., 100 for dt=0.05
    run_case(
        case_name=f"dt_{dt}",
        fluid_updates={},
        solid_updates={},
        coupling_updates={
            "total_time": TOTAL_TIME,
            "n_steps": n_steps
        }
    )
