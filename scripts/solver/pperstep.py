# Fluid fidelity model
# Question is does it depend on particles shed per step alone or other factors like dissipation rate, and length of wake field. 
# What about parameters which influence growth of particles like lambda_vpm

# We can sweep over shedding rate

# Force along span, force time series, tip displacement time series

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from run_case import run_case

# Fluid fidelity parameter sweep
particles_per_step = [1, 2, 3]

for p_step in particles_per_step:
    run_case(
        case_name=f"particles_step_{p_step}",
        fluid_updates={
            "particles_per_step": p_step,
        },
        solid_updates={
            # Kept constant to isolate fluid fidelity effects
        },
        coupling_updates={
            # Kept constant to isolate fluid fidelity effects
        }
    )
