#check page 6 of the byu paper for this

# dt plots (drag coefficient v/s t at dt=0.002,0.001,0.0005 seconds) 
# dt plots (lift coefficient v/s t at dt=0.002,0.001,0.0005 seconds)
#dt plots (drag coefficient v/s spanwise location at dt =0.002,0.001,0.0005 seconds)
#dt plots (lift coefficient v/s spanwise location at dt =0.002,0.001,0.0005 seconds)
 
#the dependent quantities can be calculated from each of the run of dt.

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from run_case import run_case

timesteps = [0.001,0.002,0.0005]

for dt in timesteps:
    run_case(
        case_name=f"dt_{dt}",
        fluid_updates={
        },
        solid_updates={

        },
        coupling_updates={
            "total_time":1.0,
            "n_steps":int(1.0/dt)
        }
    )



