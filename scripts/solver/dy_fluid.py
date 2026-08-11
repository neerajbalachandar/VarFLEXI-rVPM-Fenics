#look at page 7 of byu paper for this

# drag coefficient v/s time for n_cp=10,20,40,80 (control points) and compare them
# lift coefficient v/s time for n_cp=10,20,40,80 (control points) and compare them
# drag coefficient v/s spanwise loc for n_cp=10,20,40,80 (control points) and compare them
# lift coefficient v/s chordwise for n_cp=10,20,40,80 (control points) and compare them


# dont know what discretization to use for the solid, so just keep it constant for now and vary the fluid discretization.


from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from run_case import run_case

span_disc = [10,60,150]

for span in span_disc:
    run_case(
        case_name=f"n_span_{span}",
        fluid_updates={
            "n_span": span
        },
        solid_updates={

        },
        coupling_updates={
        }
    )



