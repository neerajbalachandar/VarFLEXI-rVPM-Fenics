#look at page 7 of byu paper for this

# drag coefficient v/s time for nx,ny=10,20,40,80 (control points) and compare them
# lift coefficient v/s time for nx,ny=10,20,40,80 (control points) and compare them
# drag coefficient v/s spanwise loc for nx,ny=10,20,40,80 (control points) and compare them
# lift coefficient v/s chordwise for nx,ny=10,20,40,80 (control points) and compare them


# dont know what discretization to use for the fluid, so just keep it constant for now and vary the fluid discretization.


from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from run_case import run_case

#can be changed .
nx = [10,20,40,80]
ny = [10,20,40,80]

for x in nx:
    for y in ny:
        run_case(
            case_name=f"n_x_{x}_n_y_{y}",
            fluid_updates={
                "n_x": x,
                "n_y": y
            },
            solid_updates={

            },
            coupling_updates={
            }
        )
