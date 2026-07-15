from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from run_case import run_case

youngs = [

7.0e10,

4.0e10,

2.0e10

]

for E in youngs:

    run_case(

        case_name=f"E_{E:.2e}",

        solid_updates={

            "material.E":E

        }

    )