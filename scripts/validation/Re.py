from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from run_case import run_case

Re = [

10000,

20000,

30000

]

for re in Re:

    run_case(

        case_name=f"Re_{re}",

        fluid_updates={

            "Re":re

        }

    )