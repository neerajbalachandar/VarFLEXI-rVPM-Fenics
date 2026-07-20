#run for work conservation error for crm and then rbf and compare them 

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from run_case import run_case

force_trans_mode = ["rbf","crm"]

for ftm in force_trans_mode:
    run_case(
        case_name=f"force_transfer_{ftm}",
        fluid_updates={
        },
        solid_updates={

        },
        coupling_updates={
            "force_transfer_mode": ftm
        }
    )