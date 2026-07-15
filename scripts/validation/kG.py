from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from run_case import run_case

# -----------------------------
# Sweep values
# -----------------------------

kG_values = [
    0.25,
    0.50,
    0.75,
    1.00,
    1.25,
    1.50,
    1.75,
    2.00,
]

for kG in kG_values:

    run_case(

        case_name=f"kG_{kG:.2f}",

        fluid_updates={
            "reduced_frequency": kG,
        },

        solid_updates={

        },

        coupling_updates={

        }

    )