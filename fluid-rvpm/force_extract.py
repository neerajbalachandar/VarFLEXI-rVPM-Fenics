import os
import time
import pyvista as pv
import pandas as pd

INPUT_DIR = "/home/dysco/FLOWUnsteady/VarFLEXI-rVPM-Fenics/fluid-rvpm/fluid-result"
OUTPUT_DIR = os.path.join(INPUT_DIR, "control_point_forces")

os.makedirs(OUTPUT_DIR, exist_ok=True)

RUN_NAME = "simple-wing"
VTK_PREFIX = f"{RUN_NAME}_Wing_vlm"
EXT = ".vtk"

START_STEP = 0


def extract_forces(vtk_path, step):
    print(f"[INFO] Reading {vtk_path}")

    mesh = pv.read(vtk_path)

    # --- CELL CENTERS (control points) ---
    centers = mesh.cell_centers().points

    data = {
        "x": centers[:, 0],
        "y": centers[:, 1],
        "z": centers[:, 2],
    }

    found_any = False

    for field in ["ftot", "Ftot", "L", "D", "Gamma"]:
        if field in mesh.cell_data:
            vec = mesh.cell_data[field]

            # Scalar vs vector safety
            if vec.ndim == 2:
                data[f"{field}_x"] = vec[:, 0]
                data[f"{field}_y"] = vec[:, 1]
                data[f"{field}_z"] = vec[:, 2]
            else:
                data[field] = vec

            found_any = True

    if not found_any:
        print(f"[INFO] No force data at step {step}, skipping")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    outfile = os.path.join(
        OUTPUT_DIR,
        f"forces_step_{step:04d}.csv"
    )

    pd.DataFrame(data).to_csv(outfile, index=False)
    print(f"[OK] Saved {outfile}")


def main():
    step = START_STEP

    while True:
        vtk_file = os.path.join(
            INPUT_DIR,
            f"{VTK_PREFIX}.{step}{EXT}"
        )

        # Wait until file appears
        while not os.path.exists(vtk_file):
            time.sleep(0.5)

        extract_forces(vtk_file, step)
        step += 1


if __name__ == "__main__":
    main()
