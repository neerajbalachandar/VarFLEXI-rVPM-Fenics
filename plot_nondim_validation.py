import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# -----------------------------
# USER INPUTS
# -----------------------------

csv_file = "tip_displacement.csv"

f = 25.65                 # Hz
T = 1.0 / f

c = 0.1                   # chord
a_root = 0.0175           # root amplitude

# -----------------------------
# LOAD DATA
# -----------------------------

data = pd.read_csv(csv_file)

time = data["time"].values
tip_disp = data["tip_z"].values
root_disp = data["root_z"].values

# -----------------------------
# NON-DIMENSIONALIZATION
# -----------------------------

t_nd = time / T

tip_nd = tip_disp / a_root
root_nd = root_disp / a_root

# -----------------------------
# PLOT
# -----------------------------

plt.figure(figsize=(8,5))

plt.plot(
    t_nd,
    root_nd,
    linewidth=2,
    label="Root"
)

plt.plot(
    t_nd,
    tip_nd,
    linewidth=2,
    label="Tip"
)

plt.xlabel(r"$t/T$")
plt.ylabel(r"$s/a_{root}$")

plt.xlim([0,2])

plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()