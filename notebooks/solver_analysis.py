from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit   # kept for possible future use

# ------------------- GLOBAL CONFIGURATION -------------------
# Choose which sweep to analyse: 'dx_dy' or 'dt'
SWEEP_TYPE = 'dx_dy'   # <<< CHANGE HERE

# ---- OPTIONAL FILTERS (empty list = include all) ----
SELECTED_NX = [10]      # e.g. [30, 60] – only these nx appear (dx_dy)
SELECTED_NY = [30,150,300]      # e.g. [10]    – only these ny appear (dx_dy)
SELECTED_DT = []      # e.g. [0.01]  – only this dt appear (dt)

# Base directory where your sweep results are stored
BASE = Path("/home/dysco/FLOWUnsteady/VarFLEXI-rVPM-Fenics/results/parameter_sweeps/")

# Physical parameters (same for both sweeps)
chord = 0.10        # [m]
Uinf = 0.30         # [m/s]
kG = 1.82           # reduced frequency (fixed)
a_root = 0.175 * chord   # root heaving amplitude [m]
T = np.pi * chord / (kG * Uinf)   # oscillation period [s]

print(f"a_root = {a_root:.4f} m, T = {T:.4f} s")

# Plot style
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 8,
    "axes.labelsize": 9,
    "legend.fontsize": 7,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "axes.linewidth": 0.5,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "pdf.fonttype": 42,
    "ps.fonttype": 42
})

COLORS = plt.cm.tab10(np.linspace(0, 1, 10))

# ------------------- LOADING FUNCTION -------------------
def load_tip_displacement(case_dir):
    """Load time and tip displacement from solid diagnostics CSV."""
    path = case_dir / "results" / "solid" / "v18_reissner_mindlin_plate" / "solid_v18_diagnostics.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing: {path}")
    df = pd.read_csv(path)

    # Find time column
    time_col = None
    for col in df.columns:
        if 'time' in col.lower():
            time_col = col
            break
    if time_col is None:
        raise KeyError(f"No 'time' column found. Available: {df.columns.tolist()}")

    # Find tip column
    tip_col = None
    for col in df.columns:
        if 'tip' in col.lower():
            tip_col = col
            break
    if tip_col is None:
        raise KeyError(f"No 'tip' column found. Available: {df.columns.tolist()}")

    return {
        "time": df[time_col].to_numpy(dtype=float),
        "tip":  df[tip_col].to_numpy(dtype=float),
    }

# ------------------- OUTPUT DIRECTORY -------------------
output_dir = Path.cwd() / "plots"
output_dir.mkdir(parents=True, exist_ok=True)

# =====================================================================
#  dx_dy SWEEP
# =====================================================================
if SWEEP_TYPE == 'dx_dy':
    SWEEP_DIR = BASE / "dx_dy_solid"
    if not SWEEP_DIR.exists():
        raise FileNotFoundError(f"Sweep directory not found: {SWEEP_DIR}")

    # Get all case directories matching the pattern n_x_..._n_y_...
    all_case_dirs = [d for d in SWEEP_DIR.iterdir() if d.is_dir()]
    case_dirs = [d for d in all_case_dirs if d.name.startswith("n_x_") and "_n_y_" in d.name]

    # ---- Apply nx/ny filters if provided ----
    if SELECTED_NX or SELECTED_NY:
        filtered = []
        for d in case_dirs:
            parts = d.name.split("_")
            try:
                nx_val = int(parts[2])
                ny_val = int(parts[5])
            except (IndexError, ValueError):
                continue
            if SELECTED_NX and nx_val not in SELECTED_NX:
                continue
            if SELECTED_NY and ny_val not in SELECTED_NY:
                continue
            filtered.append(d)
        case_dirs = filtered

    if not case_dirs:
        raise RuntimeError("No dx-dy case directories found after filtering. "
                           "Check folder names or SELECTED_NX/SELECTED_NY.")

    # Load data for each case
    results = {}
    for case_dir in case_dirs:
        case_name = case_dir.name
        print(f"Processing {case_name} ...")
        data = load_tip_displacement(case_dir)
        t_nd = data["time"] / T
        tip_nd = data["tip"] / a_root
        results[case_name] = {"t_nd": t_nd, "tip_nd": tip_nd}

    # Extract unique nx and ny (only those present in results)
    unique_nx = sorted({int(k.split("_")[2]) for k in results})
    unique_ny = sorted({int(k.split("_")[5]) for k in results})
    print(f"unique_nx = {unique_nx}, unique_ny = {unique_ny}")

    # -------------------------------------------------
    # A. NY‑based plots, grouped into two multi‑panel figures
    # -------------------------------------------------
    nx_colors = {nx: COLORS[i % len(COLORS)] for i, nx in enumerate(unique_nx)}

    # --- Figure 1: first 3 ny values ---
    ny_group1 = unique_ny[:3]
    if ny_group1:
        n_sub = len(ny_group1)
        fig1, axes1 = plt.subplots(1, n_sub, figsize=(3.5 * n_sub, 3),
                                   dpi=300, sharex=True, sharey=True)
        if n_sub == 1:
            axes1 = [axes1]
        for ax, ny in zip(axes1, ny_group1):
            for nx in unique_nx:
                case_name = f"n_x_{nx}_n_y_{ny}"
                if case_name not in results:
                    continue
                t = results[case_name]["t_nd"]
                tip = results[case_name]["tip_nd"]
                step = max(1, len(t)//20)
                ax.plot(t, tip, color=nx_colors[nx], linewidth=0.8,
                        marker='o', markevery=step, markersize=2.5,
                        markeredgewidth=0.3, markerfacecolor='white',
                        label=f"nx={nx}")
            ax.set_title(f"ny = {ny}")
            ax.set_xlabel(r"$t/T$")
            ax.set_ylabel(r"$s_{\mathrm{tip}}/a_{\mathrm{root}}$")
            ax.set_xlim(0, 2)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.grid(False)
            ax.legend(frameon=False, loc="upper right", handletextpad=0.3, labelspacing=0.2)
        fig1.tight_layout()
        fig1.savefig(output_dir / "tip_disp_ny_subplots_group1.pdf", bbox_inches="tight")
        plt.show()

    # --- Figure 2: remaining ny values (max 4) ---
    ny_group2 = unique_ny[3:]
    if ny_group2:
        ncols = 2
        nrows = 2
        fig2, axes2 = plt.subplots(nrows, ncols, figsize=(7, 6), dpi=300,
                                   sharex=True, sharey=True)
        axes2 = axes2.flatten()
        for idx, ny in enumerate(ny_group2):
            ax = axes2[idx]
            for nx in unique_nx:
                case_name = f"n_x_{nx}_n_y_{ny}"
                if case_name not in results:
                    continue
                t = results[case_name]["t_nd"]
                tip = results[case_name]["tip_nd"]
                step = max(1, len(t)//20)
                ax.plot(t, tip, color=nx_colors[nx], linewidth=0.8,
                        marker='o', markevery=step, markersize=2.5,
                        markeredgewidth=0.3, markerfacecolor='white',
                        label=f"nx={nx}")
            ax.set_title(f"ny = {ny}")
            ax.set_xlabel(r"$t/T$")
            ax.set_ylabel(r"$s_{\mathrm{tip}}/a_{\mathrm{root}}$")
            ax.set_xlim(0, 2)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.grid(False)
            ax.legend(frameon=False, loc="upper right", handletextpad=0.3, labelspacing=0.2)
        # Hide any unused panels
        for ax in axes2[len(ny_group2):]:
            ax.set_visible(False)
        fig2.tight_layout()
        fig2.savefig(output_dir / "tip_disp_ny_subplots_group2.pdf", bbox_inches="tight")
        plt.show()

    # -------------------------------------------------
    # B. NX‑based plots – one separate figure per nx
    # -------------------------------------------------
    ny_colors = {ny: COLORS[i % len(COLORS)] for i, ny in enumerate(unique_ny)}
    for nx in unique_nx:
        fig, ax = plt.subplots(figsize=(4, 3), dpi=300)
        for ny in unique_ny:
            case_name = f"n_x_{nx}_n_y_{ny}"
            if case_name not in results:
                continue
            t = results[case_name]["t_nd"]
            tip = results[case_name]["tip_nd"]
            step = max(1, len(t)//20)
            ax.plot(t, tip, color=ny_colors[ny], linewidth=0.8,
                    marker='o', markevery=step, markersize=2.5,
                    markeredgewidth=0.3, markerfacecolor='white',
                    label=f"ny={ny}")
        ax.set_title(f"nx = {nx} (all ny)")
        ax.set_xlabel(r"$t/T$")
        ax.set_ylabel(r"$s_{\mathrm{tip}}/a_{\mathrm{root}}$")
        ax.set_xlim(0, 2)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(False)
        ax.legend(frameon=False, loc="upper right", handletextpad=0.3, labelspacing=0.2)
        fig.tight_layout()
        fig.savefig(output_dir / f"tip_disp_nx_{nx}.pdf", bbox_inches="tight")
        plt.show()

# =====================================================================
#  dt SWEEP
# =====================================================================
elif SWEEP_TYPE == 'dt':
    SWEEP_DIR = BASE / "dt"
    # Use SELECTED_DT if provided, otherwise the default three values
    dt_whitelist = SELECTED_DT if SELECTED_DT else [0.05, 0.01, 0.001]

    if not SWEEP_DIR.exists():
        raise FileNotFoundError(f"Sweep directory not found: {SWEEP_DIR}")

    all_case_dirs = [d for d in SWEEP_DIR.iterdir() if d.is_dir()]
    case_dirs = []
    for d in all_case_dirs:
        try:
            dt_val = float(d.name.split('_')[1])
            if dt_val in dt_whitelist:
                case_dirs.append(d)
        except (IndexError, ValueError):
            pass

    if not case_dirs:
        raise RuntimeError("No dt cases found matching the requested dt values. "
                           "Check folder names or SELECTED_DT.")

    # Load data
    results = {}
    for case_dir in case_dirs:
        case_name = case_dir.name
        print(f"Processing {case_name} ...")
        data = load_tip_displacement(case_dir)
        t_nd = data["time"] / T
        tip_nd = data["tip"] / a_root
        dt_val = float(case_name.split('_')[1])
        results[case_name] = {"t_nd": t_nd, "tip_nd": tip_nd, "dt": dt_val}

    sorted_cases = sorted(results.items(), key=lambda x: x[1]["dt"])

    # ---------- Overlay plot ----------
    fig_overlay, ax_overlay = plt.subplots(figsize=(4, 3), dpi=300)
    for idx, (case_name, data) in enumerate(sorted_cases):
        dt = data["dt"]
        label = f"dt = {dt:.3f} s"
        t = data["t_nd"]
        tip = data["tip_nd"]
        step = max(1, len(t)//30)
        ax_overlay.plot(t, tip, color=COLORS[idx % len(COLORS)], linewidth=0.8,
                        marker='o', markevery=step, markersize=2.5,
                        markeredgewidth=0.3, markerfacecolor='white',
                        label=label)
    ax_overlay.set_xlabel(r"$t/T$")
    ax_overlay.set_ylabel(r"$s_{\mathrm{tip}}/a_{\mathrm{root}}$")
    ax_overlay.set_xlim(0, 2)
    ax_overlay.spines["top"].set_visible(False)
    ax_overlay.spines["right"].set_visible(False)
    ax_overlay.grid(False)
    ax_overlay.legend(frameon=False, loc="upper right", handletextpad=0.3, labelspacing=0.2)
    fig_overlay.tight_layout()
    fig_overlay.savefig(output_dir / "tip_disp_dt_overlay.pdf", bbox_inches="tight")
    plt.show()

    # ---------- Subplots (one per dt) ----------
    if sorted_cases:
        n_dt = len(sorted_cases)
        fig_sub, axes_sub = plt.subplots(1, n_dt, figsize=(3.5 * n_dt, 3.5), dpi=300,
                                         sharex=True, sharey=True)
        if n_dt == 1:
            axes_sub = [axes_sub]
        else:
            axes_sub = axes_sub.flatten()
        for ax, (case_name, data) in zip(axes_sub, sorted_cases):
            dt = data["dt"]
            t = data["t_nd"]
            tip = data["tip_nd"]
            ax.plot(t, tip, color='#2C5596', linewidth=0.8)
            ax.set_title(f"dt = {dt:.3f} s")
            ax.set_xlabel(r"$t/T$")
            ax.set_ylabel(r"$s_{\mathrm{tip}}/a_{\mathrm{root}}$")
            ax.set_xlim(0, 2)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.grid(False)
        fig_sub.tight_layout()
        fig_sub.savefig(output_dir / "tip_disp_dt_subplots.pdf", bbox_inches="tight")
        plt.show()

else:
    raise ValueError("SWEEP_TYPE must be 'dx_dy' or 'dt'")