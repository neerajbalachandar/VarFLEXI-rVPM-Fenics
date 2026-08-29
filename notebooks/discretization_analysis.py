from pathlib import Path
import os
import re
import numpy as np
import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import yaml

# ------------------- GLOBAL CONFIGURATION -------------------
# Choose which sweep to analyse: 'dx_dy' or 'dt'
SWEEP_TYPE = 'dx_dy'   # <<< CHANGE HERE

# ---- OPTIONAL FILTERS (empty list = include all) ----
SELECTED_NX = []      # e.g. [10, 20] - only these nx appear (dx_dy)
SELECTED_NY = []      # e.g. [60, 120] - only these ny appear (dx_dy)
SELECTED_DT = []      # e.g. [0.001] - only these dt appear (dt)

# Base directory where your sweep results are stored
BASE = Path("/media/dysco/New Volume/Neeraj/varflexi solver/publish_result/results_solver/")
DX_DY_SWEEP_NAME = "dx_dy_solid"
DT_SWEEP_NAME = "dt"

# Fallback scales used only if a case folder does not contain copied configs.
FALLBACK_CHORD = 0.12       # [m]
FALLBACK_UINF = 8.0         # [m/s]

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
DX_DY_RE = re.compile(r"^n_x_(?P<nx>\d+)_n_y_(?P<ny>\d+)$")
DT_RE = re.compile(r"^dt_(?P<dt>[0-9.eE+-]+)$")

# ------------------- LOADING FUNCTION -------------------
def load_tip_displacement(case_dir):
    """Load time and tip displacement from solid diagnostics CSV."""
    path = case_dir / "results" / "solid" / "v18_reissner_mindlin_plate" / "solid_v18_diagnostics.csv"
    if not path.exists():
        matches = sorted((case_dir / "results").glob("solid/**/solid*_diagnostics.csv"))
        if matches:
            path = matches[0]
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


def load_yaml(path):
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as stream:
        return yaml.safe_load(stream) or {}


def case_scales(case_dir):
    """Return reference chord and free-stream speed for nondimensional plots."""
    solid_cfg = load_yaml(case_dir / "config" / "solid_params.yaml")
    fluid_cfg = load_yaml(case_dir / "config" / "fluid_params.yaml")

    root_chord = float(solid_cfg.get("root_chord", fluid_cfg.get("root_chord", FALLBACK_CHORD)))
    tip_chord = float(solid_cfg.get("tip_chord", fluid_cfg.get("tip_chord", root_chord)))
    c_ref = 0.5 * (root_chord + tip_chord)
    u_inf = float(fluid_cfg.get("vinf", FALLBACK_UINF))

    if c_ref <= 0.0:
        raise ValueError(f"Invalid chord scale c_ref={c_ref} for {case_dir}")
    if u_inf <= 0.0:
        raise ValueError(f"Invalid free-stream speed Uinf={u_inf} for {case_dir}")

    return c_ref, u_inf


def nondimensional_tip(case_dir, data):
    c_ref, u_inf = case_scales(case_dir)
    return {
        "t_nd": data["time"] * u_inf / c_ref,
        "tip_nd": data["tip"] / c_ref,
        "c_ref": c_ref,
        "u_inf": u_inf,
    }

# ------------------- OUTPUT DIRECTORY -------------------
output_dir = Path(__file__).resolve().parent / "plots"
output_dir.mkdir(parents=True, exist_ok=True)

# =====================================================================
#  dx_dy SWEEP
# =====================================================================
if SWEEP_TYPE == 'dx_dy':
    SWEEP_DIR = BASE / DX_DY_SWEEP_NAME
    if not SWEEP_DIR.exists():
        raise FileNotFoundError(f"Sweep directory not found: {SWEEP_DIR}")

    # Get all case directories matching the pattern n_x_..._n_y_...
    all_case_dirs = [d for d in SWEEP_DIR.iterdir() if d.is_dir()]
    parsed_cases = []
    for d in all_case_dirs:
        match = DX_DY_RE.match(d.name)
        if match:
            parsed_cases.append((d, int(match.group("nx")), int(match.group("ny"))))

    # ---- Apply nx/ny filters if provided ----
    parsed_cases = [
        (d, nx, ny)
        for d, nx, ny in parsed_cases
        if (not SELECTED_NX or nx in SELECTED_NX) and (not SELECTED_NY or ny in SELECTED_NY)
    ]
    parsed_cases = sorted(parsed_cases, key=lambda item: (item[1], item[2]))

    if not parsed_cases:
        raise RuntimeError("No dx-dy case directories found after filtering. "
                           "Check folder names or SELECTED_NX/SELECTED_NY.")

    # Load data for each case
    results = {}
    for case_dir, nx_val, ny_val in parsed_cases:
        case_name = case_dir.name
        print(f"Processing {case_name} ...")
        data = load_tip_displacement(case_dir)
        nd = nondimensional_tip(case_dir, data)
        results[case_name] = {**nd, "nx": nx_val, "ny": ny_val}

    # Extract unique nx and ny (only those present in results)
    unique_nx = sorted({data["nx"] for data in results.values()})
    unique_ny = sorted({data["ny"] for data in results.values()})
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
            ax.set_xlabel(r"$tU_\infty/c_{\mathrm{ref}}$")
            ax.set_ylabel(r"$s_{\mathrm{tip}}/c_{\mathrm{ref}}$")
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
            ax.set_xlabel(r"$tU_\infty/c_{\mathrm{ref}}$")
            ax.set_ylabel(r"$s_{\mathrm{tip}}/c_{\mathrm{ref}}$")
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
        ax.set_xlabel(r"$tU_\infty/c_{\mathrm{ref}}$")
        ax.set_ylabel(r"$s_{\mathrm{tip}}/c_{\mathrm{ref}}$")
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
    SWEEP_DIR = BASE / DT_SWEEP_NAME

    if not SWEEP_DIR.exists():
        raise FileNotFoundError(f"Sweep directory not found: {SWEEP_DIR}")

    all_case_dirs = [d for d in SWEEP_DIR.iterdir() if d.is_dir()]
    parsed_cases = []
    for d in all_case_dirs:
        match = DT_RE.match(d.name)
        if not match:
            continue
        dt_val = float(match.group("dt"))
        if SELECTED_DT and dt_val not in SELECTED_DT:
            continue
        parsed_cases.append((d, dt_val))
    parsed_cases = sorted(parsed_cases, key=lambda item: item[1])

    if not parsed_cases:
        raise RuntimeError("No dt cases found matching the requested dt values. "
                           "Check folder names or SELECTED_DT.")

    # Load data
    results = {}
    for case_dir, dt_val in parsed_cases:
        case_name = case_dir.name
        print(f"Processing {case_name} ...")
        data = load_tip_displacement(case_dir)
        nd = nondimensional_tip(case_dir, data)
        results[case_name] = {**nd, "dt": dt_val}

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
    ax_overlay.set_xlabel(r"$tU_\infty/c_{\mathrm{ref}}$")
    ax_overlay.set_ylabel(r"$s_{\mathrm{tip}}/c_{\mathrm{ref}}$")
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
            ax.set_xlabel(r"$tU_\infty/c_{\mathrm{ref}}$")
            ax.set_ylabel(r"$s_{\mathrm{tip}}/c_{\mathrm{ref}}$")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.grid(False)
        fig_sub.tight_layout()
        fig_sub.savefig(output_dir / "tip_disp_dt_subplots.pdf", bbox_inches="tight")
        plt.show()

else:
    raise ValueError("SWEEP_TYPE must be 'dx_dy' or 'dt'")
