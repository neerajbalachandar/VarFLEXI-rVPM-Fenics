from pathlib import Path
import json
import os
import re
import numpy as np
import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import yaml

# ------------------- GLOBAL CONFIGURATION -------------------
# Choose which fluid-fidelity sweep to analyse: 'pperstep' or 'dy_fluid'
SWEEP_TYPE = "dy_fluid"   # <<< CHANGE HERE

# ---- OPTIONAL FILTERS (empty list = include all) ----
SELECTED_PARTICLES_PER_STEP = []   # e.g. [1, 2]
SELECTED_N_SPAN = []               # e.g. [20, 110]

# Base directory where your published sweep results are stored.
BASE = Path("/media/dysco/New Volume/Neeraj/varflexi solver/publish_result/results_solver/")

# Fallback scales used only if a case folder does not contain copied configs.
FALLBACK_CHORD = 0.12       # [m]
FALLBACK_UINF = 8.0         # [m/s]

plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "legend.fontsize": 7,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,

    "axes.linewidth": 0.6,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 3.0,
    "ytick.major.size": 3.0,

    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


# Publication plot scheme:
# Case 1 -> red triangle
# Case 2 -> blue square
# Case 3 -> black circle

COLORS = [
    "#C00000FF",   # red
    "#1F4E79FF",   # blue
    "#00000052",   # black
]

MARKERS = [
    "^",         # triangle
    "s",         # square
    "o",         # circle
]


PSTEP_RE = re.compile(r"^particles_step_(?P<value>\d+)$")
NSPAN_RE = re.compile(r"^n_span_(?P<value>\d+)$")


def load_yaml(path):
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as stream:
        return yaml.safe_load(stream) or {}


def load_tip_displacement(case_dir):
    """Load time and tip displacement from a case's solid diagnostics CSV."""
    path = case_dir / "results" / "solid" / "v18_reissner_mindlin_plate" / "solid_v18_diagnostics.csv"
    if not path.exists():
        matches = sorted((case_dir / "results").glob("solid/**/solid*_diagnostics.csv"))
        if matches:
            path = matches[0]
    if not path.exists():
        raise FileNotFoundError(f"Missing solid diagnostics CSV below: {case_dir / 'results' / 'solid'}")

    df = pd.read_csv(path)

    time_col = next((col for col in df.columns if "time" in col.lower()), None)
    if time_col is None:
        raise KeyError(f"No time column found in {path}. Available: {df.columns.tolist()}")

    tip_col = next((col for col in df.columns if "tip" in col.lower()), None)
    if tip_col is None:
        raise KeyError(f"No tip displacement column found in {path}. Available: {df.columns.tolist()}")

    return {
        "time": df[time_col].to_numpy(dtype=float),
        "tip": df[tip_col].to_numpy(dtype=float),
    }


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


def coupling_dt(case_dir):
    coupling_cfg = load_yaml(case_dir / "config" / "coupling_params.yaml")
    total_time = float(coupling_cfg.get("total_time", 0.0))
    n_steps = int(coupling_cfg.get("n_steps", coupling_cfg.get("nsteps", 0)))
    if total_time <= 0.0 or n_steps <= 0:
        raise ValueError(f"Invalid coupling time scale in {case_dir / 'config' / 'coupling_params.yaml'}")
    return total_time / n_steps


def tip_force_from_payload(payload):
    forces = np.asarray(payload.get("force_sent", payload.get("force_received", [])), dtype=float)
    if forces.size == 0:
        return None

    n_span = int(payload.get("n_span", len(forces)))
    n_chord = int(payload.get("n_chord", 1))
    if forces.shape[0] != n_span * n_chord:
        return forces[-1]

    indexing = str(payload.get("indexing", "span-major"))
    if indexing == "chord-major":
        grid = forces.reshape((n_chord, n_span, 3)).transpose((1, 0, 2))
    else:
        grid = forces.reshape((n_span, n_chord, 3))

    return grid[-1, :, :].sum(axis=0)


def load_tip_panel_force(case_dir):
    """Load net force vector at the tip span station from coupling JSONL history."""
    path = case_dir / "results" / "coupling" / "forces_sent_received_history.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Missing force history: {path}")

    c_ref, u_inf = case_scales(case_dir)
    dt = coupling_dt(case_dir)
    t_nd = []
    force_mag = []
    force_x = []
    force_y = []
    force_z = []

    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            payload = json.loads(line)
            tip_force = tip_force_from_payload(payload)
            if tip_force is None:
                continue
            step = int(payload.get("step", len(t_nd) + 1))
            t_nd.append(step * dt * u_inf / c_ref)
            force_x.append(float(tip_force[0]))
            force_y.append(float(tip_force[1]))
            force_z.append(float(tip_force[2]))
            force_mag.append(float(np.linalg.norm(tip_force)))

    if not t_nd:
        raise RuntimeError(f"No tip-panel force rows found in {path}")

    return {
        "force_t_nd": np.asarray(t_nd, dtype=float),
        "tip_force": np.asarray(force_mag, dtype=float),
        "tip_force_x": np.asarray(force_x, dtype=float),
        "tip_force_y": np.asarray(force_y, dtype=float),
        "tip_force_z": np.asarray(force_z, dtype=float),
    }


def discover_cases(sweep_dir, pattern, filter_values):
    cases = []
    for case_dir in sweep_dir.iterdir():
        if not case_dir.is_dir():
            continue
        match = pattern.match(case_dir.name)
        if not match:
            continue
        value = int(match.group("value"))
        if filter_values and value not in filter_values:
            continue
        cases.append((case_dir, value))
    return sorted(cases, key=lambda item: item[1])


def load_sweep_cases():
    if SWEEP_TYPE == "pperstep":
        sweep_dir = BASE / "pperstep"
        value_key = "particles_per_step"
        label = "particles/step"
        filename_key = "pperstep"
    elif SWEEP_TYPE == "dy_fluid":
        sweep_dir = BASE / "dy_fluid"
        value_key = "n_span"
        label = r"$n_{\mathrm{span}}$"
        filename_key = "dy_fluid"
    else:
        raise ValueError("SWEEP_TYPE must be 'pperstep' or 'dy_fluid'")

    if not sweep_dir.exists():
        raise FileNotFoundError(f"Sweep directory not found: {sweep_dir}")
    if SWEEP_TYPE == "pperstep":
        cases = discover_cases(sweep_dir, PSTEP_RE, SELECTED_PARTICLES_PER_STEP)
    else:
        cases = discover_cases(sweep_dir, NSPAN_RE, SELECTED_N_SPAN)
    if not cases:
        raise RuntimeError(f"No {SWEEP_TYPE} cases found in {sweep_dir}. Check folder names or filters.")

    results = {}
    for case_dir, value in cases:
        print(f"Processing {case_dir.name} ...")
        raw = load_tip_displacement(case_dir)
        nd = nondimensional_tip(case_dir, raw)
        force = load_tip_panel_force(case_dir)
        results[case_dir.name] = {**nd, **force, value_key: value, "case_dir": case_dir}

    return results, value_key, label, filename_key


def plot_overlay(results, value_key, label, filename_key, output_dir):
    sorted_cases = sorted(
        results.items(),
        key=lambda item: item[1][value_key]
    )

    fig, ax = plt.subplots(figsize=(4, 3), dpi=300)

    for idx, (_, data) in enumerate(sorted_cases):
        value = data[value_key]
        t = data["t_nd"]
        tip = data["tip_nd"]

        step = max(1, len(t) // 30)

        color = COLORS[idx % len(COLORS)]
        marker = MARKERS[idx % len(MARKERS)]

        # Shift marker locations between curves so they do not overlap.
        offset = int(idx * step / len(sorted_cases))
        marker_indices = np.arange(offset, len(t), step)

        ax.plot(
            t,
            tip,
            color=color,
            linewidth=1.0,
            marker=marker,
            markevery=marker_indices,
            markersize=4.0,
            markeredgewidth=0.6,
            markerfacecolor="white",
            markeredgecolor=color,
            label=f"{label} = {value}",
        )

    ax.set_xlabel(r"$tU_\infty/c_{\mathrm{ref}}$")
    ax.set_ylabel(r"$s_{\mathrm{tip}}/c_{\mathrm{ref}}$")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)

    ax.legend(
        frameon=False,
        loc="best",
        handletextpad=0.3,
        labelspacing=0.2
    )

    ax.set_xlim(0, 200)

    fig.tight_layout()

    fig.savefig(
        output_dir / f"tip_disp_{filename_key}_overlay.pdf",
        bbox_inches="tight"
    )

    plt.show()


def plot_subplots(results, value_key, label, filename_key, output_dir):
    sorted_cases = sorted(results.items(), key=lambda item: item[1][value_key])
    n_cases = len(sorted_cases)
    fig, axes = plt.subplots(1, n_cases, figsize=(3.5 * n_cases, 3.5), dpi=300, sharex=True, sharey=True)
    if n_cases == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for ax, (_, data) in zip(axes, sorted_cases):
        value = data[value_key]
        ax.plot(data["t_nd"], data["tip_nd"], color="#2C5596", linewidth=0.8)
        ax.set_title(f"{label} = {value}")
        ax.set_xlabel(r"$tU_\infty/c_{\mathrm{ref}}$")
        ax.set_ylabel(r"$s_{\mathrm{tip}}/c_{\mathrm{ref}}$")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(False)

    fig.tight_layout()
    fig.savefig(output_dir / f"tip_disp_{filename_key}_subplots.pdf", bbox_inches="tight")
    plt.xlim(0,200)
    plt.show()


def plot_force_overlay(results, value_key, label, filename_key, output_dir):
    sorted_cases = sorted(
        results.items(),
        key=lambda item: item[1][value_key]
    )

    fig, ax = plt.subplots(figsize=(4, 3), dpi=300)

    for idx, (_, data) in enumerate(sorted_cases):
        value = data[value_key]
        t = data["force_t_nd"]
        force = data["tip_force"]

        step = max(1, len(t) // 30)

        color = COLORS[idx % len(COLORS)]
        marker = MARKERS[idx % len(MARKERS)]

        offset = int(idx * step / len(sorted_cases))
        marker_indices = np.arange(offset, len(t), step)

        ax.plot(
            t,
            force,
            color=color,
            linewidth=1.0,
            marker=marker,
            markevery=marker_indices,
            markersize=4.0,
            markeredgewidth=0.6,
            markerfacecolor="white",
            markeredgecolor=color,
            label=f"{label} = {value}",
        )

    ax.set_xlabel(r"$tU_\infty/c_{\mathrm{ref}}$")
    ax.set_ylabel(r"$\|\mathbf{F}_{\mathrm{tip}}\|$ [N]")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)

    ax.legend(
        frameon=False,
        loc="best",
        handletextpad=0.3,
        labelspacing=0.2
    )

    ax.set_xlim(0, 200)

    fig.tight_layout()

    fig.savefig(
        output_dir / f"tip_force_{filename_key}_overlay.pdf",
        bbox_inches="tight"
    )

    plt.show()



def plot_force_subplots(results, value_key, label, filename_key, output_dir):
    sorted_cases = sorted(results.items(), key=lambda item: item[1][value_key])
    n_cases = len(sorted_cases)
    fig, axes = plt.subplots(1, n_cases, figsize=(3.5 * n_cases, 3.5), dpi=300, sharex=True, sharey=True)
    if n_cases == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for ax, (_, data) in zip(axes, sorted_cases):
        value = data[value_key]
        ax.plot(data["force_t_nd"], data["tip_force"], color="#8A3FFC", linewidth=0.8)
        ax.set_title(f"{label} = {value}")
        ax.set_xlabel(r"$tU_\infty/c_{\mathrm{ref}}$")
        ax.set_ylabel(r"$\|\mathbf{F}_{\mathrm{tip}}\|$ [N]")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(False)

    fig.tight_layout()
    plt.xlim(0,200)
    fig.savefig(output_dir / f"tip_force_{filename_key}_subplots.pdf", bbox_inches="tight")
    plt.show()


output_dir = Path(__file__).resolve().parent / "plots"
output_dir.mkdir(parents=True, exist_ok=True)

results, value_key, label, filename_key = load_sweep_cases()
plot_overlay(results, value_key, label, filename_key, output_dir)
plot_subplots(results, value_key, label, filename_key, output_dir)
plot_force_overlay(results, value_key, label, filename_key, output_dir)
plot_force_subplots(results, value_key, label, filename_key, output_dir)
