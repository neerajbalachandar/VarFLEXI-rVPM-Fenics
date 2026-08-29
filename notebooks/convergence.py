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
# Choose one setting or "all": "dt", "pperstep", "dy_fluid", "dx_dy", "all"
SETTINGS = ["dt","pperstep","dy_fluid"]  # <<< CHANGE HERE

# Empty lists include all matching cases.
SELECTED_DT = []
SELECTED_PARTICLES_PER_STEP = []
SELECTED_N_SPAN = []
SELECTED_NX = []
SELECTED_NY = []

BASE = Path("/media/dysco/New Volume/Neeraj/varflexi solver/publish_result/results_solver/")
OUTPUT_DIR = Path(__file__).resolve().parent / "plots" / "convergence"

FALLBACK_CHORD = 0.12
FALLBACK_UINF = 8.0
FALLBACK_RHO = 1.0

# ------------------- COLOUR SCHEME -------------------
# Edit these hex values to change the colours used across all plots.
COLOUR_SCHEME = {
    "series": [
        "#0b70b9",  # blue
        "#c71919",  # red
        "#1ca21c",  # green
        "#d1813a",  # orange
        
        "#9467bd",  # purple
        "#17becf",  # cyan
        "#8c564b",  # brown
        "#e377c2",  # pink
        "#7f7f7f",  # gray
        "#bcbd22",  # olive
    ],
    "single_tip": "#2C5596",
    "single_force": "#8A3FFC",
    "marker_face": "white",
}


def series_color(index):
    return COLOUR_SCHEME["series"][index % len(COLOUR_SCHEME["series"])]

CASE_PATTERNS = {
    "dt": re.compile(r"^dt_(?P<dt>[0-9.eE+-]+)$"),
    "pperstep": re.compile(r"^particles_step_(?P<particles_per_step>\d+)$"),
    "dy_fluid": re.compile(r"^n_span_(?P<n_span>\d+)$"),
    "dx_dy": re.compile(r"^n_x_(?P<nx>\d+)_n_y_(?P<ny>\d+)$"),
}

SWEEP_DIRS = {
    "dt": "dt",
    "pperstep": "pperstep",
    "dy_fluid": "dy_fluid",
    "dx_dy": "dx_dy_solid",
}

LABEL_KEYS = {
    "dt": "dt",
    "pperstep": "particles_per_step",
    "dy_fluid": "n_span",
    "dx_dy": "case_label",
}

LABEL_NAMES = {
    "dt": "dt",
    "pperstep": "particles/step",
    "dy_fluid": r"$n_{\mathrm{span}}$",
    "dx_dy": "case",
}


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
    "ps.fonttype": 42,
})


def load_yaml(path):
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as stream:
        return yaml.safe_load(stream) or {}


def case_configs(case_dir):
    return (
        load_yaml(case_dir / "config" / "fluid_params.yaml"),
        load_yaml(case_dir / "config" / "solid_params.yaml"),
        load_yaml(case_dir / "config" / "coupling_params.yaml"),
    )


def case_scales(case_dir):
    fluid, solid, coupling = case_configs(case_dir)
    root_chord = float(solid.get("root_chord", fluid.get("root_chord", FALLBACK_CHORD)))
    tip_chord = float(solid.get("tip_chord", fluid.get("tip_chord", root_chord)))
    span = float(solid.get("span", fluid.get("span", 1.0)))
    c_ref = 0.5 * (root_chord + tip_chord)
    u_inf = float(fluid.get("vinf", FALLBACK_UINF))
    rho = float(fluid.get("rho", FALLBACK_RHO))
    total_time = float(coupling.get("total_time", fluid.get("total_time", 0.0)))
    n_steps = int(coupling.get("n_steps", coupling.get("nsteps", fluid.get("nsteps", 0))))
    dt = total_time / n_steps if total_time > 0.0 and n_steps > 0 else np.nan
    return {
        "c_ref": c_ref,
        "u_inf": u_inf,
        "rho": rho,
        "span": span,
        "dt": dt,
        "q_inf": 0.5 * rho * u_inf * u_inf,
    }


def style_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)


def load_tip_displacement(case_dir):
    path = case_dir / "results" / "solid" / "v18_reissner_mindlin_plate" / "solid_v18_diagnostics.csv"
    if not path.exists():
        matches = sorted((case_dir / "results").glob("solid/**/solid*_diagnostics.csv"))
        if matches:
            path = matches[0]
    if not path.exists():
        raise FileNotFoundError(f"Missing solid diagnostics CSV below {case_dir}")
    df = pd.read_csv(path)
    time_col = next((col for col in df.columns if "time" in col.lower()), None)
    tip_col = next((col for col in df.columns if "tip" in col.lower()), None)
    if time_col is None or tip_col is None:
        raise KeyError(f"Could not find time/tip columns in {path}: {df.columns.tolist()}")
    scales = case_scales(case_dir)
    return {
        "t_nd": df[time_col].to_numpy(dtype=float) * scales["u_inf"] / scales["c_ref"],
        "tip_nd": df[tip_col].to_numpy(dtype=float) / scales["c_ref"],
    }


def load_coupling_history(case_dir):
    path = case_dir / "results" / "coupling" / "coupling_history.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing coupling history: {path}")
    df = pd.read_csv(path)
    scales = case_scales(case_dir)
    if "step" in df.columns and np.isfinite(scales["dt"]):
        t_nd = df["step"].to_numpy(dtype=float) * scales["dt"] * scales["u_inf"] / scales["c_ref"]
    else:
        t_nd = np.arange(len(df), dtype=float)
    return {
        "force_t_nd": t_nd,
        "cl": df["cl"].to_numpy(dtype=float),
        "cd": df["cd"].to_numpy(dtype=float),
    }


def force_grid(payload):
    forces = np.asarray(payload.get("force_sent", payload.get("force_received", [])), dtype=float)
    if forces.size == 0:
        return None
    n_span = int(payload.get("n_span", len(forces)))
    n_chord = int(payload.get("n_chord", 1))
    if forces.shape[0] != n_span * n_chord:
        n_span = forces.shape[0]
        n_chord = 1
    indexing = str(payload.get("indexing", "span-major"))
    if indexing == "chord-major":
        return forces.reshape((n_chord, n_span, 3)).transpose((1, 0, 2))
    return forces.reshape((n_span, n_chord, 3))


def load_force_json(case_dir):
    path = case_dir / "results" / "coupling" / "forces_sent_received_history.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Missing force JSONL: {path}")
    scales = case_scales(case_dir)
    profiles_lift = []
    profiles_drag = []
    tip_force = []
    t_nd = []
    eta = None
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            payload = json.loads(line)
            grid = force_grid(payload)
            if grid is None:
                continue
            n_span = grid.shape[0]
            panel_forces = grid.sum(axis=1)
            panel_area = scales["c_ref"] * scales["span"] / max(n_span, 1)
            denom = max(scales["q_inf"] * panel_area, 1.0e-16)
            profiles_lift.append(panel_forces[:, 2] / denom)
            profiles_drag.append(-panel_forces[:, 0] / denom)
            tip_force.append(np.linalg.norm(panel_forces[-1]))
            step = int(payload.get("step", len(t_nd) + 1))
            t_nd.append(step * scales["dt"] * scales["u_inf"] / scales["c_ref"])
            if eta is None:
                eta = np.asarray(payload.get("eta_span", []), dtype=float)
                if eta.size != n_span:
                    eta = (np.arange(n_span, dtype=float) + 0.5) / n_span
    if not profiles_lift:
        raise RuntimeError(f"No force payloads found in {path}")
    return {
        "force_t_nd": np.asarray(t_nd, dtype=float),
        "tip_force": np.asarray(tip_force, dtype=float),
        "eta_span": eta,
        "cl_span": np.asarray(profiles_lift, dtype=float),
        "cd_span": np.asarray(profiles_drag, dtype=float),
    }


def discover_cases(setting):
    sweep_dir = BASE / SWEEP_DIRS[setting]
    if not sweep_dir.exists():
        raise FileNotFoundError(f"Sweep directory not found: {sweep_dir}")
    pattern = CASE_PATTERNS[setting]
    cases = []
    for case_dir in sweep_dir.iterdir():
        if not case_dir.is_dir():
            continue
        match = pattern.match(case_dir.name)
        if not match:
            continue
        meta = {}
        for key, value in match.groupdict().items():
            meta[key] = float(value) if key == "dt" else int(value)
        if setting == "dt" and SELECTED_DT and meta["dt"] not in SELECTED_DT:
            continue
        if setting == "pperstep" and SELECTED_PARTICLES_PER_STEP and meta["particles_per_step"] not in SELECTED_PARTICLES_PER_STEP:
            continue
        if setting == "dy_fluid" and SELECTED_N_SPAN and meta["n_span"] not in SELECTED_N_SPAN:
            continue
        if setting == "dx_dy":
            if SELECTED_NX and meta["nx"] not in SELECTED_NX:
                continue
            if SELECTED_NY and meta["ny"] not in SELECTED_NY:
                continue
            meta["case_label"] = f"nx={meta['nx']}, ny={meta['ny']}"
        cases.append((case_dir, meta))
    return sorted(cases, key=lambda item: tuple(item[1].values()))


def load_cases(setting):
    cases = discover_cases(setting)
    if not cases:
        raise RuntimeError(f"No cases found for {setting}")
    results = {}
    for case_dir, meta in cases:
        print(f"[{setting}] Processing {case_dir.name} ...")
        data = {}
        data.update(meta)
        try:
            data.update(load_tip_displacement(case_dir))
            data["has_tip"] = True
        except FileNotFoundError as exc:
            print(f"  Skipping tip displacement for {case_dir.name}: {exc}")
            data["t_nd"] = np.asarray([], dtype=float)
            data["tip_nd"] = np.asarray([], dtype=float)
            data["has_tip"] = False
        data.update(load_coupling_history(case_dir))
        data.update(load_force_json(case_dir))
        data["case_dir"] = case_dir
        results[case_dir.name] = data
    return results


def sorted_cases(results, setting):
    key = LABEL_KEYS[setting]
    return sorted(results.items(), key=lambda item: item[1][key])


def label_for(data, setting):
    key = LABEL_KEYS[setting]
    return f"{LABEL_NAMES[setting]} = {data[key]}"


def save_show(fig, output_dir, name):
    fig.tight_layout()
    fig.savefig(output_dir / name, bbox_inches="tight")
    plt.show()
    plt.close(fig)


def plot_tip_overlay(results, setting, output_dir):
    cases = [(name, data) for name, data in sorted_cases(results, setting) if data.get("has_tip")]
    if not cases:
        print(f"[{setting}] No tip displacement data available; skipping overlay plot.")
        return
    fig, ax = plt.subplots(figsize=(4, 3), dpi=300)
    for idx, (_, data) in enumerate(cases):
        step = max(1, len(data["t_nd"]) // 30)
        ax.plot(data["t_nd"], data["tip_nd"], color=series_color(idx), linewidth=0.8,
                marker="o", markevery=step, markersize=2.5, markeredgewidth=0.3,
                markerfacecolor=COLOUR_SCHEME["marker_face"], label=label_for(data, setting))
    ax.set_xlabel(r"$tU_\infty/c_{\mathrm{ref}}$")
    ax.set_ylabel(r"$s_{\mathrm{tip}}/c_{\mathrm{ref}}$")
    style_axes(ax)
    ax.legend(frameon=False, loc="best", handletextpad=0.3, labelspacing=0.2)
    save_show(fig, output_dir, f"tip_disp_{setting}_overlay.pdf")


def plot_tip_subplots(results, setting, output_dir):
    cases = [(name, data) for name, data in sorted_cases(results, setting) if data.get("has_tip")]
    if not cases:
        print(f"[{setting}] No tip displacement data available; skipping subplot figure.")
        return
    fig, axes = plt.subplots(1, len(cases), figsize=(3.5 * len(cases), 3.5), dpi=300, sharex=True, sharey=True)
    axes = [axes] if len(cases) == 1 else axes.flatten()
    for ax, (_, data) in zip(axes, cases):
        ax.plot(data["t_nd"], data["tip_nd"], color=COLOUR_SCHEME["single_tip"], linewidth=0.8)
        ax.set_title(label_for(data, setting))
        ax.set_xlabel(r"$tU_\infty/c_{\mathrm{ref}}$")
        ax.set_ylabel(r"$s_{\mathrm{tip}}/c_{\mathrm{ref}}$")
        style_axes(ax)
    save_show(fig, output_dir, f"tip_disp_{setting}_subplots.pdf")


def plot_tip_force(results, setting, output_dir):
    fig, ax = plt.subplots(figsize=(4, 3), dpi=300)
    for idx, (_, data) in enumerate(sorted_cases(results, setting)):
        step = max(1, len(data["force_t_nd"]) // 30)
        ax.plot(data["force_t_nd"], data["tip_force"], color=series_color(idx), linewidth=0.8,
                marker="o", markevery=step, markersize=2.5, markeredgewidth=0.3,
                markerfacecolor=COLOUR_SCHEME["marker_face"], label=label_for(data, setting))
    ax.set_xlabel(r"$tU_\infty/c_{\mathrm{ref}}$")
    ax.set_ylabel(r"$\|\mathbf{F}_{\mathrm{tip}}\|$ [N]")
    style_axes(ax)
    ax.legend(frameon=False, loc="best", handletextpad=0.3, labelspacing=0.2)
    save_show(fig, output_dir, f"tip_force_{setting}_overlay.pdf")


def plot_coeff_time(results, setting, coeff, output_dir):
    fig, ax = plt.subplots(figsize=(4, 3), dpi=300)

    for idx, (_, data) in enumerate(sorted_cases(results, setting)):
        step = max(1, len(data["force_t_nd"]) // 30)

        ax.plot(
            data["force_t_nd"],
            data[coeff],
            color=series_color(idx),
            linewidth=0.8,
            marker="o",
            markevery=step,
            markersize=2.5,
            markeredgewidth=0.3,
            markerfacecolor=COLOUR_SCHEME["marker_face"],
            label=label_for(data, setting),
        )

    ax.set_xlim(0, 100)
    ax.set_xlabel(r"$tU_\infty/c_{\mathrm{ref}}$")

    if coeff.lower() == "cl":
        ax.set_ylabel(r"$C_L$")
    elif coeff.lower() == "cd":
        ax.set_ylabel(r"$C_D$")
    else:
        ax.set_ylabel(rf"${coeff.upper()}$")

    style_axes(ax)
    ax.legend(
        frameon=False,
        loc="best",
        handletextpad=0.3,
        labelspacing=0.2,
    )

    save_show(fig, output_dir, f"{coeff}_{setting}_time.pdf")


def mean_span_profile(values):
    start = max(0, int(0.5 * values.shape[0]))
    return values[start:, :].mean(axis=0)


def plot_coeff_span(results, setting, coeff, output_dir):
    fig, ax = plt.subplots(figsize=(4, 3), dpi=300)
    profile_key = f"{coeff}_span"

    for idx, (_, data) in enumerate(sorted_cases(results, setting)):
        ax.plot(
            data["eta_span"],
            mean_span_profile(data[profile_key]),
            color=series_color(idx),
            linewidth=0.8,
            marker="o",
            markersize=2.5,
            markeredgewidth=0.3,
            markerfacecolor=COLOUR_SCHEME["marker_face"],
            label=label_for(data, setting),
        )

    ax.set_xlabel(r"$y/b$")

    if coeff.lower() == "cl":
        ax.set_ylabel(r"$\overline{C}_L(y/b)$")
    elif coeff.lower() == "cd":
        ax.set_ylabel(r"$\overline{C}_D(y/b)$")

    style_axes(ax)
    ax.legend(frameon=False, loc="best",
              handletextpad=0.3, labelspacing=0.2)
    save_show(fig, output_dir, f"{coeff}_{setting}_span.pdf")


def plot_dx_dy_existing(results, output_dir):
    unique_nx = sorted({data["nx"] for data in results.values()})
    unique_ny = sorted({data["ny"] for data in results.values()})
    nx_colors = {nx: series_color(i) for i, nx in enumerate(unique_nx)}
    for group_idx, ny_group in enumerate((unique_ny[:3], unique_ny[3:]), start=1):
        if not ny_group:
            continue
        fig, axes = plt.subplots(1, len(ny_group), figsize=(3.5 * len(ny_group), 3), dpi=300, sharex=True, sharey=True)
        axes = [axes] if len(ny_group) == 1 else axes
        for ax, ny in zip(axes, ny_group):
            for nx in unique_nx:
                case_name = f"n_x_{nx}_n_y_{ny}"
                if case_name not in results or not results[case_name].get("has_tip"):
                    continue
                data = results[case_name]
                step = max(1, len(data["t_nd"]) // 20)
                ax.plot(data["t_nd"], data["tip_nd"], color=nx_colors[nx], linewidth=0.8,
                        marker="o", markevery=step, markersize=2.5, markeredgewidth=0.3,
                        markerfacecolor=COLOUR_SCHEME["marker_face"], label=f"nx={nx}")
            ax.set_title(f"ny = {ny}")
            ax.set_xlabel(r"$tU_\infty/c_{\mathrm{ref}}$")
            ax.set_ylabel(r"$s_{\mathrm{tip}}/c_{\mathrm{ref}}$")
            style_axes(ax)
            ax.legend(frameon=False, loc="best", handletextpad=0.3, labelspacing=0.2)
        save_show(fig, output_dir, f"tip_disp_dx_dy_ny_group{group_idx}.pdf")
    ny_colors = {ny: series_color(i) for i, ny in enumerate(unique_ny)}
    for nx in unique_nx:
        fig, ax = plt.subplots(figsize=(4, 3), dpi=300)
        for ny in unique_ny:
            case_name = f"n_x_{nx}_n_y_{ny}"
            if case_name not in results or not results[case_name].get("has_tip"):
                continue
            data = results[case_name]
            step = max(1, len(data["t_nd"]) // 20)
            ax.plot(data["t_nd"], data["tip_nd"], color=ny_colors[ny], linewidth=0.8,
                    marker="o", markevery=step, markersize=2.5, markeredgewidth=0.3,
                    markerfacecolor=COLOUR_SCHEME["marker_face"], label=f"ny={ny}")
        ax.set_title(f"nx = {nx} (all ny)")
        ax.set_xlabel(r"$tU_\infty/c_{\mathrm{ref}}$")
        ax.set_ylabel(r"$s_{\mathrm{tip}}/c_{\mathrm{ref}}$")
        style_axes(ax)
        ax.legend(frameon=False, loc="best", handletextpad=0.3, labelspacing=0.2)
        save_show(fig, output_dir, f"tip_disp_dx_dy_nx_{nx}.pdf")


def run_setting(setting):
    output_dir = OUTPUT_DIR / setting
    output_dir.mkdir(parents=True, exist_ok=True)
    results = load_cases(setting)
    if setting == "dx_dy":
        plot_dx_dy_existing(results, output_dir)
    else:
        plot_tip_overlay(results, setting, output_dir)
        plot_tip_subplots(results, setting, output_dir)
    plot_tip_force(results, setting, output_dir)
    for coeff in ("cl", "cd"):
        plot_coeff_time(results, setting, coeff, output_dir)
        plot_coeff_span(results, setting, coeff, output_dir)


for setting in SETTINGS:
    run_setting(setting)
