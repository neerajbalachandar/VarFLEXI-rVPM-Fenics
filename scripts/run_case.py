#!/usr/bin/env python3
"""Run coupled VarFlExI/FLOWUnsteady parameter sweeps from one entry point."""

from __future__ import annotations

import argparse
import copy
import inspect
import os
import runpy
import shutil
import subprocess
import sys
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_DIR = REPO_ROOT / "config"
DEFAULT_RESULTS_ROOT = REPO_ROOT / "results" / "parameter_sweeps"
CONFIG_FILES = {
    "fluid": "fluid_params.yaml",
    "solid": "solid_params.yaml",
    "coupling": "coupling_params.yaml",
}


@dataclass(frozen=True)
class CaseSpec:
    name: str
    fluid_updates: dict[str, Any] | None = None
    solid_updates: dict[str, Any] | None = None
    coupling_updates: dict[str, Any] | None = None


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def dump_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False)


def apply_update(data: dict[str, Any], key_path: str, value: Any) -> None:
    keys = key_path.split(".")
    cursor: Any = data
    for key in keys[:-1]:
        if key not in cursor or not isinstance(cursor[key], dict):
            cursor[key] = {}
        cursor = cursor[key]
    cursor[keys[-1]] = value


def sanitize_name(value: Any) -> str:
    text = str(value).strip().replace(" ", "_")
    for char in ("/", "\\", ":", "=", ","):
        text = text.replace(char, "_")
    return text


def relative_to_repo(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT)).replace(os.sep, "/")
    except ValueError:
        return str(path)


def config_paths(config_dir: Path) -> dict[str, Path]:
    return {key: config_dir / filename for key, filename in CONFIG_FILES.items()}


def load_configs(config_dir: Path) -> dict[str, dict[str, Any]]:
    return {key: load_yaml(path) for key, path in config_paths(config_dir).items()}


def snapshot_config_text(config_dir: Path) -> dict[Path, bytes]:
    return {path: path.read_bytes() for path in config_paths(config_dir).values() if path.exists()}


def restore_config_text(snapshot: dict[Path, bytes]) -> None:
    for path, payload in snapshot.items():
        path.write_bytes(payload)


def write_configs(config_dir: Path, configs: dict[str, dict[str, Any]]) -> None:
    for key, filename in CONFIG_FILES.items():
        dump_yaml(config_dir / filename, configs[key])


def ensure_unique_dir(path: Path) -> Path:
    if not path.exists() or not any(path.iterdir()):
        return path

    suffix = 2
    while True:
        candidate = path.with_name(f"{path.name}_{suffix:02d}")
        if not candidate.exists():
            return candidate
        suffix += 1


def copy_run_configs(config_dir: Path, run_config_dir: Path) -> None:
    run_config_dir.mkdir(parents=True, exist_ok=True)
    for filename in CONFIG_FILES.values():
        shutil.copy2(config_dir / filename, run_config_dir / filename)


def write_manifest(run_dir: Path, manifest: dict[str, Any]) -> None:
    with (run_dir / "run_manifest.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(manifest, handle, sort_keys=False)


def normalize_case_updates(
    fluid_updates: dict[str, Any] | None,
    solid_updates: dict[str, Any] | None,
    coupling_updates: dict[str, Any] | None,
) -> dict[str, dict[str, Any]]:
    return {
        "fluid": dict(fluid_updates or {}),
        "solid": dict(solid_updates or {}),
        "coupling": dict(coupling_updates or {}),
    }


def apply_case_updates(configs: dict[str, dict[str, Any]], updates: dict[str, dict[str, Any]]) -> None:
    for config_key, config_updates in updates.items():
        if config_key not in configs:
            raise KeyError(f"Unknown config group: {config_key}")
        for key_path, value in config_updates.items():
            apply_update(configs[config_key], key_path, value)

    fluid_updates = updates.get("fluid", {})
    if "n_span" in fluid_updates:
        span = fluid_updates["n_span"]
        if "n_span_comm" not in fluid_updates:
            configs["fluid"]["n_span_comm"] = span
        if "comm_nspan" not in updates.get("coupling", {}):
            configs["coupling"]["comm_nspan"] = span


def infer_study_name() -> str:
    run_case_path = Path(__file__).resolve()
    for frame in inspect.stack()[1:]:
        caller = Path(frame.filename).resolve()
        if caller != run_case_path:
            return caller.stem
    return "manual"


def run_and_tee(command: list[str], cwd: Path, log_path: Path) -> None:
    with log_path.open("w", encoding="utf-8") as log_handle:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log_handle.write(line)
        returncode = process.wait()

    if returncode != 0:
        raise subprocess.CalledProcessError(returncode, command)


def run_case(
    case_name: str,
    fluid_updates: dict[str, Any] | None = None,
    solid_updates: dict[str, Any] | None = None,
    coupling_updates: dict[str, Any] | None = None,
    *,
    study_name: str | None = None,
    config_dir: Path | str = DEFAULT_CONFIG_DIR,
    results_root: Path | str = DEFAULT_RESULTS_ROOT,
    dry_run: bool = False,
) -> Path:
    """Run one simulation case and save configs, manifest, log, and solver output together."""

    config_dir = Path(config_dir).expanduser().resolve()
    results_root = Path(results_root).expanduser().resolve()
    study_name = study_name or infer_study_name()
    study_dir = results_root / sanitize_name(study_name)
    run_dir = study_dir / sanitize_name(case_name)
    run_results_dir = run_dir / "results"
    run_config_dir = run_dir / "config"
    run_dir.mkdir(parents=True, exist_ok=True)
    run_results_dir.mkdir(parents=True, exist_ok=True)

    baseline_text = snapshot_config_text(config_dir)
    baseline_configs = load_configs(config_dir)
    configs_to_run = copy.deepcopy(baseline_configs)
    updates = normalize_case_updates(fluid_updates, solid_updates, coupling_updates)
    apply_case_updates(configs_to_run, updates)

    output_root = relative_to_repo(run_results_dir)
    configs_to_run["fluid"]["save_directory"] = f"{output_root}/fluid"
    configs_to_run["solid"]["results_root"] = output_root
    configs_to_run["coupling"]["results_root"] = output_root

    manifest = {
        "study_name": study_name,
        "case_name": case_name,
        "run_dir": str(run_dir),
        "updates": updates,
        "status": "dry_run" if dry_run else "prepared",
    }

    try:
        write_configs(config_dir, configs_to_run)
        copy_run_configs(config_dir, run_config_dir)
        write_manifest(run_dir, manifest)

        print(f"[{study_name}] prepared {case_name}: {relative_to_repo(run_dir)}")
        if dry_run:
            return run_dir

        run_and_tee(["bash", "run.sh"], REPO_ROOT, run_dir / "run.log")

        manifest["status"] = "completed"
        write_manifest(run_dir, manifest)
        print(f"[{study_name}] completed {case_name}")
        return run_dir
    except subprocess.CalledProcessError as exc:
        manifest["status"] = "failed"
        manifest["returncode"] = exc.returncode
        write_manifest(run_dir, manifest)
        print(f"[{study_name}] failed {case_name}; see {relative_to_repo(run_dir / 'run.log')}", file=sys.stderr)
        raise
    finally:
        restore_config_text(baseline_text)


def built_in_studies() -> dict[str, list[CaseSpec]]:
    total_time = 1.0
    dt_values = [0.01, 0.001, 0.0005]

    return {
        "dt": [
            CaseSpec(
                name=f"dt_{sanitize_name(dt)}",
                coupling_updates={"total_time": total_time, "n_steps": int(total_time / dt)},
            )
            for dt in dt_values
        ],
        "tip_displacement": [
            CaseSpec(
                name=f"dt_{sanitize_name(dt)}",
                coupling_updates={"total_time": total_time, "n_steps": int(total_time / dt)},
            )
            for dt in dt_values
        ],
        "dx_dy_solid": [
            CaseSpec(name=f"n_x_{nx}_n_y_{ny}", solid_updates={"nx": nx, "ny": ny})
            for nx, ny in product([10, 20, 40, 80], [40, 120, 240])
        ],
        "pperstep": [
            CaseSpec(name=f"particles_step_{p_step}", fluid_updates={"particles_per_step": p_step})
            for p_step in [1, 2, 3, 4]
        ],
        "crm_rbf": [
            CaseSpec(name=f"force_transfer_{mode}", coupling_updates={"force_transfer_mode": mode})
            for mode in ["rbf", "crm"]
        ],
        "dx_dy_fluid": [
            CaseSpec(name=f"n_span_{span}", fluid_updates={"n_span": span})
            for span in [20, 40, 80, 160]
        ],
        "dx_dy": [
            CaseSpec(name=f"n_span_{span}", fluid_updates={"n_span": span})
            for span in [10, 20, 40, 80]
        ],
    }


def discover_sweep_file(repo_root: Path, explicit: str | None) -> Path | None:
    if explicit:
        return Path(explicit).expanduser().resolve()

    candidates = [
        repo_root / "scripts" / "sweep_params.yaml",
        repo_root / "scripts" / "params.yaml",
        repo_root / "scripts" / "param_sweep.yaml",
        repo_root / "scripts" / "run_params.yaml",
    ]
    return next((candidate for candidate in candidates if candidate.exists()), None)


def parse_sweep_spec(raw_spec: Any) -> list[CaseSpec]:
    if raw_spec is None:
        return []

    if isinstance(raw_spec, dict) and "cases" in raw_spec:
        raw_cases = raw_spec["cases"]
        if not isinstance(raw_cases, list):
            raise ValueError("'cases' must be a list")
        return [
            CaseSpec(
                name=str(case["name"]),
                fluid_updates=case.get("fluid_updates") or case.get("fluid") or {},
                solid_updates=case.get("solid_updates") or case.get("solid") or {},
                coupling_updates=case.get("coupling_updates") or case.get("coupling") or {},
            )
            for case in raw_cases
        ]

    if isinstance(raw_spec, dict) and "params" in raw_spec:
        params = raw_spec["params"]
        if not isinstance(params, dict):
            raise ValueError("'params' must map config groups to update dictionaries")

        groups: list[dict[str, dict[str, Any]]] = [{"fluid": {}, "solid": {}, "coupling": {}}]
        for group_name, updates in params.items():
            if group_name not in groups[0]:
                raise ValueError(f"Unknown parameter group '{group_name}'. Use fluid, solid, or coupling.")
            if not isinstance(updates, dict):
                raise ValueError(f"Parameter group '{group_name}' must be a dictionary")

            keys = list(updates.keys())
            values = [value if isinstance(value, list) else [value] for value in updates.values()]
            next_groups: list[dict[str, dict[str, Any]]] = []
            for existing in groups:
                for combination in product(*values):
                    candidate = copy.deepcopy(existing)
                    for key, value in zip(keys, combination):
                        candidate[group_name][str(key)] = value
                    next_groups.append(candidate)
            groups = next_groups

        cases = []
        for index, updates in enumerate(groups, start=1):
            label_parts = []
            for group_name, group_updates in updates.items():
                for key, value in group_updates.items():
                    label_parts.append(f"{group_name}_{key}_{sanitize_name(value)}")
            name = f"case_{index:03d}_{'_'.join(label_parts)}" if label_parts else f"case_{index:03d}"
            cases.append(
                CaseSpec(
                    name=name,
                    fluid_updates=updates["fluid"],
                    solid_updates=updates["solid"],
                    coupling_updates=updates["coupling"],
                )
            )
        return cases

    raise ValueError("Sweep YAML must contain either 'cases' or 'params'")


def selected_studies(selection: str) -> dict[str, list[CaseSpec]]:
    studies = built_in_studies()
    if selection == "all":
        return studies
    if selection not in studies:
        valid = ", ".join(["all", *studies.keys()])
        raise ValueError(f"Unknown study '{selection}'. Valid choices: {valid}")
    return {selection: studies[selection]}


def solver_studies() -> dict[str, Path]:
    solver_dir = REPO_ROOT / "scripts" / "solver"
    return {
        path.stem: path
        for path in sorted(solver_dir.glob("*.py"))
        if not path.name.startswith("_")
    }


def run_solver_file(path: Path) -> None:
    runpy.run_path(str(path), run_name="__main__")


def main() -> None:
    studies = solver_studies()
    parser = argparse.ArgumentParser(description="Run coupled solver parameter sweeps")
    parser.add_argument(
        "--study",
        default=None,
        choices=["all", *studies.keys()],
        help="Solver sweep file to run from scripts/solver/. Use all to run every solver file.",
    )
    parser.add_argument("--config-dir", default=str(DEFAULT_CONFIG_DIR), help="Directory containing YAML config files")
    parser.add_argument("--results-root", default=str(DEFAULT_RESULTS_ROOT), help="Root folder for sweep results")
    parser.add_argument("--sweep-file", default=None, help="Optional YAML file with custom cases or params")
    parser.add_argument("--dry-run", action="store_true", help="Prepare folders/configs without launching run.sh")
    parser.add_argument("--list-studies", action="store_true", help="Print available built-in studies and exit")
    args = parser.parse_args()

    if args.list_studies:
        for name, path in studies.items():
            print(f"{name}: {relative_to_repo(path)}")
        return

    config_dir = Path(args.config_dir)
    results_root = Path(args.results_root)
    sweep_file = discover_sweep_file(REPO_ROOT, args.sweep_file) if args.sweep_file else None

    if sweep_file:
        cases = parse_sweep_spec(load_yaml(sweep_file))
        if not cases:
            raise ValueError(f"No cases found in sweep file: {sweep_file}")
        for case in cases:
            run_case(
                case.name,
                case.fluid_updates,
                case.solid_updates,
                case.coupling_updates,
                study_name=sweep_file.stem,
                config_dir=config_dir,
                results_root=results_root,
                dry_run=args.dry_run,
            )
        return

    if args.study is None:
        parser.print_help()
        print("\nChoose a solver file with --study, or run the file directly, for example:")
        print("  python3 scripts/solver/dx_dy_solid.py")
        return

    if args.dry_run:
        print("--dry-run is only supported for run_case(...) calls and --sweep-file cases.")
        print("Run the solver file directly after setting dry_run=True in its run_case(...) call if needed.")
        return

    if args.study == "all":
        for path in studies.values():
            run_solver_file(path)
        return

    run_solver_file(studies[args.study])


if __name__ == "__main__":
    main()
