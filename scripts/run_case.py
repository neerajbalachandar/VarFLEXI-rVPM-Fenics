#!/usr/bin/env python3
"""Run one or more coupled cases by updating YAML config files and launching the project runner."""

from __future__ import annotations

import argparse
import copy
import os
import shutil
import subprocess
import sys
from itertools import product
from pathlib import Path
from typing import Any

import yaml


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
    return str(value).replace(" ", "_").replace("/", "_").replace("\\", "_")


def discover_sweep_file(repo_root: Path, explicit: str | None) -> Path | None:
    if explicit:
        return Path(explicit).expanduser().resolve()

    candidates = [
        repo_root / "scripts" / "sweep_params.yaml",
        repo_root / "scripts" / "params.yaml",
        repo_root / "scripts" / "param_sweep.yaml",
        repo_root / "scripts" / "run_params.yaml",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def parse_sweep_spec(raw_spec: Any) -> list[dict[str, dict[str, Any]]]:
    if raw_spec is None:
        return [{}]

    if isinstance(raw_spec, list):
        combos: list[dict[str, dict[str, Any]]] = []
        for entry in raw_spec:
            if isinstance(entry, dict) and "file" in entry and "key" in entry and "values" in entry:
                target_file = entry.get("file")
                key_path = entry.get("key")
                values = entry.get("values") or []
                if target_file is None or key_path is None:
                    continue
                if not isinstance(values, list):
                    values = [values]
                for value in values:
                    combos.append({str(target_file): {str(key_path): value}})
            else:
                raise ValueError("Each sweep entry must contain file, key, and values")
        return combos

    if isinstance(raw_spec, dict):
        if "sweep" in raw_spec:
            return parse_sweep_spec(raw_spec["sweep"])

        if "params" in raw_spec:
            params = raw_spec["params"]
            if isinstance(params, dict):
                grouped: list[dict[str, dict[str, Any]]] = [{}]
                for target_file, updates in params.items():
                    if not isinstance(updates, dict):
                        raise ValueError("Sweep params must map files to dictionaries of updates")
                    current_grouped: list[dict[str, dict[str, Any]]] = []
                    for existing in grouped:
                        key_names = []
                        values_per_key: list[tuple[str, list[Any]]] = []
                        for key_path, candidate_values in updates.items():
                            if isinstance(candidate_values, list):
                                values = candidate_values
                            else:
                                values = [candidate_values]
                            key_names.append(str(key_path))
                            values_per_key.append((str(key_path), values))
                        if not values_per_key:
                            current_grouped.append(existing)
                            continue
                        for combination in product(*[values for _, values in values_per_key]):
                            updated = copy.deepcopy(existing)
                            file_updates = updated.setdefault(str(target_file), {})
                            for (key_path, _), value in zip(values_per_key, combination):
                                file_updates[key_path] = value
                            current_grouped.append(updated)
                    grouped = current_grouped
                return grouped

        if isinstance(raw_spec.get("file"), str) and isinstance(raw_spec.get("key"), str):
            values = raw_spec.get("values") or []
            if not isinstance(values, list):
                values = [values]
            return [{raw_spec["file"]: {raw_spec["key"]: value}} for value in values]

        # Fallback: treat the dict as a single set of updates for a single file.
        return [raw_spec]

    raise ValueError("Unsupported sweep spec format")


def build_run_label(updates: dict[str, dict[str, Any]], index: int) -> str:
    if not updates:
        return f"run_{index:03d}"

    labels: list[str] = []
    for file_name, file_updates in sorted(updates.items()):
        for key_path, value in sorted(file_updates.items()):
            labels.append(f"{file_name}_{key_path}_{sanitize_name(value)}")
    return f"run_{index:03d}_{'_'.join(labels)}"


def write_config_files(config_dir: Path, configs: dict[str, dict[str, Any]]) -> None:
    for filename, payload in configs.items():
        dump_yaml(config_dir / filename, payload)


def move_results_into_run_dir(repo_root: Path, run_dir: Path) -> None:
    run_results_dir = run_dir / "results"
    run_results_dir.mkdir(parents=True, exist_ok=True)

    source_results_dir = repo_root / "results"
    if not source_results_dir.exists():
        return

    for child in source_results_dir.iterdir():
        destination = run_results_dir / child.name
        if destination.exists():
            shutil.rmtree(destination)
        shutil.move(str(child), str(destination))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the coupled solver workflow and optionally sweep YAML parameters")
    parser.add_argument("--config-dir", default=None, help="Directory containing the YAML config files")
    parser.add_argument("--sweep-file", default=None, help="YAML file describing sweep parameters under scripts/")
    parser.add_argument("--results-root", default=None, help="Root folder for storing per-run results")
    parser.add_argument("--dry-run", action="store_true", help="Show the planned runs without launching the solver")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    config_dir = Path(args.config_dir).expanduser().resolve() if args.config_dir else repo_root / "config"
    results_root = Path(args.results_root).expanduser().resolve() if args.results_root else repo_root / "results" / "runs"
    sweep_file = discover_sweep_file(repo_root, args.sweep_file)

    config_files = {
        "fluid_params.yaml": config_dir / "fluid_params.yaml",
        "solid_params.yaml": config_dir / "solid_params.yaml",
        "coupling_params.yaml": config_dir / "coupling_params.yaml",
    }

    baseline_configs = {name: load_yaml(path) for name, path in config_files.items()}
    sweep_specs = parse_sweep_spec(load_yaml(sweep_file) if sweep_file else None)
    if not sweep_specs:
        sweep_specs = [{}]

    results_root.mkdir(parents=True, exist_ok=True)

    for index, update_spec in enumerate(sweep_specs, start=1):
        run_label = build_run_label(update_spec, index)
        run_dir = results_root / run_label
        run_dir.mkdir(parents=True, exist_ok=True)
        run_config_dir = run_dir / "config"
        run_config_dir.mkdir(parents=True, exist_ok=True)
        run_results_dir = run_dir / "results"
        run_results_dir.mkdir(parents=True, exist_ok=True)

        configs_to_run = copy.deepcopy(baseline_configs)
        for file_name, file_updates in update_spec.items():
            target_path = config_files.get(file_name)
            if target_path is None:
                raise FileNotFoundError(f"Unknown config file in sweep spec: {file_name}")
            target_payload = configs_to_run[file_name]
            for key_path, value in file_updates.items():
                apply_update(target_payload, key_path, value)

        # Make each run write its own output folder.
        run_output_root = str(run_dir.relative_to(repo_root)).replace(os.sep, "/")
        fluid_payload = configs_to_run["fluid_params.yaml"]
        solid_payload = configs_to_run["solid_params.yaml"]
        fluid_payload["save_directory"] = f"{run_output_root}/results/fluid"
        solid_payload["results_root"] = f"{run_output_root}/results"

        write_config_files(config_dir, configs_to_run)

        # Copy the YAML files into the run results folder.
        for file_name in config_files:
            shutil.copy2(config_dir / file_name, run_config_dir / file_name)

        manifest = {
            "run_label": run_label,
            "sweep_file": str(sweep_file) if sweep_file else None,
            "updates": update_spec,
        }
        with (run_dir / "run_manifest.yaml").open("w", encoding="utf-8") as handle:
            yaml.safe_dump(manifest, handle, sort_keys=False)

        print(f"Prepared run {run_label} in {run_dir}")
        if args.dry_run:
            print("Dry run enabled; skipping launch of run.sh")
            continue

        try:
            subprocess.run(["bash", "run.sh"], cwd=repo_root, check=True)
        finally:
            move_results_into_run_dir(repo_root, run_dir)
            # Restore the repository config after each run so the next iteration starts cleanly.
            write_config_files(config_dir, baseline_configs)


if __name__ == "__main__":
    main()
