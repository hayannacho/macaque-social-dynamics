# results_io.py — Serialize / deserialize ExperimentResults for S4-T1 checkpoints
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from macaque_sim.experiments.monte_carlo import ExperimentCell, ExperimentResults, TrialResult

FORMAT_VERSION = 1


def _jsonify_val(v: Any) -> Any:
    if hasattr(v, "item") and callable(v.item):
        try:
            return v.item()
        except (ValueError, AttributeError):
            pass
    if isinstance(v, float):
        return float(v)
    if isinstance(v, (int, str, bool)) or v is None:
        return v
    if isinstance(v, dict):
        return {str(k): _jsonify_val(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_jsonify_val(x) for x in v]
    return str(v)


def experiment_results_to_dict(results: ExperimentResults) -> dict[str, Any]:
    cells_out: list[dict[str, Any]] = []
    for coords in sorted(results.cells.keys(), key=lambda t: t):
        cell = results.cells[coords]
        cells_out.append(
            {
                "coords": list(coords),
                "param_values": {k: _jsonify_val(v) for k, v in cell.param_values.items()},
                "runs": [
                    {
                        "alpha_survived": r.alpha_survived,
                        "num_swaps": int(r.num_swaps),
                        "final_ranks": [int(x) for x in r.final_ranks],
                    }
                    for r in cell.runs
                ],
            }
        )
    sweep_values = {
        k: [_jsonify_val(x) for x in vals] for k, vals in results.sweep_values.items()
    }
    return {
        "format_version": FORMAT_VERSION,
        "scenario_name": results.scenario_name,
        "sweep_keys": list(results.sweep_keys),
        "sweep_values": sweep_values,
        "shape": list(results.shape),
        "elapsed_seconds": float(results.elapsed_seconds),
        "cells": cells_out,
    }


def experiment_results_from_dict(d: dict[str, Any]) -> ExperimentResults:
    if int(d.get("format_version", 0)) != FORMAT_VERSION:
        raise ValueError(f"Unsupported results format_version: {d.get('format_version')!r}")
    cells: dict[tuple[int, ...], ExperimentCell] = {}
    for c in d["cells"]:
        coords = tuple(int(x) for x in c["coords"])
        runs = [
            TrialResult(
                alpha_survived=bool(r["alpha_survived"]),
                num_swaps=int(r["num_swaps"]),
                final_ranks=[int(x) for x in r["final_ranks"]],
                raw_logger=None,
            )
            for r in c["runs"]
        ]
        cells[coords] = ExperimentCell(
            coords=coords,
            param_values=dict(c["param_values"]),
            runs=runs,
        )
    sweep_values: dict[str, list[Any]] = {
        k: list(v) for k, v in d.get("sweep_values", {}).items()
    }
    return ExperimentResults(
        scenario_name=str(d["scenario_name"]),
        sweep_keys=list(d["sweep_keys"]),
        sweep_values=sweep_values,
        shape=tuple(int(x) for x in d["shape"]),
        cells=cells,
        elapsed_seconds=float(d.get("elapsed_seconds", 0.0)),
    )


def save_experiment_results(path: str | Path, results: ExperimentResults) -> None:
    """Atomically write JSON results (strip raw logger payloads)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(experiment_results_to_dict(results), indent=2)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(payload, encoding="utf-8")
    tmp.replace(path)


def load_experiment_results(path: str | Path) -> ExperimentResults:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(str(path))
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        raise ValueError(f"Empty or corrupted results file: {path}")
    return experiment_results_from_dict(json.loads(text))


def verify_experiment_file(
    path: str | Path,
    *,
    expected_scenario_name: str,
    expected_num_mc_runs: int,
    expected_num_cells: int | None = None,
) -> None:
    """Raise ValueError if file is missing data, wrong grid size, or wrong run counts."""
    r = load_experiment_results(path)
    if r.scenario_name != expected_scenario_name:
        raise ValueError(f"{path}: expected scenario {expected_scenario_name!r}, got {r.scenario_name!r}")
    if not r.cells:
        raise ValueError(f"{path}: no cells")
    if expected_num_cells is not None and len(r.cells) != expected_num_cells:
        raise ValueError(
            f"{path}: expected {expected_num_cells} grid cells, got {len(r.cells)} (incomplete sweep?)"
        )
    for coords, cell in r.cells.items():
        if len(cell.runs) != expected_num_mc_runs:
            raise ValueError(
                f"{path}: cell {coords} has {len(cell.runs)} runs, expected {expected_num_mc_runs}"
            )
