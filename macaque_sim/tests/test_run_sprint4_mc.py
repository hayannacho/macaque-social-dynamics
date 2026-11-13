"""S4-T1: production MC driver + verification."""

import json
from pathlib import Path

import pytest

from macaque_sim.experiments.monte_carlo import run_experiment
from macaque_sim.experiments.results_io import save_experiment_results, verify_experiment_file
from macaque_sim.experiments.run_sprint4_mc import _JOBS, _select_jobs, main, verify_all
from macaque_sim.tests.test_monte_carlo import dummy_sim_fn


def test_verify_all_passes_for_dummy_checkpointed_sweeps(tmp_path: Path) -> None:
    """Full grid × 1 MC per cell, fast dummy sim — matches S4-T1 verify path."""
    num_mc = 1
    for _, factory, fname in _JOBS:
        scenario = factory(num_mc_runs=num_mc)
        run_experiment(
            scenario,
            dummy_sim_fn,
            checkpoint_path=tmp_path / fname,
            resume=False,
        )
    verify_all(tmp_path, num_mc)


def test_verify_experiment_file_rejects_wrong_cell_count(tmp_path: Path) -> None:
    from macaque_sim.experiments.monte_carlo import ExperimentCell, ExperimentResults, TrialResult

    one_cell = ExperimentResults(
        scenario_name="noise_sweep",
        sweep_keys=["sigma"],
        sweep_values={"sigma": [0.1]},
        shape=(1,),
        cells={
            (0,): ExperimentCell(
                coords=(0,),
                param_values={"sigma": 0.1},
                runs=[TrialResult(True, 0, [0, 1, 2])],
            ),
        },
        elapsed_seconds=0.0,
    )
    path = tmp_path / "noise_sweep.json"
    save_experiment_results(path, one_cell)
    with pytest.raises(ValueError, match="expected 11 grid cells"):
        verify_experiment_file(
            path,
            expected_scenario_name="noise_sweep",
            expected_num_mc_runs=1,
            expected_num_cells=11,
        )


def test_select_jobs_subset_and_errors() -> None:
    j = _select_jobs(["noise_sweep", "alpha_removal"])
    assert j is not None and len(j) == 2
    assert {row[0] for row in j} == {"noise_sweep", "alpha_removal"}
    assert _select_jobs(None) is None
    with pytest.raises(ValueError, match="Unknown"):
        _select_jobs(["not_a_key"])


def test_main_verify_only_zero(tmp_path: Path) -> None:
    num_mc = 1
    for _, factory, fname in _JOBS:
        scenario = factory(num_mc_runs=num_mc)
        run_experiment(scenario, dummy_sim_fn, checkpoint_path=tmp_path / fname, resume=False)
    code = main(["--verify-only", "--output-dir", str(tmp_path), "--num-mc-runs", str(num_mc)])
    assert code == 0


def test_run_all_summary_has_s4_fields(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import macaque_sim.experiments.run_sprint4_mc as rs4

    monkeypatch.setattr(rs4, "run_single_trial", dummy_sim_fn)
    rs4.run_all(
        output_dir=tmp_path,
        num_mc_runs=1,
        progress=False,
        resume=False,
        jobs=[_JOBS[0]],
        budget_wall_clock_hours=8.0,
    )
    summary = json.loads((tmp_path / "sprint4_runtime.json").read_text(encoding="utf-8"))
    assert summary["sprint_task"] == "S4-T1"
    assert "within_budget" in summary
    assert summary["experiments"][0]["num_grid_cells"] >= 1
