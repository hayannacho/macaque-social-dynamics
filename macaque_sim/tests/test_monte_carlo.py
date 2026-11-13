"""Unit tests for Sprint 2 Monte Carlo runner skeleton."""

import json
from pathlib import Path

from macaque_sim.config.scenarios import ExperimentScenario
from macaque_sim.experiments.monte_carlo import (
    ExperimentCell,
    ExperimentResults,
    TrialResult,
    expand_param_grid,
    run_experiment,
)
from macaque_sim.experiments.results_io import (
    experiment_results_from_dict,
    experiment_results_to_dict,
    load_experiment_results,
    save_experiment_results,
)


def dummy_sim_fn(config: dict) -> dict:
    """Synthetic logger payload used for Sprint 2 skeleton tests."""
    sigma = float(config.get("sigma", 0.1))
    lambda_decay = float(config.get("lambda_decay", 0.1))
    seed = int(config.get("random_seed", 0))

    alpha_survived = sigma < 0.5
    num_swaps = int(round((sigma + lambda_decay) * 10)) + (seed % 3)
    final_ranks = [0, 1, 2] if alpha_survived else [1, 0, 2]

    return {
        "alpha_survived": alpha_survived,
        "num_swaps": num_swaps,
        "final_ranks": final_ranks,
    }


def test_grid_expansion_5x5_produces_25_cells():
    _, _, grid_cells = expand_param_grid(
        {
            "sigma": (0.1, 0.5, 5),
            "lambda_decay": (0.1, 0.5, 5),
        }
    )
    assert len(grid_cells) == 25


def test_run_experiment_cell_contains_num_mc_runs_results():
    scenario = ExperimentScenario(
        name="s2_t7_test",
        description="Two-dimensional sweep",
        param_overrides={
            "sigma": (0.1, 0.5, 3),
            "lambda_decay": (0.1, 0.5, 2),
        },
        num_mc_runs=7,
    )
    result = run_experiment(scenario, dummy_sim_fn)
    assert isinstance(result, ExperimentResults)

    assert len(result) == 6
    for cell in result.cells.values():
        assert len(cell.runs) == 7
        for run_metric in cell.runs:
            assert isinstance(run_metric, TrialResult)
            assert isinstance(run_metric.alpha_survived, bool)
            assert isinstance(run_metric.num_swaps, int)
            assert isinstance(run_metric.final_ranks, list)


def test_results_indexable_by_grid_coordinates():
    scenario = ExperimentScenario(
        name="index_test",
        description="Indexing by tuple coordinates",
        param_overrides={
            "sigma": (0.1, 0.5, 4),
            "lambda_decay": (0.2, 0.6, 3),
        },
        num_mc_runs=2,
    )
    result = run_experiment(scenario, dummy_sim_fn)

    cell = result[(2, 1)]
    assert cell.coords == (2, 1)
    assert "sigma" in cell.param_values
    assert "lambda_decay" in cell.param_values
    assert len(cell.runs) == 2


def test_fixed_override_single_cell_still_runs():
    scenario = ExperimentScenario(
        name="fixed_only",
        description="No sweep",
        param_overrides={"sigma": 0.2, "lambda_decay": 0.3},
        num_mc_runs=3,
    )
    result = run_experiment(scenario, dummy_sim_fn)
    assert result.shape == (1,)
    assert len(result) == 1
    assert len(result[(0,)].runs) == 3


def test_experiment_results_json_roundtrip():
    cells = {
        (0,): ExperimentCell(
            coords=(0,),
            param_values={"sigma": 0.1},
            runs=[
                TrialResult(alpha_survived=True, num_swaps=1, final_ranks=[1, 2, 3]),
            ],
        ),
    }
    original = ExperimentResults(
        scenario_name="roundtrip",
        sweep_keys=["sigma"],
        sweep_values={"sigma": [0.1]},
        shape=(1,),
        cells=cells,
        elapsed_seconds=1.5,
    )
    blob = json.dumps(experiment_results_to_dict(original))
    restored = experiment_results_from_dict(json.loads(blob))
    assert restored.scenario_name == "roundtrip"
    assert restored[(0,)].alpha_survival_rate == 1.0
    assert len(restored[(0,)].runs) == 1
    assert restored[(0,)].runs[0].final_ranks == [1, 2, 3]


def test_run_experiment_checkpoint_resume(tmp_path: Path):
    scenario = ExperimentScenario(
        name="ckpt_test",
        description="checkpoint",
        param_overrides={"sigma": (0.1, 0.2, 2)},
        num_mc_runs=2,
    )
    out = tmp_path / "ckpt.json"
    run_experiment(scenario, dummy_sim_fn, checkpoint_path=out, resume=False)
    first = load_experiment_results(out)
    assert len(first.cells) == 2
    run_experiment(scenario, dummy_sim_fn, checkpoint_path=out, resume=True)
    second = load_experiment_results(out)
    assert len(second.cells) == 2
    for cell in second.cells.values():
        assert len(cell.runs) == 2


def test_save_load_experiment_results_file(tmp_path: Path):
    scenario = ExperimentScenario(
        name="file_io",
        description="io",
        param_overrides={"sigma": 0.1},
        num_mc_runs=1,
    )
    result = run_experiment(scenario, dummy_sim_fn)
    path = tmp_path / "r.json"
    save_experiment_results(path, result)
    loaded = load_experiment_results(path)
    assert loaded.scenario_name == "file_io"
    assert len(loaded[(0,)].runs) == 1