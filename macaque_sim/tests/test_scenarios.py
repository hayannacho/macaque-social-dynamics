"""
test_scenarios.py — Unit tests for config/scenarios.py.
 
Covers:
  - All four factory functions return well-formed ExperimentScenario objects.
  - param_overrides are non-empty where expected.
  - Sweep tuples have the correct (start, stop, num_steps) structure.
  - validate() catches malformed scenarios.
  - JSON round-trip (save_scenario → load_scenario) preserves all fields exactly.
  - Convenience helpers: is_sweep(), sweep_keys(), fixed_keys().
 
Run with:  python -m pytest tests/test_scenarios.py -v
"""
 
import json
import tempfile
from pathlib import Path

import pytest

from macaque_sim.config.scenarios import (
    ExperimentScenario,
    load_scenario,
    save_scenario,
    simconfig_field_names,
    scenario_noise,
    scenario_grooming_bias,
    scenario_decay,
    scenario_alpha_removal,
)
 
 
# ============================================================
#  Helpers
# ============================================================
 
ALL_FACTORIES = [
    scenario_noise,
    scenario_grooming_bias,
    scenario_decay,
    scenario_alpha_removal,
]
 
EXPECTED_NAMES = {
    scenario_noise:          "noise_sweep",
    scenario_grooming_bias:  "grooming_bias",
    scenario_decay:          "decay_sweep",
    scenario_alpha_removal:  "alpha_removal",
}
 
 
# ============================================================
#  Factory: well-formed output
# ============================================================
 
class TestFactoryOutputs:
 
    @pytest.mark.parametrize("factory", ALL_FACTORIES)
    def test_returns_experiment_scenario(self, factory):
        assert isinstance(factory(), ExperimentScenario)
 
    @pytest.mark.parametrize("factory", ALL_FACTORIES)
    def test_name_is_non_empty_string(self, factory):
        s = factory()
        assert isinstance(s.name, str) and s.name.strip()
 
    @pytest.mark.parametrize("factory", ALL_FACTORIES)
    def test_description_is_non_empty_string(self, factory):
        s = factory()
        assert isinstance(s.description, str) and s.description.strip()
 
    @pytest.mark.parametrize("factory", ALL_FACTORIES)
    def test_num_mc_runs_positive_int(self, factory):
        s = factory()
        assert isinstance(s.num_mc_runs, int) and s.num_mc_runs >= 1
 
    @pytest.mark.parametrize("factory", ALL_FACTORIES)
    def test_perturbations_is_list(self, factory):
        s = factory()
        assert isinstance(s.perturbations, list)
 
    @pytest.mark.parametrize("factory", ALL_FACTORIES)
    def test_validate_passes(self, factory):
        """validate() must not raise for any built-in factory scenario."""
        factory().validate(valid_param_keys=simconfig_field_names())   # raises on failure
 
    @pytest.mark.parametrize("factory,expected", EXPECTED_NAMES.items())
    def test_correct_name(self, factory, expected):
        assert factory().name == expected
 
    def test_num_mc_runs_override(self):
        """Factory num_mc_runs parameter is respected."""
        assert scenario_noise(num_mc_runs=42).num_mc_runs == 42
 
 
# ============================================================
#  Factory: param_overrides content
# ============================================================
 
class TestParamOverrides:
 
    def test_noise_sweep_has_sigma(self):
        s = scenario_noise()
        assert "sigma" in s.param_overrides
 
    def test_noise_sweep_sigma_is_tuple(self):
        s = scenario_noise()
        assert s.is_sweep("sigma")
 
    def test_noise_sweep_sigma_range(self):
        start, stop, steps = scenario_noise().param_overrides["sigma"]
        assert start > 0.0
        assert stop  == 1.0
        assert steps >= 2
 
    def test_decay_sweep_has_lambda(self):
        s = scenario_decay()
        assert "lambda_decay" in s.param_overrides

    def test_decay_sweep_lambda_is_tuple(self):
        assert scenario_decay().is_sweep("lambda_decay")
 
    def test_decay_sweep_lambda_range(self):
        start, stop, steps = scenario_decay().param_overrides["lambda_decay"]
        assert start > 0        # λ must be positive (model constraint)
        assert stop  > start
        assert steps >= 2
 
    def test_grooming_bias_has_strategy(self):
        s = scenario_grooming_bias()
        assert "grooming_strategy" in s.param_overrides
 
    def test_grooming_bias_strategy_is_categorical_sweep(self):
        s = scenario_grooming_bias()
        val = s.param_overrides["grooming_strategy"]
        assert isinstance(val, tuple)
        assert set(val) == {"upward_biased", "coalition"}
        assert s.is_sweep("grooming_strategy")
 
    def test_alpha_removal_overrides_empty(self):
        """Alpha removal has no parameter sweep — perturbation only."""
        s = scenario_alpha_removal()
        assert s.param_overrides == {}
 
    def test_noise_sweep_keys_helper(self):
        s = scenario_noise()
        assert "sigma" in s.sweep_keys()
        assert "sigma" not in s.fixed_keys()
 
    def test_grooming_fixed_keys_helper(self):
        s = scenario_grooming_bias()
        assert "grooming_strategy" in s.sweep_keys()
        assert "grooming_strategy" not in s.fixed_keys()
 
 
# ============================================================
#  Factory: perturbations content
# ============================================================
 
class TestPerturbations:
 
    def test_alpha_removal_has_one_perturbation(self):
        assert len(scenario_alpha_removal().perturbations) == 1
 
    def test_alpha_removal_perturbation_keys(self):
        p = scenario_alpha_removal().perturbations[0]
        assert {"type", "time", "target", "params"} <= set(p.keys())
 
    def test_alpha_removal_type_string(self):
        p = scenario_alpha_removal().perturbations[0]
        assert p["type"] == "alpha_removal"
 
    def test_alpha_removal_sentinels(self):
        """time=-1 and target=-1 are sentinels resolved at runtime."""
        p = scenario_alpha_removal().perturbations[0]
        assert p["time"]   == -1
        assert p["target"] == -1
 
    def test_noise_no_perturbations(self):
        assert scenario_noise().perturbations == []
 
    def test_decay_no_perturbations(self):
        assert scenario_decay().perturbations == []
 
    def test_grooming_no_perturbations(self):
        assert scenario_grooming_bias().perturbations == []
 
 
# ============================================================
#  validate() — catches bad inputs
# ============================================================
 
class TestValidate:
 
    def test_empty_name_raises(self):
        s = scenario_noise()
        s.name = ""
        with pytest.raises(ValueError, match="name"):
            s.validate()
 
    def test_whitespace_name_raises(self):
        s = scenario_noise()
        s.name = "   "
        with pytest.raises(ValueError, match="name"):
            s.validate()
 
    def test_empty_description_raises(self):
        s = scenario_noise()
        s.description = ""
        with pytest.raises(ValueError, match="description"):
            s.validate()
 
    def test_zero_mc_runs_raises(self):
        s = scenario_noise()
        s.num_mc_runs = 0
        with pytest.raises(ValueError, match="num_mc_runs"):
            s.validate()
 
    def test_bad_sweep_length_raises(self):
        s = scenario_noise()
        s.param_overrides["sigma"] = (0.0, 1.0)   # missing num_steps
        with pytest.raises(ValueError, match="sigma"):
            s.validate()
 
    def test_bad_num_steps_raises(self):
        s = scenario_noise()
        s.param_overrides["sigma"] = (0.0, 1.0, 0)   # num_steps=0 invalid
        with pytest.raises(ValueError, match="num_steps"):
            s.validate()
 
    def test_perturbation_missing_key_raises(self):
        s = scenario_alpha_removal()
        del s.perturbations[0]["type"]
        with pytest.raises(ValueError, match="type"):
            s.validate()

    def test_unknown_param_override_key_raises_against_simconfig(self):
        s = scenario_noise()
        s.param_overrides["not_a_real_simconfig_key"] = 1.23
        with pytest.raises(ValueError, match="Unknown param_overrides key"):
            s.validate(valid_param_keys=simconfig_field_names())

    def test_all_factory_override_keys_exist_in_simconfig(self):
        valid = simconfig_field_names()
        for factory in ALL_FACTORIES:
            s = factory()
            assert set(s.param_overrides.keys()) <= valid
 
 
# ============================================================
#  JSON round-trip
# ============================================================
 
class TestJsonRoundTrip:
 
    @pytest.mark.parametrize("factory", ALL_FACTORIES)
    def test_round_trip_preserves_name(self, factory, tmp_path):
        s = factory()
        path = tmp_path / f"{s.name}.json"
        save_scenario(s, path)
        loaded = load_scenario(path)
        assert loaded.name == s.name
 
    @pytest.mark.parametrize("factory", ALL_FACTORIES)
    def test_round_trip_preserves_description(self, factory, tmp_path):
        s = factory()
        path = tmp_path / f"{s.name}.json"
        save_scenario(s, path)
        loaded = load_scenario(path)
        assert loaded.description == s.description
 
    @pytest.mark.parametrize("factory", ALL_FACTORIES)
    def test_round_trip_preserves_num_mc_runs(self, factory, tmp_path):
        s = factory()
        path = tmp_path / f"{s.name}.json"
        save_scenario(s, path)
        loaded = load_scenario(path)
        assert loaded.num_mc_runs == s.num_mc_runs
 
    def test_round_trip_sweep_tuple_restored(self, tmp_path):
        """Sweep tuples stored as JSON lists are restored as tuples on load."""
        s = scenario_noise()
        path = tmp_path / "noise.json"
        save_scenario(s, path)
        loaded = load_scenario(path)
        assert loaded.is_sweep("sigma")
        assert isinstance(loaded.param_overrides["sigma"], tuple)
 
    def test_round_trip_sweep_values_preserved(self, tmp_path):
        s = scenario_noise()
        path = tmp_path / "noise.json"
        save_scenario(s, path)
        loaded = load_scenario(path)
        assert loaded.param_overrides["sigma"] == s.param_overrides["sigma"]
 
    def test_round_trip_perturbations_preserved(self, tmp_path):
        s = scenario_alpha_removal()
        path = tmp_path / "alpha.json"
        save_scenario(s, path)
        loaded = load_scenario(path)
        assert loaded.perturbations == s.perturbations
 
    def test_load_nonexistent_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_scenario(tmp_path / "does_not_exist.json")
 
    def test_saved_file_is_valid_json(self, tmp_path):
        """The output file must be parseable as plain JSON."""
        s = scenario_decay()
        path = tmp_path / "decay.json"
        save_scenario(s, path)
        with open(path) as f:
            data = json.load(f)
        assert "name" in data
        assert "param_overrides" in data
 