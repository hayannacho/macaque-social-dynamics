"""
test_config.py - Unit tests for config/sim_config.py.

Covers:
  - default_config() returns a valid SimConfig object.
  - load_config() overrides only the keys present in JSON/YAML files.
  - Type validation: invalid types raise TypeError.
  - Stability validation: invalid dt_sde values raise ValueError.
  - Unknown config fields raise KeyError.
  - YAML loading works correctly if PyYAML is installed.

Run with:
  pytest tests/test_config.py
"""

import json
import importlib.util
import pytest

from macaque_sim.config.sim_config import SimConfig, default_config, load_config

def test_default_config_returns_valid_object():
    cfg = default_config()
    assert isinstance(cfg, SimConfig)
    assert cfg.lambda_decay > 0
    assert cfg.sigma > 0
    assert cfg.epsilon > 0
    assert 30 <= cfg.N <= 50


def test_load_config_overrides_only_present_keys(tmp_path):
    cfg_file = tmp_path / "custom.json"
    cfg_file.write_text(
        json.dumps({
            "N": 40,
            "sigma": 0.08
        }),
        encoding="utf-8"
    )

    cfg = load_config(cfg_file)

    assert cfg.N == 40
    assert cfg.sigma == 0.08

    default = default_config()
    assert cfg.lambda_decay == default.lambda_decay
    assert cfg.epsilon == default.epsilon
    assert cfg.cooldown_duration == default.cooldown_duration


def test_invalid_type_raises(tmp_path):
    cfg_file = tmp_path / "bad.json"
    cfg_file.write_text(
        json.dumps({
            "N": "thirty"
        }),
        encoding="utf-8"
    )

    with pytest.raises(TypeError):
        load_config(cfg_file)


def test_invalid_dt_stability_raises(tmp_path):
    cfg_file = tmp_path / "bad_dt.json"
    cfg_file.write_text(
        json.dumps({
            "lambda_decay": 0.1,
            "dt_sde": 25.0
        }),
        encoding="utf-8"
    )

    with pytest.raises(ValueError):
        load_config(cfg_file)


def test_unknown_field_raises(tmp_path):
    cfg_file = tmp_path / "unknown.json"
    cfg_file.write_text(
        json.dumps({
            "banana": 123
        }),
        encoding="utf-8"
    )

    with pytest.raises(KeyError):
        load_config(cfg_file)


def test_load_yaml_overrides_only_present_keys(tmp_path):
    if importlib.util.find_spec("yaml") is None:
        pytest.skip("PyYAML is not installed.")

    cfg_file = tmp_path / "custom.yaml"
    cfg_file.write_text(
        "N: 40\nsigma: 0.08\n",
        encoding="utf-8"
    )

    cfg = load_config(cfg_file)

    assert cfg.N == 40
    assert cfg.sigma == 0.08

    default = default_config()
    assert cfg.lambda_decay == default.lambda_decay
    assert cfg.epsilon == default.epsilon
