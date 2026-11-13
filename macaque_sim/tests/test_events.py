"""
test_events.py — Unit tests for event tracking (logger, hysteresis, cooldown).

Covers:
  - SimLogger initialization and state management
  - SimLogger recording of rank, power, and time data
  - Event recording and retrieval
  - Save/load functionality for checkpointing
  - Placeholders for hysteresis and cooldown when implemented

Run with:
  pytest tests/test_events.py -v
"""

import numpy as np
import pytest
import tempfile
from pathlib import Path

from macaque_sim.events.logger import SimLogger
from macaque_sim.config.sim_config import default_config


class TestSimLogger:
    """Tests for SimLogger state recording and event tracking."""
    
    def test_simlogger_initializes_correctly(self):
        """Verify logger correctly initializes pre-allocated arrays."""
        cfg = default_config()
        logger = SimLogger(cfg)
        
        assert logger.N == cfg.N
        assert logger.step_idx == 0
        assert len(logger.swap_events) == 0
        assert logger.expected_steps > 0
        assert logger.A.shape[0] > 0
        assert logger.P.shape[0] > 0
        assert logger.R.shape[0] > 0
    
    def test_simlogger_records_state(self):
        """Verify logger correctly records state at each timestep."""
        cfg = default_config()
        logger = SimLogger(cfg)
        
        t_val = 0.1
        R_t = np.array(list(range(1, cfg.N + 1)))
        P_t = np.ones(cfg.N) / cfg.N
        
        logger.record(t_val, R_t, P_t)
        
        assert logger.step_idx == 1
        assert logger.time[0] == t_val
        assert np.array_equal(logger.R[0], R_t)
        assert np.array_equal(logger.P[0], P_t)
    
    def test_simlogger_records_multiple_timesteps(self):
        """Verify logger handles multiple timestep recordings."""
        cfg = default_config()
        logger = SimLogger(cfg)
        
        num_records = 5
        for i in range(num_records):
            t_val = i * 0.1
            R_t = np.array(list(range(1, cfg.N + 1)))
            P_t = np.ones(cfg.N) / cfg.N
            logger.record(t_val, R_t, P_t)
        
        assert logger.step_idx == num_records
        assert logger.get_time_axis().shape[0] == num_records
        assert logger.get_rank_history().shape[0] == num_records
        assert logger.get_power_history().shape[0] == num_records
    
    def test_simlogger_records_events(self):
        """Verify logger correctly records event metadata."""
        cfg = default_config()
        logger = SimLogger(cfg)
        
        logger.record_event(0.5, "overthrow", {"loser": 2, "winner": 1})
        logger.record_event(1.0, "overtake", {"loser": 3, "winner": 2})
        
        events = logger.get_events()
        assert len(events) == 2
        assert events[0]["type"] == "overthrow"
        assert events[0]["t"] == 0.5
        assert events[1]["type"] == "overtake"
    
    def test_simlogger_truncates_arrays_on_save(self):
        """Verify logger saves only used portion of pre-allocated arrays."""
        cfg = default_config()
        cfg.T = 1.0  # Longer simulation
        logger = SimLogger(cfg)
        
        # Record fewer steps than pre-allocated
        num_records = min(3, logger.expected_steps - 1)
        for i in range(num_records):
            t_val = i * cfg.dt_ode
            R_t = np.array(list(range(1, cfg.N + 1)))
            P_t = np.ones(cfg.N) / cfg.N
            logger.record(t_val, R_t, P_t)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "test_log.npz"
            logger.save(save_path)
            
            # Verify saved arrays are truncated
            data = np.load(save_path, allow_pickle=True)
            assert data["P"].shape[0] == num_records
            assert data["R"].shape[0] == num_records
            assert data["time"].shape[0] == num_records
            data.close()  # Explicitly close to avoid file locking issues
    
    def test_simlogger_saves_and_loads(self):
        """Verify logger can save and restore state correctly."""
        cfg = default_config()
        logger = SimLogger(cfg)
        
        # Record some data
        R_t = np.array(list(range(1, cfg.N + 1)))
        P_t = np.ones(cfg.N) / cfg.N
        logger.record(0.1, R_t, P_t)
        logger.record_event(0.1, "overthrow", {"loser": 3, "winner": 2})
        
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "test_log.npz"
            logger.save(save_path)
            
            # Load and verify
            loaded_logger = SimLogger.load(save_path)
            
            assert loaded_logger.N == cfg.N
            assert loaded_logger.step_idx == 1
            assert np.array_equal(loaded_logger.get_rank_history()[0], R_t)
            assert np.array_equal(loaded_logger.get_power_history()[0], P_t)
            assert len(loaded_logger.get_events()) == 1
            assert loaded_logger.get_events()[0]["type"] == "overthrow"


# Hysteresis and cooldown have dedicated test files
# (test_hysteresis_and_cooldown.py, test_cooldown.py) now that they're implemented.
