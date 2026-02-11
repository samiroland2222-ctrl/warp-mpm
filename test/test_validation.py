"""
Tests for validate_initial_state function
"""

import sys
import os
import numpy as np
import warp as wp
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mpm_solver_warp import MPM_Simulator_WARP

wp.init()


def test_validate_catches_uninitialized_F_trial():
    """validate_initial_state should catch uninitialized particle_F_trial"""
    solver = MPM_Simulator_WARP(n_particles=10, n_grid=20, device="cpu")

    # Set positions and volumes
    positions = np.random.rand(10, 3) * 0.5 + 0.25
    solver.mpm_state.particle_x = wp.array(positions, dtype=wp.vec3, device="cpu")

    volumes = np.ones(10) * 0.001
    solver.mpm_state.particle_vol = wp.array(volumes, dtype=float, device="cpu")

    # Set masses
    solver.set_parameters_dict({'density': 1000.0})

    # F_trial is still all zeros - should raise error
    with pytest.raises(RuntimeError, match="particle_F_trial is all zeros"):
        solver.validate_initial_state()


def test_validate_catches_uninitialized_positions():
    """validate_initial_state should catch uninitialized positions"""
    solver = MPM_Simulator_WARP(n_particles=10, n_grid=20, device="cpu")

    # Initialize F_trial
    solver.initialize_shell_particles(range(10))

    # Set volumes and masses
    volumes = np.ones(10) * 0.001
    solver.mpm_state.particle_vol = wp.array(volumes, dtype=float, device="cpu")
    solver.set_parameters_dict({'density': 1000.0})

    # Positions are still all zeros - should raise error
    with pytest.raises(RuntimeError, match="particle_x is all zeros"):
        solver.validate_initial_state()


def test_validate_catches_zero_volumes():
    """validate_initial_state should catch zero volumes"""
    solver = MPM_Simulator_WARP(n_particles=10, n_grid=20, device="cpu")

    # Set positions
    positions = np.random.rand(10, 3) * 0.5 + 0.25
    solver.mpm_state.particle_x = wp.array(positions, dtype=wp.vec3, device="cpu")

    # Initialize F_trial
    solver.initialize_shell_particles(range(10))

    # Volumes still zero - should raise error
    with pytest.raises(RuntimeError, match="particle_vol values are zero"):
        solver.validate_initial_state()


def test_validate_catches_zero_masses():
    """validate_initial_state should catch zero masses"""
    solver = MPM_Simulator_WARP(n_particles=10, n_grid=20, device="cpu")

    # Set positions
    positions = np.random.rand(10, 3) * 0.5 + 0.25
    solver.mpm_state.particle_x = wp.array(positions, dtype=wp.vec3, device="cpu")

    # Initialize F_trial
    solver.initialize_shell_particles(range(10))

    # Set volumes
    volumes = np.ones(10) * 0.001
    solver.mpm_state.particle_vol = wp.array(volumes, dtype=float, device="cpu")

    # Masses still zero - should raise error
    with pytest.raises(RuntimeError, match="particle_mass values are zero"):
        solver.validate_initial_state()


def test_validate_passes_with_proper_initialization():
    """validate_initial_state should pass with fully initialized state"""
    solver = MPM_Simulator_WARP(n_particles=10, n_grid=20, device="cpu")

    # Set positions
    positions = np.random.rand(10, 3) * 0.5 + 0.25
    solver.mpm_state.particle_x = wp.array(positions, dtype=wp.vec3, device="cpu")

    # Set volumes
    volumes = np.ones(10) * 0.001
    solver.mpm_state.particle_vol = wp.array(volumes, dtype=float, device="cpu")

    # Initialize shell particles (sets F_trial and v)
    solver.initialize_shell_particles(range(10))

    # Set material properties
    solver.set_parameters_dict({'E': 1e6, 'nu': 0.3, 'density': 1000.0})
    solver.finalize_mu_lam_bulk()

    # Should not raise any error
    solver.validate_initial_state()


def test_validate_with_load_from_torch():
    """validate_initial_state should work with load_initial_data_from_torch"""
    import torch

    n_particles = 20
    positions = torch.rand(n_particles, 3) * 0.5 + 0.25
    volumes = torch.ones(n_particles) * 0.001

    solver = MPM_Simulator_WARP(n_particles, n_grid=20, device="cpu")
    solver.load_initial_data_from_torch(positions, volumes)

    # Set material properties
    solver.set_parameters_dict({'E': 1e6, 'nu': 0.3, 'density': 1000.0})
    solver.finalize_mu_lam_bulk()

    # Should pass validation (load_initial_data_from_torch initializes F_trial)
    solver.validate_initial_state()


def test_validate_reports_F_trial_trace():
    """validate_initial_state should report F_trial trace range"""
    solver = MPM_Simulator_WARP(n_particles=10, n_grid=20, device="cpu")

    # Set up properly
    positions = np.random.rand(10, 3) * 0.5 + 0.25
    solver.mpm_state.particle_x = wp.array(positions, dtype=wp.vec3, device="cpu")

    volumes = np.ones(10) * 0.001
    solver.mpm_state.particle_vol = wp.array(volumes, dtype=float, device="cpu")

    solver.initialize_shell_particles(range(10))
    solver.set_parameters_dict({'density': 1000.0})

    # Capture output
    import io
    from contextlib import redirect_stdout

    f = io.StringIO()
    with redirect_stdout(f):
        solver.validate_initial_state()
    output = f.getvalue()

    # Should report F_trial trace of 3.0 (identity matrix)
    assert "F_trial trace range: [3.000, 3.000]" in output
    assert "✓ Initial state validation passed" in output


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

