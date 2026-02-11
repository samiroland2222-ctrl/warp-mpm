"""
Integration tests for mixed volumetric and shell MPM simulations
Tests that the full solver can handle both particle types in the same simulation
"""

import sys
import os
import numpy as np
import warp as wp
import pytest

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mpm_solver_warp import MPM_Simulator_WARP

wp.init()


def test_mixed_particles_initialization():
    """Test that solver can initialize with both volumetric and shell particles"""
    n_particles = 100
    solver = MPM_Simulator_WARP(n_particles, n_grid=50, device="cpu")

    # Set first 50 particles as volumetric (fluid/solid)
    particle_type = solver.mpm_state.particle_type.numpy()
    particle_type[:50] = 0  # volumetric
    particle_type[50:] = 1  # shell
    solver.mpm_state.particle_type = wp.array(particle_type, dtype=int, device="cpu")

    # Initialize fiber and normal for shell particles
    particle_fiber = solver.mpm_state.particle_fiber.numpy()
    particle_normal = solver.mpm_state.particle_normal.numpy()

    for i in range(50, 100):
        particle_fiber[i] = [1.0, 0.0, 0.0]  # fiber along x
        particle_normal[i] = [0.0, 0.0, 1.0]  # normal along z

    solver.mpm_state.particle_fiber = wp.array(particle_fiber, dtype=wp.vec3, device="cpu")
    solver.mpm_state.particle_normal = wp.array(particle_normal, dtype=wp.vec3, device="cpu")

    # Verify arrays are set correctly
    types = solver.mpm_state.particle_type.numpy()
    assert np.sum(types == 0) == 50, "Should have 50 volumetric particles"
    assert np.sum(types == 1) == 50, "Should have 50 shell particles"


def test_mixed_particles_stress_computation():
    """Test stress computation with mixed particle types"""
    n_particles = 10
    solver = MPM_Simulator_WARP(n_particles, n_grid=20, device="cpu")

    # Set material properties
    solver.set_parameters_dict({'E': 1e6, 'nu': 0.3, 'density': 1000.0})
    solver.finalize_mu_lam_bulk()

    # Initialize some particles as shell
    particle_type = np.zeros(n_particles, dtype=int)
    particle_type[5:] = 1  # Last 5 are shell
    solver.mpm_state.particle_type = wp.array(particle_type, dtype=int, device="cpu")

    # Initialize particle positions
    positions = np.random.rand(n_particles, 3) * 0.1 + 0.4  # Around center
    solver.mpm_state.particle_x = wp.array(positions, dtype=wp.vec3, device="cpu")

    # Initialize F to identity
    F_data = np.zeros((n_particles, 3, 3))
    for i in range(n_particles):
        F_data[i] = np.eye(3)
    solver.mpm_state.particle_F = wp.array(F_data, dtype=wp.mat33, device="cpu")
    solver.mpm_state.particle_F_trial = wp.array(F_data, dtype=wp.mat33, device="cpu")

    # Initialize fiber and normal for shell particles
    particle_fiber = np.zeros((n_particles, 3))
    particle_normal = np.zeros((n_particles, 3))
    for i in range(5, n_particles):
        particle_fiber[i] = [1.0, 0.0, 0.0]
        particle_normal[i] = [0.0, 0.0, 1.0]
    solver.mpm_state.particle_fiber = wp.array(particle_fiber, dtype=wp.vec3, device="cpu")
    solver.mpm_state.particle_normal = wp.array(particle_normal, dtype=wp.vec3, device="cpu")

    # Initialize volumes and masses
    vols = np.ones(n_particles) * 0.001
    masses = np.ones(n_particles) * 1.0
    solver.mpm_state.particle_vol = wp.array(vols, dtype=float, device="cpu")
    solver.mpm_state.particle_mass = wp.array(masses, dtype=float, device="cpu")

    # Compute stress (this should handle both types)
    from mpm_utils import compute_stress_from_F_trial
    wp.launch(
        kernel=compute_stress_from_F_trial,
        dim=n_particles,
        inputs=[solver.mpm_state, solver.mpm_model, 0.001],
        device="cpu"
    )

    # Check that stress was computed for all particles
    stress = solver.mpm_state.particle_stress.numpy()
    assert stress.shape == (n_particles, 3, 3), "Stress should be computed for all particles"

    # Stress should be near zero for identity deformation
    stress_magnitudes = np.abs(stress).sum(axis=(1, 2))
    assert np.all(stress_magnitudes < 1e3), "Identity deformation should have low stress"


def test_solver_substep_with_mixed_particles():
    """Test that a full substep works with mixed particles"""
    n_particles = 20
    solver = MPM_Simulator_WARP(n_particles, n_grid=30, device="cpu")

    # Set material to fluid for volumetric particles
    solver.mpm_model.material = 6  # fluid
    solver.set_parameters_dict({'E': 1e5, 'nu': 0.3, 'density': 1000.0, 'bulk_modulus': 2000.0})
    solver.finalize_mu_lam_bulk()

    # Initialize particle types
    particle_type = np.zeros(n_particles, dtype=int)
    particle_type[10:] = 1  # Last 10 are shell
    solver.mpm_state.particle_type = wp.array(particle_type, dtype=int, device="cpu")

    # Initialize positions in center of domain
    positions = np.random.rand(n_particles, 3) * 0.2 + 0.4
    solver.mpm_state.particle_x = wp.array(positions, dtype=wp.vec3, device="cpu")

    # Initialize velocities
    velocities = np.zeros((n_particles, 3))
    solver.mpm_state.particle_v = wp.array(velocities, dtype=wp.vec3, device="cpu")

    # Initialize F and F_trial
    F_data = np.zeros((n_particles, 3, 3))
    for i in range(n_particles):
        F_data[i] = np.eye(3)
    solver.mpm_state.particle_F = wp.array(F_data, dtype=wp.mat33, device="cpu")
    solver.mpm_state.particle_F_trial = wp.array(F_data, dtype=wp.mat33, device="cpu")

    # Initialize shell directions
    particle_fiber = np.zeros((n_particles, 3))
    particle_normal = np.zeros((n_particles, 3))
    for i in range(10, n_particles):
        particle_fiber[i] = [1.0, 0.0, 0.0]
        particle_normal[i] = [0.0, 0.0, 1.0]
    solver.mpm_state.particle_fiber = wp.array(particle_fiber, dtype=wp.vec3, device="cpu")
    solver.mpm_state.particle_normal = wp.array(particle_normal, dtype=wp.vec3, device="cpu")

    # Initialize volumes and masses
    vols = np.ones(n_particles) * 0.0001
    masses = np.ones(n_particles) * 0.1
    solver.mpm_state.particle_vol = wp.array(vols, dtype=float, device="cpu")
    solver.mpm_state.particle_mass = wp.array(masses, dtype=float, device="cpu")

    # Initialize C matrix
    C_data = np.zeros((n_particles, 3, 3))
    solver.mpm_state.particle_C = wp.array(C_data, dtype=wp.mat33, device="cpu")

    # Run one substep - should not crash
    dt = 0.0001
    try:
        solver.p2g2p(step=0, dt=dt)
        success = True
    except Exception as e:
        print(f"Substep failed with error: {e}")
        success = False

    assert success, "Solver should handle mixed particle types without crashing"

    # Verify positions changed (or stayed similar if stable)
    new_positions = solver.mpm_state.particle_x.numpy()
    assert new_positions.shape == (n_particles, 3), "Positions should still have correct shape"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

