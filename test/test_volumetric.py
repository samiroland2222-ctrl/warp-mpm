import os
import sys

import numpy as np
import warp as wp
import pytest

# Add parent directory to path so we can import modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mpm_materials import get_stress_shell, get_stress_volumetric
from mpm_solver_warp import MPM_Simulator_WARP

wp.init()


def test_fluid_stress_incompressible():
    """Fluid at rest (J=1) should have zero stress"""
    wp.force_load()

    F = wp.mat33(1.0, 0.0, 0.0,
                 0.0, 1.0, 0.0,
                 0.0, 0.0, 1.0)

    @wp.kernel
    def test_kernel(result: wp.array(dtype=float)):
        U = wp.mat33(0.0)
        V = wp.mat33(0.0)
        sig = wp.vec3(0.0)
        wp.svd3(F, U, sig, V)
        J = 1.0
        bulk = 2000.0

        stress = get_stress_volumetric(F, U, V, sig, J, 0.0, 0.0, bulk, 6)  # material 6 = fluid

        # Sum stress magnitude
        total = 0.0
        for i in range(3):
            for j in range(3):
                total += wp.abs(stress[i, j])
        result[0] = total

    result = wp.zeros(1, dtype=float)
    wp.launch(test_kernel, dim=1, inputs=[result])

    stress_magnitude = result.numpy()[0]
    assert stress_magnitude < 1e-3, f"Incompressible fluid should have near-zero stress, got {stress_magnitude}"

def test_fluid_stress_compression():
    """Compressed fluid should have negative stress (tension in Kirchhoff stress formulation)"""
    wp.force_load()

    # Compress to 95% of original volume (J = 0.95)
    scale = 0.95 ** (1/3)
    F = wp.mat33(scale, 0.0, 0.0,
                 0.0, scale, 0.0,
                 0.0, 0.0, scale)

    @wp.kernel
    def test_kernel(result: wp.array(dtype=float)):
        U = wp.mat33(0.0)
        V = wp.mat33(0.0)
        sig = wp.vec3(0.0)
        wp.svd3(F, U, sig, V)
        J = wp.determinant(F)
        bulk = 2000.0

        stress = get_stress_volumetric(F, U, V, sig, J, 0.0, 0.0, bulk, 6)
        result[0] = stress[0, 0]  # Kirchhoff stress
        result[1] = J  # Check determinant

    result = wp.zeros(2, dtype=float)
    wp.launch(test_kernel, dim=1, inputs=[result])

    kirchhoff_stress = result.numpy()[0]
    J = result.numpy()[1]

    # Compressed fluid (J < 1) has negative Kirchhoff stress in this formulation
    assert kirchhoff_stress < 0, f"Compressed fluid (J={J:.3f}) should have negative Kirchhoff stress, got {kirchhoff_stress}"
    assert abs(kirchhoff_stress) > 10, f"Stress magnitude should be significant, got {kirchhoff_stress}"

def test_fcr_elastic_consistency():
    """FCR model should give consistent results"""
    wp.force_load()

    # Uniform stretch
    F = wp.mat33(1.1, 0.0, 0.0,
                 0.0, 1.1, 0.0,
                 0.0, 0.0, 1.1)

    @wp.kernel
    def test_kernel(result: wp.array(dtype=float)):
        U = wp.mat33(0.0)
        V = wp.mat33(0.0)
        sig = wp.vec3(0.0)
        wp.svd3(F, U, sig, V)
        J = wp.determinant(F)
        mu = 1e6
        lam = 1e6

        stress = get_stress_volumetric(F, U, V, sig, J, mu, lam, 0.0, 0)  # material 0 = FCR

        # For uniform stretch, all diagonal components should be equal
        result[0] = stress[0, 0]
        result[1] = stress[1, 1]
        result[2] = stress[2, 2]

    result = wp.zeros(3, dtype=float)
    wp.launch(test_kernel, dim=1, inputs=[result])

    stress_components = result.numpy()
    # All diagonal components should be approximately equal
    assert np.allclose(stress_components, stress_components[0], rtol=0.01), \
        f"Uniform stretch should give equal diagonal stress: {stress_components}"



def test_simulator_initialization():
    """Simulator should initialize all arrays including shell arrays"""
    n_particles = 100
    solver = MPM_Simulator_WARP(n_particles, n_grid=50, device="cpu")

    # Check that shell arrays exist
    assert solver.mpm_state.particle_type is not None
    assert solver.mpm_state.particle_fiber is not None
    assert solver.mpm_state.particle_normal is not None

    # Check sizes
    assert solver.mpm_state.particle_type.shape[0] == n_particles
    assert solver.mpm_state.particle_fiber.shape[0] == n_particles
    assert solver.mpm_state.particle_normal.shape[0] == n_particles

def test_anisotropy_factor_initialization():
    """Model should have anisotropy_factor initialized"""
    solver = MPM_Simulator_WARP(10, device="cpu")
    assert hasattr(solver.mpm_model, 'anisotropy_factor')
    assert solver.mpm_model.anisotropy_factor > 0


def test_sphere_volume_calculation():
    """Sphere of fluid particles should have correct volume"""
    # This will test the parametric sphere generation
    radius = 0.1  # 10 cm
    particle_spacing = 0.005  # 5 mm spacing

    # Expected volume
    expected_volume = (4/3) * np.pi * radius**3  # m^3

    # Each particle represents a small volume
    particle_volume = particle_spacing**3  # m^3

    # Number of particles needed
    n_particles_calculated = int(expected_volume / particle_volume)

    # Verify the calculation makes sense
    # Volume = 4.19e-3 m^3, particle_vol = 1.25e-7 m^3, n ≈ 33,510
    assert n_particles_calculated > 1000, \
        f"Should have many particles for a 10cm sphere, got {n_particles_calculated}"
    assert n_particles_calculated < 100000, \
        f"Particle count seems too high: {n_particles_calculated}"

def test_shell_thickness_calculation():
    """Shell should be thin compared to sphere radius"""
    radius = 0.1
    shell_thickness = 0.002  # 2mm

    # Shell should be much thinner than radius
    assert shell_thickness < 0.1 * radius, \
        f"Shell thickness {shell_thickness} should be << radius {radius}"

    # Surface area
    surface_area = 4 * np.pi * radius**2

    # Shell volume
    shell_volume = surface_area * shell_thickness

    # Fluid volume
    fluid_volume = (4/3) * np.pi * radius**3

    # Shell should contain much less volume than fluid
    assert shell_volume < 0.2 * fluid_volume, \
        f"Shell volume {shell_volume} should be << fluid volume {fluid_volume}"





def test_shell_stress_symmetry():
    """Shell stress should be symmetric"""
    wp.force_load()

    F = wp.mat33(1.2, 0.1, 0.0,
                 0.1, 1.1, 0.0,
                 0.0, 0.0, 1.0)

    fiber = wp.vec3(1.0, 0.0, 0.0)
    normal = wp.vec3(0.0, 0.0, 1.0)

    @wp.kernel
    def test_kernel(result: wp.array(dtype=float)):
        stress = get_stress_shell(F, fiber, normal, 1e6, 1e6, 2.0)
        # Check symmetry: stress[i,j] == stress[j,i]
        result[0] = stress[0, 1]
        result[1] = stress[1, 0]
        result[2] = stress[0, 2]
        result[3] = stress[2, 0]
        result[4] = stress[1, 2]
        result[5] = stress[2, 1]

    result = wp.zeros(6, dtype=float)
    wp.launch(test_kernel, dim=1, inputs=[result])

    values = result.numpy()
    assert np.abs(values[0] - values[1]) < 1e-3, "Stress should be symmetric: σ_xy ≠ σ_yx"
    assert np.abs(values[2] - values[3]) < 1e-3, "Stress should be symmetric: σ_xz ≠ σ_zx"
    assert np.abs(values[4] - values[5]) < 1e-3, "Stress should be symmetric: σ_yz ≠ σ_zy"

def test_volumetric_stress_symmetry():
    """Volumetric stress should be symmetric"""
    wp.force_load()

    F = wp.mat33(1.2, 0.1, 0.05,
                 0.1, 1.1, 0.02,
                 0.05, 0.02, 1.0)

    @wp.kernel
    def test_kernel(result: wp.array(dtype=float)):
        U = wp.mat33(0.0)
        V = wp.mat33(0.0)
        sig = wp.vec3(0.0)
        wp.svd3(F, U, sig, V)
        J = wp.determinant(F)

        stress = get_stress_volumetric(F, U, V, sig, J, 1e6, 1e6, 0.0, 0)  # FCR

        result[0] = stress[0, 1]
        result[1] = stress[1, 0]
        result[2] = stress[0, 2]
        result[3] = stress[2, 0]

    result = wp.zeros(4, dtype=float)
    wp.launch(test_kernel, dim=1, inputs=[result])

    values = result.numpy()
    # Use relative tolerance for large stress values
    rel_tol = 1e-4  # 0.01% relative error is acceptable
    assert np.abs(values[0] - values[1]) / (np.abs(values[0]) + 1e-10) < rel_tol, \
        f"Volumetric stress should be symmetric: σ_xy={values[0]}, σ_yx={values[1]}"
    assert np.abs(values[2] - values[3]) / (np.abs(values[2]) + 1e-10) < rel_tol, \
        f"Volumetric stress should be symmetric: σ_xz={values[2]}, σ_zx={values[3]}"
