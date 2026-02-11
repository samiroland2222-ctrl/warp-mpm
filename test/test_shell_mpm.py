"""
Unit tests for Shell MPM (Codimensional MPM) implementation
Tests material models, stress calculations, and water balloon initialization
"""

import sys
import os
import numpy as np
import warp as wp
import pytest

# Add parent directory to path so we can import modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mpm_materials import get_stress_volumetric, get_stress_shell
from warp_utils import MPMModelStruct, MPMStateStruct
from mpm_solver_warp import MPM_Simulator_WARP

wp.init()


"""Test shell material stress calculations"""

def test_shell_stress_zero_deformation():
    """Shell with identity deformation should have near-zero stress"""
    wp.force_load()

    # Identity deformation gradient
    F = wp.mat33(1.0, 0.0, 0.0,
                 0.0, 1.0, 0.0,
                 0.0, 0.0, 1.0)

    # Define shell orientation
    fiber = wp.vec3(1.0, 0.0, 0.0)
    normal = wp.vec3(0.0, 0.0, 1.0)

    # Material parameters (typical rubber)
    mu = 1e6  # Pa
    lam = 1e6  # Pa
    anisotropy = 2.0

    @wp.kernel
    def test_kernel(result: wp.array(dtype=float)):
        stress = get_stress_shell(F, fiber, normal, mu, lam, anisotropy)
        # Sum all stress components to check magnitude
        total = 0.0
        for i in range(3):
            for j in range(3):
                total += wp.abs(stress[i, j])
        result[0] = total

    result = wp.zeros(1, dtype=float)
    wp.launch(test_kernel, dim=1, inputs=[result])

    stress_magnitude = result.numpy()[0]
    assert stress_magnitude < 1e3, f"Identity deformation should have near-zero stress, got {stress_magnitude}"

def test_shell_stress_tension():
    """Shell under tension should produce positive stress"""
    wp.force_load()

    # Stretch by 10% in x direction
    F = wp.mat33(1.1, 0.0, 0.0,
                 0.0, 1.0, 0.0,
                 0.0, 0.0, 1.0)

    fiber = wp.vec3(1.0, 0.0, 0.0)
    normal = wp.vec3(0.0, 0.0, 1.0)

    mu = 1e6
    lam = 1e6
    anisotropy = 2.0

    @wp.kernel
    def test_kernel(result: wp.array(dtype=float)):
        stress = get_stress_shell(F, fiber, normal, mu, lam, anisotropy)
        # Check stress along fiber direction (should be positive)
        result[0] = stress[0, 0]  # xx component

    result = wp.zeros(1, dtype=float)
    wp.launch(test_kernel, dim=1, inputs=[result])

    stress_xx = result.numpy()[0]
    assert stress_xx > 0, f"Tension should produce positive stress, got {stress_xx}"

def test_shell_anisotropy():
    """Stress along fiber should be scaled by anisotropy factor"""
    wp.force_load()

    # Stretch along fiber direction
    F = wp.mat33(1.1, 0.0, 0.0,
                 0.0, 1.0, 0.0,
                 0.0, 0.0, 1.0)

    fiber = wp.vec3(1.0, 0.0, 0.0)
    normal = wp.vec3(0.0, 0.0, 1.0)
    mu = 1e6
    lam = 1e6

    @wp.kernel
    def test_kernel(aniso: float, result: wp.array(dtype=float)):
        stress = get_stress_shell(F, fiber, normal, mu, lam, aniso)
        result[0] = stress[0, 0]

    # Test with different anisotropy factors
    result_iso = wp.zeros(1, dtype=float)
    result_aniso = wp.zeros(1, dtype=float)

    wp.launch(test_kernel, dim=1, inputs=[1.0, result_iso])
    wp.launch(test_kernel, dim=1, inputs=[3.0, result_aniso])

    stress_iso = result_iso.numpy()[0]
    stress_aniso = result_aniso.numpy()[0]

    # Anisotropic should be roughly 3x larger
    ratio = stress_aniso / (stress_iso + 1e-10)
    assert 2.5 < ratio < 3.5, f"Anisotropy ratio should be ~3.0, got {ratio}"

def test_shell_plane_stress():
    """Shell should have minimal normal stress (plane stress condition)"""
    wp.force_load()

    F = wp.mat33(1.1, 0.0, 0.0,
                 0.0, 1.1, 0.0,
                 0.0, 0.0, 1.0)

    fiber = wp.vec3(1.0, 0.0, 0.0)
    normal = wp.vec3(0.0, 0.0, 1.0)
    mu = 1e6
    lam = 1e6
    anisotropy = 2.0

    @wp.kernel
    def test_kernel(result: wp.array(dtype=float)):
        stress = get_stress_shell(F, fiber, normal, mu, lam, anisotropy)
        # Check stress normal to shell (should be near zero)
        result[0] = wp.abs(stress[2, 2])  # zz component
        result[1] = wp.abs(stress[0, 0])  # xx component for comparison

    result = wp.zeros(2, dtype=float)
    wp.launch(test_kernel, dim=1, inputs=[result])

    stress_zz = result.numpy()[0]
    stress_xx = result.numpy()[1]

    # Normal stress should be much smaller than in-plane stress
    assert stress_zz < 0.1 * stress_xx, f"Normal stress {stress_zz} should be << in-plane stress {stress_xx}"


