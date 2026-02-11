"""
MPM Material Models
This module contains material constitutive models for MPM simulation.
Refactored from mpm_utils.py to support codimensional (shell) materials.
"""

import warp as wp
from warp_utils import *


# ============================================================================
# Volumetric (3D) Material Models
# ============================================================================

@wp.func
def kirchoff_stress_FCR(
    F: wp.mat33, U: wp.mat33, V: wp.mat33, J: float, mu: float, lam: float
):
    """Fixed Corotated (FCR) stress model - good for elastic jelly-like materials"""
    R = U * wp.transpose(V)
    id = wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    return 2.0 * mu * (F - R) * wp.transpose(F) + id * lam * J * (J - 1.0)


@wp.func
def kirchoff_stress_water(
    J: float, bulk: float
):
    """Weakly compressible fluid model"""
    gamma = 1.1  # gamma is set to be a little greater than 1 for weakly compressible fluids
    pressure = -bulk * (wp.pow(J, -gamma) - 1.)
    id = wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    cauchy_stress = id * pressure
    return J * cauchy_stress


@wp.func
def kirchoff_stress_neoHookean(
    F: wp.mat33, U: wp.mat33, V: wp.mat33, J: float, sig: wp.vec3, mu: float, lam: float
):
    """Neo-Hookean hyperelastic model"""
    b = wp.vec3(sig[0] * sig[0], sig[1] * sig[1], sig[2] * sig[2])
    b_hat = b - wp.vec3(
        (b[0] + b[1] + b[2]) / 3.0,
        (b[0] + b[1] + b[2]) / 3.0,
        (b[0] + b[1] + b[2]) / 3.0,
    )
    tau = mu * J ** (-2.0 / 3.0) * b_hat + lam / 2.0 * (J * J - 1.0) * wp.vec3(
        1.0, 1.0, 1.0
    )
    return (
        U
        * wp.mat33(tau[0], 0.0, 0.0, 0.0, tau[1], 0.0, 0.0, 0.0, tau[2])
        * wp.transpose(V)
        * wp.transpose(F)
    )


@wp.func
def kirchoff_stress_StVK(
    F: wp.mat33, U: wp.mat33, V: wp.mat33, sig: wp.vec3, mu: float, lam: float
):
    """St. Venant-Kirchhoff stress model"""
    sig = wp.vec3(
        wp.max(sig[0], 0.01), wp.max(sig[1], 0.01), wp.max(sig[2], 0.01)
    )  # add this to prevent NaN in extreme cases
    epsilon = wp.vec3(wp.log(sig[0]), wp.log(sig[1]), wp.log(sig[2]))
    log_sig_sum = wp.log(sig[0]) + wp.log(sig[1]) + wp.log(sig[2])
    ONE = wp.vec3(1.0, 1.0, 1.0)
    tau = 2.0 * mu * epsilon + lam * log_sig_sum * ONE
    return (
        U
        * wp.mat33(tau[0], 0.0, 0.0, 0.0, tau[1], 0.0, 0.0, 0.0, tau[2])
        * wp.transpose(V)
        * wp.transpose(F)
    )


@wp.func
def kirchoff_stress_drucker_prager(
    F: wp.mat33, U: wp.mat33, V: wp.mat33, sig: wp.vec3, mu: float, lam: float
):
    """Drucker-Prager model for granular materials"""
    log_sig_sum = wp.log(sig[0]) + wp.log(sig[1]) + wp.log(sig[2])
    center00 = 2.0 * mu * wp.log(sig[0]) * (1.0 / sig[0]) + lam * log_sig_sum * (
        1.0 / sig[0]
    )
    center11 = 2.0 * mu * wp.log(sig[1]) * (1.0 / sig[1]) + lam * log_sig_sum * (
        1.0 / sig[1]
    )
    center22 = 2.0 * mu * wp.log(sig[2]) * (1.0 / sig[2]) + lam * log_sig_sum * (
        1.0 / sig[2]
    )
    center = wp.mat33(center00, 0.0, 0.0, 0.0, center11, 0.0, 0.0, 0.0, center22)
    return U * center * wp.transpose(V) * wp.transpose(F)


@wp.func
def get_stress_volumetric(
    F: wp.mat33,
    U: wp.mat33,
    V: wp.mat33,
    sig: wp.vec3,
    J: float,
    mu: float,
    lam: float,
    bulk: float,
    material: int
):
    """
    Unified interface for volumetric (3D) material stress calculation.

    Args:
        F: Deformation gradient
        U, V, sig: SVD components of F
        J: Determinant of F
        mu, lam: Lamé parameters
        bulk: Bulk modulus (for fluids)
        material: Material type ID

    Returns:
        Kirchhoff stress tensor
    """
    stress = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    if material == 0 or material == 5:  # FCR (jelly/elastic)
        stress = kirchoff_stress_FCR(F, U, V, J, mu, lam)
    elif material == 1:  # St. Venant-Kirchhoff
        stress = kirchoff_stress_StVK(F, U, V, sig, mu, lam)
    elif material == 2:  # Drucker-Prager (sand)
        stress = kirchoff_stress_drucker_prager(F, U, V, sig, mu, lam)
    elif material == 3:  # Viscoplastic (temporarily StVK)
        stress = kirchoff_stress_StVK(F, U, V, sig, mu, lam)
    elif material == 6:  # Fluid
        stress = kirchoff_stress_water(J, bulk)

    return stress


# ============================================================================
# Shell (2D Codimensional) Material Models
# ============================================================================

@wp.func
def get_stress_shell(
    F: wp.mat33,
    particle_fiber: wp.vec3,
    particle_normal: wp.vec3,
    mu: float,
    lam: float,
    anisotropy_factor: float
):
    """
    Anisotropic Plane Stress model for thin shell materials (cloth/membrane).

    This implements a simplified 2D stress model that:
    1. Projects F onto the shell plane (defined by normal)
    2. Applies plane stress condition (no resistance to thinning)
    3. Scales stress along fiber direction for anisotropy

    Args:
        F: Deformation gradient (3D)
        particle_fiber: Fiber direction in material space
        particle_normal: Shell normal direction
        mu, lam: Lamé parameters
        anisotropy_factor: Multiplier for stress along fiber direction (> 1.0 = stiffer along fibers)

    Returns:
        Kirchhoff stress tensor (plane stress, 2D embedded in 3D)
    """

    # Compute two orthogonal tangent vectors in the shell plane
    # tangent1 is aligned with fiber direction
    tangent1 = particle_fiber
    tangent1 = tangent1 - particle_normal * wp.dot(tangent1, particle_normal)  # project onto plane
    tangent1_len = wp.length(tangent1)
    if tangent1_len > 1e-6:
        tangent1 = tangent1 / tangent1_len
    else:
        # If fiber is parallel to normal, pick arbitrary tangent
        if wp.abs(particle_normal[0]) < 0.9:
            tangent1 = wp.normalize(wp.cross(particle_normal, wp.vec3(1.0, 0.0, 0.0)))
        else:
            tangent1 = wp.normalize(wp.cross(particle_normal, wp.vec3(0.0, 1.0, 0.0)))

    tangent2 = wp.normalize(wp.cross(particle_normal, tangent1))

    # Project F onto the 2D shell space
    # The key insight: F maps material vectors to spatial vectors
    # For shell, we want to know how material tangent vectors deform
    # F_tangent = [F*t1, F*t2] gives us the deformed tangent vectors
    # Then we measure their length/angle changes in the tangent plane

    # Get deformed tangent vectors
    F_t1 = F * tangent1  # Where tangent1 goes after deformation
    F_t2 = F * tangent2  # Where tangent2 goes after deformation

    # Extract 2x2 in-plane deformation gradient
    # F_shell[i,j] = how much of deformed tangent_j is along tangent_i direction
    F11 = wp.dot(F_t1, tangent1)
    F12 = wp.dot(F_t2, tangent1)
    F21 = wp.dot(F_t1, tangent2)
    F22 = wp.dot(F_t2, tangent2)

    # Right Cauchy-Green deformation tensor in shell coordinates: C = F^T * F
    # C11 = F11^2 + F21^2
    # C22 = F12^2 + F22^2
    # C12 = F11*F12 + F21*F22
    C11 = F11 * F11 + F21 * F21
    C22 = F12 * F12 + F22 * F22
    C12 = F11 * F12 + F21 * F22

    # Green strain: E = 0.5 * (C - I)
    E11 = 0.5 * (C11 - 1.0)
    E22 = 0.5 * (C22 - 1.0)
    E12 = 0.5 * C12

    # Plane stress constitutive relation (isotropic base)
    # sigma = (E / (1 - nu^2)) * [ [1, nu, 0], [nu, 1, 0], [0, 0, (1-nu)/2] ] * epsilon
    # Using Lamé parameters: mu = E/(2(1+nu)), lam = E*nu/((1+nu)(1-2nu))
    # For plane stress, effective_lam = 2*mu*lam/(lam + 2*mu)
    effective_lam = 2.0 * mu * lam / (lam + 2.0 * mu + 1e-10)

    # Base 2D stress (isotropic)
    S11_base = 2.0 * mu * E11 + effective_lam * (E11 + E22)
    S22_base = 2.0 * mu * E22 + effective_lam * (E11 + E22)
    S12_base = 2.0 * mu * E12

    # Apply anisotropy: scale stress along fiber direction (tangent1)
    # Simple model: multiply normal stress in fiber direction by anisotropy factor
    S11 = S11_base * anisotropy_factor
    S22 = S22_base
    S12 = S12_base

    # Convert 2D stress to 3D (in material configuration)
    # S_material = S11 * (t1 ⊗ t1) + S22 * (t2 ⊗ t2) + S12 * (t1 ⊗ t2 + t2 ⊗ t1)
    S_material = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    S_material = S_material + wp.outer(tangent1, tangent1) * S11
    S_material = S_material + wp.outer(tangent2, tangent2) * S22
    S_material = S_material + wp.outer(tangent1, tangent2) * S12
    S_material = S_material + wp.outer(tangent2, tangent1) * S12

    # Push forward to spatial configuration: Kirchhoff stress = F * S * F^T
    # This transformation is essential for the MPM solver to correctly apply forces
    stress = F * S_material * wp.transpose(F)

    return stress


