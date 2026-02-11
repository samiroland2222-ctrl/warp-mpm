"""Test the stress function directly with known inputs"""
import numpy as np
import warp as wp

wp.init()

# Copy the stress function but add debug output
@wp.kernel
def test_stress_kernel(
    F: wp.array(dtype=wp.mat33),
    fiber: wp.array(dtype=wp.vec3),
    normal: wp.array(dtype=wp.vec3),
    mu: float,
    lam: float,
    aniso: float,
    result_stress: wp.array(dtype=wp.mat33),
    result_E11: wp.array(dtype=float),
    result_E22: wp.array(dtype=float),
    result_E12: wp.array(dtype=float)
):
    tid = wp.tid()

    F_val = F[tid]
    particle_fiber = fiber[tid]
    particle_normal = normal[tid]

    # Compute tangent basis
    tangent1 = particle_fiber
    tangent1 = tangent1 - particle_normal * wp.dot(tangent1, particle_normal)
    tangent1_len = wp.length(tangent1)
    if tangent1_len > 1e-6:
        tangent1 = tangent1 / tangent1_len
    else:
        if wp.abs(particle_normal[0]) < 0.9:
            tangent1 = wp.normalize(wp.cross(particle_normal, wp.vec3(1.0, 0.0, 0.0)))
        else:
            tangent1 = wp.normalize(wp.cross(particle_normal, wp.vec3(0.0, 1.0, 0.0)))

    tangent2 = wp.normalize(wp.cross(particle_normal, tangent1))

    # Get deformed tangent vectors
    F_t1 = F_val * tangent1
    F_t2 = F_val * tangent2

    # Extract 2x2 in-plane deformation gradient
    F11 = wp.dot(F_t1, tangent1)
    F12 = wp.dot(F_t2, tangent1)
    F21 = wp.dot(F_t1, tangent2)
    F22 = wp.dot(F_t2, tangent2)

    # Right Cauchy-Green deformation tensor
    C11 = F11 * F11 + F21 * F21
    C22 = F12 * F12 + F22 * F22
    C12 = F11 * F12 + F21 * F22

    # Green strain
    E11 = 0.5 * (C11 - 1.0)
    E22 = 0.5 * (C22 - 1.0)
    E12 = 0.5 * C12

    result_E11[tid] = E11
    result_E22[tid] = E22
    result_E12[tid] = E12

    # Plane stress constitutive
    effective_lam = 2.0 * mu * lam / (lam + 2.0 * mu + 1e-10)

    S11_base = 2.0 * mu * E11 + effective_lam * (E11 + E22)
    S22_base = 2.0 * mu * E22 + effective_lam * (E11 + E22)
    S12_base = 2.0 * mu * E12

    S11 = S11_base * aniso
    S22 = S22_base
    S12 = S12_base

    # Convert to 3D
    S_material = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    S_material = S_material + wp.outer(tangent1, tangent1) * S11
    S_material = S_material + wp.outer(tangent2, tangent2) * S22
    S_material = S_material + wp.outer(tangent1, tangent2) * S12
    S_material = S_material + wp.outer(tangent2, tangent1) * S12

    # Push forward
    stress = F_val * S_material * wp.transpose(F_val)
    result_stress[tid] = stress

# Test with a deformed F
F_test = np.array([
    [[1.1, 0.0, 0.0],    # 10% stretch in X
     [0.0, 1.0, 0.0],
     [0.0, 0.0, 1.0]],

    [[1.0, 0.0, 0.0],    # Identity (no deformation)
     [0.0, 1.0, 0.0],
     [0.0, 0.0, 1.0]],

    [[1.0, 0.1, 0.0],    # Shear in XY
     [0.0, 1.0, 0.0],
     [0.0, 0.0, 1.0]],
], dtype=np.float32)

fiber_test = np.array([
    [1.0, 0.0, 0.0],
    [1.0, 0.0, 0.0],
    [1.0, 0.0, 0.0]
], dtype=np.float32)

normal_test = np.array([
    [0.0, 0.0, 1.0],
    [0.0, 0.0, 1.0],
    [0.0, 0.0, 1.0]
], dtype=np.float32)

n_test = 3

# Material parameters
E = 1e6
nu = 0.4
mu = E / (2 * (1 + nu))
lam = E * nu / ((1 + nu) * (1 - 2 * nu))

print(f'Material parameters:')
print(f'  E = {E:.2e}')
print(f'  nu = {nu}')
print(f'  mu = {mu:.2e}')
print(f'  lam = {lam:.2e}')

# Run test
F_wp = wp.array(F_test, dtype=wp.mat33)
fiber_wp = wp.array(fiber_test, dtype=wp.vec3)
normal_wp = wp.array(normal_test, dtype=wp.vec3)
stress_wp = wp.zeros(n_test, dtype=wp.mat33)
E11_wp = wp.zeros(n_test, dtype=float)
E22_wp = wp.zeros(n_test, dtype=float)
E12_wp = wp.zeros(n_test, dtype=float)

wp.launch(
    kernel=test_stress_kernel,
    dim=n_test,
    inputs=[F_wp, fiber_wp, normal_wp, mu, lam, 2.0, stress_wp, E11_wp, E22_wp, E12_wp]
)

stress_result = stress_wp.numpy()
E11_result = E11_wp.numpy()
E22_result = E22_wp.numpy()
E12_result = E12_wp.numpy()

print(f'\n=== Test Results ===')
for i in range(n_test):
    print(f'\nCase {i}:')
    print(f'  F:\n{F_test[i]}')
    print(f'  Strain: E11={E11_result[i]:.6f}, E22={E22_result[i]:.6f}, E12={E12_result[i]:.6f}')
    print(f'  Stress norm: {np.linalg.norm(stress_result[i]):.2f}')
    if np.linalg.norm(stress_result[i]) > 0.1:
        print(f'  Stress:\n{stress_result[i]}')

