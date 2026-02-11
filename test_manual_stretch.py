"""Test: Manually stretch membrane to check if stress is computed"""
import numpy as np
import warp as wp
from mpm_solver_warp import MPM_Simulator_WARP

wp.init()

# Create 4 particles in a square, manually
positions = np.array([
    [0.4, 0.4, 2.9],
    [0.4, 0.5, 2.9],
    [0.5, 0.4, 2.9],
    [0.5, 0.5, 2.9]
])

fibers = np.array([[1.0, 0.0, 0.0]] * 4)
normals = np.array([[0.0, 0.0, 1.0]] * 4)
n_particles = 4

# Initialize solver
solver = MPM_Simulator_WARP(n_particles, n_grid=64, grid_lim=3.0, device='cpu')
solver.mpm_state.particle_x = wp.array(positions, dtype=wp.vec3, device='cpu')

particle_volume = 0.1 * 0.1 * 0.002
volumes = np.ones(n_particles) * particle_volume
solver.mpm_state.particle_vol = wp.array(volumes, dtype=float, device='cpu')

solver.initialize_shell_particles(
    particle_indices=range(n_particles),
    fiber_directions=fibers,
    normal_directions=normals
)

solver.set_parameters_dict({
    'material': 'jelly',
    'E': 1e6,
    'nu': 0.4,
    'density': 1100.0,
    'g': [0.0, 0.0, 0.0]  # NO GRAVITY
})
solver.mpm_model.anisotropy_factor = 1.0
solver.finalize_mu_lam_bulk()
solver.add_bounding_box()

print('Initial setup complete')
print(f'mu = {solver.mpm_model.mu.numpy()[0]:.2f}')
print(f'lam = {solver.mpm_model.lam.numpy()[0]:.2f}')

# Run one step with no movement - should have zero stress
solver.p2g2p(step=0, dt=0.001)

stress = solver.mpm_state.particle_stress.numpy()
F = solver.mpm_state.particle_F.numpy()
print(f'\nAfter 1 step (no deformation):')
print(f'F[0]:\n{F[0]}')
print(f'Stress[0] norm: {np.linalg.norm(stress[0]):.6f}')

# Now manually stretch the membrane by 10% in X direction
print(f'\n=== Manually stretching by 10% in X ===')
pos = solver.mpm_state.particle_x.numpy()
# Stretch particles in X direction
for i in range(n_particles):
    pos[i, 0] = 0.45 + (pos[i, 0] - 0.45) * 1.1

solver.mpm_state.particle_x = wp.array(pos, dtype=wp.vec3, device='cpu')

print(f'New positions:')
for i in range(4):
    print(f'  {i}: {pos[i]}')

# Run a few steps to let F update from the new positions
for step in range(3):
    solver.p2g2p(step=step, dt=0.001)

stress = solver.mpm_state.particle_stress.numpy()
F = solver.mpm_state.particle_F.numpy()

print(f'\nAfter stretching:')
for i in range(4):
    print(f'Particle {i}:')
    print(f'  F:\n{F[i]}')
    print(f'  det(F): {np.linalg.det(F[i]):.6f}')
    print(f'  stress norm: {np.linalg.norm(stress[i]):.6f}')
    if np.linalg.norm(stress[i]) > 0:
        print(f'  stress:\n{stress[i]}')

