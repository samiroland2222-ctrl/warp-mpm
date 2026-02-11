"""Debug: Check if deformation gradient is being computed correctly"""
import numpy as np
import warp as wp
from mpm_solver_warp import MPM_Simulator_WARP
from run_shell_membrane import create_rectangular_membrane

wp.init()

# Create a larger membrane to see deformation
positions, fibers, normals = create_rectangular_membrane(
    center=[0.5, 0.5, 2.9], width=0.15, height=0.15, particle_spacing=0.05
)
n_particles = len(positions)
print(f'Created {n_particles} particles')

# Initialize solver
solver = MPM_Simulator_WARP(n_particles, n_grid=64, grid_lim=3.0, device='cpu')
solver.mpm_state.particle_x = wp.array(positions, dtype=wp.vec3, device='cpu')

membrane_thickness = 0.002
particle_volume = 0.05 * 0.05 * membrane_thickness
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
    'g': [0.0, 0.0, -9.8]
})
solver.mpm_model.anisotropy_factor = 2.0
solver.finalize_mu_lam_bulk()
solver.add_bounding_box()

print(f'\nInitial positions (first 5):')
for i in range(min(5, n_particles)):
    print(f'  Particle {i}: {positions[i]}')

# Pin top-left corner
pin_idx = 0
print(f'\nPinning particle {pin_idx} at {positions[pin_idx]}')

# Run 20 steps
dt = 0.002  # Larger timestep to see more deformation
for step in range(20):
    solver.p2g2p(step=step, dt=dt)

    # Pin the particle by resetting position AND velocity
    pos = solver.mpm_state.particle_x.numpy()
    vel = solver.mpm_state.particle_v.numpy()
    pos[pin_idx] = positions[pin_idx]
    vel[pin_idx] = [0, 0, 0]
    solver.mpm_state.particle_x = wp.array(pos, dtype=wp.vec3, device='cpu')
    solver.mpm_state.particle_v = wp.array(vel, dtype=wp.vec3, device='cpu')

    if step in [0, 5, 10, 19]:
        F = solver.mpm_state.particle_F.numpy()
        stress = solver.mpm_state.particle_stress.numpy()
        pos = solver.mpm_state.particle_x.numpy()

        print(f'\n--- Step {step+1} ---')
        # Check particle at opposite corner from pinned
        test_idx = n_particles - 1
        print(f'Pinned particle {pin_idx}:')
        print(f'  pos: {pos[pin_idx]}')
        print(f'  F:\n{F[pin_idx]}')
        print(f'  stress norm: {np.linalg.norm(stress[pin_idx]):.6f}')

        print(f'Corner particle {test_idx}:')
        print(f'  pos: {pos[test_idx]}')
        print(f'  F:\n{F[test_idx]}')
        print(f'  det(F): {np.linalg.det(F[test_idx]):.6f}')
        print(f'  stress norm: {np.linalg.norm(stress[test_idx]):.6f}')

        # Check overall deformation
        z_coords = pos[:, 2]
        print(f'Overall z-range: [{z_coords.min():.4f}, {z_coords.max():.4f}]')

print(f'\n=== Final Check ===')
final_pos = solver.mpm_state.particle_x.numpy()
final_F = solver.mpm_state.particle_F.numpy()
final_stress = solver.mpm_state.particle_stress.numpy()

print(f'Position range:')
print(f'  x: [{final_pos[:, 0].min():.4f}, {final_pos[:, 0].max():.4f}]')
print(f'  y: [{final_pos[:, 1].min():.4f}, {final_pos[:, 1].max():.4f}]')
print(f'  z: [{final_pos[:, 2].min():.4f}, {final_pos[:, 2].max():.4f}]')

print(f'\nF determinant range: [{np.linalg.det(final_F).min():.6f}, {np.linalg.det(final_F).max():.6f}]')
print(f'Stress norm range: [{np.linalg.norm(final_stress, axis=(1,2)).min():.6f}, {np.linalg.norm(final_stress, axis=(1,2)).max():.6f}]')

# Check if there's ANY non-zero stress
nonzero_stress = np.where(np.linalg.norm(final_stress, axis=(1,2)) > 1e-6)[0]
print(f'\nParticles with non-zero stress: {len(nonzero_stress)} / {n_particles}')
if len(nonzero_stress) > 0:
    print(f'  Indices: {nonzero_stress[:10]}')

