"""Debug script to check if shell stress is being computed correctly"""
import numpy as np
import warp as wp
from mpm_solver_warp import MPM_Simulator_WARP
from run_shell_membrane import create_rectangular_membrane

wp.init()

# Create a simple 3x3 membrane for easier debugging
positions, fibers, normals = create_rectangular_membrane(
    center=[0.5, 0.5, 2.9], width=0.06, height=0.06, particle_spacing=0.03
)
n_particles = len(positions)
print(f'Created {n_particles} particles (should be 9 for 3x3 grid)')
print(f'Positions:\n{positions}')

# Initialize solver
solver = MPM_Simulator_WARP(n_particles, n_grid=64, grid_lim=3.0, device='cpu')

# Set particle positions
solver.mpm_state.particle_x = wp.array(positions, dtype=wp.vec3, device='cpu')

# Set particle volumes
membrane_thickness = 0.002
particle_volume = 0.03 * 0.03 * membrane_thickness
volumes = np.ones(n_particles) * particle_volume
solver.mpm_state.particle_vol = wp.array(volumes, dtype=float, device='cpu')

# Initialize as shell particles
solver.initialize_shell_particles(
    particle_indices=range(n_particles),
    fiber_directions=fibers,
    normal_directions=normals
)

# Set material properties
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

print(f'\n=== Initial State ===')
print(f'Particle types: {solver.mpm_state.particle_type.numpy()}')
print(f'Particle masses: {solver.mpm_state.particle_mass.numpy()}')
print(f'Particle volumes: {solver.mpm_state.particle_vol.numpy()}')

# Check initial F
F = solver.mpm_state.particle_F.numpy()
print(f'\nInitial F (should be identity):')
for i in range(min(3, n_particles)):
    print(f'  Particle {i}: trace={np.trace(F[i]):.3f}, det={np.linalg.det(F[i]):.3f}')

# Check initial stress (should be zero)
stress = solver.mpm_state.particle_stress.numpy()
print(f'\nInitial stress (should be ~0):')
for i in range(min(3, n_particles)):
    print(f'  Particle {i}: norm={np.linalg.norm(stress[i]):.6f}')

# Pin the center particle (particle 4 in a 3x3 grid)
center_idx = n_particles // 2
print(f'\n=== Pinning particle {center_idx} ===')
print(f'Position before: {positions[center_idx]}')

# Run simulation for a few steps
dt = 0.001
for step in range(5):
    solver.p2g2p(step=step, dt=dt)

    # Pin center particle by resetting position AND velocity
    pos = solver.mpm_state.particle_x.numpy()
    vel = solver.mpm_state.particle_v.numpy()

    # Save original pinned position
    pinned_pos = positions[center_idx].copy()
    pos[center_idx] = pinned_pos
    vel[center_idx] = [0, 0, 0]  # Also zero out velocity!

    solver.mpm_state.particle_x = wp.array(pos, dtype=wp.vec3, device='cpu')
    solver.mpm_state.particle_v = wp.array(vel, dtype=wp.vec3, device='cpu')

    # Check stress after pinning
    if step == 0 or step == 4:
        stress = solver.mpm_state.particle_stress.numpy()
        F = solver.mpm_state.particle_F.numpy()
        pos = solver.mpm_state.particle_x.numpy()

        print(f'\n--- After step {step+1} ---')
        print(f'Positions (z-coords): {pos[:, 2]}')
        print(f'Stress norms: {[np.linalg.norm(stress[i]) for i in range(n_particles)]}')
        print(f'F determinants: {[np.linalg.det(F[i]) for i in range(n_particles)]}')

        # Check if neighbors of pinned particle have stress
        print(f'\nStress detail for center and neighbors:')
        for i in [center_idx-1, center_idx, center_idx+1]:
            if 0 <= i < n_particles:
                print(f'  Particle {i}: stress_norm={np.linalg.norm(stress[i]):.6f}, det(F)={np.linalg.det(F[i]):.6f}')

print('\n=== Final Positions ===')
final_pos = solver.mpm_state.particle_x.numpy()
for i in range(n_particles):
    dz = final_pos[i, 2] - positions[i, 2]
    print(f'  Particle {i}: z={final_pos[i, 2]:.4f} (delta={dz:.4f})')

