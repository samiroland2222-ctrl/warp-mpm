"""Test if membrane generates stress when stretched"""
import numpy as np
import warp as wp
from mpm_solver_warp import MPM_Simulator_WARP
from run_shell_membrane import create_rectangular_membrane

wp.init()

# Create membrane with proper grid spacing
positions, fibers, normals = create_rectangular_membrane(
    center=[0.75, 0.75, 1.3],
    width=0.6,
    height=0.6,
    particle_spacing=0.01
)
n_particles = len(positions)
print(f'Created {n_particles} particles')

# Grid parameters for proper spacing
n_grid = 128
grid_lim = 1.5
dx = grid_lim / n_grid
print(f'Grid: {n_grid}³, dx={dx*1000:.2f}mm')
print(f'Particle spacing: 10mm')
print(f'Spacing/dx ratio: {0.01/dx:.2f}')

# Initialize solver
solver = MPM_Simulator_WARP(n_particles, n_grid=n_grid, grid_lim=grid_lim, device='cpu')
solver.mpm_state.particle_x = wp.array(positions, dtype=wp.vec3, device='cpu')

membrane_thickness = 0.002
particle_volume = 0.01 * 0.01 * membrane_thickness
volumes = np.ones(n_particles) * particle_volume
solver.mpm_state.particle_vol = wp.array(volumes, dtype=float, device='cpu')

solver.initialize_shell_particles(
    particle_indices=range(n_particles),
    fiber_directions=fibers,
    normal_directions=normals
)

# Use stiffer material to make effect more visible
solver.set_parameters_dict({
    'material': 'jelly',
    'E': 5e6,  # 5 MPa - stiffer
    'nu': 0.4,
    'density': 1100.0,
    'g': [0.0, 0.0, -9.8]
})
solver.mpm_model.anisotropy_factor = 2.0
solver.finalize_mu_lam_bulk()
solver.add_bounding_box()

print(f'\nMaterial properties:')
print(f'  E = {5e6:.2e} Pa')
print(f'  mu = {solver.mpm_model.mu.numpy()[0]:.2e} Pa')
print(f'  lam = {solver.mpm_model.lam.numpy()[0]:.2e} Pa')

# Pin center region using proper velocity enforcement
center_point = [0.75, 0.75, 1.3]
pin_size = [0.015, 0.015, 0.015]
print(f'\nPinning region {pin_size} around {center_point}')

solver.enforce_particle_velocity_translation(
    point=center_point,
    size=pin_size,
    velocity=[0.0, 0.0, 0.0],
    start_time=0.0,
    end_time=1000.0,
    device='cpu'
)

# Run simulation
dt = 0.0005
for step in range(100):
    solver.p2g2p(step=step, dt=dt)


    if step in [0, 25, 50, 99]:
        F = solver.mpm_state.particle_F.numpy()
        stress = solver.mpm_state.particle_stress.numpy()
        pos = solver.mpm_state.particle_x.numpy()

        # Check a few key particles
        # Center particle is around index n_particles // 2
        center_idx = n_particles // 2
        test_indices = [
            center_idx,  # center (pinned region)
            center_idx - 30,  # above
            center_idx + 30,  # below
            0,  # corner
            n_particles - 1  # opposite corner
        ]

        print(f'\n=== Step {step+1} ===')
        z_coords = pos[:, 2]
        print(f'Z range: [{z_coords.min():.4f}, {z_coords.max():.4f}]')

        stress_norms = np.linalg.norm(stress, axis=(1,2))
        print(f'Stress norms: min={stress_norms.min():.2f}, max={stress_norms.max():.2f}, mean={stress_norms.mean():.2f}')

        nonzero_stress = np.sum(stress_norms > 1.0)
        print(f'Particles with stress > 1 Pa: {nonzero_stress} / {n_particles}')

        if step == 99:
            # Detailed check
            print(f'\nDetailed particle info:')
            for idx in test_indices[:3]:
                F_val = F[idx]
                F11, F22 = F_val[0,0], F_val[1,1]
                print(f'  Particle {idx}: F11={F11:.4f}, F22={F22:.4f}, stress_norm={stress_norms[idx]:.2f}')

            # Check if there's any in-plane deformation
            F11_vals = F[:, 0, 0]
            F22_vals = F[:, 1, 1]
            print(f'\nIn-plane F components:')
            print(f'  F11: min={F11_vals.min():.6f}, max={F11_vals.max():.6f}')
            print(f'  F22: min={F22_vals.min():.6f}, max={F22_vals.max():.6f}')

print(f'\n=== Summary ===')
final_stress = solver.mpm_state.particle_stress.numpy()
final_stress_norms = np.linalg.norm(final_stress, axis=(1,2))
print(f'Final max stress: {final_stress_norms.max():.2f} Pa')
if final_stress_norms.max() > 1.0:
    print('✓ Stress is being computed!')
    max_idx = np.argmax(final_stress_norms)
    print(f'  Max stress at particle {max_idx}')
else:
    print('✗ No significant stress detected - particles may not be coupled properly')

