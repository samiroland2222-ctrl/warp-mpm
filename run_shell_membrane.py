"""
Example: Simple Shell (Membrane) Simulation
Demonstrates how to use the codimensional MPM features for thin shells/membranes.
"""
import argparse
import threading
from argparse import ArgumentParser

import numpy as np
import torch
import warp as wp
from mpm_solver_warp import MPM_Simulator_WARP
from mpm_viewer import MPM_Viewer

wp.init()

def create_rectangular_membrane(center, width, height, particle_spacing):
    """Create a rectangular membrane of particles"""
    nx = int(width / particle_spacing)
    ny = int(height / particle_spacing)

    positions = []
    fibers = []
    normals = []

    for i in range(nx):
        for j in range(ny):
            x = center[0] - width/2 + i * particle_spacing
            y = center[1] - height/2 + j * particle_spacing
            z = center[2]

            positions.append([x, y, z])
            fibers.append([1.0, 0.0, 0.0])  # Fibers along x-axis
            normals.append([0.0, 0.0, 1.0])  # Normal along z-axis

    return np.array(positions), np.array(fibers), np.array(normals)

should_stop = False

def main(steps: int=100, viewer: MPM_Viewer = None):
    print("=" * 60)
    print("Shell MPM Example: Rectangular Membrane")
    print("=" * 60)

    # Simulation parameters
    n_grid = 64
    grid_lim = 1.5
    dt = 0.001
    n_steps = 50
    device = "cpu"  # Use CPU for this example

    print(f"\nDevice: {device}")

    # Create membrane geometry
    # Adjust spacing to match grid: dx = 1.5/128 ≈ 0.0117, use spacing ≈ 0.01
    membrane_spacing = 0.01  # 1cm spacing
    positions, fibers, normals = create_rectangular_membrane(
        center=[0.75, 0.75, 1.3],  # Center in smaller domain
        width=0.6,  # Larger membrane
        height=0.6,
        particle_spacing=membrane_spacing
    )

    n_particles = len(positions)
    print(f"Membrane particles: {n_particles}")

    # Initialize solver
    solver = MPM_Simulator_WARP(n_particles, n_grid=n_grid, grid_lim=grid_lim, device=device)

    # Set particle positions
    solver.mpm_state.particle_x = wp.array(positions, dtype=wp.vec3, device=device)

    # Set particle volumes (thin membrane, so small volume per particle)
    membrane_thickness = 0.002  # 2mm thick
    particle_volume = membrane_spacing * membrane_spacing * membrane_thickness
    volumes = np.ones(n_particles) * particle_volume
    solver.mpm_state.particle_vol = wp.array(volumes, dtype=float, device=device)

    # Initialize as shell particles
    solver.initialize_shell_particles(
        particle_indices=range(n_particles),
        fiber_directions=fibers,
        normal_directions=normals
    )

    # Set material properties (rubber-like material)
    solver.set_parameters_dict({
        'material': 'jelly',  # Use elastic model
        'E': 1e6,            # 1 MPa (soft rubber)
        'nu': 0.4,           # Nearly incompressible
        'density': 1100.0,   # kg/m³ (rubber density)
        'g': [0.0, 0.0, -9.8]  # Gravity
    })

    # Set anisotropy (membrane is 2x stiffer along fiber direction)
    solver.mpm_model.anisotropy_factor = 2.0

    # Finalize material parameters
    solver.finalize_mu_lam_bulk()

    # Validate initial state before running
    solver.validate_initial_state()

    # Add ground plane
    solver.add_surface_collider(
        point=(0.0, 0.0, 0.1),
        normal=(0.0, 0.0, 1.0),
        surface='slip',
        friction=0.1
    )

    # Add bounding box
    solver.add_bounding_box()

    # Pin the center particle using proper velocity enforcement
    # This creates constraint forces that propagate through the grid
    center_point = [0.75, 0.75, 1.3]
    pin_size = [0.015, 0.015, 0.015]  # Small region around center
    solver.enforce_particle_velocity_translation(
        point=center_point,
        size=pin_size,
        velocity=[0.0, 0.0, 0.0],
        start_time=0.0,
        end_time=1000.0,
        device=device
    )

    print(f"\nSimulation setup complete!")
    print(f"  Grid: {n_grid}³")
    print(f"  Particles: {n_particles}")
    print(f"  Material: Elastic membrane")
    print(f"  Anisotropy: {solver.mpm_model.anisotropy_factor}x")
    print(f"  Time step: {dt}s")
    print(f"  Steps: {n_steps}")
    print(f"  Pinned region: {pin_size} around {center_point}")

    if viewer:
        viewer.add_colliders(solver.collider_params)

    # Run simulation
    print(f"\nRunning simulation...")
    step = 0
    while (steps is None or step < steps) and not should_stop:
        solver.p2g2p(step=step, dt=dt)
        step += 1


        if viewer:
            viewer.update_data(solver.mpm_state.particle_x.numpy(),
                               solver.mpm_state.particle_v.numpy(),
                               solver.mpm_state.particle_F.numpy())
        else:
            if step % 10 == 0:
                # Get current positions
                positions_current = solver.mpm_state.particle_x.numpy()
                z_min = positions_current[:, 2].min()
                z_max = positions_current[:, 2].max()
                print(f"  Step {step:3d}: z_range = [{z_min:.3f}, {z_max:.3f}]")

    # Show final statistics
    final_positions = solver.mpm_state.particle_x.numpy()
    final_velocities = solver.mpm_state.particle_v.numpy()

    print(f"\nFinal Statistics:")
    print(f"  Position range:")
    print(f"    x: [{final_positions[:, 0].min():.3f}, {final_positions[:, 0].max():.3f}]")
    print(f"    y: [{final_positions[:, 1].min():.3f}, {final_positions[:, 1].max():.3f}]")
    print(f"    z: [{final_positions[:, 2].min():.3f}, {final_positions[:, 2].max():.3f}]")
    print(f"  Max velocity: {np.linalg.norm(final_velocities, axis=1).max():.3f} m/s")


if __name__ == "__main__":

    arg_parser = ArgumentParser()
    arg_parser.add_argument("--n_steps", type=int, default=-1, help="number of steps, or -1 to run forever")
    arg_parser.add_argument("--headless", action=argparse.BooleanOptionalAction, help="run in headless mode (implies --steps 100 unless overridden")

    args = arg_parser.parse_args()

    steps = None if args.n_steps < 0 else args.n_steps
    if args.headless and steps is None:
        steps = 100

    if args.headless:
        main(steps=steps)
    else:
        viewer = MPM_Viewer()
        simulation_thread = threading.Thread(target=main, kwargs={"steps": steps, "viewer": viewer})
        simulation_thread.start()
        viewer.launch_window()
        should_stop = True


