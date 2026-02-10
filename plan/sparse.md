

### Summary of Tasks for the Programmer

1.  **Refactor** existing material math into `mpm_materials.py` (Pure Copy/Paste job first).
2.  **Refactor** grid access into accessor functions (Search/Replace job).
3.  **Implement** `get_stress_shell` using Anisotropic Plane Stress math.
4.  **Replace** the 3D grid arrays with `wp.HashGrid` + 1D Flat Arrays.

### Part 1: Implementing Codimensional (Shell) MPM

**Objective:** Teach the simulator to treat specific particles as 2D fabric sheets rather than sand or fluid.

#### Step 1: Isolate the Material Logic (Refactoring)
*Context: Currently, your stress calculation is a giant block of code inside the kernel.*

1.  **Write unit tests:** Before refactoring, write a few unit tests that call `compute_stress_from_F_trial` with known inputs and verify the outputs. This ensures you have a safety net when moving code around.
2.  **Create a new file** named `mpm_materials.py`.
3.  **Move the math:** Take the existing Neo-Hookean/Jelly stress logic from `compute_stress_from_F_trial` and move it into a standalone function annotated with `@wp.func` called `get_stress_volumetric()`.
4.  **Define the Interface:** Ensure this function takes only the necessary local data (F, mu, lam) and returns `stress` (mat33).
5.  **Verify:** Update the main solver to call this function. The simulation should run exactly as before. *Do not proceed until this passes.*

#### Step 2: Implement the Shell Data Structure
*Context: Cloth needs to know "which way is up" (Normal) and "which way the fibers run" (Anisotropy).*

1.  **Modify `MPMStateStruct`:** Add a new array `particle_fiber` (type `wp.vec3`).
2.  **Modify `MPMStateStruct`:** Add a new array `particle_type` (type `int`).
    *   `0` = Volumetric (fluid)
    *   `1` = Shell (cloth)
3.  **Create a helper:** Write a Python function `initialize_fabric_fibers()` that runs once at startup. It should iterate over cloth particles and set their `particle_fiber` to align with the fabric weave (usually derived from the UV coordinates of your CAD file).

#### Step 3: Implement the Shell Physics Kernel
1.  **In `mpm_materials.py`,** add a new function `@wp.func get_stress_shell()`.
2.  **The Logic:**
    *   This function should project the Deformation Gradient (`F`) onto the 2D plane defined by the particle's normal.
    *   It should apply a "Plane Stress" condition (meaning the material doesn't resist thinning, only stretching).
    *   **Anisotropy:** Scale the stress higher if the stretch direction aligns with the `particle_fiber` vector.
3.  **Wire it up:** In your main `compute_stress` kernel, add a simple switch:
    ```python
    if state.particle_type[p] == 1:
        stress = get_stress_shell(...)
    else:
        stress = get_stress_volumetric(...)
    ```

---

### Part 2: Switching from Dense to Sparse Grids

**Objective:** Allow the simulation to run high-resolution (2mm) grids without running out of RAM, by only allocating memory where the cloth actually touches.

#### Step 1: Abstract the Grid Access (The Interface)
*Context: Currently, the code accesses `grid_m[x,y,z]` directly. We need to hide this behind a function so we can change the storage method later without breaking the physics.*

1.  **Create a new file** named `mpm_grid.py`.
2.  **Create Accessor Functions:** Write `@wp.func` wrappers for reading/writing grid data:
    *   `get_grid_mass(state, x, y, z)`
    *   `add_grid_mass(state, x, y, z, val)`
    *   `get_grid_velocity(state, x, y, z)`
3.  **Refactor Main Solver:** Go through `mpm_solver_warp.py` and replace every `state.grid_m[i,j,k]` with these function calls.
4.  **Verify:** Run the sim. It should still work (using the old dense arrays underneath).

#### Step 2: Introduce the Sparse Structure
*Context: Use Warp’s built-in HashGrid to track active cells.*

1.  **In `MPM_Simulator_WARP` init:**
    *   Remove `self.mpm_state.grid_m = wp.zeros(...)`.
    *   Initialize `self.hash_grid = wp.HashGrid(dim_x, dim_y, dim_z, device)`.
    *   Allocate "Flat" arrays for mass and velocity. Instead of shape `(N,N,N)`, they will be shape `(MAX_ACTIVE_BLOCKS * BLOCK_SIZE)`.
2.  **Update the Accessors (in `mpm_grid.py`):**
    *   Change `get_grid_mass` to:
        1.  Compute the flat index using `hash_grid.grid_hash(x, y, z)`.
        2.  Return the value from the new flat array.

#### Step 3: Change the Loop Logic
*Context: P2G (Particle-to-Grid) stays mostly the same, but Grid Update and G2P need to change how they iterate.*

1.  **The "Build" Step:** At the very start of every time step, call `self.hash_grid.build(particle_positions)`. This figures out which cells are active.
2.  **Update P2G:** Use `wp.atomic_add` onto the flat arrays using the hash index.
3.  **Update Grid Processing:**
    *   *Old Way:* `wp.launch(dim=(N,N,N))` (Iterates empty space).
    *   *New Way:* Use `wp.launch(dim=hash_grid.block_count())`.
    *   Inside the kernel, use `hash_grid.get_block_index()` to find where you are in 3D space.

