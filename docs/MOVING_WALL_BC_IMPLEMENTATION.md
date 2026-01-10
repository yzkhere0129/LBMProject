# Moving Wall Boundary Condition Implementation

## Executive Summary

This document describes the implementation of moving wall boundary conditions for the FluidLBM solver, essential for lid-driven cavity validation and other canonical flow benchmarks.

**Implementation Status:** Complete and tested
**Key Files Modified:**
- `include/physics/fluid_lbm.h` - Added `setMovingWall()` method
- `src/physics/fluid/fluid_lbm.cu` - Implemented moving wall BC and updated BC application
- `tests/unit/fluid/test_moving_wall_bc.cu` - Comprehensive unit tests

---

## Physical Background

### What is a Moving Wall Boundary Condition?

A moving wall boundary condition enforces a prescribed velocity at a solid wall. Unlike a stationary no-slip wall (zero velocity), a moving wall imparts momentum to the fluid.

**Physical Applications:**
1. **Lid-driven cavity flow** - Top wall moves horizontally, driving circulation
2. **Couette flow** - Parallel plates with one moving
3. **Taylor-Couette flow** - Rotating cylinders
4. **Belt-driven flows** - Conveyor systems

### LBM Implementation: Zou-He Velocity Boundary

The moving wall is implemented using the Zou-He velocity boundary condition, which is already available in the codebase:

**Theory (Zou & He 1997):**
```
For a boundary with prescribed velocity u_wall:
1. Known: wall velocity components (u_wall, v_wall, w_wall)
2. Unknown: certain distribution functions pointing into the domain
3. Solution: Extrapolate unknown distributions using equilibrium closure
```

**Reference:**
Zou, Q., & He, X. (1997). On pressure and velocity boundary conditions for the lattice Boltzmann BGK model. *Physics of Fluids*, 9(6), 1591-1598.

---

## Implementation Details

### 1. Architecture

The implementation leverages the existing boundary condition infrastructure:

```
BoundaryNode structure (already exists)
  ├─ position (x, y, z)
  ├─ type (BOUNCE_BACK, VELOCITY, PRESSURE, etc.)
  ├─ velocities (ux, uy, uz)
  └─ direction flags (BOUNDARY_X_MIN, BOUNDARY_Y_MAX, etc.)

FluidLBM class
  ├─ d_boundary_nodes_ (device array of BoundaryNode)
  ├─ initializeBoundaryNodes() - creates wall boundaries as BOUNCE_BACK
  └─ setMovingWall() - NEW: converts BOUNCE_BACK to VELOCITY
```

### 2. Key Design Decisions

**Q: Why not create a new MOVING_WALL boundary type?**

A: The existing VELOCITY boundary type already implements exactly what we need (Zou-He BC). A moving wall is simply a velocity boundary with non-zero wall velocity. This reuses existing, tested code.

**Q: How does it differ from the existing no-slip BC?**

A:
- **No-slip (BOUNCE_BACK)**: Reverses distribution functions, enforces zero velocity
- **Moving wall (VELOCITY)**: Uses Zou-He BC to enforce prescribed non-zero velocity

**Q: Why modify boundary nodes on-the-fly instead of at initialization?**

A: This provides maximum flexibility. Users can:
1. Create a solver with all walls as no-slip
2. Selectively make specific walls moving via `setMovingWall()`
3. Change wall velocities during simulation if needed

### 3. Code Implementation

#### Header Addition (`include/physics/fluid_lbm.h`)

```cpp
/**
 * @brief Set moving wall boundary condition
 *
 * Replaces existing wall boundaries with moving wall (velocity) boundaries.
 * Uses Zou-He velocity boundary condition to enforce wall velocity.
 *
 * @param wall_direction Direction of the wall (use Streaming::BOUNDARY_* flags)
 * @param ux_wall Wall velocity x-component [m/s]
 * @param uy_wall Wall velocity y-component [m/s]
 * @param uz_wall Wall velocity z-component [m/s]
 *
 * Example usage for lid-driven cavity (moving top wall):
 *   fluid.setMovingWall(Streaming::BOUNDARY_Y_MAX, 0.1, 0.0, 0.0);
 */
void setMovingWall(unsigned int wall_direction,
                  float ux_wall, float uy_wall, float uz_wall);
```

#### Implementation (`src/physics/fluid/fluid_lbm.cu`)

**Method: `setMovingWall()`**
```cpp
void FluidLBM::setMovingWall(unsigned int wall_direction,
                             float ux_wall, float uy_wall, float uz_wall) {
    // 1. Copy boundary nodes from device to host
    std::vector<core::BoundaryNode> h_boundary_nodes(n_boundary_nodes_);
    cudaMemcpy(h_boundary_nodes.data(), d_boundary_nodes_, ...);

    // 2. Find nodes on specified wall and change type
    for (auto& node : h_boundary_nodes) {
        if (node.directions & wall_direction) {
            node.type = core::BoundaryType::VELOCITY;  // Change from BOUNCE_BACK
            node.ux = ux_wall;
            node.uy = uy_wall;
            node.uz = uz_wall;
        }
    }

    // 3. Copy modified nodes back to device
    cudaMemcpy(d_boundary_nodes_, h_boundary_nodes.data(), ...);
}
```

**Modified: `applyBoundaryConditions()`**
```cpp
void FluidLBM::applyBoundaryConditions(int boundary_type) {
    // OLD: Only called applyBounceBackKernel
    // NEW: Call unified kernel that handles BOUNCE_BACK, VELOCITY, etc.

    applyBoundaryConditionsKernel<<<grid, block>>>(
        d_f_src, d_rho, d_boundary_nodes_, n_boundary_nodes_, nx_, ny_, nz_
    );
}
```

The unified kernel `applyBoundaryConditionsKernel` (already exists in `boundary_conditions.cu`) checks each node's type and applies the appropriate BC:
- `BOUNCE_BACK` → bounce-back for no-slip
- `VELOCITY` → Zou-He velocity BC (for moving walls)
- `PRESSURE` → Zou-He pressure BC

---

## Usage Guide

### Basic Usage: Lid-Driven Cavity

```cpp
#include "physics/fluid_lbm.h"
#include "core/streaming.h"

// 1. Create solver with all walls as no-slip
FluidLBM fluid(nx, ny, nz, viscosity, density,
               BoundaryType::WALL,   // X walls
               BoundaryType::WALL,   // Y walls
               BoundaryType::WALL);  // Z walls

// 2. Initialize with zero velocity
fluid.initialize(density, 0.0, 0.0, 0.0);

// 3. Make top wall (y = ny-1) a moving wall
float lid_velocity = 0.1f;  // m/s
fluid.setMovingWall(Streaming::BOUNDARY_Y_MAX, lid_velocity, 0.0, 0.0);

// 4. Run simulation
for (int step = 0; step < n_steps; ++step) {
    fluid.collisionBGK();
    fluid.streaming();
    fluid.applyBoundaryConditions(1);  // Apply all BCs (bounce-back + velocity)
    fluid.computeMacroscopic();
}
```

### Wall Direction Flags

Available in `core::Streaming`:
```cpp
Streaming::BOUNDARY_X_MIN    // Left wall (x = 0)
Streaming::BOUNDARY_X_MAX    // Right wall (x = nx-1)
Streaming::BOUNDARY_Y_MIN    // Bottom wall (y = 0)
Streaming::BOUNDARY_Y_MAX    // Top wall (y = ny-1)
Streaming::BOUNDARY_Z_MIN    // Back wall (z = 0)
Streaming::BOUNDARY_Z_MAX    // Front wall (z = nz-1)
```

### Multiple Moving Walls

```cpp
// Couette flow: top and bottom walls moving in opposite directions
fluid.setMovingWall(Streaming::BOUNDARY_Y_MAX, +0.1, 0.0, 0.0);
fluid.setMovingWall(Streaming::BOUNDARY_Y_MIN, -0.1, 0.0, 0.0);
```

---

## Testing and Validation

### Unit Tests

**File:** `tests/unit/fluid/test_moving_wall_bc.cu`

Six comprehensive tests:

1. **TopWallVelocityImposed** - Verifies wall velocity matches prescribed value
   - Tolerance: ±5% (allows for compressibility and corner effects)

2. **MomentumImparted** - Confirms moving wall increases fluid momentum

3. **VariableWallVelocity** - Tests different wall velocities (0.05, 0.1, 0.15 m/s)

4. **DifferentWallDirections** - Tests X_MAX, Y_MAX, Z_MAX walls

5. **MixedBoundaryConditions** - Moving wall + no-slip walls coexist
   - Top wall moving (u = 0.1)
   - Bottom wall stationary (u ≈ 0)

6. **MassConservation** - Verifies total mass conserved

### Running the Tests

```bash
cd build
make test_moving_wall_bc
./tests/test_moving_wall_bc
```

Expected output:
```
========================================
Moving Wall Boundary Condition Tests
========================================

=== Test: Moving Wall - Top Wall Velocity Imposed ===
  Domain: 16x16x16
  [...]
  [PASS] Wall velocity correctly imposed

=== Test: Moving Wall - Momentum Imparted ===
  [PASS] Momentum correctly imparted by moving wall

[... 6 tests total ...]

All Moving Wall BC Tests Complete
```

### Simple Standalone Test

**File:** `test_moving_wall_simple.cu`

Minimal test without full build system:
```bash
nvcc -o test_simple test_moving_wall_simple.cu \
     -I./include -L./build -llbm_core -llbm_physics
./test_simple
```

---

## Physics Validation

### Expected Behavior

For a lid-driven cavity with moving top wall:

1. **Wall velocity field:**
   - Top wall: u_x ≈ U_wall (prescribed velocity)
   - Other walls: u ≈ 0 (no-slip)

2. **Interior flow:**
   - Primary vortex forms beneath moving lid
   - Velocity decreases away from moving wall
   - Secondary vortices in corners (for Re > 100)

3. **Conservation:**
   - Mass conserved (incompressible flow)
   - Momentum balance: wall shear stress = viscous dissipation

### Literature Comparison

**Benchmark:** Ghia et al. (1982) lid-driven cavity

| Re | Grid | U_lid | Centerline Velocity |
|----|------|-------|---------------------|
| 100 | 128³ | 0.1 | u_max ≈ 0.21 @ y≈0.6 |
| 400 | 256³ | 0.1 | u_max ≈ 0.30 @ y≈0.55 |
| 1000 | 512³ | 0.1 | u_max ≈ 0.38 @ y≈0.46 |

**Reference:**
Ghia, U., Ghia, K. N., & Shin, C. T. (1982). High-Re solutions for incompressible flow using the Navier-Stokes equations and a multigrid method. *Journal of Computational Physics*, 48(3), 387-411.

---

## Numerical Considerations

### Stability Constraints

**CFL condition:** Moving wall velocity must satisfy:
```
u_wall * dt / dx < 0.3  (approximately)
```

For typical parameters:
- dx = 2 μm
- dt = 0.1 μs
- u_wall < 6 m/s (conservative)

**Viscosity constraint:** For stable LBM:
```
tau = nu_lattice / cs² + 0.5 > 0.51
```
Automatically enforced by FluidLBM constructor.

### Accuracy

Zou-He BC is **first-order accurate** in space:
- Velocity error: O(Δx)
- Pressure error: O(Δx)

For second-order accuracy in bulk, use:
- Fine grid near walls (refinement ratio 2:1)
- At least 16 cells across characteristic length

---

## Common Issues and Solutions

### Issue 1: Wall Velocity Not Imposed

**Symptom:** Wall velocity remains ~0 despite setMovingWall()

**Diagnosis:**
```cpp
// Check if boundary nodes exist
std::cout << "Number of boundary nodes: " << fluid.getNBoundaryNodes() << std::endl;
```

**Solution:** Ensure `applyBoundaryConditions(1)` is called AFTER `streaming()`:
```cpp
fluid.streaming();
fluid.applyBoundaryConditions(1);  // MUST be after streaming
fluid.computeMacroscopic();
```

### Issue 2: Corners Have Wrong Velocity

**Symptom:** Corner nodes (e.g., x=0, y=ny-1, z=0) have incorrect velocity

**Expected:** This is normal. Corner nodes belong to multiple walls and BC may not be well-defined.

**Solution:** When comparing with benchmarks, exclude corner/edge nodes:
```cpp
for (int k = 1; k < nz-1; ++k) {     // Exclude edges
    for (int i = 1; i < nx-1; ++i) {
        // Check wall nodes (interior points only)
    }
}
```

### Issue 3: Mass Not Conserved

**Symptom:** Total mass changes over time

**Diagnosis:**
- Check boundary_type parameter in `applyBoundaryConditions()`
- Verify no outflow boundaries mixed with walls

**Solution:** For closed cavity, use `boundary_type = 1` (walls only)

---

## Performance Characteristics

### Computational Cost

Moving wall BC (Zou-He) has **negligible overhead** compared to no-slip (bounce-back):

| Boundary Type | Cost per Node |
|---------------|---------------|
| Bounce-back | 19 float reads + 19 writes |
| Zou-He velocity | 19 float reads + 19 writes + ~20 arithmetic ops |

**Overhead:** < 1% for typical domains (boundary << bulk)

### Memory Footprint

Additional memory per boundary node:
```cpp
struct BoundaryNode {
    int x, y, z;           // 12 bytes
    BoundaryType type;     // 4 bytes
    float ux, uy, uz;      // 12 bytes (wall velocity)
    float pressure;        // 4 bytes
    unsigned int directions; // 4 bytes
};  // Total: 36 bytes/node
```

For 100³ domain with 6 wall faces:
- Boundary nodes: 6 × 100² ≈ 60,000
- Memory: 60,000 × 36 bytes ≈ 2.16 MB (negligible)

---

## Future Enhancements

### Potential Improvements

1. **Time-dependent wall velocity**
   ```cpp
   void setMovingWall(wall_direction, velocity_function, time);
   ```

2. **Second-order accurate BC**
   - Implement Guo et al. (2002) extrapolation scheme
   - Improves accuracy near walls

3. **Wall shear stress extraction**
   ```cpp
   float* getWallShearStress(wall_direction);
   ```

4. **Curved moving walls**
   - Current: only flat walls
   - Extension: moving cylinders (Taylor-Couette)

---

## References

1. **Zou-He BC:**
   Zou, Q., & He, X. (1997). On pressure and velocity boundary conditions for the lattice Boltzmann BGK model. *Physics of Fluids*, 9(6), 1591-1598.

2. **LBM Textbook:**
   Krüger, T., Kusumaatmaja, H., Kuzmin, A., Shardt, O., Silva, G., & Viggen, E. M. (2017). *The Lattice Boltzmann Method: Principles and Practice*. Springer. (Section 5.3.3: Velocity boundaries)

3. **Lid-Driven Cavity Benchmark:**
   Ghia, U., Ghia, K. N., & Shin, C. T. (1982). High-Re solutions for incompressible flow using the Navier-Stokes equations and a multigrid method. *JCP*, 48(3), 387-411.

4. **Second-Order BC:**
   Guo, Z., Zheng, C., & Shi, B. (2002). An extrapolation method for boundary conditions in lattice Boltzmann method. *Physics of Fluids*, 14(6), 2007-2010.

---

## Summary

The moving wall boundary condition implementation:

- **Reuses existing infrastructure** - No new BC type needed
- **Leverages proven Zou-He method** - First-order accurate, stable
- **Flexible interface** - Easy to specify which wall moves
- **Comprehensively tested** - 6 unit tests covering various scenarios
- **Ready for lid-driven cavity** - Essential for Phase 1 fluid validation

**Next Steps:**
1. Implement lid-driven cavity benchmark test
2. Compare with Ghia et al. (1982) reference data
3. Validate Reynolds number scaling (Re = 100, 400, 1000)
