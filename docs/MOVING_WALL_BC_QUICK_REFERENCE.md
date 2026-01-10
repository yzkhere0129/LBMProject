# Moving Wall Boundary Condition - Quick Reference

## TL;DR

```cpp
#include "physics/fluid_lbm.h"
#include "core/streaming.h"

// Create solver with walls
FluidLBM fluid(nx, ny, nz, viscosity, density,
               BoundaryType::WALL,
               BoundaryType::WALL,
               BoundaryType::WALL);

// Initialize
fluid.initialize(density, 0.0, 0.0, 0.0);

// Make top wall move
fluid.setMovingWall(Streaming::BOUNDARY_Y_MAX, 0.1, 0.0, 0.0);

// Time loop
for (int step = 0; step < n_steps; ++step) {
    fluid.collisionBGK();
    fluid.streaming();
    fluid.applyBoundaryConditions(1);  // Apply BCs
    fluid.computeMacroscopic();
}
```

---

## Wall Direction Constants

| Constant | Wall Position | Description |
|----------|--------------|-------------|
| `Streaming::BOUNDARY_X_MIN` | x = 0 | Left wall |
| `Streaming::BOUNDARY_X_MAX` | x = nx-1 | Right wall |
| `Streaming::BOUNDARY_Y_MIN` | y = 0 | Bottom wall |
| `Streaming::BOUNDARY_Y_MAX` | y = ny-1 | Top wall |
| `Streaming::BOUNDARY_Z_MIN` | z = 0 | Back wall |
| `Streaming::BOUNDARY_Z_MAX` | z = nz-1 | Front wall |

---

## Common Use Cases

### 1. Lid-Driven Cavity (Moving Top Wall)

```cpp
// Standard benchmark: top wall moves in x-direction
fluid.setMovingWall(Streaming::BOUNDARY_Y_MAX, U_lid, 0.0, 0.0);
```

**Reynolds number:**
```cpp
float Re = U_lid * L / viscosity;  // L = domain height
```

**Typical values:**
- Re = 100: U_lid ≈ 0.1 m/s, L = 0.1 m, ν = 1e-4 m²/s
- Re = 1000: U_lid ≈ 1.0 m/s, L = 0.1 m, ν = 1e-4 m²/s

### 2. Couette Flow (Two Moving Walls)

```cpp
// Top wall moves right, bottom wall moves left
fluid.setMovingWall(Streaming::BOUNDARY_Y_MAX, +U_wall, 0.0, 0.0);
fluid.setMovingWall(Streaming::BOUNDARY_Y_MIN, -U_wall, 0.0, 0.0);
```

**Analytical solution:**
```
u(y) = U_wall * (2*y/H - 1)  // Linear velocity profile
```

### 3. Shear-Driven Flow (Parallel Walls)

```cpp
// Both walls move in same direction at different speeds
fluid.setMovingWall(Streaming::BOUNDARY_Y_MAX, U_top, 0.0, 0.0);
fluid.setMovingWall(Streaming::BOUNDARY_Y_MIN, U_bottom, 0.0, 0.0);
```

---

## Stability Guidelines

### CFL Condition
```cpp
float CFL = U_wall * dt / dx;
// CFL < 0.3 for stability
// CFL < 0.1 for accuracy
```

### Reynolds Number Limits
```cpp
// LBM stable up to:
// Re ~ 1000 - 5000 (depends on resolution)
// For higher Re, use turbulence model
```

### Grid Resolution
```cpp
// Minimum cells across channel:
int min_cells = 16;  // For Re < 100
int min_cells = 32;  // For Re ~ 1000
int min_cells = 64;  // For Re > 1000
```

---

## Troubleshooting Checklist

- [ ] `applyBoundaryConditions(1)` called AFTER `streaming()`?
- [ ] Boundary type parameter = 1 (not 0)?
- [ ] Wall velocity within CFL limit?
- [ ] Sufficient grid resolution (>16 cells)?
- [ ] `setMovingWall()` called BEFORE time loop?
- [ ] Correct wall direction constant used?

---

## Expected Results (Lid-Driven Cavity)

### Qualitative
- Primary vortex beneath moving lid
- Velocity maximum at y ≈ 0.5-0.6
- Secondary vortices in corners (Re > 100)

### Quantitative (Re = 100, nx=ny=128)
```
Centerline velocity profile:
y = 0.0:  u_x ≈ 0.00 (no-slip wall)
y = 0.5:  u_x ≈ 0.21 (max velocity)
y = 1.0:  u_x ≈ 0.10 (moving wall)
```

### Convergence Check
```cpp
// Monitor max velocity over time
float u_max = *std::max_element(h_ux.begin(), h_ux.end());

// Steady state when:
// |u_max(t) - u_max(t-100)| / u_max(t) < 1e-4
```

---

## File Locations

**Implementation:**
- `include/physics/fluid_lbm.h` - `setMovingWall()` declaration
- `src/physics/fluid/fluid_lbm.cu` - `setMovingWall()` implementation

**Boundary Kernels:**
- `include/core/boundary_conditions.h` - Zou-He velocity BC
- `src/core/boundary/boundary_conditions.cu` - Kernel implementations

**Tests:**
- `tests/unit/fluid/test_moving_wall_bc.cu` - Unit tests
- `test_moving_wall_simple.cu` - Standalone test

**Documentation:**
- `docs/MOVING_WALL_BC_IMPLEMENTATION.md` - Full documentation

---

## API Reference

### Function Signature
```cpp
void FluidLBM::setMovingWall(
    unsigned int wall_direction,  // Which wall (use Streaming::BOUNDARY_*)
    float ux_wall,                // Wall velocity x-component [m/s]
    float uy_wall,                // Wall velocity y-component [m/s]
    float uz_wall                 // Wall velocity z-component [m/s]
);
```

### Example: 3D Lid-Driven Cavity
```cpp
// Domain: 64 x 64 x 64
const int nx = 64, ny = 64, nz = 64;
const float nu = 1e-4f;      // m²/s
const float rho = 1000.0f;   // kg/m³
const float U_lid = 0.1f;    // m/s

// Create solver
FluidLBM fluid(nx, ny, nz, nu, rho,
               BoundaryType::WALL,
               BoundaryType::WALL,
               BoundaryType::WALL);

// Initialize
fluid.initialize(rho, 0.0, 0.0, 0.0);

// Set moving top wall
fluid.setMovingWall(Streaming::BOUNDARY_Y_MAX, U_lid, 0.0, 0.0);

// Run to steady state
const int max_steps = 10000;
for (int step = 0; step < max_steps; ++step) {
    fluid.collisionBGK();
    fluid.streaming();
    fluid.applyBoundaryConditions(1);
    fluid.computeMacroscopic();

    // Check convergence every 100 steps
    if (step % 100 == 0) {
        // Extract and check velocity field
    }
}
```

---

## Physics Summary

**What it does:** Enforces prescribed velocity at wall boundaries

**How it works:** Uses Zou-He velocity BC (equilibrium extrapolation)

**When to use:** Lid-driven cavity, Couette flow, belt-driven flows

**Limitations:**
- First-order accurate (O(Δx) error)
- Best for moderate Re (< 1000 without turbulence model)
- Corners may have artifacts (exclude from validation)

**Validation:** Ghia et al. (1982) lid-driven cavity benchmark
