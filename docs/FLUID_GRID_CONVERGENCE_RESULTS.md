# Fluid Grid Convergence Study Results

**Date:** 2026-01-10
**Test:** `test_fluid_grid_convergence`
**Objective:** Validate second-order spatial accuracy of fluid LBM solver

---

## Executive Summary

This document presents the results of the fluid grid convergence study for the LBM-CUDA Poiseuille flow solver. The test validates that the fluid solver achieves second-order spatial accuracy by comparing numerical solutions against the analytical parabolic velocity profile on four grid resolutions.

**Status:** Test implementation COMPLETE and ready to run
**Expected Result:** Convergence order p ≈ 2.00 (same as thermal solver)

---

## Test Configuration

### Physical Problem: 2D Poiseuille Flow

Poiseuille flow is the laminar flow between two parallel plates driven by a constant body force. It is a fundamental benchmark for viscous incompressible flow solvers.

**Governing Equation:**
```
∂u/∂t + (u·∇)u = -∇p/ρ + ν∇²u + F
```

**Analytical Solution:**
```
u(y) = u_max · [1 - (2y/H - 1)²]
u_max = F·H²/(8·ν)
```

where:
- `y`: Distance from bottom wall [m]
- `H`: Channel height [m]
- `F`: Body force per unit mass [m/s²]
- `ν`: Kinematic viscosity [m²/s]
- `u_max`: Maximum velocity at centerline [m/s]

### Test Parameters

| Parameter | Value | Unit | Description |
|-----------|-------|------|-------------|
| Channel height (H) | 100 | μm | Distance between parallel walls |
| Density (ρ₀) | 1000 | kg/m³ | Reference fluid density |
| Kinematic viscosity (ν) | 1.0×10⁻⁵ | m²/s | Fluid viscosity |
| Body force (F) | 1000 | m/s² | Constant driving force |
| Reynolds number (Re) | ~8.3 | - | Based on u_max and H |
| Analytical u_max | 0.125 | m/s | Maximum velocity at centerline |

**Note:** Low Reynolds number (Re ≈ 8.3) ensures well-resolved laminar flow suitable for convergence testing.

### Grid Resolutions

Four systematically refined grids are tested:

| Grid | ny (cells) | dy (μm) | Refinement Ratio | Domain |
|------|------------|---------|------------------|---------|
| Coarse | 25 | 4.0 | - | 3×25×3 |
| Medium | 50 | 2.0 | 2× | 3×50×3 |
| Fine | 100 | 1.0 | 2× | 3×100×3 |
| Finest | 200 | 0.5 | 2× | 3×200×3 |

**Grid Design:**
- Resolution across channel (y-direction) is varied
- Minimal resolution in x and z (periodic boundaries)
- Uniform grid spacing: dx = dy = dz
- Consistent refinement ratio of 2× between levels

### Computational Parameters

For each grid resolution, the following adaptive parameters are used:

**Timestep (CFL-based):**
```
CFL = 0.1 (conservative for accuracy)
dt = CFL × dx²/(ν × 6)
```

**LBM Relaxation:**
```
nu_lattice = ν × dt / dx²
tau = 3 × nu_lattice + 0.5
omega = 1 / tau
```

**Simulation Time:**
```
t_diff = H²/ν  (diffusion time scale)
t_final = 5 × t_diff  (run for 5 diffusion times to reach steady state)
```

---

## Methodology

### 1. Problem Setup

1. **Initialize** fluid solver with zero velocity
2. **Apply** no-slip boundary conditions on y-walls (top/bottom)
3. **Apply** periodic boundary conditions in x and z directions
4. **Drive** flow with constant body force F in x-direction

### 2. Time Integration

1. Apply boundary conditions (bounce-back at walls)
2. BGK collision with body force
3. Streaming step
4. Repeat until steady state (5 diffusion times)

### 3. Error Analysis

**Extract velocity profile:**
- Sample u_x at centerline (i=nx/2, k=nz/2) across channel (varying j)
- Compare with analytical solution at cell centers

**Compute L2 error:**
```
L2 error = sqrt(Σ(u_num - u_ana)²) / sqrt(Σ u_ana²)
```

**Compute convergence order:**
```
p = log(E_coarse / E_fine) / log(dx_coarse / dx_fine)
```

For second-order accuracy, expect p ≈ 2.0.

---

## Expected Results

Based on the thermal grid convergence study (which achieved p = 2.00), we expect:

### Convergence Table (Expected)

| Grid | ny | dy [μm] | L2 Error [%] | Order p |
|------|-----|---------|--------------|---------|
| Coarse | 25 | 4.0 | ~2.0% | - |
| Medium | 50 | 2.0 | ~0.5% | ~2.0 |
| Fine | 100 | 1.0 | ~0.13% | ~2.0 |
| Finest | 200 | 0.5 | ~0.03% | ~2.0 |

**Expected average convergence order:** p ≈ 2.00 ± 0.1

### Acceptance Criteria

The test will PASS if all three criteria are met:

1. ✓ **Convergence order:** 1.8 ≤ p ≤ 2.2 (second-order accuracy)
2. ✓ **Finest grid error:** L2 error < 0.5%
3. ✓ **Monotonic decrease:** Error decreases with each refinement

---

## Implementation Details

### Files Created

1. **Analytical solution header:**
   ```
   /home/yzk/LBMProject/tests/validation/analytical/poiseuille.h
   ```
   - Contains analytical Poiseuille flow functions
   - Provides velocity profile, u_max calculation, L2 error computation
   - Includes Reynolds number and flow rate calculations

2. **Test executable:**
   ```
   /home/yzk/LBMProject/tests/validation/test_fluid_grid_convergence.cu
   ```
   - Main convergence study implementation
   - Runs 4 grid resolutions systematically
   - Computes errors and convergence orders
   - Generates detailed formatted output

3. **CMake integration:**
   - Added to `tests/validation/CMakeLists.txt`
   - Target: `test_fluid_grid_convergence`
   - Test name: `FluidGridConvergenceValidation`
   - Labels: `validation;convergence;grid;fluid;poiseuille;critical`
   - Timeout: 20 minutes

### How to Run

**Build the test:**
```bash
cd /home/yzk/LBMProject/build_test
cmake ..
cmake --build . --target test_fluid_grid_convergence
```

**Run the test:**
```bash
cd /home/yzk/LBMProject/build_test/tests/validation
./test_fluid_grid_convergence
```

**Run via CTest:**
```bash
cd /home/yzk/LBMProject/build_test
ctest -R FluidGridConvergenceValidation -V
```

---

## Comparison to Thermal Solver

The thermal grid convergence study (completed earlier) provides a direct comparison:

| Metric | Thermal Solver | Fluid Solver (Expected) |
|--------|----------------|-------------------------|
| Test problem | 1D Gaussian diffusion | 2D Poiseuille flow |
| Analytical solution | T(x,t) = T₀ + A·exp(-x²/2σ(t)²) | u(y) = u_max·[1-(2y/H-1)²] |
| Convergence order | p = 2.00 | p ≈ 2.00 |
| Finest grid error | < 1% | < 0.5% (stricter) |
| Grid resolutions | 25, 50, 100, 200 | 25, 50, 100, 200 |
| Physical process | Heat diffusion | Viscous flow |

**Key insight:** Both thermal and fluid solvers use the same LBM framework with BGK collision. The second-order accuracy comes from the Chapman-Enskog expansion, which applies to both D3Q7 (thermal) and D3Q19 (fluid) lattices.

---

## Theoretical Background

### Why Second-Order?

The Lattice Boltzmann Method achieves second-order spatial accuracy through the Chapman-Enskog multi-scale expansion:

1. **Zeroth order:** Recovers mass conservation
2. **First order:** Recovers Euler equations (inviscid flow)
3. **Second order:** Recovers Navier-Stokes equations with correct viscosity

The BGK collision operator with equilibrium distribution:
```
f_i^eq = w_i · ρ · [1 + 3(e_i·u) + 9/2(e_i·u)² - 3/2(u·u)]
```

ensures that:
- Spatial discretization error: O(Δx²)
- Temporal discretization error: O(Δt²)

### Poiseuille Flow Validity

Poiseuille flow is ideal for convergence testing because:

1. **Exact analytical solution** exists (no modeling uncertainty)
2. **Steady state** eliminates temporal discretization effects
3. **Simple geometry** isolates spatial discretization errors
4. **Low Reynolds number** ensures numerical stability
5. **No-slip boundaries** test boundary condition implementation

The parabolic profile is sensitive to:
- Viscosity implementation (relaxation time τ)
- Boundary condition accuracy (bounce-back at walls)
- Force implementation (Guo forcing scheme)

### Convergence Theory

For a second-order method solving Poiseuille flow:

**Error scaling:**
```
E(Δx) ≈ C · Δx²
```

**Convergence order:**
```
p = log(E_coarse / E_fine) / log(Δx_coarse / Δx_fine)
```

For consecutive 2× refinements:
```
p ≈ 2.0 (ideal)
1.8 ≤ p ≤ 2.2 (acceptable)
```

Deviations from p = 2.0 can occur due to:
- Boundary condition approximations (reduces p)
- Insufficient steady-state convergence (noise)
- Compressibility effects (LBM is weakly compressible)
- Floating-point precision limits (finest grids)

---

## Diagnostic Output

The test produces detailed formatted output:

```
╔═══════════════════════════════════════════════════════════════╗
║       GRID CONVERGENCE STUDY - FLUID LBM SOLVER               ║
╚═══════════════════════════════════════════════════════════════╝

=== Problem Setup ===
Channel height: 100 μm
Density: ρ₀ = 1000 kg/m³
Kinematic viscosity: ν = 0.01 mm²/s
Body force: F = 1000 m/s²
Analytical u_max = 0.125000 m/s
Reynolds number: Re = 8.33

[For each grid:]
=== Grid: Coarse ===
  ny = 25 cells
  dy = 4.00 μm
  dx = 4.00 μm
  dt = XXX ns
  CFL = 0.1
  nu_lattice = XXX
  tau = XXX
  omega = XXX
  Steps = XXXX
  Simulation time = XXX μs
  Diffusion time = XXX μs
  u_max (LBM): X.XXXXXX m/s
  u_max (Analytical): 0.125000 m/s
  u_max error: X.XX%
  L2 error: X.XX%
  Reynolds number: 8.33

╔═══════════════════════════════════════════════════════════════╗
║                    CONVERGENCE ANALYSIS                       ║
╚═══════════════════════════════════════════════════════════════╝

=== Grid Refinement Results ===
        Grid          ny        dy [μm]     L2 Error [%]         Order p
---------------------------------------------------------------------------
      Coarse          25           4.00            X.XX               -
      Medium          50           2.00            X.XX            X.XX
        Fine         100           1.00            X.XX            X.XX
      Finest         200           0.50            X.XX            X.XX

=== Summary ===
Average convergence order (refined grids): X.XX
Finest grid error: X.XX%
Error decreases monotonically: YES

=== Acceptance Criteria ===
✓ Convergence order 1.8-2.2: PASS (actual: X.XX)
✓ Finest grid error < 0.5%: PASS (actual: X.XX%)
✓ Monotonic error decrease: PASS

╔═══════════════════════════════════════════════════════════════╗
║                    ✓ TEST PASSED                              ║
║      Grid independence verified (second-order convergence)    ║
╚═══════════════════════════════════════════════════════════════╝
```

---

## Next Steps

After confirming second-order convergence (p ≈ 2.0):

### Phase 1 Completion

1. **Document results** in this file with actual run data
2. **Compare** with thermal solver results (p = 2.00)
3. **Mark** Phase 1 fluid validation as COMPLETE

### Future Validation Tests

1. **Lid-driven cavity flow** (Re = 100, 400, 1000)
   - Validate against Ghia et al. (1982) benchmark
   - Test pressure-velocity coupling
   - Verify vortex formation

2. **Timestep convergence study**
   - Validate temporal accuracy (first-order for BGK)
   - Similar to thermal timestep convergence test

3. **Taylor-Green vortex decay**
   - Validate viscous dissipation
   - Test periodic boundary conditions
   - Analytical solution available

4. **Couette flow**
   - Validate wall shear stress
   - Test moving wall boundary conditions

---

## References

### Analytical Solutions

- **White, F. M.** (2006). *Viscous Fluid Flow* (3rd ed.). McGraw-Hill.
  - Chapter 3: Exact solutions of Navier-Stokes equations
  - Poiseuille flow derivation and properties

### LBM Theory

- **Succi, S.** (2001). *The Lattice Boltzmann Equation for Fluid Dynamics and Beyond.* Oxford University Press.
  - Chapman-Enskog expansion and accuracy analysis
  - Chapter 3: From Boltzmann to Navier-Stokes

- **Krüger, T., Kusumaatmaja, H., Kuzmin, A., Shardt, O., Silva, G., & Viggen, E. M.** (2017). *The Lattice Boltzmann Method: Principles and Practice.* Springer.
  - Chapter 5: Boundary conditions (bounce-back accuracy)
  - Chapter 7: Benchmarking and validation

### Numerical Methods

- **Roache, P. J.** (1997). Quantification of uncertainty in computational fluid dynamics. *Annual Review of Fluid Mechanics*, 29(1), 123-160.
  - Grid convergence index methodology
  - Richardson extrapolation

- **He, X., & Luo, L. S.** (1997). A priori derivation of the lattice Boltzmann equation. *Physical Review E*, 55(6), R6333.
  - Theoretical foundation for second-order accuracy

### Benchmarks

- **Ghia, U., Ghia, K. N., & Shin, C. T.** (1982). High-Re solutions for incompressible flow using the Navier-Stokes equations and a multigrid method. *Journal of Computational Physics*, 48(3), 387-411.
  - Lid-driven cavity benchmark data

---

## Appendix: Code Snippets

### Analytical Solution Usage

```cpp
#include "analytical/poiseuille.h"

// Compute maximum velocity
float u_max = analytical::poiseuille_u_max(F, H, nu);

// Get velocity at specific y
float y = 0.5f * H;  // centerline
float u = analytical::poiseuille_velocity(y, H, u_max);

// Compute L2 error
float error = analytical::poiseuille_l2_error(u_numerical, u_analytical, n);

// Reynolds number
float Re = analytical::poiseuille_reynolds(u_max, H, nu);
```

### FluidLBM Setup for Poiseuille Flow

```cpp
#include "physics/fluid_lbm.h"

// Initialize with no-slip walls in y-direction
FluidLBM fluid(nx, ny, nz, nu, rho0,
               BoundaryType::PERIODIC,  // x: periodic
               BoundaryType::WALL,      // y: no-slip walls
               BoundaryType::PERIODIC,  // z: periodic
               dt, dx);

// Initialize with zero velocity
fluid.initialize(rho0, 0.0f, 0.0f, 0.0f);

// Time loop
for (int step = 0; step < num_steps; ++step) {
    fluid.applyBoundaryConditions(1);  // 1 = no-slip walls
    fluid.collisionBGK(body_force, 0.0f, 0.0f);  // Force in x-direction
    fluid.streaming();
}

// Extract velocity
std::vector<float> h_ux(num_cells);
std::vector<float> h_uy(num_cells);
std::vector<float> h_uz(num_cells);
fluid.copyVelocityToHost(h_ux.data(), h_uy.data(), h_uz.data());
```

---

**Status:** Documentation complete. Test ready to run.
**Next action:** Execute test and record actual convergence order in this document.
