# VOF+LBM Quick Reference Guide

**Project:** LBMProject - Metal AM CFD Framework
**Purpose:** Quick lookup for equations, parameters, and key implementation details
**Date:** 2025-12-17

---

## Mathematical Formulas Summary

### VOF Advection

```
Governing Equation:
  ∂f/∂t + ∇·(f·u) = S_evap + S_shrink

Upwind Discretization:
  f^{n+1} = f^n - dt[u·∂f/∂x + v·∂f/∂y + w·∂f/∂z]_upwind

CFL Constraint:
  v_max · dt / dx < 0.5
```

### Interface Geometry

```
Interface Normal:
  n = -∇f / |∇f|

Gradient (central difference):
  ∂f/∂x = [f(i+1) - f(i-1)] / (2dx)

Curvature:
  κ = ∇·n = ∂n_x/∂x + ∂n_y/∂y + ∂n_z/∂z

Analytical Values:
  Sphere (R):   κ = 2/R
  Cylinder (R): κ = 1/R
  Plane:        κ = 0
```

### Surface Tension (CSF Model)

```
Volumetric Force:
  F_σ = σ · κ · ∇f  [N/m³]

Units Check:
  [N/m] × [1/m] × [1/m] = [N/m³] ✓

Material Property (Ti-6Al-4V):
  σ = 1.5 N/m
```

### Marangoni Effect

```
Surface Tension Gradient:
  ∇_s σ = (dσ/dT) · ∇_s T

Tangential Temperature Gradient:
  ∇_s T = ∇T - (n·∇T)n

Volumetric Force:
  F_M = (dσ/dT) · ∇_s T · |∇f|  [N/m³]

Material Property (Ti-6Al-4V):
  dσ/dT = -3.5×10⁻⁴ N/(m·K)

Gradient Limiter:
  |∇T| < 5×10⁸ K/m (physical maximum from laser)
```

### Recoil Pressure

```
Vapor Pressure:
  P_sat = 101325 Pa (atmospheric)
  P_recoil = P_sat · exp((T - T_boil) / ΔT)

Where:
  T_boil = 3560 K (Ti-6Al-4V)
  ΔT = 100 K (temperature scale)

Volumetric Force:
  F_R = P_recoil · n · |∇f|  [N/m³]
```

### LBM Collision (D3Q19 BGK)

```
BGK with Guo Forcing:
  f_i^{n+1} = f_i^n + ω(f_i^eq - f_i^n) + S_i

Equilibrium:
  f_i^eq = w_i ρ [1 + c_i·u/c_s² + (c_i·u)²/(2c_s⁴) - u²/(2c_s²)]

Guo Force Term:
  S_i = w_i(1-ω/2)[(c_i-u)·F/c_s² + (c_i·u)(c_i·F)/c_s⁴]

Relaxation:
  ω = 1/τ
  τ = ν/c_s² + 0.5

D3Q19 Parameters:
  c_s² = 1/3
  w_0 = 1/3,  w_1-6 = 1/18,  w_7-18 = 1/36
```

### Unit Conversions

```
Velocity (Lattice → Physical):
  u_phys [m/s] = u_lattice × (dx/dt)

Force (Physical → Lattice):
  F_lattice = F_phys [N/m³] × (dt²/dx)

Viscosity (Physical → Lattice):
  ν_lattice = ν_phys × dt / dx²

Example:
  dx = 2×10⁻⁶ m,  dt = 1×10⁻⁷ s
  u_lattice = 0.1 → u_phys = 2.0 m/s
  ν_phys = 4.5×10⁻⁷ m²/s → ν_lattice = 0.01125
```

### Evaporation Mass Loss

```
Mass Flux (Langmuir-Knudsen):
  J_evap = 0.01 · M · P_sat / √(2πMRT)  [kg/(m²·s)]

Where:
  M = 0.0479 kg/mol (Ti-6Al-4V molar mass)
  R = 8.314 J/(mol·K)
  T > T_boil

VOF Update:
  df/dt = -J_evap / (ρ · dx)

Stability Limiter:
  |df| < 0.02 per time step (2% maximum)
```

### Solidification Shrinkage

```
Volume Change:
  dV/V = -β · dfl

Shrinkage Factor:
  β = 1 - ρ_liquid/ρ_solid ≈ 0.07 (metals)

VOF Update:
  df = β · (dfl/dt) · dt

Constraints:
  • Only at interface (0.01 < f < 0.99)
  • Only during solidification (dfl/dt < 0)
  • Limited to 1% change per step
```

---

## Key Parameters Table

### Material Properties (Ti-6Al-4V)

| Property | Symbol | Value | Units |
|----------|--------|-------|-------|
| Density (liquid) | ρ | 4110 | kg/m³ |
| Density (solid) | ρ_s | 4430 | kg/m³ |
| Kinematic viscosity | ν | 4.5×10⁻⁷ | m²/s |
| Surface tension | σ | 1.5 | N/m |
| dσ/dT | dσ/dT | -3.5×10⁻⁴ | N/(m·K) |
| Melting point | T_melt | 1933 | K |
| Boiling point | T_boil | 3560 | K |
| Solidus | T_solidus | 1878 | K |
| Liquidus | T_liquidus | 1933 | K |
| Thermal conductivity | κ | 35 | W/(m·K) |
| Specific heat | c_p | 670 | J/(kg·K) |
| Latent heat (fusion) | L_f | 2.86×10⁵ | J/kg |

### Numerical Parameters

| Parameter | Typical Value | Constraint | Purpose |
|-----------|---------------|------------|---------|
| Grid spacing (dx) | 2×10⁻⁶ m | > 0.1 μm | Resolution |
| Time step (dt) | 1×10⁻⁷ s | CFL < 0.5 | Stability |
| LBM relaxation (τ) | 0.51-2.0 | > 0.5 | Viscosity |
| LBM velocity (u_lattice) | < 0.15 | Ma < 0.3 | Compressibility |
| VOF CFL | < 0.5 | Upwind stability | Advection |
| Interface thickness | 3-5 cells | > 2 cells | Accuracy |
| Marangoni gradient limit | 5×10⁸ K/m | Physical | Stability |
| Force limiter (v_target) | 0.3-0.5 | Region-based | CFL |

### Domain Sizes (Typical)

| Application | Grid | dx | Domain Size | Cells |
|-------------|------|----|-----------|-|
| LIFT microdroplet | 32³ | 2 μm | 64×64×64 μm | 32K |
| LPBF melt pool | 64³ | 2 μm | 128×128×128 μm | 262K |
| DED track | 128×64×64 | 5 μm | 640×320×320 μm | 524K |
| Large-scale LPBF | 128³ | 2 μm | 256×256×256 μm | 2.1M |

---

## Algorithm Execution Checklist

### Initialization

- [ ] Allocate GPU memory
  - [ ] VOF fields: f, n, κ, cell_flags
  - [ ] LBM distributions: f_i (D3Q19), g_i (D3Q7)
  - [ ] Macroscopic: ρ, u, v, w, T, p
  - [ ] Forces: F_x, F_y, F_z

- [ ] Initialize fields
  - [ ] VOF fill level (droplet, pool, or uniform)
  - [ ] Temperature (ambient + laser spot)
  - [ ] Velocity (zero or prescribed)
  - [ ] Distributions (equilibrium)

- [ ] Set parameters
  - [ ] Material properties
  - [ ] Grid spacing dx, time step dt
  - [ ] LBM relaxation τ
  - [ ] Boundary conditions

### Time Step Loop

```
FOR n = 1 to num_steps:

  ┌─ THERMAL ──────────────────────────┐
  │ 1. Compute laser heat source       │
  │ 2. Thermal LBM collision           │
  │ 3. Thermal LBM streaming           │
  │ 4. Update temperature T^{n+1}      │
  └────────────────────────────────────┘
           ↓
  ┌─ PHASE CHANGE ─────────────────────┐
  │ 1. Compute liquid fraction fl      │
  │ 2. Compute phase rate dfl/dt       │
  │ 3. Compute evaporation flux J_evap │
  └────────────────────────────────────┘
           ↓
  ┌─ VOF ──────────────────────────────┐
  │ 1. Convert velocity (lattice→phys) │
  │ 2. Determine subcycling n_sub      │
  │ FOR sub = 1 to n_sub:              │
  │   • Advect fill level (upwind)     │
  │   • Reconstruct interface normals  │
  │   • Compute curvature              │
  │   • Update cell flags              │
  │ 3. Apply evaporation mass loss     │
  │ 4. Apply solidification shrinkage  │
  └────────────────────────────────────┘
           ↓
  ┌─ FORCES ───────────────────────────┐
  │ 1. Zero force arrays               │
  │ 2. Add surface tension (CSF)       │
  │ 3. Add Marangoni force             │
  │ 4. Add recoil pressure             │
  │ 5. Convert to lattice units        │
  │ 6. Apply CFL limiter               │
  └────────────────────────────────────┘
           ↓
  ┌─ FLUID LBM ────────────────────────┐
  │ 1. Collision (BGK + Guo forcing)   │
  │ 2. Streaming                       │
  │ 3. Boundary conditions             │
  │ 4. Update velocity u^{n+1}         │
  └────────────────────────────────────┘
           ↓
  ┌─ DIAGNOSTICS ──────────────────────┐
  │ 1. Check stability (NaN, CFL)      │
  │ 2. Compute derived quantities      │
  │ 3. Output fields (if needed)       │
  └────────────────────────────────────┘

END FOR
```

---

## File Locations Quick Reference

### Core Implementation

```
VOF Solver:
  Header:  include/physics/vof_solver.h
  Source:  src/physics/vof/vof_solver.cu
  Lines:   892 (implementation)

Surface Tension:
  Header:  include/physics/surface_tension.h
  Source:  src/physics/vof/surface_tension.cu
  Lines:   197

Marangoni:
  Header:  include/physics/marangoni.h
  Source:  src/physics/vof/marangoni.cu
  Lines:   366

Fluid LBM:
  Header:  include/physics/fluid_lbm.h
  Source:  src/physics/fluid/fluid_lbm.cu
  Lines:   700+

Multiphysics Coupling:
  Header:  include/physics/multiphysics_solver.h
  Source:  src/physics/multiphysics/multiphysics_solver.cu
  Lines:   1500+
```

### Tests

```
Unit Tests:
  tests/unit/vof/test_vof_advection.cu
  tests/unit/vof/test_vof_curvature.cu
  tests/unit/vof/test_vof_marangoni.cu
  tests/unit/vof/test_vof_surface_tension.cu

Validation Tests:
  tests/validation/vof/test_vof_advection_rotation.cu    (Zalesak)
  tests/validation/vof/test_vof_curvature_sphere.cu
  tests/validation/vof/test_vof_curvature_cylinder.cu

Integration Tests:
  tests/integration/multiphysics/test_vof_fluid_coupling.cu
  tests/integration/multiphysics/test_thermal_vof_coupling.cu
```

### Documentation

```
Algorithm Analysis:
  docs/VOF_LBM_ALGORITHM_ANALYSIS.md       (comprehensive)

Flowcharts:
  docs/VOF_LBM_ALGORITHM_FLOWCHARTS.md     (visual)

This Document:
  docs/VOF_LBM_QUICK_REFERENCE.md          (quick lookup)

Validation Results:
  tests/validation/vof/README.md
  benchmark/vof/VOF_RESULTS.md
```

---

## Common Debugging Scenarios

### Problem: VOF Mass Loss

**Symptoms:**
- `computeTotalMass()` decreases over time
- No evaporation source active

**Diagnostic Steps:**
1. Check advection CFL: `v_max × dt / dx`
2. Verify periodic boundaries in all 3 directions
3. Check for NaN in velocity field
4. Examine boundary cells for leakage

**Expected Behavior:**
- With periodic BC: mass conserved to machine precision
- With evaporation: controlled loss matching J_evap integral

---

### Problem: Velocity Explosion

**Symptoms:**
- Velocity goes from 1 m/s → 100 m/s in few steps
- Simulation diverges with NaN

**Diagnostic Steps:**
1. Check Marangoni gradient: `|∇T|` should be < 5e8 K/m
2. Verify force limiting is active
3. Check LBM stability: `τ > 0.5`
4. Examine recoil pressure magnitude

**Solution:**
- Enable adaptive CFL force limiter
- Reduce time step dt
- Check temperature gradient limiter

---

### Problem: Interface Diffusion

**Symptoms:**
- Sharp interface becomes smeared over time
- Fill level 0.01 < f < 0.99 region expands

**Diagnostic Steps:**
1. Check advection scheme (should be upwind, not central)
2. Verify VOF CFL < 0.5
3. Check for subcycling if velocities high
4. Examine interface thickness evolution

**Expected Behavior:**
- First-order upwind: inherently diffusive
- Interface thickness: ~2-3 cells stable

**Solutions:**
- Use subcycling for better CFL
- Consider higher-order schemes (future work)
- Accept controlled diffusion as trade-off for stability

---

### Problem: Incorrect Curvature

**Symptoms:**
- Measured κ differs significantly from analytical
- Surface tension forces incorrect

**Diagnostic Steps:**
1. Check sphere test: R=20 should give κ ≈ 0.1
2. Verify interface normal calculation
3. Examine fill level smoothness
4. Check resolution: cells/radius > 10?

**Expected Accuracy:**
- R > 15 cells: error < 10%
- R = 10 cells: error < 25%
- R < 5 cells: error > 50% (under-resolved)

---

## Performance Optimization Checklist

### Current Performance (64³ grid, RTX 3060)

- [ ] Total time per step: ~14 ms
- [ ] Breakdown:
  - [ ] Fluid LBM: 5 ms (35%)
  - [ ] VOF advection: 3 ms (21%)
  - [ ] Thermal LBM: 3 ms (21%)
  - [ ] Forces: 2 ms (14%)
  - [ ] Other: 1 ms (9%)

### Optimization Strategies

**Short-term (Kernel-level):**
- [ ] Fuse VOF kernels (advect + reconstruct + curvature)
  - Expected: 30% reduction in VOF time
  - Difficulty: Medium
  - Files: vof_solver.cu

- [ ] Add shared memory caching for stencil operations
  - Expected: 2x speedup for central differences
  - Difficulty: Medium-High
  - Impact: VOF, forces

- [ ] Optimize force accumulation memory access
  - Expected: 20% reduction in force time
  - Difficulty: Low
  - Files: surface_tension.cu, marangoni.cu

**Medium-term (Architecture):**
- [ ] Asynchronous streams for thermal + fluid LBM
  - Expected: 20% reduction in total time
  - Difficulty: Medium
  - Requires: Dependency analysis

- [ ] Persistent kernels for time stepping
  - Expected: 10% reduction (kernel launch overhead)
  - Difficulty: High
  - Complexity: Significant refactor

**Long-term (Scaling):**
- [ ] Multi-GPU domain decomposition
  - Expected: 1.8x per GPU (up to 4 GPUs)
  - Difficulty: High
  - Requires: MPI or NCCL

- [ ] Adaptive mesh refinement
  - Expected: 5-10x speedup (fewer cells)
  - Difficulty: Very High
  - Complexity: Major architectural change

---

## Validation Test Quick Run

```bash
# Build all tests
cd /home/yzk/LBMProject/build
cmake .. && cmake --build . -j4

# Run VOF unit tests
./tests/test_vof_advection
./tests/test_vof_curvature
./tests/test_vof_marangoni

# Run validation tests
./tests/test_vof_advection_rotation     # Zalesak's disk
./tests/test_vof_curvature_sphere       # Spherical curvature
./tests/test_vof_curvature_cylinder     # Cylindrical curvature

# Run integration tests
./tests/test_vof_fluid_coupling
./tests/test_thermal_vof_coupling

# Expected output: All tests PASS
```

**Pass Criteria:**
- Zalesak: Mass error < 5%, shape error < 0.15
- Sphere curvature: Error < 10% (R=20)
- Cylinder curvature: Error < 15% (R=16)
- Coupling: No NaN, stable for 1000 steps

---

## Critical Code Sections for Review

### 1. Upwind Advection (VOF Core)

**File:** `src/physics/vof/vof_solver.cu:24-110`

**Critical Lines:**
```cuda
// Line 83-86: Upwind derivative computation
if (u >= 0.0f) {
    dfdt_x = -u * (fill_level[idx] - fill_level[idx_x]) / dx;
} else {
    dfdt_x = -u * (fill_level[idx_x] - fill_level[idx]) / dx;
}

// Line 103: Time integration
float f_new = fill_level[idx] + dt * (dfdt_x + dfdt_y + dfdt_z);

// Line 109: Clamping to physical bounds
fill_level_new[idx] = fminf(1.0f, fmaxf(0.0f, f_new));
```

**Bug History:**
- Fixed: Missing minus sign in advection equation
- Fixed: Incorrect upwind direction for u < 0
- Current: Stable, validated

---

### 2. Marangoni Force (Interface Physics)

**File:** `src/physics/vof/marangoni.cu:28-141`

**Critical Lines:**
```cuda
// Line 95-104: Gradient limiting
float grad_T_mag = sqrtf(grad_T_x*grad_T_x + grad_T_y*grad_T_y + grad_T_z*grad_T_z);
if (grad_T_mag > max_gradient_limit) {
    float scale = max_gradient_limit / grad_T_mag;
    grad_T_x *= scale;
    grad_T_y *= scale;
    grad_T_z *= scale;
}

// Line 109-115: Tangential projection
float n_dot_gradT = n.x*grad_T_x + n.y*grad_T_y + n.z*grad_T_z;
float grad_Ts_x = grad_T_x - n_dot_gradT * n.x;
float grad_Ts_y = grad_T_y - n_dot_gradT * n.y;
float grad_Ts_z = grad_T_z - n_dot_gradT * n.z;

// Line 136-140: CSF formulation (NO division by h_interface)
float coeff = dsigma_dT * grad_f_mag;
force_x[idx] = coeff * grad_Ts_x;
force_y[idx] = coeff * grad_Ts_y;
force_z[idx] = coeff * grad_Ts_z;
```

**Physical Justification:**
- Gradient limit: Derived from laser parameters (P=20W, κ=35 W/(m·K), r=50μm)
- No h_phys division: |∇f| already provides delta function
- Tangential projection: Ensures force parallel to interface

---

### 3. Force Limiting (Stability Guardian)

**File:** `src/physics/multiphysics/multiphysics_solver.cu:382-440`

**Critical Lines:**
```cuda
// Line 399-400: Cell classification
float fill = fill_level[idx];
float liq_frac = liquid_fraction[idx];

// Region-based velocity targets
float v_target;
if (fill < interface_lo) {
    v_target = 0.1f; // Gas
} else if (fill > interface_hi) {
    if (liq_frac < 0.01f) {
        v_target = 0.0f; // Solid
    } else {
        v_target = v_target_bulk; // Bulk liquid (0.3)
    }
} else {
    v_target = v_target_interface; // Interface (0.5)
}

// Line 313-320: Compute scale factor
if (v_new > v_target) {
    float delta_v_allowed = fmaxf(0.0f, v_target - v_current);
    scale = delta_v_allowed / (f_mag + 1e-12f);
    scale = fminf(scale, 1.0f);
}

// Apply scaling
fx[idx] *= scale;
fy[idx] *= scale;
fz[idx] *= scale;
```

**Design Rationale:**
- Region-based: Different physics require different velocity limits
- Predictive: Estimates v_new before applying forces
- Conservative: Never amplifies forces (scale ≤ 1)
- Physically motivated: Represents viscous damping, inertia

---

## Glossary of Abbreviations

| Term | Meaning | Context |
|------|---------|---------|
| VOF | Volume of Fluid | Interface tracking method |
| LBM | Lattice Boltzmann Method | CFD solver |
| CSF | Continuum Surface Force | Surface tension model |
| BGK | Bhatnagar-Gross-Krook | Single-relaxation collision |
| CFL | Courant-Friedrichs-Lewy | Stability condition |
| D3Q19 | 3D 19-velocity | Fluid lattice |
| D3Q7 | 3D 7-velocity | Thermal lattice |
| AM | Additive Manufacturing | 3D printing |
| LPBF | Laser Powder Bed Fusion | AM process |
| DED | Directed Energy Deposition | AM process |
| LIFT | Laser-Induced Forward Transfer | Microdroplet AM |
| AoS | Array of Structures | Memory layout |
| SoA | Structure of Arrays | Memory layout |

---

## Contact and Further Information

**Project Documentation:**
- Main analysis: `docs/VOF_LBM_ALGORITHM_ANALYSIS.md`
- Flowcharts: `docs/VOF_LBM_ALGORITHM_FLOWCHARTS.md`
- This guide: `docs/VOF_LBM_QUICK_REFERENCE.md`

**Code Organization:**
- See: `CLAUDE.md` for project philosophy
- Architecture: Modular, extensible, tested
- Style: Concise, efficient, good taste

**Testing:**
- Unit tests: Validate individual components
- Validation tests: Compare to analytical solutions
- Integration tests: Multi-physics coupling

**Performance:**
- Current: 14 ms/step (64³ grid, RTX 3060)
- Target: < 10 ms/step (with optimizations)
- Scaling: Multi-GPU capable architecture

---

**Document Version:** 1.0
**Last Updated:** 2025-12-17
**Author:** LBM-CUDA Chief Architect
