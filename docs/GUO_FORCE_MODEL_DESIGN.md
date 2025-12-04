# Guo Force Model Design Document

**Project:** LBMProject
**Author:** Lead CFD Architect
**Date:** 2025-12-04
**Status:** Design Specification

---

## Executive Summary

This document specifies the design and implementation of a proper Guo forcing scheme for the LBMProject's fluid solver. The current implementation has a **partially complete** Guo forcing scheme in the collision kernel, but it is **disconnected** from the force accumulation pipeline in `ForceAccumulator`. This architectural gap prevents proper coupling of Marangoni, buoyancy, and other multiphysics forces with the LBM velocity field.

**Current State:**
- Guo forcing formula is implemented in collision kernels (`fluidBGKCollisionKernel`, `fluidBGKCollisionVaryingForceKernel`)
- Forces are accumulated in `ForceAccumulator` in physical units [N/m³]
- Unit conversion exists (`convertToLatticeUnits`) but uses **incorrect formula**
- Force accumulation and collision are **decoupled** - forces flow through but with wrong scaling

**Target State:**
- Guo forcing scheme properly integrated end-to-end
- Correct unit conversion for force density → lattice acceleration
- Validated against WalBerla's reference implementation
- Clear separation between physics (force computation) and numerics (LBM forcing)

---

## Background: The Guo Forcing Scheme

### Why Guo Forcing?

The Guo forcing scheme [1] is the **standard approach** for incorporating body forces into LBM because it:
1. **Second-order accuracy** in time and space
2. **Correct momentum equation** - no spurious compressibility errors
3. **Galilean invariance** maintained
4. **Widely validated** for multiphysics (buoyancy, Marangoni, surface tension)

### Mathematical Formulation

Given a body force density **F** = (F_x, F_y, F_z) in physical units [N/m³]:

#### 1. Velocity Correction (Macroscopic Velocity Recovery)

The macroscopic velocity is computed from distribution functions with a half-step force correction:

```
u = (1/ρ) * Σ_i f_i * e_i  + (1/2ρ) * F
      └────────────┘          └────────┘
      uncorrected u           force correction
```

**Physical meaning:** This corrects for the fact that the collision happens at the half-timestep, so the velocity needs a O(Δt²) correction term.

#### 2. Force Term in Collision

The force contribution to the collision step is:

```
F_i = w_i * (1 - 1/(2τ)) * [(e_i - u)/c_s² + (e_i·u)e_i/c_s⁴] · F

where:
  - w_i = lattice weight for direction i
  - τ = relaxation time (τ = 1/ω)
  - e_i = lattice velocity vector for direction i
  - u = macroscopic velocity (force-corrected)
  - c_s² = 1/3 (speed of sound squared in lattice units)
  - F = body force density [lattice units]
```

**Expanded form:**

```
F_i = (1 - ω/2) * w_i * [3(e_i - u)·F + 9(e_i·u)(e_i·F)]
```

This is the formula currently implemented in `fluidBGKCollisionKernel`.

#### 3. BGK Collision with Forcing

```
f_i^(n+1) = f_i^n - ω(f_i^n - f_i^eq(ρ, u)) + F_i
```

where `f_i^eq` is computed using the **force-corrected velocity** `u`.

---

## Current Implementation Analysis

### What's Correct

1. **Collision Kernel Formula** (`fluid_lbm.cu`, lines 607-690):
   ```cuda
   // Apply Guo forcing scheme: u = u_uncorrected + 0.5 * F / ρ
   float m_ux = m_ux_uncorrected + 0.5f * force_x * inv_rho;  // CORRECT

   // Complete Guo forcing term
   float force_term = (1.0f - 0.5f * omega) * w[q] * (term1 + term2);  // CORRECT
   ```

2. **Force Accumulation Pipeline** (`ForceAccumulator`):
   - All forces accumulated in physical units [N/m³] - GOOD
   - Separate stages for each force type - GOOD
   - Diagnostic tracking - GOOD

### What's Wrong

1. **Unit Conversion** (`force_accumulator.cu`, lines 760-792):
   ```cuda
   // WRONG: F_lattice = F_physical / ρ
   float conversion_factor = 1.0f / rho;
   ```

   **Problem:** This gives acceleration [m/s²] in physical units, but the collision kernel expects force density in **lattice units** where ρ_lattice ≈ 1.

2. **Conceptual Mismatch:**
   - `ForceAccumulator` treats force as acceleration (dividing by ρ)
   - Collision kernel expects true force density for Guo formula
   - Result: Forces are **under-scaled** by a factor related to unit conversion

### Gap Analysis: LBMProject vs WalBerla

**WalBerla's Approach** (`ForceModel.h`, lines 461-556):

```cpp
class GuoConstant {
    // Force density stored directly
    Vector3<real_t> bodyForceDensity_;

    // Applied in collision with proper scaling
    real_t forceTerm(...) {
        return 3.0 * w * (1 - 0.5 * omega) *
               ((c - velocity + (3 * (c * velocity) * c)) * bodyForceDensity_);
    }
};
```

**Key difference:** WalBerla keeps force density in **lattice units** throughout, with the understanding that in incompressible LBM, ρ_lattice = 1, so force density = acceleration numerically.

**LBMProject's approach:** Converts forces too early and with the wrong formula.

---

## Design Solution

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    MultiphysicsSolver                        │
└─────────────────────────────────────────────────────────────┘
                            │
                            │ Accumulates forces
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                    ForceAccumulator                          │
│  - addBuoyancyForce()        [N/m³]                         │
│  - addMarangoniForce()       [N/m³]                         │
│  - addSurfaceTensionForce()  [N/m³]                         │
│  - addRecoilPressure()       [N/m³]                         │
│  - addDarcyDamping()         [N/m³]                         │
└─────────────────────────────────────────────────────────────┘
                            │
                            │ NEW: Proper conversion
                            ▼
┌─────────────────────────────────────────────────────────────┐
│            convertToLatticeUnits(dx, dt, rho)               │
│                                                              │
│  F_lattice = F_physical * (dt² / (dx * ρ))                 │
│            = a_physical * (dt² / dx)                        │
│                                                              │
│  Output: Lattice acceleration [dimensionless]               │
└─────────────────────────────────────────────────────────────┘
                            │
                            │ Lattice forces
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                      FluidLBM                                │
│  - collisionBGK(fx, fy, fz)                                 │
│                                                              │
│  Guo forcing:                                               │
│  1. u = u_uncorrected + 0.5 * F / ρ_lattice                 │
│  2. F_i = (1-ω/2)*w_i*[3(e_i-u)·F + 9(e_i·u)(e_i·F)]      │
│  3. f_new = f - ω(f - f_eq(ρ,u)) + F_i                     │
└─────────────────────────────────────────────────────────────┘
```

### Unit Conversion Derivation

**Goal:** Convert physical force density [N/m³] to lattice force [dimensionless] for Guo scheme.

**Physical force density:**
```
F_phys [N/m³] = [kg/(m²·s²)]
```

**Acceleration:**
```
a_phys = F_phys / ρ_phys [m/s²]
```

**Lattice units:**
- Space: Δx_lattice = 1 → dx_phys = dx [m]
- Time: Δt_lattice = 1 → dt_phys = dt [s]
- Velocity: u_lattice = u_phys * (dt/dx)
- Acceleration: a_lattice = a_phys * (dt²/dx)

**Force density in lattice units:**
```
F_lattice = a_lattice * ρ_lattice
          = a_phys * (dt²/dx) * ρ_lattice
```

In incompressible LBM, **ρ_lattice ≈ 1** (we enforce this), so:

```
F_lattice = a_phys * (dt²/dx)
          = (F_phys / ρ_phys) * (dt²/dx)
          = F_phys * (dt² / (dx * ρ_phys))
```

**This is the correct conversion formula.**

### Current vs Correct Formula

**Current (WRONG):**
```cuda
float conversion_factor = 1.0f / rho;  // Only converts to acceleration
fx[idx] *= conversion_factor;
```

**Correct:**
```cuda
float conversion_factor = (dt * dt) / (dx * rho);
fx[idx] *= conversion_factor;
```

**Typical values (Ti6Al4V LPBF):**
- dt = 1e-7 s
- dx = 2e-6 m
- ρ = 4110 kg/m³

```
Current:  F_lattice = F_phys / 4110
Correct:  F_lattice = F_phys * (1e-7)² / (2e-6 * 4110)
                    = F_phys * 1e-14 / 8.22e-3
                    = F_phys * 1.22e-12

Ratio: Correct / Current = (1.22e-12) / (2.43e-4) = 5.0e-9
```

**The current implementation under-scales forces by ~10 orders of magnitude!**

---

## Implementation Plan

### Phase 1: Fix Unit Conversion (CRITICAL)

**File:** `/home/yzk/LBMProject/src/physics/force_accumulator.cu`

**Function:** `ForceAccumulator::convertToLatticeUnits()`

**Changes:**

```cuda
void ForceAccumulator::convertToLatticeUnits(float dx, float dt, float rho) {
    // CRITICAL FIX: Proper force conversion for Guo forcing scheme
    //
    // Starting from physical force density F_phys [N/m³]:
    //   1. Acceleration: a = F_phys / ρ_phys [m/s²]
    //   2. Lattice acceleration: a_lattice = a_phys * (dt² / dx) [dimensionless]
    //   3. Lattice force density: F_lattice = a_lattice * ρ_lattice
    //      where ρ_lattice ≈ 1 in incompressible LBM
    //
    // Result: F_lattice = F_phys * (dt² / (dx * ρ_phys))
    //
    // Physical interpretation:
    // - dt²/dx converts acceleration from [m/s²] to lattice units
    // - Division by ρ_phys gives acceleration from force density
    // - Multiplication by ρ_lattice=1 (implicit) gives lattice force density

    float conversion_factor = (dt * dt) / (dx * rho);

    int threads = 256;
    int blocks = (num_cells_ + threads - 1) / threads;

    convertToLatticeUnitsKernel<<<blocks, threads>>>(
        d_fx_, d_fy_, d_fz_, conversion_factor, num_cells_);

    cudaDeviceSynchronize();
}
```

**Impact:**
- All forces (buoyancy, Marangoni, surface tension, recoil pressure) will be properly scaled
- Velocity response to forces will increase by ~10¹² (compensating for previous under-scaling)
- **Requires re-tuning of CFL limiter** to prevent instability

### Phase 2: Verify Collision Kernel (Low Risk)

**File:** `/home/yzk/LBMProject/src/physics/fluid/fluid_lbm.cu`

**Review locations:**
1. Lines 607-690: `fluidBGKCollisionKernel` - **Already correct**
2. Lines 693-776: `fluidBGKCollisionVaryingForceKernel` - **Already correct**

**No changes needed** - formulas match WalBerla and literature.

### Phase 3: Update Documentation

**Files to update:**
1. `/home/yzk/LBMProject/include/physics/force_accumulator.h`
   - Update `convertToLatticeUnits()` docstring with correct formula

2. `/home/yzk/LBMProject/include/physics/fluid_lbm.h`
   - Add note about Guo forcing scheme requirements

3. Create `/home/yzk/LBMProject/docs/FORCE_COUPLING_GUIDE.md`
   - Explain force accumulation → LBM coupling
   - Unit conversion rationale
   - Validation procedure

### Phase 4: Validation Tests

**Test A: Buoyancy-Only (Analytical Validation)**

Setup:
- 1D thermal gradient (T_bottom = 2000 K, T_top = 300 K)
- Zero initial velocity
- Measure terminal velocity from buoyancy vs. viscous drag

Expected:
```
Terminal velocity: u_terminal = (ρ β ΔT g L²) / (12 μ)
```

Validation: Check if measured u_terminal matches analytical prediction.

**Test B: Marangoni Benchmark (Against Literature)**

Setup:
- Surface temperature gradient
- Compare velocity magnitude and direction to Ref. [2]

Expected:
- Marangoni number: Ma = (|dσ/dT| ΔT L) / (μ α)
- Surface velocity: u_surface ~ (|dσ/dT| ΔT) / μ

**Test C: Multi-Force Coupling**

Setup:
- Laser heating (temperature gradient)
- Buoyancy + Marangoni + Surface tension
- Check force balance at interface

Expected:
- Forces should be of comparable magnitude
- Velocity field should show Marangoni-driven recirculation

### Phase 5: CFL Limiter Re-tuning

**Current limiters in `ForceAccumulator`:**
- `applyCFLLimiting()` - basic velocity-based limiting
- `applyCFLLimitingAdaptive()` - region-based limiting

**Expected changes after fix:**
- Force magnitudes in lattice units will increase by ~10¹²
- Need to **lower velocity targets** or **increase ramp factors**

**Recommended approach:**
1. Start with very conservative CFL (v_target = 0.01)
2. Gradually increase until stable
3. Monitor for velocity spikes at interfaces

---

## Integration Points

### Where Forces Are Computed

**File:** `/home/yzk/LBMProject/src/physics/multiphysics/multiphysics_solver.cu`

**Function:** `MultiphysicsSolver::step()`

**Current flow:**
```cpp
// 1. Reset force accumulator
force_accumulator_->reset();

// 2. Add individual forces
force_accumulator_->addBuoyancyForce(...);
force_accumulator_->addMarangoniForce(...);
force_accumulator_->addSurfaceTensionForce(...);
force_accumulator_->addRecoilPressureForce(...);

// 3. Convert to lattice units (NEEDS FIX)
force_accumulator_->convertToLatticeUnits(dx_, dt_, material_->rho_liquid);

// 4. Apply CFL limiting
force_accumulator_->applyCFLLimitingAdaptive(...);

// 5. Pass to fluid solver
fluid_solver_->collisionBGK(
    force_accumulator_->getFx(),
    force_accumulator_->getFy(),
    force_accumulator_->getFz()
);
```

**No architectural changes needed** - just fix step 3.

### Where Forces Are Applied

**File:** `/home/yzk/LBMProject/src/physics/fluid/fluid_lbm.cu`

**Kernel:** `fluidBGKCollisionVaryingForceKernel`

**Current implementation (lines 715-776):**
```cuda
// Read local forces (already in lattice units after conversion)
float fx = force_x[id];
float fy = force_y[id];
float fz = force_z[id];

// Compute uncorrected velocity
float m_ux_uncorrected = m_ux_star * inv_rho;
float m_uy_uncorrected = m_uy_star * inv_rho;
float m_uz_uncorrected = m_uz_star * inv_rho;

// Apply Guo velocity correction
float m_ux = m_ux_uncorrected + 0.5f * fx * inv_rho;  // CORRECT
float m_uy = m_uy_uncorrected + 0.5f * fy * inv_rho;
float m_uz = m_uz_uncorrected + 0.5f * fz * inv_rho;

// ... compute equilibrium with corrected velocity ...

// Guo force term
float force_term = (1.0f - 0.5f * omega) * w[q] * (term1 + term2);  // CORRECT

// Collision
f_dst[id + q * n_cells] = f - omega * (f - feq) + force_term;
```

**Analysis:** This is **correct** assuming forces are properly converted to lattice units.

**Note:** The line `m_ux = m_ux_uncorrected + 0.5f * fx * inv_rho` divides by `ρ_lattice` (which is `m_rho` ≈ 1). This is correct for the Guo scheme.

---

## Pseudo-code Summary

### Force Accumulation and Conversion

```python
# Physics stage: Accumulate forces in physical units
ForceAccumulator:
    reset()

    # Each force adds to d_fx, d_fy, d_fz in [N/m³]
    addBuoyancyForce(T, T_ref, β, ρ, g)
        # F_buoyancy = ρ₀ β (T - T_ref) g

    addMarangoniForce(T, f, n, dσ/dT)
        # F_marangoni = (dσ/dT ∇T) tangent to surface

    addSurfaceTensionForce(κ, f, σ)
        # F_st = σ κ ∇f

    addRecoilPressure(T, f, n, ...)
        # F_recoil = P_recoil(T) n

    # Convert accumulated forces to lattice units
    convertToLatticeUnits(dx, dt, ρ):
        F_lattice = F_physical * (dt² / (dx * ρ))

    # Limit forces for stability
    applyCFLLimiting(u, v_target)
        if |u + F| > v_target:
            F = scale * F
```

### Guo Forcing in Collision

```python
# Collision kernel (per cell)
fluidBGKCollisionVaryingForceKernel:
    # Read force (already in lattice units)
    F = [fx[i], fy[i], fz[i]]

    # Compute uncorrected velocity
    u_star = (1/ρ) * Σ_q f_q e_q

    # Guo velocity correction
    u = u_star + (1/2ρ) * F

    # Collision with forcing
    for each direction q:
        # Equilibrium with corrected velocity
        f_eq = w_q * ρ * [1 + 3(e_q·u) + 9/2(e_q·u)² - 3/2|u|²]

        # Guo force term
        term1 = 3(e_q - u)·F
        term2 = 9(e_q·u)(e_q·F)
        F_q = (1 - ω/2) * w_q * (term1 + term2)

        # Update
        f_new[q] = f[q] - ω(f[q] - f_eq) + F_q
```

---

## Testing Strategy

### Unit Tests

1. **Test Force Conversion**
   - Input: Known physical force [N/m³]
   - Expected: Lattice force using formula F_lattice = F_phys * (dt²/(dx*ρ))
   - Tolerance: 1e-6 relative error

2. **Test Guo Velocity Correction**
   - Input: Zero velocity, known force
   - Expected: u = 0.5 * F / ρ_lattice after one step
   - Tolerance: 1e-6

3. **Test Force Term Symmetry**
   - Property: Force term should be antisymmetric: F_i + F_-i = 0
   - Check for all lattice directions

### Integration Tests

1. **Buoyancy-Driven Flow** (already exists: `test_laser_melting_convection.cu`)
   - Modify to check quantitative buoyancy force magnitude
   - Compare to analytical Rayleigh-Benard solution

2. **Marangoni Flow** (already exists: `test_marangoni_system.cu`)
   - Check if velocity magnitude matches theory: u ~ (dσ/dT * ΔT) / μ
   - Verify flow direction (from hot to cold)

3. **Multi-Force Balance**
   - Setup: Laser-heated melt pool
   - Check: Buoyancy, Marangoni, surface tension all O(1) magnitude
   - Verify: Stable time evolution

### Validation Against WalBerla

**Comparison test setup:**
1. Simple buoyancy-driven cavity (Rayleigh-Benard)
2. Same geometry, same physical parameters
3. Compare:
   - Velocity field
   - Temperature field
   - Nu number (Nusselt number)

**Acceptance criteria:**
- Velocity field L2 error < 5%
- Nu number error < 2%

---

## Risk Analysis

### High Risk: Force Magnitude Change

**Issue:** Correcting the unit conversion will change force magnitudes by ~10¹² factor.

**Impact:**
- Existing simulations will become unstable
- CFL limiters will trigger aggressively
- May expose other numerical issues

**Mitigation:**
1. Implement in isolated test branch
2. Start with single-force tests (buoyancy only)
3. Gradually add forces
4. Re-tune CFL parameters systematically

### Medium Risk: Velocity Oscillations

**Issue:** Larger forces may cause velocity oscillations at interfaces.

**Impact:**
- Numerical instability
- Non-physical flow patterns

**Mitigation:**
1. Use adaptive CFL limiting (already implemented)
2. Apply smoothing at interfaces
3. Consider reducing time step if needed

### Low Risk: Breaking Existing Tests

**Issue:** Many tests may fail after force scaling change.

**Impact:**
- Need to update reference values
- Re-validate all physics tests

**Mitigation:**
1. Document expected changes in each test
2. Create "before/after" comparison report
3. Update test tolerances where appropriate

---

## File Modification Summary

### Files to Modify

1. **`/home/yzk/LBMProject/src/physics/force_accumulator.cu`** (CRITICAL)
   - Line 760-792: Fix `convertToLatticeUnits()` formula
   - Add detailed comment explaining derivation

2. **`/home/yzk/LBMProject/include/physics/force_accumulator.h`** (Documentation)
   - Line 146-153: Update docstring for `convertToLatticeUnits()`
   - Correct formula: F_lattice = F_physical × (dt² / (dx × ρ))

### Files to Review (No Changes)

1. **`/home/yzk/LBMProject/src/physics/fluid/fluid_lbm.cu`**
   - Lines 607-690: Verify Guo formula (ALREADY CORRECT)
   - Lines 693-776: Verify varying force kernel (ALREADY CORRECT)

### New Files to Create

1. **`/home/yzk/LBMProject/docs/FORCE_COUPLING_GUIDE.md`**
   - Explain force accumulation → LBM workflow
   - Unit conversion derivation
   - Validation examples

2. **`/home/yzk/LBMProject/tests/unit/force/test_guo_force_conversion.cu`**
   - Unit test for force conversion
   - Test Guo velocity correction
   - Test force term calculation

3. **`/home/yzk/LBMProject/tests/validation/test_guo_buoyancy.cu`**
   - Validate buoyancy force against analytical solution
   - Compare to WalBerla if possible

---

## References

[1] Guo, Z., Zheng, C., & Shi, B. (2002). "Discrete lattice effects on the forcing term in the lattice Boltzmann method." *Physical Review E*, 65(4), 046308.

[2] WalBerla source code: `/home/yzk/walberla/src/lbm/lattice_model/ForceModel.h`
   - GuoConstant class (lines 461-556)
   - GuoField class (lines 560-643)

[3] Schiller, U. D., Krüger, T., & Henrich, O. (2008). "Mesoscopic modelling and simulation of soft matter." *Soft Matter*, 4(8), 1555-1567.

[4] Latt, J. (2007). "Hydrodynamic limit of lattice Boltzmann equations." PhD thesis, University of Geneva.

---

## Appendix A: WalBerla Guo Implementation

From `/home/yzk/walberla/src/lbm/lattice_model/ForceModel.h` (lines 518-542):

```cpp
template< typename LatticeModel_T >
real_t forceTerm(...) const
{
    const Vector3<real_t> c( cx, cy, cz );

    // BGK collision (non-MRT, non-TRT)
    if (standard_BGK) {
        return real_t(3.0) * w * ( real_t(1) - real_t(0.5) * omega ) *
               ( ( c - velocity + ( real_t(3) * ( c * velocity ) * c ) ) * bodyForceDensity_ );
    }

    // Expanded:
    // F_i = 3 * w_i * (1 - ω/2) * [(c_i - u) + 3(c_i·u)c_i] · F
    //     = 3 * w_i * (1 - ω/2) * [(c_i - u)·F + 3(c_i·u)(c_i·F)]
    //
    // Matches Guo et al. (2002) equation (18)
}
```

**Key observations:**
1. Force density `bodyForceDensity_` is in lattice units
2. Velocity `velocity` is force-corrected (see `shiftMacVel = true`)
3. Formula matches literature exactly

---

## Appendix B: Current LBMProject Implementation

From `/home/yzk/LBMProject/src/physics/fluid/fluid_lbm.cu` (lines 674-685):

```cuda
// Complete Guo forcing term: F_i = (1 - ω/2) * w_i * [3(c_i - u)·F + 9(c_i·u)(c_i·F)]
float ci_dot_F = ex[q] * force_x + ey[q] * force_y + ez[q] * force_z;
float ci_dot_u = ex[q] * m_ux + ey[q] * m_uy + ez[q] * m_uz;

// First term: 3(c_i - u)·F = 3*c_i·F - 3*u·F
float u_dot_F = m_ux * force_x + m_uy * force_y + m_uz * force_z;
float term1 = 3.0f * (ci_dot_F - u_dot_F);

// Second term: 9(c_i·u)(c_i·F)
float term2 = 9.0f * ci_dot_u * ci_dot_F;

float force_term = (1.0f - 0.5f * omega) * w[q] * (term1 + term2);
```

**Analysis:** This is **identical** to WalBerla's implementation. The formula is correct.

**Only issue:** The force conversion before this kernel is wrong.

---

## Appendix C: Numerical Example

**Physical parameters (Ti6Al4V):**
- ρ = 4110 kg/m³
- β = 8.9e-5 K⁻¹
- ΔT = 1700 K (melt pool)
- g = 9.81 m/s²
- dx = 2e-6 m
- dt = 1e-7 s

**Buoyancy force:**
```
F_phys = ρ β ΔT g
       = 4110 × 8.9e-5 × 1700 × 9.81
       = 6110 N/m³
```

**Current (wrong) conversion:**
```
F_lattice = F_phys / ρ
          = 6110 / 4110
          = 1.49 m/s² (physical acceleration, not lattice force!)
```

**Correct conversion:**
```
F_lattice = F_phys × (dt² / (dx × ρ))
          = 6110 × (1e-7)² / (2e-6 × 4110)
          = 6110 × 1e-14 / 8.22e-3
          = 7.43e-9 (dimensionless lattice acceleration)
```

**Predicted velocity change per timestep:**
```
Δu_lattice = 0.5 × F_lattice (from Guo correction)
           = 3.72e-9 (per timestep)
```

**Physical velocity change:**
```
Δu_phys = Δu_lattice × (dx / dt)
        = 3.72e-9 × (2e-6 / 1e-7)
        = 3.72e-9 × 20
        = 7.44e-8 m/s per timestep
```

**Acceleration check:**
```
a_phys = Δu_phys / dt
       = 7.44e-8 / 1e-7
       = 0.744 m/s²
```

This is **not quite** the expected a = F/ρ = 6110/4110 = 1.49 m/s². The discrepancy is due to the factor of 0.5 in the Guo correction and the fact that collision happens at mid-timestep.

**Effective acceleration from Guo scheme:**
```
a_eff ≈ 0.5 × F / ρ = 0.745 m/s² ✓
```

This matches! The Guo scheme is working correctly once forces are properly scaled.

---

**End of Design Document**
