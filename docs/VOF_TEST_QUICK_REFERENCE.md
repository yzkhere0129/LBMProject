# VOF Test Suite - Quick Reference

**Document:** VOF_TEST_SPECS.md (1637 lines)
**Location:** `/home/yzk/LBMProject/docs/VOF_TEST_SPECS.md`
**Total Tests:** 12 comprehensive validation tests

---

## Test Overview Matrix

| # | Test Name | Type | Runtime | Priority | Status |
|---|-----------|------|---------|----------|--------|
| 1 | Zalesak's Disk | Validation | 2-5 min | Phase 2 | To Implement |
| 2 | Laplace Pressure | Validation | 1-3 min | Phase 2 | To Implement |
| 3 | Oscillating Droplet | Validation | 5-10 min | Phase 3 | To Implement |
| 4 | Thermocapillary Migration | Validation | 3-5 min | Phase 3 | To Implement |
| 5 | Contact Angle | Validation | 2-4 min | Phase 4 | To Implement |
| 6 | Evaporation Mass Loss | Unit | <1 min | Phase 2 | EXISTS |
| 7 | Recoil Pressure | Validation | 2-4 min | Phase 3 | To Implement |
| 8 | Interface Reconstruction | Unit | <1 min | Phase 1 | EXISTS |
| 9 | Curvature Calculation | Unit | <1 min | Phase 1 | EXISTS |
| 10 | Mass Conservation | Unit | 2-5 min | Phase 1 | EXISTS |
| 11 | CFL Stability | Validation | 3-5 min | Phase 4 | EXISTS |
| 12 | Marangoni Force | Unit | <1 min | Phase 2 | EXISTS |

**Legend:**
- EXISTS: Test file already implemented
- To Implement: Test specification complete, needs implementation

---

## Quick Test Summaries

### Test 1: Zalesak's Disk
**What:** Slotted disk rotates 360° and returns to original position
**Validates:** VOF advection accuracy with sharp gradients
**Key Metric:** Shape error < 5%
**Physics:** Interface advection in rotating flow
**Grid:** 100×100×1 cells (2D)
**Time:** 2000 steps

### Test 2: Laplace Pressure
**What:** Spherical droplet generates pressure jump ΔP = 2σ/R
**Validates:** Surface tension (CSF method)
**Key Metric:** Pressure jump error < 15%
**Physics:** Capillary pressure equilibrium
**Grid:** 64×64×64 cells
**Time:** 1000 steps (equilibration)

### Test 3: Oscillating Droplet
**What:** Elliptical droplet oscillates at natural frequency
**Validates:** Dynamic surface tension
**Key Metric:** Frequency error < 15%
**Physics:** f = (1/2π)√[8σ/(ρR³)] ≈ 8.6 kHz
**Grid:** 64×64×64 cells
**Time:** 500 μs (4 periods)

### Test 4: Thermocapillary Migration
**What:** Droplet migrates in temperature gradient
**Validates:** Marangoni effect (Young-Goldstein-Block theory)
**Key Metric:** Migration velocity error < 30%
**Physics:** U = (2/3) × |dσ/dT| × ∇T × R / μ
**Grid:** 100×50×50 cells
**Time:** 10 μs

### Test 5: Contact Angle
**What:** Droplet forms static contact angle on substrate
**Validates:** Wall wetting boundary condition
**Key Metric:** Angle error < 5°
**Physics:** Young's equation, equilibrium shape
**Grid:** 64×64×32 cells
**Time:** 20 μs (relaxation)

### Test 6: Evaporation Mass Loss
**What:** Liquid layer evaporates at constant rate
**Validates:** VOF-thermal coupling for mass loss
**Key Metric:** Mass loss rate error < 10%
**Physics:** df/dt = -J_evap / (ρ × dx)
**Grid:** 32×32×32 cells
**Time:** 10 μs

### Test 7: Recoil Pressure
**What:** High-temperature surface depresses due to vapor recoil
**Validates:** Recoil pressure force (Anisimov model)
**Key Metric:** Pressure magnitude error < 5%
**Physics:** P_recoil = 0.54 × P_sat(T)
**Grid:** 64×64×64 cells
**Time:** 5 μs

### Test 8: Interface Reconstruction
**What:** Compute interface normals for known geometries
**Validates:** PLIC (Piecewise Linear Interface) accuracy
**Key Metric:** Angular error < 5°
**Physics:** n = -∇f / |∇f|
**Grid:** Various (32-64 cells)
**Time:** Single snapshot

### Test 9: Curvature Calculation
**What:** Measure curvature of sphere, cylinder, plane
**Validates:** Geometric curvature computation
**Key Metric:** Relative error < 10%
**Physics:** κ = ∇·n (sphere: κ = 2/R)
**Grid:** 64×64×64 cells
**Time:** Single snapshot

### Test 10: Mass Conservation
**What:** Track total mass over long-term advection
**Validates:** Global conservation property
**Key Metric:** Mass variation < 1%
**Physics:** M(t) = Σf_i = constant
**Grid:** 64×64×64 cells
**Time:** 50 μs (complex flow)

### Test 11: CFL Stability
**What:** Test stability under CFL limit violations
**Validates:** Numerical robustness
**Key Metric:** No bound violations (f ∈ [0,1])
**Physics:** CFL = v × dt / dx < 0.5
**Grid:** 32×32×32 cells
**Time:** 1000 steps

### Test 12: Marangoni Force
**What:** Verify force magnitude for known temperature gradient
**Validates:** Marangoni force calculation (isolated)
**Key Metric:** Force magnitude error < 20%
**Physics:** F = (dσ/dT) × ∇_s T × |∇f|
**Grid:** 32×32×32 cells
**Time:** Single snapshot

---

## Physical Parameters Reference

### Ti6Al4V Material Properties
```
Density (liquid):      ρ = 4110 kg/m³
Surface tension:       σ = 1.5 N/m at 2000 K
Surface tension coeff: dσ/dT = -0.26e-3 N/(m·K)
Viscosity (liquid):    μ = 0.003-0.005 Pa·s
Boiling temperature:   T_boil = 3560 K
Latent heat (vap):     L_vap = 8.878e6 J/kg
Molar mass:            M = 0.0479 kg/mol
```

### Typical Simulation Parameters
```
Grid spacing:          dx = 2 μm (typical)
Time step:             dt = 1e-7 to 1e-9 s (depends on physics)
Domain size:           32-100 cells per dimension
CFL limit:             CFL < 0.5
```

### Characteristic Scales
```
Capillary velocity:    U_cap = σ/μ = 300-500 m/s
Capillary time:        t_cap = ρR²/σ = 112 ns (R=10μm)
Oscillation period:    T_osc = 2π√(ρR³/(8σ)) = 116 μs (R=10μm)
Marangoni velocity:    U_Ma = |dσ/dT| × ∇T × R / μ ≈ 0.1-1 m/s
```

---

## Implementation Roadmap

### Phase 1: Core Numerical Properties (PRIORITY)
**Goal:** Verify fundamental VOF correctness
**Tests:** 8, 9, 10
**Estimated Time:** 2-3 days
**Deliverables:**
- Verified interface normal accuracy (angular error < 5°)
- Verified curvature computation (error < 10%)
- Verified mass conservation (< 1% drift)

### Phase 2: Surface Tension Physics
**Goal:** Validate capillary forces
**Tests:** 2, 1, 12, 6
**Estimated Time:** 4-5 days
**Deliverables:**
- Laplace pressure benchmark (ΔP = 2σ/R)
- Zalesak's disk advection benchmark
- Marangoni force verification
- Evaporation coupling test

### Phase 3: Advanced Interfacial Dynamics
**Goal:** Complex physical phenomena
**Tests:** 4, 7, 3
**Estimated Time:** 5-7 days
**Deliverables:**
- Thermocapillary migration (YGB theory)
- Recoil pressure (Anisimov model)
- Dynamic oscillations

### Phase 4: Boundary Conditions & Robustness
**Goal:** Wall interactions and stability
**Tests:** 5, 11
**Estimated Time:** 3-4 days
**Deliverables:**
- Contact angle validation
- CFL stability analysis

---

## Test Execution Commands

### Run All Unit Tests
```bash
cd /home/yzk/LBMProject/build

# Existing unit tests
./tests/unit/vof/test_vof_reconstruction
./tests/unit/vof/test_vof_curvature
./tests/unit/vof/test_vof_marangoni
./tests/unit/vof/test_vof_evaporation_mass_loss
./tests/unit/vof/test_vof_mass_conservation
./tests/unit/vof/test_vof_advection
./tests/unit/vof/test_vof_contact_angle
```

### Run All Validation Tests
```bash
# Existing validation tests
./tests/validation/vof/test_vof_curvature_sphere
./tests/validation/vof/test_vof_curvature_cylinder
./tests/validation/vof/test_vof_advection_rotation
./tests/validation/test_cfl_stability

# New tests to implement
./tests/validation/vof/test_vof_zalesak_disk
./tests/validation/vof/test_vof_laplace_pressure
./tests/validation/vof/test_vof_oscillating_droplet
./tests/validation/vof/test_vof_thermocapillary_migration
./tests/validation/vof/test_vof_contact_angle_static
./tests/validation/vof/test_vof_recoil_pressure_depression
```

### Quick Smoke Test (< 5 min)
```bash
# Fast tests to verify basic functionality
./tests/unit/vof/test_vof_reconstruction
./tests/unit/vof/test_vof_curvature
./tests/unit/vof/test_vof_marangoni
```

### Full Validation Suite (30-60 min)
```bash
# Run all tests (existing + new)
cd /home/yzk/LBMProject/build
ctest -R vof -V
```

---

## Key Validation Metrics Summary

| Test | Primary Metric | Secondary Metrics | Tolerance |
|------|----------------|-------------------|-----------|
| 1. Zalesak | Shape error | Mass conservation, centroid | 5% / 0.5% / 1 cell |
| 2. Laplace | ΔP error | Sphericity, spurious currents | 15% / 1.05 / 1% |
| 3. Oscillating | Frequency error | Damping rate, volume | 15% / 30% / 1% |
| 4. Thermocapillary | Velocity error | Flow pattern, shape | 30% / qual / 1.1 |
| 5. Contact Angle | Angle error | Symmetry, volume | 5° / 5% / 1% |
| 6. Evaporation | Mass rate error | Interface position | 10% / qual |
| 7. Recoil | Pressure error | Force sign, depression | 5% / qual / qual |
| 8. Reconstruction | Angular error | Normal magnitude | 5° / 0.05 |
| 9. Curvature | Relative error | Consistency (std dev) | 10% / qual |
| 10. Mass Conserv | Mass variation | Monotonicity | 1% / no drift |
| 11. CFL | Bound violations | Warnings issued | 0 / yes |
| 12. Marangoni | Force error | Force direction | 20% / qual |

---

## Common Implementation Patterns

### Test Structure Template
```cpp
#include <gtest/gtest.h>
#include "physics/vof_solver.h"

TEST(VOFValidation, TestName) {
    // 1. Setup domain
    int nx = 64, ny = 64, nz = 64;
    float dx = 2e-6f;
    VOFSolver vof(nx, ny, nz, dx);

    // 2. Initialize conditions
    // (geometry, temperature, velocity)

    // 3. Run simulation
    for (int step = 0; step < num_steps; ++step) {
        // Physics update
        // Diagnostics
    }

    // 4. Extract results
    std::vector<float> results(num_cells);
    vof.copyToHost(results.data());

    // 5. Compute metrics
    float measured = computeMetric(results);
    float theoretical = analyticalSolution();

    // 6. Validate
    float error = abs(measured - theoretical) / theoretical;
    EXPECT_LT(error, tolerance);
}
```

### Analytical Solution Helpers
```cpp
// Sphere curvature
float curvatureSphere(float R) { return 2.0f / R; }

// Laplace pressure
float laplacepressure(float sigma, float R) { return 2.0f * sigma / R; }

// Oscillation frequency
float dropletFrequency(float sigma, float rho, float R) {
    return sqrt(8.0f * sigma / (rho * R * R * R)) / (2.0f * M_PI);
}

// Marangoni velocity
float marangoniVelocity(float dsigma_dT, float grad_T, float R, float mu) {
    return (2.0f / 3.0f) * fabs(dsigma_dT) * grad_T * R / mu;
}
```

---

## Expected Deliverables

### For Each Test
1. **Test implementation file** (.cu)
2. **CMakeLists.txt entry** (test registration)
3. **Analytical reference solution** (documented)
4. **Validation report** (results summary)
5. **Visualization output** (optional VTK files)

### Documentation Requirements
- Test purpose and physics
- Setup parameters (grid, materials, BC)
- Expected behavior (analytical solution)
- Validation metrics (formulas, tolerances)
- Implementation notes (kernels, algorithms)
- Pass/fail criteria (specific thresholds)

---

## References to Full Specifications

**Main Document:** `/home/yzk/LBMProject/docs/VOF_TEST_SPECS.md`

**Test Details:**
- Test 1: Line 28 (Zalesak's Disk)
- Test 2: Line 177 (Laplace Pressure)
- Test 3: Line 326 (Oscillating Droplet)
- Test 4: Line 475 (Thermocapillary Migration)
- Test 5: Line 624 (Contact Angle)
- Test 6: Line 773 (Evaporation Mass Loss)
- Test 7: Line 872 (Recoil Pressure)
- Test 8: Line 1021 (Interface Reconstruction)
- Test 9: Line 1170 (Curvature Calculation)
- Test 10: Line 1269 (Mass Conservation)
- Test 11: Line 1368 (CFL Stability)
- Test 12: Line 1517 (Marangoni Force)

**Summary Table:** Line 1616
**References:** Line 1626

---

**Total Specification Length:** 1637 lines
**Estimated Reading Time:** 45-60 minutes
**Estimated Implementation Time:** 15-20 days (all 12 tests)
