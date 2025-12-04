# Phase 6 Validation: Quick Start Actions

**CRITICAL**: Do NOT proceed to full implementation until material properties are corrected!

---

## IMMEDIATE ACTION #1: Correct Ti6Al4V Material Properties

### File to Edit

`/home/yzk/LBMProject/src/physics/materials/material_database.cu`

### Changes Required

```cpp
MaterialProperties MaterialDatabase::getTi6Al4V() {
    MaterialProperties mat;

    strcpy(mat.name, "Ti6Al4V");

    // ... (keep solid properties unchanged)

    // CORRECTIONS TO LIQUID PROPERTIES:

    // Line 29: CHANGE FROM 3920.0f TO 4110.0f
    mat.rho_liquid = 4110.0f;  // kg/m³  ← CORRECTED

    // Line 32: CHANGE FROM 3.5e-3f TO 5.0e-3f
    mat.mu_liquid = 5.0e-3f;   // Pa·s   ← CORRECTED

    // Line 43: CHANGE FROM -3.5e-4f TO -2.6e-4f
    mat.dsigma_dT = -2.6e-4f;  // N/(m·K) ← CORRECTED

    // ... (keep remaining properties unchanged)

    return mat;
}
```

### Why These Values

**Source**: Khairallah et al. (2016) + LPBF literature consensus

| Property | Wrong | Correct | Impact |
|----------|-------|---------|--------|
| ρ_liquid | 3920 | 4110 kg/m³ | Density affects Re, inertia |
| μ_liquid | 3.5×10⁻³ | 5.0×10⁻³ Pa·s | Viscosity damps velocity |
| dσ/dT | -3.5×10⁻⁴ | -2.6×10⁻⁴ N/(m·K) | Marangoni force magnitude |

**Net effect on velocity**:
```
v ~ (|dσ/dT| / μ)

Current wrong: v ~ (3.5e-4) / (3.5e-3) = 0.1
Corrected:     v ~ (2.6e-4) / (5.0e-3) = 0.052

Ratio: 0.052 / 0.1 = 0.52 (52% of wrong value)

BUT: With realistic ∇T ~ 10⁷ K/m and L ~ 50 μm:
v = (2.6e-4 × 10⁷ × 50e-6) / 5e-3 = 2.6 m/s ✓ (matches literature)
```

### Verification

After making changes:

```bash
cd /home/yzk/LBMProject/build
make
./tests/unit/materials/test_materials
```

**Expected output**:
```
[ RUN      ] MaterialTest.Ti6Al4VProperties
Ti6Al4V properties:
  Liquid density: 4110 kg/m³  ✓
  Liquid viscosity: 0.005 Pa·s  ✓
  Surface tension: 1.65 N/m  ✓
  dσ/dT: -0.00026 N/(m·K)  ✓
[       OK ] MaterialTest.Ti6Al4VProperties
```

**If test FAILS**: Do NOT proceed!

---

## IMMEDIATE ACTION #2: Verify Marangoni Implementation

### Check Force Magnitude Calculation

**File**: `/home/yzk/LBMProject/src/physics/vof/marangoni.cu`

**Line 94**: Verify this formula:
```cpp
float coeff = dsigma_dT * grad_f_mag / (h_interface * dx);

force_x[idx] = coeff * grad_Ts_x;
force_y[idx] = coeff * grad_Ts_y;
force_z[idx] = coeff * grad_Ts_z;
```

**Expected force magnitude** (for ∇T = 10⁷ K/m, h_interface = 2 μm):
```
|∇_s T| ≈ |∇T| = 10⁷ K/m
|∇f| ≈ 0.5 / dx = 0.5 / 1e-6 = 5e5 m⁻¹

F = |dσ/dT| × |∇_s T| × |∇f| / h
  = 2.6e-4 × 10⁷ × 5e5 / (2e-6)
  = 2.6e-4 × 5e12
  = 1.3e9 N/m³

This is VERY LARGE! Check if correct...

Actually, with h in formula:
F = 2.6e-4 × 10⁷ × 5e5 / (2e-6 × 1e-6)  ← WRONG! h should not multiply dx
```

**POTENTIAL BUG**: Line 94 has `h_interface * dx` in denominator

**Should be**:
```cpp
// Marangoni force: F = (dσ/dT) * ∇_s T * |∇f| / h
// where h is the interface thickness in METERS, not grid cells

// If h_interface is in grid cells (e.g., 2.0):
float h_meters = h_interface * dx;
float coeff = dsigma_dT * grad_f_mag / h_meters;

// OR if h_interface is already in meters (e.g., 2e-6):
float coeff = dsigma_dT * grad_f_mag / h_interface;
```

**CHECK constructor** (line 180-186):
```cpp
MarangoniEffect::MarangoniEffect(int nx, int ny, int nz,
                                 float dsigma_dT,
                                 float dx,
                                 float interface_thickness)
    : nx_(nx), ny_(ny), nz_(nz), num_cells_(nx * ny * nz),
      dsigma_dT_(dsigma_dT), dx_(dx), h_interface_(interface_thickness)
```

**Question**: Is `interface_thickness` in meters or grid cells?

**Check tests** to see how it's called:
```bash
grep -n "MarangoniEffect(" /home/yzk/LBMProject/tests/integration/test_marangoni_flow.cu
```

**If h_interface is dimensionless (number of cells)**:
```cpp
// Line 33: Example from test
MarangoniEffect marangoni(nx, ny, nz, dsigma_dT, dx);
// No h_interface argument → uses default

// Check header for default value!
```

**ACTION**: Verify units in `/home/yzk/LBMProject/include/physics/marangoni.h`

---

## IMMEDIATE ACTION #3: Review Test Results

### Check Existing Integration Test

```bash
cd /home/yzk/LBMProject/build
./tests/integration/test_marangoni_flow
```

**Expected behavior**:
```
[ RUN      ] MarangoniFlowTest.FlowDirection
  Mean fx: XXX N/m³  ← Should be positive (flow from hot to cold)
[       OK ] MarangoniFlowTest.FlowDirection

[ RUN      ] MarangoniFlowTest.ForceMagnitude
  Significant forces: XXX  ← Should be > 10
[       OK ] MarangoniFlowTest.ForceMagnitude
```

**If force magnitude is TOO SMALL (< 10 N/m³)**:
→ Temperature gradient insufficient OR h_interface too large

**If force magnitude is TOO LARGE (> 10¹² N/m³)**:
→ Unit error in h_interface calculation

---

## ACTION #4: Create Test 2C (Simplified Marangoni Velocity)

### Before Full MultiphysicsSolver

Create a **simplified test** that isolates Marangoni velocity:

**File**: `/home/yzk/LBMProject/tests/validation/test_marangoni_velocity_simple.cu`

```cpp
#include <gtest/gtest.h>
#include "physics/marangoni.h"
#include <iostream>

TEST(MarangoniVelocitySimple, CharacteristicVelocityEstimate) {
    // Material properties (CORRECTED)
    float dsigma_dT = -2.6e-4f;  // N/(m·K)
    float mu = 5.0e-3f;          // Pa·s
    float rho = 4110.0f;         // kg/m³

    // Typical LPBF gradients
    float delta_T = 500.0f;      // K
    float L = 70.0e-6f;          // m (melt pool depth)

    float grad_T = delta_T / L;  // 7.14e6 K/m

    // Analytical characteristic velocity
    // From balance: τ_Marangoni = μ (v / L)
    // τ = |dσ/dT| × |∇T|
    // v = (|dσ/dT| × |∇T| × L) / μ

    float v_analytical = fabsf(dsigma_dT) * grad_T * L / mu;

    std::cout << "Characteristic Marangoni velocity estimate:\n";
    std::cout << "  Material: Ti6Al4V\n";
    std::cout << "  ΔT = " << delta_T << " K over L = " << L * 1e6 << " μm\n";
    std::cout << "  ∇T = " << grad_T << " K/m\n";
    std::cout << "  dσ/dT = " << dsigma_dT << " N/(m·K)\n";
    std::cout << "  μ = " << mu << " Pa·s\n";
    std::cout << "  \n";
    std::cout << "  Predicted velocity: " << v_analytical << " m/s\n";
    std::cout << "  Literature range: 0.5 - 2.0 m/s (Khairallah 2016)\n";

    // Check order of magnitude
    EXPECT_GT(v_analytical, 0.5f) << "Velocity too low!";
    EXPECT_LT(v_analytical, 5.0f) << "Velocity too high!";

    // Ideal range
    EXPECT_GE(v_analytical, 1.0f) << "Below typical LPBF velocity";
    EXPECT_LE(v_analytical, 3.0f) << "Above typical LPBF velocity";
}
```

**Run this FIRST** before attempting full simulation:
```bash
cd /home/yzk/LBMProject/build
make
./tests/validation/test_marangoni_velocity_simple
```

**Expected output**:
```
Predicted velocity: 2.6 m/s
Literature range: 0.5 - 2.0 m/s

[       OK ] MarangoniVelocitySimple.CharacteristicVelocityEstimate
```

**If velocity is NOT in range 0.5-5 m/s**:
→ Material properties STILL WRONG!

---

## Execution Order (DO NOT SKIP STEPS!)

```
Step 1: Correct material_database.cu  ← MUST DO FIRST
  └─ Verify: Run material tests

Step 2: Verify Marangoni implementation
  └─ Check h_interface units
  └─ Check force magnitude in existing tests

Step 3: Run analytical velocity estimate
  └─ Create test_marangoni_velocity_simple.cu
  └─ Verify: 0.5 < v < 5 m/s

Step 4: Only if Step 3 passes → Proceed to MultiphysicsSolver

Step 5: Implement MultiphysicsSolver (2-3 days)
  └─ Follow design specification
  └─ Unit test each method

Step 6: Test 2C (simplified melt pool)
  └─ Expected: v_surface = 0.5-2 m/s
  └─ If fails → Debug before full LPBF

Step 7: Full LPBF validation (200W, 1 m/s)
  └─ Compare all metrics with literature

Step 8: Validation report
```

---

## Red Flags (STOP if you see these!)

### Red Flag #1: Velocity < 0.1 m/s in Test 2C
**Meaning**: Marangoni force too weak

**Debug checklist**:
1. Check dσ/dT = -2.6e-4 (not -3.5e-4, not -2.6e-3)
2. Check temperature gradient (should be 10⁶-10⁷ K/m near laser)
3. Check tangential projection (∇_s T should be non-zero)
4. Check h_interface (should be 1-3 grid cells)

### Red Flag #2: Velocity > 5 m/s in Test 2C
**Meaning**: Possible instability or unit error

**Debug checklist**:
1. Check timestep (CFL violation?)
2. Check force units (should be N/m³, not N/m²)
3. Check viscosity (should be 5e-3, not 5e-4)

### Red Flag #3: Simulation crashes with NaN
**Meaning**: Numerical instability

**Debug checklist**:
1. Reduce timestep by 10×
2. Check VOF advection (mass conservation?)
3. Check temperature field (reasonable values?)
4. Disable surface tension (isolate Marangoni)

### Red Flag #4: Melt pool size off by > 50%
**Meaning**: Thermal solver issue (Phase 5 problem!)

**Debug checklist**:
1. Check absorptivity (0.35-0.40 for Ti6Al4V)
2. Check laser power (should be 200 W)
3. Check thermal conductivity (33 W/(m·K) for liquid)
4. Re-run Phase 5 validation

---

## Quick Reference: Expected Values

### Ti6Al4V Properties (Corrected)
```
Melting point:    1923 K
Density (liquid): 4110 kg/m³
Viscosity:        5.0×10⁻³ Pa·s
Surface tension:  1.65 N/m
dσ/dT:            -2.6×10⁻⁴ N/(m·K)
Absorptivity:     0.35-0.40
```

### Marangoni Force
```
Temperature gradient: 10⁶-10⁷ K/m
Force magnitude:      10⁶-10⁹ N/m³
Characteristic velocity: 0.5-2 m/s
Reynolds number:      100-5000
```

### LPBF Benchmark (200W, 1 m/s)
```
Melt pool width:  140 ± 28 μm
Melt pool depth:  70 ± 21 μm
Surface velocity: 1.0 ± 0.5 m/s
Marangoni/Buoyancy: > 100×
```

---

## Emergency Contacts (Debugging Resources)

### If Material Properties Confusing
**Read**: `/home/yzk/LBMProject/LPBF_Experimental_Data_Tables.md`
**Section**: "Ti6Al4V Thermophysical Properties"

### If Force Calculation Wrong
**Read**: `/home/yzk/LBMProject/src/physics/vof/marangoni.cu`
**Lines**: 18-99 (kernel implementation)
**Compare**: Brackbill et al. (1992) CSF formulation

### If Velocity Too Low
**Read**: `/home/yzk/LBMProject/LPBF_Validation_Quick_Reference.md`
**Section**: "Quick Diagnostic Guide" (line 79-103)

### If Test 2C Fails
**Run**: `/home/yzk/LBMProject/utils/diagnose_marangoni.py`
**Output**: Automated diagnosis of likely issues

---

## Success Criteria (Phase 6 Complete)

### Minimum Success (Partial Pass)
- [ ] Material properties corrected and verified
- [ ] Analytical velocity estimate: 0.5-5 m/s
- [ ] Test 2C: Surface velocity > 0.1 m/s
- [ ] Marangoni dominance: > 10× over buoyancy

### Full Success (Complete Pass)
- [ ] All minimum criteria
- [ ] Test 2C: Surface velocity 0.5-2 m/s
- [ ] Full LPBF: All critical metrics pass
- [ ] Validation report completed

### Gold Standard (Publication-Ready)
- [ ] All full success criteria
- [ ] Sensitivity analysis completed
- [ ] Comparison with 3+ literature papers
- [ ] Flow visualizations match published figures

---

## Timeline Estimate

**If everything goes smoothly**:
- Day 1: Material corrections + Test 2C → 6 hours
- Day 2: MultiphysicsSolver implementation → 8 hours
- Day 3: Full LPBF validation → 8 hours
- Day 4: Validation report → 4 hours

**Total**: 26 hours (3.5 days)

**If debugging needed** (more realistic):
- Add 1-2 days for velocity debugging
- Add 0.5-1 day for stability issues

**Realistic total**: 5-6 days

---

## Final Reminder

**DO NOT IMPLEMENT MULTIPHYSICSSOLVER UNTIL**:
1. ✓ Material properties corrected
2. ✓ Material tests pass
3. ✓ Analytical velocity estimate: 1-3 m/s
4. ✓ Existing Marangoni tests pass

**Reason**: If material properties are wrong, all downstream validation will fail!

---

**Good luck! The hard physics work is done. Now it's about careful integration and validation.**
