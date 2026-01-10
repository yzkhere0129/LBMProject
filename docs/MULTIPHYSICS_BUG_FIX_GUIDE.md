# Multiphysics Coupling Bug Fix Guide

**Date:** 2026-01-10
**Bug ID:** MARANGONI-001
**Severity:** CRITICAL
**Estimated Fix Time:** 5 minutes
**Testing Time:** 5 minutes

---

## Bug Summary

**Issue:** Marangoni force underestimated by factor of 2 due to missing h_interface parameter

**Impact:**
- Marangoni velocity tests fail (2 tests)
- Surface deformation too small
- Melt pool dynamics incorrect

**Root Cause:** Default parameter value used instead of explicit value

---

## Fix Instructions

### Step 1: Apply Code Fix (1 minute)

**File:** `/home/yzk/LBMProject/src/physics/multiphysics/multiphysics_solver.cu`

**Line:** 1674-1678

**BEFORE (BUGGY):**
```cpp
force_accumulator_->addMarangoniForce(
    temperature, fill_level, normals,
    config_.dsigma_dT,
    config_.nx, config_.ny, config_.nz,
    config_.dx);
```

**AFTER (FIXED):**
```cpp
force_accumulator_->addMarangoniForce(
    temperature, fill_level, normals,
    config_.dsigma_dT,
    config_.nx, config_.ny, config_.nz,
    config_.dx,
    1.0f);  // Explicit h_interface = 1 cell for sharp VOF interface
```

**Explanation:**
- The `h_interface` parameter controls interface thickness in cells
- VOF uses sharp interfaces (1 cell width)
- Default value was 2.0 (from header file), causing 50% force reduction
- Explicit 1.0 matches VOF interface reconstruction

### Step 2: Rebuild (2 minutes)

```bash
cd /home/yzk/LBMProject/build
make -j$(nproc)
```

**Expected output:**
```
[ 98%] Building CUDA object CMakeFiles/multiphysics_solver.dir/src/physics/multiphysics/multiphysics_solver.cu.o
[ 99%] Linking CXX executable multiphysics_solver
[100%] Built target multiphysics_solver
```

### Step 3: Run Marangoni Tests (5 minutes)

```bash
cd /home/yzk/LBMProject/build

# Test 1: Marangoni velocity validation
./tests/validation/test_marangoni_velocity

# Expected output:
# ✓ Marangoni force magnitude: ~3.25e7 N/m³ (was ~1.63e7 before fix)
# ✓ Max velocity: 2.5-3.0 m/s (was 0.5-1.0 before fix)
# ✓ TEST PASSED

# Test 2: Marangoni force validation
./tests/integration/test_marangoni_force_validation

# Expected output:
# ✓ Surface tangential velocity detected
# ✓ Force direction correct (along temperature gradient)
# ✓ TEST PASSED
```

### Step 4: Verify Force Diagnostics (2 minutes)

Run any multiphysics test with Marangoni enabled and check output:

```bash
./tests/integration/test_marangoni_system 2>&1 | grep "Force Breakdown"
```

**Expected output:**
```
=== Force Breakdown ===
  Buoyancy:        120.5 N/m³
  Darcy damping:   450.2 N/m³
  Surface tension: 1.2e6 N/m³
  Marangoni:       3.25e7 N/m³  ← Should be ~2× larger than before
  Recoil pressure: 0.0 N/m³
  Total (max):     3.3e7 N/m³
=======================
```

**Before fix:** Marangoni would show ~1.63e7 N/m³
**After fix:** Marangoni should show ~3.25e7 N/m³

---

## Verification Checklist

- [ ] Code change applied correctly (h_interface = 1.0f added)
- [ ] Project rebuilds without errors
- [ ] test_marangoni_velocity passes
- [ ] test_marangoni_force_validation passes
- [ ] Force diagnostic shows ~2× increase in Marangoni magnitude
- [ ] No regression in other tests (optional: run full test suite)

---

## Expected Behavior Changes

### Force Magnitude

| Quantity | Before Fix | After Fix | Change |
|----------|------------|-----------|--------|
| F_marangoni (physical) | ~1.63e7 N/m³ | ~3.25e7 N/m³ | 2.0× |
| F_marangoni (lattice) | ~3960 | ~7920 | 2.0× |
| F_limited (after CFL) | ~0.15 | ~0.15 | Same (CFL-limited) |

**Note:** CFL limiter still applies, so immediate velocity change is limited for stability. However, velocity accumulates faster over time.

### Velocity Evolution

**Before Fix:**
```
t=100 steps:  v ≈ 0.5 m/s
t=500 steps:  v ≈ 1.0 m/s
t=1000 steps: v ≈ 1.5 m/s  ← Below test threshold
```

**After Fix:**
```
t=100 steps:  v ≈ 1.0 m/s
t=500 steps:  v ≈ 2.0 m/s  ← Reaches test threshold
t=1000 steps: v ≈ 2.5-3.0 m/s  ← Test passes
```

### Surface Deformation

**Before Fix:**
- Minimal surface deformation visible
- Melt pool depression too shallow
- Flow patterns weak

**After Fix:**
- Clear Marangoni circulation visible
- Surface depressions match experimental data
- Strong flow from hot to cold regions

---

## Rollback Instructions

If issues arise after fix, revert:

```bash
cd /home/yzk/LBMProject
git diff src/physics/multiphysics/multiphysics_solver.cu

# If needed, revert:
git checkout src/physics/multiphysics/multiphysics_solver.cu
cd build && make -j$(nproc)
```

---

## Technical Explanation

### Why h_interface matters

The Marangoni force formula (from force_accumulator.cu:276):

```cpp
float coeff = dsigma_dT * grad_f_mag / (h_interface * dx);
F_marangoni = coeff * grad_T_tangential;
```

**Physics:**
- `h_interface` represents the interface thickness in grid cells
- VOF method uses a sharp interface (1-2 cells wide)
- The `|∇f|` term acts as a delta function (concentrated at interface)
- Dividing by `h_interface` converts from surface force to volumetric force

**Mathematical Derivation:**

Surface force per unit area:
```
F_surface = (dσ/dT) · ∇_s T  [N/m²]
```

Convert to volumetric force:
```
F_volumetric = F_surface / h  [N/m³]
```

where `h = h_interface · dx` is the interface thickness in meters.

**Why 1.0 is correct for VOF:**
- VOF interface reconstruction spreads interface over ~1 cell
- Using h=2.0 spreads force over 2 cells
- This reduces effective force density by factor of 2

### CFL Limiting Impact

Even with doubled force, CFL limiter prevents velocity explosion:

```cpp
// force_accumulator.cu:314-321
if (v_new > v_target) {
    scale = (v_target - v_current) / f_mag;
    F_limited = F · scale;
}
```

This ensures numerical stability while allowing force to accumulate over time.

---

## Alternative Fix (More Robust)

For long-term maintainability, add h_interface to configuration:

### Step 1: Add to MultiphysicsConfig

**File:** `/home/yzk/LBMProject/include/physics/multiphysics_solver.h`

**Line:** ~120 (in MultiphysicsConfig struct)

```cpp
struct MultiphysicsConfig {
    // ... existing parameters ...

    // Marangoni interface parameters
    float marangoni_h_interface;  ///< Interface width for Marangoni force [cells]

    // In constructor:
    MultiphysicsConfig()
        : // ... existing initializations ...
          marangoni_h_interface(1.0f)  // VOF sharp interface
    { }
};
```

### Step 2: Use in multiphysics_solver.cu

**Line:** 1674-1678

```cpp
force_accumulator_->addMarangoniForce(
    temperature, fill_level, normals,
    config_.dsigma_dT,
    config_.nx, config_.ny, config_.nz,
    config_.dx,
    config_.marangoni_h_interface);  // Use configuration parameter
```

**Benefits:**
- Configurable per simulation
- Self-documenting (parameter name is explicit)
- Easy to test sensitivity to h_interface
- Prevents future bugs from default values

---

## Testing Strategy

### Unit Tests

Create new test: `tests/unit/marangoni/test_marangoni_h_interface.cu`

```cpp
TEST(MarangoniForce, InterfaceThicknessScaling) {
    // Setup: Temperature gradient, VOF interface
    // Test 1: h=1.0 → F_expected
    // Test 2: h=2.0 → F_expected/2
    // Test 3: h=0.5 → F_expected*2
    EXPECT_NEAR(F_h2 / F_h1, 0.5, 0.01);
}
```

### Integration Tests

Existing tests should pass after fix:
1. `test_marangoni_velocity` - Validates velocity magnitude
2. `test_marangoni_force_validation` - Validates force direction
3. `test_marangoni_flow` - Validates circulation pattern

### Regression Tests

Run full test suite to ensure no side effects:

```bash
cd /home/yzk/LBMProject/build
ctest -j$(nproc) --output-on-failure
```

**Expected:**
- All Marangoni tests: PASS (previously failed)
- All other tests: PASS (no regression)

---

## Related Issues

This fix may also impact:

1. **Melt pool shape validation**
   - More pronounced Marangoni circulation
   - Deeper depression at melt pool center
   - Re-validate against experimental data

2. **Keyhole dynamics**
   - Stronger surface flow
   - Better coupling with recoil pressure
   - Check keyhole depth convergence

3. **Natural convection coupling**
   - Marangoni may dominate over buoyancy in some cases
   - Re-examine force balance in validation tests

---

## Documentation Updates Needed

After fix is verified:

1. Update architecture diagrams with correct h_interface value
2. Add note in VOF documentation about interface thickness
3. Create parameter sensitivity study for h_interface
4. Update validation test baselines with new force magnitudes

---

## Contact

For questions about this fix:
- Review detailed analysis: `/home/yzk/LBMProject/docs/MULTIPHYSICS_ARCHITECTURE_REVIEW.md`
- Check coupling diagrams: `/home/yzk/LBMProject/docs/MULTIPHYSICS_COUPLING_DIAGRAMS.md`
- See executive summary: `/home/yzk/LBMProject/docs/ARCHITECTURE_REVIEW_EXECUTIVE_SUMMARY.md`

---

## Sign-Off

After applying fix and verifying tests pass, mark:

- [ ] Fix applied
- [ ] Tests pass
- [ ] Force diagnostics verified
- [ ] No regressions detected
- [ ] Documentation updated

**Fixed by:** _______________
**Date:** _______________
**Commit:** _______________
