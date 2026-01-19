# VOF Mass Conservation Test Suite

## Overview

Comprehensive test suite for validating VOF advection mass conservation, designed to measure baseline performance and compare first-order upwind vs TVD schemes.

**File**: `/home/yzk/LBMProject/tests/validation/vof/test_vof_mass_conservation_tvd.cu`

## Purpose

This test suite was created to:
1. Establish baseline mass conservation metrics for first-order upwind VOF
2. Provide comparison framework for TVD (Total Variation Diminishing) schemes
3. Isolate VOF advection from complex multiphysics interactions
4. Validate improvements after implementing higher-order advection

## Background

**Problem**: Current first-order upwind VOF shows ~20% mass loss in Rayleigh-Taylor instability tests

**Root Cause**: First-order upwind advection is inherently diffusive, causing interface smearing and mass loss over long simulations

**Solution Path**: Implement TVD schemes (van Leer, Superbee, MUSCL) to reduce numerical diffusion

**This Test Suite**: Provides quantitative metrics to validate the improvement

## Test Cases

### Test 1: Translation - Uniform Velocity

**Physics**:
- Circular interface translates horizontally with constant velocity
- Periodic boundaries (material wraps around)
- Pure advection, no deformation
- After 64 steps, circle completes one full crossing

**Parameters**:
- Domain: 64×64×4 (2D thin in z)
- Circle radius: 12 cells
- Velocity: u_x = 0.25 (CFL = 0.25)
- Duration: 64 timesteps (one period)

**Baseline Results (First-Order Upwind)**:
```
Initial mass: M0 = 1850.9
Final mass: M = 1850.9
Mass error: 0.000020% (EXCELLENT)
L2 error: 0.348
Interface diffusion: +33 cells (significant)
```

**Analysis**:
- Mass conservation is EXCELLENT due to CFL subcycling (CFL_target=0.10)
- Interface significantly diffuses (33 cells = 2.75× original width)
- L2 error shows shape distortion from diffusion
- With subcycling disabled: mass error would be ~5-10%

**Success Criteria**:
- First-order baseline: <5% mass error
- TVD target: <0.5% mass error
- Interface sharpness: TVD should show <10 cells diffusion

---

### Test 2: Rotation - Solid Body 360°

**Physics**:
- Circular interface rotates 360° around domain center
- Solid body rotation (no shearing)
- Tests geometric fidelity under rotation
- Should return to original configuration

**Parameters**:
- Domain: 64×64×4
- Circle radius: 10 cells
- Angular velocity: ω = 0.04 rad/step
- Duration: 158 steps (2π/ω)

**Baseline Results (First-Order Upwind)**:
```
Initial mass: M0 = 1297.97
Final mass: M = 1297.97
Mass error: 0.00047% (EXCELLENT)
L2 error: 0.095
Interface diffusion: +101 cells (severe)
Centroid shift: 0.008 cells (excellent)
```

**Analysis**:
- Mass conservation EXCELLENT with subcycling
- Severe interface diffusion (10× original width)
- Centroid returns almost exactly to origin
- Demonstrates first-order upwind geometric diffusion
- CFL violations handled by adaptive subcycling (19 substeps)

**Success Criteria**:
- First-order baseline: <10% mass error
- TVD target: <1% mass error
- Geometric accuracy: Centroid shift <1 cell

---

### Test 3: Shear - Linear Velocity Gradient

**Physics**:
- Circular interface subjected to linear shear flow
- Deforms into ellipse (expected behavior)
- Tests advection with velocity gradients
- Shear strain ε = γ·t = 1.0 (significant deformation)

**Parameters**:
- Domain: 64×64×4
- Circle radius: 12 cells
- Shear rate: γ = 0.01 1/step
- Velocity field: u(y) = γ·(y - y0)
- Duration: 100 steps

**Baseline Results (First-Order Upwind)**:
```
Initial mass: M0 = 1850.9
Final mass: M = 1850.88
Mass error: 0.00015% (EXCELLENT)
L2 error: 0.177 (deformation expected)
Interface diffusion: +14 cells
```

**Analysis**:
- Mass conservation EXCELLENT
- Interface deformation expected (circle → ellipse)
- L2 error reflects geometric deformation, not numerical error
- Less diffusion than rotation (lower max velocity)

**Success Criteria**:
- First-order baseline: <15% mass error
- TVD target: <2% mass error

---

### Test 4: Translation - High CFL (Stress Test)

**Physics**:
- Same as Test 1, but with aggressive CFL = 0.4
- Stresses stability and mass conservation
- Exposes first-order diffusion more clearly

**Parameters**:
- Domain: 64×64×4
- Circle radius: 12 cells
- Velocity: u_x = 0.4 (CFL = 0.4)
- Duration: 64 timesteps

**Baseline Results (First-Order Upwind)**:
```
Initial mass: M0 = 1850.9
Final mass: M = 1850.9
Mass error: 0.0000066% (EXCELLENT)
```

**Analysis**:
- Even at high CFL, subcycling maintains mass conservation
- Without subcycling: would expect 10-20% mass loss
- Demonstrates robustness of CFL-adaptive timestepping

**Success Criteria**:
- First-order baseline: <20% mass error at CFL=0.4
- TVD target: <1% mass error (should handle high CFL better)

---

## Key Findings

### Current Implementation (First-Order Upwind + Subcycling)

**Strengths**:
1. **Excellent mass conservation** (<0.001% error) due to:
   - Conservative flux formulation
   - CFL-adaptive subcycling (target CFL=0.10)
   - Zero-flux boundary conditions

2. **Stable** across all test cases:
   - Translation: CFL 0.25-0.40
   - Rotation: CFL up to 1.81 (handled by subcycling)
   - Shear: CFL 0.32

**Weaknesses**:
1. **Severe interface diffusion**:
   - Translation: 33 cells (2.75× original)
   - Rotation: 101 cells (10× original)
   - Shear: 14 cells

2. **Geometric distortion**:
   - L2 error 0.09-0.35 after one period
   - Interface smearing reduces feature resolution

3. **Computational cost**:
   - Subcycling overhead: 3-19× per timestep
   - Required for mass conservation

### Expected Improvements with TVD

**TVD schemes** (van Leer, Superbee, MUSCL) should provide:

1. **Sharper interfaces**:
   - Interface width: <5 cells diffusion (vs 30-100 current)
   - Better preservation of sharp features

2. **Better geometric fidelity**:
   - L2 error: <0.05 (vs 0.09-0.35 current)
   - Reduced shape distortion

3. **Less subcycling**:
   - Can handle higher CFL (0.3-0.4) stably
   - Lower computational overhead

4. **Comparable or better mass conservation**:
   - Target: <1% mass error
   - Already have <0.001% with subcycling, but want this WITHOUT subcycling

## Implementation Notes

### CFL-Adaptive Subcycling

The test revealed that mass conservation is excellent WITH subcycling (CFL_target=0.10). The key issue is:

**Problem**: Subcycling adds 3-19× computational overhead

**Solution Path**:
1. Implement TVD scheme to reduce diffusion at higher CFL
2. Increase CFL_target from 0.10 to 0.30 (3× speedup)
3. Maintain <1% mass error without excessive subcycling

### Test Architecture

The test suite provides:

1. **Baseline metrics** for regression detection
2. **Multiple velocity fields** to test different flow regimes
3. **Detailed diagnostics**:
   - Total mass (Σf_i)
   - L2 error (shape preservation)
   - Interface width (diffusion measure)
   - Centroid position (geometric accuracy)

4. **Flexible comparison framework**:
   - Easy to add new advection schemes
   - Compare multiple schemes side-by-side
   - Quantitative metrics for validation

## Usage

### Running the Test

```bash
cd /home/yzk/LBMProject/build
make test_vof_mass_conservation_tvd
./tests/validation/test_vof_mass_conservation_tvd
```

### Expected Output

```
[==========] Running 4 tests from 1 test suite.
[ RUN      ] VOFMassConservationTest.Translation_UniformVelocity
  Initial mass: M0 = 1850.9
  Final mass: M = 1850.9
  Mass error: 0.000020%
  EXCELLENT: Mass error <1% (TVD-level)
[       OK ] VOFMassConservationTest.Translation_UniformVelocity (425 ms)

[ RUN      ] VOFMassConservationTest.Rotation_SolidBody360
  Mass error: 0.00047%
  EXCELLENT: Mass error <1% (TVD-level)
[       OK ] VOFMassConservationTest.Rotation_SolidBody360 (680 ms)

[ RUN      ] VOFMassConservationTest.Shear_LinearGradient
  Mass error: 0.00015%
  EXCELLENT: Mass error <2% (TVD-level)
[       OK ] VOFMassConservationTest.Shear_LinearGradient (123 ms)

[ RUN      ] VOFMassConservationTest.Translation_HighCFL
  Mass error: 0.0000066%
  EXCELLENT: Mass error <1% even at CFL=0.4
[       OK ] VOFMassConservationTest.Translation_HighCFL (74 ms)

[  PASSED  ] 4 tests.
```

### Adding New Advection Schemes

To add TVD scheme comparison:

1. Add new advection kernel in `vof_solver.cu`:
```cpp
__global__ void advectFillLevelTVDKernel(...) {
    // TVD scheme implementation
}
```

2. Add scheme selection in `VOFSolver::advectFillLevel()`:
```cpp
if (scheme == UPWIND) {
    advectFillLevelUpwindKernel<<<...>>>();
} else if (scheme == TVD_VANLEER) {
    advectFillLevelTVDKernel<<<...>>>();
}
```

3. Update test to compare both schemes:
```cpp
// Run with first-order
vof_upwind.advectFillLevel(...);
float error_upwind = compute_error();

// Run with TVD
vof_tvd.advectFillLevel(...);
float error_tvd = compute_error();

// Compare
EXPECT_LT(error_tvd, error_upwind * 0.1); // TVD should be 10× better
```

## References

1. **Hirt, C. W., & Nichols, B. D. (1981)**. "Volume of fluid (VOF) method."
   *Journal of Computational Physics*, 39(1), 201-225.
   - Original VOF method, discusses donor-cell advection

2. **Rider, W. J., & Kothe, D. B. (1998)**. "Reconstructing volume tracking."
   *Journal of Computational Physics*, 141(2), 112-152.
   - Comprehensive review of VOF advection schemes
   - Compares first-order vs higher-order methods

3. **Rudman, M. (1997)**. "Volume-tracking methods for interfacial flow calculations."
   *International Journal for Numerical Methods in Fluids*, 24(7), 671-691.
   - Practical VOF implementation details
   - Mass conservation strategies

4. **van Leer, B. (1979)**. "Towards the ultimate conservative difference scheme."
   *Journal of Computational Physics*, 32(1), 101-136.
   - TVD scheme theory and implementation

5. **LeVeque, R. J. (2002)**. *Finite Volume Methods for Hyperbolic Problems*.
   Cambridge University Press.
   - Comprehensive treatment of conservative advection schemes

## Next Steps

1. **Implement TVD advection kernel**:
   - Start with van Leer limiter (simplest, robust)
   - Add Superbee for sharper interfaces
   - Compare MUSCL variants

2. **Run comparison tests**:
   - Use this test suite to measure improvement
   - Verify <1% mass error with TVD
   - Measure interface sharpness improvement

3. **Optimize performance**:
   - Reduce CFL_target from 0.10 to 0.30
   - Minimize subcycling overhead
   - Profile GPU kernel performance

4. **Validate on complex flows**:
   - Rayleigh-Taylor instability (target: <1% mass loss)
   - Rising bubble (geometric fidelity)
   - Oscillating droplet (frequency accuracy)

## Summary

This test suite provides:
- ✓ Baseline metrics for first-order upwind (EXCELLENT mass conservation with subcycling)
- ✓ Quantitative measures of interface diffusion (30-100 cells)
- ✓ Framework for comparing TVD schemes
- ✓ Regression detection for future changes
- ✓ Clear success criteria (<1% mass error, <10 cells diffusion)

**Key Insight**: Current implementation has excellent mass conservation due to aggressive subcycling (CFL=0.10), but suffers from severe interface diffusion. TVD schemes should maintain mass conservation while reducing diffusion and allowing higher CFL (less subcycling overhead).

---

**Test File**: `/home/yzk/LBMProject/tests/validation/vof/test_vof_mass_conservation_tvd.cu`
**CMake Entry**: Line 2990-3036 in `/home/yzk/LBMProject/tests/validation/CMakeLists.txt`
**Test Label**: `validation;vof;mass_conservation;tvd;advection`
**Timeout**: 600 seconds (4 test cases)
