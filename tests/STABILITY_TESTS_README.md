# Stability Regression Test Suite

## Overview

This test suite ensures that critical stability fixes for the thermal LBM solver remain in place and functional. It guards against regressions that could reintroduce numerical instabilities observed at high Peclet numbers (Pe > 10).

## Background

### Problem Context

In Laser Powder Bed Fusion (LPBF) simulations, the thermal solver encounters challenging conditions:

- **High Peclet Number**: Pe = v·L/α can reach 10-20 in melt pool
- **Strong Advection**: Marangoni convection creates velocities >> sound speed
- **Numerical Instability**: Without proper limiters, negative populations → NaN → divergence

### Stability Fixes Implemented

1. **TVD Flux Limiter** (lattice_d3q7.cu::computeThermalEquilibrium)
   - Prevents negative thermal populations at high velocity
   - Applied when |c_i·u/cs²| > threshold

2. **Temperature Bounds** (thermal_lbm.cu::computeTemperatureKernel)
   - Clamps temperature to [0, 7000] K
   - Prevents radiation BC from generating NaN (T^4 term)

3. **Omega Reduction** (thermal_lbm.cu::ThermalLBM constructor)
   - Caps omega_T at 1.50 (previously 1.95)
   - Provides safety margin from instability limit (omega = 2.0)

## Test Structure

```
tests/
├── unit/stability/
│   ├── test_flux_limiter.cu              # Unit test for TVD limiter
│   └── test_temperature_bounds.cu        # Unit test for T bounds
├── integration/stability/
│   └── test_high_pe_stability.cu         # End-to-end stability test
├── regression/stability/
│   └── test_omega_reduction.cu           # Omega cap regression test
└── performance/
    └── test_flux_limiter_overhead.cu     # Performance overhead test
```

## Test Descriptions

### 1. Unit Test: Flux Limiter (`test_flux_limiter.cu`)

**Purpose**: Verify flux limiter prevents negative populations

**Test Cases**:
- `PreventNegativePopulations`: Extreme velocity (v=5.0 >> cs=0.577)
- `ConservesTemperature`: Mass conservation with limiter
- `VaryingPecletNumbers`: Range Pe ~ 0.1 to 50
- `LowVelocityUnaffected`: Limiter inactive at Pe < 1
- `MultidirectionalHighVelocity`: 3D high-velocity flow
- `GPUFluxLimiterCorrectness`: Device kernel validation
- `RandomizedStressTest`: 100 random high-velocity scenarios

**Pass Criteria**:
- ALL g_eq ≥ 0 for all velocities
- Temperature conserved: Σ g_eq = T (within 0.1%)
- Low-velocity matches analytical solution

**Failure Modes**:
- Negative populations → Immediate FAIL
- Temperature not conserved → Limiter broken
- Low-velocity altered → Limiter over-active

---

### 2. Unit Test: Temperature Bounds (`test_temperature_bounds.cu`)

**Purpose**: Verify temperature clamping to [0, 7000] K

**Test Cases**:
- `UpperBoundEnforcement`: Extreme heating → T ≤ 7000 K
- `LowerBoundEnforcement`: Extreme cooling → T ≥ 0 K
- `NormalRangeUnaffected`: T ∈ [300, 5000] K unchanged
- `GPUBoundsKernelCorrectness`: Device kernel validation
- `BoundsWithPhaseChange`: Bounds work with phase change solver
- `RadiationBCWithBounds`: No NaN from T^4 term

**Pass Criteria**:
- All T ∈ [0, 7000] K always
- No NaN, no Inf
- Normal temps (< 5000 K) unaffected

**Failure Modes**:
- T < 0 → Unphysical
- T > 7000 K → Evaporation model overflow
- NaN in radiation BC → Critical failure

---

### 3. Integration Test: High-Pe Stability (`test_high_pe_stability.cu`)

**Purpose**: End-to-end stability test at Pe ≈ 10

**Test Cases**:
- `ThermalSolver500Steps`: 500-step run at v=5.0, T=2500 K
- `VaryingVelocityStability`: Velocity ramp 0 → 10
- `LocalizedHighVelocityRegion`: Gaussian melt pool profile
- `HeatSourceWithHighVelocity`: Laser heating + Marangoni flow

**Pass Criteria**:
- No divergence (NaN, Inf) in 500 steps
- T_max < 15000 K (reasonable bound)
- T_min > 0 K (physical)
- Simulation completes successfully

**Failure Modes**:
- Divergence → Stability fix broken
- T runaway → Bounds not enforced
- Crash → Numerical issue

**Benchmark**: Without fixes, divergence occurs at step ~50-200

---

### 4. Regression Test: Omega Reduction (`test_omega_reduction.cu`)

**Purpose**: Ensure omega_T never exceeds 1.50

**Test Cases**:
- `OmegaNeverExceeds1_50`: Test range of diffusivities
- `OmegaCalculationFormula`: Verify tau = α/cs² + 0.5
- `CappedOmegaReasonableDiffusivity`: Effective α > 0
- `OldLimitNotUsed`: Confirm 1.50 cap (not 1.95)
- `MultipleInstancesConsistency`: Same α → same omega
- `BackwardCompatibleConstructor`: Old API also capped
- `OmegaCapAppliedAtConstruction`: Cap immediate, not runtime

**Pass Criteria**:
- omega_T ≤ 1.50 for ALL diffusivities
- omega_T ≥ 0.5 (not over-capped)
- Warning printed when capping applied

**Failure Modes**:
- omega_T > 1.50 → Regression to unsafe value
- omega_T > 1.95 → Using old (unsafe) cap
- omega_T changes during runtime → Not constant

---

### 5. Performance Test: Flux Limiter Overhead (`test_flux_limiter_overhead.cu`)

**Purpose**: Ensure limiter doesn't degrade performance

**Test Cases**:
- `GPUEquilibriumThroughput`: 10M eq/s target
- `CPUEquilibriumThroughput`: 7M eq in < 500ms
- `FullLBMStepThroughput`: > 100k cells/s
- `LowVsHighVelocityOverhead`: < 20% overhead
- `ScalingWithProblemSize`: O(1) complexity

**Pass Criteria**:
- GPU: > 10 M eq/s
- CPU: < 500 ms for 7M equilibria
- Overhead: < 20% vs uncapped
- No memory bandwidth impact

**Failure Modes**:
- > 20% overhead → Limiter too expensive
- Scaling not O(1) → Algorithmic issue

---

## Running Tests

### Quick Test (Pre-Commit)

```bash
cd /home/yzk/LBMProject/build
ctest -R "flux_limiter|temperature_bounds|omega_reduction" --output-on-failure
```

**Expected Time**: < 30 seconds
**Purpose**: Fast unit tests before every commit

### Medium Test (Pre-Push)

```bash
ctest -R "high_pe_stability" --output-on-failure
```

**Expected Time**: < 2 minutes
**Purpose**: Integration test before pushing to repository

### Full Test Suite

```bash
ctest -R "stability" --output-on-failure
```

**Expected Time**: < 5 minutes
**Purpose**: Complete stability verification

### Performance Benchmarks

```bash
./tests/performance/test_flux_limiter_overhead --gtest_filter="*Throughput*"
```

**Expected Time**: < 1 minute
**Purpose**: Measure computational overhead

---

## Continuous Integration

### GitHub Actions Workflow

Create `.github/workflows/stability_checks.yml`:

```yaml
name: Stability Regression Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Install CUDA
        uses: Jimver/cuda-toolkit@v0.2.11
        with:
          cuda: '12.0.0'

      - name: Build Project
        run: |
          mkdir build && cd build
          cmake ..
          make -j4

      - name: Run Unit Tests
        run: |
          cd build
          ctest -R "flux_limiter|temperature_bounds" --output-on-failure

      - name: Run Regression Tests
        run: |
          cd build
          ctest -R "omega_reduction" --output-on-failure

      - name: Run Integration Tests
        run: |
          cd build
          ctest -R "high_pe_stability" --output-on-failure

      - name: Performance Benchmarks
        run: |
          cd build
          ./tests/performance/test_flux_limiter_overhead
```

---

## Expected Test Results

### All Tests Passing

```
[==========] Running 5 test suites
[----------] 8 tests from FluxLimiterTest
[ RUN      ] FluxLimiterTest.PreventNegativePopulations
[       OK ] FluxLimiterTest.PreventNegativePopulations (12 ms)
...
[  PASSED  ] 42 tests.
```

### Regression Detected

```
[ RUN      ] FluxLimiterTest.PreventNegativePopulations
/path/test_flux_limiter.cu:45: Failure
REGRESSION: Negative population detected at q=2 (ux=5, T=1000)
[  FAILED  ] FluxLimiterTest.PreventNegativePopulations (8 ms)
```

**Action**: DO NOT MERGE. Investigate code change that removed limiter.

---

## Debugging Failed Tests

### Negative Population Failure

**Symptom**: `PreventNegativePopulations` fails
**Cause**: Flux limiter removed or disabled
**Fix**: Check `lattice_d3q7.cu::computeThermalEquilibrium()`

```cpp
// MUST HAVE:
float cu_cs2 = cu / CS2;
if (fabs(cu_cs2) > 2.0f) {
    // Apply TVD limiter
    ...
}
```

### Temperature Bounds Failure

**Symptom**: `UpperBoundEnforcement` fails, T > 7000 K
**Cause**: Clamp removed from `computeTemperatureKernel`
**Fix**: Check `thermal_lbm.cu`:

```cpp
// MUST HAVE:
T = fmaxf(T, 0.0f);
T = fminf(T, 7000.0f);
```

### Omega Regression

**Symptom**: `OmegaNeverExceeds1_50` fails, omega > 1.50
**Cause**: Cap removed from constructor
**Fix**: Check `ThermalLBM::ThermalLBM()`:

```cpp
// MUST HAVE:
if (omega_T_ >= 1.50f) {
    omega_T_ = 1.50f;  // NOT 1.95f!
    tau_T_ = 1.0f / omega_T_;
}
```

### Integration Test Divergence

**Symptom**: `ThermalSolver500Steps` diverges at step 127
**Cause**: Multiple stability fixes missing
**Fix**: Run unit tests to isolate which fix is broken

---

## Acceptance Criteria

Before merging ANY code that touches thermal solver:

1. ALL unit tests must pass
2. Integration test must complete 500 steps without divergence
3. Omega regression test must pass
4. Performance overhead must be < 20%

**Zero tolerance for stability regressions.**

---

## Contact

For questions about these tests:
- **Test Designer**: Testing Specialist
- **Stability Fixes**: CFD-CUDA Architect
- **Issue Tracking**: Create GitHub issue with tag `stability-regression`

## References

1. He, X., Chen, S., & Doolen, G. D. (1998). *A novel thermal model for the lattice Boltzmann method*. JCP, 146(1), 282-300.
2. Mohamad, A.A. (2011). *Lattice Boltzmann Method*. Springer.
3. Project Documentation: `/home/yzk/LBMProject/THERMAL_INVERSION_DIAGNOSIS.md`
