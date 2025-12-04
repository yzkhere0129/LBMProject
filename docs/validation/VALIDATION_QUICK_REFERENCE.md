# Validation Framework - Quick Reference

**Quick access guide for developers implementing the validation strategy**

---

## Current Status at a Glance

| Metric | Current | Target | Status |
|--------|---------|--------|--------|
| Test Coverage | 65% | 85% | ⚠️ Needs improvement |
| T_max (150W) | 4,478 K | 2,400-2,800 K | ✗ Too high (1.7×) |
| v_max (150W) | 19 mm/s | 100-500 mm/s | ✗ Too low (5×) |
| Energy Balance | 4.8% error | <5% | ✓ PASS |
| Mass Conservation | 0.18% error | <0.5% | ✓ PASS |
| Grid Convergence | p = -0.69 | p > 0.5 | ✗ CRITICAL ISSUE |

---

## Regression Test Suite (Run Before Every Commit)

```bash
cd /home/yzk/LBMProject/build

# Quick smoke test (10 sec)
./tests/regression/test_baseline_150W

# Full regression suite (2 min)
ctest -L regression --output-on-failure

# Results
# ✓ PASS: All 5 critical tests passed → Safe to commit
# ✗ FAIL: Any critical test failed → DO NOT COMMIT, investigate
```

### Critical Tests (Must Pass)

1. **test_baseline_150W** (10s) - Reference simulation
   - T_max = 4,478 ± 70 K
   - v_max = 19.0 ± 2.0 mm/s
   - E_error < 5%

2. **test_stability_500step** (15s) - No divergence
   - No NaN, no T > 10,000 K
   - Runs 500 steps without crash

3. **test_energy_conservation** (5s) - Zero laser test
   - E_total conserved to <1e-5

4. **test_pure_conduction** (1s) - Analytical comparison
   - L2 error < 5%

5. **test_static_droplet** (5s) - Laplace pressure
   - ΔP within ±10% of 2σ/R

---

## Validation Workflow for New Features

### Step 1: Define Acceptance Criteria BEFORE Coding

```yaml
Feature: Substrate Cooling BC

Acceptance Criteria:
  CRITICAL:
    - T_max reduction > 500 K (compared to adiabatic)
    - Energy balance still < 5%
    - No instability (500 steps)

  HIGH:
    - Substrate flux P_sub > 0 (heat flows out)
    - Magnitude: 10-50% of P_laser

  STRETCH:
    - T_max reaches 2,400-2,800 K (literature target)
```

### Step 2: Implement Test BEFORE Feature

```cpp
// File: tests/validation/test_substrate_cooling.cu

TEST(SubstrateCooling, TemperatureReduction) {
    // Baseline: adiabatic
    float T_max_adiabatic = run_simulation(BC_ADIABATIC);

    // Test: convective BC
    float T_max_convective = run_simulation(BC_CONVECTIVE, h=1000);

    // Acceptance
    EXPECT_LT(T_max_convective, T_max_adiabatic - 500.0f);
    EXPECT_GT(T_max_convective, 2000.0f);  // Still physical
}
```

### Step 3: Implement Feature

```cpp
// Implement substrate cooling BC
// ...code...
```

### Step 4: Run Tests

```bash
make test_substrate_cooling
./tests/validation/test_substrate_cooling

# If PASS: Feature works, add to regression suite
# If FAIL: Debug using checklist (see below)
```

---

## Debugging Checklist (When Tests Fail)

### Quick Diagnosis

```bash
# Check for common issues
grep -i "nan\|inf\|diverged" simulation.log

# Visualize problem
paraview output_*.vtk
# Look for: NaN (holes), extreme values, wrong location
```

### Systematic Debug (8 Steps)

1. **Reproduce** - Can you get failure 3 times in a row?
2. **Isolate** - Does it fail with feature disabled?
3. **Energy** - Is energy conserved? (check diagnostics)
4. **Physics** - Temperature in 0-10,000 K? Velocity reasonable?
5. **Visualize** - Plot T, v fields in ParaView
6. **Stability** - CFL < 0.3? omega in (0,2)? Pe < 100?
7. **Code Review** - Index bounds? Unit conversions? Signs?
8. **Bisect** - If all else fails, git bisect to find breaking commit

### Common LPBF Bugs

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| T_max = NaN | Negative population | Check TVD limiter, omega |
| T_max > 10,000 K | Laser power too high | Scale by domain volume |
| v_max = 0 | Forces not applied | Check force→velocity coupling |
| v_max = NaN | Division by zero | Check ρ > 0 in macroscopic |
| Energy error > 10% | Missing sink/source | Check BC fluxes, radiation |
| Mass loss | VOF advection bug | Check flux limiter, BCs |

---

## Literature Validation Protocol

### Target: Mohr et al. 2020 (Ti6Al4V, 195W)

**Step 1: Exact Replication**
```bash
./run_simulation --config configs/validation/mohr2020_replication.conf
```

**Step 2: Compare Results**
```python
# Expected (Mohr 2020)
T_max_lit = 2600  # K (pyrometer measurement)
v_max_lit = 200   # mm/s (CFD, mid-range)

# Your results
T_max_sim = 2850  # K (from simulation)
v_max_sim = 180   # mm/s

# Check
T_error = abs(T_max_sim - T_max_lit) / T_max_lit * 100
v_error = abs(v_max_sim - v_max_lit) / v_max_lit * 100

# Accept if within tolerance
assert T_error < 15, f"Temperature error {T_error:.1f}% > 15%"
assert v_error < 50, f"Velocity error {v_error:.1f}% > 50%"
```

**Step 3: Sensitivity Analysis**
```bash
# Vary uncertain parameters ±10%
./param_sweep.sh --param emissivity --values 0.45,0.5,0.55
./param_sweep.sh --param penetration_depth --values 1.8e-6,2e-6,2.2e-6

# Check how much results change
# If T varies by ±200 K → uncertainty is ±200 K
```

**Acceptance Tiers:**
- **Excellent:** ±10% of literature → Publish without caveats
- **Acceptable:** ±30% of literature → Publish with discussion
- **Qualitative:** ±50% of literature → Trends only, not quantitative
- **Failed:** >50% deviation → Debug before publishing

---

## CI Pipeline (Continuous Integration)

### Local Pre-Commit (2 min)

```bash
# Run automatically via git hook
# File: .git/hooks/pre-commit

#!/bin/bash
cd /home/yzk/LBMProject/build
make -j8 > /dev/null || exit 1
ctest -L critical --output-on-failure || exit 1
echo "Tests passed, safe to commit"
```

### Pre-Merge (5 min)

```yaml
# GitHub Actions: .github/workflows/ci.yml
# Runs on self-hosted GPU runner

on: [pull_request]

jobs:
  test:
    runs-on: self-hosted
    steps:
      - uses: actions/checkout@v3
      - run: cmake -B build
      - run: make -C build -j8
      - run: cd build && ctest -L regression
```

### Nightly Build (30 min)

```bash
# Cron job: 2 AM every day
# /etc/cron.d/lbm-nightly

0 2 * * * /home/yzk/LBMProject/scripts/nightly_build.sh

# nightly_build.sh:
# - Pull latest main
# - Build
# - Run all tests (including slow ones)
# - Generate report
# - Email team if any failures
```

---

## Test Report Interpretation

### Sample Report

```
Test Results: 69/72 passed
Duration: 4m 32s

Failed Tests:
  1. test_grid_convergence_order
     Status: KNOWN ISSUE (hardware limited)
     Impact: LOW - defer to cloud GPU

  2. test_velocity_magnitude
     Status: ACTIVE WORK (calibration in progress)
     Impact: MEDIUM - affects accuracy, not stability

  3. test_realistic_temperature
     Status: ACTIVE WORK (needs substrate BC)
     Impact: HIGH - primary objective
```

### What To Do

| Failed Tests | Action |
|-------------|--------|
| 0 | ✓ All good, safe to merge |
| 1-2 KNOWN ISSUES | ⚠️ Check if expected, document |
| 1-2 NEW FAILURES | ⚠️ Investigate, may be feature bug |
| 3+ FAILURES | ✗ BLOCK merge, something is broken |

---

## Key Metrics Summary

### Before Improvements (Current Baseline)

```
T_max: 4,478 K  (1.7× too high vs Mohr 2020)
v_max: 19 mm/s  (5-26× too low vs literature)
E_error: 4.8%   (acceptable)
M_error: 0.18%  (excellent)
Stability: OK   (500 steps no divergence)
Grid conv: FAIL (p = -0.69, hardware limited)
```

### After Improvements (Target)

```
T_max: 2,400-3,200 K  (within ±15% of literature)
v_max: 70-650 mm/s    (within ±30% of literature)
E_error: < 5%         (maintain current)
M_error: < 0.5%       (maintain current)
Stability: OK         (500+ steps)
Grid conv: p > 0.5    (run on cloud GPU)
```

### Improvement Roadmap

| Priority | Improvement | Expected Δ T_max | Expected Δ v_max |
|----------|-------------|------------------|------------------|
| 1 | Substrate cooling BC | -500 to -1000 K | <10% change |
| 2 | Parameter calibration (ε, δ) | -300 to -600 K | <10% change |
| 3 | Variable viscosity μ(T) | <100 K change | +100% to +400% |
| 4 | Grid convergence (cloud GPU) | (Validation only) | - |

**Timeline:** 4-6 weeks for full implementation + validation

---

## Quick Commands Reference

```bash
# Build
cd /home/yzk/LBMProject/build && make -j8

# Run baseline test
./tests/regression/test_baseline_150W

# Run all regression tests
ctest -L regression

# Run validation tests (long)
ctest -L validation

# Generate test report
python3 ../scripts/generate_test_report.py

# View test dashboard
xdg-open ../test_results/dashboard.html

# Visualize results
paraview output_*.vtk

# Check energy balance
python3 ../scripts/analyze_energy.py diagnostics.csv

# Compare to literature
python3 ../scripts/compare_to_mohr2020.py results.csv
```

---

## Files and Locations

```
/home/yzk/LBMProject/
├── tests/
│   ├── regression/          # 10 regression tests
│   │   ├── test_baseline_150W.cu
│   │   ├── test_stability_500step.cu
│   │   └── ...
│   ├── validation/          # Literature comparison
│   │   ├── test_mohr2020_replication.cu
│   │   └── ...
│   └── unit/                # 49 unit tests
│
├── scripts/
│   ├── generate_test_report.py
│   ├── analyze_energy.py
│   └── compare_to_mohr2020.py
│
├── docs/validation/
│   ├── VALIDATION_FRAMEWORK_COMPREHENSIVE.md  (Full spec)
│   └── VALIDATION_QUICK_REFERENCE.md         (This file)
│
└── .github/workflows/
    └── ci.yml               # CI configuration
```

---

## Getting Help

**When to Ask:**
- Test fails and you don't understand why
- Acceptance criteria unclear
- Need help setting up CI
- Results don't match literature and you've tried everything

**Where to Ask:**
- File issue on GitHub with "validation" label
- Tag @testing-specialist in PR comments
- Email team with test report attached

**What to Include:**
- Test name that failed
- Full error message
- Commit hash (git rev-parse HEAD)
- Hardware info (nvidia-smi)
- VTK files if visualization-related

---

**Document:** Quick Reference for Validation Framework
**Version:** 1.0
**Date:** 2025-11-19
**Full Documentation:** VALIDATION_FRAMEWORK_COMPREHENSIVE.md
