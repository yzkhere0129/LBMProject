# Week 2: Automated Convergence Validation Test Suite

## Overview

This directory contains a comprehensive automated test suite to validate grid and temporal convergence of the LBM solver. The tests are designed to detect the critical bugs discovered during Week 2 verification and prevent regressions in future development.

## Critical Bugs Addressed

**Week 2 Verification Findings:**
1. **Configuration Parser Bug** - `num_steps` parameter hardcoded to 6000, ignoring config file values
2. **Temporal Divergence** - 60-145% error between different timesteps, indicating time integration bug
3. **Energy Conservation Concern** - Need verification that energy is conserved during laser heating

## Test Suite Components

### Test 1: Grid Independence Validator
**File:** `test_grid_convergence.cu`
**Purpose:** Validates spatial discretization convergence

**Configuration:**
- Power: 50W
- Physical time: 300 microseconds
- Grid levels: 4μm (coarse), 2μm (baseline), 1μm (fine)
- Metrics: T_max, melt pool volume, energy balance

**Success Criteria:**
- All metrics converge within 5% between grid levels
- Recommended grid: 2μm (baseline)

**Expected Output (when fixed):**
```
Grid Convergence Test: PASS
  Coarse vs Baseline: 3.2% ✓ (< 5%)
  Baseline vs Fine:   2.1% ✓ (< 5%)
  Recommended grid:   2μm (baseline)
```

**Current Status:** Expected to FAIL until grid-dependent issues are resolved

---

### Test 2: Timestep Convergence Validator
**File:** `test_timestep_convergence.cu`
**Purpose:** Validates temporal discretization convergence

**Configuration:**
- Power: 50W
- Grid: 2μm (fixed)
- Timestep levels: 0.2μs (coarse), 0.1μs (baseline), 0.05μs (fine)
- All cases run to same physical time (300μs)

**Success Criteria:**
- Temperature curves overlap within 5% at all times
- CFL numbers within stability limits
- Recommended timestep: 0.1μs (baseline)

**Expected Output (before fix):**
```
Timestep Convergence Test: FAIL
  Coarse vs Baseline: 60.6% ✗ (> 5%)
  Baseline vs Fine:   53.3% ✗ (> 5%)

  ERROR MAGNITUDE: >50% (SEVERE - time integration bug)
  This matches Week 2 findings (60-145% divergence)
```

**Expected Output (after fix):**
```
Timestep Convergence Test: PASS
  Coarse vs Baseline: 2.8% ✓ (< 5%)
  Baseline vs Fine:   1.5% ✓ (< 5%)
  Recommended timestep: 0.1μs (baseline)
```

---

### Test 3: Energy Conservation Validator
**File:** `test_energy_conservation_timestep.cu`
**Purpose:** Validates thermodynamic consistency across timesteps

**Configuration:**
- Power: 50W
- Three timestep levels (0.2μs, 0.1μs, 0.05μs)
- Energy balance check: |P_in - P_out - dE/dt| < 5%

**Success Criteria:**
- Energy conserved (<5% error) for ALL timesteps

**Expected Output:**
```
Energy Conservation Test: PASS
  dt=0.2μs: Energy error 3.2% ✓ (< 5%)
  dt=0.1μs: Energy error 2.1% ✓ (< 5%)
  dt=0.05μs: Energy error 1.8% ✓ (< 5%)
```

---

### Test 4: Regression Test (50W Baseline)
**File:** `test_regression_50W.cu`
**Purpose:** Prevents breaking changes to validated baseline case

**Known-Good Values (Week 1):**
- T_max = 2563K ± 50K
- Energy error < 5%
- No NaN or divergence

**Success Criteria:**
- Current results match known-good values within tolerances

**Expected Output:**
```
Regression Test: PASS
  T_max:        2560 K (diff = 3 K < 50 K)
  Energy error: 3.6% (< 5%)
  No NaN/Inf:   OK
  Configuration parser: OK
```

---

### Test 5: CFL Stability Check
**File:** `test_cfl_stability.cu`
**Purpose:** Verifies numerical stability criteria

**CFL Criteria:**
- Thermal: CFL_thermal = α × dt / dx² < 0.5
- Fluid: CFL_fluid = u_max × dt / dx < 0.1

**Success Criteria:**
- All test cases satisfy CFL criteria
- Warn if close to stability limit (>80% of limit)

**Expected Output:**
```
CFL Stability Check: PASS

  Test Case                  | CFL_thermal | CFL_fluid | Status
  ---------------------------|-------------|-----------|-------
  Grid 4μm, dt=0.1μs        |      0.363  |    0.050  | OK
  Grid 2μm, dt=0.1μs        |      0.145  |    0.100  | WARNING
  Grid 1μm, dt=0.1μs        |      0.580  |    0.200  | UNSTABLE
```

---

### Test 6: Configuration Parser Test
**File:** `test_config_parser.cu`
**Purpose:** Detects hardcoded parameter bug

**Test Method:**
1. Create config with unusual `num_steps` value (12345)
2. Run simulation
3. Verify simulation executes EXACTLY 12345 steps

**Success Criteria:**
- Simulation respects config file parameter

**Expected Output (before fix):**
```
Configuration Parser Test: FAIL

  Expected: 12345 steps
  Executed: 6000 steps
  Error:    51.2%

  BUG: Configuration parser uses hardcoded 6000 steps
  This is the EXACT bug found in Week 2 timestep study

  CRITICAL: Do NOT trust any convergence studies until this is fixed!
```

**Expected Output (after fix):**
```
Configuration Parser Test: PASS

  Expected: 12345 steps
  Executed: 12345 steps
  Match:    EXACT

  The Week 2 configuration parser bug has been FIXED!
```

---

## Build Instructions

### Prerequisites
- CUDA Toolkit
- CMake 3.18+
- C++17 compiler

### Build All Tests

```bash
cd /home/yzk/LBMProject
mkdir -p build
cd build
cmake ..
make test_grid_convergence test_timestep_convergence test_energy_conservation_timestep \
     test_regression_50W test_cfl_stability test_config_parser -j4
```

### Build Individual Test

```bash
cd /home/yzk/LBMProject/build
make test_config_parser
```

---

## Running Tests

### Automated Test Suite (Recommended)

```bash
cd /home/yzk/LBMProject
./tests/validation/run_convergence_validation.sh
```

**Options:**
- `--quick` : Skip expensive convergence studies (fast tests only)
- `--verbose` : Show detailed output from each test

**Exit codes:**
- `0` : All tests passed (safe to proceed to Week 3)
- `1` : One or more tests failed (DO NOT PROCEED)

### Run Individual Tests

```bash
cd /home/yzk/LBMProject

# Fast tests (run these first)
./build/tests/validation/test_config_parser
./build/tests/validation/test_cfl_stability
./build/tests/validation/test_regression_50W

# Expensive tests (3 simulations each, 5-10 minutes)
./build/tests/validation/test_grid_convergence
./build/tests/validation/test_timestep_convergence
./build/tests/validation/test_energy_conservation_timestep
```

### Run via CTest

```bash
cd /home/yzk/LBMProject/build

# Run all validation tests
ctest -L "validation;week2" --output-on-failure

# Run specific test
ctest -R ConfigurationParserValidation --verbose

# Run only fast tests
ctest -L "validation" -E "Convergence|Energy" --output-on-failure
```

---

## Interpreting Results

### All Tests PASS
```
========================================
ALL TESTS PASSED
========================================

Code is verified for Week 3 development!

Validated:
  - Grid independence achieved
  - Temporal convergence verified
  - Energy conservation confirmed
  - Numerical stability ensured
  - Configuration parser working
  - No regressions detected
```

**Action:** Safe to proceed with Week 3 development

---

### Configuration Parser FAIL
```
CRITICAL: Configuration Parser Failed

The configuration parser bug prevents all other tests from being valid.
This is the hardcoded num_steps bug discovered in Week 2.

REQUIRED ACTION:
  1. Fix configuration parser to read num_steps/total_steps correctly
  2. Re-run this test suite
  3. Do NOT proceed to other tests until this PASSES
```

**Action:**
1. Check `src/config/simulation_config.cpp`
2. Verify `TimeConfig::n_steps` parameter reading
3. Fix hardcoded default value (likely 6000)
4. Re-run: `./build/tests/validation/test_config_parser`

---

### Timestep Convergence FAIL (>50% error)
```
FAIL: Temporal convergence not achieved

Reason: Temperature errors exceed 5% tolerance
  Coarse vs Baseline: 60.6%
  Baseline vs Fine:   53.3%

Diagnosis:
  ERROR MAGNITUDE: >50% (SEVERE - likely time integration bug)
  This matches Week 2 findings (60-145% divergence)

  Possible causes:
    - Laser energy deposition discretization error
    - Boundary condition timestep dependency
    - Phase change coupling instability
```

**Action:**
1. Run `./build/apps/diagnose_energy_balance` for detailed analysis
2. Check laser source term discretization
3. Verify boundary conditions are timestep-independent
4. Review phase change solver coupling

---

### Energy Conservation FAIL
```
FAIL: Energy conservation violated

  dt=0.2μs: Energy error 8.5% (exceeds 5% tolerance)
  dt=0.1μs: Energy error 7.2% (exceeds 5% tolerance)
  dt=0.05μs: Energy error 6.9% (exceeds 5% tolerance)

Error is CONSISTENT across timesteps:
  Possible causes:
    - Missing energy term (e.g., kinetic energy)
    - Incorrect formula for heat loss (evap/radiation)
    - Boundary flux not properly accounted
```

**Action:**
1. Check Week 1 evaporation fix (660× bug)
2. Verify radiation BC formula (Stefan-Boltzmann)
3. Check substrate cooling implementation
4. Review energy balance calculation

---

## Configuration Files

Test configuration files are located in:
```
/home/yzk/LBMProject/configs/validation/
```

**Grid convergence configs:**
- `lpbf_50W_grid_4um.conf` (coarse: 100×50×25 cells)
- `lpbf_50W_grid_2um.conf` (baseline: 200×100×50 cells)
- `lpbf_50W_grid_1um.conf` (fine: 400×200×100 cells)

**Timestep convergence configs:**
- `lpbf_50W_dt_020us.conf` (coarse: dt=0.2μs, 1500 steps → 300μs)
- `lpbf_50W_dt_010us.conf` (baseline: dt=0.1μs, 3000 steps → 300μs)
- `lpbf_50W_dt_005us.conf` (fine: dt=0.05μs, 6000 steps → 300μs)

---

## Integration with CI/CD

### Add to Pre-Commit Hook

```bash
#!/bin/bash
# .git/hooks/pre-commit

echo "Running Week 2 validation tests..."
./tests/validation/run_convergence_validation.sh --quick

if [ $? -ne 0 ]; then
    echo "ERROR: Validation tests failed"
    echo "Fix failing tests before committing"
    exit 1
fi
```

### Weekly Regression Testing

```bash
# crontab entry
0 2 * * 1 cd /home/yzk/LBMProject && ./tests/validation/run_convergence_validation.sh > /tmp/weekly_validation.log 2>&1
```

---

## Troubleshooting

### Test executable not found
```bash
ERROR: Test executable not found
Expected: ./build/tests/validation/test_config_parser
```

**Solution:**
```bash
cd /home/yzk/LBMProject/build
cmake ..
make test_config_parser
```

---

### Simulation fails during test
```bash
ERROR: Simulation failed with return code 1
FAIL: Simulation failed for Coarse (dt=0.2μs)
```

**Solution:**
1. Check log file: `cat /tmp/timestep_convergence_020us.log`
2. Look for NaN, segfault, or CUDA errors
3. Verify GPU memory availability
4. Check config file syntax

---

### CFL warning but test passes
```bash
WARNING: 1 case(s) close to stability limit
  Grid 2μm, dt=0.1μs: CFL_fluid = 0.100 (at limit 0.1)
```

**Solution:**
- This is a warning, not a failure
- Consider reducing timestep for production runs
- Monitor velocity field for oscillations

---

## File Manifest

**Test executables:**
- `test_grid_convergence.cu` (520 lines)
- `test_timestep_convergence.cu` (480 lines)
- `test_energy_conservation_timestep.cu` (380 lines)
- `test_regression_50W.cu` (350 lines)
- `test_cfl_stability.cu` (290 lines)
- `test_config_parser.cu` (370 lines)

**Infrastructure:**
- `run_convergence_validation.sh` (220 lines) - Automated test runner
- `CMakeLists.txt` (updated) - Build configuration
- `CONVERGENCE_TEST_README.md` (this file) - Documentation

**Configuration files:**
- 6 validation configs in `/configs/validation/`

**Total:** 6 test executables + 1 shell script + documentation

---

## Expected Test Run Time

**Quick mode (--quick):**
- Test 6: Config Parser (~30 seconds)
- Test 5: CFL Stability (<1 second)
- Test 4: Regression Test (~2 minutes)
- **Total: ~3 minutes**

**Full mode (all tests):**
- Test 1: Grid Convergence (~10 minutes, 3 simulations)
- Test 2: Timestep Convergence (~10 minutes, 3 simulations)
- Test 3: Energy Conservation (~10 minutes, 3 simulations)
- Test 4: Regression Test (~2 minutes)
- Test 5: CFL Stability (<1 second)
- Test 6: Config Parser (~30 seconds)
- **Total: ~35 minutes**

---

## Success Metrics

### Week 2 Completion Criteria

- [x] 6 automated tests created
- [x] Test runner script implemented
- [x] CMakeLists.txt integration complete
- [x] Configuration files generated
- [x] Documentation written

### Week 3 Readiness Criteria

Before proceeding to Week 3, ensure:

- [ ] Test 6 (Config Parser) **PASSES** (CRITICAL)
- [ ] Test 5 (CFL Stability) **PASSES**
- [ ] Test 4 (Regression) **PASSES**
- [ ] Test 1 (Grid Convergence) **PASSES** (within 5%)
- [ ] Test 2 (Timestep Convergence) **PASSES** (within 5%)
- [ ] Test 3 (Energy Conservation) **PASSES** (<5% error)

---

## Contact & Support

**Issues:** Report bugs discovered by tests to the development team
**Logs:** All test logs saved to `/tmp/test_*.log`
**Questions:** See main project README or Week 2 validation reports

---

**Last Updated:** 2025-11-20
**Test Suite Version:** 1.0
**Status:** Ready for initial testing (expected failures before bug fixes)
