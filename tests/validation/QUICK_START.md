# Week 2 Convergence Test Suite - Quick Start

## One-Command Test Execution

```bash
cd /home/yzk/LBMProject
./tests/validation/run_convergence_validation.sh
```

**That's it!** The script will:
- Run all 6 validation tests
- Display color-coded PASS/FAIL results
- Provide detailed diagnostics on failures
- Exit 0 if all pass, 1 if any fail

---

## Quick Validation (3 minutes)

Skip expensive convergence studies:

```bash
./tests/validation/run_convergence_validation.sh --quick
```

Runs only:
- Configuration Parser Test
- CFL Stability Check
- Regression Test (50W baseline)

---

## Build Tests First Time

```bash
cd /home/yzk/LBMProject
mkdir -p build && cd build
cmake ..
make test_grid_convergence test_timestep_convergence \
     test_energy_conservation_timestep test_regression_50W \
     test_cfl_stability test_config_parser -j4
cd ..
```

---

## Expected Results

### Before Bug Fixes
```
Test 1: Configuration Parser          ✗ FAIL
Test 2: CFL Stability Check            ✓ PASS
Test 3: Regression (50W baseline)      ✓ PASS
Test 4: Grid Independence              ✗ FAIL
Test 5: Timestep Convergence           ✗ FAIL
Test 6: Energy Conservation            ✗ FAIL

Results: 2 passed, 4 failed
✗ FAILURES DETECTED - Do not proceed to Week 3
```

### After Bug Fixes
```
Test 1: Configuration Parser          ✓ PASS
Test 2: CFL Stability Check            ✓ PASS
Test 3: Regression (50W baseline)      ✓ PASS
Test 4: Grid Independence              ✓ PASS
Test 5: Timestep Convergence           ✓ PASS
Test 6: Energy Conservation            ✓ PASS

Results: 6 passed, 0 failed
✓ ALL TESTS PASSED - Code verified for Week 3
```

---

## Critical: Fix Order

1. **Fix Config Parser FIRST** (blocks all other tests)
2. Fix Time Integration (60-145% temporal divergence)
3. Fix Energy Balance (if needed)
4. Re-run full test suite
5. Confirm ALL PASS before Week 3

---

## Individual Test Commands

```bash
cd /home/yzk/LBMProject

# CRITICAL: Run this first
./build/tests/validation/test_config_parser

# Fast tests
./build/tests/validation/test_cfl_stability
./build/tests/validation/test_regression_50W

# Expensive tests (5-10 min each)
./build/tests/validation/test_grid_convergence
./build/tests/validation/test_timestep_convergence
./build/tests/validation/test_energy_conservation_timestep
```

---

## What Each Test Does

| Test | What It Checks | Time |
|------|----------------|------|
| Config Parser | Reads num_steps from file (not hardcoded) | 30s |
| CFL Stability | All timesteps numerically stable | <1s |
| Regression | 50W baseline matches Week 1 results | 2min |
| Grid Convergence | Results independent of grid spacing | 10min |
| Timestep Convergence | Results independent of timestep | 10min |
| Energy Conservation | Energy balanced at all timesteps | 10min |

---

## Common Issues

### "ERROR: Test executable not found"
```bash
cd /home/yzk/LBMProject/build
cmake .. && make test_config_parser -j4
```

### "Configuration parser bug detected"
This is EXPECTED before fixes. Fix `src/config/simulation_config.cpp`.

### "Temporal divergence 60-145%"
This is EXPECTED before fixes. Indicates time integration bug.

---

## Full Documentation

- **Complete Guide:** `/home/yzk/LBMProject/tests/validation/CONVERGENCE_TEST_README.md`
- **Delivery Report:** `/home/yzk/LBMProject/WEEK2_CONVERGENCE_TEST_SUITE_DELIVERY.md`

---

**Pro Tip:** Add to pre-commit hook:
```bash
./tests/validation/run_convergence_validation.sh --quick
```
