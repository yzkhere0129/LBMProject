# Marangoni Limiter Removal - Automated Validation Framework

**Complete automated testing suite for validating the removal of Marangoni gradient and CFL limiters in LPBF simulations.**

## Table of Contents

1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [Test Strategy](#test-strategy)
4. [File Structure](#file-structure)
5. [Usage Guide](#usage-guide)
6. [Validation Framework](#validation-framework)
7. [Interpreting Results](#interpreting-results)
8. [Troubleshooting](#troubleshooting)
9. [Technical Details](#technical-details)

---

## Overview

### Problem

The current LPBF simulation has two limiters that artificially constrain Marangoni-driven flow:

1. **Gradient Limiter** (`marangoni.cu`, lines 109-114, 199-204)
   - Caps temperature gradient at 5×10⁸ K/m
   - Reduces Marangoni force by 87% in critical 2-5 μm range
   - Limits velocity to ~7 mm/s instead of expected 50-500 mm/s

2. **CFL Limiter** (`multiphysics_solver.cu`, lines 624-636)
   - Currently inactive (1400x safety margin)
   - May be removable without stability impact

### Solution

This framework provides **fully automated testing** to validate safe removal of these limiters through 6 progressive stages:

1. **BASELINE** - Current code (reference)
2. **GRAD-2X** - Relax gradient limit by 2x
3. **GRAD-10X** - Relax gradient limit by 10x
4. **GRAD-REMOVED** - Remove gradient limiter entirely
5. **CFL-REMOVED** - Remove both limiters
6. **LONG-STABILITY** - Extended run for long-term stability

---

## Quick Start

### Prerequisites

```bash
# Required
- CUDA-capable GPU
- CMake, Make
- Python 3.6+
- Compiled LBMProject in /home/yzk/LBMProject/build

# Verify
cd /home/yzk/LBMProject/build
ls visualize_lpbf_scanning  # Should exist
```

### Run Full Validation Suite

```bash
cd /home/yzk/LBMProject/validation_tests/scripts

# Make scripts executable
chmod +x *.sh *.py

# Run complete validation (takes ~3-5 hours)
./run_validation_suite.sh
```

**What happens:**
1. Backs up source files
2. Runs 6 test stages sequentially
3. Each stage: modify → compile → simulate → validate
4. Restores original source code
5. Generates comprehensive report

### View Results

```bash
# Final report
cat ../reports/validation_summary.md

# Individual test logs
ls -lh ../data/*.log

# Individual validation reports
ls -lh ../reports/*_validation.txt
```

---

## Test Strategy

### Progressive Validation

Tests are ordered by increasing aggressiveness:

| Stage | Modification | Expected v_max | Risk Level |
|-------|--------------|----------------|------------|
| BASELINE | None | ~7 mm/s | None (reference) |
| GRAD-2X | Limit 5e8→1e9 | ~10-15 mm/s | Low |
| GRAD-10X | Limit 5e8→5e9 | ~20-30 mm/s | Low-Medium |
| GRAD-REMOVED | Remove limiter | ~50-500 mm/s | Medium |
| CFL-REMOVED | Remove both | ~50-500 mm/s | Medium-High |
| LONG-STABILITY | 6000 steps | ~50-500 mm/s | High (extended) |

### Safety Features

- **Automatic backup/restore** - Original code never lost
- **Timeout protection** - 30-minute limit per test
- **Ctrl+C handling** - Clean restoration on interrupt
- **Error isolation** - One test failure doesn't stop suite
- **Independent compilation** - Each test rebuilds from scratch

### 7-Point Validation

Each test is validated against 7 criteria:

1. **Numerical Health** - No NaN/Inf values
2. **Temperature Range** - Physical bounds (0-10000 K)
3. **Velocity Range** - Expected for configuration
4. **Simulation Completion** - All timesteps executed
5. **Field Smoothness** - VTK files generated
6. **Conservation** - Mass error < 5%
7. **Physics Realism** - Matches LPBF literature

---

## File Structure

```
validation_tests/
├── scripts/
│   ├── run_validation_suite.sh     # Main test orchestrator
│   ├── run_single_test.sh          # Run individual test
│   ├── validate_results.py         # 7-point validation framework
│   ├── extract_metrics.py          # Parse log files
│   ├── compare_baseline.py         # Comparative analysis
│   └── generate_report.py          # Final report generator
│
├── data/                            # Generated during tests
│   ├── BASELINE_output.log
│   ├── GRAD-2X_output.log
│   ├── GRAD-REMOVED_output.log
│   └── ...
│
├── reports/                         # Generated during tests
│   ├── BASELINE_validation.txt
│   ├── GRAD-REMOVED_validation.txt
│   ├── validation_summary.md       # Main output
│   └── ...
│
├── backups/                         # Source backups
│   ├── marangoni.cu.backup_TIMESTAMP
│   └── multiphysics_solver.cu.backup_TIMESTAMP
│
└── README.md                        # This file
```

---

## Usage Guide

### Option 1: Run Full Suite (Recommended)

```bash
cd /home/yzk/LBMProject/validation_tests/scripts
./run_validation_suite.sh
```

**Duration:** ~3-5 hours (6 tests × 10-30 min each)

**Output:** Comprehensive report at `../reports/validation_summary.md`

### Option 2: Run Single Test

```bash
# Syntax
./run_single_test.sh TEST_NAME [NUM_STEPS]

# Examples
./run_single_test.sh BASELINE          # Quick baseline
./run_single_test.sh GRAD-REMOVED 500  # Key test, 500 steps
./run_single_test.sh LONG-STABILITY    # Extended stability
```

**Use cases:**
- Debugging specific configuration
- Quick verification after code changes
- Focused investigation of failure

### Option 3: Manual Validation

```bash
# Extract metrics from existing log
python3 extract_metrics.py ../data/GRAD-REMOVED_output.log --pretty

# Validate specific log
python3 validate_results.py \
    --log-file ../data/GRAD-REMOVED_output.log \
    --test-name GRAD-REMOVED \
    --v-min 50 --v-max 500 \
    --output ../reports/manual_validation.txt

# Compare with baseline
python3 compare_baseline.py \
    --baseline ../data/BASELINE_metrics.json \
    --test ../data/GRAD-REMOVED_metrics.json \
    --test-name GRAD-REMOVED
```

---

## Validation Framework

### 7-Point Validation Details

#### 1. Numerical Health

**Check:** No NaN or Inf in velocity, temperature, or fill level fields

**Pass criteria:**
- No NaN detected in logs
- No CUDA errors
- Velocities < 1000 mm/s (sanity check)

**Failure modes:**
- Numerical instability (timestep too large)
- Division by zero
- GPU memory corruption

#### 2. Temperature Range

**Check:** Temperature remains physical

**Pass criteria:**
- T_min ≥ 0 K (absolute zero)
- T_max ≤ 10000 K (well above Ti6Al4V vaporization at 3500 K)

**Failure modes:**
- Negative temperature (numerical error)
- Extreme heating (energy balance issue)

#### 3. Velocity Range

**Check:** Velocity within expected range for test

**Pass criteria (test-dependent):**
- BASELINE: 0-20 mm/s
- GRAD-2X/10X: 0-100 mm/s
- GRAD-REMOVED: 50-500 mm/s (LPBF literature)

**Failure modes:**
- No flow (Marangoni disabled)
- Excessive flow (runaway instability)

#### 4. Simulation Completion

**Check:** All timesteps executed

**Pass criteria:**
- Reached target step count
- No timeout
- No early termination

**Failure modes:**
- Timeout (30 minutes exceeded)
- Crash/exception
- Infinite loop

#### 5. Field Smoothness

**Check:** VTK output files generated

**Pass criteria:**
- ≥ 5 VTK files present
- Regular output intervals

**Failure modes:**
- I/O errors
- Disk full
- Corrupted fields

#### 6. Conservation

**Check:** Mass conserved within tolerance

**Pass criteria:**
- Mass error < 5%

**Failure modes:**
- VOF advection errors
- Boundary condition leaks
- Numerical diffusion

#### 7. Physics Realism

**Check:** Results match expected physics

**Pass criteria (for GRAD-REMOVED):**
- Velocity in LPBF range (50-500 mm/s)
- Velocity > baseline (Marangoni enhanced)

**Failure modes:**
- Below literature range (limiter still active)
- Above range (unphysical acceleration)

---

## Interpreting Results

### Success Indicators

**GRAD-REMOVED test passed if:**

```
✓ All 7 validation checks passed
✓ v_max = 50-500 mm/s (LPBF range)
✓ Velocity increased 10-100x vs baseline
✓ No NaN/Inf detected
✓ Simulation completed all steps
```

**Recommendation:** Remove gradient limiter in production

### Warning Signs

**Investigate if:**

```
⚠ Velocity < 50 mm/s (below expected)
⚠ Velocity increase < 10x (limiter still effective?)
⚠ Mass error 3-5% (borderline conservation)
⚠ CFL approaching 0.5 (stability margin low)
```

**Action:** Review configuration, check for other constraints

### Failure Indicators

**Do not deploy if:**

```
✗ NaN or Inf detected
✗ Simulation timeout or crash
✗ Velocity > 1000 mm/s (unphysical)
✗ Mass error > 5%
```

**Action:** Investigate root cause, try incremental approach (GRAD-2X/10X)

### Example Report Interpretation

```markdown
## Executive Summary

**Tests Run:** 6/6
**Tests Passed:** 5/6

### Key Findings

- Baseline velocity: 6.9 mm/s
- Gradient limiter removed: 234 mm/s
- Increase factor: 34x
- Physics validation: PASS (within LPBF range 50-500 mm/s)
- Stability: STABLE (no NaN/Inf, simulation completed)

## Recommendations

### PRIMARY RECOMMENDATION: REMOVE GRADIENT LIMITER

Evidence:
- Velocity increased 34x (from 6.9 to 234 mm/s)
- Simulation remained numerically stable
- Results match expected LPBF physics
```

**Interpretation:** ✓ Safe to remove gradient limiter

---

## Troubleshooting

### Build Failures

**Symptom:** CMake or Make errors

**Solutions:**
```bash
# Check build directory
ls /home/yzk/LBMProject/build

# Clean build
cd /home/yzk/LBMProject/build
rm -rf *
cmake ..
make -j8

# Check CUDA
nvcc --version
nvidia-smi
```

### Timeout Issues

**Symptom:** Tests exceed 30-minute limit

**Solutions:**
```bash
# Increase timeout in run_validation_suite.sh
TEST_TIMEOUT=3600  # 60 minutes

# Reduce steps for quick tests
SHORT_STEPS=500

# Run on fewer cores
make -j4  # Instead of -j8
```

### Python Errors

**Symptom:** Validation scripts fail

**Solutions:**
```bash
# Check Python version
python3 --version  # Should be 3.6+

# Test scripts directly
python3 scripts/validate_results.py --help

# Check for syntax errors
python3 -m py_compile scripts/*.py
```

### Source Restoration

**Symptom:** Original code not restored

**Solutions:**
```bash
# Manual restore from backup
TIMESTAMP=20251117_123456  # Use actual timestamp
cp backups/marangoni.cu.backup_$TIMESTAMP \
   /home/yzk/LBMProject/src/physics/vof/marangoni.cu

cp backups/multiphysics_solver.cu.backup_$TIMESTAMP \
   /home/yzk/LBMProject/src/physics/multiphysics/multiphysics_solver.cu

# Verify restoration
cd /home/yzk/LBMProject/build
make -j8
```

### Log Parsing Issues

**Symptom:** Metrics not extracted correctly

**Solutions:**
```bash
# Check log format
head -100 ../data/BASELINE_output.log

# Test metric extraction
python3 extract_metrics.py ../data/BASELINE_output.log --pretty

# Verify regex patterns in validate_results.py
# Look for lines like:
#   v_max = 234.5 mm/s
#   T_max = 4305.2 K
```

---

## Technical Details

### Source Code Modifications

#### Gradient Limiter (marangoni.cu)

**Original (lines 109-114):**
```cuda
if (grad_T_mag > MAX_PHYSICAL_GRAD_T) {
    float scale = MAX_PHYSICAL_GRAD_T / grad_T_mag;
    grad_T_x *= scale;
    grad_T_y *= scale;
    grad_T_z *= scale;
}
```

**GRAD-2X:** Change line 31: `MAX_PHYSICAL_GRAD_T = 1.0e9f`

**GRAD-10X:** Change line 31: `MAX_PHYSICAL_GRAD_T = 5.0e9f`

**GRAD-REMOVED:** Comment lines 109-114 and 199-204

#### CFL Limiter (multiphysics_solver.cu)

**Original (lines 624-636):**
```cuda
// STABILITY FIX: Apply CFL-based force limiter
const float MAX_CFL = 0.5f;

limitForcesByCFL_kernel<<<blocks, threads>>>(
    d_force_x_, d_force_y_, d_force_z_,
    fluid_->getVelocityX(),
    fluid_->getVelocityY(),
    fluid_->getVelocityZ(),
    dt, config_.dx, MAX_CFL,
    num_cells
);
cudaDeviceSynchronize();
```

**CFL-REMOVED:** Comment lines 624-636

### Expected Physics

**Marangoni velocity scale:**
```
v ~ (dσ/dT) × ΔT / μ

For Ti6Al4V LPBF:
  dσ/dT ≈ -3.5×10⁻⁴ N/(m·K)
  ΔT ≈ 2000-3000 K (laser heating)
  μ ≈ 4×10⁻³ Pa·s

  v ~ 175-262 mm/s
```

**Current limiter effect:**
- Limits ∇T to 5×10⁸ K/m
- Actual ∇T near laser: 1-4×10⁹ K/m
- Suppression: 87% in critical 2-5 μm range
- Result: v_max = 7 mm/s (25-40x below expected)

**After removal:**
- Full ∇T allowed: up to 4×10⁹ K/m
- Expected v_max: 50-500 mm/s
- Matches LPBF literature and experimental data

### Performance Notes

**Test duration estimates:**
- BASELINE: 10-15 minutes (1000 steps)
- GRAD-2X/10X: 10-15 minutes each
- GRAD-REMOVED: 15-20 minutes (more flow = slower)
- CFL-REMOVED: 15-20 minutes
- LONG-STABILITY: 60-90 minutes (6000 steps)

**Total suite runtime:** 3-5 hours

**Resource usage:**
- GPU: ~4-6 GB VRAM
- CPU: 4-8 cores (during compilation)
- Disk: ~2-5 GB (logs + VTK files)

---

## Next Steps After Validation

### If Tests Pass

1. **Review final report**
   ```bash
   cat reports/validation_summary.md
   ```

2. **Commit changes**
   ```bash
   cd /home/yzk/LBMProject
   git add src/physics/vof/marangoni.cu
   git commit -m "Remove Marangoni gradient limiter

   Validation results:
   - Velocity: 6.9 → 234 mm/s (34x increase)
   - Stability: Confirmed (no NaN/Inf)
   - Physics: Matches LPBF literature (50-500 mm/s)

   See validation_tests/reports/validation_summary.md"
   ```

3. **Extended testing**
   - Run production LPBF simulations
   - Test multiple materials
   - Validate against experimental data

### If Tests Fail

1. **Analyze failure mode**
   ```bash
   # Check which test failed
   grep "FAILED" reports/validation_summary.md

   # Review error details
   cat reports/GRAD-REMOVED_validation.txt
   ```

2. **Try incremental approach**
   ```bash
   # If GRAD-REMOVED failed, try partial relaxation
   ./run_single_test.sh GRAD-10X

   # Review if 10x is stable but full removal isn't
   ```

3. **Investigate root cause**
   - Review temperature fields in VTK files
   - Check CFL numbers in logs
   - Examine interface behavior

---

## Support

**Documentation:**
- This README
- Individual script help: `./script.sh --help`
- Python module docstrings

**Debugging:**
- Enable verbose output in scripts
- Check individual log files in `data/`
- Run single tests for isolation

**Code Review:**
- Source modifications are minimal (commenting out limiters)
- All changes are automatically reverted
- Backups stored with timestamps

---

**Report Issues:** Review logs and consult LPBF simulation literature

**Last Updated:** 2025-11-17
