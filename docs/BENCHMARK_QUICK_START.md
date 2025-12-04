# Benchmark Comparison Quick Start Guide

## Overview

This guide explains how to run the LBM-CUDA validation tests and compare results with waLBerla.

## File Locations

### Test Plan and Documentation
```
/home/yzk/LBMProject/docs/BENCHMARK_COMPARISON_TEST_PLAN.md  - Full test plan
/home/yzk/LBMProject/docs/BENCHMARK_QUICK_START.md          - This guide
```

### Existing Validation Tests (CUDA/C++)
```
/home/yzk/LBMProject/tests/validation/test_pure_conduction.cu     - 1D diffusion
/home/yzk/LBMProject/tests/validation/test_stefan_problem.cu      - Phase change
/home/yzk/LBMProject/tests/validation/test_grid_convergence.cu    - Grid study
/home/yzk/LBMProject/tests/unit/thermal/test_thermal_lbm.cu       - Thermal solver
/home/yzk/LBMProject/tests/unit/laser/test_laser_source.cu        - Laser source
```

### Benchmark Scripts
```
/home/yzk/LBMProject/scripts/benchmark/run_benchmark_comparison.sh     - Main runner
/home/yzk/LBMProject/scripts/benchmark/compare_thermal_solutions.py    - Python analysis
```

### Configuration Files
```
/home/yzk/LBMProject/configs/benchmark/pure_conduction_2um.conf   - 1D diffusion config
/home/yzk/LBMProject/configs/benchmark/gaussian_source_2um.conf   - Laser source config
/home/yzk/LBMProject/configs/benchmark/stefan_problem_2um.conf    - Phase change config
```

### waLBerla Reference
```
/home/yzk/walberla/tests/lbm/DiffusionTest.cpp                    - Diffusion test
/home/yzk/walberla/apps/showcases/Thermocapillary/                - Thermal + Marangoni
```

---

## Quick Start

### 1. Build the Tests

```bash
cd /home/yzk/LBMProject/build
cmake ..
make test_pure_conduction test_stefan_problem test_grid_convergence test_thermal_lbm test_laser_source
```

### 2. Run Individual Tests

```bash
# Pure conduction (1D diffusion)
./test_pure_conduction

# Stefan problem (phase change)
./test_stefan_problem

# Grid convergence study
./test_grid_convergence

# Thermal LBM unit tests
./test_thermal_lbm

# Laser source tests
./test_laser_source
```

### 3. Run Benchmark Suite

```bash
cd /home/yzk/LBMProject
./scripts/benchmark/run_benchmark_comparison.sh all

# Or run specific tests:
./scripts/benchmark/run_benchmark_comparison.sh pure_conduction
./scripts/benchmark/run_benchmark_comparison.sh stefan
./scripts/benchmark/run_benchmark_comparison.sh grid_convergence
```

### 4. Run Python Validation

```bash
cd /home/yzk/LBMProject
python3 scripts/benchmark/compare_thermal_solutions.py --test all --output benchmark_results/
```

### 5. Include waLBerla Comparison (optional)

```bash
# First, build waLBerla DiffusionTest
cd /home/yzk/walberla/build
make DiffusionTest

# Then run comparison
cd /home/yzk/LBMProject
./scripts/benchmark/run_benchmark_comparison.sh pure_conduction --walberla
```

---

## Validation Test Summary

### Test 1: Pure Conduction (1D Heat Diffusion)

**Purpose:** Validate thermal diffusion accuracy

**Analytical Solution:**
```
T(x,t) = T_amb + (T_peak - T_amb) * sqrt(t0/(t+t0)) * exp(-x^2/(4*alpha*(t+t0)))
```

**Success Criteria:**
- L2 error < 5% at t = 0.1, 0.5, 1.0 ms
- Energy conservation < 0.1%

**Existing test:** `tests/validation/test_pure_conduction.cu`

---

### Test 2: Grid Convergence Study

**Purpose:** Verify 2nd order spatial accuracy

**Method:**
- Run on 3 grid levels: 4um, 2um, 1um
- Compute L2 error vs analytical at fixed time
- Calculate convergence order: p = log2(e_h / e_{h/2})

**Success Criteria:**
- Convergence order >= 1.9 (expect 2.0)
- GCI < 5%

**Existing test:** `tests/validation/test_grid_convergence.cu`

---

### Test 3: Stefan Problem (Phase Change)

**Purpose:** Validate melting front tracking

**Analytical Solution:**
```
Front position: s(t) = 2 * lambda * sqrt(alpha * t)
Stefan number:  St = cp * dT / L_fusion
```

For Ti6Al4V: St = 0.095, lambda = 0.10

**Success Criteria:**
- Front position error < 5%
- Latent heat properly tracked

**Existing test:** `tests/validation/test_stefan_problem.cu`

---

### Test 4: Gaussian Heat Source

**Purpose:** Validate laser-thermal coupling

**Semi-Analytical (steady state):**
```
T_excess ~ P_absorbed / (2 * pi * k * r)
```

**Success Criteria:**
- Peak T within 10% of estimate
- Energy balance within 5%
- Gaussian surface profile

**Existing test:** `tests/unit/laser/test_laser_source.cu`

---

## Comparison Metrics

### Error Metrics
```python
# L2 Norm (global accuracy)
L2 = sqrt(sum((T_num - T_ana)^2) / sum(T_ana^2))

# L_inf Norm (worst case)
Linf = max(abs(T_num - T_ana)) / max(T_ana)

# RMS Error
RMS = sqrt(mean((T_num - T_ana)^2))
```

### Grid Convergence
```python
# Convergence order
p = log(error_h / error_h2) / log(2)

# Grid Convergence Index
GCI = 1.25 * abs(f_h2 - f_h) / (2^p - 1)
```

---

## waLBerla Parameter Mapping

| LBMProject | waLBerla | Notes |
|------------|----------|-------|
| `thermal_diffusivity` | `-d` | m^2/s |
| `dx` | `-dx` | m |
| `dt` | `-dt` | Computed from CFL |
| `tau_T` | `tau = 3*D + 0.5` | D = alpha*dt/dx^2 |

---

## Expected Results

### Pure Conduction
- L2 error: 2-5%
- Energy conservation: < 0.1%

### Grid Convergence
- Order: 1.9 - 2.0
- GCI: < 3%

### Stefan Problem
- Front position error: 3-5%
- Latent heat storage: Working

### waLBerla Comparison
- Temperature profiles within 5%
- LBM-CUDA faster on GPU

---

## Troubleshooting

### Tests not building
```bash
# Ensure CUDA is available
nvcc --version

# Rebuild from clean
cd /home/yzk/LBMProject/build
rm -rf CMakeCache.txt
cmake ..
make -j4
```

### waLBerla build issues
```bash
cd /home/yzk/walberla/build
cmake .. -DWALBERLA_BUILD_TESTS=ON
make DiffusionTest
```

### Python script errors
```bash
# Install dependencies
pip install numpy scipy matplotlib
```

---

## Next Steps

1. Run existing tests and verify they pass
2. Compare with waLBerla on pure diffusion case
3. Document any discrepancies
4. Add new tests as needed for full LPBF validation
