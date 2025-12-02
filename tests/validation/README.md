# LBM-CUDA Validation Framework

**Purpose:** Rigorously validate that BGK high-Peclet instability fixes are treating the **root cause (治本)** not merely **superficial symptoms (治标)**.

**Author:** LBM-CUDA Architecture Team
**Date:** 2025-01-19
**Status:** Ready for Execution

---

## Overview

This validation framework consists of **5 comprehensive test suites** designed to prove:

1. **Numerical Soundness** - Solutions converge to true physics as grid is refined
2. **Robustness** - Stability maintained across all flow regimes (low to high Peclet numbers)
3. **Thermodynamic Consistency** - Energy is conserved within acceptable error bounds
4. **Physical Accuracy** - Results match peer-reviewed literature benchmarks
5. **Efficiency** - Fixes provide computational benefits without sacrificing accuracy

---

## Quick Start

### Run Standard Validation (Recommended)
Runs all tests **except** literature benchmark (~1 hour total):

```bash
cd /home/yzk/LBMProject/tests/validation
./run_all_validation.sh
```

### Run Quick Check (~30 minutes):

```bash
./run_all_validation.sh --quick
```

### Run Full Validation (~3 hours):

```bash
./run_all_validation.sh --full
```

---

## Validation Tests

### Test 1: Grid Convergence Study
**Duration:** ~30-45 minutes | **Critical:** Yes

**Purpose:** Prove solution converges with order ≥ 1.0 as dx → 0

**Success Criteria:**
- ✓ Temperature convergence order ≥ 1.0
- ✓ Velocity convergence order ≥ 1.0
- ✓ Peak temperature decreases as dx → 0

### Test 2: Peclet Number Sweep
**Duration:** ~20-30 minutes | **Critical:** Yes

**Purpose:** Verify stability across different advection-diffusion regimes

**Success Criteria:**
- ✓ All cases remain stable (T_max < 10,000 K)
- ✓ Energy error < 5% for all Pe
- ✓ Higher Pe shows sharper gradients

### Test 3: Energy Conservation
**Duration:** ~15-20 minutes | **Critical:** Yes

**Purpose:** Prove fixes don't violate thermodynamic consistency

**Success Criteria:**
- ✓ Average energy error < 5%
- ✓ Maximum error < 10%
- ✓ No systematic drift

### Test 4: Literature Benchmark
**Duration:** ~2-3 hours | **Critical:** No (Optional)

**Purpose:** Quantitative validation against Mohr et al. (2020)

**Success Criteria:**
- ✓ Peak temperature within ±20% of literature
- ✓ Melt pool geometry within ±30%

### Test 5: Flux Limiter Impact
**Duration:** ~20-30 minutes | **Critical:** No

**Purpose:** Quantify accuracy vs. efficiency trade-off

**Success Criteria:**
- ✓ Temperature difference < 5%
- ✓ Speedup > 2×

---

## Understanding Results

### Exit Codes
- **0**: All tests PASSED → Fixes are treating root cause (治本)
- **1**: Some tests FAILED → Further investigation needed

### Output Location
Results: `/home/yzk/LBMProject/validation_results/`

---

## Analysis

### Jupyter Notebook:
```bash
cd /home/yzk/LBMProject/analysis
jupyter lab validation_analysis.ipynb
```

### ParaView Visualization:
```bash
paraview /home/yzk/LBMProject/validation_results/grid_convergence/fine/*.vtk
```

---

## Success Metrics Summary

| Test | Metric | Target | Critical? |
|------|--------|--------|-----------|
| Grid Convergence | Order ≥ 1.0 | Temperature & Velocity | ✓ |
| Peclet Sweep | All Pe stable | T_max < 10k K | ✓ |
| Energy Conservation | Error < 5% | Average | ✓ |
| Literature Benchmark | ±20% match | Temperature | - |
| Flux Limiter Impact | Accuracy < 5%, Speedup > 2× | Both | - |

---

See full documentation in this README for detailed test descriptions, troubleshooting, and interpretation guidance.
