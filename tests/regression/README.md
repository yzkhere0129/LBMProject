# Regression Test Suite - Quick Reference

This directory contains regression tests that validate critical bug fixes to prevent reintroduction through future code changes.

## Quick Start

### Build All Regression Tests
```bash
cd /home/yzk/LBMProject/build
cmake --build . --target test_vof_advection_bulk_cells
cmake --build . --target test_timestep_omega_clamping
cmake --build . --target test_material_database_values
```

### Run All Regression Tests
```bash
cd /home/yzk/LBMProject/build
ctest -R "Regression" -V
```

### Run Individual Tests
```bash
# VOF advection bulk cell test
./tests/test_vof_advection_bulk_cells

# Timestep omega clamping test
./tests/test_timestep_omega_clamping

# Material database values test
./tests/test_material_database_values
```

## Tests Overview

### 1. VOF Advection Bulk Cells (`test_vof_advection_bulk_cells.cu`)

**Purpose:** Validates that bulk liquid (f=1.0) and bulk gas (f=0.0) cells properly propagate during advection.

**Bug Fixed:** Interface compression kernel returned early for bulk cells without copying advected values (vof_solver.cu:419-421).

**Runtime:** 2-5 minutes

**Outputs:**
- `vof_bulk_advection_mass.csv` - Mass conservation history

**Pass Criteria:**
- Mass conservation error < 5%
- Bulk regions move with expected velocity
- Center of mass shifts by u·t

### 2. Timestep Omega Clamping (`test_timestep_omega_clamping.cu`)

**Purpose:** Validates that timestep convergence study uses stable omega values (omega < 1.90).

**Bug Fixed:** Finest timestep (0.5 μs) caused omega=1.92, breaking convergence analysis (test_timestep_convergence.cu:279-284).

**Runtime:** 30-60 seconds

**Outputs:**
- Console tables showing omega for each timestep

**Pass Criteria:**
- All new timesteps have omega < 1.90
- Old timesteps demonstrate instability
- Timestep ratios equal 2.0

### 3. Material Database Values (`materials/test_material_database_values.cu`)

**Purpose:** Validates that Ti6Al4V material properties match walberla validation values.

**Bug Fixed:** Test expected wrong cp_liquid value; should be 526 J/(kg·K) same as solid (test_materials.cu:48).

**Runtime:** 1-2 minutes

**Outputs:**
- `material_properties_regression.csv` - Full property table

**Pass Criteria:**
- cp_liquid = 526.0 (CRITICAL for walberla match)
- All Ti6Al4V properties match database
- Thermal diffusivity calculation correct

## Critical Failure Modes

### If VOF Test Fails

**Error Message:** "REGRESSION: Bulk liquid did not advect!"

**Likely Cause:** The bug has been reintroduced. Check:
```cpp
// File: src/physics/vof/vof_solver.cu
// Lines: ~419-421
if (f < 0.01f || f > 0.99f) {
    fill_level[idx] = f;  // ← This line must be present!
    return;
}
```

### If Omega Test Fails

**Error Message:** "REGRESSION: New timesteps have instability!"

**Likely Cause:** Timestep values or thermal parameters changed. Check:
```cpp
// File: tests/validation/test_timestep_convergence.cu
// Lines: ~279-284
std::vector<float> timesteps = {
    8.0e-6f,   // Must be these values!
    4.0e-6f,
    2.0e-6f,
    1.0e-6f
};
```

### If Material Test Fails

**Error Message:** "REGRESSION: Liquid cp no longer matches walberla!"

**Likely Cause:** Material database changed. Check:
```cpp
// File: src/physics/materials/material_database.cu
// Ti6Al4V section
mat.cp_solid = 526.0f;
mat.cp_liquid = 526.0f;  // ← Must match solid!
```

## Plotting Results

### VOF Mass Conservation
```python
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('vof_bulk_advection_mass.csv')
plt.plot(df['step'], df['mass'])
plt.axhline(df['mass'][0], color='r', linestyle='--', label='Initial mass')
plt.xlabel('Timestep')
plt.ylabel('Total Mass')
plt.title('VOF Mass Conservation During Advection')
plt.legend()
plt.show()
```

### Material Properties Comparison
```python
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('material_properties_regression.csv')
df.plot(x='Material', y=['rho_solid', 'rho_liquid'], kind='bar')
plt.ylabel('Density (kg/m³)')
plt.title('Material Density Comparison')
plt.show()
```

## When to Run These Tests

**Always run before:**
- Merging to main branch
- Releasing new version
- Making changes to:
  - VOF solver (`src/physics/vof/`)
  - Thermal LBM (`src/physics/thermal/`)
  - Material database (`src/physics/materials/`)

**Run as part of:**
- Pre-commit hooks
- CI/CD pipeline
- Nightly regression suite

## Labels and Organization

Tests are labeled for easy filtering:
- `regression` - All regression tests
- `critical` - Tests that must never fail
- `2026-01-10` - Date bug fix was implemented

**Filter by date:**
```bash
ctest -L "2026-01-10" -V
```

**Filter critical tests:**
```bash
ctest -L "critical" -V
```

## Adding New Regression Tests

1. **Create test file** in this directory
2. **Document bug** in file header:
   ```cpp
   /**
    * @file test_my_regression.cu
    * @brief Regression test for [bug description]
    *
    * BUG FIXED ([file]:[lines]):
    * [Detailed description of what was wrong]
    *
    * FIX:
    * [What was changed to fix it]
    */
   ```
3. **Add to CMakeLists.txt** with labels
4. **Update** this README
5. **Update** `/home/yzk/LBMProject/docs/REGRESSION_TEST_SUITE_2026-01-10.md`

## Contact

For questions about these tests:
- See detailed documentation: `/home/yzk/LBMProject/docs/REGRESSION_TEST_SUITE_2026-01-10.md`
- Check git history for bug fix commits
- Review test source code comments

---

**Last Updated:** 2026-01-10
**Test Suite Version:** 1.0
