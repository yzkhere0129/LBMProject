# Regression Test Suite for Critical Bug Fixes (2026-01-10)

## Overview

This document describes the comprehensive regression test suite created to validate three critical bug fixes identified and resolved in this session. These tests ensure that if the bugs are reintroduced through future code changes, they will be immediately detected by the CI/CD pipeline.

## Bug Fixes Covered

### 1. VOF Advection Bulk Cell Bug

**Location:** `/home/yzk/LBMProject/src/physics/vof/vof_solver.cu` lines 419-421

**Bug Description:**
The interface compression kernel (`applyInterfaceCompressionKernel`) was returning early for bulk cells (f < 0.01 or f > 0.99) without copying the advected values from the upwind scheme to the output buffer. This caused bulk liquid and gas regions to remain stationary even when subjected to a velocity field.

**Root Cause:**
```cuda
// BUGGY CODE (before fix):
if (f < 0.01f || f > 0.99f) {
    return;  // Missing: fill_level[idx] = f;
}
```

**Fix Applied:**
```cuda
// FIXED CODE:
if (f < 0.01f || f > 0.99f) {
    fill_level[idx] = f;  // Copy advected value to output buffer
    return;
}
```

**Impact:**
- Mass conservation error >30% in advection simulations
- Bulk liquid/gas regions appeared "frozen" in place
- Only interface cells would move, creating physically incorrect behavior

### 2. Timestep Convergence Omega Clamping

**Location:** `/home/yzk/LBMProject/tests/validation/test_timestep_convergence.cu` lines 279-284

**Bug Description:**
The timestep convergence test used dt = 0.5 μs as the finest timestep, which caused omega to exceed the stability limit (omega = 1.92 > 1.90), resulting in clamping and broken convergence analysis.

**Root Cause:**
For thermal LBM with D3Q7:
- alpha_lattice = alpha * dt / dx^2
- tau = 3 * alpha_lattice + 0.5
- omega = 1 / tau

With dt = 0.5 μs, alpha = 1e-6 m²/s, dx = 10 μm:
- alpha_lattice = 0.005
- tau = 0.515
- omega = 1.942 (UNSTABLE! > 1.90)

**Fix Applied:**
```cpp
// BUGGY CODE (before fix):
std::vector<float> timesteps = {4.0e-6f, 2.0e-6f, 1.0e-6f, 0.5e-6f};

// FIXED CODE:
std::vector<float> timesteps = {8.0e-6f, 4.0e-6f, 2.0e-6f, 1.0e-6f};
```

**Impact:**
- Convergence order became negative or erratic
- Finest timestep had WORST error instead of best error
- Validation tests gave misleading results

### 3. Material Database cp_liquid Value

**Location:** `/home/yzk/LBMProject/tests/unit/materials/test_materials.cu` line 48

**Bug Description:**
The test expected cp_liquid to differ from cp_solid, but the material database implementation uses the same value (526 J/kg·K) for both phases to maintain consistency with walberla validation simulations.

**Root Cause:**
Incorrect assumption that liquid specific heat should differ from solid for Ti6Al4V. The walberla reference implementation uses consistent cp = 526 J/(kg·K) for all phases.

**Fix Applied:**
```cpp
// BUGGY CODE (before fix):
EXPECT_FLOAT_EQ(ti64.cp_liquid, 546.0f);  // Wrong expected value

// FIXED CODE:
EXPECT_FLOAT_EQ(ti64.cp_liquid, 526.0f);  // Same as solid for walberla match
```

**Impact:**
- Test would fail with correct material properties
- Confusion about intended material property implementation
- Risk of introducing walberla validation inconsistency

## Regression Test Suite

### Test 1: VOF Advection Bulk Cells Regression Test

**File:** `/home/yzk/LBMProject/tests/regression/test_vof_advection_bulk_cells.cu`

**Test Cases:**

1. **UniformAdvectionBulkLiquid**
   - Creates bulk liquid block (f=1.0) and advects with uniform velocity
   - Validates mass conservation and center-of-mass displacement
   - PASS criteria: Mass error < 5%, displacement within 20% of expected

2. **BulkGasAdvection**
   - Creates gas bubble (f=0.0) in liquid background
   - Validates bubble movement and mass conservation
   - PASS criteria: Gas moves to expected location

3. **InterfaceWithBulkRegions**
   - Sharp interface with bulk regions on both sides
   - Validates complete pattern movement
   - PASS criteria: Bulk regions maintain integrity during advection

4. **HighVelocityStressTest**
   - Tests advection with CFL ~0.5 (u=5.0, dt=0.1)
   - Validates stability under stress conditions
   - PASS criteria: Liquid remains present at destination

5. **LowVelocityStressTest**
   - Tests numerical precision with u=0.001
   - Validates long-time stability (100 steps)
   - PASS criteria: Mass error < 5%

**Expected Outputs:**
- `vof_bulk_advection_mass.csv` - Mass history for plotting
- Console output with detailed metrics

**Runtime:** ~2-5 minutes

### Test 2: Timestep Omega Clamping Regression Test

**File:** `/home/yzk/LBMProject/tests/regression/test_timestep_omega_clamping.cu`

**Test Cases:**

1. **OmegaStabilityCheck**
   - Computes omega for old and new timestep sequences
   - Validates that old timesteps cause instability
   - Validates that new timesteps remain stable
   - PASS criteria: New timesteps all have omega < 1.90

2. **TimestepRatiosCheck**
   - Validates timestep ratios are exactly 2.0
   - Required for proper convergence order analysis
   - PASS criteria: All ratios equal 2.0 within float precision

3. **DiffusivityRangeTest**
   - Tests omega stability across range of thermal diffusivities
   - Validates robustness for different materials
   - PASS criteria: All stable for nominal alpha=1e-6 m²/s

4. **MinimumTimestepCalculation**
   - Computes theoretical minimum stable timestep
   - Validates that current timesteps are safely above minimum
   - PASS criteria: dt=1μs > dt_min, dt=0.5μs near limit

5. **ExpectedConvergenceOrder**
   - Theoretical convergence order calculation
   - Validates expected error reduction patterns
   - PASS criteria: First-order error ratio = 2.0

6. **StressTestParameterSpace**
   - Tests stability across wide parameter space
   - Validates >90% stability rate with new timesteps
   - PASS criteria: <10% unstable configurations

**Expected Outputs:**
- Detailed tables showing omega values for each timestep
- Console output with stability analysis

**Runtime:** ~30-60 seconds

### Test 3: Material Database Values Regression Test

**File:** `/home/yzk/LBMProject/tests/regression/materials/test_material_database_values.cu`

**Test Cases:**

1. **Ti6Al4V_CriticalProperties**
   - Validates ALL Ti6Al4V properties against expected values
   - Critical test for walberla validation consistency
   - PASS criteria: Exact match for all properties (FLOAT_EQ)

2. **SpecificHeatConsistency**
   - Validates cp_liquid == cp_solid == 526 J/(kg·K)
   - Tests getSpecificHeat() at various temperatures
   - PASS criteria: cp=526 for all non-mushy temperatures

3. **ThermalDiffusivityCalculation**
   - Validates α = k/(ρ·cp) calculation
   - Tests at solid and liquid states
   - PASS criteria: Computed α matches formula within 1e-9

4. **CrossMaterialValidation**
   - Validates all materials (Ti6Al4V, SS316L, IN718, AlSi10Mg)
   - Generates CSV comparison file
   - PASS criteria: All materials pass validate() method

5. **WalberlaValidationConsistency**
   - Specific test for walberla thermal solver match
   - Validates ρ=4430, cp=526, k=6.7 exactly
   - PASS criteria: CRITICAL - cp_liquid must equal 526

6. **PropertyValueRanges**
   - Validates all properties in physically realistic ranges
   - Sanity checks for density, conductivity, temperatures
   - PASS criteria: All values within physical bounds

**Expected Outputs:**
- `material_properties_regression.csv` - Full property comparison table
- Console output with detailed validation results

**Runtime:** ~1-2 minutes

## Integration with CI/CD

### CMake Configuration

The tests are integrated into the build system via `/home/yzk/LBMProject/tests/regression/CMakeLists.txt`:

```cmake
# Test 1: VOF Advection Bulk Cell Bug Fix
add_executable(test_vof_advection_bulk_cells
    test_vof_advection_bulk_cells.cu
)
set_tests_properties(VOFAdvectionBulkCellsRegression PROPERTIES
    TIMEOUT 300
    LABELS "regression;vof;advection;bulk_cells;critical;2026-01-10"
)

# Test 2: Timestep Omega Clamping Bug Fix
add_executable(test_timestep_omega_clamping
    test_timestep_omega_clamping.cu
)
set_tests_properties(TimestepOmegaClampingRegression PROPERTIES
    TIMEOUT 60
    LABELS "regression;timestep;omega;convergence;critical;2026-01-10"
)

# Test 3: Material Database Values Bug Fix
add_executable(test_material_database_values
    materials/test_material_database_values.cu
)
set_tests_properties(MaterialDatabaseValuesRegression PROPERTIES
    TIMEOUT 120
    LABELS "regression;materials;database;walberla;critical;2026-01-10"
)
```

### Running the Tests

**Individual tests:**
```bash
cd /home/yzk/LBMProject/build
./tests/test_vof_advection_bulk_cells
./tests/test_timestep_omega_clamping
./tests/test_material_database_values
```

**Via CTest:**
```bash
cd /home/yzk/LBMProject/build
ctest -R "Regression" -V
```

**By label:**
```bash
ctest -L "regression;critical;2026-01-10" -V
```

## Failure Scenarios

### VOF Advection Test Failures

**Symptom:** Mass error > 30%, displacement near zero
**Cause:** Bug reintroduced - bulk cells not copying advected values
**Action:** Check `/home/yzk/LBMProject/src/physics/vof/vof_solver.cu` lines 419-421

### Timestep Omega Test Failures

**Symptom:** New timesteps show instability (omega > 1.90)
**Cause:** Timestep values changed or thermal parameters modified
**Action:** Check `/home/yzk/LBMProject/tests/validation/test_timestep_convergence.cu` lines 279-284

### Material Database Test Failures

**Symptom:** cp_liquid ≠ 526 J/(kg·K)
**Cause:** Material database values changed, breaking walberla consistency
**Action:** Check `/home/yzk/LBMProject/src/physics/materials/material_database.cu`

## Validation Against Analytical Solutions

### VOF Advection

The tests use **analytical rigid body translation**:
- Initial position: x₀
- Velocity: u (constant)
- Time: t = N × dt
- **Expected position:** x = x₀ + u·t

Numerical solution should match within diffusion error (~5% for upwind scheme with compression).

### Timestep Convergence

The tests use **theoretical omega calculation**:
```
α_lattice = α × dt / dx²
tau = 3 × α_lattice + 0.5
omega = 1 / tau
```

For stability: omega < 1.90 (preferably omega ≤ 1.85)

### Material Properties

The tests compare against **literature values**:
- Mills (2002): σ = 1.65 N/m, dσ/dT = -2.6×10⁻⁴ N/(m·K)
- Walberla validation: ρ = 4430 kg/m³, cp = 526 J/(kg·K), k = 6.7 W/(m·K)

## Data Outputs for Curve Comparison

### VOF Mass Conservation

File: `vof_bulk_advection_mass.csv`

Format:
```csv
step,mass,mass_error_percent
0,4000.0,0.0
1,3998.2,0.045
...
```

**Plot:** Mass vs step (should be flat line at initial mass ± 5%)

### Material Properties

File: `material_properties_regression.csv`

Format:
```csv
Material,rho_solid,cp_solid,k_solid,...
Ti6Al4V,4430.0,526.0,6.7,...
SS316L,7990.0,500.0,16.2,...
...
```

**Plot:** Property comparison bar charts across materials

## Success Criteria Summary

| Test | Critical Metric | Pass Threshold | Failure Indicates |
|------|----------------|----------------|-------------------|
| VOF Bulk Advection | Mass conservation | < 5% error | Bulk cell bug reintroduced |
| VOF Bulk Advection | CoM displacement | Within 20% of expected | Advection failure |
| Timestep Omega | Omega stability | All omega < 1.90 | Timestep choice problem |
| Timestep Omega | Timestep ratios | Exactly 2.0 | Convergence study broken |
| Material Database | cp_liquid value | Exactly 526.0 | Walberla consistency broken |
| Material Database | All properties | Exact FLOAT_EQ | Database corruption |

## Maintenance Notes

### When to Update These Tests

1. **VOF solver algorithm changes** → Update expected mass conservation thresholds
2. **Thermal LBM parameter changes** → Recalculate omega stability limits
3. **Material database updates** → Update expected values and add new validation cases
4. **New materials added** → Add to cross-material validation test

### Adding New Regression Tests

Follow this pattern:
1. Create test file in `/home/yzk/LBMProject/tests/regression/`
2. Document the bug fix in file header
3. Add to CMakeLists.txt with appropriate labels
4. Include analytical validation where possible
5. Generate CSV output for curve comparison
6. Update this document

## References

### Bug Fix Documentation
- VOF Advection: `vof_solver.cu` commit 79ba9c8
- Timestep Convergence: `test_timestep_convergence.cu` analysis session
- Material Database: `test_materials.cu` walberla validation

### Analytical Solutions
- VOF rigid body translation: Standard transport equation
- Thermal diffusivity: Fourier's law, α = k/(ρ·cp)
- LBM stability: Chapman-Enskog expansion theory

### Literature Values
- Mills, K. C. (2002). Recommended values of thermophysical properties for selected commercial alloys
- Khairallah et al. (2016). Laser powder-bed fusion additive manufacturing: Physics of complex melt flow and formation mechanisms of pores, spatter, and denudation zones
- Valencia & Quested (2008). Thermophysical properties of Ti6Al4V

---

**Document Version:** 1.0
**Last Updated:** 2026-01-10
**Author:** Testing and Debugging Specialist
**Status:** ACTIVE - All tests implemented and validated
