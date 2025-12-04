# LBM Multiphysics Test Suite - Quick Reference

## Quick Commands

### Run Everything
```bash
cd /home/yzk/LBMProject
./RUN_ALL_PHYSICS_TESTS.sh
```

### Pre-Commit (< 30s)
```bash
cd /home/yzk/LBMProject/build
ctest -R "flux_limiter|temperature_bounds|omega_reduction" --output-on-failure
```

### Pre-Push (< 2min)
```bash
ctest -R "high_pe_stability|test_vof_mass_conservation" --output-on-failure
```

---

## Test Categories at a Glance

| Category | Test Pattern | Count | Runtime | Critical |
|----------|-------------|-------|---------|----------|
| VOF Solver | `test_vof` | 20+ | 5-60s | Yes |
| Marangoni | `test_marangoni\|test_interface_geometry` | 10+ | 5-300s | Yes |
| Thermal | `test_thermal\|test_lattice_d3q7` | 15+ | 5-600s | Yes |
| Fluid LBM | `test_fluid\|test_poiseuille` | 12+ | 5-180s | Yes |
| Phase Change | `test_liquid_fraction\|test_enthalpy\|test_stefan` | 8+ | 5-300s | Yes |
| Multiphysics | `test_.*_coupling\|test_mp` | 25+ | 60-300s | Yes |
| Energy | `test_energy\|test_evaporation_energy` | 8+ | 30-180s | Yes |
| Stability | `test_.*stability\|test_cfl\|test_omega` | 10+ | 5-300s | **CRITICAL** |

---

## Critical Tests (MUST PASS)

### Stability Tests
```bash
# These prevent simulation divergence
ctest -R "flux_limiter" --output-on-failure           # Prevents negative populations
ctest -R "temperature_bounds" --output-on-failure      # Prevents NaN from radiation BC
ctest -R "omega_reduction" --output-on-failure         # Prevents omega > 1.50
ctest -R "high_pe_stability" --output-on-failure       # End-to-end stability at Pe~10
```

**Why Critical**: Without these, simulations diverge at step ~50-200

### Physics Correctness
```bash
ctest -R "divergence_free" --output-on-failure        # Incompressibility for VOF
ctest -R "test_energy_conservation" --output-on-failure  # Energy accounting
ctest -R "test_vof_mass_conservation" --output-on-failure # VOF mass conservation
```

---

## Test by Physics Module

### 1. VOF Solver
```bash
# Basic tests (fast)
ctest -R "test_vof_advection_uniform" --output-on-failure
ctest -R "test_vof_reconstruction" --output-on-failure
ctest -R "test_vof_curvature" --output-on-failure

# Analytical validation (medium)
ctest -R "test_vof_advection_rotation" --output-on-failure  # Zalesak's disk
ctest -R "test_vof_curvature_sphere" --output-on-failure    # Sphere curvature
ctest -R "test_vof_curvature_cylinder" --output-on-failure  # Cylinder curvature

# All VOF tests
ctest -R "vof" --output-on-failure
```

**Key Metrics**:
- Displacement error < 0.5 cells
- Mass error < 1-2%
- Curvature error < 10-30% (resolution dependent)

### 2. Marangoni Effects
```bash
# Unit tests (fast)
ctest -R "test_marangoni_force_magnitude" --output-on-failure
ctest -R "test_marangoni_force_direction" --output-on-failure
ctest -R "test_marangoni_gradient_limiter" --output-on-failure

# Integration tests (slow)
ctest -R "test_marangoni_flow" --output-on-failure
ctest -R "test_marangoni_velocity" --output-on-failure      # Literature validation

# All Marangoni tests
ctest -R "marangoni" --output-on-failure
```

**Key Metrics**:
- Force magnitude ~10^8-10^9 N/m³
- Velocity 0.5-2 m/s (Ti6Al4V)
- Force tangent to interface

### 3. Thermal Solver
```bash
# Core tests (fast)
ctest -R "test_thermal_lbm" --output-on-failure
ctest -R "test_lattice_d3q7" --output-on-failure

# Stability tests (CRITICAL, fast)
ctest -R "test_flux_limiter" --output-on-failure
ctest -R "test_temperature_bounds" --output-on-failure
ctest -R "test_omega_reduction" --output-on-failure

# Validation tests (slow)
ctest -R "test_pure_conduction" --output-on-failure         # Diffusion equation
ctest -R "test_stefan_problem" --output-on-failure          # Phase change
ctest -R "test_high_pe_stability" --output-on-failure       # Pe~10 stability

# All thermal tests
ctest -R "thermal" --output-on-failure
```

**Key Metrics**:
- All g_eq ≥ 0 (no negative populations)
- T ∈ [0, 7000] K always
- omega_T ≤ 1.50
- L2 error < 5% vs analytical

### 4. Fluid LBM
```bash
# Core tests (fast)
ctest -R "test_fluid_lbm" --output-on-failure
ctest -R "test_d3q19" --output-on-failure
ctest -R "test_bgk" --output-on-failure
ctest -R "test_streaming" --output-on-failure

# Validation tests (medium)
ctest -R "test_poiseuille" --output-on-failure
ctest -R "test_divergence_free" --output-on-failure         # CRITICAL

# All fluid tests
ctest -R "fluid|poiseuille" --output-on-failure
```

**Key Metrics**:
- |div(u)| < 1e-3 (incompressibility)
- Velocity profile error < 5%
- Mass conservation < 0.1%

### 5. Phase Change
```bash
# Unit tests (fast)
ctest -R "test_liquid_fraction" --output-on-failure
ctest -R "test_enthalpy" --output-on-failure
ctest -R "test_phase_properties" --output-on-failure

# Validation tests (slow)
ctest -R "test_stefan" --output-on-failure
ctest -R "test_phase_change_robustness" --output-on-failure

# All phase change tests
ctest -R "phase_change|liquid_fraction|enthalpy|stefan" --output-on-failure
```

**Key Metrics**:
- Correct phase identification
- Temperature recovery error < 1 K
- Interface position error < 10%

### 6. Multiphysics Coupling
```bash
# Coupling tests (medium-slow)
ctest -R "test_thermal_fluid_coupling" --output-on-failure
ctest -R "test_vof_fluid_coupling" --output-on-failure
ctest -R "test_thermal_vof_coupling" --output-on-failure
ctest -R "test_phase_fluid_coupling" --output-on-failure

# Force balance tests (fast)
ctest -R "test_force_balance" --output-on-failure
ctest -R "test_force_magnitude_ordering" --output-on-failure

# All multiphysics tests
ctest -R "multiphysics|coupling|test_mp" --output-on-failure
```

**Key Metrics**:
- Bidirectional coupling functional
- Force directions correct
- Net force < 1% at equilibrium

### 7. Energy Conservation
```bash
# Energy tests (medium)
ctest -R "test_energy_conservation_no_source" --output-on-failure
ctest -R "test_energy_conservation_laser_only" --output-on-failure
ctest -R "test_energy_conservation_full" --output-on-failure
ctest -R "test_evaporation_energy_balance" --output-on-failure

# All energy tests
ctest -R "energy" --output-on-failure
```

**Key Metrics**:
- No source: energy drift < 1% per 1000 steps
- With source: balance error < 5-10%

---

## Test by Speed

### Fast Tests (< 30s) - Daily Development
```bash
ctest -R "flux_limiter|temperature_bounds|omega_reduction|test_d3q19|test_bgk|test_streaming|test_lattice_d3q7|test_liquid_fraction|test_enthalpy|test_laser_source|test_unit_converter" --output-on-failure
```

### Medium Tests (30s - 5min) - Before Push
```bash
ctest -R "test_vof_advection_uniform|test_vof_curvature|test_marangoni_force|test_thermal_lbm|test_poiseuille|test_high_pe_stability|test_thermal_fluid_coupling|test_energy_conservation" --output-on-failure
```

### Slow Tests (> 5min) - Nightly/Release
```bash
ctest -R "test_pure_conduction|test_stefan_problem|test_grid_convergence|test_timestep_convergence|test_regression_50W|test_marangoni_velocity|test_vof_advection_rotation" --output-on-failure
```

---

## Common Workflows

### After Making Code Changes
```bash
cd /home/yzk/LBMProject/build
cmake --build . -j8                    # Rebuild
ctest -R "flux_limiter|temperature_bounds" --output-on-failure  # Quick check
```

### Before Committing
```bash
ctest -R "flux_limiter|temperature_bounds|omega_reduction" --output-on-failure
# If pass: git commit
```

### Before Pushing
```bash
ctest -R "stability|high_pe|divergence_free|vof_mass_conservation" --output-on-failure
# If pass: git push
```

### Full Validation
```bash
cd /home/yzk/LBMProject
./RUN_ALL_PHYSICS_TESTS.sh
# Check report in test_reports/
```

### Debug Specific Module
```bash
# Rebuild specific test
cmake --build . --target test_vof_advection_uniform -j4

# Run with verbose output
ctest -R test_vof_advection_uniform --output-on-failure --verbose

# Run with gtest filters
./tests/unit/vof/test_vof_advection_uniform --gtest_filter="*InterfaceDisplacement*"
```

---

## Failure Diagnosis

### Symptom: NaN or Inf in thermal solver
**Check**:
```bash
ctest -R "flux_limiter|temperature_bounds" --output-on-failure
```
**Action**: Verify flux limiter and temperature bounds are active

### Symptom: Simulation diverges at step ~50-200
**Check**:
```bash
ctest -R "omega_reduction" --output-on-failure
```
**Action**: Verify omega_T ≤ 1.50

### Symptom: VOF mass not conserved
**Check**:
```bash
ctest -R "divergence_free|vof_mass_conservation" --output-on-failure
```
**Action**: Verify incompressibility and advection scheme

### Symptom: Energy balance error > 10%
**Check**:
```bash
ctest -R "test_energy_conservation" --output-on-failure
```
**Action**: Check individual energy terms, verify timestep scaling

### Symptom: Excessive numerical diffusion
**Check**:
```bash
ctest -R "test_vof_advection_uniform|test_vof_advection_shear" --output-on-failure
```
**Action**: Monitor interface thickness, adjust CFL/subcycling

---

## Pass Criteria Summary

| Test Category | Key Metric | Pass Criteria |
|---------------|-----------|---------------|
| VOF Advection | Displacement error | < 0.5 cells |
| VOF Mass | Mass error | < 1-2% |
| VOF Curvature | Error vs analytical | < 10-30% (resolution dependent) |
| Thermal Flux | All g_eq | ≥ 0 (no negative) |
| Thermal Bounds | Temperature | ∈ [0, 7000] K |
| Thermal Omega | omega_T | ≤ 1.50 |
| Thermal Diffusion | L2 error | < 5% |
| Fluid Divergence | \|div(u)\| | < 1e-3 |
| Fluid Poiseuille | Velocity profile error | < 5% |
| Phase Change | Temperature recovery | < 1 K error |
| Marangoni Force | Magnitude | 10^8-10^9 N/m³ |
| Marangoni Velocity | Velocity | 0.5-2 m/s (Ti6Al4V) |
| Energy No Source | Drift | < 1% per 1000 steps |
| Energy With Source | Balance error | < 5-10% |

---

## Build Commands

### Full Build
```bash
cd /home/yzk/LBMProject/build
cmake ..
cmake --build . -j8
```

### Rebuild Tests Only
```bash
cmake --build . --target test_vof_advection_uniform -j4
cmake --build . --target test_thermal_lbm -j4
# ... etc
```

### Clean Build
```bash
rm -rf /home/yzk/LBMProject/build/*
cd /home/yzk/LBMProject/build
cmake ..
cmake --build . -j8
```

---

## Test Output Interpretation

### Good Output
```
[==========] Running 3 tests from 1 test suite.
[----------] 3 tests from VOFAdvectionTest
[ RUN      ] VOFAdvectionTest.InterfaceDisplacement
[       OK ] VOFAdvectionTest.InterfaceDisplacement (237 ms)
[ RUN      ] VOFAdvectionTest.MassConservation
[       OK ] VOFAdvectionTest.MassConservation (412 ms)
[ RUN      ] VOFAdvectionTest.InterfaceShapePreservation
[       OK ] VOFAdvectionTest.InterfaceShapePreservation (523 ms)
[----------] 3 tests from VOFAdvectionTest (1172 ms total)
[==========] 3 tests from 1 test suite ran. (1172 ms total)
[  PASSED  ] 3 tests.
```

### Bad Output
```
[ RUN      ] FluxLimiterTest.PreventNegativePopulations
/path/test_flux_limiter.cu:45: Failure
REGRESSION: Negative population detected at q=2 (ux=5, T=1000)
g_eq[2] = -0.0234 (MUST BE >= 0)
[  FAILED  ] FluxLimiterTest.PreventNegativePopulations (8 ms)
```

**Action**: DO NOT MERGE. Flux limiter broken or disabled.

---

## File Locations

### Test Source Files
```
/home/yzk/LBMProject/tests/
├── unit/                    # Fast unit tests
├── integration/             # Medium integration tests
├── validation/              # Slow validation tests
├── debug/                   # Debug utilities
├── diagnostic/              # Diagnostic tools
├── performance/             # Performance tests
└── regression/              # Regression tests
```

### Test Documentation
```
/home/yzk/LBMProject/docs/
├── TEST_GUIDE.md            # Comprehensive guide (this file)
├── TEST_QUICK_REFERENCE.md  # Quick reference
```

### Test Reports
```
/home/yzk/LBMProject/test_reports/
├── test_summary_YYYYMMDD_HHMMSS.txt
└── failed_tests_YYYYMMDD_HHMMSS.txt
```

### Test Runner
```
/home/yzk/LBMProject/RUN_ALL_PHYSICS_TESTS.sh
```

---

## Contact

For test issues:
1. Check `/home/yzk/LBMProject/docs/TEST_GUIDE.md` for detailed documentation
2. Review individual test file headers for specific physics
3. Check existing documentation in `/home/yzk/LBMProject/tests/*.md`

---

**Quick Reference Version 1.0**
**Last Updated**: 2025-12-02
