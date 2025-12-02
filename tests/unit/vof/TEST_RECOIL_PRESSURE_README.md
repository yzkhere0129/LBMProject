# Recoil Pressure Module Test Suite

## Overview

This test suite validates the recoil pressure computation for evaporating metal surfaces. Recoil pressure is a critical driver of keyhole formation in high-power laser welding and LPBF (Laser Powder Bed Fusion).

## Physical Model

The recoil pressure is computed using the Anisimov model:

```
P_sat(T) = P_ref * exp[(L_vap * M / R) * (1/T_boil - 1/T)]     (Clausius-Clapeyron)
P_recoil = C_r * P_sat                                          (Anisimov coefficient)
F_recoil = P_recoil * (-n) * |grad(f)| / h_interface            (Volumetric force)
```

### Physical Constants (Ti6Al4V)

| Parameter | Value | Description |
|-----------|-------|-------------|
| T_boil | 3533 K | Boiling temperature |
| L_vap | 8.878 MJ/kg | Latent heat of vaporization |
| M_molar | 0.0479 kg/mol | Molar mass |
| C_r | 0.54 | Anisimov/Knight recoil coefficient |
| P_ref | 101325 Pa | Reference pressure (1 atm) |
| T_activation | 3033 K | Temperature activation threshold |

## Test Cases

### Unit Test 1: P_sat Calculation Verification

| Test ID | Description | Expected Result | Pass Criteria |
|---------|-------------|-----------------|---------------|
| PsatCalculationTest.AtBoilingPoint | P_sat at T = 3533 K | P_sat = 101325 Pa | Error < 1% |
| PsatCalculationTest.BelowBoilingPoint_3100K | P_sat at T = 3100 K | P_sat ~ 13400 Pa | 100 < P_sat < P_ref |
| PsatCalculationTest.AboveBoilingPoint_4000K | P_sat at T = 4000 K | P_sat ~ 549 kPa | P_ref < P_sat < 10 MPa |
| PsatCalculationTest.ExponentialTemperatureDependence | Temperature dependence | Monotonic increase | Ratio > 1 for all pairs |

### Unit Test 2: Recoil Pressure Coefficient

| Test ID | Description | Expected Result | Pass Criteria |
|---------|-------------|-----------------|---------------|
| RecoilCoefficientTest.AnisimovCoefficientCorrect | C_r = 0.54 applied correctly | P_recoil = 0.54 * P_sat | Error < 1% for all test values |
| RecoilCoefficientTest.CoefficientGetSet | Getter/setter verification | Coefficient can be changed | Value matches after set |
| RecoilCoefficientTest.RecoilAtBoilingPoint | P_recoil at T_boil | P_recoil = 54715 Pa | Error < 100 Pa |

### Unit Test 3: Force Direction Verification

| Test ID | Description | Expected Result | Pass Criteria |
|---------|-------------|-----------------|---------------|
| ForceDirectionTest.HorizontalInterfaceForceDownward | Force on horizontal surface | Force points -z (into liquid) | >50% cells with F_z < 0 |
| ForceDirectionTest.ForceAlongNormal | Force on spherical droplet | Force points toward center | >70% direction accuracy |

### Unit Test 4: Boundary Conditions

| Test ID | Description | Expected Result | Pass Criteria |
|---------|-------------|-----------------|---------------|
| BoundaryConditionTest.ZeroPsatBelowThreshold | T < 3033 K | P_sat = 0 | P_sat < 1 Pa |
| BoundaryConditionTest.ZeroForceInGasRegion | fill_level = 0 | Force = 0 | Max force < 1 N/m3 |
| BoundaryConditionTest.ZeroForceInLiquidRegion | fill_level = 1 | Force = 0 | Max force < 1 N/m3 |
| BoundaryConditionTest.NoNaNOrInf | Random temperatures | No numerical issues | NaN count = 0, Inf count = 0 |
| BoundaryConditionTest.PressureLimiter | P_sat = 5 MPa, limit = 1 MPa | P_recoil capped | P_recoil <= 1 MPa |

### Unit Test 5: Physical Consistency

| Test ID | Description | Expected Result | Pass Criteria |
|---------|-------------|-----------------|---------------|
| PhysicalConsistencyTest.KeyholeRegimePressure | T = 4500 K (keyhole regime) | P_recoil > 1 MPa | 1 MPa < P_recoil < 100 MPa |
| PhysicalConsistencyTest.ForceMagnitudeReasonable | Force order of magnitude | Consistent with formula | Ratio within 0.1 - 10 |

### Integration Test

| Test ID | Description | Expected Result | Pass Criteria |
|---------|-------------|-----------------|---------------|
| IntegrationTest.SurfaceDepressionSetup | Gaussian hot spot on surface | Downward force at center | >10 cells with F_z < 0, total_fz < 0 |

## Running the Tests

```bash
# Build the test
cd /home/yzk/LBMProject/build
make test_recoil_pressure

# Run all tests
./tests/test_recoil_pressure

# Run specific test suite
./tests/test_recoil_pressure --gtest_filter=PsatCalculationTest.*

# Run with verbose output
./tests/test_recoil_pressure --gtest_print_time=1
```

## Expected Test Output Summary

```
[==========] Running 17 tests from 6 test suites.
[  PASSED  ] 17 tests.
```

## Reference Values for Validation

### P_sat vs Temperature (Ti6Al4V)

| Temperature [K] | P_sat [Pa] | Description |
|-----------------|------------|-------------|
| 3000 | ~0 | Below activation threshold |
| 3100 | ~13,400 | Below boiling point |
| 3200 | ~22,500 | Below boiling point |
| 3400 | ~57,500 | Approaching boiling |
| 3533 | 101,325 | Boiling point (1 atm) |
| 3600 | ~132,700 | Above boiling |
| 4000 | ~549,000 | High temperature |
| 4500 | ~2,274,000 | Keyhole regime |

### P_recoil = 0.54 * P_sat

| P_sat [Pa] | P_recoil [Pa] |
|------------|---------------|
| 101,325 (at T_boil) | 54,716 |
| 500,000 | 270,000 |
| 1,000,000 | 540,000 |
| 2,274,264 (at 4500K) | 1,228,103 |

## File Locations

- Test source: `/home/yzk/LBMProject/tests/unit/vof/test_recoil_pressure.cu`
- Recoil pressure header: `/home/yzk/LBMProject/include/physics/recoil_pressure.h`
- CMake configuration: `/home/yzk/LBMProject/tests/CMakeLists.txt`

## Author

Test suite created for validating recoil pressure physics in LPBF simulations.
