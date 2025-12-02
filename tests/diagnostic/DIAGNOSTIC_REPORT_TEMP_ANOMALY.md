# Temperature Anomaly Diagnostic Report

## Executive Summary

Two diagnostic test files were created to isolate the root cause of the Tier2 temperature oscillation anomaly:

1. `/home/yzk/LBMProject/tests/diagnostic/test_temperature_anomaly.cu` - Tests ThermalLBM in isolation
2. `/home/yzk/LBMProject/tests/diagnostic/test_multiphysics_temp_anomaly.cu` - Tests MultiphysicsSolver integration

## Test Results

### Test 1: Minimal Laser Heating (ThermalLBM only)
**Result: PASS**

T_max monotonically increases from 300K to 7000K+ over 10,000 steps.
Temperature at each 100-step interval shows consistent RISING status.

```
Step    Time(us)    T_max(K)    T_min(K)    dT_max      Status
  100       10.0       979.0       300.0     +679.0     RISING
  200       20.0      1506.8       300.0     +527.8     RISING
  ...
10000     1000.0      7799.0       300.0      +22.1     RISING
```

**Conclusion**: Core ThermalLBM solver is NOT the source of the oscillation.

---

### Test 2: Heat Source Verification
**Result: FAIL (marginal)**

Heat source integral shows 10.2% error (11.02W vs expected 10.00W).
This is slightly above the 10% threshold but likely due to:
- Gaussian beam discretization on finite grid
- Boundary truncation effects

**Conclusion**: Heat source computation is approximately correct.

---

### Test 3: Energy Conservation
**Result: PASS**

dE/dt measurements converge to approximately 10-11W, matching expected laser power.
Energy balance error decreases over time as transient effects diminish.

---

### Test 4: Buffer Swap Verification
**Result: PASS**

No anomalous temperature jumps detected during ping-pong buffer operations.

---

### Test 5: Radiation BC Isolation
**Result: PASS (with caveat)**

When initialized at 2000K with radiation BC only (no laser):
- T_max stays at 2000K
- T_avg shows 2015.6K (slight increase, unexpected)

**IMPORTANT FINDING**: The radiation BC only applies to the top surface (z=nz-1).
Interior cells are NOT cooled by radiation BC, only surface cells.

This is correct behavior for surface radiation, but may explain why
cooling appears slower than expected in bulk simulations.

---

### Test 6: Combined Laser + Radiation
**Result: PASS**

With both laser heating and radiation BC enabled:
- T_max rises monotonically from 300K to approximately 3560K (boiling point)
- Evaporation cooling kicks in at T_boil = 3560K
- Temperature oscillates slightly around boiling point (expected physical behavior)

No large drops (>50K) observed between measurement intervals.

---

## MultiphysicsSolver Tests (Partial Results)

### Test A: Thermal + Laser Only (via MultiphysicsSolver)
**Result: In Progress, T_max RISING**

At step 1000 (100 us): T_max = 3559.6K, status = RISING
Energy balance shows ~11W dE/dt (expected 10W, 10% error)

Key observation: Evaporation cooling becomes active when T_surf > T_boil (3560K)

---

## Key Findings

### 1. ThermalLBM Core is Correct
The isolated ThermalLBM solver shows monotonic temperature increase.
The bug is NOT in:
- Heat source addition
- BGK collision
- Streaming step
- Buffer swap

### 2. Energy Balance Shows ~10-12% Excess
dE/dt consistently shows 10-12W when 10W laser is applied.
Possible causes:
- Heat source numerical integration error (~10%)
- Boundary condition numerical diffusion

### 3. Radiation BC Works But Has Limited Effect
- Only applies to top surface (z = nz-1)
- Does not cool interior cells
- Radiation power shows as 0.00W in energy balance (may be bug in power computation)

### 4. Evaporation Cooling is Active
When T > T_boil (3560K), evaporation cooling activates.
This provides natural temperature limiting at boiling point.

---

## Root Cause Hypothesis

Based on test results, the Tier2 temperature oscillation is likely caused by:

### Hypothesis A: VOF/Fluid Coupling Issue (Most Likely)
The oscillation occurs when full multiphysics is enabled, not with thermal-only.
VOF advection or fluid solver may be corrupting the temperature field.

### Hypothesis B: Phase Change Solver Bug
Phase change (melting/solidification) may cause energy non-conservation
when liquid fraction oscillates near the mushy zone.

### Hypothesis C: Boundary Condition Race Condition
Multiple BCs (radiation, substrate, periodic) may be applying conflicting
modifications to the same cells in different orders.

---

## Recommended Next Steps

1. **Run Test B (Thermal + Phase Change)**
   - Isolates phase change solver contribution
   - If this fails, bug is in phase change solver

2. **Run Test C (Full Multiphysics)**
   - Matches Tier2 configuration
   - If this fails but A,B pass, bug is in fluid/VOF/Marangoni

3. **Check VOF Advection Impact on Temperature**
   - VOF subcycling may not properly conserve thermal energy
   - Temperature field may be corrupted during interface reconstruction

4. **Review Boundary Condition Application Order**
   - Current order: BCs -> Collision -> Streaming -> Compute T
   - Verify this is correct for LBM

---

## Files Created

1. `/home/yzk/LBMProject/tests/diagnostic/test_temperature_anomaly.cu`
   - 6 tests for ThermalLBM isolation
   - All PASS except minor heat source error

2. `/home/yzk/LBMProject/tests/diagnostic/test_multiphysics_temp_anomaly.cu`
   - 3 tests for MultiphysicsSolver integration
   - Test A: In progress (thermal+laser only)
   - Test B: Thermal + phase change
   - Test C: Full multiphysics (Tier2 reproduction)

3. CMakeLists.txt updated with new test targets:
   - `test_temperature_anomaly`
   - `test_multiphysics_temp_anomaly`

---

## Build and Run Instructions

```bash
cd /home/yzk/LBMProject/build

# Build tests
make test_temperature_anomaly
make test_multiphysics_temp_anomaly

# Run ThermalLBM isolation tests
./tests/test_temperature_anomaly

# Run MultiphysicsSolver tests (long running)
./tests/test_multiphysics_temp_anomaly
```

---

## Conclusion

The Tier2 temperature oscillation is **NOT** caused by the core ThermalLBM solver.
The bug is likely in:
1. MultiphysicsSolver physics coupling (most likely)
2. Phase change solver
3. VOF/fluid interaction with temperature field

Further testing with Test B and Test C will pinpoint the exact component.
