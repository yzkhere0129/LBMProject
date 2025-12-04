# Test Cases for Energy Verification

This document defines specific test cases for verifying energy conservation and validating against literature.

---

## Test H (Baseline - Full Physics Stack)

**File**: `configs/lpbf_195W_test_H_full_physics.conf`

**Status**: RUNNING (need energy diagnostics)

**Configuration**:
```
Domain: 200×100×50 cells, dx=2 μm (400×200×100 μm³)
Time: 500 μs (5000 steps × 0.1 μs)
Laser: 195 W, r=50 μm, α=0.35, δ=5 μm
Material: Ti6Al4V
Physics: ALL ENABLED
  - Thermal diffusion + advection
  - Phase change (melting/solidification)
  - Fluid flow (LBM)
  - VOF advection
  - Marangoni effect
  - Surface tension
  - Darcy damping
  - Buoyancy
  - Radiation BC (ε=0.3)
```

**Expected Results**:
- Energy balance error: < 5%
- Peak temperature: 2,400-2,800 K
- Melt pool length: 150-300 μm
- Peak velocity: 0.5-3.0 m/s

**Current Results** (PRELIMINARY):
- Peak temperature: ~27,000 K (TOO HIGH!)
- Peak velocity: ~1 mm/s
- Status: NEEDS INVESTIGATION

**Action Items**:
1. Add energy diagnostics
2. Measure energy balance
3. Identify cause of high temperature
4. Adjust cooling mechanisms if needed

---

## Test H-RAD-1: Radiation BC Sensitivity (Low Emissivity)

**File**: `configs/lpbf_test_H_rad_low.conf`

**Purpose**: Test if lower emissivity allows higher temperature

**Changes from Test H**:
```
emissivity = 0.1  (was 0.3)
```

**Expected**:
- Higher peak temperature (less radiative cooling)
- Higher energy storage (dE/dt)
- Energy balance should still hold (< 5% error)

**Hypothesis**: If emissivity is too low, peak T will exceed 3,000 K

---

## Test H-RAD-2: Radiation BC Sensitivity (High Emissivity)

**File**: `configs/lpbf_test_H_rad_high.conf`

**Purpose**: Test if higher emissivity reduces peak temperature

**Changes from Test H**:
```
emissivity = 0.8  (was 0.3)
```

**Expected**:
- Lower peak temperature (more radiative cooling)
- Smaller melt pool (more cooling)
- Energy balance should still hold

**Hypothesis**: If emissivity is too high, peak T will be < 2,000 K (too cold)

---

## Test H-PEN-1: Shallow Laser Penetration

**File**: `configs/lpbf_test_H_pen_shallow.conf`

**Purpose**: Test surface heating vs volumetric heating

**Changes from Test H**:
```
laser_penetration_depth = 2.0e-6  (was 5.0e-6)
```

**Expected**:
- Higher surface temperature (more concentrated heating)
- Shallower melt pool
- Higher evaporation rate (hotter surface)
- Energy balance should show more P_evap

**Hypothesis**: Shallow penetration → more surface cooling → lower peak T

---

## Test H-PEN-2: Deep Laser Penetration

**File**: `configs/lpbf_test_H_pen_deep.conf`

**Purpose**: Test deep volumetric heating

**Changes from Test H**:
```
laser_penetration_depth = 20.0e-6  (was 5.0e-6)
```

**Expected**:
- Lower surface temperature (spread out heating)
- Deeper melt pool
- Lower evaporation rate (cooler surface)
- Energy balance should show less P_evap, more dE/dt

---

## Test H-POW-1: Lower Laser Power (Mohr 2020 Low)

**File**: `configs/lpbf_test_H_pow_low.conf`

**Purpose**: Replicate Mohr 2020 low power case

**Changes from Test H**:
```
laser_power = 100.0  (was 195.0)
```

**Expected**:
- Peak temperature: ~2,200 K (lower end of Mohr range)
- Smaller melt pool: ~100 μm length
- Energy balance: P_in = 35 W → expect dE/dt ~20-25 W

**Literature match**: Mohr 2020, 100W case

---

## Test H-POW-2: Higher Laser Power (Mohr 2020 High)

**File**: `configs/lpbf_test_H_pow_high.conf`

**Purpose**: Replicate Mohr 2020 high power case

**Changes from Test H**:
```
laser_power = 200.0  (was 195.0)
```

**Expected**:
- Peak temperature: ~2,700 K (high end of Mohr range)
- Larger melt pool: ~250 μm length
- Energy balance: P_in = 70 W

**Literature match**: Mohr 2020, 200W case

---

## Test H-SCAN-1: Moving Laser (Khairallah 2016 Replication)

**File**: `configs/lpbf_test_H_scan.conf`

**Purpose**: Test scanning laser (dynamic case)

**Changes from Test H**:
```
laser_power = 200.0
laser_scan_vx = 0.8  (was 0.0, stationary)
laser_start_x = 50.0e-6
laser_end_x = 350.0e-6
total_steps = 10000  (1 ms total, full scan)
```

**Expected**:
- Elongated melt pool (tracks laser)
- Peak velocity: 0.5-2.8 m/s (Khairallah 2016 range)
- Keyhole formation at high power (optional - may not capture)

**Literature match**: Khairallah 2016 (316L SS, but similar Ti6Al4V)

---

## Test H-GRID-1: Coarse Grid (Convergence Test)

**File**: `configs/lpbf_test_H_grid_coarse.conf`

**Purpose**: Test grid independence

**Changes from Test H**:
```
nx = 100, ny = 50, nz = 25  (was 200×100×50)
dx = 4.0e-6  (was 2.0e-6)
```

**Expected**:
- Similar peak temperature (±10%)
- Similar melt pool size (±20%)
- Energy balance should still hold
- Faster runtime (~8× faster)

**Purpose**: If results match Test H, grid is converged

---

## Test H-GRID-2: Fine Grid (Convergence Test)

**File**: `configs/lpbf_test_H_grid_fine.conf`

**Purpose**: Test grid independence (expensive!)

**Changes from Test H**:
```
nx = 400, ny = 200, nz = 100  (was 200×100×50)
dx = 1.0e-6  (was 2.0e-6)
total_steps = 10000  (need smaller timestep for CFL)
dt = 5.0e-8  (was 1.0e-7)
```

**Expected**:
- Same peak temperature as Test H (±5%)
- More detailed melt pool shape
- Slower runtime (~64× slower!)

**Purpose**: Confirm Test H grid is sufficient (not under-resolved)

**NOTE**: This test is very expensive - only run if Test H fails

---

## Test H-NOEVAP: No Evaporation (Diagnostic)

**File**: `configs/lpbf_test_H_noevap.conf`

**Purpose**: Isolate evaporation cooling contribution

**Changes from Test H**:
```
(Disable evaporation cooling in code - requires code change)
```

**Expected**:
- Higher peak temperature (no evap cooling)
- Energy balance: P_evap = 0, dE/dt increases

**Purpose**: Quantify how much evaporation cools the melt pool

---

## Test H-NORAD: No Radiation (Diagnostic)

**File**: `configs/lpbf_test_H_norad.conf`

**Purpose**: Isolate radiation cooling contribution

**Changes from Test H**:
```
enable_radiation_bc = false  (was true)
```

**Expected**:
- Higher peak temperature (no radiation cooling)
- Energy balance: P_rad = 0, dE/dt increases

**Purpose**: Quantify how much radiation cools the melt pool

**WARNING**: Temperature may diverge (go to infinity) without cooling!

---

## Test ENERGY-UNIT: Unit Test for Energy Diagnostics

**File**: `tests/test_energy_conservation.cpp`

**Purpose**: Verify energy diagnostic methods are correct

**Test Cases**:

### TC-1: Uniform Heating
```cpp
// Setup: Uniform temperature field, no flow
// Apply: Constant heat source Q [W/m³]
// Check: dE/dt = Q × Volume (to within 1%)
```

### TC-2: Radiation Only
```cpp
// Setup: Hot surface (T=3000 K), cold ambient (T=300 K)
// Apply: No heat source
// Check: P_rad = ε·σ·A·(T⁴ - T_amb⁴) (to within 2%)
```

### TC-3: Evaporation Only
```cpp
// Setup: Surface at T_boil
// Apply: Evaporation model
// Check: P_evap = ṁ × L_v (to within 5%)
```

### TC-4: Energy Conservation (Closed System)
```cpp
// Setup: Isolated system (adiabatic BC)
// Apply: Internal heat source Q
// Check: E(t) = E(0) + ∫Q dt (to within 1%)
```

---

## Summary Table

| Test ID | Purpose | Parameter Varied | Expected Outcome | Priority |
|---------|---------|------------------|------------------|----------|
| Test H | Baseline | - | Energy balance + literature match | HIGH |
| H-RAD-1 | Cooling sensitivity | ε = 0.1 | Higher T_max | MEDIUM |
| H-RAD-2 | Cooling sensitivity | ε = 0.8 | Lower T_max | MEDIUM |
| H-PEN-1 | Heating distribution | δ = 2 μm | Shallower melt pool | MEDIUM |
| H-PEN-2 | Heating distribution | δ = 20 μm | Deeper melt pool | MEDIUM |
| H-POW-1 | Literature match | P = 100 W | Match Mohr 2020 low | HIGH |
| H-POW-2 | Literature match | P = 200 W | Match Mohr 2020 high | HIGH |
| H-SCAN-1 | Dynamic case | v_scan = 0.8 m/s | Match Khairallah 2016 | MEDIUM |
| H-GRID-1 | Convergence | dx = 4 μm | Similar results | LOW |
| H-GRID-2 | Convergence | dx = 1 μm | Same results | LOW |
| H-NOEVAP | Diagnostic | No evap | Isolate P_evap | LOW |
| H-NORAD | Diagnostic | No rad | Isolate P_rad | LOW |
| ENERGY-UNIT | Code verification | - | Pass all unit tests | HIGH |

---

## Execution Order

**Phase 1: Baseline Verification** (CURRENT)
1. Run Test H with energy diagnostics
2. Measure energy balance
3. If error > 5%, debug before proceeding

**Phase 2: Literature Validation**
1. Run H-POW-1 (100 W)
2. Run H-POW-2 (200 W)
3. Compare with Mohr 2020 data
4. Adjust parameters if needed

**Phase 3: Sensitivity Analysis**
1. Run H-RAD-1, H-RAD-2 (emissivity sweep)
2. Run H-PEN-1, H-PEN-2 (penetration sweep)
3. Identify optimal parameters

**Phase 4: Advanced Tests** (Optional)
1. Run H-SCAN-1 (moving laser)
2. Run H-GRID-1, H-GRID-2 (convergence)
3. Document limitations

---

## Acceptance Criteria

**Pass**: Test H meets ALL of:
1. Energy balance error < 5%
2. Peak temperature 2,400-2,800 K
3. Melt pool length 150-300 μm
4. Simulation stable (no NaN/Inf)

**Pass with Adjustment**: Test H fails, but H-RAD or H-PEN passes
- Document the adjusted parameters
- Explain physical justification

**Fail**: All tests fail
- Energy balance error > 10%
- Temperature outside 1,500-5,000 K range
- Fundamental physics error suspected

---

**Last updated**: 2025-11-19
**Author**: Claude Code
**Status**: Design phase (ready for implementation)
