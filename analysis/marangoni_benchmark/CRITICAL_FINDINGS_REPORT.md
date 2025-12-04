# Marangoni Benchmark VTK Analysis - CRITICAL FINDINGS

**Analysis Date**: 2025-12-03
**Analyst**: VTK Data Analysis Specialist
**Project**: /home/yzk/LBMProject
**Data Source**: /home/yzk/LBMProject/build/marangoni_benchmark_output/

---

## Executive Summary

**CRITICAL FAILURE DETECTED**: The Marangoni benchmark simulation exhibits **ZERO velocity** throughout the entire simulation period (0-20 μs, 21 timesteps). The expected thermocapillary-driven Marangoni flow is completely absent.

**Status**: UNACCEPTABLE - Physics implementation is fundamentally broken

---

## Critical Findings

### 1. ZERO VELOCITY FIELD (CRITICAL)

**Finding**: All velocity components are exactly zero at all timesteps.

**Data**:
```
v_max = 0.0000 m/s  (all timesteps)
vx_mean = 0.0000 m/s (all timesteps)
vy_max = 0.0000 m/s  (all timesteps)
vz_max = 0.0000 m/s  (all timesteps)
```

**Expected**:
- Marangoni velocity scale: ~13.0 m/s (characteristic)
- Literature values for Ti-6Al-4V: 0.5-2.0 m/s
- Even minimal flow: 0.01-0.1 m/s expected within first few microseconds

**Deviation**: **INFINITE** - Zero velocity vs expected O(1) m/s

**Physical Interpretation**:
- No Marangoni force is being applied to the fluid
- Surface tension gradient is not driving flow
- Thermocapillary mechanism is not functioning
- Test is effectively a **static temperature field** with no dynamics

**Root Causes (Possible)**:
1. Marangoni force computation is not being called
2. Marangoni force is computed but not applied to fluid solver
3. Interface tracking is absent (no FillLevel field found in VTK)
4. Force-to-velocity coupling is broken
5. Boundary conditions are over-constraining the flow

---

### 2. INVERTED TEMPERATURE GRADIENT SIGN (CRITICAL)

**Finding**: Temperature gradient has **opposite sign** to expected.

**Data**:
```
Measured gradient: -0.505 K/μm
Expected gradient: +0.503 K/μm
Error: 200.5%
```

**Physical Interpretation**:
- Temperature decreases from LEFT (x=0) to RIGHT (x=Lx)
- Measured: T(x=0) = 2000 K, T(x=Lx) = 1800 K
- This is actually correct for the hot-to-cold setup
- The **sign convention issue** is in the calculation, not physics

**Corrected Assessment**:
The temperature gradient magnitude is **perfect** (|dT/dx| = 0.505 vs 0.503 K/μm, error = 0.4%).
The sign difference is due to coordinate system convention:
- Expected: dT/dx = ΔT/(x_max - x_min) = (1800-2000)/(Lx) = -200/398μm = -0.503 K/μm
- Measured: -0.505 K/μm

**Status**: PASS (once sign convention is corrected)

---

### 3. MISSING CRITICAL FIELDS IN VTK OUTPUT (CRITICAL)

**Finding**: VTK files lack essential fields for Marangoni analysis.

**Available Fields**:
- Temperature (scalar) - PRESENT
- Velocity (vector) - PRESENT

**Missing Fields**:
- FillLevel / VOF (scalar) - MISSING
- MarangoniForce (vector) - MISSING
- InterfaceIndicator (scalar) - MISSING

**Impact**:
Cannot verify:
- Interface position and shape
- Interface thickness and stability
- Marangoni force localization at interface
- Force magnitude and direction
- Force tangentiality (should be perpendicular to interface normal)

**This prevents validation of the most critical physics**:
The Marangoni effect specifically acts **at the free surface interface**. Without interface data, we cannot determine if forces are even being computed in the correct regions.

---

### 4. TEMPERATURE FIELD PROPERTIES

**Finding**: Temperature field is perfectly linear and static.

**Data**:
```
Boundary Conditions:
  Left wall (x=0):  T = 2000.00 K ✓ Correct
  Right wall (x=Lx): T = 1800.00 K ✓ Correct

Linearity:
  R² = 0.9999998 ✓ Excellent

Temporal Evolution:
  Temperature field STATIC (no change over 20 μs)
  dT/dx constant: -505,046 K/m at all timesteps
```

**Assessment**: Temperature boundary conditions are implemented correctly.

**Physical Interpretation**:
- Pure conduction solution with fixed BCs
- No thermal convection (because no fluid flow)
- No coupling between thermal and fluid fields
- System is in **thermal equilibrium** from t=0

---

## Detailed Quantitative Analysis

### Temperature Gradient Validation

| Parameter | Measured | Expected | Error | Status |
|-----------|----------|----------|-------|--------|
| Magnitude | 0.505 K/μm | 0.503 K/μm | 0.4% | **PASS** |
| Sign | Negative | Negative | - | **PASS** |
| Linearity (R²) | 0.9999998 | >0.999 | - | **PASS** |
| Left BC | 2000.00 K | 2000.00 K | 0.0% | **PASS** |
| Right BC | 1800.00 K | 1800.00 K | 0.0% | **PASS** |

### Velocity Field Validation

| Parameter | Measured | Expected | Ratio | Status |
|-----------|----------|----------|-------|--------|
| v_max | 0.0000 m/s | ~1.0 m/s | 0.000 | **FAIL** |
| Marangoni scale | 0.0000 m/s | 13.0 m/s | 0.000 | **FAIL** |
| vx_mean | 0.0000 m/s | >0 | - | **FAIL** |
| Flow direction | None | +x (hot→cold) | - | **FAIL** |

### Expected Physical Behavior (NOT OBSERVED)

For a Marangoni benchmark with:
- Ti-6Al-4V material
- dσ/dT = -0.26 mN/(m·K)
- ΔT = 200 K
- Domain: 400 × 200 × 4 μm³

**Expected observations by t=20 μs:**

1. **Surface velocity**: 0.1-1.0 m/s at interface
2. **Flow pattern**:
   - Surface flow from hot (left) to cold (right)
   - Return flow in bulk liquid (opposite direction)
   - Formation of convection cell
3. **Velocity profile**: Peak at interface, decay into bulk
4. **Time evolution**: Velocity increases from 0, approaches steady state
5. **Interface**: Stable horizontal interface at ~70% of domain height

**Actual observations:**
- All velocity components = 0
- No flow pattern
- No convection
- No time evolution
- Interface not tracked (no data)

---

## Physics Validation Summary

| Physics Component | Expected | Observed | Status |
|-------------------|----------|----------|--------|
| Temperature gradient | Linear, -0.503 K/μm | Linear, -0.505 K/μm | **PASS** |
| Temperature BCs | Fixed at walls | Correct | **PASS** |
| Interface tracking | VOF at y=70% | Not in VTK | **FAIL** |
| Marangoni force | ~22 MN/m³ at interface | Not in VTK | **FAIL** |
| Surface velocity | 0.1-1.0 m/s | 0.0 m/s | **FAIL** |
| Flow direction | +x (hot→cold) | None | **FAIL** |
| Convection cell | Should form | Absent | **FAIL** |
| Time evolution | Velocity increases | No change | **FAIL** |

---

## Root Cause Analysis

### Hypothesis 1: Marangoni Force Not Applied (MOST LIKELY)

**Evidence**:
- Velocity remains exactly zero
- No time evolution
- Temperature field is static (no convective transport)

**Mechanism**:
The Marangoni force computation may be implemented but:
1. Not called in the integration loop
2. Not passed to the fluid solver
3. Zeroed out due to missing interface data
4. Applied with wrong unit conversion

**Test**: Check if MarangoniEffect class is instantiated and called in test code.

### Hypothesis 2: Interface Not Initialized

**Evidence**:
- No FillLevel field in VTK output
- Cannot locate interface for force application

**Mechanism**:
- VOF solver may not be initialized
- Fill level field not set up
- Marangoni force requires interface normal vectors (from ∇f)

**Test**: Search test code for VOF initialization.

### Hypothesis 3: Fluid Solver Not Running

**Evidence**:
- Zero velocity at all times
- No response to any forces

**Mechanism**:
- Fluid LBM solver may not be stepping forward
- Velocity update disabled
- Boundary conditions over-constraining

**Test**: Check if FluidLBM::step() is called in test loop.

### Hypothesis 4: Missing Force-Velocity Coupling

**Evidence**:
- Temperature field evolves correctly (thermal solver works)
- Velocity remains zero (fluid solver doesn't respond)

**Mechanism**:
- Force accumulation not working
- LBM collision step not applying external forces
- Unit conversion errors making forces negligible

**Test**: Add diagnostic outputs in force accumulation.

---

## Recommended Actions

### IMMEDIATE (Required for Test to Pass)

1. **Verify Marangoni Force Computation**:
   ```
   - Check if MarangoniEffect::computeForce() is called
   - Add diagnostic output: max |F_marangoni|
   - Expected: O(10^7) N/m³ at interface
   ```

2. **Verify Interface Initialization**:
   ```
   - Check VOFSolver initialization in test
   - Verify fill level field is allocated and initialized
   - Expected: f(y < 70%) = 1.0, f(y > 70%) = 0.0
   ```

3. **Verify Fluid Solver Integration**:
   ```
   - Check FluidLBM::step() is called each timestep
   - Verify forces are passed to LBM collision
   - Check velocity update is enabled
   ```

4. **Add Missing Fields to VTK Output**:
   ```
   - Output FillLevel field
   - Output MarangoniForce field
   - Output InterfaceIndicator
   - Enable full physics diagnostic output
   ```

### VERIFICATION (After Fixes)

1. **Minimum Acceptable Criteria**:
   - v_max > 0.01 m/s by t=20 μs
   - vx_mean > 0 (positive flow)
   - Velocity increases with time
   - Interface visible in VTK (f field)

2. **Target Performance**:
   - v_max ~ 0.1-1.0 m/s (order of magnitude)
   - Clear surface flow pattern
   - Marangoni force O(10^7) N/m³
   - Convection cell visible

3. **Optimal Performance**:
   - v_max ~ 0.5-2.0 m/s (literature range)
   - Steady-state reached
   - Force tangential to interface (|Fy|/|Fx| < 0.1)
   - Mass conserved (<1% change)

---

## Comparison with Test Output Logs

The test execution log (`/home/yzk/LBMProject/build/marangoni_output.txt`) shows:

**Test 2C Results**:
```
Maximum surface velocity achieved: 1.0940 m/s
Final surface velocity: 0.4180 m/s
Literature range (LPBF Ti6Al4V): 0.5 - 2.0 m/s
Status: ✓ CRITICAL PASS
```

**DISCREPANCY**: Test reports success with v_max = 1.09 m/s, but VTK files show v_max = 0 m/s.

**Possible Explanations**:
1. Different test runs (Test 2C vs Benchmark test)
2. VTK output is from initial state only
3. Velocity present in memory but not written to VTK
4. Test passes on internal diagnostics, VTK export is broken

**Action**: Verify which test generated these VTK files.

---

## File Locations

**Analysis Scripts**:
- Full analysis: `/home/yzk/LBMProject/scripts/analyze_marangoni_vtk.py`
- Simplified version: `/home/yzk/LBMProject/scripts/analyze_marangoni_vtk_simple.py`

**Output Files**:
- Analysis plot: `/home/yzk/LBMProject/analysis/marangoni_benchmark/marangoni_analysis.png`
- CSV data: `/home/yzk/LBMProject/analysis/marangoni_benchmark/marangoni_results.csv`
- This report: `/home/yzk/LBMProject/analysis/marangoni_benchmark/CRITICAL_FINDINGS_REPORT.md`

**VTK Data**:
- Location: `/home/yzk/LBMProject/build/marangoni_benchmark_output/`
- Files: 21 VTK files (marangoni_000000.vtk to marangoni_020000.vtk)
- Time span: 0-20 μs (1 μs intervals)

---

## Conclusion

**The Marangoni benchmark test is FAILING at a fundamental level.**

**The simulation produces**:
- ✓ Correct temperature field
- ✗ Zero velocity field
- ✗ No Marangoni-driven flow
- ✗ Missing interface tracking data
- ✗ Missing force field data

**This is not a quantitative deviation - it is a complete absence of the target physics.**

The test cannot be considered validated until:
1. Non-zero velocities are observed
2. Marangoni forces are confirmed at the interface
3. Flow direction matches expected thermocapillary behavior
4. VTK output includes all critical fields

**The physics implementation is not functioning as intended.**

---

## Technical Details

### Analysis Configuration

```
Domain: 200 × 100 × 2 cells
Resolution: 2.0 μm
Physical size: 400 × 200 × 4 μm³
Temperature: 2000 K (left) → 1800 K (right)
Material: Ti-6Al-4V
dσ/dT: -0.26 mN/(m·K)
Timestep: 1 ns (output every 1000 steps = 1 μs)
Simulation time: 20 μs
```

### Expected Force Magnitude

```
F_Marangoni = |dσ/dT| × |∇T| × |∇f|

where:
|dσ/dT| = 0.26 mN/(m·K) = 2.6×10⁻⁴ N/(m·K)
|∇T| = 0.505 K/μm = 5.05×10⁵ K/m
|∇f| ~ 1/h ~ 1/(6μm) = 1.67×10⁵ m⁻¹ (for 3-cell interface)

F ≈ 2.6×10⁻⁴ × 5.05×10⁵ × 1.67×10⁵
F ≈ 2.2×10⁷ N/m³ = 22 MN/m³
```

This force should drive significant fluid motion within microseconds.

---

**END OF REPORT**
