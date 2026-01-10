# Phase 2 Implementation Checklist
# Quick Reference for Validation Tasks

**Last Updated:** 2026-01-10
**Related:** See PHASE2_VALIDATION_PLAN.md for full details

---

## Quick Status Overview

| Benchmark | Status | Priority | Estimated Effort | Files to Modify |
|-----------|--------|----------|-----------------|-----------------|
| **Stefan Problem** | ⚠️ DISABLED | HIGH | 1-2 days | `test_stefan_problem.cu` (enable), `phase_change.cu` (verify) |
| **Natural Convection** | ❌ NOT STARTED | HIGH | 2-3 days | Create `test_natural_convection.cu` |
| **Marangoni Analytical** | ⚠️ PARTIAL | MEDIUM | 1 day | Modify `test_marangoni_velocity.cu` |
| **Khairallah Melt Pool** | ❌ NOT STARTED | LOW | 3-4 days | Create `test_khairallah_melt_pool.cu` |

---

## Task 1: Stefan Problem (HIGH PRIORITY)

### Status
- Tests exist but are DISABLED
- File: `/home/yzk/LBMProject/tests/validation/test_stefan_problem.cu`
- Issue: Expected error 50-150% with current temperature-based method
- Root cause: Enthalpy-based phase change not properly activated

### Action Items

- [ ] **Step 1.1:** Verify enthalpy method is correctly called
  ```bash
  cd /home/yzk/LBMProject
  grep -n "updateTemperatureFromEnthalpy" tests/validation/test_stefan_problem.cu
  grep -n "updateEnthalpyFromTemperature" tests/validation/test_stefan_problem.cu
  ```

- [ ] **Step 1.2:** Add diagnostic output to phase_change.cu
  - Log Newton-Raphson convergence rate
  - Check for NaN/Inf in enthalpy field
  - Verify latent heat absorption: `E_latent = Σ(fl · ρ · L · V_cell)`

- [ ] **Step 1.3:** Enable ShortTime test (0.5 ms)
  ```cpp
  // In test_stefan_problem.cu, line 255
  TEST_F(StefanProblemTest, ShortTime) {  // Remove "DISABLED_"
      // ...
  }
  ```

- [ ] **Step 1.4:** Run test and analyze error
  ```bash
  cd /home/yzk/LBMProject/build
  ./tests/validation/test_stefan_problem --gtest_filter=*ShortTime*
  ```
  Expected outcomes:
  - ✓ Error < 20%: PASS (move to longer times)
  - ⚠ Error 20-50%: Check convergence parameters
  - ✗ Error > 50%: Enthalpy update not active

- [ ] **Step 1.5:** Grid convergence study
  - Run with NX = 250, 500, 1000
  - Verify error decreases with refinement
  - Plot: interface position vs time (numerical + analytical)

### Success Criteria
```
✓ Interface position error < 20% at t = 0.5, 1.0, 2.0 ms
✓ Temperature profile shows correct liquid/mushy/solid regions
✓ Latent heat absorption verified (energy balance < 5% error)
```

### Estimated Time: 8-12 hours

---

## Task 2: Natural Convection (HIGH PRIORITY)

### Status
- No existing test
- Buoyancy force implementation exists but not validated
- Need quantitative validation against Davis (1983) benchmark

### Action Items

- [ ] **Step 2.1:** Create test file
  ```bash
  cd /home/yzk/LBMProject/tests/validation
  cp test_marangoni_velocity.cu test_natural_convection.cu
  # Modify to implement Rayleigh-Benard configuration
  ```

- [ ] **Step 2.2:** Implement buoyancy force kernel
  ```cuda
  __global__ void computeBuoyancyForceKernel(
      const float* temperature,
      float* force_y,
      float g, float beta, float T_ref,
      int num_cells)
  {
      // F = ρ·g·β·(T - T_ref)
      // For LBM: provide acceleration [m/s²]
  }
  ```

- [ ] **Step 2.3:** Set up domain
  - 2D cavity: 128 × 128 × 1 cells
  - L = 100 µm (LPBF-relevant)
  - Boundary conditions:
    - Bottom (y=0): T_hot, no-slip
    - Top (y=ny-1): T_cold, no-slip
    - Sides: adiabatic, no-slip

- [ ] **Step 2.4:** Compute dimensionless numbers
  ```cpp
  float Ra = g * beta * Delta_T * pow(L, 3) / (nu * alpha);
  float Pr = nu / alpha;
  // Target: Ra ~ 1.67e4, Pr ~ 0.21 (Ti6Al4V liquid)
  ```

- [ ] **Step 2.5:** Implement Nusselt number calculation
  ```cpp
  float computeNusseltNumber(ThermalLBM& thermal, float dx, float Delta_T) {
      // Nu = q_conv / q_cond
      // q_conv = vertical heat flux at mid-height
      // q_cond = k·ΔT / L
  }
  ```

- [ ] **Step 2.6:** Run to steady state
  - Criterion: dNu/dt < 1% over 100 time steps
  - Monitor: T_max, v_max, Nu vs time
  - Output: VTK files for visualization

- [ ] **Step 2.7:** Compare with Davis (1983)
  - Extract centerline velocity profile
  - Compare Nu with empirical correlation: `Nu ≈ 0.069·Ra^(1/3)·Pr^0.074`
  - Compute L2 error

### Success Criteria
```
✓ Nu within ±20% of Davis (1983) benchmark
✓ Flow pattern correct (upward at hot wall, downward at cold wall)
✓ Steady state reached (Nu oscillation < 5%)
```

### Estimated Time: 16-24 hours

---

## Task 3: Marangoni Analytical (MEDIUM PRIORITY)

### Status
- Velocity magnitude validated (0.7-0.8 m/s)
- Missing: analytical velocity profile comparison
- Missing: scaling law validation (v ∝ ∇T)

### Action Items

- [ ] **Step 3.1:** Modify existing test
  ```bash
  cd /home/yzk/LBMProject/tests/validation
  # Edit test_marangoni_velocity.cu
  # Add TEST: AnalyticalProfileComparison
  ```

- [ ] **Step 3.2:** Set up linear temperature gradient
  ```cpp
  // Instead of radial, use linear: T(x) = T_cold + (T_hot - T_cold)·x/Lx
  void initializeLinearTemperatureField(float* d_temperature, ...);
  ```

- [ ] **Step 3.3:** Extract velocity profile
  ```cpp
  std::vector<float> extractVelocityProfile_xz(FluidLBM& fluid, int j_mid);
  // u(x, z) at centerline y = j_mid
  ```

- [ ] **Step 3.4:** Compare with analytical solution
  ```cpp
  // Expected: u(z) = u_max for z > z_interface (liquid)
  //           u(z) = 0     for z < z_interface (vapor)
  // u_max = |dσ/dT| · |∇T| / μ
  ```

- [ ] **Step 3.5:** Scaling law parametric study
  - Vary dT/dx: 1e5, 2e5, 5e5, 1e6 K/m
  - Measure v_max for each
  - Fit linear regression: v_max = c · (dT/dx)
  - Compare c_fit vs c_analytical = |dσ/dT| / μ

### Success Criteria
```
✓ Velocity magnitude within ±20% of analytical prediction
✓ Profile shape correct (flat above interface, zero below)
✓ Scaling law: (c_fit - c_analytical)/c_analytical < 15%
```

### Estimated Time: 4-8 hours

---

## Task 4: Khairallah Melt Pool (LOW PRIORITY)

### Status
- Full multiphysics validation
- Most complex benchmark
- Requires all previous validations complete
- File: Create new `test_khairallah_melt_pool.cu`

### Action Items

- [ ] **Step 4.1:** Create test file
  ```bash
  cd /home/yzk/LBMProject/tests/validation
  cp test_laser_melting_senior.cu test_khairallah_melt_pool.cu
  ```

- [ ] **Step 4.2:** Configure laser parameters (Khairallah 2016)
  ```cpp
  laser.power = 195.0f;              // W
  laser.scan_speed = 1.0f;           // m/s
  laser.spot_radius = 55e-6f;        // m (1/e² radius)
  laser.absorptivity_solid = 0.35f;  // Ti6Al4V
  laser.absorptivity_liquid = 0.70f; // Enhanced for liquid
  laser.penetration_depth = 10e-6f;  // Beer-Lambert
  ```

- [ ] **Step 4.3:** Set up powder bed
  - Powder layer: z > 70 µm (porous, 50% packing)
  - Substrate: z < 70 µm (solid, 100% dense)
  - Initial temperature: 300 K

- [ ] **Step 4.4:** Implement melt pool measurements
  ```cpp
  float computeMeltPoolWidth(ThermalLBM& thermal, int k_surface);
  float computeMeltPoolDepth(ThermalLBM& thermal, int i_laser, int j_mid);
  float computeKeyholeDepth(VOFSolver& vof, int i_laser, int j_mid);
  ```

- [ ] **Step 4.5:** Run full simulation
  - Total time: 200 µs
  - Output interval: 10 µs (20 frames)
  - Monitor: width, depth, keyhole, T_max, v_max

- [ ] **Step 4.6:** Compare with Khairallah 2016 Table 2
  ```
  Reference data:
  - Width: 150 µm ± 15%
  - Depth: 100 µm ± 20%
  - Keyhole depth: 35 µm ± 30%
  - Marangoni velocity: 0.5-2.0 m/s
  ```

- [ ] **Step 4.7:** Generate visualizations
  - ParaView: temperature + velocity fields
  - Time series plots: width, depth vs time
  - Cross-section: melt pool shape

### Success Criteria
```
✓ Width within ±25% of Khairallah 2016
✓ Depth within ±25%
✓ Keyhole forms (qualitative)
✓ Marangoni velocity 0.5-2.0 m/s
```

### Estimated Time: 16-24 hours

---

## Priority Order

### Week 1: Foundation
1. **Stefan Problem** (1-2 days)
   - Highest ROI: Single physics validation
   - Unblocks phase change confidence

2. **Marangoni Analytical** (1 day)
   - Quick win: Modify existing test
   - Validates surface tension implementation

### Week 2: Multi-Physics Coupling
3. **Natural Convection** (2-3 days)
   - Validates thermal-fluid coupling
   - Foundation for buoyancy-driven flows

4. **Khairallah Melt Pool** (3-4 days)
   - Full validation of all physics
   - Industry-relevant benchmark

---

## Common Pitfalls and Solutions

### Issue 1: Stefan Problem Still Fails After Enabling Enthalpy
**Symptom:** Error > 50% even with enthalpy method
**Diagnosis:**
```bash
# Check if updateTemperatureFromEnthalpy is called
grep -A 5 "computeTemperature" tests/validation/test_stefan_problem.cu
```
**Solution:**
```cpp
// After thermal LBM step
thermal.computeTemperature();
phase_change.updateEnthalpyFromTemperature(thermal.getTemperature());

// After heat source
thermal.addHeatSource(d_heat_source, dt);

// CRITICAL: Solve T from H
phase_change.updateTemperatureFromEnthalpy(thermal.getTemperature());
```

### Issue 2: Natural Convection Nu Diverges
**Symptom:** Nu increases without bound
**Diagnosis:**
```cpp
// Check CFL number
float CFL = v_max * dt / dx;
std::cout << "CFL = " << CFL << " (should be < 0.3)" << std::endl;
```
**Solution:**
- Reduce time step: `dt = 0.5e-7f` (50 ns)
- Or reduce domain size: `L = 50e-6f` (50 µm)

### Issue 3: Marangoni Velocity Too Low
**Symptom:** v_max < 0.1 m/s (expected > 0.5 m/s)
**Diagnosis:**
```cpp
// Check force magnitude
float F_max = computeMaxForceMagnitude(d_fx, d_fy, d_fz);
std::cout << "Max Marangoni force: " << F_max << " N/m³" << std::endl;
// Expected: 10⁶-10⁹ N/m³
```
**Solution:**
- Verify force conversion: `F_lattice = F_physical * (dt² / dx)`
- Check temperature gradient: `|∇T| ~ 10⁶ K/m` (should be present)
- Verify dsigma_dT = -2.6e-4 N/(m·K) (correct sign)

### Issue 4: Melt Pool Too Small/Large
**Symptom:** Width/Depth off by > 50%
**Diagnosis:**
```cpp
// Check total laser energy deposition
float P_absorbed = laser.power * laser.absorptivity;
float E_total = P_absorbed * simulation_time;
float E_required = volume_melted * rho * L_fusion;
std::cout << "Energy balance: " << E_total / E_required << std::endl;
// Should be ~ 1.0-2.0 (accounting for conduction losses)
```
**Solution:**
- Adjust laser absorptivity: Try 0.4-0.6 for solid, 0.6-0.8 for liquid
- Check substrate cooling BC: `h_conv = 1000 W/(m²·K)`
- Verify penetration depth: 5-15 µm typical for metals

---

## Files Modified Summary

### New Files to Create
1. `/home/yzk/LBMProject/tests/validation/test_natural_convection.cu`
2. `/home/yzk/LBMProject/tests/validation/test_marangoni_analytical.cu`
3. `/home/yzk/LBMProject/tests/validation/test_khairallah_melt_pool.cu`

### Existing Files to Modify
1. `/home/yzk/LBMProject/tests/validation/test_stefan_problem.cu`
   - Remove `DISABLED_` from test names (lines 255, 273, 288)
   - Add enthalpy update diagnostics

2. `/home/yzk/LBMProject/src/physics/phase_change/phase_change.cu`
   - Add verbose logging for convergence (optional)
   - Verify Newton-Raphson fallback works

3. `/home/yzk/LBMProject/tests/validation/CMakeLists.txt`
   - Add new test executables

### Documentation to Create
1. `/home/yzk/LBMProject/benchmark/STEFAN_VALIDATION.md`
2. `/home/yzk/LBMProject/benchmark/NATURAL_CONVECTION_VALIDATION.md`
3. `/home/yzk/LBMProject/benchmark/MARANGONI_ANALYTICAL_VALIDATION.md`
4. `/home/yzk/LBMProject/benchmark/KHAIRALLAH_MELT_POOL_VALIDATION.md`

---

## Final Checklist

### Before Starting
- [ ] Read PHASE2_VALIDATION_PLAN.md (full details)
- [ ] Verify all solvers compile: `cd build && make -j8`
- [ ] Run existing tests: `ctest -R validation`
- [ ] Check git status: commit any pending changes

### Phase 2 Complete When
- [ ] Stefan problem tests pass (error < 20%)
- [ ] Natural convection: Nu within ±20%
- [ ] Marangoni: analytical profile validated
- [ ] Khairallah: melt pool dimensions within ±25%
- [ ] All validation reports documented
- [ ] Results reviewed and approved

---

**End of Checklist**
