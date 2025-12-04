# ThermalLBM Test Suite Quick Reference

## Test Coverage Overview

**Total Test Suites:** 10
**Total Tests:** 30+
**Estimated Runtime:** < 10 minutes (NVIDIA RTX 3090)

---

## Test Suite Summary

| Suite | Tests | Focus | Success Criteria |
|-------|-------|-------|------------------|
| **1. Pure Conduction** | 4 | 1D diffusion, steady-state | L2 error < 1-5% |
| **2. Variable Properties** | 1 | Temperature-dependent k(T), cp(T) | Correct interpolation in mushy zone |
| **3. Stefan Problem** | 1 | Moving phase interface | s(t) ∝ √t scaling |
| **4. Advection-Diffusion** | 1 | Passive scalar transport | Peak tracking < 2 cells |
| **5. Energy Conservation** | 2 | Isolated system, heat source balance | Energy drift < 0.1% |
| **6. Boundary Conditions** | 4 | Dirichlet, Neumann, radiation, substrate | BC enforcement, energy balance |
| **7. Stability** | 3 | High Pe, sharp gradients, omega capping | No oscillations, no blow-up |
| **8. Phase Change** | 3 | Melting, solidification, mushy zone | Latent heat plateaus, f_l correctness |
| **9. Evaporation** | 2 | Mass flux, cooling power | Hertz-Knudsen model validation |
| **10. Kernel Unit Tests** | 2 | Equilibrium, temperature clamping | Correct low-level operations |

---

## Critical Test Cases

### Must-Pass Tests (Blocking)

1. **Test 1.1: Steady-State 1D Conduction**
   - Validates: Basic diffusion correctness
   - Criterion: L2 error < 1%, R² > 0.999

2. **Test 5.1: Energy Conservation (Isolated)**
   - Validates: No spurious energy creation/loss
   - Criterion: |E(t) - E(0)| / E(0) < 0.1%

3. **Test 6.1: Dirichlet Boundary Condition**
   - Validates: BC enforcement
   - Criterion: T_boundary ± 1 K

4. **Test 7.1: High Peclet Stability**
   - Validates: No numerical blow-up
   - Criterion: Simulation completes without NaN/Inf

### High-Value Tests (Validation)

5. **Test 1.2: Gaussian Diffusion**
   - Validates: Transient accuracy
   - Criterion: L2 error < 5% at t = 0.1, 0.5, 1.0 ms

6. **Test 3.1: Stefan Problem**
   - Validates: Phase change interface dynamics
   - Criterion: Interface position follows √t

7. **Test 8.1: Latent Heat Absorption**
   - Validates: Temperature plateau during melting
   - Criterion: Detectable plateau in mushy zone

---

## Analytical Solutions Reference

### 1D Steady Conduction
```
T(x) = T_cold + (T_hot - T_cold) * x / L
```

### Gaussian Diffusion
```
sigma(t) = sqrt(sigma0² + 2*alpha*t)
T(x,t) = T0 + (Tp - T0) * (sigma0/sigma_t) * exp(-(x-x0)²/(2*sigma_t²))
```

### Stefan Problem
```
s(t) = 2*lambda*sqrt(alpha*t)
where lambda from: lambda*exp(lambda²)*erf(lambda) = Ste/sqrt(pi)
Ste = cp*(T_hot - T_melt) / L_fusion  (Stefan number)
```

### Radiation Cooling
```
dT/dt = -epsilon*sigma*(T⁴ - T_amb⁴) / (rho*cp*dx)
```

---

## Physical Parameters for Testing

### Ti-6Al-4V Properties
```cpp
T_solidus      = 1878 K
T_liquidus     = 1943 K
T_vaporization = 3560 K
L_fusion       = 286,000 J/kg
L_vaporization = 9,830,000 J/kg
k_solid        = 21.9 W/(m·K)
k_liquid       = 28.5 W/(m·K)
rho_solid      = 4430 kg/m³
cp_solid       = 546 J/(kg·K)
alpha_solid    = 9.05e-6 m²/s
```

### Typical Domain Parameters
```cpp
nx = 50-200       (varies by test)
dx = 2e-6 m       (2 microns)
dt = 1e-7 s       (0.1 microseconds, from CFL)
alpha_lattice = alpha_physical * dt / dx²  (~0.02-0.15)
```

### CFL Condition
```cpp
dt <= 0.5 * dx² / alpha  (thermal CFL)
alpha_lattice = alpha * dt / dx² <= 0.25  (for stability)
```

---

## Expected Accuracy Ranges

| Test Type | Expected L2 Error | Notes |
|-----------|-------------------|-------|
| Pure diffusion | 1-5% | BGK has inherent diffusion error |
| Advection-diffusion | 5-10% | Higher error at high Pe |
| Steady-state | < 1% | Better accuracy at equilibrium |
| Phase change | 5-10% | Latent heat introduces complexity |
| Energy conservation | < 0.1% | Should be near machine precision |

---

## Common Failure Modes

### 1. Energy Conservation Failure
**Symptom:** E(t) drifts from E(0)
**Causes:**
- Incorrect T_ref in energy calculation
- Boundary condition energy leaks
- Missing source term accounting

**Fix:**
- Use T_initial as reference (not T_solidus)
- Verify adiabatic BCs via bounce-back
- Account for all heat sources/sinks

---

### 2. Instability at High Pe
**Symptom:** Oscillations or NaN after few steps
**Causes:**
- Omega too high (ω > 1.85)
- Peclet number Pe > 10
- Sharp velocity gradients

**Fix:**
- Enable omega capping (ω ≤ 1.85)
- Reduce velocity or increase diffusivity
- Use MRT collision operator (future work)

---

### 3. Phase Change Not Working
**Symptom:** No temperature plateau during melting
**Causes:**
- Phase change solver not initialized
- applyPhaseChangeCorrection() not called
- Latent heat too large (overcorrection)

**Fix:**
- Enable phase change in constructor: `enable_phase_change=true`
- Call correction after computeTemperature()
- Check L_fusion / cp ratio (should be ~ 100-500 K)

---

### 4. Boundary Conditions Not Enforced
**Symptom:** T_boundary ≠ T_expected
**Causes:**
- BCs applied before streaming (wrong order)
- BC kernel not launched
- Distribution functions not updated

**Fix:**
- Order: collision → streaming → BC → computeTemperature
- Verify BC kernel launch configuration
- Check that g[idx] is modified, not just T[idx]

---

## Test Execution Commands

### Run All Thermal Tests
```bash
cd /home/yzk/LBMProject/build
ctest -R "ThermalLBM*" --output-on-failure
```

### Run Specific Test Suite
```bash
./test_thermal_conduction          # Suite 1
./test_thermal_phase_change        # Suite 8
./test_thermal_boundaries          # Suite 6
./test_thermal_energy_conservation # Suite 5
```

### Debug Single Test
```bash
./test_thermal_conduction --gtest_filter="*SteadyStateConduction1D*"
```

### Verbose Output
```bash
./test_thermal_conduction --gtest_print_time=1 -v
```

---

## Validation Checklist

Before declaring ThermalLBM production-ready:

- [ ] All Suite 1 tests pass (pure conduction baseline)
- [ ] All Suite 5 tests pass (energy conservation)
- [ ] All Suite 6 tests pass (boundary conditions)
- [ ] At least 2/3 Suite 7 tests pass (stability)
- [ ] At least 2/3 Suite 8 tests pass (phase change)
- [ ] No memory leaks (valgrind/cuda-memcheck)
- [ ] Performance benchmark: 1M cells/sec throughput
- [ ] Code coverage: > 80% of thermal_lbm.cu

---

## Performance Benchmarks

### Expected Throughput (RTX 3090)
- Pure diffusion: ~15 M cells/sec
- Advection-diffusion: ~12 M cells/sec
- With phase change: ~8 M cells/sec

### Memory Usage
- Per cell: 7 floats (distribution) + 1 float (T) = 32 bytes
- 1M cells: ~32 MB GPU memory
- 10M cells: ~320 MB GPU memory

---

## Implementation Priority

### Phase 1: Core Validation (Week 1)
1. Implement Test 1.1 (steady-state)
2. Implement Test 5.1 (energy conservation)
3. Implement Test 6.1 (Dirichlet BC)
→ Goal: Validate basic diffusion correctness

### Phase 2: Transient Accuracy (Week 2)
4. Implement Test 1.2 (Gaussian diffusion)
5. Implement Test 4.1 (advection-diffusion)
→ Goal: Validate time-dependent behavior

### Phase 3: Advanced Physics (Week 3)
6. Implement Test 8.1, 8.2 (phase change)
7. Implement Test 3.1 (Stefan problem)
→ Goal: Validate complex thermophysics

### Phase 4: Robustness (Week 4)
8. Implement Test 7.1, 7.2, 7.3 (stability)
9. Implement Test 9.1, 9.2 (evaporation)
→ Goal: Stress testing and edge cases

---

## Document Links

- **Full Specification:** `/home/yzk/LBMProject/docs/THERMAL_TEST_SPECS.md`
- **Existing Test Example:** `/home/yzk/LBMProject/tests/validation/test_pure_conduction.cu`
- **ThermalLBM Header:** `/home/yzk/LBMProject/include/physics/thermal_lbm.h`
- **ThermalLBM Source:** `/home/yzk/LBMProject/src/physics/thermal/thermal_lbm.cu`

---

**Document Version:** 1.0
**Last Updated:** 2025-12-03
**Status:** Ready for Implementation
