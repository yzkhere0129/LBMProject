# Validation Action Plan: Phase 1 - Fluid Dynamics
## Critical Gap Remediation (4-Week Sprint)

**Start Date:** 2026-01-10
**Target Completion:** 2026-02-07
**Priority:** CRITICAL (Blocks publication)
**Owner:** Development Team

---

## OBJECTIVE

Establish **quantitative validation** of the fluid dynamics solver (D3Q19 LBM) through standard CFD benchmarks. Currently, the fluid solver has **no Reynolds-dependent validation**, making its accuracy **unknown** for the melt pool convection simulations central to the AM application.

**Success Criteria:**
1. Taylor-Green vortex decay rate within 5% of analytical
2. Lid-driven cavity velocities within 1% of Ghia et al. (1982) at Re=100, 400
3. Grid convergence proven (second-order spatial accuracy)
4. Poiseuille Reynolds sweep validates kinematic viscosity

---

## BACKGROUND: WHY THIS IS CRITICAL

### Current Fluid Validation Status
- ✅ Poiseuille flow: **Single Reynolds number** tested
- ✅ Unit tests: Collision, streaming, boundaries
- ❌ **NO Reynolds-dependent validation**
- ❌ **NO grid convergence study**
- ❌ **NO benchmark comparison to literature**

### Impact on AM Simulation Confidence
The fluid solver drives:
- Melt pool convection (thermal transport)
- Marangoni flow (thermocapillary circulation)
- Buoyancy-driven mixing (Rayleigh-Benard)

**Without fluid validation, the entire multiphysics simulation is untrustworthy.**

### Comparison to Thermal Solver (85% confidence)
- Thermal: 2.01% walberla error, second-order convergence, analytical Gaussian
- Fluid: Poiseuille only, no Re sweep, **no quantitative benchmarks**

**The gap is unacceptable for publication-quality work.**

---

## PHASE 1 TESTS (4 TESTS, 4 WEEKS)

### Test 1: Taylor-Green Vortex 2D (Week 1)

**File:** `/home/yzk/LBMProject/tests/validation/test_taylor_green_2d.cu`

**Purpose:** Validate momentum diffusion (viscosity) through vortex decay.

**Benchmark:**
- Initial condition: u = U₀ sin(kx) cos(ky), v = -U₀ cos(kx) sin(ky)
- Analytical decay: Kinetic energy E(t) = E₀ exp(-2νk²t)
- Reference: Taylor & Green (1937), He et al. (1997) for LBM

**Test Configuration:**
```cpp
Domain: 256×256 (periodic BC)
Reynolds: Re = U₀L/ν = 100 (well-resolved)
Grid spacings: dx = L/128, L/256, L/512 (convergence study)
Duration: t = 5 τ_visc (where τ_visc = L²/ν)
Output: Kinetic energy vs time
```

**Acceptance Criteria:**
- Decay rate error < 5% vs analytical
- Grid convergence order 1.8 < p < 2.2
- Energy dissipation matches ν exactly

**Estimated Effort:** 3 days (implementation) + 1 day (analysis)

**Deliverables:**
- `test_taylor_green_2d.cu` (test file)
- `taylor_green_analytical.h` (analytical solution header)
- `TAYLOR_GREEN_RESULTS.md` (validation report)

---

### Test 2: Lid-Driven Cavity Re=100 (Week 2)

**File:** `/home/yzk/LBMProject/tests/validation/test_lid_driven_cavity_re100.cu`

**Purpose:** Validate pressure-velocity coupling in confined flow.

**Benchmark:**
- Reference: Ghia, Ghia, & Shin (1982) - **standard CFD benchmark**
- Geometry: 1×1 square, top wall moving at U=1, other walls stationary
- Reynolds: Re = UL/ν = 100
- Mesh: 129×129 (standard resolution)

**Test Configuration:**
```cpp
Domain: 129×129 cells
Boundary conditions:
  - Top: Moving wall (u=1, v=0) - bounce-back with velocity
  - Others: No-slip walls (u=0, v=0)
Reynolds: 100 (ν tuned via tau)
Convergence: |u_max(t) - u_max(t-Δt)| / u_max < 1e-6
Output: u(x) along vertical centerline, v(y) along horizontal centerline
```

**Acceptance Criteria:**
- u-velocity at centerline: L∞ error < 1% vs Ghia data
- v-velocity at centerline: L∞ error < 1% vs Ghia data
- Primary vortex center location within 1 cell of Ghia
- Secondary vortex (if present) location within 2 cells

**Ghia Reference Data (Re=100):**
```
y=0.5 centerline: u_max ≈ 0.18
x=0.5 centerline: v_min ≈ -0.25
Primary vortex: (x,y) ≈ (0.62, 0.74)
```

**Estimated Effort:** 4 days (moving wall BC, convergence) + 1 day (analysis)

**Deliverables:**
- `test_lid_driven_cavity_re100.cu`
- `ghia_1982_data.h` (reference data)
- `LID_DRIVEN_CAVITY_RE100_RESULTS.md`

---

### Test 3: Lid-Driven Cavity Re=400 (Week 2)

**File:** `/home/yzk/LBMProject/tests/validation/test_lid_driven_cavity_re400.cu`

**Purpose:** Validate higher Reynolds number (stronger inertia effects).

**Benchmark:**
- Same as Re=100 but more complex flow structure
- Secondary corner vortices become stronger
- Tests stability at higher Re

**Test Configuration:**
```cpp
Domain: 129×129 cells (standard)
Reynolds: 400
Convergence criterion: Same as Re=100
```

**Acceptance Criteria:**
- u-velocity: L∞ error < 2% vs Ghia (slightly relaxed due to higher Re)
- v-velocity: L∞ error < 2% vs Ghia
- Both secondary vortices captured
- No instability or oscillations

**Estimated Effort:** 2 days (reuse Re=100 code, different ν) + 1 day (analysis)

**Deliverables:**
- `test_lid_driven_cavity_re400.cu`
- `LID_DRIVEN_CAVITY_RE400_RESULTS.md`

---

### Test 4: Fluid Grid Convergence Study (Week 3-4)

**File:** `/home/yzk/LBMProject/tests/validation/test_fluid_grid_convergence.cu`

**Purpose:** Prove second-order spatial accuracy for fluid solver.

**Benchmark:**
- Use Poiseuille flow (has analytical solution)
- Vary grid resolution: 25, 50, 100, 200 cells across channel
- Compute L2 error vs analytical parabolic profile

**Test Configuration:**
```cpp
Setup: 2D channel flow
Resolutions: Ny = 25, 50, 100, 200 (across channel)
Reynolds: Re = 10 (low Re, well-resolved)
Body force: Constant (drives Poiseuille)
Analytical: u(y) = u_max [1 - (2y/H - 1)²]
```

**Acceptance Criteria:**
- Convergence order: 1.8 < p < 2.2 (second-order)
- Finest grid error < 0.5% (L2 norm)
- Error decreases monotonically with refinement

**Computation:**
```
E(dx) = ||u_numerical - u_analytical||_L2
p = log(E₁/E₂) / log(dx₁/dx₂)
```

**Estimated Effort:** 3 days (reuse thermal convergence framework) + 2 days (analysis)

**Deliverables:**
- `test_fluid_grid_convergence.cu`
- `poiseuille_analytical.h` (if not exists)
- `FLUID_GRID_CONVERGENCE_RESULTS.md`

---

### Bonus Test: Poiseuille Reynolds Sweep (Week 4)

**File:** `/home/yzk/LBMProject/tests/validation/test_poiseuille_reynolds_sweep.cu`

**Purpose:** Validate that kinematic viscosity ν = cs²(τ - 0.5)Δt is correct.

**Test Configuration:**
```cpp
Reynolds range: Re = 1, 10, 50, 100, 200
For each Re:
  - Adjust tau to get correct ν
  - Run to steady state
  - Compare u_max to analytical
```

**Acceptance Criteria:**
- All Re: u_max within 1% of analytical
- Proves ν formula is correct
- Proves no Re-dependent errors

**Estimated Effort:** 2 days (if time permits)

---

## IMPLEMENTATION STRATEGY

### Week 1: Taylor-Green Vortex

**Day 1-2: Implementation**
- Create analytical solution header (`analytical/taylor_green.h`)
  ```cpp
  namespace analytical {
    float taylor_green_u(float x, float y, float t, float U0, float nu, float k);
    float taylor_green_energy(float t, float E0, float nu, float k);
  }
  ```
- Implement test (`test_taylor_green_2d.cu`)
  - Initialize velocity field
  - Run simulation (periodic BC)
  - Measure kinetic energy decay
  - Compare to analytical

**Day 3: Grid Convergence**
- Run on 3 grids (128², 256², 512²)
- Compute convergence rate
- Generate plots (energy vs time)

**Day 4: Analysis & Documentation**
- Write `TAYLOR_GREEN_RESULTS.md`
- Add test to CMakeLists.txt
- Commit to validation branch

---

### Week 2: Lid-Driven Cavity

**Day 1-2: Moving Wall Boundary Condition**
- Implement moving wall BC in `src/physics/fluid/boundary_conditions.cu`
  ```cpp
  __global__ void applyMovingWallBC(float* f, float u_wall, ...);
  // Bounce-back with velocity correction
  ```
- Test BC in isolation (unit test)

**Day 3-4: Re=100 Implementation**
- Set up 129×129 domain
- Tune tau for Re=100
- Run to steady state (monitor u_max convergence)
- Extract centerline profiles

**Day 5: Ghia Comparison**
- Load Ghia reference data
- Compute L∞ and L2 errors
- Generate comparison plots
- Write `LID_DRIVEN_CAVITY_RE100_RESULTS.md`

**Day 6-7: Re=400**
- Adjust tau for Re=400
- Run to steady state (may take longer)
- Compare to Ghia
- Document results

---

### Week 3-4: Grid Convergence & Polishing

**Day 1-3: Grid Convergence**
- Reuse Poiseuille test
- Run on 4 grids (25, 50, 100, 200)
- Compute L2 errors
- Calculate convergence order
- Generate convergence plots

**Day 4: Reynolds Sweep (if time)**
- Implement Re sweep (1, 10, 50, 100, 200)
- Verify u_max vs analytical

**Day 5: Integration & Testing**
- Ensure all tests pass in CI
- Update CMakeLists.txt
- Add labels (`validation;fluid;benchmark;critical`)

**Day 6-7: Documentation**
- Update `VALIDATION_STATUS.md` with new scores
- Write summary report
- Prepare presentation (if needed)

---

## DEPENDENCIES & REQUIREMENTS

### Code Dependencies
- Existing FluidLBM class (`src/physics/fluid/fluid_lbm.cu`)
- Periodic boundary conditions (exists)
- **NEW: Moving wall BC** (need to implement)
- Analytical solution headers (create)

### Reference Data
- Ghia et al. (1982) paper - **obtain digitized data**
- Taylor-Green analytical formula (standard)

### Computational Resources
- Taylor-Green 512²: ~5 min per run (estimate)
- Lid-driven cavity 129²: ~10 min to steady state
- Grid convergence: 4 runs × ~5-10 min each
- **Total compute: ~2-3 hours** (modest)

### Tools
- Python scripts for data extraction and plotting
- ParaView for visual verification (optional)

---

## ACCEPTANCE CRITERIA (PHASE 1 SUCCESS)

### Minimum Requirements (Must Pass)
1. ✅ Taylor-Green decay rate within 5% of analytical
2. ✅ Lid-driven cavity Re=100 centerline velocities within 1% of Ghia
3. ✅ Fluid grid convergence order 1.8 < p < 2.2
4. ✅ All tests pass CI (no NaN, no crashes)

### Stretch Goals (Desired)
5. ✅ Lid-driven cavity Re=400 within 2% of Ghia
6. ✅ Poiseuille Re sweep validates ν formula
7. ✅ Documentation complete (RESULTS.md for each test)

### Quality Metrics
- Code coverage: New BC code has unit tests
- Documentation: Clear headers with references
- Reproducibility: CMake targets, clear instructions

---

## DELIVERABLES CHECKLIST

### Code
- [ ] `tests/validation/test_taylor_green_2d.cu`
- [ ] `tests/validation/test_lid_driven_cavity_re100.cu`
- [ ] `tests/validation/test_lid_driven_cavity_re400.cu`
- [ ] `tests/validation/test_fluid_grid_convergence.cu`
- [ ] `tests/validation/analytical/taylor_green.h`
- [ ] `tests/validation/analytical/ghia_1982_data.h`
- [ ] `src/physics/fluid/boundary_conditions.cu` (moving wall BC)

### Documentation
- [ ] `TAYLOR_GREEN_RESULTS.md`
- [ ] `LID_DRIVEN_CAVITY_RE100_RESULTS.md`
- [ ] `LID_DRIVEN_CAVITY_RE400_RESULTS.md`
- [ ] `FLUID_GRID_CONVERGENCE_RESULTS.md`
- [ ] Update `VALIDATION_STATUS.md` (Fluid score: 55% → 85%)

### Integration
- [ ] CMakeLists.txt updates (add new tests)
- [ ] CI configuration (ensure tests run)
- [ ] Git branch: `feature/fluid-validation` merged to main

---

## RISK MITIGATION

### Risk 1: Moving Wall BC Implementation Difficulty
**Likelihood:** Medium
**Impact:** High (blocks lid-driven cavity)

**Mitigation:**
- Start with moving wall BC **first** (Week 1-2 overlap)
- Unit test BC separately before full cavity
- Reference: Kruger et al. LBM textbook, Section 5.3.3
- Fallback: Use existing no-slip BC with external forcing (less accurate)

---

### Risk 2: Convergence to Steady State Takes Too Long
**Likelihood:** Medium (especially Re=400)
**Impact:** Medium (delays testing)

**Mitigation:**
- Monitor convergence criteria carefully
- Use multigrid initialization (if available)
- Reduce domain size if needed (65×65 also acceptable for Ghia)
- Increase relaxation (higher omega, closer to 1.9 limit)

---

### Risk 3: Ghia Reference Data Not Digitized
**Likelihood:** Low (paper is standard benchmark)
**Impact:** Medium (need data for comparison)

**Mitigation:**
- Ghia data widely available online (CFD benchmarks repository)
- Backup: Use other references (Bruneau & Saad 2006, Erturk et al. 2005)
- Manual digitization (if necessary, ~1 hour effort)

---

### Risk 4: Tests Fail (Fluid Solver Has Bugs)
**Likelihood:** Medium (unknown accuracy currently)
**Impact:** **CRITICAL** (invalidates entire fluid solver)

**Mitigation:**
- **This is WHY we do validation** - finding bugs is success
- If tests fail:
  1. Check BC implementation first (most common error source)
  2. Verify tau-to-ν conversion
  3. Check force implementation (if used)
  4. Debug with simple tests (channel flow, Couette)
- Document all bugs found and fixed
- If fundamental issue found, escalate to team for redesign

---

## SUCCESS METRICS

### Quantitative
- Fluid validation score: **55% → 85%** (target)
- Number of standard benchmarks: **1 → 4** (Poiseuille, T-G, Cavity×2)
- Grid convergence proven: **Thermal only → Thermal + Fluid**

### Qualitative
- **Confidence in melt pool convection:** Low → High
- **Publication readiness:** Not ready → Fluid section ready
- **Team morale:** Validation demonstrates rigor

### Timeline
- Week 1: Taylor-Green complete
- Week 2: Cavity Re=100, Re=400 complete
- Week 3: Grid convergence complete
- Week 4: Documentation, integration, testing

**Overall Success:** 4/4 tests passing within acceptance criteria by 2026-02-07.

---

## PHASE 2 PREVIEW (Weeks 5-8)

After Phase 1 fluid validation, proceed to:

1. **Natural convection** (differentially heated cavity)
   - Validates Ra, Pr, Nu
   - Critical for thermal-fluid coupling

2. **Stefan problem completion** (interface tracking)
   - Validates phase change accuracy

3. **Melt pool benchmark** (Khairallah 2016)
   - **CRITICAL** for AM validation
   - Quantitative D, W, L comparison

4. **Marangoni analytical** (thermocapillary migration)
   - Validates Ma number

**Phase 2 enables publication of multiphysics validation paper.**

---

## RESOURCES & REFERENCES

### Papers to Obtain
1. **Ghia, U., Ghia, K. N., & Shin, C. T. (1982).** "High-Re solutions for incompressible flow using the Navier-Stokes equations and a multigrid method." *Journal of Computational Physics*, 48(3), 387-411.
   - **Essential** for lid-driven cavity comparison

2. **Taylor, G. I., & Green, A. E. (1937).** "Mechanism of the production of small eddies from large ones." *Proceedings of the Royal Society of London A*, 158(895), 499-521.
   - Background for Taylor-Green vortex

3. **He, X., & Luo, L. S. (1997).** "Theory of the lattice Boltzmann method: From the Boltzmann equation to the lattice Boltzmann equation." *Physical Review E*, 56(6), 6811.
   - LBM validation methodology

### Online Resources
- CFD Benchmarks: https://www.cfd-online.com/Wiki/Lid-driven_cavity_problem
- Ghia data tables: Available in supplementary materials or CFD repositories

### LBM Textbooks
- Kruger, T., et al. (2017). *The Lattice Boltzmann Method: Principles and Practice.* Springer.
  - Chapter 5: Boundary conditions (moving wall)
  - Chapter 9: Validation benchmarks

---

## TEAM ROLES (If Applicable)

- **Lead Developer:** Implement tests, moving wall BC
- **Validation Specialist:** Analyze results, compare to references
- **Documentation Lead:** Write RESULTS.md files, update VALIDATION_STATUS.md
- **Code Reviewer:** Ensure code quality, test coverage

---

## CONCLUSION

Phase 1 is a **4-week focused sprint** to close the **critical fluid validation gap**. Success means:
- Fluid solver validated to same standard as thermal (85% confidence)
- Publication-ready fluid dynamics section
- Foundation for Phase 2 multiphysics validation

**The effort is HIGH but ESSENTIAL.** Without this, the entire AM simulation lacks credibility.

**Start Date:** 2026-01-10
**Go/No-Go Review:** 2026-01-17 (after Taylor-Green)
**Target Completion:** 2026-02-07

**Questions? Concerns? Escalate immediately to Chief Architect.**

---

**Next Document:** After Phase 1 completion, create `PHASE_2_MULTIPHYSICS_VALIDATION.md`

**Related Documents:**
- `/home/yzk/LBMProject/docs/VALIDATION_ARCHITECTURE_REVIEW.md` (Full review)
- `/home/yzk/LBMProject/docs/VALIDATION_STATUS.md` (Current status dashboard)
