# Validation Test Suite Implementation Roadmap

## Executive Summary

This document provides a practical roadmap for implementing the comprehensive validation test suite specified in `TEST_SUITE_DESIGN.md`. It prioritizes tests based on impact, provides effort estimates, and tracks implementation status.

**Objective**: Establish a rigorous validation framework that demonstrates the LBM-CUDA solver achieves:
- Second-order spatial accuracy (p ≈ 2.0)
- Conservation of mass, momentum, and energy (<1% drift)
- Correct physical scaling with dimensionless numbers
- Stable operation within established parameter bounds

---

## Current Status

### Existing Tests (Already Implemented)

| Test | File | Status | Notes |
|------|------|--------|-------|
| Poiseuille Flow | `tests/integration/test_poiseuille_flow.cu` | Good | Has analytical comparison, needs convergence study |
| Stefan Problem | `tests/integration/test_stefan_1d.cu` | Qualitative | Demonstrates phase change, quantitative accuracy needs tuning |
| Energy Conservation | `tests/integration/multiphysics/test_energy_conservation_full.cu` | Excellent | Full balance with laser, evaporation, radiation, substrate |
| VOF Mass Conservation | `tests/unit/vof/test_vof_mass_conservation.cu` | Good | Tracks mass in VOF advection |
| Divergence-Free Test | `tests/validation/test_divergence_free.cu` | Good | Verifies incompressibility |
| Temperature Bounds | `tests/unit/stability/test_temperature_bounds.cu` | Good | Prevents unphysical temperatures |
| Flux Limiter | `tests/unit/stability/test_flux_limiter.cu` | Good | Stabilizes steep gradients |
| High Pe Stability | `tests/integration/stability/test_high_pe_stability.cu` | Good | Tests advection-dominated transport |

### Test Coverage Summary

**What We Have**:
- 1 analytical solution test (Poiseuille)
- 1 conservation law test (energy balance)
- 1 phase change validation (Stefan - qualitative)
- Several stability tests (temperature bounds, flux limiter, high Pe)
- VOF-specific tests (mass conservation, curvature, interface geometry)

**What's Missing**:
- Systematic convergence studies (grid h-refinement, temporal dt-refinement)
- Multiple analytical solution benchmarks (Couette, heat conduction, Taylor-Green)
- Momentum conservation verification
- Entropy production checks
- Dimensionless number scaling tests (Re, Pe, Ma)
- Force balance validation
- Stability boundary characterization

---

## Implementation Phases

### Phase 1: Core Validation (Highest Priority) - 3-4 weeks

These tests establish fundamental solver accuracy and are prerequisites for publishing results.

#### 1.1 Grid Convergence Study - Poiseuille Flow
**File**: `/tests/validation/convergence/test_grid_convergence_poiseuille.cu`

**Effort**: 1-2 days

**Description**: Extend existing `test_poiseuille_flow.cu` to run at multiple resolutions (ny = 16, 32, 64, 128) and measure convergence order.

**Implementation Steps**:
1. Create loop over grid sizes
2. For each size, run Poiseuille simulation to steady state
3. Compute L2 error vs analytical solution
4. Calculate convergence order: `p = log(e1/e2) / log(h1/h2)`
5. Verify `1.8 < p < 2.2` (second-order accuracy)
6. Generate convergence plot (log-log: error vs h)

**Success Criteria**:
- Convergence order p ≈ 2.0 ± 0.2
- Error decreases monotonically with refinement
- Finest grid (ny=128) has error < 2%

**Deliverable**: Test passes, convergence plot exported to `poiseuille_convergence.png`

---

#### 1.2 Heat Conduction - 1D Analytical Validation
**File**: `/tests/validation/analytical/test_heat_conduction_1d.cu`

**Effort**: 2-3 days

**Description**: Validate thermal LBM against exact solution for 1D steady-state heat conduction.

**Implementation Steps**:
1. Set up quasi-1D domain (200×3×3) with fixed boundary temperatures
2. Run thermal solver to steady state
3. Extract temperature profile along x-direction
4. Compare with analytical: `T(x) = T_left + (T_right - T_left) * x/L`
5. Verify heat flux conservation: `q = -k * dT/dx = constant`
6. Test with Ti6Al4V properties

**Success Criteria**:
- L2 error < 4% at steady state
- Temperature gradient uniform (std < 3%)
- Boundary temperatures accurate to ±0.1 K

**Deliverable**: Test passes, temperature profile exported

---

#### 1.3 Momentum Conservation Verification
**File**: `/tests/validation/conservation/test_momentum_conservation.cu`

**Effort**: 2-3 days

**Description**: Verify total momentum changes only due to external forces.

**Test Cases**:
- **Case A**: Periodic domain, no forces → `P(t) = constant`
- **Case B**: Uniform body force → `dP/dt = F * Volume`
- **Case C**: Wall boundaries → measure momentum transfer

**Implementation Steps**:
1. Create momentum calculation utility: `P = sum(rho * u * dV)`
2. For each case, initialize system and apply forces
3. Track momentum evolution over 1000 timesteps
4. Verify momentum balance: `dP/dt = sum(F_external)`
5. Test component-wise (Px, Py, Pz separately)

**Success Criteria**:
- Case A: `||P(t) - P(0)|| / P(0) < 1e-6`
- Case B: `|dP/dt - F*V| / (F*V) < 2%`
- No spurious momentum generation in bulk

**Deliverable**: Three test cases pass, momentum time series exported

---

#### 1.4 Mass Conservation - Detailed Breakdown
**File**: `/tests/validation/conservation/test_mass_conservation_detailed.cu`

**Effort**: 2 days

**Description**: Enhance existing VOF mass conservation test with detailed accounting.

**Test Cases**:
- **Case A**: Closed system (no evaporation) → absolute conservation
- **Case B**: With evaporation → track liquid + vapor mass

**Implementation Steps**:
1. Extend existing `test_vof_mass_conservation.cu`
2. Add evaporated mass tracking
3. Separate local vs global conservation checks
4. Monitor interface cells specifically
5. Test with different VOF advection schemes

**Success Criteria**:
- Case A: `|dm/dt| / m < 1e-4` over 10,000 steps
- Case B: `m_liquid(t) + m_evap(t) = m_initial` within 1%
- No mass loss in bulk cells (only at interfaces)

**Deliverable**: Enhanced test with detailed mass budget reporting

---

#### 1.5 Temporal Convergence - Heat Diffusion
**File**: `/tests/validation/convergence/test_temporal_convergence_heat.cu`

**Effort**: 2 days

**Description**: Verify temporal convergence order for thermal solver.

**Implementation Steps**:
1. Use 1D transient heat diffusion with analytical solution
2. Run with `dt = dt0, dt0/2, dt0/4, dt0/8`
3. Compare at fixed physical time (e.g., t = 0.1 ms)
4. Calculate temporal convergence order
5. Verify p ≈ 1.0 (first-order time integration)

**Success Criteria**:
- Temporal order p ≈ 1.0 ± 0.15
- Error decreases with smaller dt
- All tested dt values are stable

**Deliverable**: Temporal convergence plot (error vs dt)

---

**Phase 1 Summary**:
- **Total Effort**: 3-4 weeks
- **Tests Delivered**: 5 new validation tests
- **Impact**: Establishes core solver accuracy (spatial order, temporal order, conservation)
- **Milestone**: Can confidently state "Second-order spatial accuracy demonstrated" in publications

---

### Phase 2: Extended Validation (Medium Priority) - 4-5 weeks

These tests expand validation coverage and support advanced physics claims.

#### 2.1 Couette Flow Validation
**File**: `/tests/validation/analytical/test_couette_flow.cu`

**Effort**: 1-2 days

**Description**: Validate moving wall boundary condition and shear-driven flow.

**Success Criteria**: L2 error < 3%, linear velocity profile

---

#### 2.2 Taylor-Green Vortex
**File**: `/tests/validation/analytical/test_taylor_green_vortex.cu`

**Effort**: 3-4 days

**Description**: Classic vorticity decay test with analytical solution.

**Success Criteria**:
- Velocity field L2 error < 6%
- Kinetic energy decay rate error < 5%
- Enstrophy conservation verified

---

#### 2.3 Reynolds Number Scaling
**File**: `/tests/validation/scaling/test_reynolds_number_scaling.cu`

**Effort**: 4-5 days

**Description**: Verify flow behavior depends only on Re, not individual u, nu.

**Test Cases**:
- Drag on sphere: compare Cd vs Re to theory
- Vortex shedding: measure Strouhal number

**Success Criteria**:
- Drag coefficient within 8% of theory
- Strouhal number within 10% for cylinder

---

#### 2.4 Peclet Number Consistency
**File**: `/tests/validation/scaling/test_peclet_number_scaling.cu`

**Effort**: 3-4 days

**Description**: Validate thermal transport scaling with Pe = U*L/alpha.

**Test Case**: Heated sphere in flow at Pe = 0.1, 1, 10, 100

**Success Criteria**: Nusselt number follows correlations within 10%

---

#### 2.5 Combined Space-Time Convergence
**File**: `/tests/validation/convergence/test_combined_convergence_taylor_green.cu`

**Effort**: 2-3 days

**Description**: Simultaneous h and dt refinement for Taylor-Green vortex.

**Success Criteria**: Overall error reduces with both refinements

---

**Phase 2 Summary**:
- **Total Effort**: 4-5 weeks
- **Tests Delivered**: 5 additional validation tests
- **Impact**: Demonstrates broad solver capability (various flows, thermal coupling, scaling laws)
- **Milestone**: Comprehensive validation comparable to top CFD codes

---

### Phase 3: Advanced Tests (Lower Priority) - 3-4 weeks

These tests are valuable for research-grade validation but not critical for initial deployment.

#### 3.1 Entropy Production Check
**File**: `/tests/validation/conservation/test_entropy_production.cu`

**Effort**: 4-5 days

**Description**: Verify second law of thermodynamics (dS_universe/dt ≥ 0).

**Complexity**: Requires careful entropy calculation including surroundings.

---

#### 3.2 Mach Number Limit Study
**File**: `/tests/validation/scaling/test_mach_number_limit.cu`

**Effort**: 2 days

**Description**: Characterize compressibility errors as Ma increases.

**Success Criteria**: Density fluctuations scale as Ma^2

---

#### 3.3 Force Balance in Melt Pool
**File**: `/tests/validation/scaling/test_force_balance_melt_pool.cu`

**Effort**: 5-6 days

**Description**: Verify force equilibrium in multiphysics melt pool simulation.

**Complexity**: Couples surface tension, Marangoni, recoil, buoyancy, viscous forces.

---

#### 3.4 Stability Boundary Characterization
**Files**: Various in `/tests/validation/stability/`

**Effort**: 3-4 days total

**Description**: Map out stability boundaries (max velocity, max temperature gradient, CFL limits).

**Deliverable**: Stability map in parameter space

---

**Phase 3 Summary**:
- **Total Effort**: 3-4 weeks
- **Tests Delivered**: 4+ advanced tests
- **Impact**: Research-grade validation for specialized applications
- **Milestone**: Publication-ready validation suite comparable to literature standards

---

## Estimated Timeline

```
Week 1-2:   Phase 1.1-1.2  (Grid convergence, 1D heat conduction)
Week 3-4:   Phase 1.3-1.4  (Momentum, Mass conservation)
Week 5:     Phase 1.5      (Temporal convergence)
            --- Phase 1 Complete: Core Validation ---

Week 6-7:   Phase 2.1-2.2  (Couette, Taylor-Green)
Week 8-9:   Phase 2.3-2.4  (Reynolds, Peclet scaling)
Week 10:    Phase 2.5      (Combined convergence)
            --- Phase 2 Complete: Extended Validation ---

Week 11-13: Phase 3.1-3.3  (Entropy, Mach, Force balance)
Week 14:    Phase 3.4      (Stability boundaries)
            --- Phase 3 Complete: Advanced Tests ---

Total: 14 weeks (3.5 months)
```

**Parallel Development Note**: Multiple tests can be developed simultaneously by different team members. With 2-3 developers, Phase 1 could complete in 2 weeks, total project in 6-8 weeks.

---

## Test Infrastructure Setup

Before implementing tests, ensure infrastructure is in place:

### Directory Structure
```
/tests/validation/
├── analytical/          # Analytical solution tests
│   ├── test_poiseuille_flow.cu
│   ├── test_couette_flow.cu
│   ├── test_heat_conduction_1d.cu
│   └── test_taylor_green_vortex.cu
├── conservation/        # Conservation law tests
│   ├── test_mass_conservation_detailed.cu
│   ├── test_momentum_conservation.cu
│   ├── test_energy_conservation_detailed.cu (exists)
│   └── test_entropy_production.cu
├── convergence/         # Convergence studies
│   ├── test_grid_convergence_poiseuille.cu
│   ├── test_temporal_convergence_heat.cu
│   └── test_combined_convergence_taylor_green.cu
├── scaling/            # Dimensionless number tests
│   ├── test_reynolds_number_scaling.cu
│   ├── test_peclet_number_scaling.cu
│   ├── test_mach_number_limit.cu
│   └── test_force_balance_melt_pool.cu
└── stability/          # Stability tests (some exist)
    ├── test_max_stable_velocity.cu
    ├── test_max_temperature_gradient.cu
    ├── test_cfl_condition_coupling.cu
    └── test_vof_subcycling_stability.cu
```

### CMake Integration
```cmake
# tests/validation/CMakeLists.txt
add_subdirectory(analytical)
add_subdirectory(conservation)
add_subdirectory(convergence)
add_subdirectory(scaling)
add_subdirectory(stability)

# Master validation target
add_custom_target(validation_suite
    COMMAND ctest --output-on-failure -R "validation_.*"
    COMMENT "Running complete validation test suite"
)

# Quick validation (Phase 1 only)
add_custom_target(validation_core
    COMMAND ctest --output-on-failure -R "validation_(analytical_poiseuille|conservation_mass|conservation_momentum|convergence_grid)"
    COMMENT "Running core validation tests"
)
```

### Utility Library
Create shared utilities in `/tests/validation/utils/`:

**Files**:
- `validation_utils.h` - Error metrics, profile extraction
- `analytical_solutions.h` - Implementations of analytical solutions
- `convergence_analysis.h` - Convergence order calculation
- `conservation_checks.h` - Mass, momentum, energy accounting
- `test_reporting.h` - Standardized output formatting

---

## Success Metrics

### Per-Test Metrics
Each test should report:
- **Pass/Fail Status**: Based on tolerance criteria
- **Error Magnitude**: L2, Linf, average relative error
- **Comparison to Tolerance**: How much margin exists
- **Runtime**: Execution time on reference hardware
- **Output Files**: Profiles, plots, data for post-processing

### Suite-Level Metrics
Overall validation suite is successful when:
- **Coverage**: ≥90% of tests pass
- **Accuracy**: Mean L2 error across analytical tests < 5%
- **Conservation**: All conserved quantities within <1% drift
- **Convergence**: Measured orders match theory (p ≈ 2 spatial, p ≈ 1 temporal)
- **Scaling**: Dimensionless number tests within 10% of theory
- **Stability**: No crashes, all simulations complete

### Publication Readiness Checklist
Before submitting results for publication:
- [ ] Phase 1 tests (Core Validation) all pass
- [ ] At least 3 analytical solution tests implemented and passing
- [ ] Grid convergence study demonstrates p ≈ 2.0
- [ ] All conservation laws verified (<1% error)
- [ ] Energy balance matches theory within 5%
- [ ] Validation results documented in manuscript methods section
- [ ] Convergence plots included in supplementary material
- [ ] Test suite publicly available (GitHub) for reproducibility

---

## Risk Assessment and Mitigation

### Risk: Tests Fail to Meet Tolerances

**Likelihood**: Medium (first implementation often reveals solver issues)

**Impact**: High (delays validation, questions solver correctness)

**Mitigation**:
1. Start with simplest tests (1D problems) to isolate issues
2. Use debug builds with extensive checking
3. Validate solver components individually before integration tests
4. Compare with waLBerla or other established LBM codes
5. Adjust tolerances based on realistic LBM accuracy (5-6% is expected)

---

### Risk: Convergence Studies Show Wrong Order

**Likelihood**: Low-Medium

**Impact**: Critical (indicates fundamental discretization error)

**Mitigation**:
1. Verify boundary condition implementation (major source of order degradation)
2. Check force application scheme (Guo vs simple, exact difference)
3. Test with different collision operators (BGK vs MRT)
4. Ensure boundary cells are excluded from error calculation
5. Run on multiple grid sizes to confirm asymptotic regime

---

### Risk: Conservation Laws Violated

**Likelihood**: Medium (multiphysics coupling can introduce errors)

**Impact**: High (physics correctness questioned)

**Mitigation**:
1. Test each physics module separately first
2. Verify operator splitting is conservative
3. Check boundary fluxes are properly accounted
4. Use double precision for accumulation
5. Implement flux correction schemes if needed

---

### Risk: Insufficient GPU Memory for Fine Grids

**Likelihood**: Low (modern GPUs have 8-24 GB)

**Impact**: Medium (limits maximum grid size for convergence studies)

**Mitigation**:
1. Use quasi-1D or 2D domains for analytical tests (memory efficient)
2. Implement domain decomposition for multi-GPU
3. Run finest grids on workstation/cluster GPUs
4. Focus on 3D tests at moderate resolutions (128^3 is usually sufficient)

---

## Continuous Integration Setup

### Automated Testing
**Strategy**: Run core validation suite on every PR to main branch.

**CI Configuration** (GitHub Actions / GitLab CI):
```yaml
name: Validation Tests

on: [push, pull_request]

jobs:
  validation:
    runs-on: self-hosted-gpu  # Requires GPU runner
    steps:
      - uses: actions/checkout@v2
      - name: Build
        run: |
          mkdir build && cd build
          cmake -DCMAKE_BUILD_TYPE=Release ..
          make -j8
      - name: Run Core Validation
        run: |
          cd build
          ctest --output-on-failure -R "validation_core"
      - name: Upload Results
        uses: actions/upload-artifact@v2
        with:
          name: validation-results
          path: build/test_results/
```

### Nightly Testing
Run full validation suite (Phases 1-3) nightly on main branch:
- Longer timeout (1 hour)
- More comprehensive tests
- Generate HTML report with plots
- Email results to team

---

## Documentation and Reporting

### Test Documentation
Each test file should include:
```cpp
/**
 * @file test_name.cu
 * @brief One-line description
 *
 * Physics Description:
 * - Problem setup and geometry
 * - Governing equations
 * - Physical parameters
 *
 * Analytical Solution:
 * - Exact solution formula
 * - Derivation reference (textbook, paper)
 *
 * Validation Criteria:
 * - Error tolerances and rationale
 * - Expected convergence order
 * - Physical constraints verified
 *
 * References:
 * - [1] Author, "Title", Journal (Year)
 */
```

### Validation Report Generation
**Script**: `/scripts/generate_validation_report.py`

**Functionality**:
- Parse all test output files
- Generate summary table (test, status, error, tolerance)
- Create convergence plots (matplotlib)
- Compile into PDF report (LaTeX) or HTML dashboard

**Example Output**:
```
Validation Test Suite Report
=============================
Generated: 2025-12-02 14:30:00

Summary:
  Total Tests: 18
  Passed: 17 (94%)
  Failed: 1 (6%)
  Average Error: 3.2%

Core Validation (Phase 1):
  ✓ Grid Convergence - Poiseuille: p = 1.98, PASS
  ✓ Heat Conduction 1D: L2 = 3.1%, PASS
  ✓ Momentum Conservation: error = 1.5%, PASS
  ✓ Mass Conservation: drift < 0.01%, PASS
  ✓ Temporal Convergence: p = 1.02, PASS

Extended Validation (Phase 2):
  ✓ Couette Flow: L2 = 2.8%, PASS
  ✓ Taylor-Green Vortex: L2 = 5.4%, PASS
  ✗ Reynolds Scaling: Cd error = 12%, FAIL (threshold: 8%)
  ...

[Plots: Convergence curves, profile comparisons, error distributions]
```

---

## Maintenance and Evolution

### Test Suite Maintenance
**Frequency**: Continuous

**Activities**:
- Monitor test pass rates on CI
- Update tolerances if solver improvements warrant
- Add regression tests when bugs are fixed
- Keep analytical solution implementations up-to-date

### Adding New Tests
**Process**:
1. Define physics problem and analytical solution
2. Document in `TEST_SUITE_DESIGN.md`
3. Implement following patterns in `TEST_IMPLEMENTATION_GUIDE.md`
4. Add to CMake build system
5. Run locally and verify passing
6. Submit PR with test and documentation
7. Review for scientific correctness and code quality

### Benchmark Updates
**Trigger**: Major solver changes (new collision operator, boundary conditions)

**Action**: Re-run full validation suite (Phases 1-3) and update baseline results.

---

## Key Resources

### Internal Documentation
- `/docs/TEST_SUITE_DESIGN.md` - Test specifications
- `/docs/TEST_IMPLEMENTATION_GUIDE.md` - Implementation patterns
- `/docs/VALIDATION_ROADMAP.md` - This document

### External References
1. **Krüger et al.**, "The Lattice Boltzmann Method: Principles and Practice" (2017)
   - Chapter 8: Validation and Verification
2. **AIAA CFD Verification and Validation** (2022)
   - Standards for computational fluid dynamics
3. **waLBerla Documentation** - Example validation test suite
4. **Palabos Benchmark Suite** - Reference LBM validation

### Code Examples
- Existing tests in `/tests/integration/`, `/tests/validation/`
- Utility functions in physics modules (`physics/thermal_lbm.h`, `physics/fluid_lbm.h`)

---

## Conclusion

This roadmap provides a structured approach to implementing a comprehensive validation test suite for the LBM-CUDA framework. By following the phased implementation plan, the team will:

1. **Establish credibility**: Second-order accuracy, conservation laws verified
2. **Enable publications**: Rigorous validation matching literature standards
3. **Prevent regressions**: Automated testing catches solver errors early
4. **Support users**: Clear validation results build confidence in the code

**Recommended Start**: Begin with Phase 1 (Core Validation). These 5 tests are high-impact and achievable in 3-4 weeks. Once Phase 1 is complete, the solver has a solid validation foundation suitable for initial publications and applications.

**Next Steps**:
1. Review this roadmap with the team
2. Assign Phase 1 tests to developers
3. Set up test infrastructure (directories, CMake, CI)
4. Begin implementation following `TEST_IMPLEMENTATION_GUIDE.md`
5. Schedule weekly validation meetings to track progress

---

**Document Version**: 1.0
**Date**: 2025-12-02
**Author**: LBM-CUDA Architecture Team
**Status**: Active Roadmap
