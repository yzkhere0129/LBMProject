# VOF Solver Testing - Complete Documentation Index

**Generated:** 2025-12-03
**Project:** LBM-CUDA CFD Framework for Metal AM
**Module:** Volume of Fluid (VOF) Solver

---

## Documentation Structure

### 1. Primary Specification Document
**File:** `VOF_TEST_SPECS.md` (46 KB, 1637 lines)
**Path:** `/home/yzk/LBMProject/docs/VOF_TEST_SPECS.md`

**Contents:**
- Complete test specifications for 12 comprehensive VOF tests
- Detailed physical descriptions and analytical solutions
- Implementation guidelines with code examples
- Validation metrics and acceptance criteria
- References to scientific literature

**Sections:**
1. Test 1: Zalesak's Disk (Advection Accuracy) - Line 28
2. Test 2: Spherical Droplet (Laplace Pressure) - Line 177
3. Test 3: Oscillating Droplet (Dynamic Surface Tension) - Line 326
4. Test 4: Thermocapillary Migration (Marangoni Effect) - Line 475
5. Test 5: Contact Angle (Wall Wetting) - Line 624
6. Test 6: Evaporation Mass Loss (VOF-Thermal Coupling) - Line 773
7. Test 7: Recoil Pressure (Keyhole Formation) - Line 872
8. Test 8: Interface Reconstruction (PLIC) - Line 1021
9. Test 9: Curvature Calculation - Line 1170
10. Test 10: Mass Conservation (Long-Term) - Line 1269
11. Test 11: CFL Stability - Line 1368
12. Test 12: Marangoni Force Verification - Line 1517
13. Summary Table - Line 1616
14. Testing Workflow - Line 1626
15. Implementation Priorities - Line 1647
16. References - Line 1665

---

### 2. Quick Reference Guide
**File:** `VOF_TEST_QUICK_REFERENCE.md` (12 KB, 373 lines)
**Path:** `/home/yzk/LBMProject/docs/VOF_TEST_QUICK_REFERENCE.md`

**Contents:**
- Test overview matrix (status, priority, runtime)
- One-paragraph summaries for each test
- Material properties reference
- Execution commands
- Implementation roadmap
- Common patterns and templates

**Use Cases:**
- Daily reference during implementation
- Quick lookup for test parameters
- Command-line test execution
- Progress tracking

---

## Test Coverage Overview

### Physical Phenomena Covered
1. **Advection:** Interface transport in complex flows (Tests 1, 10)
2. **Surface Tension:** Capillary forces and pressure jumps (Tests 2, 3, 9)
3. **Marangoni Effect:** Thermocapillary flow and migration (Tests 4, 12)
4. **Evaporation:** Mass loss and VOF-thermal coupling (Test 6)
5. **Recoil Pressure:** High-temperature vapor forces (Test 7)
6. **Contact Angle:** Wall wetting and boundary conditions (Test 5)
7. **Interface Geometry:** Normal reconstruction and curvature (Tests 8, 9)
8. **Numerical Stability:** CFL limits and conservation (Tests 10, 11)

### Test Types
- **Unit Tests:** 8, 9, 10, 6, 12 (isolated component validation)
- **Validation Tests:** 1, 2, 3, 4, 5, 7, 11 (physics benchmarks)
- **Integration Tests:** All tests verify coupling between components

### Implementation Status (as of 2025-12-03)
- **Implemented:** 6 tests (Tests 6, 8, 9, 10, 11, 12)
- **To Implement:** 6 tests (Tests 1, 2, 3, 4, 5, 7)
- **Coverage:** 50% complete, 50% specified

---

## Quick Start Guide

### For Test Implementers
1. Read test specification in `VOF_TEST_SPECS.md`
2. Use template from Quick Reference
3. Implement test in `/home/yzk/LBMProject/tests/validation/vof/`
4. Register in CMakeLists.txt
5. Run and validate results

### For Test Users
1. Check Quick Reference for test list
2. Run specific test: `./tests/validation/vof/test_name`
3. Interpret results against acceptance criteria
4. Report pass/fail status

### For Project Managers
1. Review implementation roadmap (Phase 1-4)
2. Track progress using test status matrix
3. Monitor deliverables (test files, reports, visualizations)
4. Estimated timeline: 15-20 days for full suite

---

## Test File Locations

### Existing Tests (Implemented)
```
/home/yzk/LBMProject/tests/unit/vof/
  - test_vof_reconstruction.cu         (Test 8)
  - test_vof_curvature.cu              (Test 9 - partial)
  - test_vof_mass_conservation.cu      (Test 10)
  - test_vof_evaporation_mass_loss.cu  (Test 6)
  - test_vof_marangoni.cu              (Test 12)
  - test_vof_advection.cu              (advection unit tests)
  - test_vof_contact_angle.cu          (Test 5 - partial)

/home/yzk/LBMProject/tests/validation/vof/
  - test_vof_curvature_sphere.cu       (Test 9 - sphere case)
  - test_vof_curvature_cylinder.cu     (Test 9 - cylinder case)
  - test_vof_advection_rotation.cu     (rotation test)

/home/yzk/LBMProject/tests/validation/
  - test_cfl_stability.cu              (Test 11)
```

### New Tests (To Implement)
```
/home/yzk/LBMProject/tests/validation/vof/
  - test_vof_zalesak_disk.cu                      (Test 1)
  - test_vof_laplace_pressure.cu                  (Test 2)
  - test_vof_oscillating_droplet.cu               (Test 3)
  - test_vof_thermocapillary_migration.cu         (Test 4)
  - test_vof_contact_angle_static.cu              (Test 5)
  - test_vof_recoil_pressure_depression.cu        (Test 7)
```

---

## Key Metrics and Tolerances

| Test | Key Metric | Tolerance | Physical Constant |
|------|------------|-----------|-------------------|
| 1 | Shape error | < 5% | - |
| 2 | Pressure jump | < 15% | ΔP = 2σ/R |
| 3 | Oscillation frequency | < 15% | f = √(8σ/ρR³)/(2π) |
| 4 | Migration velocity | < 30% | U = (2/3)|dσ/dT|∇TR/μ |
| 5 | Contact angle | < 5° | Young's equation |
| 6 | Mass loss rate | < 10% | df/dt = -J/(ρdx) |
| 7 | Recoil pressure | < 5% | P = 0.54×P_sat(T) |
| 8 | Angular error | < 5° | n = -∇f/|∇f| |
| 9 | Curvature | < 10% | κ = 2/R (sphere) |
| 10 | Mass variation | < 1% | M = Σf_i = const |
| 11 | Bound violations | 0 | CFL < 0.5 |
| 12 | Force magnitude | < 20% | F = (dσ/dT)∇_sT|∇f| |

---

## Material Properties Reference

### Ti6Al4V (Titanium Alloy)
Used in all tests for consistency with LPBF application.

**Thermophysical Properties:**
```
Density (liquid):             ρ_l = 4110 kg/m³
Density (solid):              ρ_s = 4420 kg/m³
Surface tension (2000 K):     σ = 1.5 N/m
Surface tension gradient:     dσ/dT = -0.26e-3 N/(m·K)
Dynamic viscosity:            μ = 0.003 - 0.005 Pa·s
Melting temperature:          T_melt = 1923 K
Boiling temperature:          T_boil = 3560 K
Latent heat (vaporization):   L_vap = 8.878e6 J/kg
Latent heat (fusion):         L_fus = 2.86e5 J/kg
Molar mass:                   M = 0.0479 kg/mol
Thermal conductivity:         κ = 35 W/(m·K)
Specific heat:                c_p = 670 J/(kg·K)
```

**Derived Scales:**
```
Capillary velocity:           U_cap = σ/μ = 300-500 m/s
Capillary time (R=10μm):      t_cap = ρR²/σ = 112 ns
Capillary length:             l_cap = √(σ/ρg) = 620 μm
Reynolds number (typical):    Re = ρUR/μ ≈ 1-100
Weber number (typical):       We = ρU²R/σ ≈ 0.01-1
Bond number (R=10μm):         Bo = ρgR²/σ ≈ 2.7e-4
```

---

## Implementation Phases

### Phase 1: Core Numerical Properties (2-3 days)
**Priority:** CRITICAL
**Tests:** 8, 9, 10
**Goal:** Verify fundamental VOF correctness

**Deliverables:**
- Interface normal accuracy: angular error < 5°
- Curvature accuracy: relative error < 10%
- Mass conservation: drift < 1%

**Rationale:** These tests validate the numerical foundation. All other tests depend on correct interface reconstruction, curvature, and mass conservation.

---

### Phase 2: Surface Tension Physics (4-5 days)
**Priority:** HIGH
**Tests:** 2, 1, 12, 6
**Goal:** Validate capillary forces

**Deliverables:**
- Laplace pressure benchmark
- Zalesak's disk benchmark
- Marangoni force verification
- Evaporation coupling

**Rationale:** Surface tension is the dominant force in LPBF melt pools. These tests ensure correct force application.

---

### Phase 3: Advanced Interfacial Dynamics (5-7 days)
**Priority:** MEDIUM
**Tests:** 4, 7, 3
**Goal:** Complex physical phenomena

**Deliverables:**
- Thermocapillary migration
- Recoil pressure (keyhole)
- Dynamic oscillations

**Rationale:** These tests validate advanced physics needed for high-fidelity LPBF simulations.

---

### Phase 4: Boundary Conditions & Robustness (3-4 days)
**Priority:** MEDIUM-LOW
**Tests:** 5, 11
**Goal:** Wall interactions and stability

**Deliverables:**
- Contact angle validation
- CFL stability analysis

**Rationale:** These tests ensure robustness for production simulations with complex geometries.

---

## Usage Examples

### Run Single Test
```bash
cd /home/yzk/LBMProject/build
./tests/validation/vof/test_vof_laplace_pressure
```

### Run All VOF Tests
```bash
ctest -R vof -V
```

### Run Quick Smoke Test (< 5 min)
```bash
./tests/unit/vof/test_vof_reconstruction
./tests/unit/vof/test_vof_curvature
./tests/unit/vof/test_vof_marangoni
```

### Run Full Validation Suite (30-60 min)
```bash
ctest -R "vof|validation" -V > vof_test_results.log 2>&1
```

### Generate Test Report
```bash
python3 scripts/analyze_test_results.py vof_test_results.log --output VOF_TEST_REPORT.md
```

---

## Expected Outputs

### For Each Test
1. **Console Output:**
   - Test name and description
   - Setup parameters
   - Progress updates
   - Measured vs expected values
   - Pass/fail status

2. **VTK Files (Optional):**
   - Initial condition
   - Final state
   - Time series (for dynamic tests)
   - Visualization-ready format

3. **Validation Report:**
   - Test summary
   - Metrics table
   - Error analysis
   - Plots (if applicable)

---

## Scientific References

### VOF Method
1. Hirt, C.W. & Nichols, B.D. (1981). "Volume of fluid (VOF) method for the dynamics of free boundaries." *Journal of Computational Physics*, 39(1), 201-225.

### Surface Tension (CSF)
2. Brackbill, J.U., Kothe, D.B., & Zemach, C. (1992). "A continuum method for modeling surface tension." *Journal of Computational Physics*, 100(2), 335-354.

### Benchmark Tests
3. Zalesak, S.T. (1979). "Fully multidimensional flux-corrected transport algorithms for fluids." *Journal of Computational Physics*, 31(3), 335-362.

### Thermocapillary Flow
4. Young, N.O., Goldstein, J.S., & Block, M.J. (1959). "The motion of bubbles in a vertical temperature gradient." *Journal of Fluid Mechanics*, 6(3), 350-356.

### Recoil Pressure
5. Anisimov, S.I. (1968). "Vaporization of metal absorbing laser radiation." *Soviet Physics JETP*, 27, 182-183.
6. Khairallah, S.A. et al. (2016). "Laser powder-bed fusion additive manufacturing: Physics of complex melt flow and formation mechanisms of pores, spatter, and denudation zones." *Acta Materialia*, 108, 36-45.

### Contact Angle
7. Young, T. (1805). "An essay on the cohesion of fluids." *Philosophical Transactions of the Royal Society of London*, 95, 65-87.

---

## Contact and Contribution

### Questions or Issues
- Check test specification document first
- Review existing test implementations for patterns
- Consult project documentation in `/home/yzk/LBMProject/docs/`

### Contributing New Tests
1. Follow test template in Quick Reference
2. Document analytical solution
3. Include validation metrics
4. Add to CMakeLists.txt
5. Update this index

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2025-12-03 | Initial comprehensive test specification |

---

**Total Documentation:** 2010 lines, 58 KB
**Estimated Implementation Time:** 15-20 days (all 12 tests)
**Current Status:** 6/12 tests implemented (50%)
