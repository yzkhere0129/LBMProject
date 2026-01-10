# Thermal LBM Validation Framework: Architectural Design

**Date:** 2025-12-24
**Status:** Architecture Proposal
**Priority:** CRITICAL - Foundation for all physics validation

---

## Executive Summary

The current thermal validation approach (`test_thermal_walberla_match.cu`) achieves 2% agreement with walberla on peak temperature but lacks fundamental rigor. This document provides a comprehensive architectural framework for establishing reliable thermal solver validation.

**Key Finding:** Single-point validation is insufficient. A robust validation framework requires:
1. Analytical benchmarks (mathematical correctness)
2. Grid/timestep convergence studies (discretization error quantification)
3. Boundary condition coverage (implementation verification)
4. Energy conservation tests (fundamental physics)
5. Spatial/temporal error distribution analysis

---

## 1. CRITICAL GAPS IN CURRENT VALIDATION

### 1.1 Existing Test: `test_thermal_walberla_match.cu`

**Strengths:**
- Clean isolation of thermal physics (no multiphysics coupling)
- Exact parameter matching with walberla reference
- Achieves 2% agreement: LBM 4,017 K vs walberla 4,099 K
- Includes FD reference comparison (`FDReferenceComparison` test)

**Critical Deficiencies:**

| Category | Gap | Impact |
|----------|-----|--------|
| **Spatial Coverage** | Only validates peak temperature (single point) | No proof of spatial accuracy away from laser |
| **Temporal Coverage** | Only validates at t=50μs (peak time) | No proof of temporal evolution accuracy |
| **Convergence** | Fixed dx=2μm, dt=100ns (arbitrary) | No proof of grid independence |
| **Error Metrics** | Only L∞ norm (max error) | No RMS error or spatial distribution |
| **Boundary Conditions** | Only Dirichlet BC tested | Adiabatic/radiation BC unvalidated |
| **Analytical Benchmarks** | None | No mathematical proof of correctness |

### 1.2 Fundamental Question: What Does 2% Agreement Mean?

**Current interpretation:** "The solver is accurate"

**Reality check:**
- 2% at peak temperature does NOT guarantee:
  - Correct diffusion away from peak
  - Correct temporal evolution
  - Correct boundary flux
  - Correct energy conservation
  - Grid independence

**Analogy:** A clock showing correct time twice a day doesn't mean it works.

---

## 2. VALIDATION HIERARCHY

### Level 0: Unit Tests (Kernel Correctness)
**Purpose:** Verify individual CUDA kernels work as designed

```
├─ Collision operator (BGK relaxation)
├─ Streaming operator (propagation + bounce-back)
├─ Heat source deposition (addHeatSourceKernel)
├─ Boundary conditions (Dirichlet, adiabatic, radiation)
└─ Temperature computation (moment extraction)
```

**Status:** Partially covered by `test_thermal_walberla_match.cu`
**Gap:** No isolated kernel tests

---

### Level 1: Analytical Benchmarks (Mathematical Correctness)
**Purpose:** Prove the method solves the heat equation correctly

#### **Benchmark 1A: 1D Gaussian Diffusion**

**Problem:**
```
∂T/∂t = α ∂²T/∂x²
T(x,0) = T₀ + ΔT exp(-x²/(2σ₀²))
```

**Analytical solution:**
```
T(x,t) = T₀ + ΔT × (σ₀/σ(t)) × exp(-x²/(2σ(t)²))
where σ(t) = √(σ₀² + 2αt)
```

**Validation metrics:**
- L₂ norm: ||T_LBM - T_analytical||₂ / ||T_analytical||₂ < 1%
- Peak temperature error < 0.1%
- Width evolution error < 1%

**Implementation:**
```cuda
// Domain: 1D slab (nx × 1 × 1)
// BC: Periodic or zero-gradient at far boundaries
// Initial: Gaussian pulse at center
// Run: t = 0 to 10 × σ₀²/α (10 diffusion times)
// Output: Compare T(x,t) vs analytical every 1 diffusion time
```

**Expected convergence:** 2nd order in space (LBM Chapman-Enskog)

**Status:** NOT IMPLEMENTED
**Priority:** CRITICAL

---

#### **Benchmark 1B: 3D Gaussian Diffusion**

**Problem:**
```
∂T/∂t = α ∇²T
T(x,y,z,0) = T₀ + ΔT exp(-(x²+y²+z²)/(2σ₀²))
```

**Analytical solution:**
```
T(r,t) = T₀ + ΔT × (σ₀/σ(t))³ × exp(-r²/(2σ(t)²))
where σ(t) = √(σ₀² + 6αt)  (3D diffusion)
```

**Validation metrics:**
- Radial profile error < 1%
- Total energy conservation: |E(t) - E(0)| / E(0) < 0.1%
- Spherical symmetry error < 0.5%

**Implementation:**
```cuda
// Domain: 100×100×100 cells
// BC: Adiabatic (zero-flux) at all boundaries
// Initial: Gaussian sphere at center
// Run: 5 diffusion times
// Verify: Energy conservation, symmetry, spatial distribution
```

**Status:** Partially in `test_3d_heat_diffusion_senior.cu`
**Gap:** No energy conservation check, no symmetry verification

---

#### **Benchmark 1C: Stefan Problem (1D Phase Change)**

**Problem:** 1D melting of semi-infinite solid with moving boundary

**Setup:**
```
T_left = T_hot > T_melt (constant)
T_initial = T_cold < T_melt
Interface position: s(t) = 2λ√(αt)  (Stefan solution)
```

**Validation metrics:**
- Interface position vs √t (should be linear)
- Temperature profile in liquid/solid regions
- Energy balance across interface

**Implementation:**
```cuda
// Domain: 1D slab (500 × 1 × 1)
// BC: T(x=0) = 1600 K (hot), T(x=L) = 300 K (cold)
// Material: Ti6Al4V (T_solidus=1878K, T_liquidus=1923K)
// Initial: T = 300 K everywhere
// Run: Until interface reaches x = 0.5*L
```

**Expected:** Interface velocity proportional to 1/√t

**Status:** NOT IMPLEMENTED
**Priority:** HIGH (validates phase change coupling)

---

#### **Benchmark 1D: Rosenthal Equation (Moving Heat Source)**

**Problem:** Analytical solution for moving point heat source

**Analytical solution (semi-infinite domain):**
```
T(x,y,z) - T₀ = (P·η)/(2πkr) × exp(-v(r+x)/(2α))
where r = √(x² + y² + z²)
```

**Setup:**
```
Power: P = 200 W
Velocity: v = 0.5 m/s (laser scan speed)
Absorptivity: η = 0.35
Material: Ti6Al4V (k=6.7 W/(m·K), α=2.87e-6 m²/s)
```

**Validation metrics:**
- Temperature profile along scan direction (x-axis)
- Melt pool dimensions (length, width, depth)
- Peak temperature location (should be ahead of source)

**Implementation:**
```cuda
// Domain: 200×200×100 cells (moving reference frame)
// BC: Far-field T = T₀
// Run: Until steady state (temperature stops changing)
// Compare: T(x,y,z) vs Rosenthal analytical
```

**Status:** Mentioned in docs but NOT IMPLEMENTED
**Priority:** HIGH (validates moving heat source for LPBF)

---

### Level 2: Grid/Timestep Convergence Studies

**Purpose:** Quantify discretization error and prove numerical convergence

#### **Convergence Study 2A: Spatial Convergence**

**Test matrix:**
```
dx = [4.0, 2.0, 1.0, 0.5] μm
dt = fixed (maintains CFL stability)
Domain: Fixed physical size (adjust nx, ny, nz)
```

**Metrics:**
```
L₂_error(dx) = ||T_LBM(dx) - T_ref||₂
Convergence rate: log(L₂_error(dx)) / log(dx)
Expected: -2 (second-order method)
```

**Implementation:**
```cuda
// Use Benchmark 1B (3D Gaussian) as reference
// Run 4 grid resolutions
// Plot: log(error) vs log(dx)
// Verify: Slope ≈ -2
```

**Acceptance criteria:**
- Convergence rate > 1.5 (at least 1st order)
- Target: > 1.9 (approaching 2nd order)

**Status:** NOT IMPLEMENTED
**Priority:** CRITICAL

---

#### **Convergence Study 2B: Temporal Convergence**

**Test matrix:**
```
dt = [200, 100, 50, 25] ns
dx = fixed (2 μm)
Domain: Fixed physical size
```

**Expected:** 2nd order temporal accuracy

**Implementation:**
```cuda
// Same as 2A but vary dt instead of dx
// Verify: error ∝ dt²
```

**Status:** NOT IMPLEMENTED
**Priority:** CRITICAL

---

#### **Convergence Study 2C: Combined Richardson Extrapolation**

**Purpose:** Estimate true solution using grid refinement

```
T_exact ≈ (4*T(dx/2) - T(dx)) / 3  (Richardson extrapolation)
```

**Use:** Verify that LBM converges to the correct solution, not just a consistent wrong answer

**Status:** NOT IMPLEMENTED
**Priority:** MEDIUM

---

### Level 3: Boundary Condition Validation

**Purpose:** Verify all BC types work correctly

#### **BC Test 3A: Dirichlet BC (Constant Temperature)**

**Test:**
```
Domain: Hot center (1500 K), cold boundaries (300 K)
BC: T(boundaries) = 300 K (enforced)
Expected: Steady-state with smooth gradient
```

**Validation:**
- Boundary temperature matches prescribed value
- Flux at boundary: q = -k ∂T/∂n (compare with FD)

**Status:** IMPLEMENTED (in `test_thermal_walberla_match.cu`)
**Gap:** No flux validation

---

#### **BC Test 3B: Adiabatic BC (Zero-Flux)**

**Test:**
```
Domain: Hot center (1500 K), insulated boundaries
BC: ∂T/∂n = 0 at boundaries
Expected: Total energy conservation
```

**Validation:**
```
E(t) = ∫∫∫ ρ cp T dV
|E(t) - E(0)| / E(0) < 0.01%  (energy drift < 0.01%)
```

**Status:** NOT IMPLEMENTED
**Priority:** HIGH (required for energy conservation tests)

---

#### **BC Test 3C: Radiation BC (Stefan-Boltzmann)**

**Test:**
```
Domain: Hot plate (T=2000 K) cooling by radiation
BC: q = ε σ (T⁴ - T_ambient⁴)
Expected: Cooling rate follows radiation law
```

**Validation:**
```
dT/dt = -(ε σ A / (ρ V cp)) (T⁴ - T_ambient⁴)
Compare: Numerical dT/dt vs analytical
```

**Status:** Implemented but NOT VALIDATED
**Priority:** MEDIUM

---

### Level 4: Energy Conservation Tests

**Purpose:** Verify fundamental thermodynamic laws

#### **Energy Test 4A: Adiabatic System Energy Balance**

**Setup:**
```
Domain: Isolated system (adiabatic BC on all faces)
Heat source: Q(x,y,z,t) = known function
Expected: E(t) = E(0) + ∫₀ᵗ P_input(t') dt'
```

**Test procedure:**
```cuda
1. Initialize: T = 300 K everywhere
2. Apply laser: P = 200 W, η = 0.35 for 50 μs
3. Compute:
   - E_input = ∫∫∫∫ Q(x,y,z,t) dV dt
   - E_stored = ∫∫∫ ρ cp (T(t) - T(0)) dV
4. Verify: |E_stored - E_input| / E_input < 1%
```

**Expected:** Perfect conservation in adiabatic system

**Status:** Implemented in `test_energy_budget.cu`
**Gap:** Needs stricter tolerance (currently 5%, should be <1%)

---

#### **Energy Test 4B: Boundary Flux Balance**

**Setup:**
```
Domain: With heat loss at boundaries (radiation/convection)
Expected: dE/dt = P_in - P_out
```

**Test:**
```cuda
E(t+dt) - E(t) = P_laser × dt - (P_radiation + P_substrate) × dt
Verify: |LHS - RHS| / P_laser < 2%
```

**Status:** Partially implemented in multiphysics solver
**Gap:** No standalone thermal test

---

### Level 5: Method Comparison

**Purpose:** Cross-validate against other methods

#### **Comparison 5A: LBM vs FD (Same Problem)**

**Status:** IMPLEMENTED in `test_thermal_walberla_match.cu::FDReferenceComparison`
**Result:** <10% error between LBM and FD
**Gap:** No spatial error distribution analysis

---

#### **Comparison 5B: LBM vs Analytical (Known Solutions)**

**Status:** Partially (needs Benchmarks 1A-1D)

---

## 3. RECOMMENDED IMPLEMENTATION ROADMAP

### Phase 1: Analytical Benchmarks (Week 1-2)

**Priority: CRITICAL**

```
Day 1-2:   Implement Benchmark 1A (1D Gaussian)
Day 3-4:   Implement Benchmark 1B (3D Gaussian)
Day 5-7:   Implement Benchmark 1C (Stefan problem)
Day 8-10:  Implement Benchmark 1D (Rosenthal equation)
Day 11-14: Document results and establish baseline
```

**Deliverables:**
- 4 new test files in `tests/validation/analytical/`
- Validation report with error metrics
- Decision: GO/NO-GO for continuing development

---

### Phase 2: Convergence Studies (Week 3)

**Priority: CRITICAL**

```
Day 1-2:  Spatial convergence (dx refinement)
Day 3-4:  Temporal convergence (dt refinement)
Day 5-7:  Richardson extrapolation and error analysis
```

**Acceptance criteria:**
- Spatial convergence rate > 1.5
- Temporal convergence rate > 1.5
- Energy conservation error < 1%

---

### Phase 3: Boundary Condition Coverage (Week 4)

**Priority: HIGH**

```
Day 1-2:  Adiabatic BC with energy conservation
Day 3-4:  Radiation BC with cooling rate validation
Day 5-7:  Mixed BC (Dirichlet + adiabatic + radiation)
```

---

### Phase 4: Enhanced Error Analysis (Week 5)

**Priority: MEDIUM**

```
- Spatial error distribution maps
- Temporal error evolution plots
- Error budget analysis (truncation vs round-off)
- Adaptive mesh refinement recommendations
```

---

## 4. ARCHITECTURAL RECOMMENDATIONS

### 4.1 Test Organization

**Proposed structure:**
```
tests/
├── validation/
│   ├── analytical/
│   │   ├── test_1d_gaussian_diffusion.cu
│   │   ├── test_3d_gaussian_diffusion.cu
│   │   ├── test_stefan_problem.cu
│   │   └── test_rosenthal_equation.cu
│   ├── convergence/
│   │   ├── test_spatial_convergence.cu
│   │   ├── test_temporal_convergence.cu
│   │   └── test_richardson_extrapolation.cu
│   ├── boundary_conditions/
│   │   ├── test_dirichlet_bc.cu
│   │   ├── test_adiabatic_bc.cu
│   │   ├── test_radiation_bc.cu
│   │   └── test_mixed_bc.cu
│   └── energy_conservation/
│       ├── test_adiabatic_energy_balance.cu
│       └── test_boundary_flux_balance.cu
```

### 4.2 Validation Utilities

**Create reusable validation framework:**

```cpp
// include/validation/thermal_validation.h

namespace lbm {
namespace validation {

// Analytical solutions
float gaussian_1d(float x, float t, float alpha, float sigma0);
float gaussian_3d(float r, float t, float alpha, float sigma0);
float stefan_interface(float t, float alpha, float L_fusion);
float rosenthal_temperature(float x, float y, float z,
                            float P, float v, float k, float alpha);

// Error metrics
struct ErrorMetrics {
    float l_inf;     // Maximum error
    float l2;        // RMS error
    float l1;        // Mean absolute error
    float energy;    // Energy conservation error
};

ErrorMetrics computeErrors(const float* T_numerical,
                           const float* T_analytical,
                           int num_cells);

// Convergence analysis
struct ConvergenceRate {
    float spatial_order;   // log(error) / log(dx)
    float temporal_order;  // log(error) / log(dt)
    bool is_converging;    // true if rate > 0.5
};

ConvergenceRate analyzeConvergence(
    const std::vector<float>& dx_values,
    const std::vector<float>& errors);

} // namespace validation
} // namespace lbm
```

### 4.3 Automated Regression Testing

**Add to CI/CD pipeline:**

```yaml
# .github/workflows/validation.yml
validation_suite:
  - analytical_benchmarks:
      tolerance: 1%
      required: true
  - convergence_studies:
      min_order: 1.5
      required: true
  - energy_conservation:
      tolerance: 0.5%
      required: true
```

---

## 5. ACCEPTANCE CRITERIA

### For declaring thermal solver "reliable"

**Must pass ALL of the following:**

| Test Category | Criterion | Current Status |
|---------------|-----------|----------------|
| 1D Gaussian | L₂ error < 1% | ❌ Not implemented |
| 3D Gaussian | L₂ error < 1% | ⚠️ Partial (no error metric) |
| Stefan problem | Interface position error < 2% | ❌ Not implemented |
| Rosenthal | Peak temperature error < 5% | ❌ Not implemented |
| Spatial convergence | Order > 1.5 | ❌ Not implemented |
| Temporal convergence | Order > 1.5 | ❌ Not implemented |
| Energy conservation | Error < 1% (adiabatic) | ⚠️ Partial (5% tolerance) |
| Dirichlet BC | Flux error < 2% | ⚠️ Partial (no flux check) |
| Adiabatic BC | Energy drift < 0.01% | ❌ Not implemented |
| Radiation BC | Cooling rate error < 5% | ❌ Not validated |

**Current score: 1/10 fully validated, 3/10 partially validated**

---

## 6. SPECIFIC IMPLEMENTATION GUIDANCE

### 6.1 Example: 1D Gaussian Diffusion Test

**File:** `tests/validation/analytical/test_1d_gaussian_diffusion.cu`

```cuda
#include <gtest/gtest.h>
#include "physics/thermal_lbm.h"
#include <cmath>
#include <vector>

// Analytical solution for 1D Gaussian diffusion
float gaussian_1d_analytical(float x, float t, float T0, float dT,
                             float sigma0, float alpha) {
    float sigma_t = sqrt(sigma0*sigma0 + 2.0f*alpha*t);
    float amplitude = dT * (sigma0 / sigma_t);
    return T0 + amplitude * exp(-x*x / (2.0f*sigma_t*sigma_t));
}

TEST(ThermalAnalytical, GaussianDiffusion1D) {
    // Setup
    const int nx = 512;  // High resolution for accuracy
    const int ny = 1;
    const int nz = 1;
    const float dx = 1.0e-6f;  // 1 μm
    const float dt = 10.0e-9f; // 10 ns

    // Material (arbitrary for diffusion test)
    const float alpha = 5.0e-6f;  // m²/s
    const float rho = 4430.0f;
    const float cp = 526.0f;
    const float k = alpha * rho * cp;

    // Initial Gaussian
    const float T0 = 300.0f;
    const float dT = 1000.0f;
    const float sigma0 = 50.0e-6f;  // 50 μm
    const float x_center = nx * dx / 2.0f;

    // Initialize LBM solver
    ThermalLBM thermal(nx, ny, nz, alpha, rho, cp, dt, dx);

    std::vector<float> T_initial(nx);
    for (int i = 0; i < nx; ++i) {
        float x = i * dx - x_center;
        T_initial[i] = gaussian_1d_analytical(x, 0.0f, T0, dT, sigma0, alpha);
    }
    thermal.initialize(T_initial.data());

    // Run simulation
    const float t_final = 5.0f * sigma0*sigma0 / (2.0f*alpha);  // 5 diffusion times
    const int num_steps = static_cast<int>(t_final / dt);

    for (int step = 0; step < num_steps; ++step) {
        thermal.applyBoundaryConditions(2);  // Adiabatic
        thermal.computeTemperature();
        thermal.collisionBGK(nullptr, nullptr, nullptr);
        thermal.streaming();
    }

    // Compare with analytical solution
    std::vector<float> T_numerical(nx);
    thermal.copyTemperatureToHost(T_numerical.data());

    std::vector<float> T_analytical(nx);
    for (int i = 0; i < nx; ++i) {
        float x = i * dx - x_center;
        T_analytical[i] = gaussian_1d_analytical(x, t_final, T0, dT, sigma0, alpha);
    }

    // Compute L2 error
    float l2_error = 0.0f;
    float l2_norm = 0.0f;
    for (int i = 0; i < nx; ++i) {
        float diff = T_numerical[i] - T_analytical[i];
        l2_error += diff * diff;
        l2_norm += T_analytical[i] * T_analytical[i];
    }
    l2_error = sqrt(l2_error / nx);
    l2_norm = sqrt(l2_norm / nx);
    float relative_error = l2_error / l2_norm * 100.0f;

    std::cout << "L2 relative error: " << relative_error << "%" << std::endl;

    // Acceptance: <1% error
    EXPECT_LT(relative_error, 1.0f);
}
```

### 6.2 Example: Spatial Convergence Test

**File:** `tests/validation/convergence/test_spatial_convergence.cu`

```cuda
TEST(ThermalConvergence, SpatialOrder) {
    // Test on 3D Gaussian diffusion
    std::vector<float> dx_values = {4.0e-6, 2.0e-6, 1.0e-6, 0.5e-6};  // μm
    std::vector<float> l2_errors;

    for (float dx : dx_values) {
        int nx = static_cast<int>(200.0e-6 / dx);  // 200 μm domain
        // ... run simulation ...
        float error = computeL2Error(T_numerical, T_analytical);
        l2_errors.push_back(error);
    }

    // Fit log(error) = p*log(dx) + c
    float convergence_rate = computeConvergenceRate(dx_values, l2_errors);

    std::cout << "Spatial convergence rate: " << convergence_rate << std::endl;
    std::cout << "Expected: ~2.0 for 2nd-order method" << std::endl;

    // Accept if at least 1st order
    EXPECT_GT(convergence_rate, 1.5f);
}
```

---

## 7. DECISION FRAMEWORK

### Question: "Is the thermal solver reliable?"

**Answer based on validation coverage:**

| Coverage | Confidence Level | Recommendation |
|----------|------------------|----------------|
| < 30% | Low | DO NOT USE in production |
| 30-60% | Moderate | Use with caution, known limitations |
| 60-80% | High | Suitable for research |
| > 80% | Very High | Production-ready |

**Current status: ~20% coverage** (2/10 tests fully validated)

**Recommendation:**
- HALT multiphysics development
- COMPLETE analytical benchmarks (Phase 1)
- ESTABLISH convergence properties (Phase 2)
- THEN proceed with confidence

---

## 8. SUMMARY

### What we currently know:
1. Peak temperature matches walberla within 2% (single point, single time)
2. LBM and FD agree within 10% on a specific test case

### What we DON'T know:
1. Does LBM solve the heat equation correctly? (no analytical proof)
2. What is the spatial accuracy order? (no convergence study)
3. What is the temporal accuracy order? (no convergence study)
4. Do boundary conditions work correctly? (partial validation)
5. Is energy conserved? (partial, 5% tolerance)
6. Does it work for different problems? (only laser heating tested)

### Path forward:
**Implement the 10 validation tests outlined in Section 5.**

Only after passing these tests can we confidently say the thermal solver is "reliable" (可靠).

---

**Architectural Principle:**

> "In computational physics, agreement with one reference case proves nothing.
> Systematic validation against analytical solutions, convergence studies, and
> conservation laws is the ONLY path to reliability."

---

## References

1. He, X., Chen, S., & Doolen, G. D. (1998). A novel thermal model for the lattice Boltzmann method in incompressible limit. *J. Comput. Phys.*, 146(1), 282-300.

2. Mohamad, A. A. (2011). *Lattice Boltzmann Method: Fundamentals and Engineering Applications with Computer Codes*. Springer.

3. Krüger, T., et al. (2017). *The Lattice Boltzmann Method: Principles and Practice*. Springer.

4. Roache, P. J. (1998). Verification of codes and calculations. *AIAA Journal*, 36(5), 696-702.

5. Oberkampf, W. L., & Roy, C. J. (2010). *Verification and Validation in Scientific Computing*. Cambridge University Press.
