# Phase 2 Multi-Physics Validation Plan
# LBM-CUDA Platform

**Document Version:** 1.0
**Created:** 2026-01-10
**Author:** LBM-CUDA Chief Architect
**Purpose:** Comprehensive validation plan for Phase 2 benchmarks

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Current Implementation Status](#2-current-implementation-status)
3. [Benchmark 1: Natural Convection (Rayleigh-Benard)](#3-benchmark-1-natural-convection)
4. [Benchmark 2: Stefan Problem (Phase Change)](#4-benchmark-2-stefan-problem)
5. [Benchmark 3: Melt Pool Validation (Khairallah 2016)](#5-benchmark-3-melt-pool-validation)
6. [Benchmark 4: Marangoni Analytical Validation](#6-benchmark-4-marangoni-analytical-validation)
7. [Implementation Priority and Timeline](#7-implementation-priority-and-timeline)
8. [Appendix: Literature References](#8-appendix-literature-references)

---

## 1. Executive Summary

### 1.1 Phase 2 Validation Goals

Phase 2 focuses on validating **multi-physics coupling** through four critical benchmarks:

| Benchmark | Physics Validated | Acceptance Criteria | Literature Reference |
|-----------|------------------|---------------------|---------------------|
| **Natural Convection** | Thermal-Fluid Coupling | Ra, Nu match ±10% | Davis (1983), de Vahl Davis (1983) |
| **Stefan Problem** | Phase Change Interface Tracking | Interface position error < 5% | Crank (1984) |
| **Melt Pool Size** | Full Multi-Physics | Pool dimensions match ±15% | Khairallah et al. (2016) |
| **Marangoni Flow** | Surface Tension Gradient | Velocity profile shape match | Young et al. (1959) |

### 1.2 Current Implementation Readiness

**Strengths:**
- ✅ All core solvers implemented (Thermal LBM, Fluid LBM, VOF, Phase Change)
- ✅ Thermal validation complete (vs waLBerla FD reference)
- ✅ Marangoni velocity magnitude validated (0.7-0.8 m/s range)
- ✅ Multiphysics coupling operational

**Gaps Identified:**
- ⚠️ **Natural convection:** Not yet validated quantitatively (Ra, Pr, Nu)
- ⚠️ **Stefan problem:** Test exists but DISABLED due to known issues
- ⚠️ **Buoyancy forces:** Implemented but not validated against benchmark
- ⚠️ **Full LPBF simulation:** No direct comparison with Khairallah 2016 data

**Priority Actions:**
1. Enable and fix Stefan problem validation (1-2 days)
2. Implement natural convection benchmark (2-3 days)
3. Set up Khairallah melt pool comparison (3-4 days)
4. Validate Marangoni analytical profile (1-2 days)

---

## 2. Current Implementation Status

### 2.1 Solver Status Matrix

| Module | Status | Validation Level | Test Files |
|--------|--------|-----------------|------------|
| **Thermal LBM (D3Q7)** | ✅ Production | Quantitative | `test_3d_heat_diffusion_senior.cu` |
| **Fluid LBM (D3Q19)** | ✅ Production | Qualitative | `test_lid_driven_cavity_re100.cu` |
| **VOF Solver** | ✅ Production | Qualitative | `test_vof_advection_rotation.cu` |
| **Phase Change** | ✅ Implemented | Disabled Tests | `test_stefan_problem.cu` (DISABLED) |
| **Marangoni** | ✅ Production | Quantitative | `test_marangoni_velocity.cu` |
| **Buoyancy** | ✅ Implemented | Not Validated | None (needs benchmark) |
| **Multiphysics** | ✅ Operational | Integration Only | 35 integration tests |

### 2.2 Known Issues from Test Analysis

#### 2.2.1 Stefan Problem (test_stefan_problem.cu)

**Issue:** All tests DISABLED with comment:
```cpp
// NOTE: This test is DISABLED due to known physics limitation.
// Current implementation uses temperature-based phase change.
// Stefan problem requires enthalpy-based advection for accuracy.
```

**Root Cause Analysis:**
- Current phase change uses temperature-based liquid fraction: `fl(T)`
- Stefan problem requires enthalpy conservation: `H = ρcp·T + fl·ρL`
- Interface velocity `v_interface = q/(ρL)` depends on heat flux at interface
- Current implementation: Interface tracking via VOF, but no explicit enthalpy advection

**Expected Error:** 50-150% interface position error (acknowledged in test)

**Fix Strategy:**
1. Verify enthalpy method implementation in `phase_change.cu`
2. Ensure `updateTemperatureFromEnthalpy()` is called after thermal advection
3. Add diagnostic: check if latent heat is correctly absorbed/released
4. Validate Newton-Raphson convergence in mushy zone

#### 2.2.2 Marangoni Velocity (test_marangoni_velocity.cu)

**Current Status:** PASSES with velocity 0.7-0.8 m/s

**Validation Achieved:**
- Magnitude: Matches literature range (0.5-2.0 m/s for LPBF Ti6Al4V)
- Direction: Radial outward flow from hot to cold (verified)
- Force direction sanity check: 70%+ correct

**Missing Validation:**
- ❌ Analytical velocity profile comparison
- ❌ Spatial distribution of Marangoni stress
- ❌ Time evolution of surface deformation

### 2.3 Material Properties Database

**Available Materials:**
- Ti6Al4V (primary validation material)
- 316L Stainless Steel
- Inconel 718
- AlSi10Mg

**Ti6Al4V Properties Used:**
```cpp
T_solidus = 1878 K
T_liquidus = 1928 K (mushy zone width: 50 K)
T_vaporization = 3560 K
L_fusion = 286,000 J/kg
cp_solid = 546 J/(kg·K)
rho_liquid = 4110 kg/m³
mu_liquid = 5.0e-3 Pa·s
dsigma_dT = -2.6e-4 N/(m·K)
```

---

## 3. Benchmark 1: Natural Convection

### 3.1 Problem Definition

**Classical Rayleigh-Benard Convection:**

Heated cavity with temperature difference driving buoyant flow.

**Governing Equations:**
```
Momentum:    ∂u/∂t + u·∇u = -∇p/ρ + ν∇²u + g·β·(T - T_ref)
Energy:      ∂T/∂t + u·∇T = α∇²T
Continuity:  ∇·u = 0
```

**Dimensionless Numbers:**
- **Rayleigh Number:** `Ra = g·β·ΔT·L³/(ν·α)`
  - Physical interpretation: ratio of buoyancy to viscous-thermal damping
  - Critical value: `Ra_crit ≈ 1708` (onset of convection)

- **Prandtl Number:** `Pr = ν/α`
  - Physical interpretation: ratio of momentum to thermal diffusivity
  - For liquid metals: `Pr ~ 0.01-0.1` (thermal diffusion dominates)
  - For water: `Pr ~ 7` (momentum and thermal similar)

- **Nusselt Number:** `Nu = q·L/(k·ΔT)`
  - Physical interpretation: ratio of convective to conductive heat transfer
  - For Ra = 10³-10⁶: `Nu ≈ 0.069·Ra^(1/3)·Pr^0.074` (empirical)

### 3.2 Benchmark Configuration

**Standard Test Case (Davis 1983):**
```
Domain:       Square cavity L × L (2D) or cube L³ (3D)
Geometry:     L = 100 µm (LPBF-relevant scale)
Temperature:  T_hot = 2000 K (bottom), T_cold = 1900 K (top)
Material:     Ti6Al4V liquid (T > T_liquidus)
Boundary:     No-slip walls, isothermal top/bottom, adiabatic sides
Initial:      Quiescent fluid (u = 0), linear T profile
```

**Parameter Selection:**
```cpp
// Physical parameters (Ti6Al4V liquid at T ~ 1950 K)
float L = 100e-6f;              // Cavity size [m]
float Delta_T = 100.0f;         // Temperature difference [K]
float T_ref = 1950.0f;          // Reference temperature [K]
float g = 9.81f;                // Gravity [m/s²]
float beta = 1.2e-4f;           // Thermal expansion coeff [1/K] (Ti6Al4V)
float nu = 1.217e-6f;           // Kinematic viscosity [m²/s]
float alpha = 5.8e-6f;          // Thermal diffusivity [m²/s]

// Dimensionless numbers
float Ra = g * beta * Delta_T * pow(L, 3) / (nu * alpha);
// Ra ≈ 9.81 × 1.2e-4 × 100 × (100e-6)³ / (1.217e-6 × 5.8e-6)
// Ra ≈ 1.67e4 (moderate convection)

float Pr = nu / alpha;
// Pr ≈ 1.217e-6 / 5.8e-6 ≈ 0.21 (liquid metal)
```

### 3.3 Implementation Strategy

**File:** `/home/yzk/LBMProject/tests/validation/test_natural_convection.cu`

**Key Components:**

1. **Domain Setup:**
```cpp
const int nx = 128, ny = 128, nz = 1;  // 2D simulation
const float dx = L / (nx - 1);         // Grid spacing
const float dt = 1e-7f;                // Time step (CFL check required)
```

2. **Buoyancy Force Calculation:**
```cuda
__global__ void computeBuoyancyForceKernel(
    const float* temperature,
    float* force_y,  // Vertical buoyancy force
    float g, float beta, float T_ref,
    int num_cells)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_cells) return;

    float T = temperature[idx];
    // Boussinesq approximation: F_buoy = ρ·g·β·(T - T_ref)
    // For FluidLBM, we provide acceleration [m/s²]
    float accel_y = g * beta * (T - T_ref);
    force_y[idx] = accel_y;
}
```

3. **Boundary Conditions:**
```cpp
// Bottom wall (y=0): T = T_hot, u = 0
thermal.applyBoundaryConditions(1, T_hot);  // Dirichlet BC
fluid.applyBoundaryConditions(1);           // No-slip wall

// Top wall (y=ny-1): T = T_cold, u = 0
// (similar setup)

// Side walls (x=0, x=nx-1): adiabatic ∂T/∂n = 0, no-slip
thermal.applyBoundaryConditions(2, 0.0f);   // Neumann BC
```

4. **Coupling Loop:**
```cpp
for (int step = 0; step < n_steps; ++step) {
    // 1. Compute buoyancy force from temperature field
    computeBuoyancyForce(thermal.getTemperature(), d_force_y);

    // 2. Fluid solver: momentum equation with buoyancy
    fluid.collisionBGK(d_force_x, d_force_y, d_force_z);
    fluid.streaming();
    fluid.applyBoundaryConditions(1);
    fluid.computeMacroscopic();

    // 3. Thermal solver: energy equation with advection
    thermal.collisionBGK(fluid.getVelocityX(),
                        fluid.getVelocityY(),
                        fluid.getVelocityZ());
    thermal.streaming();
    thermal.applyBoundaryConditions(1, T_hot);
    thermal.computeTemperature();

    // 4. Compute diagnostics every N steps
    if (step % diag_interval == 0) {
        float Nu = computeNusseltNumber(thermal, dx, Delta_T);
        float v_max = computeMaxVelocity(fluid);
        // Log: step, t, Nu, v_max, Ra, Pr
    }
}
```

### 3.4 Post-Processing and Validation

**Quantitative Metrics:**

1. **Steady-State Nusselt Number:**
```cpp
float computeNusseltNumber(ThermalLBM& thermal, float dx, float Delta_T) {
    // Nu = (q_conv / q_cond)
    // q_conv = total heat flux through top surface
    // q_cond = k·ΔT / L (pure conduction)

    // Compute vertical heat flux at mid-height
    std::vector<float> T_host(nx * ny * nz);
    thermal.copyTemperatureToHost(T_host.data());

    float q_conv = 0.0f;
    int j_mid = ny / 2;
    for (int i = 0; i < nx; ++i) {
        int idx = i + j_mid * nx;
        int idx_up = i + (j_mid + 1) * nx;
        float dT_dy = (T_host[idx_up] - T_host[idx]) / dx;
        q_conv += material.k_liquid * dT_dy;  // Fourier's law
    }
    q_conv /= nx;  // Average

    float q_cond = material.k_liquid * Delta_T / L;
    return q_conv / q_cond;
}
```

2. **Velocity Profile Comparison:**
- Extract centerline velocity (u vs y at x = L/2)
- Compare with DNS/experimental data (Davis 1983)
- Compute L2 error norm

**Acceptance Criteria:**
```
✓ PASS if:
  - Nu within ±10% of literature value (Davis 1983)
  - Steady state reached (dNu/dt < 1% over 100 time steps)
  - Velocity profile matches qualitatively (similar peak location and magnitude)

⚠ ACCEPTABLE if:
  - Nu within ±20% (LBM numerical diffusion may cause deviation)
  - Flow pattern correct (cells, circulation direction)

✗ FAIL if:
  - Nu error > 30%
  - Flow direction incorrect (upward at hot wall, downward at cold wall)
  - Simulation diverges (NaN, Inf)
```

### 3.5 Literature References for Comparison

**Benchmark Data Sources:**

1. **Davis (1983):** "Natural convection of air in a square cavity: A bench mark numerical solution"
   - Domain: 2D square cavity
   - Ra = 10³, 10⁴, 10⁵, 10⁶
   - Pr = 0.71 (air)
   - Provides: Nu, velocity profiles, streamlines

2. **de Vahl Davis & Jones (1983):** "Natural convection in a square cavity: A comparison exercise"
   - Multiple numerical methods compared
   - Standard test case for CFD validation
   - Our target: Ra ~ 1.67×10⁴, Pr ~ 0.21

3. **Khairallah et al. (2016):** Reports buoyancy effects in LPBF melt pools
   - Ra ~ 10⁴-10⁶ (depending on melt pool size)
   - Nu ~ 2-10 (effective heat transfer enhancement)

**Expected Results for Ra = 1.67×10⁴, Pr = 0.21:**
```
Nu_analytical ≈ 0.069 × (1.67e4)^(1/3) × (0.21)^0.074
Nu_analytical ≈ 3.5-4.5 (estimated)

v_max ≈ (g·β·ΔT·L)^0.5 ~ 0.1-0.5 m/s (order of magnitude)
```

---

## 4. Benchmark 2: Stefan Problem

### 4.1 Problem Definition

**Classical 1D Melting Problem:**

Semi-infinite solid initially at `T = T_solidus`, with boundary held at `T = T_liquidus`. Interface moves as `s(t) = 2λ√(αt)`.

**Governing Equations:**
```
Solid region (x > s(t)):   ρc_p ∂T_s/∂t = k ∂²T_s/∂x²
Liquid region (x < s(t)):  ρc_p ∂T_l/∂t = k ∂²T_l/∂x²
Interface (x = s(t)):       k(∂T_l/∂x - ∂T_s/∂x) = ρL_f ds/dt
```

**Analytical Solution:**
```
Interface position:  s(t) = 2λ√(α·t)

where λ solves:  λ·exp(λ²)·erf(λ) = St / √π

Stefan number:   St = c_p·ΔT / L_f
                 ΔT = T_liquidus - T_solidus
```

### 4.2 Current Test Status Analysis

**File:** `/home/yzk/LBMProject/tests/validation/test_stefan_problem.cu`

**Test Structure:**
- ✅ Analytical solution implemented (`computeLambda()` via Newton-Raphson)
- ✅ Front tracking via liquid fraction: `findMeltingFront()` at `fl = 0.5`
- ✅ Three time points: 0.5 ms, 1.0 ms, 2.0 ms
- ❌ **All tests DISABLED** with physics limitation warning

**Acknowledged Issue (lines 257-260):**
```cpp
std::cout << "NOTE: This test is DISABLED due to known physics limitation." << std::endl;
std::cout << "      Current implementation uses temperature-based phase change." << std::endl;
std::cout << "      Stefan problem requires enthalpy-based advection for accuracy." << std::endl;
std::cout << "  Expected error with current method: 50-150%" << std::endl;
```

### 4.3 Root Cause Analysis

**Problem:** Temperature-based vs Enthalpy-based phase change

**Current Implementation (phase_change.cu):**
```cpp
// Temperature determines liquid fraction
float fl = d_material.liquidFraction(T);

// Enthalpy computed from temperature
float H = rho_ref * cp_ref * T + fl * rho_ref * L_fusion;
```

**What Stefan Problem Requires:**
```
Enthalpy advection:  ∂H/∂t + u·∇H = ∇·(k∇T)

where H(T) = {
  ρcp·T                           if T < T_solidus
  ρcp·T_solidus + ρL·(T-T_solidus)/(T_liq-T_sol)  if T_solidus ≤ T ≤ T_liquidus
  ρcp·T + ρL                      if T > T_liquidus
}

Then solve for T from H using Newton-Raphson (already implemented!)
```

**Why Current Method Fails:**
- Interface velocity `v_interface = q/(ρL)` depends on **heat flux** at interface
- Temperature-based method: Heat flux not directly coupled to interface motion
- Enthalpy-based method: Latent heat absorption automatically captured in `∇·(k∇T)` term
- **Expected error:** 50-150% because interface moves too fast (heat diffuses without absorbing latent heat correctly)

### 4.4 Fix Strategy

**Option A: Enable Existing Enthalpy Method (RECOMMENDED)**

The infrastructure already exists in `phase_change.cu`:

1. **Verify Enthalpy Update Sequence:**
```cpp
// In thermal LBM step
thermal.collisionBGK();
thermal.streaming();
thermal.computeTemperature();  // T from distribution functions

// Update enthalpy from temperature
phase_change.updateEnthalpyFromTemperature(thermal.getTemperature());

// After applying heat sources
thermal.addHeatSource(d_heat_source, dt);

// Solve T from H (this is where latent heat is accounted for)
phase_change.updateTemperatureFromEnthalpy(thermal.getTemperature(),
                                           tolerance=1e-4f, max_iter=50);
```

2. **Add Diagnostic: Latent Heat Absorption Check**
```cpp
// After each time step
float E_latent = computeLatentHeatStored(phase_change);
float E_sensible = computeSensibleHeatStored(thermal);
float E_total = E_latent + E_sensible;

// Check energy conservation
float E_input = integrateHeatSource(laser, dt);
float dE = E_total - E_total_prev;
float energy_error = (dE - E_input) / E_input;

EXPECT_LT(abs(energy_error), 0.05f)
    << "Energy not conserved: " << energy_error * 100 << "% error";
```

3. **Enable Tests Incrementally:**
```cpp
// Start with shortest time (easiest)
TEST_F(StefanProblemTest, ShortTime) {  // Remove DISABLED_
    float test_time = 0.5e-3f;
    runSimulation(test_time);
    float error = testFrontPosition(test_time);

    // Relaxed criterion first: 20% error acceptable for LBM
    EXPECT_LT(error, 0.20f) << "Front position error exceeds 20% threshold";
}
```

**Option B: Implement Explicit Interface Tracking (COMPLEX)**

Use VOF solver to track interface explicitly:
- ❌ More complex coupling
- ❌ Requires interface velocity: `v_interface = (k_l·∇T_l - k_s·∇T_s)/(ρ·L)`
- ✅ More accurate for sharp interfaces

**RECOMMENDATION:** Start with Option A (enable enthalpy method). Only pursue Option B if error remains > 20%.

### 4.5 Validation Methodology

**Test Setup (from existing test_stefan_problem.cu):**
```cpp
// Domain: 1D grid (quasi-1D: NX=500, NY=1, NZ=1)
const int NX = 500;
const float DOMAIN_LENGTH = 2000e-6f;  // 2 mm
const float DX = DOMAIN_LENGTH / (NX - 1);  // 4 µm resolution

// Material: Ti6Al4V
MaterialProperties material = MaterialDatabase::getTi6Al4V();

// Stefan number
float St = material.cp_solid * (material.T_liquidus - material.T_solidus)
         / material.L_fusion;
// St ≈ 546 × 50 / 286000 ≈ 0.095

// Lambda (solve transcendental equation)
float lambda = computeLambda(St);  // lambda ≈ 0.069

// Time discretization
float dt = 0.05 * DX * DX / alpha;  // Conservative CFL
float test_time = 0.5e-3f;  // 0.5 ms
```

**Metrics:**
1. **Interface Position Error:**
```cpp
float s_numerical = findMeltingFront(h_temp, h_fl);  // fl = 0.5 criterion
float s_analytical = 2.0f * lambda * sqrtf(alpha * t);
float error = abs(s_numerical - s_analytical) / s_analytical;
```

2. **Temperature Profile:**
- Check that `T(x) = T_liquidus` for `x < s(t)` (fully liquid)
- Check that `T(x) = T_solidus` for `x > s(t) + δ` (fully solid)
- Check smooth transition in mushy zone `s(t) ± 2·DX`

**Acceptance Criteria:**
```
✓ EXCELLENT if error < 5%
  - Published LBM papers typically achieve 2-8% for Stefan problem
  - Requires careful enthalpy treatment

✓ GOOD if error < 10%
  - Acceptable for engineering applications

⚠ ACCEPTABLE if error < 20%
  - Physics is correct, numerical diffusion causes lag

✗ FAIL if error > 30%
  - Indicates fundamental issue with phase change implementation
```

### 4.6 Implementation Steps

**Day 1: Enable Enthalpy Method**
1. Add diagnostic output to `phase_change.cu`:
   - Log convergence of Newton-Raphson solver
   - Check for NaN/Inf in enthalpy or temperature
   - Verify `updateTemperatureFromEnthalpy()` is called

2. Run `ShortTime` test (0.5 ms) with verbose output:
   ```bash
   cd /home/yzk/LBMProject/build
   ./tests/validation/test_stefan_problem --gtest_filter=*ShortTime*
   ```

3. Analyze error sources:
   - If error ~ 5-10%: PASS, move to longer times
   - If error ~ 20-50%: Check Newton-Raphson convergence
   - If error > 100%: Enthalpy not being updated correctly

**Day 2: Refinement and Full Validation**
1. Run all three time points (0.5, 1.0, 2.0 ms)
2. Grid convergence study: NX = 250, 500, 1000
3. Time step convergence: dt × [0.5, 1.0, 2.0]
4. Generate publication-quality plots:
   - Interface position vs time (numerical vs analytical)
   - Temperature profile at t = 1.0 ms
   - Liquid fraction field evolution

---

## 5. Benchmark 3: Melt Pool Validation

### 5.1 Problem Definition

**Reference:** Khairallah et al. (2016), *Acta Materialia* 108:36-45
"Laser powder-bed fusion additive manufacturing: Physics of complex melt flow and formation mechanisms of pores, spatter, and denudation zones"

**Validation Target:**
Compare simulated melt pool dimensions with experimental/simulation data from Khairallah for Ti6Al4V LPBF.

**Key Physics:**
- Laser absorption (Beer-Lambert)
- Surface evaporation (recoil pressure)
- Marangoni convection (temperature-dependent surface tension)
- Heat conduction (solid + liquid)
- Phase change (melting + solidification)
- Free surface tracking (VOF)

### 5.2 Khairallah 2016 Reference Data

**Simulation Parameters:**
```
Material:        Ti6Al4V powder bed
Laser Power:     195 W
Scan Speed:      1.0 m/s
Spot Size:       55 µm (1/e² radius)
Powder Layer:    30 µm thickness
Substrate:       Solid Ti6Al4V (initially 300 K)
Atmosphere:      Argon (1 atm)
```

**Reported Melt Pool Dimensions:**
```
Depth:           80-120 µm (depends on power density)
Width:           120-180 µm (surface)
Length:          150-250 µm (quasi-steady state)
Lifetime:        50-200 µs (from formation to solidification)
```

**Key Observations:**
- **Keyhole formation:** Recoil pressure creates depression at laser center
- **Marangoni flow:** Outward surface flow (hot→cold) at 0.5-2.0 m/s
- **Vapor depression depth:** 20-50 µm below initial surface
- **Temperature peak:** 3000-3500 K at keyhole base (near T_vaporization)

### 5.3 Simulation Configuration

**Domain Setup:**
```cpp
// Grid resolution
const int nx = 256, ny = 128, nz = 128;  // 256 × 128 × 128 cells
const float dx = 1.0e-6f;                // 1 µm resolution
const float Lx = nx * dx;                // 256 µm
const float Ly = ny * dx;                // 128 µm
const float Lz = nz * dx;                // 128 µm

// Time discretization
const float dt = 5e-8f;                  // 50 ns (CFL-limited)
const float total_time = 200e-6f;        // 200 µs (full scan)
const int n_steps = int(total_time / dt); // 4,000 steps
```

**Laser Parameters:**
```cpp
LaserSource laser;
laser.power = 195.0f;                    // W
laser.scan_speed = 1.0f;                 // m/s
laser.spot_radius = 55e-6f;              // m (1/e² radius)
laser.absorptivity = 0.35f;              // Ti6Al4V solid (increases to 0.7 for liquid)
laser.penetration_depth = 10e-6f;        // m (Beer-Lambert decay)
laser.scan_trajectory = LINEAR_X;        // Scan along x-axis
laser.start_position = {50e-6f, 64e-6f, 100e-6f};  // Start at (50, 64, 100) µm
```

**Initial Conditions:**
```cpp
// Powder bed: porous region (z > 70 µm)
// Solid substrate: dense region (z < 70 µm)
// Temperature: T_0 = 300 K everywhere

for (int k = 0; k < nz; ++k) {
    float z = k * dx;
    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            int idx = i + nx * (j + ny * k);

            if (z > 70e-6f) {
                // Powder region (porous, ~50% packing)
                fill_level[idx] = 0.5f;
                temperature[idx] = 300.0f;
            } else {
                // Substrate (solid)
                fill_level[idx] = 1.0f;
                temperature[idx] = 300.0f;
            }
        }
    }
}
```

**Boundary Conditions:**
```cpp
// Top surface (z = nz-1):    Free surface (VOF tracks interface)
// Bottom (z = 0):            Substrate cooling: h_conv = 1000 W/(m²·K), T_sub = 300 K
// Sides (x=0, x=nx-1, etc.): Periodic (to minimize domain size)
```

### 5.4 Measurement Protocol

**Melt Pool Dimensions:**

1. **Width (W):**
```cpp
float computeMeltPoolWidth(ThermalLBM& thermal, int k_surface) {
    // At surface layer (k = k_surface), find x-extent where T > T_liquidus
    std::vector<float> T_host(nx * ny * nz);
    thermal.copyTemperatureToHost(T_host.data());

    int j_mid = ny / 2;  // Centerline
    int x_left = -1, x_right = -1;

    for (int i = 0; i < nx; ++i) {
        int idx = i + nx * (j_mid + ny * k_surface);
        if (T_host[idx] > material.T_liquidus) {
            if (x_left < 0) x_left = i;
            x_right = i;
        }
    }

    return (x_right - x_left) * dx;  // Width in meters
}
```

2. **Depth (D):**
```cpp
float computeMeltPoolDepth(ThermalLBM& thermal, int i_laser, int j_mid) {
    // At laser center (i_laser, j_mid), find z-extent where T > T_liquidus
    std::vector<float> T_host(nx * ny * nz);
    thermal.copyTemperatureToHost(T_host.data());

    int z_top = -1, z_bottom = -1;

    for (int k = 0; k < nz; ++k) {
        int idx = i_laser + nx * (j_mid + ny * k);
        if (T_host[idx] > material.T_liquidus) {
            if (z_bottom < 0) z_bottom = k;
            z_top = k;
        }
    }

    return (z_top - z_bottom) * dx;  // Depth in meters
}
```

3. **Length (L):**
```cpp
float computeMeltPoolLength(ThermalLBM& thermal, int j_mid, int k_surface) {
    // Along x-axis at centerline, measure x-extent of liquid region
    // (similar to width calculation, but along x instead of across y)
    // ...
}
```

4. **Keyhole Depth:**
```cpp
float computeKeyholeDepth(VOFSolver& vof, int i_laser, int j_mid) {
    // Find deepest depression in free surface
    std::vector<float> fill_host(nx * ny * nz);
    vof.copyFillLevelToHost(fill_host.data());

    int z_surface_undisturbed = int(70e-6f / dx);  // Initial surface height
    int z_depression = z_surface_undisturbed;

    for (int k = 0; k < nz; ++k) {
        int idx = i_laser + nx * (j_mid + ny * k);
        if (fill_host[idx] < 0.5f) {  // Vapor region
            z_depression = k;
            break;
        }
    }

    return (z_surface_undisturbed - z_depression) * dx;  // Depth below original surface
}
```

### 5.5 Comparison Methodology

**Time Series Analysis:**

Track melt pool evolution at 10 µs intervals:
```cpp
std::vector<float> time_points = {10, 20, 40, 60, 80, 100, 120, 150, 200};  // µs
std::vector<float> width_sim, depth_sim, keyhole_sim;

for (int step = 0; step < n_steps; ++step) {
    float t_current = step * dt * 1e6;  // Convert to µs

    // ... run simulation step ...

    // Record diagnostics at specified times
    if (isCloseToTimePoint(t_current, time_points)) {
        float W = computeMeltPoolWidth(thermal, k_surface);
        float D = computeMeltPoolDepth(thermal, i_laser, j_mid);
        float K = computeKeyholeDepth(vof, i_laser, j_mid);

        width_sim.push_back(W * 1e6);     // Convert to µm
        depth_sim.push_back(D * 1e6);
        keyhole_sim.push_back(K * 1e6);
    }
}
```

**Statistical Comparison:**
```cpp
// Khairallah 2016 data (Table 2, Figure 5)
float width_ref = 150e-6f;   // 150 µm ± 15% (experimental variation)
float depth_ref = 100e-6f;   // 100 µm ± 20%
float keyhole_ref = 35e-6f;  // 35 µm ± 30%

// Compute steady-state average (t > 50 µs)
float width_avg = computeAverage(width_sim, t_start=50e-6f);
float depth_avg = computeAverage(depth_sim, t_start=50e-6f);
float keyhole_avg = computeAverage(keyhole_sim, t_start=50e-6f);

// Relative errors
float err_width = abs(width_avg - width_ref) / width_ref;
float err_depth = abs(depth_avg - depth_ref) / depth_ref;
float err_keyhole = abs(keyhole_avg - keyhole_ref) / keyhole_ref;
```

### 5.6 Acceptance Criteria

```
✓ EXCELLENT if:
  - Width error < 10%
  - Depth error < 15%
  - Keyhole depth error < 20%
  - Marangoni velocity 0.5-2.0 m/s (matches literature)

✓ GOOD if:
  - Width error < 15%
  - Depth error < 20%
  - Keyhole forms (qualitative)
  - Temperature peak near T_vaporization

⚠ ACCEPTABLE if:
  - Width/Depth within ±25% (order of magnitude correct)
  - General melt pool shape correct
  - Flow patterns qualitatively correct

✗ FAIL if:
  - Width/Depth error > 30%
  - No keyhole formation (recoil pressure not working)
  - Temperature runaway (T >> T_vaporization)
  - Simulation crashes or diverges
```

**Note on Uncertainty:**
- Khairallah's simulations show ±10-20% variation depending on numerical parameters
- Experimental melt pool dimensions vary ±15-25% due to powder packing, shielding gas flow, etc.
- Our target: Match Khairallah simulation results within their reported uncertainty

### 5.7 Implementation File

**File:** `/home/yzk/LBMProject/tests/validation/test_khairallah_melt_pool.cu`

**Key Features:**
- Full multiphysics coupling via `MultiphysicsSolver`
- VTK output every 10 µs for visualization
- Diagnostic log: time, width, depth, keyhole_depth, T_max, v_max, CFL
- Comparison table: simulation vs Khairallah 2016 reference

**Estimated Runtime:**
- 4,000 steps × 2 ms/step ≈ 8 seconds (RTX 3090)
- Total with I/O: ~15-20 seconds

---

## 6. Benchmark 4: Marangoni Analytical Validation

### 6.1 Problem Definition

**Reference:** Young, Goldstein & Block (1959), *Journal of Fluid Mechanics* 6:350-356
"The motion of bubbles in a vertical temperature gradient"

**Physics:**
Surface tension gradient drives tangential flow along interface:
```
Marangoni stress:  τ = dσ/dT · ∇_s T

where ∇_s T = (I - n⊗n)·∇T  (tangential temperature gradient)
      n = interface normal
      σ(T) = surface tension
```

**Analytical Solution (Simplified Case):**

For a **planar interface** with **linear temperature gradient** `∇T = constant`:

```
Velocity profile:  u(z) = (1/μ) · (dσ/dT) · (∇T_tangential) · z

Maximum velocity:  u_max = (1/μ) · |dσ/dT| · |∇T| · δ

where δ = thermal boundary layer thickness
```

### 6.2 Current Test Status

**File:** `/home/yzk/LBMProject/tests/validation/test_marangoni_velocity.cu`

**Achievements:**
- ✅ Velocity magnitude validated: 0.7-0.8 m/s (matches LPBF literature)
- ✅ Direction validated: Radial outward flow from hot to cold
- ✅ Force magnitude validated: ~10⁶-10⁹ N/m³ range

**Missing:**
- ❌ Analytical velocity profile comparison
- ❌ Spatial distribution of Marangoni stress
- ❌ Scaling law validation: `v ~ (dσ/dT)·∇T·L/μ`

### 6.3 Analytical Benchmark Configuration

**Simplified Geometry:**

Instead of radial temperature gradient (current test), use **1D shear flow** along planar interface:

```
Domain:      Lx × Ly × Lz = 200 µm × 200 µm × 100 µm
Interface:   Planar at z = 50 µm (horizontal)
Temperature: T(x) = T_cold + (T_hot - T_cold) · x/Lx
             Linear gradient along x-axis
             Uniform in y, z directions
Fluid:       Liquid Ti6Al4V above interface (z > 50 µm)
             Vapor below interface (z < 50 µm) - zero velocity
```

**Analytical Solution:**

For Stokes flow with Marangoni forcing:
```
Governing equation:  μ ∂²u/∂z² = 0  (in bulk)

Boundary conditions:
  z = 0:       u = 0  (no-slip substrate)
  z = δ:       μ ∂u/∂z = τ_Marangoni = (dσ/dT)·(dT/dx)
  z → ∞:       u → 0  (far field)

Solution:  u(z) = (1/μ)·(dσ/dT)·(dT/dx)·z  for 0 < z < δ
           u(z) = u_max                      for z ≥ δ

where δ ~ √(μ·Lx/ρ·v_max) (boundary layer thickness)
```

**Expected Velocity:**
```cpp
// Ti6Al4V liquid properties
float mu = 5.0e-3f;           // Pa·s
float dsigma_dT = -2.6e-4f;   // N/(m·K)
float dT_dx = (T_hot - T_cold) / Lx;
                               // (2500 - 2000) / 200e-6 = 2.5e6 K/m

// Maximum velocity
float u_max = abs(dsigma_dT * dT_dx) / mu;
// u_max = 2.6e-4 × 2.5e6 / 5.0e-3 = 130 m/s (too high! Need to adjust ΔT)

// More realistic: ΔT = 100 K over 200 µm
float dT_dx_realistic = 100.0f / 200e-6f;  // 5e5 K/m
float u_max_realistic = abs(dsigma_dT * dT_dx_realistic) / mu;
// u_max = 2.6e-4 × 5e5 / 5.0e-3 = 26 m/s (still high, but physical)
```

**Note:** The high velocities indicate strong Marangoni effect. In reality, inertia and viscous damping limit velocities to O(1-10 m/s).

### 6.4 Implementation Strategy

**Test Structure:**

1. **Setup Linear Temperature Gradient:**
```cuda
__global__ void initializeLinearTemperatureKernel(
    float* temperature,
    float T_cold, float T_hot,
    float Lx,
    int nx, int ny, int nz, float dx)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i >= nx || j >= ny || k >= nz) return;

    int idx = i + nx * (j + ny * k);
    float x = i * dx;

    // Linear T profile along x
    temperature[idx] = T_cold + (T_hot - T_cold) * (x / Lx);
}
```

2. **Run Simulation Until Steady State:**
```cpp
// Criterion: steady state when du/dt < 1e-6 m/s per time step
bool reachedSteadyState(FluidLBM& fluid, float threshold = 1e-6f) {
    static std::vector<float> u_prev(fluid.getNx() * fluid.getNy() * fluid.getNz());
    std::vector<float> u_curr(fluid.getNx() * fluid.getNy() * fluid.getNz());
    fluid.copyVelocityToHost(u_curr.data(), nullptr, nullptr);

    if (u_prev.empty()) {
        u_prev = u_curr;
        return false;
    }

    float max_change = 0.0f;
    for (size_t i = 0; i < u_curr.size(); ++i) {
        max_change = std::max(max_change, abs(u_curr[i] - u_prev[i]));
    }

    u_prev = u_curr;
    return (max_change < threshold);
}
```

3. **Extract Velocity Profile:**
```cpp
std::vector<float> extractVelocityProfile_xz(FluidLBM& fluid, int j_mid) {
    // Extract u(x, z) at centerline y = j_mid
    int nx = fluid.getNx(), nz = fluid.getNz();
    std::vector<float> ux_host(nx * ny * nz);
    fluid.copyVelocityToHost(ux_host.data(), nullptr, nullptr);

    std::vector<float> profile(nx * nz);
    for (int k = 0; k < nz; ++k) {
        for (int i = 0; i < nx; ++i) {
            int idx_3d = i + nx * (j_mid + ny * k);
            int idx_2d = i + nx * k;
            profile[idx_2d] = ux_host[idx_3d];
        }
    }
    return profile;
}
```

4. **Compare with Analytical Solution:**
```cpp
void compareWithAnalyticalSolution(
    const std::vector<float>& u_sim,
    float u_analytical_max,
    float z_interface,
    float dx, int nx, int nz)
{
    // Analytical: u(z) = u_max for z > z_interface (above interface)
    //             u(z) = 0     for z < z_interface (below interface, vapor)

    int k_interface = int(z_interface / dx);
    int i_mid = nx / 2;  // Sample at mid-x

    std::vector<float> u_sim_vertical;
    std::vector<float> u_analytical_vertical;

    for (int k = 0; k < nz; ++k) {
        int idx = i_mid + nx * k;
        u_sim_vertical.push_back(u_sim[idx]);

        if (k < k_interface) {
            u_analytical_vertical.push_back(0.0f);  // Vapor
        } else {
            u_analytical_vertical.push_back(u_analytical_max);  // Liquid
        }
    }

    // Compute L2 error
    float error = computeL2Error(u_sim_vertical, u_analytical_vertical);

    // Visualize
    plotVelocityProfile(u_sim_vertical, u_analytical_vertical, "marangoni_profile.png");
}
```

### 6.5 Acceptance Criteria

```
✓ EXCELLENT if:
  - Velocity magnitude within ±10% of analytical prediction
  - Profile shape matches (piecewise constant above interface)
  - L2 error < 15%

✓ GOOD if:
  - Velocity magnitude within ±20%
  - Correct flow direction (along temperature gradient)
  - Qualitative profile match (flat above interface, zero below)

⚠ ACCEPTABLE if:
  - Velocity within order of magnitude
  - Flow pattern correct
  - Scaling law confirmed: v ∝ (dσ/dT)·∇T

✗ FAIL if:
  - Wrong flow direction
  - Velocity order of magnitude wrong (factor > 10×)
  - Strong mesh dependence (result changes > 50% with refinement)
```

### 6.6 Additional Validation: Scaling Law

**Parametric Study:**

Vary temperature gradient and confirm linear scaling:
```cpp
std::vector<float> dT_dx_values = {1e5, 2e5, 5e5, 1e6};  // K/m
std::vector<float> v_max_sim;

for (float dT_dx : dT_dx_values) {
    // Set up simulation with this gradient
    float T_hot = T_cold + dT_dx * Lx;
    initializeLinearTemperature(thermal, T_cold, T_hot, Lx);

    // Run to steady state
    runUntilSteadyState(multiphysics);

    // Measure max velocity
    float v = computeMaxVelocity(fluid);
    v_max_sim.push_back(v);
}

// Expected: v_max = c · dT_dx where c = |dσ/dT| / μ
float c_analytical = abs(dsigma_dT) / mu;  // 0.052 m²·K/(s·kg)

// Fit linear regression: v_max_sim = c_fit · dT_dx
float c_fit = linearRegression(dT_dx_values, v_max_sim);

// Validation
float error = abs(c_fit - c_analytical) / c_analytical;
EXPECT_LT(error, 0.15f) << "Scaling law not satisfied: "
                        << c_fit << " vs " << c_analytical;
```

---

## 7. Implementation Priority and Timeline

### 7.1 Recommended Sequence

**Week 1: Foundation (Days 1-2)**
```
Day 1: Stefan Problem
  - Enable enthalpy method in phase_change.cu
  - Add diagnostics (latent heat, energy conservation)
  - Run test_stefan_problem.cu (remove DISABLED)
  - Target: <20% error on interface position
  - Deliverable: test_stefan_problem passing

Day 2: Stefan Problem Refinement
  - Grid/time convergence study
  - Generate comparison plots (s(t) vs analytical)
  - Document results in benchmark/STEFAN_VALIDATION.md
  - Deliverable: Publication-quality validation report
```

**Week 2: Natural Convection (Days 3-5)**
```
Day 3: Test Implementation
  - Create test_natural_convection.cu
  - Implement buoyancy force kernel
  - Set up 2D Rayleigh-Benard configuration
  - Run to steady state (monitor Nu convergence)
  - Deliverable: First Nu measurement

Day 4: Validation and Tuning
  - Compare Nu with Davis (1983) benchmark
  - Extract velocity profiles
  - Adjust Ra/Pr if needed (material properties)
  - Target: Nu within ±20% of literature
  - Deliverable: Natural convection validated

Day 5: Documentation and Visualization
  - Generate streamlines, temperature contours
  - Plot Nu vs time (convergence to steady state)
  - Compare velocity profiles with reference
  - Deliverable: benchmark/NATURAL_CONVECTION_VALIDATION.md
```

**Week 3: Marangoni Analytical (Day 6)**
```
Day 6: Analytical Profile Validation
  - Modify test_marangoni_velocity.cu
  - Add linear temperature gradient setup
  - Extract velocity profile u(z)
  - Compare with analytical solution
  - Run scaling law parametric study
  - Deliverable: test_marangoni_analytical.cu passing
```

**Week 4: Melt Pool Comparison (Days 7-9)**
```
Day 7: Test Setup
  - Create test_khairallah_melt_pool.cu
  - Configure laser parameters (195 W, 1.0 m/s)
  - Set up powder bed initial conditions
  - Implement melt pool dimension measurement
  - Deliverable: Simulation runs to completion

Day 8: Comparison and Tuning
  - Extract width, depth, keyhole dimensions
  - Compare with Khairallah 2016 Table 2
  - Adjust laser absorptivity if needed
  - Check Marangoni velocity (0.5-2.0 m/s)
  - Deliverable: Melt pool dimensions within ±25%

Day 9: Visualization and Documentation
  - Generate ParaView visualizations
  - Time series plots (width, depth vs time)
  - Temperature and velocity field snapshots
  - Deliverable: benchmark/KHAIRALLAH_MELT_POOL_VALIDATION.md
```

### 7.2 Effort Estimation

| Task | Effort (hours) | Dependencies | Risk Level |
|------|---------------|--------------|------------|
| Stefan Problem Fix | 8-12 | None | LOW (infrastructure exists) |
| Natural Convection | 16-24 | Stefan complete | MEDIUM (new benchmark) |
| Marangoni Analytical | 4-8 | None | LOW (modify existing test) |
| Khairallah Melt Pool | 16-24 | All above | HIGH (full multiphysics) |
| **TOTAL** | **44-68 hours** | Sequential | - |

**Calendar Time:** 9-12 working days (2-3 weeks)

### 7.3 Success Criteria Summary

**Phase 2 COMPLETE when:**
- [x] Stefan problem: Interface position error < 20%
- [x] Natural convection: Nu within ±20% of Davis (1983)
- [x] Marangoni: Velocity profile matches analytical shape
- [x] Khairallah: Melt pool dimensions within ±25%
- [x] All validation reports documented in `benchmark/`

**Publication-Ready Quality (Stretch Goal):**
- [ ] Stefan problem: < 10% error
- [ ] Natural convection: Nu within ±10%
- [ ] Marangoni: Scaling law confirmed (v ∝ ∇T)
- [ ] Khairallah: Melt pool dimensions within ±15%

---

## 8. Appendix: Literature References

### 8.1 Natural Convection

1. **Davis, G.D.V.** (1983). "Natural convection of air in a square cavity: A bench mark numerical solution." *International Journal for Numerical Methods in Fluids*, 3(3):249-264.

2. **de Vahl Davis, G., & Jones, I.P.** (1983). "Natural convection in a square cavity: A comparison exercise." *International Journal for Numerical Methods in Fluids*, 3(3):227-248.

3. **Incropera, F.P., et al.** (2007). *Fundamentals of Heat and Mass Transfer*, 6th ed. Chapter 9: Free Convection.

### 8.2 Stefan Problem

1. **Crank, J.** (1984). *Free and Moving Boundary Problems*. Oxford University Press. Chapter 1: Classical Stefan Problem.

2. **Voller, V.R.** (1997). "An overview of numerical methods for solving phase change problems." *Advances in Numerical Heat Transfer*, 1:341-380.

3. **Hu, H., & Argyropoulos, S.A.** (1996). "Mathematical modelling of solidification and melting: A review." *Modelling and Simulation in Materials Science and Engineering*, 4(4):371.

### 8.3 Melt Pool and LPBF

1. **Khairallah, S.A., et al.** (2016). "Laser powder-bed fusion additive manufacturing: Physics of complex melt flow and formation mechanisms of pores, spatter, and denudation zones." *Acta Materialia*, 108:36-45.

2. **King, W.E., et al.** (2015). "Observation of keyhole-mode laser melting in laser powder-bed fusion additive manufacturing." *Journal of Materials Processing Technology*, 214(12):2915-2925.

3. **Panwisawas, C., et al.** (2017). "Mesoscale modelling of selective laser melting: Thermal fluid dynamics and microstructural evolution." *Computational Materials Science*, 126:479-490.

### 8.4 Marangoni Effect

1. **Young, N.O., Goldstein, J.S., & Block, M.J.** (1959). "The motion of bubbles in a vertical temperature gradient." *Journal of Fluid Mechanics*, 6(3):350-356.

2. **Scriven, L.E., & Sternling, C.V.** (1960). "The Marangoni effects." *Nature*, 187(4733):186-188.

3. **Kidess, A., et al.** (2016). "Marangoni driven turbulence in high energy surface melting processes." *International Journal of Thermal Sciences*, 104:412-426.

### 8.5 LBM Validation

1. **Kruger, T., et al.** (2017). *The Lattice Boltzmann Method: Principles and Practice*. Springer. Chapter 6: Thermal LBM.

2. **Guo, Z., & Shu, C.** (2013). *Lattice Boltzmann Method and Its Applications in Engineering*. World Scientific. Chapter 7: Multiphase Flows.

3. **Korner, C., et al.** (2013). "Lattice Boltzmann model for free surface flow for modeling foaming." *Journal of Statistical Physics*, 121(1-2):179-196.

---

## Revision History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2026-01-10 | LBM-CUDA Architect | Initial comprehensive validation plan |

---

**End of Document**
