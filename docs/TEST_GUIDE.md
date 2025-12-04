# LBM Multiphysics Test Suite Guide

## Overview

This document provides comprehensive documentation for the entire physics test suite in the LBM multiphysics simulation framework. The test suite validates all major physics modules including VOF solver, Marangoni effects, thermal solver, fluid dynamics, phase change, and multiphysics coupling.

**Total Test Coverage**: 150+ tests across 8 major categories

## Table of Contents

1. [Quick Start](#quick-start)
2. [Test Organization](#test-organization)
3. [VOF Solver Tests](#1-vof-solver-tests)
4. [Marangoni Effect Tests](#2-marangoni-effect-tests)
5. [Thermal Solver Tests](#3-thermal-solver-tests)
6. [Fluid LBM Tests](#4-fluid-lbm-tests)
7. [Phase Change Tests](#5-phase-change-tests)
8. [Multiphysics Coupling Tests](#6-multiphysics-coupling-tests)
9. [Energy Conservation Tests](#7-energy-conservation-tests)
10. [Stability & Validation Tests](#8-stability--validation-tests)
11. [Running Tests](#running-tests)
12. [Interpreting Results](#interpreting-results)
13. [Adding New Tests](#adding-new-tests)

---

## Quick Start

### Run All Tests
```bash
cd /home/yzk/LBMProject
./RUN_ALL_PHYSICS_TESTS.sh
```

### Run Specific Category
```bash
cd /home/yzk/LBMProject/build
ctest -R "vof" --output-on-failure          # VOF tests only
ctest -R "marangoni" --output-on-failure     # Marangoni tests only
ctest -R "thermal" --output-on-failure       # Thermal tests only
ctest -R "energy" --output-on-failure        # Energy tests only
```

### Run Quick Validation (Pre-commit)
```bash
cd /home/yzk/LBMProject/build
ctest -R "flux_limiter|temperature_bounds|omega_reduction" --output-on-failure
# Expected time: < 30 seconds
```

---

## Test Organization

### Directory Structure
```
tests/
├── unit/                      # Component-level tests (fast, < 5s)
│   ├── vof/                  # VOF solver unit tests
│   ├── marangoni/            # Marangoni force tests
│   ├── thermal/              # Thermal solver tests
│   ├── fluid/                # Fluid LBM tests
│   ├── phase_change/         # Phase change tests
│   ├── collision/            # Collision operator tests
│   ├── streaming/            # Streaming operator tests
│   └── boundary/             # Boundary condition tests
├── integration/               # Multi-module tests (medium, 1-5 min)
│   ├── multiphysics/         # Full coupling tests
│   ├── stability/            # Stability tests
│   └── *.cu                  # Various integration tests
├── validation/                # Physics validation (slow, 5-30 min)
│   ├── vof/                  # VOF analytical validation
│   ├── multiphysics/         # Multiphysics validation
│   └── *.cu                  # Benchmark tests
├── debug/                     # Debugging/diagnostic tests
├── diagnostic/                # Diagnostic tools
├── performance/               # Performance benchmarks
└── regression/                # Regression tests
```

### Test Categories by Speed

**Fast (< 30s)**: Unit tests, quick validation
- Run before every commit
- Examples: `test_flux_limiter`, `test_vof_advection_uniform`

**Medium (30s - 5min)**: Integration tests
- Run before pushing to repository
- Examples: `test_marangoni_flow`, `test_thermal_fluid_coupling`

**Slow (> 5min)**: Full validation tests
- Run nightly or before major releases
- Examples: `test_grid_convergence`, `test_timestep_convergence`

---

## 1. VOF Solver Tests

The Volume-of-Fluid (VOF) solver tracks free surface interfaces and handles surface tension, evaporation, and recoil pressure effects.

### 1.1 Interface Advection Tests

#### test_vof_advection.cu
**Location**: `/home/yzk/LBMProject/tests/unit/vof/test_vof_advection.cu`

**Purpose**: Validate VOF advection equation ∂f/∂t + ∇·(f·u) = 0

**Tests**:
- Basic advection with uniform velocity
- Velocity unit conversion (lattice → physical)
- Zero velocity stability
- Multi-dimensional advection

**Physics Validated**:
- Interface transport accuracy
- Advection scheme correctness
- Unit conversion consistency

**How to Run**:
```bash
cd /home/yzk/LBMProject/build
ctest -R test_vof_advection --output-on-failure
```

**Pass Criteria**:
- Interface displacement error < 0.5 cells
- No spurious oscillations
- Velocity conversion error < 1%

**Typical Runtime**: 5-10 seconds

---

#### test_vof_advection_uniform.cu
**Location**: `/home/yzk/LBMProject/tests/unit/vof/test_vof_advection_uniform.cu`

**Purpose**: Strict validation of uniform velocity advection with analytical solutions

**Tests**:
1. **InterfaceDisplacement**
   - Physics: Interface translates distance = u × t
   - Setup: Plane interface, u = 0.1 m/s, 100 timesteps
   - Expected: Displacement error < 0.5 cells

2. **MassConservation**
   - Physics: ∫f dV = constant
   - Setup: 100×32×32 domain, 200 timesteps
   - Expected: Mass error < 1%

3. **InterfaceShapePreservation**
   - Physics: Plane interface remains planar
   - Setup: 64×64×32 domain, 300 timesteps
   - Expected: Interface std dev < 1 cell

**Physics Validated**:
- Advection equation accuracy
- Mass conservation
- Interface shape preservation
- Numerical diffusion quantification

**How to Run**:
```bash
ctest -R test_vof_advection_uniform --output-on-failure
```

**Interpreting Results**:
- **PASS**: Displacement error < 0.5 cells, mass error < 1%
- **FAIL**: Check for excessive numerical diffusion or advection bugs
- **Check**: `displacement_error`, `mass_error`, `interface_thickness`

**Typical Runtime**: 8-12 seconds

---

#### test_vof_advection_shear.cu
**Location**: `/home/yzk/LBMProject/tests/unit/vof/test_vof_advection_shear.cu`

**Purpose**: Test advection under shear flow, measure numerical diffusion

**Tests**:
1. **InterfaceTilting**
   - Physics: Interface tilts under shear u(y) = γ·y
   - Setup: γ = 1000 s⁻¹, 20 μs simulation
   - Expected: Tilt angle within 30% of theory (tan(θ) ≈ γ·t)

2. **InterfaceDiffusion**
   - Physics: Measure numerical diffusion rate
   - Expected: 0.5-10 cells growth (upwind scheme)

3. **MassConservationShear**
   - Physics: Mass conserved under complex flow
   - Expected: Mass error < 2% (relaxed for shear)

**Physics Validated**:
- Shear flow handling
- Numerical diffusion quantification
- Complex velocity field advection

**How to Run**:
```bash
ctest -R test_vof_advection_shear --output-on-failure
```

**Typical Runtime**: 10-15 seconds

---

### 1.2 Interface Reconstruction & Curvature

#### test_vof_reconstruction.cu
**Location**: `/home/yzk/LBMProject/tests/unit/vof/test_vof_reconstruction.cu`

**Purpose**: Validate PLIC (Piecewise Linear Interface Calculation) reconstruction

**Tests**:
- Interface normal calculation
- Interface plane reconstruction
- Volume consistency

**Physics Validated**:
- Normal vector accuracy: n = -∇f / |∇f|
- PLIC reconstruction correctness
- Volume conservation in reconstruction

**How to Run**:
```bash
ctest -R test_vof_reconstruction --output-on-failure
```

**Pass Criteria**:
- Normal magnitude = 1 ± 0.01
- Reconstructed volume error < 5%

**Typical Runtime**: 3-5 seconds

---

#### test_vof_curvature.cu
**Location**: `/home/yzk/LBMProject/tests/unit/vof/test_vof_curvature.cu`

**Purpose**: Validate curvature computation κ = ∇·n

**Tests**:
- Plane interface (κ = 0)
- Spherical interface (κ = 2/R)
- Cylindrical interface (κ = 1/R)

**Physics Validated**:
- Curvature calculation accuracy
- Sign convention (liquid droplet → positive κ)
- Isotropy (no grid orientation bias)

**How to Run**:
```bash
ctest -R test_vof_curvature --output-on-failure
```

**Pass Criteria**:
- Plane: |κ| < 0.01
- Sphere: Error < 10% for R > 20 cells
- Cylinder: Error < 15% for R > 16 cells

**Typical Runtime**: 5-8 seconds

---

#### test_vof_curvature_sphere.cu
**Location**: `/home/yzk/LBMProject/tests/validation/vof/test_vof_curvature_sphere.cu`

**Purpose**: Multi-resolution curvature validation for spherical interfaces

**Tests**:
1. **LargeSphere** (R=20 cells)
   - Analytical: κ = 2/R = 0.1
   - Expected: Error < 10%

2. **MediumSphere** (R=12 cells)
   - Expected: Error < 20%

3. **SmallSphere** (R=8 cells)
   - Expected: Error < 30% (poor resolution)

4. **CurvatureIsotropy**
   - Physics: Curvature uniform across sphere
   - Expected: Coefficient of variation < 0.3

**Physics Validated**:
- Multi-resolution accuracy
- Isotropy validation
- Statistical consistency

**How to Run**:
```bash
ctest -R test_vof_curvature_sphere --output-on-failure
```

**Interpreting Results**:
- Check mean curvature vs analytical value
- Check CV (coefficient of variation) for isotropy
- Larger spheres should have better accuracy

**Typical Runtime**: 10-20 seconds

---

#### test_vof_curvature_cylinder.cu
**Location**: `/home/yzk/LBMProject/tests/validation/vof/test_vof_curvature_cylinder.cu`

**Purpose**: Validate 2D curvature (cylindrical geometry)

**Tests**:
1. **LargeCylinder** (R=16 cells)
   - Analytical: κ = 1/R (in cross-section)
   - Expected: Error < 15%

2. **MediumCylinder** (R=10 cells)
   - Expected: Error < 25%

3. **CylinderVsSphere**
   - Physics: κ_cylinder / κ_sphere ≈ 0.5
   - Expected: Ratio error < 30%

**Physics Validated**:
- 2D geometry in 3D code
- Comparative validation
- Axial uniformity

**How to Run**:
```bash
ctest -R test_vof_curvature_cylinder --output-on-failure
```

**Typical Runtime**: 10-15 seconds

---

### 1.3 Mass Conservation

#### test_vof_mass_conservation.cu
**Location**: `/home/yzk/LBMProject/tests/unit/vof/test_vof_mass_conservation.cu`

**Purpose**: Verify mass conservation during VOF operations

**Tests**:
- Static interface (no flow)
- Advection with flow
- With evaporation
- With solidification shrinkage

**Physics Validated**:
- Mass conservation: M = ∫f dV = constant
- Conservation accuracy under different operations

**How to Run**:
```bash
ctest -R test_vof_mass_conservation --output-on-failure
```

**Pass Criteria**:
- Static: Mass error < 0.1%
- Dynamic: Mass error < 1-2%
- With sources/sinks: Mass change matches analytical

**Typical Runtime**: 5-8 seconds

---

### 1.4 Surface Tension & Contact Angle

#### test_vof_surface_tension.cu
**Location**: `/home/yzk/LBMProject/tests/unit/vof/test_vof_surface_tension.cu`

**Purpose**: Validate surface tension force calculation

**Tests**:
- Continuum Surface Force (CSF) method
- Force magnitude and direction
- Laplace pressure validation

**Physics Validated**:
- Surface tension force: F = σκn
- Laplace pressure: ΔP = σκ
- Force direction (points toward liquid)

**How to Run**:
```bash
ctest -R test_vof_surface_tension --output-on-failure
```

**Pass Criteria**:
- Force direction correct
- Laplace pressure error < 20%

**Typical Runtime**: 5-10 seconds

---

#### test_vof_contact_angle.cu
**Location**: `/home/yzk/LBMProject/tests/unit/vof/test_vof_contact_angle.cu`

**Purpose**: Validate contact angle implementation

**Tests**:
- Wall contact angle enforcement
- Dynamic vs static contact angles
- Wetting and non-wetting conditions

**Physics Validated**:
- Contact angle boundary condition
- Young's equation: σ_sv - σ_sl = σ_lv·cos(θ)

**How to Run**:
```bash
ctest -R test_vof_contact_angle --output-on-failure
```

**Pass Criteria**:
- Contact angle within 5° of prescribed value

**Typical Runtime**: 8-12 seconds

---

### 1.5 Evaporation Coupling

#### test_vof_evaporation_mass_loss.cu
**Location**: `/home/yzk/LBMProject/tests/unit/vof/test_vof_evaporation_mass_loss.cu`

**Purpose**: Validate VOF-thermal coupling for evaporation

**Tests**:
1. **SingleTimestepMassLoss**
   - Formula: df = -J_evap × dt / (ρ × dx)
   - Setup: J = 100 kg/(m²·s), single timestep
   - Expected: Error < 5%

2. **StabilityLimiter**
   - Physics: Extreme flux triggers 2% limiter
   - Setup: J = 50,000 kg/(m²·s) (very high)
   - Expected: df capped at -0.02 (2%)

3. **ProgressiveMassLoss**
   - Physics: Cumulative mass loss over 100 steps
   - Expected: Total error < 10%

4. **TopLayerEvaporationOnly**
   - Physics: Evaporation localized to specified cells
   - Expected: Top layer changes, interior unchanged

**Physics Validated**:
- Hertz-Knudsen mass flux formula
- Stability limiter effectiveness
- Spatial localization

**How to Run**:
```bash
ctest -R test_vof_evaporation_mass_loss --output-on-failure
```

**Interpreting Results**:
- Check `mass_loss_rate` vs analytical
- Verify stability limiter activates at high flux
- Confirm spatial localization

**Typical Runtime**: 5-10 seconds

---

#### test_evaporation_temperature_check.cu
**Location**: `/home/yzk/LBMProject/tests/unit/vof/test_evaporation_temperature_check.cu`

**Purpose**: Verify evaporation only occurs above boiling point

**Tests**:
- Evaporation activation at T > T_boil
- No evaporation at T < T_boil
- Temperature-dependent evaporation rate

**Physics Validated**:
- Temperature threshold enforcement
- Hertz-Knudsen temperature dependence

**How to Run**:
```bash
ctest -R test_evaporation_temperature_check --output-on-failure
```

**Pass Criteria**:
- Zero flux below T_boil
- Non-zero flux above T_boil

**Typical Runtime**: 3-5 seconds

---

### 1.6 Recoil Pressure

#### test_recoil_pressure.cu
**Location**: `/home/yzk/LBMProject/tests/unit/vof/test_recoil_pressure.cu`

**Purpose**: Validate recoil pressure from evaporation

**Tests**:
- Recoil pressure magnitude
- Temperature dependence
- Force direction (outward normal)

**Physics Validated**:
- Recoil pressure: P_recoil = 0.54 × P_sat(T)
- Momentum transfer from vapor jet

**How to Run**:
```bash
ctest -R test_recoil_pressure --output-on-failure
```

**Pass Criteria**:
- Pressure magnitude within 20% of theory
- Force points outward from liquid

**Typical Runtime**: 5-8 seconds

---

### 1.7 VOF Validation Tests (Analytical)

#### test_vof_advection_rotation.cu (Zalesak's Disk)
**Location**: `/home/yzk/LBMProject/tests/validation/vof/test_vof_advection_rotation.cu`

**Purpose**: Classic VOF benchmark - slotted disk rotation

**Tests**:
1. **FullRotation360**
   - Physics: Disk rotates 360° and returns to original shape
   - Setup: 64×64×4 domain, 628 timesteps (2π rotation)
   - Expected: L1 error < 0.15, mass error < 5%

2. **HalfRotation180**
   - Physics: 180° rotation symmetry check
   - Expected: Mass error < 3%

**Reference**: Zalesak (1979), "Fully multidimensional flux-corrected transport"

**Physics Validated**:
- Interface reconstruction accuracy
- Advection scheme quality
- Mass conservation during rotation

**How to Run**:
```bash
ctest -R test_vof_advection_rotation --output-on-failure
```

**Interpreting Results**:
- L1 error measures shape preservation
- Mass error measures conservation
- Compare with literature benchmarks

**Typical Runtime**: 30-60 seconds

---

## 2. Marangoni Effect Tests

Marangoni effects drive thermocapillary flow at liquid-gas interfaces due to surface tension gradients.

### 2.1 Force Calculation

#### test_marangoni_force_magnitude.cu
**Location**: `/home/yzk/LBMProject/tests/unit/marangoni/test_marangoni_force_magnitude.cu`

**Purpose**: Verify Marangoni force magnitude calculation

**Tests**:
- Simple temperature gradient
- Force magnitude vs analytical
- Expected magnitude for LPBF: ~10^8-10^9 N/m³

**Physics Validated**:
- F = |dσ/dT| × |∇T_tangential| × |∇f|
- Temperature coefficient: dσ/dT ≈ -0.00026 N/(m·K) for Ti6Al4V

**How to Run**:
```bash
ctest -R test_marangoni_force_magnitude --output-on-failure
```

**Pass Criteria**:
- Force magnitude within 20% of analytical
- Order of magnitude correct (10^8-10^9 N/m³)

**Typical Runtime**: 5-8 seconds

---

#### test_marangoni_force_direction.cu
**Location**: `/home/yzk/LBMProject/tests/unit/marangoni/test_marangoni_force_direction.cu`

**Purpose**: Verify Marangoni force direction

**Tests**:
- Force tangent to interface
- Force points from hot to cold
- No normal component

**Physics Validated**:
- Tangential projection correctness
- Flow direction (hot → cold)

**How to Run**:
```bash
ctest -R test_marangoni_force_direction --output-on-failure
```

**Pass Criteria**:
- Normal component < 1% of total
- Tangential component correct direction

**Typical Runtime**: 3-5 seconds

---

#### test_interface_geometry.cu
**Location**: `/home/yzk/LBMProject/tests/unit/vof/test_interface_geometry.cu`

**Purpose**: Validate interface normal and tangent calculations

**Tests**:
- Normal vector accuracy
- Tangent vector calculation
- Orthogonality verification

**Physics Validated**:
- Normal: n = -∇f / |∇f|
- Tangent orthogonal to normal: t · n = 0

**How to Run**:
```bash
ctest -R test_interface_geometry --output-on-failure
```

**Pass Criteria**:
- Normal magnitude = 1 ± 0.01
- Orthogonality: |t · n| < 0.01

**Typical Runtime**: 3-5 seconds

---

### 2.2 Gradient & Stability

#### test_marangoni_gradient_limiter.cu
**Location**: `/home/yzk/LBMProject/tests/unit/marangoni/test_marangoni_gradient_limiter.cu`

**Purpose**: Verify gradient limiter prevents excessive forces

**Tests**:
- Extreme temperature gradients
- Limiter activation
- Force capping behavior

**Physics Validated**:
- Gradient limiter threshold
- Numerical stability at extreme gradients

**How to Run**:
```bash
ctest -R test_marangoni_gradient_limiter --output-on-failure
```

**Pass Criteria**:
- Limiter activates above threshold
- Force capped at reasonable value

**Typical Runtime**: 5-8 seconds

---

#### test_marangoni_stability.cu
**Location**: `/home/yzk/LBMProject/tests/unit/marangoni/test_marangoni_stability.cu`

**Purpose**: Long-term stability test

**Tests**:
- 1000 timestep simulation
- No divergence
- Reasonable force magnitude throughout

**Physics Validated**:
- Numerical stability
- No NaN/Inf generation

**How to Run**:
```bash
ctest -R test_marangoni_stability --output-on-failure
```

**Pass Criteria**:
- Completes 1000 steps without NaN
- Force magnitude stays bounded

**Typical Runtime**: 15-20 seconds

---

### 2.3 Integration Tests

#### test_marangoni_flow.cu
**Location**: `/home/yzk/LBMProject/tests/integration/test_marangoni_flow.cu`

**Purpose**: Full Marangoni-driven flow simulation

**Tests**:
- Temperature gradient drives flow
- Flow direction correct (hot → cold)
- Velocity magnitude reasonable

**Physics Validated**:
- Full VOF-thermal-fluid coupling
- Marangoni convection pattern

**How to Run**:
```bash
ctest -R test_marangoni_flow --output-on-failure
```

**Pass Criteria**:
- Flow develops in correct direction
- Velocity magnitude 0.1-10 m/s (realistic)

**Typical Runtime**: 60-120 seconds

---

#### test_marangoni_velocity.cu
**Location**: `/home/yzk/LBMProject/tests/validation/test_marangoni_velocity.cu`

**Purpose**: Validate Marangoni velocity against literature

**Tests**:
- Peak velocity measurement
- Comparison with Khairallah et al. (2016)
- Expected: 0.5-2 m/s for Ti6Al4V

**Reference**: Khairallah et al. (2016), "Marangoni velocity 0.5-2 m/s for Ti6Al4V"

**Physics Validated**:
- Realistic velocity scale
- Literature agreement

**How to Run**:
```bash
ctest -R test_marangoni_velocity --output-on-failure
```

**Interpreting Results**:
- CRITICAL: Velocity in range 0.5-2.0 m/s
- ACCEPTABLE: Velocity in range 0.1-10.0 m/s

**Typical Runtime**: 120-300 seconds

---

#### test_marangoni_system.cu
**Location**: `/home/yzk/LBMProject/tests/integration/test_marangoni_system.cu`

**Purpose**: System-level Marangoni validation

**Tests**:
- Complete melt pool with Marangoni
- Flow pattern validation
- Energy consistency

**Physics Validated**:
- Full system integration
- Physical flow patterns

**How to Run**:
```bash
ctest -R test_marangoni_system --output-on-failure
```

**Typical Runtime**: 120-180 seconds

---

## 3. Thermal Solver Tests

The thermal solver uses Lattice Boltzmann Method with D3Q7 lattice for heat conduction and advection.

### 3.1 Thermal LBM Core

#### test_thermal_lbm.cu
**Location**: `/home/yzk/LBMProject/tests/unit/thermal/test_thermal_lbm.cu`

**Purpose**: Core thermal LBM functionality

**Tests**:
1. **UniformInitialization**
   - Verify uniform temperature field
   - Expected: All cells at T_init ± 1e-5

2. **CustomInitialization**
   - Temperature gradient initialization
   - Expected: Matches input field

3. **PureDiffusion**
   - 1D diffusion with step function
   - Expected: Gaussian profile development

4. **ThermalDiffusionCoefficient**
   - Verify α = (tau - 0.5) × cs²
   - Expected: Error < 1%

5. **TemperatureBounds**
   - Verify T ∈ [0, 7000] K
   - Expected: No excursions outside bounds

**Physics Validated**:
- LBM thermal diffusion equation
- Tau-diffusivity relationship
- Temperature bounds enforcement

**How to Run**:
```bash
ctest -R test_thermal_lbm --output-on-failure
```

**Typical Runtime**: 8-15 seconds

---

#### test_lattice_d3q7.cu
**Location**: `/home/yzk/LBMProject/tests/unit/thermal/test_lattice_d3q7.cu`

**Purpose**: D3Q7 lattice properties validation

**Tests**:
- Lattice vectors
- Weight factors sum to 1
- Isotropy properties
- Equilibrium distribution

**Physics Validated**:
- D3Q7 lattice correctness
- Equilibrium moments

**How to Run**:
```bash
ctest -R test_lattice_d3q7 --output-on-failure
```

**Pass Criteria**:
- Sum of weights = 1.0 ± 1e-6
- Isotropy: ∑c_i = 0

**Typical Runtime**: 2-3 seconds

---

### 3.2 Thermal Stability Tests

#### test_flux_limiter.cu
**Location**: `/home/yzk/LBMProject/tests/unit/stability/test_flux_limiter.cu`

**Purpose**: Verify TVD flux limiter prevents negative populations

**Tests**:
1. **PreventNegativePopulations**
   - Extreme velocity v=5.0 >> cs=0.577
   - Expected: All g_eq ≥ 0

2. **ConservesTemperature**
   - Mass conservation with limiter
   - Expected: ∑g_eq = T (within 0.1%)

3. **VaryingPecletNumbers**
   - Range Pe ~ 0.1 to 50
   - Expected: All stable

4. **LowVelocityUnaffected**
   - Limiter inactive at Pe < 1
   - Expected: Matches analytical

**Physics Validated**:
- TVD flux limiter correctness
- High-Pe stability
- Temperature conservation

**How to Run**:
```bash
ctest -R test_flux_limiter --output-on-failure
```

**Pass Criteria**:
- ALL g_eq ≥ 0 for all velocities
- Temperature conserved within 0.1%

**Critical**: This test MUST pass before any merge to main

**Typical Runtime**: 5-10 seconds

---

#### test_temperature_bounds.cu
**Location**: `/home/yzk/LBMProject/tests/unit/stability/test_temperature_bounds.cu`

**Purpose**: Verify temperature clamping to [0, 7000] K

**Tests**:
1. **UpperBoundEnforcement**
   - Extreme heating → T ≤ 7000 K

2. **LowerBoundEnforcement**
   - Extreme cooling → T ≥ 0 K

3. **NormalRangeUnaffected**
   - T ∈ [300, 5000] K unchanged

4. **RadiationBCWithBounds**
   - No NaN from T^4 term

**Physics Validated**:
- Temperature bounds enforcement
- Radiation BC stability

**How to Run**:
```bash
ctest -R test_temperature_bounds --output-on-failure
```

**Pass Criteria**:
- All T ∈ [0, 7000] K always
- No NaN, no Inf

**Critical**: This test MUST pass before any merge to main

**Typical Runtime**: 5-8 seconds

---

#### test_omega_reduction.cu
**Location**: `/home/yzk/LBMProject/tests/regression/stability/test_omega_reduction.cu`

**Purpose**: Ensure omega_T never exceeds 1.50

**Tests**:
1. **OmegaNeverExceeds1_50**
   - Test range of diffusivities
   - Expected: omega_T ≤ 1.50 for ALL

2. **OmegaCalculationFormula**
   - Verify tau = α/cs² + 0.5

3. **OldLimitNotUsed**
   - Confirm 1.50 cap (not 1.95)

**Physics Validated**:
- Omega cap enforcement
- Stability margin (omega < 2.0)

**How to Run**:
```bash
ctest -R test_omega_reduction --output-on-failure
```

**Pass Criteria**:
- omega_T ≤ 1.50 for ALL diffusivities
- Warning printed when capping applied

**Critical**: Regression test - MUST pass

**Typical Runtime**: 3-5 seconds

---

### 3.3 Thermal Validation Tests

#### test_pure_conduction.cu
**Location**: `/home/yzk/LBMProject/tests/validation/test_pure_conduction.cu`

**Purpose**: Validate thermal diffusion against Gaussian analytical solution

**Tests**:
- 1D heat conduction
- Gaussian initial condition
- Compare with analytical solution

**Physics Validated**:
- Diffusion equation: ∂T/∂t = α∇²T
- Thermal diffusivity accuracy

**How to Run**:
```bash
ctest -R test_pure_conduction --output-on-failure
```

**Pass Criteria**:
- L2 error < 5% vs analytical

**Typical Runtime**: 60-120 seconds

---

#### test_stefan_problem.cu
**Location**: `/home/yzk/LBMProject/tests/validation/test_stefan_problem.cu`

**Purpose**: Validate phase change against moving boundary analytical solution

**Tests**:
- 1D melting with phase change
- Stefan problem analytical solution
- Interface position tracking

**Physics Validated**:
- Phase change correctness
- Moving boundary accuracy
- Latent heat handling

**Reference**: Stefan problem analytical solution

**How to Run**:
```bash
ctest -R test_stefan_problem --output-on-failure
```

**Pass Criteria**:
- Interface position error < 10%
- Temperature profile error < 15%

**Typical Runtime**: 300-600 seconds

---

#### test_high_pe_stability.cu
**Location**: `/home/yzk/LBMProject/tests/integration/stability/test_high_pe_stability.cu`

**Purpose**: End-to-end stability test at high Peclet number (Pe ≈ 10)

**Tests**:
1. **ThermalSolver500Steps**
   - 500-step run at v=5.0, T=2500 K
   - Expected: No divergence

2. **VaryingVelocityStability**
   - Velocity ramp 0 → 10
   - Expected: Stable throughout

3. **HeatSourceWithHighVelocity**
   - Laser heating + Marangoni flow
   - Expected: No instability

**Physics Validated**:
- High-Pe stability
- Stability fixes effectiveness

**Benchmark**: Without fixes, divergence at step ~50-200

**How to Run**:
```bash
ctest -R test_high_pe_stability --output-on-failure
```

**Pass Criteria**:
- No divergence (NaN, Inf) in 500 steps
- T_max < 15000 K
- Simulation completes

**Critical**: End-to-end stability validation

**Typical Runtime**: 60-120 seconds

---

### 3.4 Laser Heating Tests

#### test_laser_source.cu
**Location**: `/home/yzk/LBMProject/tests/unit/laser/test_laser_source.cu`

**Purpose**: Validate laser heat source implementation

**Tests**:
- Gaussian beam profile
- Power deposition calculation
- Absorption coefficient handling

**Physics Validated**:
- Beer-Lambert absorption
- Gaussian spatial profile
- Power normalization

**How to Run**:
```bash
ctest -R test_laser_source --output-on-failure
```

**Pass Criteria**:
- Total power error < 1%
- Profile shape correct

**Typical Runtime**: 3-5 seconds

---

#### test_laser_heating.cu
**Location**: `/home/yzk/LBMProject/tests/integration/test_laser_heating.cu`

**Purpose**: Integration test for laser heating

**Tests**:
- Temperature rise from laser
- Spatial distribution
- Time evolution

**Physics Validated**:
- Laser-thermal coupling
- Realistic heating rates

**How to Run**:
```bash
ctest -R test_laser_heating --output-on-failure
```

**Typical Runtime**: 60-120 seconds

---

#### test_laser_melting.cu
**Location**: `/home/yzk/LBMProject/tests/integration/test_laser_melting.cu`

**Purpose**: Full laser melting with phase change

**Tests**:
- Heating from solid
- Melting transition
- Melt pool formation

**Physics Validated**:
- Laser-thermal-phase coupling
- Realistic melt pool

**How to Run**:
```bash
ctest -R test_laser_melting --output-on-failure
```

**Typical Runtime**: 120-180 seconds

---

## 4. Fluid LBM Tests

The fluid solver uses Lattice Boltzmann Method with D3Q19 lattice for incompressible Navier-Stokes equations.

### 4.1 Fluid LBM Core

#### test_fluid_lbm.cu
**Location**: `/home/yzk/LBMProject/tests/unit/fluid/test_fluid_lbm.cu`

**Purpose**: Core fluid LBM functionality

**Tests**:
1. **Initialization**
   - Verify initial density and velocity

2. **EquilibriumDistribution**
   - Validate equilibrium formula

3. **MacroscopicVariables**
   - Compute density and velocity from f

4. **CollisionOperator**
   - BGK collision correctness

**Physics Validated**:
- LBM fluid equations
- Equilibrium distribution accuracy
- Collision operator correctness

**How to Run**:
```bash
ctest -R test_fluid_lbm --output-on-failure
```

**Typical Runtime**: 8-12 seconds

---

#### test_d3q19.cu
**Location**: `/home/yzk/LBMProject/tests/unit/lattice/test_d3q19.cu`

**Purpose**: D3Q19 lattice properties validation

**Tests**:
- 19 lattice vectors
- Weight factors sum to 1
- Isotropy properties
- Galilean invariance

**Physics Validated**:
- D3Q19 lattice correctness
- Isotropy: ∑c_i c_j = cs² δ_ij

**How to Run**:
```bash
ctest -R test_d3q19 --output-on-failure
```

**Typical Runtime**: 2-3 seconds

---

#### test_bgk.cu
**Location**: `/home/yzk/LBMProject/tests/unit/collision/test_bgk.cu`

**Purpose**: BGK collision operator validation

**Tests**:
- Collision toward equilibrium
- Relaxation time correctness
- Viscosity relationship: ν = cs²(tau - 0.5)

**Physics Validated**:
- BGK operator correctness
- Tau-viscosity relationship

**How to Run**:
```bash
ctest -R test_bgk --output-on-failure
```

**Typical Runtime**: 3-5 seconds

---

#### test_streaming.cu
**Location**: `/home/yzk/LBMProject/tests/unit/streaming/test_streaming.cu`

**Purpose**: Streaming step validation

**Tests**:
- Periodic boundaries
- Proper population propagation
- Conservation during streaming

**Physics Validated**:
- Streaming step correctness
- Boundary condition handling

**How to Run**:
```bash
ctest -R test_streaming --output-on-failure
```

**Typical Runtime**: 3-5 seconds

---

### 4.2 Fluid Boundaries

#### test_fluid_boundaries.cu
**Location**: `/home/yzk/LBMProject/tests/unit/fluid/test_fluid_boundaries.cu`

**Purpose**: Fluid boundary conditions

**Tests**:
- Periodic boundaries
- Inlet/outlet boundaries
- Wall boundaries

**Physics Validated**:
- Boundary condition implementations
- Mass conservation at boundaries

**How to Run**:
```bash
ctest -R test_fluid_boundaries --output-on-failure
```

**Typical Runtime**: 5-8 seconds

---

#### test_no_slip_boundary.cu
**Location**: `/home/yzk/LBMProject/tests/unit/fluid/test_no_slip_boundary.cu`

**Purpose**: No-slip wall boundary validation

**Tests**:
- Velocity at wall = 0
- Bounce-back scheme
- Force on wall

**Physics Validated**:
- No-slip condition enforcement
- Wall stress calculation

**How to Run**:
```bash
ctest -R test_no_slip_boundary --output-on-failure
```

**Pass Criteria**:
- Wall velocity < 1e-6

**Typical Runtime**: 5-8 seconds

---

### 4.3 Fluid Validation Tests

#### test_poiseuille_flow.cu
**Location**: `/home/yzk/LBMProject/tests/integration/test_poiseuille_flow.cu`

**Purpose**: Poiseuille flow validation (channel flow)

**Tests**:
- Parabolic velocity profile
- Comparison with analytical solution
- Pressure drop validation

**Physics Validated**:
- Navier-Stokes equation accuracy
- Viscosity correctness
- Analytical: u(y) = (dp/dx) × y(H-y) / (2μ)

**How to Run**:
```bash
ctest -R test_poiseuille_flow --output-on-failure
```

**Pass Criteria**:
- Velocity profile error < 5%
- Peak velocity error < 2%

**Typical Runtime**: 60-120 seconds

---

#### test_poiseuille_flow_fluidlbm.cu
**Location**: `/home/yzk/LBMProject/tests/integration/test_poiseuille_flow_fluidlbm.cu`

**Purpose**: Poiseuille flow using FluidLBM class

**Tests**:
- Same as above but using high-level API
- Convergence to steady state

**How to Run**:
```bash
ctest -R test_poiseuille_flow_fluidlbm --output-on-failure
```

**Typical Runtime**: 120-180 seconds

---

## 5. Phase Change Tests

Phase change module handles solid-liquid-gas transitions with enthalpy method.

### 5.1 Phase Change Core

#### test_liquid_fraction.cu
**Location**: `/home/yzk/LBMProject/tests/unit/phase_change/test_liquid_fraction.cu`

**Purpose**: Liquid fraction calculation validation

**Tests**:
1. **SolidPhase**
   - T < T_solidus → f_l = 0

2. **LiquidPhase**
   - T > T_liquidus → f_l = 1

3. **MushyZone**
   - T_solidus < T < T_liquidus → 0 < f_l < 1
   - Linear interpolation

**Physics Validated**:
- Liquid fraction formula
- Phase transition behavior

**How to Run**:
```bash
ctest -R test_liquid_fraction --output-on-failure
```

**Pass Criteria**:
- Correct phase identification
- Smooth transition in mushy zone

**Typical Runtime**: 3-5 seconds

---

#### test_enthalpy.cu
**Location**: `/home/yzk/LBMProject/tests/unit/phase_change/test_enthalpy.cu`

**Purpose**: Enthalpy method validation

**Tests**:
- Enthalpy-temperature relationship
- Latent heat handling
- Inverse function (T from H)

**Physics Validated**:
- Enthalpy method: H = c_p T + f_l L
- Newton-Raphson solver for T(H)

**How to Run**:
```bash
ctest -R test_enthalpy --output-on-failure
```

**Pass Criteria**:
- Temperature recovery error < 1 K
- Latent heat correctly included

**Typical Runtime**: 5-8 seconds

---

#### test_phase_properties.cu
**Location**: `/home/yzk/LBMProject/tests/unit/phase_change/test_phase_properties.cu`

**Purpose**: Phase-dependent material properties

**Tests**:
- Density variation with phase
- Viscosity variation (Darcy model)
- Thermal conductivity variation

**Physics Validated**:
- Property interpolation
- Physical ranges

**How to Run**:
```bash
ctest -R test_phase_properties --output-on-failure
```

**Typical Runtime**: 3-5 seconds

---

### 5.2 Phase Change Validation

#### test_stefan_1d.cu
**Location**: `/home/yzk/LBMProject/tests/integration/test_stefan_1d.cu`

**Purpose**: 1D Stefan problem validation

**Tests**:
- Same as validation/test_stefan_problem.cu
- Moving boundary tracking
- Interface position vs analytical

**How to Run**:
```bash
ctest -R test_stefan_1d --output-on-failure
```

**Typical Runtime**: 180-300 seconds

---

#### test_phase_change_robustness.cu
**Location**: `/home/yzk/LBMProject/tests/validation/test_phase_change_robustness.cu`

**Purpose**: Robustness test for phase change solver

**Tests**:
1. **NoNaNDuringMelting**
   - Extreme heating
   - Expected: No NaN

2. **TemperatureInPhysicalRange**
   - T ∈ [0, 7000] K

3. **StableThroughLaserShutoff**
   - Laser on/off transitions

4. **StableInMushyZone**
   - Extended time in mushy zone

**Physics Validated**:
- Numerical robustness
- Stability at phase transitions

**How to Run**:
```bash
ctest -R test_phase_change_robustness --output-on-failure
```

**Pass Criteria**:
- No NaN/Inf
- Temperature stays physical

**Typical Runtime**: 60-120 seconds

---

### 5.3 Material Properties

#### test_materials.cu
**Location**: `/home/yzk/LBMProject/tests/unit/materials/test_materials.cu`

**Purpose**: Material database validation

**Tests**:
- Property lookup for different materials
- Temperature-dependent properties
- Physical value ranges

**Physics Validated**:
- Material property correctness
- Temperature interpolation

**How to Run**:
```bash
ctest -R test_materials --output-on-failure
```

**Typical Runtime**: 2-3 seconds

---

## 6. Multiphysics Coupling Tests

Multiphysics tests validate the interaction between different physics modules.

### 6.1 Thermal-Fluid Coupling

#### test_thermal_fluid_coupling.cu
**Location**: `/home/yzk/LBMProject/tests/integration/test_thermal_fluid_coupling.cu`

**Purpose**: Natural convection validation

**Tests**:
- Buoyancy-driven flow
- Temperature-velocity coupling
- Boussinesq approximation

**Physics Validated**:
- Buoyancy force: F = ρ β g (T - T_ref)
- Thermal advection by flow
- Bidirectional coupling

**How to Run**:
```bash
ctest -R test_thermal_fluid_coupling --output-on-failure
```

**Pass Criteria**:
- Flow develops upward in hot region
- Temperature field advects correctly

**Typical Runtime**: 120-300 seconds

---

#### test_mp_thermal_fluid_coupling.cu
**Location**: `/home/yzk/LBMProject/tests/integration/multiphysics/test_thermal_fluid_coupling.cu`

**Purpose**: Same as above but using MultiphysicsSolver

**How to Run**:
```bash
ctest -R test_mp_thermal_fluid --output-on-failure
```

**Typical Runtime**: 120-180 seconds

---

### 6.2 VOF-Fluid Coupling

#### test_vof_fluid_coupling.cu
**Location**: `/home/yzk/LBMProject/tests/integration/multiphysics/test_vof_fluid_coupling.cu`

**Purpose**: Surface tension force coupling to fluid

**Tests**:
- Surface tension drives flow
- Interface advection by fluid velocity
- Capillary wave damping

**Physics Validated**:
- Surface tension force coupling
- VOF advection by LBM velocity
- Bidirectional coupling

**How to Run**:
```bash
ctest -R test_vof_fluid_coupling --output-on-failure
```

**Pass Criteria**:
- Surface tension force affects velocity
- Interface advects with flow

**Typical Runtime**: 120-180 seconds

---

### 6.3 Thermal-VOF Coupling

#### test_thermal_vof_coupling.cu
**Location**: `/home/yzk/LBMProject/tests/integration/multiphysics/test_thermal_vof_coupling.cu`

**Purpose**: Evaporation coupling between thermal and VOF

**Tests**:
- Temperature drives evaporation
- Mass loss updates VOF field
- Energy conservation

**Physics Validated**:
- Evaporation mass flux
- VOF mass loss
- Evaporative cooling

**How to Run**:
```bash
ctest -R test_thermal_vof_coupling --output-on-failure
```

**Pass Criteria**:
- Evaporation occurs at T > T_boil
- VOF mass decreases correctly
- Energy balance satisfied

**Typical Runtime**: 120-180 seconds

---

### 6.4 Phase-Fluid Coupling

#### test_phase_fluid_coupling.cu
**Location**: `/home/yzk/LBMProject/tests/integration/multiphysics/test_phase_fluid_coupling.cu`

**Purpose**: Darcy damping in solid/mushy regions

**Tests**:
- Velocity reduction in solid
- Mushy zone damping
- No damping in liquid

**Physics Validated**:
- Darcy model: F = -C × (1 - f_l)² × u
- Phase-dependent viscosity

**How to Run**:
```bash
ctest -R test_phase_fluid_coupling --output-on-failure
```

**Pass Criteria**:
- Velocity ~ 0 in solid
- Gradual damping in mushy zone
- No damping in liquid (f_l = 1)

**Typical Runtime**: 120-180 seconds

---

### 6.5 Force Balance Tests

#### test_force_balance_static.cu
**Location**: `/home/yzk/LBMProject/tests/integration/multiphysics/test_force_balance_static.cu**

**Purpose**: Verify force equilibrium in static case

**Tests**:
- Buoyancy vs gravity
- Surface tension vs pressure
- Net force = 0 at equilibrium

**Physics Validated**:
- Force balance correctness
- Equilibrium states

**How to Run**:
```bash
ctest -R test_force_balance_static --output-on-failure
```

**Pass Criteria**:
- Net force < 1% of individual forces

**Typical Runtime**: 30-60 seconds

---

#### test_force_magnitude_ordering.cu
**Location**: `/home/yzk/LBMProject/tests/integration/multiphysics/test_force_magnitude_ordering.cu`

**Purpose**: Verify realistic force magnitude ordering

**Tests**:
- Compare magnitudes: Marangoni, buoyancy, surface tension
- Check order of magnitude consistency
- Validate against literature

**Physics Validated**:
- Realistic force scales
- Physical consistency

**How to Run**:
```bash
ctest -R test_force_magnitude_ordering --output-on-failure
```

**Typical Runtime**: 30-60 seconds

---

#### test_force_direction.cu
**Location**: `/home/yzk/LBMProject/tests/integration/multiphysics/test_force_direction.cu`

**Purpose**: Verify force directions

**Tests**:
- Buoyancy: upward in hot region
- Marangoni: hot → cold tangentially
- Surface tension: inward (toward liquid)

**Physics Validated**:
- Force direction correctness

**How to Run**:
```bash
ctest -R test_force_direction --output-on-failure
```

**Typical Runtime**: 30-60 seconds

---

## 7. Energy Conservation Tests

Energy conservation is critical for physical accuracy.

### 7.1 Energy Conservation Tests

#### test_energy_conservation_no_source.cu
**Location**: `/home/yzk/LBMProject/tests/integration/multiphysics/test_energy_conservation_no_source.cu`

**Purpose**: Energy conservation without heat source

**Tests**:
- Isolated system
- Total energy constant
- No drift over time

**Physics Validated**:
- Energy conservation: dE/dt = 0
- No spurious energy generation

**How to Run**:
```bash
ctest -R test_energy_conservation_no_source --output-on-failure
```

**Pass Criteria**:
- Energy drift < 1% over 1000 steps

**Typical Runtime**: 30-60 seconds

---

#### test_energy_conservation_laser_only.cu
**Location**: `/home/yzk/LBMProject/tests/integration/multiphysics/test_energy_conservation_laser_only.cu`

**Purpose**: Energy balance with laser source

**Tests**:
- Energy input = laser power × time
- Energy output = conduction + radiation
- Balance equation

**Physics Validated**:
- Energy balance: ΔE = Q_in - Q_out
- Laser power accounting

**How to Run**:
```bash
ctest -R test_energy_conservation_laser_only --output-on-failure
```

**Pass Criteria**:
- Energy balance error < 5%

**Typical Runtime**: 60-120 seconds

---

#### test_energy_conservation_full.cu
**Location**: `/home/yzk/LBMProject/tests/integration/multiphysics/test_energy_conservation_full.cu`

**Purpose**: Full energy balance with all effects

**Tests**:
- Laser input
- Conduction losses
- Radiation losses
- Evaporation losses (latent heat)
- Total balance

**Physics Validated**:
- Complete energy accounting
- All energy pathways

**How to Run**:
```bash
ctest -R test_energy_conservation_full --output-on-failure
```

**Pass Criteria**:
- Energy balance error < 10%

**Typical Runtime**: 120-180 seconds

---

### 7.2 Evaporation Energy Balance

#### test_evaporation_energy_balance.cu
**Location**: `/home/yzk/LBMProject/tests/integration/test_evaporation_energy_balance.cu`

**Purpose**: Validate evaporative cooling energy

**Tests**:
- Evaporation removes latent heat
- Temperature drop verification
- Energy balance with evaporation

**Physics Validated**:
- Evaporative cooling: Q = J_evap × L_v
- Energy removal rate

**How to Run**:
```bash
ctest -R test_evaporation_energy_balance --output-on-failure
```

**Pass Criteria**:
- Cooling rate matches latent heat removal

**Typical Runtime**: 120-180 seconds

---

#### test_evaporation_hertz_knudsen.cu
**Location**: `/home/yzk/LBMProject/tests/validation/test_evaporation_hertz_knudsen.cu`

**Purpose**: Validate Hertz-Knudsen formula

**Tests**:
- Mass flux calculation
- Temperature dependence
- Comparison with formula

**Physics Validated**:
- Hertz-Knudsen: J = α_e × P_sat(T) / sqrt(2πRT)

**How to Run**:
```bash
ctest -R test_evaporation_hertz_knudsen --output-on-failure
```

**Pass Criteria**:
- Mass flux error < 10%

**Typical Runtime**: 30-60 seconds

---

## 8. Stability & Validation Tests

### 8.1 CFL Stability

#### test_cfl_stability.cu
**Location**: `/home/yzk/LBMProject/tests/validation/test_cfl_stability.cu`

**Purpose**: CFL condition validation

**Tests**:
- Analytical CFL calculation
- Stability limit verification
- Timestep selection

**Physics Validated**:
- CFL condition: u·Δt/Δx < CFL_max
- Stability analysis

**How to Run**:
```bash
ctest -R test_cfl_stability --output-on-failure
```

**Typical Runtime**: 5-10 seconds

---

#### test_cfl_limiting_effectiveness.cu
**Location**: `/home/yzk/LBMProject/tests/integration/multiphysics/test_cfl_limiting_effectiveness.cu`

**Purpose**: CFL limiter prevents instability

**Tests**:
- High velocity without limiter → diverges
- High velocity with limiter → stable

**Physics Validated**:
- CFL limiter effectiveness
- Stability improvement

**How to Run**:
```bash
ctest -R test_cfl_limiting_effectiveness --output-on-failure
```

**Typical Runtime**: 60-120 seconds

---

#### test_extreme_gradients.cu
**Location**: `/home/yzk/LBMProject/tests/integration/multiphysics/test_extreme_gradients.cu`

**Purpose**: Stability under extreme conditions

**Tests**:
- Very steep temperature gradients
- High velocity gradients
- Sharp interface deformation

**Physics Validated**:
- Numerical stability
- Robustness to extremes

**How to Run**:
```bash
ctest -R test_extreme_gradients --output-on-failure
```

**Pass Criteria**:
- No divergence under extreme conditions

**Typical Runtime**: 120-180 seconds

---

### 8.2 Subcycling Tests

#### test_vof_subcycling_convergence.cu
**Location**: `/home/yzk/LBMProject/tests/integration/multiphysics/test_vof_subcycling_convergence.cu`

**Purpose**: VOF subcycling convergence validation

**Tests**:
- Compare subcycles = 1, 5, 10
- Convergence with increasing subcycles
- Accuracy improvement

**Physics Validated**:
- Subcycling correctness
- Temporal accuracy

**How to Run**:
```bash
ctest -R test_vof_subcycling_convergence --output-on-failure
```

**Pass Criteria**:
- Results converge with more subcycles
- Error decreases monotonically

**Typical Runtime**: 180-300 seconds

---

#### test_subcycling_1_vs_10.cu
**Location**: `/home/yzk/LBMProject/tests/integration/multiphysics/test_subcycling_1_vs_10.cu`

**Purpose**: Direct comparison of 1 vs 10 subcycles

**Tests**:
- Same physics, different subcycles
- Quantify accuracy improvement

**How to Run**:
```bash
ctest -R test_subcycling_1_vs_10 --output-on-failure
```

**Typical Runtime**: 120-180 seconds

---

### 8.3 Unit Conversion Tests

#### test_unit_converter.cpp
**Location**: `/home/yzk/LBMProject/tests/test_unit_converter.cpp`

**Purpose**: Unit conversion validation

**Tests**:
- Physical → lattice → physical roundtrip
- Dimensional consistency
- Conversion factor correctness

**Physics Validated**:
- Unit conversion correctness
- Dimensionless LBM parameters

**How to Run**:
```bash
ctest -R test_unit_converter --output-on-failure
```

**Pass Criteria**:
- Roundtrip error < 1e-6

**Typical Runtime**: 2-3 seconds

---

#### test_unit_conversion_roundtrip.cu
**Location**: `/home/yzk/LBMProject/tests/integration/multiphysics/test_unit_conversion_roundtrip.cu`

**Purpose**: Full system unit conversion test

**How to Run**:
```bash
ctest -R test_unit_conversion_roundtrip --output-on-failure
```

**Typical Runtime**: 10-20 seconds

---

### 8.4 Regression Tests

#### test_regression_50W.cu
**Location**: `/home/yzk/LBMProject/tests/validation/test_regression_50W.cu`

**Purpose**: Regression test with 50W baseline

**Tests**:
- Known good configuration
- Compare with baseline results
- Detect unintended changes

**Physics Validated**:
- System-level regression prevention

**How to Run**:
```bash
ctest -R test_regression_50W --output-on-failure
```

**Pass Criteria**:
- Results match baseline within tolerance

**Typical Runtime**: 300-600 seconds

---

#### test_substrate_bc_stability.cu
**Location**: `/home/yzk/LBMProject/tests/regression/test_substrate_bc_stability.cu`

**Purpose**: Substrate boundary condition stability

**Tests**:
- Long-term stability
- Cooling rate correctness
- No spurious oscillations

**Physics Validated**:
- Substrate BC implementation

**How to Run**:
```bash
ctest -R test_substrate_bc_stability --output-on-failure
```

**Typical Runtime**: 120-180 seconds

---

### 8.5 Validation Suite

#### test_week3_readiness.cu
**Location**: `/home/yzk/LBMProject/tests/validation/test_week3_readiness.cu`

**Purpose**: Comprehensive readiness validation

**Tests**:
- Critical requirement validation
- Scoring system (100 points)
- Go/no-go decision criteria

**Decision Criteria**:
- FULL GO: Score ≥ 85/100
- CONDITIONAL GO: Score 70-84/100
- NO GO: Score < 70/100

**How to Run**:
```bash
ctest -R test_week3_readiness --output-on-failure
```

**Typical Runtime**: 180-300 seconds

---

#### test_divergence_free.cu
**Location**: `/home/yzk/LBMProject/tests/validation/test_divergence_free.cu`

**Purpose**: Verify incompressibility (div(u) = 0)

**Tests**:
1. **ZeroForce**
   - |div(u)| < 1e-3 with no force

2. **UniformForce**
   - |div(u)| < 1e-3 with uniform force

3. **SpatiallyVaryingForce**
   - |div(u)| < 1e-3 with varying force

**Physics Validated**:
- Incompressibility enforcement
- LBM accuracy for ∇·u = 0

**How to Run**:
```bash
ctest -R test_divergence_free --output-on-failure
```

**Pass Criteria**:
- |div(u)| < 1e-3 everywhere

**Critical**: VOF mass conservation depends on this

**Typical Runtime**: 60-120 seconds

---

#### test_realistic_lpbf_initial_conditions.cu
**Location**: `/home/yzk/LBMProject/tests/validation/test_realistic_lpbf_initial_conditions.cu`

**Purpose**: Validate realistic LPBF starting conditions

**Tests**:
- Initial T ≈ 300 K (not pre-melted)
- Laser module enabled
- Temperature increases with laser on

**Physics Validated**:
- Realistic initial conditions
- Laser heating functionality

**How to Run**:
```bash
ctest -R test_realistic_lpbf --output-on-failure
```

**Typical Runtime**: 60-120 seconds

---

## Running Tests

### Command Line

#### Run All Tests
```bash
cd /home/yzk/LBMProject
./RUN_ALL_PHYSICS_TESTS.sh
```

This will:
1. Run all test categories sequentially
2. Generate detailed report in `test_reports/`
3. Highlight failures
4. Exit with code 0 (pass) or 1 (fail)

#### Run by Category
```bash
cd /home/yzk/LBMProject/build

# VOF tests
ctest -R "vof" --output-on-failure

# Marangoni tests
ctest -R "marangoni" --output-on-failure

# Thermal tests
ctest -R "thermal" --output-on-failure

# Fluid tests
ctest -R "fluid|poiseuille" --output-on-failure

# Multiphysics tests
ctest -R "test_mp|test_.*_coupling" --output-on-failure

# Energy conservation tests
ctest -R "energy" --output-on-failure

# Stability tests
ctest -R "stability|flux_limiter|temperature_bounds" --output-on-failure
```

#### Run Single Test
```bash
cd /home/yzk/LBMProject/build
ctest -R test_vof_advection_uniform --output-on-failure --verbose
```

#### Run Tests by Speed

**Fast tests (< 30s)** - Pre-commit:
```bash
ctest -R "flux_limiter|temperature_bounds|omega_reduction|vof_advection_uniform|test_d3q19|test_bgk" --output-on-failure
```

**Medium tests (30s - 5min)** - Pre-push:
```bash
ctest -R "test_marangoni_flow|test_thermal_fluid_coupling|test_high_pe_stability" --output-on-failure
```

**Slow tests (> 5min)** - Nightly:
```bash
ctest -R "test_grid_convergence|test_timestep_convergence|test_regression_50W|test_stefan_problem" --output-on-failure
```

#### Run Tests by Label
```bash
# Critical tests only
ctest -L critical --output-on-failure

# Integration tests only
ctest -L integration --output-on-failure

# Validation tests only
ctest -L validation --output-on-failure

# Week 1 fixes
ctest -L week1 --output-on-failure

# Week 2 validation
ctest -L week2 --output-on-failure
```

### Build Tests

#### Build All Tests
```bash
cd /home/yzk/LBMProject/build
cmake --build . -j8  # Use 8 parallel jobs
```

#### Build Specific Test
```bash
cmake --build . --target test_vof_advection_uniform -j4
```

#### Rebuild After Code Changes
```bash
cd /home/yzk/LBMProject/build
cmake --build . -j8
ctest --rerun-failed --output-on-failure
```

---

## Interpreting Results

### Test Output Format

#### Passing Test
```
[==========] Running 1 test from 1 test suite.
[----------] Global test environment set-up.
[----------] 1 test from VOFAdvectionTest
[ RUN      ] VOFAdvectionTest.InterfaceDisplacement
[       OK ] VOFAdvectionTest.InterfaceDisplacement (237 ms)
[----------] 1 test from VOFAdvectionTest (237 ms total)

[----------] Global test environment tear-down
[==========] 1 test from 1 test suite ran. (237 ms total)
[  PASSED  ] 1 test.
```

#### Failing Test
```
[==========] Running 1 test from 1 test suite.
[----------] Global test environment set-up.
[----------] 1 test from VOFAdvectionTest
[ RUN      ] VOFAdvectionTest.InterfaceDisplacement
/path/test_vof_advection_uniform.cu:123: Failure
Expected: displacement_error < 0.5
  Actual: 0.87
Interface displacement error too large!
[  FAILED  ] VOFAdvectionTest.InterfaceDisplacement (245 ms)
[----------] 1 test from VOFAdvectionTest (245 ms total)

[----------] Global test environment tear-down
[==========] 1 test from 1 test suite ran. (245 ms total)
[  PASSED  ] 0 tests.
[  FAILED  ] 1 test, listed below:
[  FAILED  ] VOFAdvectionTest.InterfaceDisplacement
```

### Key Metrics

#### VOF Tests
- **Displacement error**: Should be < 0.5 cells
- **Mass error**: Should be < 1-2%
- **Interface thickness**: Monitor numerical diffusion
- **Curvature error**: < 10-30% depending on resolution

#### Thermal Tests
- **Temperature conservation**: < 0.1% error
- **L2 error vs analytical**: < 5%
- **No negative populations**: Critical failure if violated
- **Temperature bounds**: [0, 7000] K

#### Fluid Tests
- **Velocity profile error**: < 5%
- **Divergence**: |div(u)| < 1e-3
- **Mass conservation**: < 0.1%

#### Energy Tests
- **Energy balance error**: < 5-10%
- **Drift over time**: < 1% per 1000 steps

### Common Failure Modes

#### 1. Numerical Instability
**Symptoms**: NaN, Inf, divergence

**Causes**:
- Flux limiter disabled
- Temperature bounds not enforced
- Omega > 1.50
- CFL violation

**Action**: Check stability tests first

#### 2. Mass Conservation Failure
**Symptoms**: Mass error > tolerance

**Causes**:
- Advection scheme bug
- Boundary condition leak
- Unit conversion error

**Action**: Check mass conservation tests, verify boundaries

#### 3. Energy Balance Failure
**Symptoms**: Energy balance error > 10%

**Causes**:
- Missing energy source/sink
- Unit conversion error
- Timestep scaling issue

**Action**: Check individual energy terms, verify scaling

#### 4. Accuracy Degradation
**Symptoms**: Increasing error vs analytical

**Causes**:
- Numerical diffusion increase
- Resolution too coarse
- Timestep too large

**Action**: Run convergence tests, check grid/timestep independence

---

## Adding New Tests

### Test Template

```cpp
/**
 * @file test_my_new_feature.cu
 * @brief Unit tests for MyNewFeature
 */

#include <gtest/gtest.h>
#include "physics/my_new_feature.h"
#include <cuda_runtime.h>
#include <vector>
#include <cmath>

using namespace lbm::physics;

// Test fixture
class MyNewFeatureTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Initialize CUDA, allocate memory
    }

    void TearDown() override {
        // Cleanup
    }

    // Helper functions
};

// Test 1: Basic functionality
TEST_F(MyNewFeatureTest, BasicFunctionality) {
    // Arrange
    // ... setup test conditions

    // Act
    // ... execute the code under test

    // Assert
    EXPECT_NEAR(result, expected, tolerance);
}

// Test 2: Edge case
TEST_F(MyNewFeatureTest, EdgeCase) {
    // ...
}

// Test 3: Error handling
TEST_F(MyNewFeatureTest, ErrorHandling) {
    // ...
}
```

### Adding to CMakeLists.txt

```cmake
# Add to tests/unit/my_category/CMakeLists.txt
add_cuda_test(test_my_new_feature unit/my_category/test_my_new_feature.cu)
```

### Test Design Principles

1. **Independent**: Each test should run independently
2. **Repeatable**: Same input → same output always
3. **Self-checking**: Assert conditions, don't just print
4. **Fast**: Unit tests < 5s, integration < 5min
5. **Focused**: One test per behavior/requirement

### Physics Validation Checklist

For physics validation tests:

1. Identify analytical solution or reference
2. Set up equivalent numerical scenario
3. Run to steady state or target time
4. Extract relevant quantities
5. Compare with analytical/reference
6. Document tolerance rationale
7. Include physics equations in comments

### Documentation Requirements

Each test file should include:

1. **File header**: Purpose, physics validated, references
2. **Test descriptions**: What, why, expected behavior
3. **Tolerances**: Justified pass criteria
4. **Runtime estimate**: Typical execution time

---

## Test Reports

### Report Location
```
/home/yzk/LBMProject/test_reports/
├── test_summary_YYYYMMDD_HHMMSS.txt
├── failed_tests_YYYYMMDD_HHMMSS.txt
└── ...
```

### Report Contents

1. **Header**: Date, build directory, system info
2. **Per-category results**: Pass/fail counts
3. **Summary**: Total tests, pass rate, failures
4. **Failed test list**: Names and error info

### Archiving

Reports are timestamped and accumulate over time. Recommended cleanup:

```bash
# Keep last 30 days
find /home/yzk/LBMProject/test_reports -mtime +30 -delete

# Keep only last 10 reports
ls -t /home/yzk/LBMProject/test_reports/test_summary_*.txt | tail -n +11 | xargs rm
```

---

## Continuous Integration

### Pre-commit Hook

Create `.git/hooks/pre-commit`:

```bash
#!/bin/bash
cd /home/yzk/LBMProject/build
ctest -R "flux_limiter|temperature_bounds|omega_reduction" --output-on-failure
if [ $? -ne 0 ]; then
    echo "ERROR: Critical stability tests failed. Commit blocked."
    exit 1
fi
```

### Pre-push Hook

Create `.git/hooks/pre-push`:

```bash
#!/bin/bash
cd /home/yzk/LBMProject/build
ctest -R "stability|high_pe" --output-on-failure
if [ $? -ne 0 ]; then
    echo "ERROR: Stability tests failed. Push blocked."
    exit 1
fi
```

### Nightly CI

Run comprehensive validation suite nightly:

```bash
#!/bin/bash
# nightly_tests.sh
cd /home/yzk/LBMProject
./RUN_ALL_PHYSICS_TESTS.sh
```

---

## Summary

### Test Suite Statistics

- **Total test files**: 150+
- **Unit tests**: ~80 (fast)
- **Integration tests**: ~50 (medium)
- **Validation tests**: ~20 (slow)
- **Total coverage**: 8 major categories

### Critical Tests (MUST PASS)

1. `test_flux_limiter` - Prevents thermal instability
2. `test_temperature_bounds` - Prevents NaN
3. `test_omega_reduction` - Stability regression
4. `test_high_pe_stability` - End-to-end stability
5. `test_divergence_free` - Incompressibility for VOF
6. `test_energy_conservation_*` - Energy accounting

### Recommended Workflow

**Daily Development**:
1. Make code changes
2. Build: `cmake --build . -j8`
3. Quick test: `ctest -R "flux_limiter|temperature_bounds"`
4. If pass: Commit
5. If fail: Debug, fix, repeat

**Before Push**:
1. Run medium tests: `ctest -R "stability|high_pe"`
2. If pass: Push
3. If fail: Debug critical issues

**Before Release**:
1. Run full suite: `./RUN_ALL_PHYSICS_TESTS.sh`
2. Verify all critical tests pass
3. Review validation tests
4. Check energy balance
5. Tag release

---

## References

1. **LBM Theory**:
   - He, X., Chen, S., & Doolen, G. D. (1998). *A novel thermal model for the lattice Boltzmann method*.
   - Mohamad, A.A. (2011). *Lattice Boltzmann Method*. Springer.

2. **VOF Methods**:
   - Hirt, C.W., & Nichols, B.D. (1981). *Volume of fluid (VOF) method for the dynamics of free boundaries*.
   - Zalesak, S.T. (1979). *Fully multidimensional flux-corrected transport*.

3. **LPBF Physics**:
   - Khairallah, S.A., et al. (2016). *Laser powder-bed fusion additive manufacturing*.

4. **Project Documentation**:
   - `/home/yzk/LBMProject/tests/VOF_TEST_SUITE_SUMMARY.md`
   - `/home/yzk/LBMProject/tests/STABILITY_TESTS_README.md`
   - `/home/yzk/LBMProject/tests/WEEK1_TEST_QUICK_REFERENCE.md`

---

## Contact & Support

For test-related questions:

- **Test Suite Design**: See this guide
- **Physics Validation**: Check individual test documentation
- **Bug Reports**: Include test name, output, and conditions
- **New Test Requests**: Follow template in "Adding New Tests"

**End of Test Guide**
