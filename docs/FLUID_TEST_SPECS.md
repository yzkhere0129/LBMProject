# FluidLBM Comprehensive Test Specifications

**Project**: LBM-CUDA Metal AM Simulation Framework
**Module**: FluidLBM (D3Q19 Incompressible Navier-Stokes Solver)
**Purpose**: Standalone physical validation tests for fluid flow solver
**Author**: LBM Platform Architect
**Date**: 2025-12-03

---

## Overview

This document provides comprehensive test specifications for the FluidLBM module. Each test validates a specific physical phenomenon with analytical solutions, clear success criteria, and implementation guidance.

**Design Philosophy:**
- Each test is **standalone** and **self-contained**
- Tests validate **physical correctness**, not just code execution
- All tests include **analytical solutions** for quantitative comparison
- Tests cover **fundamental fluid mechanics** phenomena
- Tests verify **numerical accuracy**, **stability**, and **boundary conditions**

---

## Test Suite Organization

### Unit Tests (Isolated Component Testing)
- Individual boundary conditions
- Force application mechanisms
- Viscosity verification

### Integration Tests (Complete Flow Solutions)
- Poiseuille flow (pressure-driven channel flow)
- Couette flow (shear-driven flow)
- Taylor-Green vortex decay
- Lid-driven cavity

### Validation Tests (Physical Benchmarks)
- Incompressibility verification
- Mass conservation
- Stability at high Reynolds numbers

---

## Test 1: Poiseuille Flow (Channel Flow)

### Physical Description
Pressure-driven flow between two parallel plates. This is the most fundamental validation test for incompressible flow solvers.

### Test Configuration
```
Domain:     nx=3, ny=64, nz=3 (2D-like geometry)
BCs:        Periodic in x,z; No-slip walls at y=0 and y=ny-1
IC:         u=0, v=0, w=0, ρ=ρ₀
Forcing:    Uniform body force fx = -dp/dx (mimics pressure gradient)
Duration:   10000 time steps (steady state)
```

### Analytical Solution
For steady-state flow between parallel plates separated by height H:

```
u(y) = (1/(2μ)) × (-dp/dx) × y × (H - y)

where:
  μ = ρ₀ × ν (dynamic viscosity)
  H = ny - 1 (channel height in lattice units)
  y ∈ [0, H]

Maximum velocity (at y = H/2):
  u_max = (H²/8μ) × (-dp/dx)

Average velocity:
  u_avg = (2/3) × u_max

Wall shear stress:
  τ_wall = μ × (du/dy)|_wall = (H/2) × (-dp/dx)
```

### LBM Parameters
```cpp
float nu = 0.1f;                    // Kinematic viscosity
float rho0 = 1.0f;                  // Reference density
float dp_dx = -1e-4f;               // Pressure gradient
float fx = -dp_dx / rho0;           // Body force per unit mass
float tau = nu / D3Q19::CS2 + 0.5f; // Relaxation time
float omega = 1.0f / tau;           // Relaxation parameter

// Stability check
ASSERT_TRUE(tau > 0.5f && omega < 2.0f);
```

### Expected Results
```
Velocity profile:  Parabolic u(y) = u_max × [1 - (2y/H - 1)²]
Symmetry:          u(y) = u(H - y)
Max velocity:      At y = H/2
Wall velocity:     u(0) = u(H) = 0 (within 1e-8)
Ratio:             u_avg / u_max ≈ 2/3
```

### Validation Metrics
```cpp
// 1. L2 relative error (interior points only)
float L2_error = sqrt(Σ(u_LBM - u_analytical)²) / sqrt(Σ u_analytical²);
EXPECT_LT(L2_error, 0.05);  // < 5%

// 2. Maximum velocity error
float u_max_error = |u_max_LBM - u_max_analytical| / u_max_analytical;
EXPECT_LT(u_max_error, 0.035);  // < 3.5%

// 3. Average velocity error
float u_avg_error = |u_avg_LBM - u_avg_analytical| / u_avg_analytical;
EXPECT_LT(u_avg_error, 0.03);  // < 3%

// 4. Profile symmetry
for (int y = 1; y < ny/2; ++y) {
    float symmetry_error = |u(y) - u(ny-1-y)| / (|u(y)| + |u(ny-1-y)|);
    EXPECT_LT(symmetry_error, 0.01);  // < 1%
}

// 5. Wall boundary condition
EXPECT_LT(|u(0)|, 1e-8);
EXPECT_LT(|u(ny-1)|, 1e-8);

// 6. Mass flow rate
float Q_numerical = Σ u(y) × dy;
float Q_analytical = u_avg × H;
EXPECT_NEAR(Q_numerical, Q_analytical, 0.03);
```

### GTest Implementation Outline
```cpp
TEST_F(FluidLBMTest, PoiseuilleFlowValidation) {
    // 1. Setup domain with wall boundaries
    const int nx=3, ny=64, nz=3;
    FluidLBM solver(nx, ny, nz, nu, rho0,
                    BoundaryType::PERIODIC,  // x
                    BoundaryType::WALL,      // y (no-slip walls)
                    BoundaryType::PERIODIC); // z

    // 2. Initialize at rest
    solver.initialize(rho0, 0.0f, 0.0f, 0.0f);

    // 3. Time evolution with body force
    for (int step = 0; step < 10000; ++step) {
        solver.computeMacroscopic();
        solver.collisionBGK(fx, 0.0f, 0.0f);
        solver.streaming();
        solver.applyBoundaryConditions(1); // Bounce-back
    }

    // 4. Extract velocity profile and compare with analytical solution
    // (See existing test_poiseuille_flow.cu for complete implementation)
}
```

---

## Test 2: Couette Flow (Shear-Driven Flow)

### Physical Description
Flow driven by a moving wall at constant velocity. Pure shear flow with linear velocity profile.

### Test Configuration
```
Domain:     nx=3, ny=64, nz=3
BCs:        Periodic in x,z
            No-slip wall at y=0 (stationary)
            Moving wall at y=ny-1 (velocity U_wall)
IC:         u=0, v=0, w=0, ρ=ρ₀
Duration:   5000 time steps (steady state)
```

### Analytical Solution
```
u(y) = U_wall × (y / H)

where:
  U_wall = velocity of top wall
  H = ny - 1
  y ∈ [0, H]

Shear stress (constant throughout domain):
  τ = μ × (du/dy) = μ × U_wall / H

Shear rate:
  γ̇ = du/dy = U_wall / H
```

### LBM Parameters
```cpp
float nu = 0.1f;
float rho0 = 1.0f;
float U_wall = 0.05f;  // Top wall velocity (Ma << 0.1)
float H = ny - 1;
```

### Expected Results
```
Velocity profile:  Linear u(y) = U_wall × (y/H)
Bottom wall:       u(0) = 0
Top wall:          u(H) = U_wall
Shear stress:      Constant τ = μ × U_wall / H
```

### Validation Metrics
```cpp
// 1. Linearity check (R² correlation)
float R_squared = computeLinearFit(y_coords, u_profile);
EXPECT_GT(R_squared, 0.999);  // Nearly perfect linearity

// 2. Boundary velocities
EXPECT_LT(|u(0)|, 1e-6);
EXPECT_NEAR(u(ny-1), U_wall, 1e-4);

// 3. Constant shear rate
for (int y = 1; y < ny-1; ++y) {
    float du_dy = (u(y+1) - u(y-1)) / 2.0f;
    float expected_shear = U_wall / H;
    EXPECT_NEAR(du_dy, expected_shear, 0.05);
}

// 4. Maximum deviation from linear profile
float max_deviation = max(|u(y) - U_wall×y/H|);
EXPECT_LT(max_deviation, 0.01 × U_wall);
```

### Implementation Note
Requires **moving wall boundary condition**. Two implementation options:

1. **Modified bounce-back** with velocity correction:
   ```
   f_opposite = f_incoming - 2 × w_i × (c_i · u_wall) / cs²
   ```

2. **Zou-He velocity boundary condition**:
   Set velocity at boundary and compute unknown distributions from known ones.

---

## Test 3: Lid-Driven Cavity

### Physical Description
Square cavity with moving top lid. Benchmark test for incompressible flow solvers. Develops primary vortex and corner vortices at high Re.

### Test Configuration
```
Domain:     nx=64, ny=64, nz=3 (2D-like)
BCs:        All walls no-slip, except top wall moving with u=U_lid
IC:         u=0, v=0, w=0, ρ=ρ₀
Duration:   50000 time steps (approach steady state)
Re:         100, 400, 1000 (three separate tests)
```

### Reference Solutions
**Ghia et al. (1982)** benchmark data for Re = 100, 400, 1000:

| Re   | u_center_y | v_center_x | Vortex center (x,y) |
|------|-----------|-----------|---------------------|
| 100  | 0.18750   | 0.17527   | (0.6172, 0.7344)   |
| 400  | 0.30203   | 0.30203   | (0.5547, 0.6055)   |
| 1000 | 0.38289   | 0.37095   | (0.5313, 0.5625)   |

### LBM Parameters
```cpp
float U_lid = 0.1f;  // Lid velocity (keep Ma < 0.1)
float L = nx - 1;    // Cavity size
float Re = 100;      // Reynolds number

// Compute required viscosity
float nu = U_lid * L / Re;

// Example for Re=100:
// nu = 0.1 × 63 / 100 = 0.063
```

### Expected Results
```
Primary vortex:     Center shifts with Re
Secondary vortices: Appear in bottom corners at Re > 400
Streamlines:        Closed loops, no flow through walls
Velocities:         u(walls) = 0 except top, v(all walls) = 0
```

### Validation Metrics
```cpp
// 1. Centerline velocity profiles
// Compare u(x, y=H/2) and v(x=L/2, y) with Ghia et al. data

// 2. Vortex center location
float (x_vortex, y_vortex) = findVorticityMaximum();
EXPECT_NEAR(x_vortex, x_ref, 0.05 × L);
EXPECT_NEAR(y_vortex, y_ref, 0.05 × L);

// 3. Wall boundary conditions
for all boundary nodes:
    EXPECT_LT(|u_normal|, 1e-6);
    EXPECT_LT(|u_tangent|, 1e-6);  // Except top wall

// 4. Incompressibility
float max_div = computeMaxDivergence();
EXPECT_LT(max_div, 1e-4);

// 5. Steady-state convergence
float velocity_change = L2_norm(u_n - u_{n-1000});
EXPECT_LT(velocity_change, 1e-5);
```

### Reference Data Location
Include Ghia et al. benchmark points in test:
```cpp
// Re=100 centerline data (excerpt)
struct BenchmarkPoint {
    float coord;
    float velocity;
};

const BenchmarkPoint u_centerline_Re100[] = {
    {0.0625, -0.03717},
    {0.5000,  0.18750},  // Peak velocity
    {0.9375, -0.32726},
    // ... full dataset
};
```

---

## Test 4: Taylor-Green Vortex Decay

### Physical Description
Decaying 2D vortex field. Tests temporal accuracy, dissipation rate, and spatial structure preservation.

### Test Configuration
```
Domain:     nx=64, ny=64, nz=3
BCs:        Periodic in x, y, z
IC:         Analytical vortex field
Duration:   Evolve until kinetic energy decays to 10% of initial
```

### Analytical Solution
**Initial condition (t=0):**
```
u(x,y,t=0) = -U₀ × sin(kx) × cos(ky)
v(x,y,t=0) =  U₀ × cos(kx) × sin(ky)
w(x,y,t=0) = 0

where:
  k = 2π / L (wavenumber)
  U₀ = initial velocity amplitude
```

**Time evolution:**
```
u(x,y,t) = -U₀ × sin(kx) × cos(ky) × exp(-2νk²t)
v(x,y,t) =  U₀ × cos(kx) × sin(ky) × exp(-2νk²t)

Kinetic energy:
  E(t) = (1/2) × Σ(u² + v²) = E₀ × exp(-2νk²t)

Decay rate:
  dE/dt = -2νk² × E(t)

Enstrophy:
  Ω(t) = Σ(ωz²) where ωz = ∂v/∂x - ∂u/∂y
```

### LBM Parameters
```cpp
float nu = 0.01f;          // Small viscosity for slow decay
float U0 = 0.05f;          // Initial velocity amplitude
float L = nx;              // Domain size
float k = 2.0f * M_PI / L; // Wavenumber
float decay_rate = 2.0f * nu * k * k;

// Example: L=64, nu=0.01, k=0.098
// decay_rate = 2 × 0.01 × 0.0096 = 1.92e-4
```

### Expected Results
```
Spatial structure: Preserved (vortex pattern remains)
Energy decay:      Exponential E(t) = E₀ exp(-2νk²t)
Velocity ratio:    u(x,y,t) / u(x,y,0) = exp(-2νk²t) at all points
Symmetry:          u(-x,-y) = u(x,y)
```

### Validation Metrics
```cpp
// 1. Energy decay rate (exponential fit)
vector<float> E_history;
for each timestep:
    float E = computeKineticEnergy();
    E_history.push_back(E);

// Fit E(t) = E₀ exp(-λt)
float lambda_fitted = fitExponentialDecay(E_history);
float lambda_analytical = 2.0f * nu * k * k;
EXPECT_NEAR(lambda_fitted, lambda_analytical, 0.05);

// 2. Spatial structure preservation
for each point (x,y):
    float ratio = u(x,y,t) / u(x,y,0);
    float expected_ratio = exp(-2νk²t);
    EXPECT_NEAR(ratio, expected_ratio, 0.02);

// 3. Zero mean velocity (periodic domain)
float u_mean = sum(u) / n_cells;
float v_mean = sum(v) / n_cells;
EXPECT_LT(|u_mean|, 1e-8);
EXPECT_LT(|v_mean|, 1e-8);

// 4. Enstrophy decay
float Omega_t = computeEnstrophy();
float Omega_0 = initial_enstrophy;
float Omega_analytical = Omega_0 * exp(-2νk²t);
EXPECT_NEAR(Omega_t, Omega_analytical, 0.1);
```

### Implementation Details
```cpp
// Initialize vortex field
for (int iy = 0; iy < ny; ++iy) {
    for (int ix = 0; ix < nx; ++ix) {
        float x = 2.0 * M_PI * ix / nx;
        float y = 2.0 * M_PI * iy / ny;

        ux[id] = -U0 * sin(x) * cos(y);
        uy[id] =  U0 * cos(x) * sin(y);
        uz[id] = 0.0f;
        rho[id] = rho0;
    }
}

// Upload to device and initialize solver
solver.initialize(d_rho, d_ux, d_uy, d_uz);
```

---

## Test 5: Bounce-Back Boundary Accuracy

### Physical Description
Tests no-slip wall implementation. Velocity at wall should be exactly zero.

### Test Configuration
```
Domain:     nx=10, ny=10, nz=10
BCs:        All walls = bounce-back (no-slip)
IC:         Uniform flow u=0.05, v=0, w=0
Forcing:    None (flow decays due to viscosity)
Duration:   1000 time steps
```

### Expected Results
```
Wall velocity:     u_wall < 1e-8 m/s (machine precision)
Interior velocity: Decays smoothly from initial to zero
No penetration:    v_wall = w_wall = 0
```

### Validation Metrics
```cpp
// Extract all boundary node velocities
vector<int> boundary_ids = identifyBoundaryNodes();

for (int id : boundary_ids) {
    EXPECT_LT(fabs(ux[id]), 1e-8) << "Non-zero x-velocity at wall";
    EXPECT_LT(fabs(uy[id]), 1e-8) << "Non-zero y-velocity at wall";
    EXPECT_LT(fabs(uz[id]), 1e-8) << "Non-zero z-velocity at wall";
}

// Verify flow direction reversal
// Distribution functions pointing into wall should equal
// those pointing out (after collision)
for boundary nodes:
    for q in outgoing_directions:
        int q_opp = opposite[q];
        EXPECT_NEAR(f[q], f[q_opp], 1e-6);
```

---

## Test 6: Body Force Response

### Physical Description
Uniform gravitational force should accelerate fluid uniformly until viscous forces balance.

### Test Configuration
```
Domain:     nx=8, ny=8, nz=8
BCs:        Periodic in all directions
IC:         u=0, v=0, w=0, ρ=ρ₀
Forcing:    Uniform fx = 1e-4 m/s²
Duration:   1000 time steps
```

### Analytical Solution (Early Time)
```
Before viscous damping dominates:
  u(t) ≈ (F/ρ) × t  (linear acceleration)

Terminal velocity (channel flow):
  u_∞ = (F × H²) / (8μ)  (for channel flow only)
```

### Expected Results
```
Velocity uniformity: All cells have same velocity
Acceleration phase:  u ∝ t (linear)
Direction:          Same as force direction
Magnitude:          |u| > 0 after sufficient steps
```

### Validation Metrics
```cpp
// 1. Uniform response
float u_mean = computeMean(ux);
float u_std = computeStdDev(ux);
EXPECT_LT(u_std / u_mean, 0.01);  // < 1% variation

// 2. Positive acceleration
EXPECT_GT(u_mean, 0.0);

// 3. Reasonable magnitude (not runaway)
EXPECT_LT(u_mean, 0.1);  // Ma < 0.3

// 4. Force direction alignment
EXPECT_GT(computeMean(ux), 0.0);  // Force in +x
EXPECT_NEAR(computeMean(uy), 0.0, 1e-6);
EXPECT_NEAR(computeMean(uz), 0.0, 1e-6);
```

---

## Test 7: Incompressibility Verification

### Physical Description
For low Mach number flows (Ma < 0.1), velocity field should satisfy ∇·u ≈ 0.

### Test Configuration
```
Domain:     Any test case (Poiseuille, Couette, etc.)
Constraint: Ma < 0.1
Metric:     Divergence |∇·u|
```

### Analytical Constraint
```
Incompressibility condition:
  ∇·u = ∂u/∂x + ∂v/∂y + ∂w/∂z ≈ 0

Finite difference:
  div(u) ≈ [u(i+1) - u(i-1)] / 2 + [v(j+1) - v(j-1)] / 2 + [w(k+1) - w(k-1)] / 2
```

### Validation Metrics
```cpp
// Compute divergence at each cell
for (int id = 0; id < n_cells; ++id) {
    float dudx = (ux[i+1,j,k] - ux[i-1,j,k]) / 2.0;
    float dvdy = (uy[i,j+1,k] - uy[i,j-1,k]) / 2.0;
    float dwdz = (uz[i,j,k+1] - uz[i,j,k-1]) / 2.0;

    float div_u = dudx + dvdy + dwdz;

    EXPECT_LT(fabs(div_u), 1e-4) << "Divergence too large at cell " << id;
}

// L∞ norm (maximum divergence)
float max_div = max(|div_u|);
EXPECT_LT(max_div, 1e-4);

// L2 norm (RMS divergence)
float rms_div = sqrt(mean(div_u²));
EXPECT_LT(rms_div, 1e-5);
```

---

## Test 8: Viscosity Verification

### Physical Description
Extract effective viscosity from simulation and compare with input value.

### Test Configuration
```
Method:     Use Poiseuille flow
Measure:    Velocity profile
Extract:    ν from u(y) = (1/2μ) × F × y(H-y)
```

### Validation Approach
```cpp
// 1. Run Poiseuille flow test to steady state
// 2. Measure u_max at channel center
// 3. Extract viscosity from analytical formula

float u_max_measured = velocity_profile[ny/2];
float F = body_force_x;
float H = ny - 1;

// From u_max = F·H²/(8μ):
float mu_extracted = (F * H * H) / (8.0f * u_max_measured);
float nu_extracted = mu_extracted / rho0;

// Compare with input
EXPECT_NEAR(nu_extracted, nu_input, 0.05 * nu_input);  // < 5% error
```

---

## Test 9: Stability at High Reynolds Number

### Physical Description
LBM should remain stable at Re ~ 1000-10000 with appropriate resolution.

### Test Configuration
```
Test cases:
  Re = 100:   Stable (baseline)
  Re = 1000:  Should remain stable
  Re = 5000:  May require increased resolution
  Re = 10000: Test stability limits
```

### Stability Criteria
```
Lattice relaxation time:  tau > 0.501 (stability limit)
Mach number:             Ma < 0.1 (incompressibility)
CFL condition:           |u| × dt/dx < 1
Grid resolution:         L/dx > 20 × sqrt(Re) (rule of thumb)
```

### Validation Metrics
```cpp
// Run simulation and monitor stability indicators
for (int step = 0; step < max_steps; ++step) {
    solver.step();

    // Check for NaN
    bool has_nan = checkForNaN(ux, uy, uz);
    ASSERT_FALSE(has_nan) << "NaN detected at step " << step;

    // Check for excessive velocities
    float u_max = computeMaxVelocity();
    ASSERT_LT(u_max, 0.3) << "Excessive velocity, instability";

    // Monitor energy growth (should not grow unbounded)
    float E = computeKineticEnergy();
    ASSERT_LT(E, 10.0 * E_initial) << "Energy runaway";
}

// Final check: simulation completed
EXPECT_TRUE(completed_successfully);
```

---

## Test 10: Mass Conservation

### Physical Description
Total mass in periodic domain must be conserved to machine precision.

### Test Configuration
```
Domain:     Any size, periodic BCs
Duration:   1000 steps
Check:      Total mass before vs. after
```

### Analytical Constraint
```
Mass conservation:
  M(t) = Σ ρ(x,t) = constant

Relative error:
  ε = |M(t) - M(0)| / M(0)
```

### Validation Metrics
```cpp
// Compute initial mass
float M_initial = 0.0f;
for (int i = 0; i < n_cells; ++i) {
    M_initial += rho[i];
}

// Run simulation
for (int step = 0; step < 1000; ++step) {
    solver.step();
}

// Compute final mass
float M_final = 0.0f;
for (int i = 0; i < n_cells; ++i) {
    M_final += rho[i];
}

// Check conservation
float relative_error = fabs(M_final - M_initial) / M_initial;
EXPECT_LT(relative_error, 1e-10);  // Machine precision level
```

---

## Test 11: Wall Shear Stress Accuracy

### Physical Description
Verify that wall shear stress is correctly computed and matches analytical value.

### Test Configuration
```
Test:       Poiseuille flow at steady state
Measure:    Velocity gradient at wall
Compute:    τ_wall = μ × (du/dy)|_wall
Compare:    With analytical τ_wall = (H/2) × |dp/dx|
```

### Validation Metrics
```cpp
// Compute wall shear stress from velocity gradient
float du_dy_wall = (uy[1] - uy[0]);  // One-sided difference at wall
float tau_numerical = mu * du_dy_wall;

// Analytical value for Poiseuille flow
float H = ny - 1;
float tau_analytical = 0.5f * H * fabs(dp_dx);

// Compare
float error = fabs(tau_numerical - tau_analytical) / tau_analytical;
EXPECT_LT(error, 0.05);  // < 5%
```

---

## Implementation Checklist for Each Test

For each test, provide:

- [ ] **Test fixture setup** (domain, BCs, parameters)
- [ ] **Analytical solution implementation** (reference values)
- [ ] **LBM solver configuration** (nu, omega, forcing)
- [ ] **Time evolution loop** (collision, streaming, BC)
- [ ] **Result extraction** (copy to host, compute profiles)
- [ ] **Quantitative comparison** (L2 error, max error, etc.)
- [ ] **ASSERT/EXPECT statements** (clear pass/fail criteria)
- [ ] **Console output** (print key metrics for debugging)
- [ ] **Optional: file output** (profiles, fields for visualization)

---

## Common Utilities (To be Implemented)

Create helper functions in `tests/utils/fluid_test_utils.h`:

```cpp
namespace FluidTestUtils {

// Compute L2 relative error between two vectors
float computeL2Error(const vector<float>& numerical,
                     const vector<float>& analytical);

// Compute velocity divergence at each cell
void computeDivergence(const float* ux, const float* uy, const float* uz,
                       float* div, int nx, int ny, int nz);

// Find vortex center (maximum vorticity)
pair<float,float> findVortexCenter(const float* ux, const float* uy,
                                    int nx, int ny);

// Compute kinetic energy
float computeKineticEnergy(const float* ux, const float* uy, const float* uz,
                           int n_cells);

// Check for NaN in arrays
bool hasNaN(const float* data, int size);

// Save velocity profile to file
void saveProfile(const string& filename,
                 const vector<float>& coords,
                 const vector<float>& values);

// Load benchmark data from file
vector<BenchmarkPoint> loadBenchmarkData(const string& filename);

} // namespace FluidTestUtils
```

---

## Suggested Test Execution Order

1. **Basic functionality** (constructor, initialization, force response)
2. **Boundary conditions** (bounce-back accuracy)
3. **Simple flows** (Couette flow - linear profile)
4. **Intermediate flows** (Poiseuille flow - parabolic profile)
5. **Complex flows** (Lid-driven cavity - vortex dynamics)
6. **Temporal accuracy** (Taylor-Green vortex decay)
7. **Physical constraints** (incompressibility, mass conservation)
8. **Stability limits** (high Re, viscosity bounds)

---

## Success Criteria Summary

| Test                  | Primary Metric              | Threshold      |
|-----------------------|-----------------------------|----------------|
| Poiseuille Flow       | L2 relative error           | < 5%           |
| Couette Flow          | Linearity (R²)              | > 0.999        |
| Lid-Driven Cavity     | Vortex center error         | < 5% of domain |
| Taylor-Green Vortex   | Decay rate error            | < 5%           |
| Bounce-Back           | Wall velocity               | < 1e-8         |
| Body Force            | Velocity uniformity (σ/μ)   | < 1%           |
| Incompressibility     | Max divergence              | < 1e-4         |
| Viscosity Extraction  | ν error                     | < 5%           |
| High Re Stability     | No NaN, |u| < 0.3           | Pass/Fail      |
| Mass Conservation     | Relative mass change        | < 1e-10        |

---

## References

1. **Ghia, U., Ghia, K. N., & Shin, C. T. (1982).** "High-Re solutions for incompressible flow using the Navier-Stokes equations and a multigrid method." *Journal of Computational Physics*, 48(3), 387-411.

2. **Guo, Z., Zheng, C., & Shi, B. (2002).** "Discrete lattice effects on the forcing term in the lattice Boltzmann method." *Physical Review E*, 65(4), 046308.

3. **Krüger, T., et al. (2017).** *The Lattice Boltzmann Method: Principles and Practice.* Springer.

4. **Succi, S. (2001).** *The Lattice Boltzmann Equation for Fluid Dynamics and Beyond.* Oxford University Press.

5. **White, F. M. (2011).** *Fluid Mechanics* (7th ed.). McGraw-Hill.

---

## Notes on Numerical Accuracy

**Expected LBM accuracy for benchmark tests:**
- **Poiseuille flow:** 1-5% error typical for D3Q19 with bounce-back BC
- **Couette flow:** <1% error (simpler geometry)
- **Lid-driven cavity:** 3-10% error in vortex position (reference-dependent)
- **Taylor-Green vortex:** 2-5% decay rate error (temporal discretization)

**Factors affecting accuracy:**
1. **Lattice resolution:** Higher ny → better accuracy (but slower)
2. **Relaxation time:** tau close to 0.5 → less stable, worse accuracy
3. **Boundary implementation:** Second-order BC better than simple bounce-back
4. **Time step:** Smaller dt → better temporal accuracy (if using physical units)

**Grid convergence study (optional advanced test):**
Run same test at multiple resolutions (ny = 16, 32, 64, 128) and verify:
```
error ∝ (Δx)^p  where p ≈ 2 (second-order convergence)
```

---

## File Structure for Tests

Suggested organization:
```
tests/
├── unit/
│   └── fluid/
│       ├── test_fluid_lbm_basic.cu          (constructor, init, getters)
│       ├── test_bounce_back_bc.cu           (Test 5)
│       ├── test_body_force.cu               (Test 6)
│       └── test_incompressibility.cu        (Test 7)
│
├── integration/
│   └── fluid/
│       ├── test_poiseuille_flow.cu          (Test 1) [EXISTS]
│       ├── test_couette_flow.cu             (Test 2)
│       ├── test_lid_driven_cavity.cu        (Test 3)
│       └── test_taylor_green_vortex.cu      (Test 4)
│
├── validation/
│   └── fluid/
│       ├── test_viscosity_extraction.cu     (Test 8)
│       ├── test_high_re_stability.cu        (Test 9)
│       └── test_mass_conservation.cu        (Test 10)
│
└── utils/
    └── fluid_test_utils.h                   (Common utilities)
```

---

## Conclusion

This test suite provides **comprehensive validation** of the FluidLBM module, covering:
- ✅ Fundamental physics (Poiseuille, Couette, vortex decay)
- ✅ Boundary conditions (no-slip walls, periodic)
- ✅ Numerical accuracy (comparison with analytical solutions)
- ✅ Physical constraints (incompressibility, mass conservation)
- ✅ Stability (high Re, viscosity limits)

Each test is **standalone**, includes **analytical solutions**, and has **clear pass/fail criteria**. Implement tests incrementally, starting with simple cases (Couette) and building to complex flows (cavity).

**Next steps:**
1. Implement Test 2 (Couette flow) as template
2. Create `fluid_test_utils.h` helper library
3. Implement remaining tests following specifications
4. Add grid convergence studies (optional)
5. Create test documentation with expected outputs

---

**End of Test Specifications**
