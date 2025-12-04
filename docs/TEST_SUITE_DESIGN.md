# Comprehensive Test Suite Design for LBM-CUDA Physics Modules

## Overview

This document specifies a scientifically rigorous test suite for the LBM-CUDA computational framework. The tests are designed to validate numerical accuracy, physical correctness, conservation laws, and computational stability across all physics modules.

**Organization**: Tests are organized into five categories:
1. `/tests/validation/analytical/` - Analytical solution validation (highest priority)
2. `/tests/validation/conservation/` - Conservation law verification
3. `/tests/validation/convergence/` - Grid and temporal convergence studies
4. `/tests/validation/scaling/` - Physical scaling and dimensionless number consistency
5. `/tests/validation/stability/` - Stability boundary identification

---

## 1. Analytical Solution Tests (Highest Priority)

These tests compare LBM simulations against exact analytical solutions to quantify numerical accuracy. Target: L2 error < 5-6% (typical for second-order LBM schemes).

### 1.1 Poiseuille Flow (Pressure-Driven Channel Flow)

**File**: `/tests/validation/analytical/test_poiseuille_flow_convergence.cu`

**Physics Description**:
- 2D steady pressure-driven flow between parallel plates
- Parabolic velocity profile: `u(y) = (dp/dx) * y * (H - y) / (2*mu)`
- Maximum velocity at centerline: `u_max = (dp/dx) * H^2 / (8*mu)`
- Tests fundamental LBM hydrodynamics without body forces

**Analytical Solution**:
```
u(y) = -(dp/dx) / (2*mu) * y * (H - y)
u_max = -(dp/dx) * H^2 / (8*mu)
Q = (2/3) * H * u_max  [volumetric flow rate per unit depth]
```

**Test Configuration**:
- Domain: `3 × 64 × 3` (thin in x and z with periodic BCs)
- Channel height: H = 63 (ny - 1)
- Pressure gradient: `dp/dx = -1e-4` (body force equivalent)
- Kinematic viscosity: `nu = 0.1` (lattice units)
- Relaxation parameter: `omega = 1 / (3*nu + 0.5)`
- Run time: 20,000 timesteps (ensure steady state)

**Validation Metrics**:
1. **L2 relative error**: `||u_LBM - u_analytical||_2 / ||u_analytical||_2 < 0.05`
2. **Maximum velocity error**: `|u_max,LBM - u_max,analytical| / u_max,analytical < 0.03`
3. **Profile shape**: Verify parabolic shape (check second derivative)
4. **Symmetry**: `|u(y) - u(H-y)| / u_max < 0.02` for all y
5. **Flow rate**: `|Q_LBM - Q_analytical| / Q_analytical < 0.04`
6. **Wall boundary condition**: `u(0) < 1e-4` and `u(H) < 1e-4`

**Implementation Notes**:
- Use Guo forcing scheme for body force: `F = (1 - omega/2) * 3*w_i*c_i*f_x`
- Apply mid-grid bounce-back for no-slip walls (second-order accurate)
- Store velocity profile every 1000 steps to verify convergence to steady state
- Export profile for visualization: `poiseuille_profile_H{ny}.txt`

**Expected Result**: L2 error ~2-4%, demonstrates second-order spatial accuracy of LBM

---

### 1.2 Couette Flow (Shear-Driven Flow)

**File**: `/tests/validation/analytical/test_couette_flow.cu`

**Physics Description**:
- Flow between two parallel plates, top plate moving with velocity U
- Linear velocity profile: `u(y) = U * y / H`
- Tests shear stress and momentum diffusion
- Validates viscosity implementation and moving wall BC

**Analytical Solution**:
```
u(y) = U * (y / H)
du/dy = U / H = constant (uniform shear rate)
tau = mu * (U / H)  [wall shear stress]
```

**Test Configuration**:
- Domain: `3 × 32 × 3`
- Top wall velocity: `U = 0.02` (lattice units, Ma < 0.1 for incompressibility)
- Bottom wall: stationary (u = 0)
- Kinematic viscosity: `nu = 0.1`
- Run time: 15,000 timesteps

**Validation Metrics**:
1. **L2 relative error**: `< 0.04`
2. **Linearity**: `max|u(y) - U*y/H| / U < 0.03`
3. **Shear rate uniformity**: `std(du/dy) / mean(du/dy) < 0.05`
4. **Top wall velocity**: `|u(H) - U| / U < 0.01`
5. **Bottom wall velocity**: `u(0) < 1e-4`
6. **Momentum flux**: Verify `tau = rho * nu * du/dy` matches analytical

**Boundary Conditions**:
- Top wall: Zou-He velocity BC or regularized BC with `u = U`
- Bottom wall: Standard bounce-back
- x, z: Periodic

**Expected Result**: L2 error ~2-3%, validates moving wall BC and viscous stress

---

### 1.3 Heat Conduction (1D Thermal Diffusion)

**File**: `/tests/validation/analytical/test_heat_conduction_1d.cu`

**Physics Description**:
- 1D transient heat diffusion with constant boundary temperatures
- Tests thermal LBM without phase change
- Validates thermal diffusivity and boundary conditions

**Analytical Solution** (transient):
```
T(x,t) = T_left + (T_right - T_left) * x/L +
         sum_{n=1}^∞ [A_n * sin(n*pi*x/L) * exp(-n^2*pi^2*alpha*t/L^2)]

Steady state: T(x) = T_left + (T_right - T_left) * x / L
```

**Test Configuration**:
- Domain: `200 × 3 × 3` (quasi-1D)
- Boundary temperatures: `T_left = 2000 K`, `T_right = 300 K`
- Initial temperature: `T_init = 300 K`
- Material: Ti6Al4V
- Thermal diffusivity: `alpha = 5.8e-6 m^2/s`
- Physical length: `L = 200 * dx` with `dx = 2e-6 m`
- Time step: `dt = 5e-11 s`
- Run time: 50,000 steps (approach steady state)

**Validation Metrics**:
1. **Steady-state L2 error**: `< 0.05` (after sufficient time)
2. **Temperature gradient**: `|dT/dx - (T_right - T_left)/L| / |(T_right - T_left)/L| < 0.03`
3. **Heat flux conservation**: `q = -k * dT/dx = constant` throughout domain
4. **Transient evolution**: Compare profile at `t = 0.1*L^2/alpha` with analytical series
5. **Boundary accuracy**: `|T(0) - T_left| < 0.1 K` and `|T(L) - T_right| < 0.1 K`

**Implementation Notes**:
- Use D3Q19 thermal lattice with BGK collision
- Apply constant temperature BC (Dirichlet): directly set f_eq based on T_boundary
- Monitor convergence: `max|T^n - T^{n-1000}| / max(T)` should decrease exponentially
- Test both transient and steady-state regimes

**Expected Result**: L2 error ~3-5%, validates thermal solver without phase change

---

### 1.4 Taylor-Green Vortex (2D Vorticity Decay)

**File**: `/tests/validation/analytical/test_taylor_green_vortex.cu`

**Physics Description**:
- Classic test for incompressible Navier-Stokes solvers
- Analytical solution for decaying 2D vortex array
- Tests vorticity transport, pressure-velocity coupling, and viscous dissipation

**Analytical Solution**:
```
u(x,y,t) = -U0 * cos(k*x) * sin(k*y) * exp(-2*k^2*nu*t)
v(x,y,t) =  U0 * sin(k*x) * cos(k*y) * exp(-2*k^2*nu*t)
p(x,y,t) = p0 - (rho*U0^2/4) * [cos(2*k*x) + cos(2*k*y)] * exp(-4*k^2*nu*t)

Kinetic energy: E(t) = E0 * exp(-4*k^2*nu*t)
Enstrophy: Omega(t) = Omega0 * exp(-4*k^2*nu*t)
```

**Test Configuration**:
- Domain: `64 × 64 × 3` (quasi-2D)
- Wave number: `k = 2*pi / L` with `L = 64`
- Initial velocity: `U0 = 0.05` (lattice units)
- Reynolds number: `Re = U0 * L / nu = 32` (moderate Re for clear decay)
- Kinematic viscosity: `nu = 0.1`
- Boundary conditions: Periodic in all directions
- Run time: 10,000 timesteps

**Validation Metrics**:
1. **Velocity field L2 error**: `< 0.06` at multiple time snapshots
2. **Kinetic energy decay rate**: `|d(ln E)/dt + 4*k^2*nu| / (4*k^2*nu) < 0.05`
3. **Enstrophy conservation**: Track `Omega = integral(|curl(u)|^2 dV)`
4. **Vorticity accuracy**: `|omega_z,LBM - omega_z,analytical| / max(omega_z) < 0.06`
5. **Incompressibility**: `max|div(u)| < 1e-3`
6. **Symmetry**: Verify 4-fold rotational symmetry is preserved

**Analysis**:
- Compute kinetic energy: `E = 0.5 * sum(u^2 + v^2) / N`
- Compute enstrophy: `Omega = sum(omega_z^2) / N` where `omega_z = dv/dx - du/dy`
- Plot `log(E)` vs `t`: slope should be `-4*k^2*nu`
- Export velocity field at `t = 0, L^2/(4*nu), L^2/(2*nu)` for visualization

**Expected Result**: Energy decay rate error ~3-5%, validates vorticity transport and viscous dissipation

---

## 2. Conservation Law Tests

These tests verify that fundamental physical quantities are conserved (or change at expected rates) throughout simulations.

### 2.1 Mass Conservation (Fluid + VOF)

**File**: `/tests/validation/conservation/test_mass_conservation_detailed.cu`

**Physics Description**:
- Track total mass in domain with fluid dynamics and VOF
- Tests: closed system (no sources), open boundaries, and with evaporation
- Distinguishes between numerical drift and physical mass loss

**Test Cases**:

**Case A: Closed System (No Sources)**
- Domain: `32 × 32 × 32` periodic
- Two-phase flow with VOF (liquid volume fraction field)
- No phase change, no boundaries
- Expected: `dm/dt = 0` (absolute conservation)

**Case B: With Evaporation**
- Domain: `40 × 40 × 20` with free surface (VOF)
- Enable evaporation at gas-liquid interface
- Measure evaporated mass separately
- Expected: `m_liquid(t) + m_evaporated(t) = m_liquid(0)` within 1%

**Validation Metrics**:
1. **Global mass conservation** (Case A): `|m(t) - m(0)| / m(0) < 1e-4` over 10,000 steps
2. **Local mass consistency**: `rho = rho_liquid * f + rho_gas * (1-f)` where f is VOF
3. **Evaporation balance** (Case B): `|dm_liquid/dt + dm_evaporated/dt| / |dm_liquid/dt| < 0.01`
4. **Interface mass flux**: Verify `dot{m} = -rho_liquid * A_interface * v_evap` matches Hertz-Knudsen
5. **No spurious sources**: Mass change only at interfaces, not in bulk

**Implementation**:
```cpp
// Global mass calculation
float compute_total_mass(float* d_rho, float* d_vof, int N) {
    float mass = 0.0f;
    for (int i = 0; i < N; ++i) {
        float rho_cell = rho[i];  // Mixture density
        float volume_cell = dx * dy * dz;
        mass += rho_cell * volume_cell;
    }
    return mass;
}

// Mass balance check
float dm_numerical = m_current - m_previous - m_evaporated;
float conservation_error = dm_numerical / m_previous;
EXPECT_LT(fabs(conservation_error), 1e-4);
```

**Expected Result**:
- Case A: <0.01% drift over 10,000 steps (numerical precision)
- Case B: Total mass balance within 1% (accounts for evaporation)

---

### 2.2 Momentum Conservation

**File**: `/tests/validation/conservation/test_momentum_conservation.cu`

**Physics Description**:
- Total momentum should change only due to external forces and boundary fluxes
- Tests: periodic domain (conserved), wall boundaries (flux), body forces

**Test Cases**:

**Case A: Periodic Domain, No Forces**
- Initial condition: Random velocity field with zero mean momentum
- Expected: `P(t) = P(0) = 0` exactly (within machine precision)

**Case B: Uniform Body Force**
- Periodic domain with constant body force `F`
- Expected: `dP/dt = F * Volume` (Newton's second law)

**Case C: Wall Boundaries**
- Channel flow with pressure gradient (walls exert force on fluid)
- Expected: `dP/dt = F_body + F_walls` where `F_walls` is momentum flux at boundaries

**Validation Metrics**:
1. **Periodic, no force**: `||P(t) - P(0)|| / (rho * U * V) < 1e-6`
2. **With body force**: `|dP/dt - F*V| / (F*V) < 0.02`
3. **Component-wise**: Verify Px, Py, Pz separately
4. **Symmetry**: If force in x-direction only, Py and Pz should remain zero
5. **Wall momentum transfer**: Measure `F_wall = sum_boundaries (tau_wall * dA)`

**Implementation**:
```cpp
// Compute total momentum
float3 compute_momentum(float* d_rho, float* d_ux, float* d_uy, float* d_uz, int N) {
    float3 P = {0, 0, 0};
    for (int i = 0; i < N; ++i) {
        float m = rho[i] * cell_volume;
        P.x += m * ux[i];
        P.y += m * uy[i];
        P.z += m * uz[i];
    }
    return P;
}

// Momentum balance
float3 dP_dt = (P_current - P_previous) / dt;
float3 F_total = {F_body.x * Volume, F_body.y * Volume, F_body.z * Volume};
float error = length(dP_dt - F_total) / length(F_total);
EXPECT_LT(error, 0.02);
```

**Expected Result**:
- Periodic: momentum conservation to machine precision
- With forces: momentum change rate matches applied force within 2%

---

### 2.3 Energy Conservation (Thermal System)

**File**: `/tests/validation/conservation/test_energy_conservation_detailed.cu`

**Physics Description**:
- First law of thermodynamics: `dE/dt = P_in - P_out`
- Tests: isolated system, laser heating, cooling mechanisms, phase change

**Energy Components**:
```
E_total = E_sensible + E_latent
E_sensible = integral[rho * cp * (T - T_ref) dV]
E_latent = integral[rho * L_fusion * f_liquid dV]

Power balance:
dE/dt = P_laser - P_evaporation - P_radiation - P_substrate - P_convection
```

**Test Cases**:

**Case A: Adiabatic System (No Sources)**
- Isolated domain (adiabatic walls)
- Initial: Hot region in center, cold elsewhere
- Expected: `E(t) = E(0)` exactly (no external energy transfer)

**Case B: Laser Heating Only**
- Laser power input: `P_laser = 100 W`
- No cooling mechanisms
- Expected: `dE/dt = P_laser * absorptivity` (all absorbed energy stored)

**Case C: Full Energy Balance**
- Laser + evaporation + radiation + substrate cooling
- Expected: `|dE/dt - (P_laser - P_evap - P_rad - P_sub)| / P_laser < 0.05`

**Validation Metrics**:
1. **Adiabatic conservation**: `|E(t) - E(0)| / E(0) < 1e-4` over 5,000 steps
2. **Laser-only balance**: `|dE/dt - P_laser*A| / (P_laser*A) < 0.03`
3. **Full balance**: Energy balance error < 5% (per existing test_energy_conservation_full.cu)
4. **Latent heat tracking**: Verify `dE_latent = rho * L_fusion * df_liquid`
5. **Temperature bounds**: No unphysical temperatures (T > 0 K, T < T_max)

**Implementation**:
```cpp
// Compute total energy
double compute_total_energy(float* d_T, float* d_fl, MaterialProperties mat, int N) {
    double E_sensible = 0.0;
    double E_latent = 0.0;

    for (int i = 0; i < N; ++i) {
        float T = T[i];
        float fl = fl[i];
        float rho = mat.getDensity(T);
        float cp = mat.getSpecificHeat(T);

        E_sensible += rho * cp * (T - T_ref) * cell_volume;
        E_latent += rho * mat.L_fusion * fl * cell_volume;
    }

    return E_sensible + E_latent;
}

// Energy balance
double dE_dt = (E_current - E_previous) / dt;
double P_net = P_laser - P_evap - P_rad - P_sub;
double balance_error = fabs(dE_dt - P_net) / fabs(P_laser);
EXPECT_LT(balance_error, 0.05);
```

**Expected Result**:
- Adiabatic: <0.01% drift
- Full system: 5% balance error (matches existing test standards)

---

### 2.4 Entropy Production (Second Law Check)

**File**: `/tests/validation/conservation/test_entropy_production.cu`

**Physics Description**:
- Second law: Total entropy must increase or remain constant (never decrease)
- Entropy production due to: viscous dissipation, heat conduction, phase change irreversibility

**Entropy Rate Equation**:
```
dS/dt = dS/dt|system + dS/dt|surroundings >= 0

Specific entropy production:
sigma = (k/T^2) * |grad(T)|^2 + (mu/T) * Phi

where Phi = viscous dissipation function
```

**Test Configuration**:
- Domain: `40 × 40 × 20` with thermal + fluid coupling
- Process: Heat conduction from hot to cold region
- Track: System entropy, surroundings entropy (energy flux × 1/T_boundary)

**Validation Metrics**:
1. **Non-negativity**: `dS_universe/dt >= 0` at all times
2. **Irreversible processes**: Heat conduction should produce entropy
3. **Viscous dissipation**: Flow with shear should increase entropy
4. **Equilibrium**: `dS/dt -> 0` as system approaches thermal equilibrium

**Implementation**:
```cpp
// Compute system entropy (simplified)
float compute_entropy(float* d_T, MaterialProperties mat, int N) {
    float S = 0.0f;
    for (int i = 0; i < N; ++i) {
        float T = T[i];
        float rho = mat.getDensity(T);
        float cp = mat.getSpecificHeat(T);
        // s = cp * log(T/T_ref) for ideal substance
        float s_specific = cp * log(T / T_ref);
        S += rho * s_specific * cell_volume;
    }
    return S;
}

// Entropy production rate
float dS_system = (S_current - S_previous) / dt;
float dS_surroundings = -Q_out / T_ambient;  // Heat lost to surroundings
float dS_universe = dS_system + dS_surroundings;

EXPECT_GE(dS_universe, -1e-6);  // Allow small numerical error
```

**Expected Result**: Total entropy production > 0 for all irreversible processes

---

## 3. Convergence Tests

These tests verify that numerical solutions converge to the correct answer as grid spacing and time step are refined.

### 3.1 Grid Convergence (h-refinement)

**File**: `/tests/validation/convergence/test_grid_convergence_poiseuille.cu`

**Physics Description**:
- Run same physical problem at multiple grid resolutions
- Measure error reduction as grid is refined
- Verify theoretical convergence rate: `error ~ h^p` where p = spatial order

**Test Configuration**:
- Problem: Poiseuille flow (known analytical solution)
- Grid resolutions: `ny = 16, 32, 64, 128` (keep Lx, Ly, Lz constant physically)
- Same physical parameters: viscosity, pressure gradient, etc.
- Adjust relaxation parameter: `omega(h)` to maintain same physical viscosity

**Convergence Rate Analysis**:
```
error(h) = ||u_LBM(h) - u_analytical||_2

Theoretical: error(h) ~ C * h^p
where p = 2 for second-order LBM

Measured convergence order:
p_measured = log(error(h1)/error(h2)) / log(h1/h2)
```

**Validation Metrics**:
1. **Convergence order**: `1.8 < p_measured < 2.2` (expect p ≈ 2)
2. **Monotonic reduction**: `error(h/2) < error(h)` for all refinements
3. **Richardson extrapolation**: Estimate exact solution from h1, h2, h3
4. **Grid independence**: `error(h_finest) < 0.02` (nearly grid-independent)

**Implementation**:
```cpp
struct GridStudy {
    int ny;
    float h;  // dy in lattice units
    float error_L2;
    float error_max;
};

// Run multiple grid sizes
std::vector<GridStudy> studies = {
    {16, 1.0}, {32, 0.5}, {64, 0.25}, {128, 0.125}
};

for (auto& study : studies) {
    // Run simulation at this resolution
    float L2_error = run_poiseuille(study.ny);
    study.error_L2 = L2_error;
}

// Compute convergence rate
float p = log(studies[0].error_L2 / studies[1].error_L2)
        / log(studies[0].h / studies[1].h);

EXPECT_GT(p, 1.8);
EXPECT_LT(p, 2.2);
```

**Expected Result**: Convergence order p ≈ 2.0 ± 0.2 (second-order spatial accuracy)

---

### 3.2 Temporal Convergence (dt-refinement)

**File**: `/tests/validation/convergence/test_temporal_convergence_heat_diffusion.cu`

**Physics Description**:
- Run same problem with multiple time step sizes
- Measure error reduction with finer time steps
- Verify temporal order of accuracy

**Test Configuration**:
- Problem: 1D transient heat diffusion with analytical solution
- Time steps: `dt = dt0, dt0/2, dt0/4, dt0/8`
- Keep grid spacing constant (h fixed)
- Compare at same physical time: `t_phys = N * dt`

**Validation Metrics**:
1. **Temporal order**: `p_t ≈ 1` for explicit schemes (BGK is first-order in time)
2. **Stable refinement**: Smaller dt should reduce error
3. **CFL consistency**: Verify stability for all dt tested
4. **Asymptotic regime**: Check error follows `error ~ dt^p` for small dt

**Expected Result**: Temporal convergence order p ≈ 1.0 (first-order in time for LBM)

---

### 3.3 Combined Space-Time Convergence

**File**: `/tests/validation/convergence/test_combined_convergence_taylor_green.cu`

**Physics Description**:
- Simultaneously refine space and time while maintaining physical scales
- Use Taylor-Green vortex (known analytical solution at all times)
- Demonstrates convergence to exact solution

**Test Configuration**:
- Resolutions: `{N=32, dt0}`, `{N=64, dt0/2}`, `{N=128, dt0/4}`
- Maintain `dt/dx^2 = constant` (same diffusion number)
- Compare at fixed physical times: `t = 0.1, 0.5, 1.0` time units

**Validation Metrics**:
1. **Overall convergence**: `error(h, dt) ~ h^2 + dt` (both contribute)
2. **Dominant error**: Identify whether spatial or temporal error dominates
3. **Optimal dt selection**: Find `dt_opt(h)` that balances errors

**Expected Result**: Combined error reduction with both h and dt refinement

---

## 4. Physical Scaling Tests

These tests verify that dimensionless numbers and physical scaling relationships are correctly implemented.

### 4.1 Reynolds Number Independence

**File**: `/tests/validation/scaling/test_reynolds_number_scaling.cu`

**Physics Description**:
- Same geometry and boundary conditions at different Reynolds numbers
- Test both low Re (Stokes flow) and moderate Re (inertial effects)
- Verify drag coefficients, flow patterns match theory

**Test Cases**:

**Case A: Drag on Sphere**
- Place sphere in uniform flow
- Measure drag force at Re = 1, 10, 100
- Compare to empirical correlations: `Cd = 24/Re + 6/(1+sqrt(Re)) + 0.4`

**Case B: Vortex Shedding (Cylinder)**
- Flow past cylinder at Re = 50, 100, 150
- Measure Strouhal number: `St = f * D / U`
- Expected: `St ≈ 0.15` for Re = 100 (Von Karman vortex street)

**Validation Metrics**:
1. **Drag coefficient**: `|Cd_LBM - Cd_theory| / Cd_theory < 0.08`
2. **Strouhal number**: `|St_LBM - St_theory| / St_theory < 0.10`
3. **Flow separation**: Verify separation point for different Re
4. **Scaling consistency**: Results should depend only on Re, not on individual u, nu

**Expected Result**: Physical behavior matches Re-scaling laws

---

### 4.2 Peclet Number Consistency (Thermal)

**File**: `/tests/validation/scaling/test_peclet_number_scaling.cu`

**Physics Description**:
- Thermal transport with varying Peclet number: `Pe = u * L / alpha`
- Pe << 1: Diffusion-dominated (symmetric profile)
- Pe >> 1: Advection-dominated (asymmetric, boundary layers)

**Test Configuration**:
- Problem: Heated sphere in flow
- Vary: Velocity and thermal diffusivity to achieve Pe = 0.1, 1, 10, 100
- Measure: Temperature distribution, Nusselt number

**Validation Metrics**:
1. **Nusselt number**: `Nu = h * L / k` should follow correlations
2. **Profile symmetry**: Symmetric for Pe < 1, asymmetric for Pe > 1
3. **Boundary layer thickness**: `delta_T ~ L / Pe^{1/2}` for high Pe
4. **Consistency**: Same Pe should give same Nu regardless of u, alpha separately

**Expected Result**: Heat transfer scales correctly with Peclet number

---

### 4.3 Mach Number Limit (Compressibility)

**File**: `/tests/validation/scaling/test_mach_number_limit.cu`

**Physics Description**:
- LBM assumes incompressibility: `Ma = u / c_s << 1` where `c_s = 1/sqrt(3)`
- Test: Run simulations at Ma = 0.01, 0.05, 0.1, 0.2, 0.3
- Measure: Density fluctuations, error in incompressibility assumption

**Validation Metrics**:
1. **Density fluctuations**: `|rho - rho0| / rho0 ~ Ma^2` (theoretical scaling)
2. **Divergence error**: `|div(u)| ~ Ma^2`
3. **Pressure-density relation**: Check `p - p0 = c_s^2 * (rho - rho0)`
4. **Accuracy threshold**: Verify Ma < 0.3 for accurate incompressible flow

**Expected Result**: Incompressibility assumption valid for Ma < 0.1

---

### 4.4 Force Balance Verification

**File**: `/tests/validation/scaling/test_force_balance_melt_pool.cu`

**Physics Description**:
- In melt pool, multiple forces act: surface tension, Marangoni, recoil pressure, gravity
- Test: Verify force balance at steady state: `sum(F) = 0`

**Force Components**:
```
F_surface_tension = sigma * kappa * n  [curvature pressure]
F_marangoni = -dsigma/dT * grad_s(T)  [tangential stress]
F_recoil = P_recoil * n               [evaporative recoil]
F_buoyancy = rho * g * (1 - beta*(T-T_ref))
F_viscous = mu * nabla^2(u)
```

**Validation Metrics**:
1. **Static droplet**: Only surface tension (F_st = sigma*kappa) balances pressure
2. **Heated droplet**: Marangoni should dominate for high Ma_T
3. **Evaporating pool**: Recoil pressure should depress surface
4. **Force magnitude ratios**: Verify dimensionless groups (Ma, Bo, Ca)

**Expected Result**: Dominant forces match physical expectations for each regime

---

## 5. Stability Tests

These tests identify stability boundaries and maximum stable parameters.

### 5.1 Maximum Stable Velocity (Fluid)

**File**: `/tests/validation/stability/test_max_stable_velocity.cu`

**Physics Description**:
- LBM has CFL-like stability limit related to Mach number
- Find maximum velocity before numerical instability

**Test Procedure**:
1. Initialize uniform flow with velocity `u`
2. Run 5000 timesteps
3. Check for: NaN, divergence, density fluctuations > 10%
4. Binary search to find `u_max`

**Expected Stability Limit**:
- `u_max ≈ 0.15 - 0.2` (lattice units) for most LBM schemes
- Depends on: viscosity (larger nu → more stable)
- Should match: `Ma_max ≈ 0.3` where `c_s = 1/sqrt(3)`

**Validation Metrics**:
1. **Stability criterion**: `u_max / c_s < 0.6` (practical limit)
2. **Viscosity dependence**: Higher nu → higher u_max
3. **Warning generation**: Code should warn if u > 0.15

---

### 5.2 Maximum Stable Temperature Gradient

**File**: `/tests/validation/stability/test_max_temperature_gradient.cu`

**Physics Description**:
- Steep temperature gradients can cause instability in thermal LBM
- Particularly problematic with: phase change, temperature-dependent properties

**Test Configuration**:
- 1D domain with extreme temperature difference: `T_left = 300 K`, `T_right varies`
- Gradually increase `deltaT = T_right - T_left`
- Test: 300 K → 1000 K, 1500 K, 2000 K, 3000 K, 5000 K

**Validation Metrics**:
1. **Stability**: No NaN or negative temperatures for all tested gradients
2. **Flux limiter effectiveness**: Check that flux limiter activates for steep gradients
3. **Temperature bounds**: T stays within [T_min, T_max] defined by material

**Expected Result**: Stable for all physically reasonable gradients (tested up to 5000 K difference)

---

### 5.3 CFL Condition (Coupled Thermal-Fluid)

**File**: `/tests/validation/stability/test_cfl_condition_coupling.cu`

**Physics Description**:
- Coupled system has stability limits from both fluid and thermal solvers
- CFL condition: `dt < min(dx/u_max, dx^2/alpha_max)`

**Test Configuration**:
- Domain with high-velocity flow AND high thermal diffusivity
- Vary dt while keeping dx fixed
- Monitor: fluid stability, thermal stability, coupling stability

**Validation Metrics**:
1. **Fluid CFL**: `u * dt / dx < 1` (explicit fluid solver limit)
2. **Thermal CFL**: `alpha * dt / dx^2 < 0.5` (diffusion stability)
3. **Coupling stability**: Check temperature advection: `u * dt / dx < 1`
4. **Critical dt**: Find maximum stable dt for given dx, u, alpha

**Expected Result**: Stability limits match theoretical CFL conditions

---

### 5.4 Subcycling Stability (VOF)

**File**: `/tests/validation/stability/test_vof_subcycling_stability.cu`

**Physics Description**:
- VOF advection may require smaller time steps than LBM
- Test subcycling: multiple VOF steps per LBM step
- Verify: stability, mass conservation, interface sharpness

**Test Configuration**:
- Interface advection in known velocity field
- Test subcycling ratios: 1:1, 2:1, 4:1, 8:1 (VOF:LBM)
- Run 5000 LBM steps

**Validation Metrics**:
1. **Interface sharpness**: `max|f - 0| + max|f - 1|` for interface cells
2. **Mass conservation**: `|m(t) - m(0)| / m(0) < 1e-4`
3. **Courant number**: `|u| * dt_VOF / dx < 0.5` for stable advection
4. **Optimal subcycling**: Find minimum ratio that maintains sharp interface

**Expected Result**: Subcycling improves interface sharpness without instability

---

## Test Infrastructure

### Tolerance Philosophy

**Tolerance Selection Guidelines**:
1. **Analytical solutions**: 5-6% L2 error (inherent LBM discretization error)
2. **Conservation laws**: 0.1% for exact conservation, 5% for balanced terms
3. **Convergence rates**: ±10% around theoretical order (e.g., 1.8 < p < 2.2)
4. **Dimensionless numbers**: 8-10% error (includes multiple discretization sources)
5. **Stability tests**: Binary (stable/unstable) with threshold criteria

**Rationale**: LBM is second-order accurate in space, first-order in time. Expect O(h^2) spatial error and O(dt) temporal error. For typical test resolutions (h~1, dt~1 in lattice units), errors of 2-5% are expected. Tolerances set at 5-6% allow for this inherent discretization error plus small numerical artifacts.

### Test Execution Framework

**Automated Test Suite**:
```cmake
# CMakeLists.txt additions
add_subdirectory(tests/validation/analytical)
add_subdirectory(tests/validation/conservation)
add_subdirectory(tests/validation/convergence)
add_subdirectory(tests/validation/scaling)
add_subdirectory(tests/validation/stability)

# Master test target
add_custom_target(test_validation
    COMMAND ctest --output-on-failure -R "validation_.*"
    COMMENT "Running validation test suite"
)

# Generate test report
add_custom_target(test_report
    COMMAND python3 ${CMAKE_SOURCE_DIR}/scripts/generate_test_report.py
    COMMENT "Generating validation test report"
)
```

**Test Naming Convention**:
```
validation_analytical_poiseuille_flow
validation_analytical_couette_flow
validation_analytical_heat_conduction_1d
validation_analytical_taylor_green_vortex
validation_conservation_mass
validation_conservation_momentum
validation_conservation_energy
validation_conservation_entropy
validation_convergence_grid_poiseuille
validation_convergence_temporal_heat
validation_convergence_combined_taylor_green
validation_scaling_reynolds_number
validation_scaling_peclet_number
validation_scaling_mach_number
validation_scaling_force_balance
validation_stability_max_velocity
validation_stability_max_temperature_gradient
validation_stability_cfl_condition
validation_stability_vof_subcycling
```

### Continuous Integration

**Regression Testing**:
- Run full validation suite on every PR
- Compare results to baseline (expected values ± tolerance)
- Flag any test that moves closer to failure boundary

**Performance Tracking**:
- Record execution time for each test
- Monitor for performance regressions
- Ensure tests complete in reasonable time (<5 min each)

### Visualization and Reporting

**Test Output Files**:
- Each test exports data: `{test_name}_results.txt`
- Contains: measured values, analytical values, errors, pass/fail
- Format: CSV or JSON for easy parsing

**Automated Plots**:
```python
# scripts/generate_test_report.py
# - Parse all test output files
# - Generate convergence plots (error vs h, dt)
# - Generate profile comparisons (LBM vs analytical)
# - Create summary dashboard with all test results
```

**Example Validation Report Sections**:
1. **Summary Table**: Test name, status (pass/fail), error metric, tolerance
2. **Convergence Plots**: log-log plots showing convergence rates
3. **Profile Comparisons**: Line plots of LBM vs analytical solutions
4. **Conservation Tracking**: Time series of conserved quantities
5. **Stability Maps**: Parameter space showing stable/unstable regions

---

## Implementation Priority

**Phase 1 (High Priority - Core Validation)**:
1. test_poiseuille_flow_convergence.cu (existing, enhance with convergence study)
2. test_heat_conduction_1d.cu (new)
3. test_mass_conservation_detailed.cu (enhance existing test_vof_mass_conservation.cu)
4. test_energy_conservation_detailed.cu (existing test_energy_conservation_full.cu is good)
5. test_grid_convergence_poiseuille.cu (new)

**Phase 2 (Medium Priority - Extended Validation)**:
6. test_couette_flow.cu (new)
7. test_taylor_green_vortex.cu (new)
8. test_momentum_conservation.cu (new)
9. test_temporal_convergence_heat_diffusion.cu (new)
10. test_reynolds_number_scaling.cu (new)

**Phase 3 (Lower Priority - Advanced Tests)**:
11. test_entropy_production.cu (new, research-grade)
12. test_combined_convergence_taylor_green.cu (new)
13. test_peclet_number_scaling.cu (new)
14. test_mach_number_limit.cu (new)
15. test_force_balance_melt_pool.cu (new)
16. Stability tests (most can leverage existing stability tests)

---

## Success Criteria

A test suite is considered comprehensive and rigorous if:

1. **Coverage**: All major physics modules have analytical validation tests
2. **Accuracy**: LBM achieves expected order of accuracy (p=2 spatial, p=1 temporal)
3. **Conservation**: All conserved quantities verified to <1% drift (or match sources/sinks)
4. **Scaling**: Dimensionless numbers correctly reproduce physical scaling laws
5. **Stability**: Stability boundaries identified and documented
6. **Reproducibility**: All tests pass consistently across different hardware/compilers
7. **Documentation**: Each test has clear physics description and expected results

**Target**: 90% of tests pass with errors below tolerances, 100% execute without crashes.

---

## References for Analytical Solutions

1. **Poiseuille Flow**: White, F.M. "Viscous Fluid Flow" (2006), Section 3.2
2. **Couette Flow**: Batchelor, G.K. "An Introduction to Fluid Dynamics" (2000), Section 4.1
3. **Heat Conduction**: Carslaw & Jaeger "Conduction of Heat in Solids" (1959), Section 2.3
4. **Taylor-Green Vortex**: Taylor, G.I. & Green, A.E., Proc. R. Soc. A (1937)
5. **Stefan Problem**: Alexiades & Solomon "Mathematical Modeling of Melting and Freezing" (1993)
6. **LBM Validation**: Krüger et al. "The Lattice Boltzmann Method" (2017), Chapter 8
7. **Dimensionless Numbers**: Bird, Stewart & Lightfoot "Transport Phenomena" (2007)

---

**Document Version**: 1.0
**Date**: 2025-12-02
**Author**: LBM-CUDA Architecture Team
**Status**: Design Specification (Implementation Pending)
