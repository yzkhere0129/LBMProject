# Marangoni Benchmark Architecture Design

## Executive Summary

This document specifies the architecture for a standalone Marangoni convection benchmark test that can be directly compared with waLBerla's thermocapillary showcase. The benchmark validates thermocapillary flow in a microchannel geometry using analytical solutions.

**Primary Goal:** Validate coupled thermal-fluid-Marangoni physics in an isolated test case without complexity from VOF, phase change, or laser heating.

---

## 1. waLBerla Reference Implementation Analysis

### 1.1 Test Case: Microchannel Thermocapillary Flow

**Reference:** [Chai et al., JCP 2013](https://www.sciencedirect.com/science/article/pii/S0021999113005986)

**Physics:**
- Two immiscible fluids (top and bottom) separated by a horizontal interface at y=0
- Temperature gradient applied along the interface (sinusoidal in x-direction)
- Marangoni stress drives flow along the interface
- No gravity, no phase change, no free surface deformation

**Domain Configuration (waLBerla):**
```
Geometry: 2D microchannel (quasi-2D in 3D code)
Size:     (2*L_ref, L_ref, 1) = (512, 256, 1) cells
Periodic: x-direction only
Walls:    y-top and y-bottom (NoSlip for hydro, fixed T for thermal)
```

**Interface Position:**
- Horizontal interface at y = L_ref/2 (middle of domain)
- Interface initialized with phase-field: φ = 0.5 + 0.5*tanh((y-y_mid)/(W/2))
- Interface thickness W = 5 cells

**Boundary Conditions:**
```
X-direction: Periodic (allows sinusoidal temperature wave)
Y-top:       NoSlip wall, Fixed temperature T = T_ref (static Dirichlet)
Y-bottom:    NoSlip wall, T = T_h + T_0*cos(π*(x-x_mid)/x_mid) (dynamic Dirichlet)
```

**Temperature Profile:**
- T_h = 20 (hot side reference)
- T_0 = 4 (amplitude of sinusoidal variation)
- T_ref = 10 (cold side reference)
- Creates sinusoidal temperature variation along interface

### 1.2 Physical Parameters

**Case 1: Equal thermal conductivity**
```yaml
rho_liquid: 1.0
rho_gas: 1.0
mu_liquid: 0.2      # tau = 3*mu = 0.6, so omega = 1/(tau+0.5) = 0.909
mu_gas: 0.2

sigma_ref: 0.025    # Reference surface tension
sigma_t: -5e-4      # Temperature coefficient dσ/dT

kappa_liquid: 0.2   # Thermal conductivity
kappa_gas: 0.2

T_ref: 10
T_h: 20
T_0: 4

interface_thickness: 5
mobility: 0.05
```

**Case 2: Different thermal conductivity**
```yaml
# Same as Case 1, except:
kappa_liquid: 0.04
kappa_gas: 0.2
```

### 1.3 Analytical Solution

waLBerla uses `lbmpy.phasefield_allen_cahn.analytical.analytical_solution_microchannel()`:

**Inputs:**
- reference_length: L_ref = 256
- length_x, length_y: Domain size
- kappa_top, kappa_bottom: Thermal conductivities
- t_h, t_c, t_0: Temperature parameters
- sigma_t: dσ/dT
- mu: Dynamic viscosity

**Outputs:**
- x, y: Coordinate meshes
- u_x(x,y): Velocity field x-component
- u_y(x,y): Velocity field y-component
- T(x,y): Temperature field

**Characteristic Velocity:**
```
u_max = -(T_0 * sigma_t / mu) * g * h
```
Where g, h are geometric factors from the analytical solution.

### 1.4 Validation Metrics

waLBerla computes:
```python
L2_T = sqrt(sum((T_sim - T_analytical)²) / sum(T_analytical²))
L2_U = sqrt(sum((|U_sim| - |U_analytical|)²) / sum(|U_analytical|²))
```

Typical convergence: L2_T < 0.01, L2_U < 0.05 for well-resolved cases

---

## 2. Our LBMProject Implementation Architecture

### 2.1 Design Philosophy

**Key Principle:** Isolate Marangoni physics from other complexities

**What to include:**
- ThermalLBM with fixed temperature BCs
- FluidLBM with no-slip walls
- Marangoni force computation
- Simplified two-phase approach (no VOF)

**What to exclude:**
- VOF solver (use simple phase indicator instead)
- Phase change
- Laser heating
- Evaporation/recoil pressure
- Gravity/buoyancy

### 2.2 Module Integration Strategy

#### Option A: Pure Marangoni Test (Recommended)
```
Components:
1. ThermalLBM - Temperature evolution with BCs
2. FluidLBM - Velocity evolution
3. Marangoni force - Surface tension gradient force
4. Phase indicator field - Static (no VOF solver)

Coupling:
ThermalLBM → Temperature field → Marangoni force → FluidLBM
FluidLBM → Velocity field → ThermalLBM (convection term)
```

**Advantages:**
- Minimal dependencies
- Fast execution
- Clear validation of Marangoni implementation
- Easy debugging

#### Option B: With VOF but No Phase Change
```
Components:
1. ThermalLBM
2. FluidLBM
3. VOFSolver (advection only, no reconstruction needed for flat interface)
4. Marangoni force

Coupling:
Similar to Option A but VOF evolves phase field
```

**Disadvantages:**
- Adds complexity
- Flat interface doesn't need VOF
- Harder to isolate bugs

**Recommendation:** Use Option A for initial validation, Option B for later stress-testing VOF.

### 2.3 Code Structure

```
File: /home/yzk/LBMProject/tests/validation/test_marangoni_microchannel.cu

Structure:
├── Domain setup (match waLBerla geometry)
├── Initialize phase field (analytical tanh profile)
├── Initialize temperature field (analytical solution)
├── Initialize velocity field (zero or analytical)
├── Time loop:
│   ├── ThermalLBM collision
│   ├── ThermalLBM streaming
│   ├── Thermal boundary conditions
│   ├── Compute temperature gradients
│   ├── Compute Marangoni force
│   ├── FluidLBM collision (with Marangoni force)
│   ├── FluidLBM streaming
│   └── Fluid boundary conditions
├── Compute L2 errors vs analytical solution
└── Output VTK for visualization
```

### 2.4 Domain Configuration

**Geometry:**
```cpp
// Match waLBerla Case 1
const int L_ref = 256;
const int NX = 2 * L_ref;  // 512 - periodic in x
const int NY = L_ref;       // 256 - walls at top/bottom
const int NZ = 1;           // Quasi-2D

// Physical units (can be lattice units = 1 for this test)
const float dx = 1.0f;  // Lattice spacing
const float dt = 1.0f;  // Time step
```

**Boundary Conditions:**
```cpp
// Thermal BC
BC_thermal_top:    Dirichlet, T = T_ref = 10.0
BC_thermal_bottom: Dirichlet, T = T_h + T_0*cos(π*(x-x_mid)/x_mid)
                              where T_h = 20, T_0 = 4

BC_thermal_x: Periodic

// Fluid BC
BC_fluid_top:    No-slip wall (bounce-back)
BC_fluid_bottom: No-slip wall (bounce-back)
BC_fluid_x:      Periodic
```

### 2.5 Phase Field Initialization

**Purpose:** Define which cells are "liquid" (top) vs "gas" (bottom)

**Implementation:**
```cpp
__global__ void initializePhaseField(float* phase, int nx, int ny, int nz) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nx*ny*nz) return;

    int x = idx % nx;
    int y = (idx / nx) % ny;
    int z = idx / (nx * ny);

    const float y_mid = ny / 2.0f;
    const float W = 5.0f;  // Interface thickness

    float y_global = y;
    phase[idx] = 0.5f + 0.5f * tanhf((y_global - y_mid) / (W / 2.0f));
    // phase = 0 (bottom fluid), phase = 1 (top fluid)
}
```

**Interpretation:**
- phase < 0.5: Bottom fluid (properties: rho_gas, mu_gas, kappa_gas)
- phase > 0.5: Top fluid (properties: rho_liquid, mu_liquid, kappa_liquid)
- Interface region: Smooth transition over ~5 cells

### 2.6 Temperature Field Initialization

**Option 1: Initialize to analytical solution**
```cpp
// Set T to analytical solution at t=0
// This gives instant steady state for validation
initTemperatureAnalytical(T, nx, ny, analytical_params);
```

**Option 2: Initialize to boundary values and evolve**
```cpp
// Top: T = T_ref
// Bottom: T = T_h + T_0*cos(...)
// Interior: Linear interpolation or T_ref
// Run thermal solver until steady state
```

**Recommendation:** Use Option 1 for quick validation, Option 2 to test thermal solver convergence.

### 2.7 Thermal LBM Configuration

**Setup:**
```cpp
// Material properties from phase field
MaterialProperties mat_liquid;
mat_liquid.thermal_conductivity = 0.2f;  // kappa_liquid
mat_liquid.density = 1.0f;
mat_liquid.specific_heat = 1.0f;  // Can set to 1 in lattice units

MaterialProperties mat_gas;
mat_gas.thermal_conductivity = 0.2f;  // kappa_gas (Case 1) or 0.04 (Case 2)
mat_gas.density = 1.0f;
mat_gas.specific_heat = 1.0f;

// Compute effective thermal diffusivity
float alpha_liquid = kappa_liquid / (rho_liquid * cp_liquid);
float alpha_gas = kappa_gas / (rho_gas * cp_gas);

// ThermalLBM needs spatially varying diffusivity
// Option A: Use average diffusivity, handle heterogeneity in source term
// Option B: Recompute tau locally based on phase field
```

**Boundary Implementation:**
```cpp
// After streaming, apply BCs
applyThermalDirichletBC_Top(g_field, T_ref, nx, ny, nz);
applyThermalDirichletBC_Bottom_Sinusoidal(g_field, T_h, T_0, L_ref, nx, ny, nz);
```

**Key Challenge:**
Our ThermalLBM may need modification to support spatially varying thermal conductivity based on phase field.

**Solution Approach:**
1. **Simple:** Use average properties, accept small error at interface
2. **Better:** Compute local tau based on phase field in collision kernel
3. **Best:** Implement harmonic mean for interface cells

### 2.8 Marangoni Force Computation

**Physical Model:**
```
Surface tension gradient force:
F_Marangoni = ∇_s σ = (dσ/dT) * ∇_s T

Where ∇_s is the surface gradient operator (tangent to interface)
```

**Implementation Strategy:**

**Step 1: Compute temperature gradient**
```cpp
__global__ void computeTemperatureGradient(
    const float* T, float* dTdx, float* dTdy, float* dTdz,
    int nx, int ny, int nz)
{
    // Central difference for interior points
    // One-sided for boundary points
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // ...
    dTdx[idx] = (T[idx+1] - T[idx-1]) / (2.0f * dx);
    dTdy[idx] = (T[idx+nx] - T[idx-nx]) / (2.0f * dx);
    // ...
}
```

**Step 2: Project gradient onto interface**

For horizontal interface at y = y_interface:
```cpp
// Interface normal: n = (0, 1, 0) pointing from bottom to top
// Tangential gradient: ∇_s T = ∇T - (∇T·n)n
//                             = (dT/dx, 0, dT/dz)
// For 2D case: ∇_s T = (dT/dx, 0, 0)

__global__ void computeMarangoniForce(
    const float* dTdx, const float* dTdy, const float* dTdz,
    const float* phase,
    float* fx, float* fy, float* fz,
    float sigma_t,  // dσ/dT
    int nx, int ny, int nz)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nx*ny*nz) return;

    // Detect interface cells (0.4 < phase < 0.6)
    float phi = phase[idx];
    if (phi < 0.4f || phi > 0.6f) {
        fx[idx] = fy[idx] = fz[idx] = 0.0f;
        return;
    }

    // Marangoni force is tangent to interface
    // For horizontal interface: F = (dσ/dT * dT/dx, 0, dσ/dT * dT/dz)
    fx[idx] = sigma_t * dTdx[idx];
    fy[idx] = 0.0f;  // No normal component
    fz[idx] = sigma_t * dTdz[idx];  // Zero for 2D case
}
```

**Alternative: Use surface tension force module**
```cpp
// If we have src/physics/vof/surface_tension_force.cu
// Modify it to compute σ(T) gradient instead of CSF force

// Current: F_CSF = σ * κ * n * δ_interface
// Needed:  F_Marangoni = (dσ/dT) * ∇_s T
```

**Recommendation:** Implement simplified kernel above first, then integrate with existing surface tension module.

### 2.9 Fluid LBM Configuration

**Setup:**
```cpp
// Kinematic viscosity
float nu_liquid = mu_liquid / rho_liquid = 0.2 / 1.0 = 0.2;
float nu_gas = mu_gas / rho_gas = 0.2 / 1.0 = 0.2;

// Relaxation time (BGK)
float tau_liquid = 3.0f * nu_liquid + 0.5f = 1.1f;
float tau_gas = 3.0f * nu_gas + 0.5f = 1.1f;
float omega_liquid = 1.0f / tau_liquid = 0.909f;

// FluidLBM constructor
FluidLBM fluid_solver(NX, NY, NZ,
                      nu_liquid,  // Use liquid viscosity (or average)
                      rho_liquid,
                      BoundaryType::PERIODIC,  // x
                      BoundaryType::WALL,      // y (top/bottom)
                      BoundaryType::PERIODIC); // z
```

**Force Coupling:**
```cpp
// In time loop:
computeMarangoniForce(dTdx, dTdy, dTdz, phase, fx, fy, fz, sigma_t, NX, NY, NZ);

// FluidLBM collision with force
fluid_solver.setBodyForce(fx, fy, fz);  // Sets force for next collision
fluid_solver.collisionBGK();
```

**Key Points:**
- Disable buoyancy force (or set thermal expansion coefficient = 0)
- Disable Darcy damping (or set porosity = 1 everywhere)
- Only Marangoni force should drive flow

### 2.10 Boundary Condition Implementation

**Thermal BC:**

```cpp
// Top boundary: y = NY-1, Dirichlet BC, T = T_ref
__global__ void applyThermalDirichletTop(
    float* g_field,  // Thermal PDFs (D3Q7)
    float T_wall,
    int nx, int ny, int nz)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int k = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= nx || k >= nz) return;

    int j = ny - 1;  // Top wall
    int idx = i + j*nx + k*nx*ny;

    // Set PDFs to equilibrium at wall temperature, zero velocity
    // g_eq^i = w_i * T_wall * (1 + u·e_i/cs^2 + ...)
    // For u=0: g_eq^i = w_i * T_wall

    g_field[idx + 0*nx*ny*nz] = w0 * T_wall;  // Center
    g_field[idx + 1*nx*ny*nz] = w1 * T_wall;  // +x
    g_field[idx + 2*nx*ny*nz] = w1 * T_wall;  // -x
    // ... etc for all 7 directions
}

// Bottom boundary: y = 0, Dirichlet BC, T = T_h + T_0*cos(π*(x-x_mid)/x_mid)
__global__ void applyThermalDirichletBottom(
    float* g_field,
    float T_h, float T_0, float L_ref,
    int nx, int ny, int nz)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int k = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= nx || k >= nz) return;

    int j = 0;  // Bottom wall
    int idx = i + j*nx + k*nx*ny;

    float x_mid = L_ref;
    float T_wall = T_h + T_0 * cosf(M_PI * (i - x_mid) / x_mid);

    // Set equilibrium PDFs
    g_field[idx + 0*nx*ny*nz] = w0 * T_wall;
    g_field[idx + 1*nx*ny*nz] = w1 * T_wall;
    // ... etc
}
```

**Fluid BC:**

```cpp
// Top and bottom: No-slip wall (bounce-back)
// This is likely already implemented in FluidLBM::applyBoundaryConditions()
// with BoundaryType::WALL

// Verify it applies bounce-back at j=0 and j=NY-1
```

### 2.11 Analytical Solution Integration

**Approach 1: Port Python code to C++**

```cpp
struct AnalyticalSolution {
    float* x;
    float* y;
    float* u_x;
    float* u_y;
    float* T;

    int nx, ny;
};

AnalyticalSolution computeAnalyticalSolution(
    int reference_length,
    int length_x, int length_y,
    float kappa_top, float kappa_bottom,
    float T_h, float T_c, float T_0,
    float sigma_t, float mu)
{
    // Port lbmpy.phasefield_allen_cahn.analytical.analytical_solution_microchannel()
    // to C++

    // This is ~90 lines of Python with numpy operations
    // Need to translate sinh, cosh, cos, sin operations

    AnalyticalSolution sol;
    sol.nx = length_x;
    sol.ny = length_y;

    // Allocate host memory
    sol.x = new float[length_x * length_y];
    sol.y = new float[length_x * length_y];
    sol.u_x = new float[length_x * length_y];
    sol.u_y = new float[length_x * length_y];
    sol.T = new float[length_x * length_y];

    // Compute analytical solution
    // ... (translation of Python code)

    return sol;
}
```

**Approach 2: Call Python from C++ (testing only)**

```cpp
// Use pybind11 or just run Python script separately
// Compare outputs via CSV files

// test_marangoni_microchannel.cu outputs: velocity_sim.csv, temp_sim.csv
// Python script computes analytical solution: velocity_analytical.csv, temp_analytical.csv
// Python script computes L2 error and generates plots
```

**Recommendation:** Use Approach 2 for initial testing (faster development), then implement Approach 1 for production benchmark.

### 2.12 Time Stepping and Convergence

**Timestep Selection:**
```cpp
// Thermal stability: CFL_thermal = alpha * dt / dx^2 < 0.5
// For alpha ~ 0.2, dx=1, dt=1: CFL = 0.2 (OK)

// Viscous stability: CFL_viscous = nu * dt / dx^2 < 0.5
// For nu = 0.2, dx=1, dt=1: CFL = 0.2 (OK)

// Convective stability: CFL_conv = u_max * dt / dx < 1.0
// Need to estimate u_max from analytical solution: u_max ~ 0.001-0.01 (very slow flow)
// For u_max = 0.01, dt=1, dx=1: CFL = 0.01 (OK)

const float DT = 1.0f;  // Lattice time step
```

**Convergence Criterion:**
```cpp
// Approach 1: Run to steady state
// Monitor max velocity change between steps
float max_du = computeMaxVelocityChange(u_old, u_new, nx*ny*nz);
if (max_du < 1e-6) {
    printf("Converged at step %d\n", timestep);
    break;
}

// Approach 2: Run fixed number of steps (if initialized to analytical solution)
const int NUM_STEPS = 10000;  // Should be steady already if T and u initialized correctly
```

**Recommendation:** Initialize to analytical solution, run 1000 steps for settling, then validate.

---

## 3. Implementation Roadmap

### Phase 1: Basic Infrastructure (Day 1)
- [ ] Create test file: `test_marangoni_microchannel.cu`
- [ ] Set up domain with waLBerla parameters
- [ ] Implement phase field initialization kernel
- [ ] Implement temperature initialization (boundary values)
- [ ] Verify BCs with VTK output

### Phase 2: Thermal Solver (Day 1-2)
- [ ] Set up ThermalLBM with Dirichlet BCs
- [ ] Implement sinusoidal bottom BC
- [ ] Run thermal solver to steady state
- [ ] Verify temperature field matches expected pattern
- [ ] Output VTK for visual inspection

### Phase 3: Marangoni Force (Day 2)
- [ ] Implement temperature gradient kernel
- [ ] Implement Marangoni force kernel
- [ ] Verify force is tangent to interface
- [ ] Check force magnitude vs analytical estimate
- [ ] Output force field to VTK

### Phase 4: Coupled Solver (Day 2-3)
- [ ] Set up FluidLBM with no-slip walls
- [ ] Couple Marangoni force to fluid solver
- [ ] Run coupled thermal-fluid-Marangoni simulation
- [ ] Monitor convergence to steady state
- [ ] Verify velocity field pattern (convection rolls)

### Phase 5: Validation (Day 3)
- [ ] Port or interface with analytical solution
- [ ] Compute L2 error for temperature field
- [ ] Compute L2 error for velocity field
- [ ] Generate comparison plots (sim vs analytical)
- [ ] Document results

### Phase 6: Parametric Study (Day 4)
- [ ] Test Case 1 (equal thermal conductivity)
- [ ] Test Case 2 (different thermal conductivity)
- [ ] Grid refinement study (128, 256, 512)
- [ ] Document convergence behavior

---

## 4. Expected Results

### 4.1 Physical Behavior

**Temperature Field:**
- Sinusoidal variation along bottom boundary
- Constant temperature at top boundary
- Smooth transition across interface
- Different temperature gradients in top/bottom fluids (Case 2)

**Velocity Field:**
- Convection rolls aligned with temperature variation
- Flow from cold to hot along interface (negative x-direction if sigma_t < 0)
- Return flow in bulk fluid above and below interface
- Maximum velocity at interface
- Zero velocity at walls

**Flow Pattern:**
```
Top wall (T=T_ref, u=0)
    ↓ slow return flow ↓
==================== Interface (T varies, u_max here)
    ↑ slow return flow ↑
Bottom wall (T=T_h+T_0*cos(), u=0)

Side view shows counter-rotating vortices driven by temperature gradients
```

### 4.2 Quantitative Metrics

**Target Accuracy (based on waLBerla results):**
```
L2_T < 0.01  (1% error in temperature field)
L2_U < 0.05  (5% error in velocity field)
```

**Typical Values:**
```
u_max ~ O(10^-3) in lattice units
T_max = 24 (at bottom hot spots)
T_min = 10 (at top)
```

**Grid Convergence:**
```
L_ref = 128:  L2_U ~ 0.10
L_ref = 256:  L2_U ~ 0.05  (baseline)
L_ref = 512:  L2_U ~ 0.02  (2nd order convergence expected)
```

### 4.3 Visualization Checklist

**VTK Output Fields:**
- [x] Temperature (scalar)
- [x] Velocity magnitude (scalar)
- [x] Velocity vectors (vector)
- [x] Phase field (scalar)
- [x] Marangoni force (vector)
- [x] Temperature gradient (vector)

**ParaView Verification:**
1. Temperature: Should show sinusoidal pattern with wavelength = 2*L_ref
2. Velocity: Should show convection cells with u_max at interface
3. Streamlines: Should show closed circulation loops
4. Marangoni force: Should be concentrated at interface, tangent to surface

---

## 5. Comparison Methodology with waLBerla

### 5.1 Direct Comparison

**Match these exactly:**
- Domain size: (512, 256, 1)
- Reference length: 256
- Physical parameters (Case 1 or Case 2)
- Boundary conditions
- Interface position and thickness

**Compare outputs:**
- VTK files at steady state
- L2 error metrics
- Velocity profiles along vertical lines
- Temperature profiles along horizontal lines

### 5.2 Validation Protocol

**Step 1: Reproduce waLBerla Case 1**
```bash
# Run our implementation
./test_marangoni_microchannel --case 1 --nx 512 --ny 256

# Expected output:
# - velocity_sim.vtk
# - temperature_sim.vtk
# - L2_T = 0.008 ± 0.002
# - L2_U = 0.045 ± 0.010
```

**Step 2: Side-by-side comparison**
```python
# Load waLBerla VTK and our VTK
# Plot temperature contours overlay
# Plot velocity quiver overlay
# Compute pointwise difference

import pyvista as pv
wlb_data = pv.read('walberla_output.vtu')
our_data = pv.read('velocity_sim.vtk')

# Compare
diff = our_data['velocity'] - wlb_data['velocity']
print(f"Max velocity difference: {np.max(np.abs(diff))}")
```

**Step 3: Analytical validation**
```python
# Python script to compute analytical solution
from lbmpy.phasefield_allen_cahn.analytical import analytical_solution_microchannel

x, y, u_x, u_y, T = analytical_solution_microchannel(
    reference_length=256,
    length_x=512, length_y=256,
    kappa_top=0.2, kappa_bottom=0.2,
    t_h=20, t_c=10, t_0=4,
    sigma_t=-5e-4,
    mu=0.2
)

# Load our simulation data
# Compute L2 errors
# Generate comparison plots
```

### 5.3 Success Criteria

**Minimum Viable Product (MVP):**
- [ ] Test compiles and runs
- [ ] Reaches steady state (convergence criterion met)
- [ ] Qualitatively correct flow pattern (convection rolls visible)
- [ ] Temperature field shows sinusoidal variation
- [ ] L2_T < 0.05 (5% error acceptable for first version)

**Production Quality:**
- [ ] L2_T < 0.01
- [ ] L2_U < 0.05
- [ ] Matches waLBerla results within 10% for all metrics
- [ ] Grid convergence study shows 2nd order accuracy
- [ ] Both Case 1 and Case 2 validated

**Stretch Goals:**
- [ ] Matches waLBerla results within 1%
- [ ] Performance benchmarking (MLUPS comparison)
- [ ] 3D extension (droplet migration test)

---

## 6. Key Architectural Decisions

### Decision 1: Use Simple Phase Field vs VOF

**Choice:** Static phase field (no VOF solver)

**Rationale:**
- Microchannel test has flat, stationary interface
- VOF adds complexity without benefit for this case
- Validates Marangoni physics independently
- Faster debugging and iteration

**Future:** Add VOF for droplet migration test (curved, moving interface)

### Decision 2: Thermal Conductivity Handling

**Challenge:** ThermalLBM needs spatially varying thermal conductivity

**Options:**
A. Single tau, average properties (simplest)
B. Compute local tau from phase field in collision kernel
C. Harmonic averaging at interface cells (most accurate)

**Choice:** Start with B (local tau), upgrade to C if needed

**Implementation:**
```cpp
__global__ void thermalCollisionBGK_variable_tau(
    float* g, const float* T, const float* u,
    const float* phase,  // NEW: phase field input
    float alpha_liquid, float alpha_gas,  // NEW: both diffusivities
    int nx, int ny, int nz)
{
    int idx = ...;

    // Compute local diffusivity based on phase
    float phi = phase[idx];
    float alpha = (1.0f - phi) * alpha_gas + phi * alpha_liquid;

    // Compute local tau
    float tau = alpha / cs2 + 0.5f;
    float omega = 1.0f / tau;

    // BGK collision with local omega
    // ...
}
```

### Decision 3: Force Application Method

**Options:**
A. Exact difference method (Guo forcing)
B. Simple force addition to equilibrium
C. Body force in collision operator

**Choice:** Use existing FluidLBM force interface (likely Guo forcing)

**Interface:**
```cpp
fluid_solver.setBodyForce(fx, fy, fz);  // Marangoni force only
// FluidLBM internally applies Guo forcing scheme
```

### Decision 4: Boundary Condition Strategy

**Thermal BC:**
- Use Zou-He scheme or equilibrium BC for Dirichlet conditions
- Need to implement dynamic BC for sinusoidal bottom

**Fluid BC:**
- Use standard bounce-back for no-slip walls
- Periodic in x-direction (likely already supported)

**Implementation:**
```cpp
// After streaming step:
applyPeriodicBC_X(f_field, nx, ny, nz);  // Existing
applyBounceBack_Y(f_field, nx, ny, nz);  // Existing
applyThermalDirichlet_Y(g_field, T_values, nx, ny, nz);  // NEW
```

---

## 7. Testing and Debugging Strategy

### 7.1 Unit Tests

**Test 1: Phase field initialization**
```cpp
// Initialize phase field
// Check: phase[x, 0, 0] = 0 (bottom)
//        phase[x, NY-1, 0] = 1 (top)
//        phase[x, NY/2, 0] = 0.5 (interface)
```

**Test 2: Temperature BCs**
```cpp
// Apply thermal BCs
// Check: T[x, NY-1, 0] = T_ref for all x
//        T[x, 0, 0] = T_h + T_0*cos(...) matches formula
```

**Test 3: Temperature gradient**
```cpp
// Set linear temperature: T = a*x + b*y + c
// Compute gradient
// Check: dT/dx = a, dT/dy = b (within 1% for interior)
```

**Test 4: Marangoni force**
```cpp
// Set T = x (linear in x)
// Phase = 0.5 (uniform interface)
// Compute force
// Check: fx = sigma_t * 1.0 (constant)
//        fy = 0 (no y-component)
```

### 7.2 Integration Tests

**Test 5: Thermal solver convergence**
```cpp
// Initialize T to boundary values
// Run 10000 thermal steps (no flow)
// Check: reaches steady state
//        T profile matches analytical (without flow)
```

**Test 6: Flow without Marangoni**
```cpp
// Disable Marangoni force
// Apply small body force
// Check: velocity field develops
//        Obeys no-slip BCs
```

**Test 7: Coupled thermal-fluid**
```cpp
// Enable convection in thermal solver
// Run coupled simulation
// Check: heat is advected by flow
//        Temperature profile changes
```

### 7.3 Debugging Workflow

**If simulation crashes:**
1. Check array bounds (nx*ny*nz indexing)
2. Check NaN in fields (print min/max each step)
3. Check BC indexing (j=0 and j=NY-1)

**If velocity field is wrong:**
1. Verify Marangoni force direction (should point from cold to hot for sigma_t < 0)
2. Check force magnitude (print max force)
3. Verify force is only at interface (print force vs phase)
4. Check fluid BC (should be no-slip)

**If temperature field is wrong:**
1. Check thermal BCs (plot T at boundaries)
2. Check thermal diffusivity (too high → oversmoothing)
3. Verify sinusoidal BC (plot T_bottom vs x)

**If L2 error is high:**
1. Check analytical solution implementation
2. Verify coordinate systems match (x offset?)
3. Increase resolution (256 → 512)
4. Run longer (may not be at steady state)

---

## 8. File Structure

```
/home/yzk/LBMProject/
├── tests/validation/test_marangoni_microchannel.cu  [NEW]
├── include/physics/marangoni_force.h                [NEW]
├── src/physics/marangoni/marangoni_force.cu         [NEW]
├── scripts/analytical_marangoni.py                  [NEW]
├── scripts/validate_marangoni.py                    [NEW]
├── docs/MARANGONI_BENCHMARK_DESIGN.md               [THIS FILE]
└── configs/marangoni_case1.yaml                     [NEW]
```

**New files needed:**

1. **test_marangoni_microchannel.cu**: Main test driver
2. **marangoni_force.h**: Interface for Marangoni force computation
3. **marangoni_force.cu**: CUDA kernels for force computation
4. **analytical_marangoni.py**: Python script with analytical solution
5. **validate_marangoni.py**: Comparison and plotting script
6. **marangoni_case1.yaml**: Configuration file (optional)

---

## 9. Success Metrics

### 9.1 Technical Validation

- [ ] Temperature field L2 error < 1%
- [ ] Velocity field L2 error < 5%
- [ ] Converges in < 10000 timesteps
- [ ] Passes Case 1 and Case 2
- [ ] Grid convergence shows O(h²) accuracy

### 9.2 Code Quality

- [ ] Self-contained test (no manual parameter tuning)
- [ ] Clear console output (reports L2 errors)
- [ ] Generates VTK for visualization
- [ ] Documented with physics equations
- [ ] < 500 lines of code (excluding analytical solution)

### 9.3 Scientific Impact

- [ ] Validates Marangoni implementation
- [ ] Provides reference benchmark for future work
- [ ] Enables debugging of Marangoni issues in full LPBF simulation
- [ ] Publishable result (comparison with waLBerla)

---

## 10. Risk Assessment

### Risk 1: Analytical Solution Porting
**Impact:** High (can't validate without it)
**Likelihood:** Medium (complex math expressions)
**Mitigation:** Use Python for validation first, port to C++ later

### Risk 2: Spatially Varying Thermal Conductivity
**Impact:** Medium (affects temperature field accuracy)
**Likelihood:** High (current ThermalLBM may not support it)
**Mitigation:** Implement local tau computation; if fails, use average properties

### Risk 3: Interface Force Localization
**Impact:** Medium (force should be at interface, not bulk)
**Likelihood:** Low (can detect with phase field)
**Mitigation:** Use phase field to mask force; validate force distribution in VTK

### Risk 4: Boundary Condition Implementation
**Impact:** High (wrong BC → wrong solution)
**Likelihood:** Medium (sinusoidal BC is non-standard)
**Mitigation:** Test BCs separately before full simulation; plot boundary values

---

## 11. Next Steps

**Immediate (Day 1):**
1. Review this design document
2. Confirm waLBerla parameters are correct
3. Create test file skeleton
4. Implement phase field initialization
5. Set up VTK output

**Short-term (Week 1):**
1. Implement all kernels
2. Run first coupled simulation
3. Debug and iterate
4. Port analytical solution
5. Compute first validation metrics

**Medium-term (Week 2):**
1. Refine implementation based on results
2. Complete both test cases
3. Generate publication-quality plots
4. Document results in test report

---

## 12. References

1. **Chai et al. (2013):** "A comparative study of local and nonlocal Allen-Cahn equations with mass conservation" - JCP, defines the analytical solution

2. **waLBerla Thermocapillary showcase:**
   - `/home/yzk/walberla/apps/showcases/Thermocapillary/`
   - Reference implementation with benchmark script

3. **lbmpy analytical solutions:**
   - `lbmpy.phasefield_allen_cahn.analytical.analytical_solution_microchannel()`
   - Python implementation of analytical solution

4. **Our project references:**
   - Existing Marangoni implementation: `src/physics/vof/marangoni.cu`
   - ThermalLBM: `include/physics/thermal_lbm.h`
   - FluidLBM: `include/physics/fluid_lbm.h`

---

## Appendix A: Parameter Tables

### Test Case 1: Equal Thermal Conductivity

| Parameter | Symbol | Value | Units | LBM Units |
|-----------|--------|-------|-------|-----------|
| Domain size | L | (512, 256, 1) | cells | - |
| Reference length | L_ref | 256 | cells | - |
| Density (both) | ρ | 1.0 | - | 1.0 |
| Viscosity (both) | μ | 0.2 | - | 0.2 |
| Kinematic viscosity | ν | 0.2 | - | 0.2 |
| Relaxation time | τ | 1.1 | - | 1.1 |
| Surface tension ref | σ_0 | 0.025 | - | 0.025 |
| Surface tension coef | dσ/dT | -5×10⁻⁴ | - | -5×10⁻⁴ |
| Thermal conductivity (both) | κ | 0.2 | - | 0.2 |
| T reference (top) | T_ref | 10 | - | 10 |
| T hot (bottom ref) | T_h | 20 | - | 20 |
| T amplitude | T_0 | 4 | - | 4 |
| Interface thickness | W | 5 | cells | 5 |
| Mobility (phase field) | M | 0.05 | - | 0.05 |

### Test Case 2: Different Thermal Conductivity

Same as Case 1 except:

| Parameter | Symbol | Value | Units |
|-----------|--------|-------|-------|
| Thermal conductivity (top) | κ_liquid | 0.04 | - |
| Thermal conductivity (bottom) | κ_gas | 0.2 | - |

---

## Appendix B: Dimensional Analysis

**Characteristic scales:**
```
Length scale: L_c = L_ref = 256 cells
Velocity scale: U_c = |T_0 * σ_t / μ| = 4 * 5e-4 / 0.2 = 0.01
Time scale: t_c = L_c / U_c = 256 / 0.01 = 25600 steps
Temperature scale: T_c = T_0 = 4
```

**Non-dimensional numbers:**
```
Marangoni number: Ma = |σ_t| * T_c * L_c / (μ * α)
                     = 5e-4 * 4 * 256 / (0.2 * 0.2)
                     = 12.8

Reynolds number: Re = U_c * L_c / ν
                    = 0.01 * 256 / 0.2
                    = 12.8

Prandtl number: Pr = ν / α
                   = 0.2 / 0.2
                   = 1.0

(Note: Ma = Re * Pr for this setup)
```

**Physical interpretation:**
- Ma ~ 10: Moderate Marangoni effect
- Re ~ 10: Laminar flow, viscous forces significant
- Pr = 1: Momentum and thermal diffusivity balanced

---

## Appendix C: Code Snippet Templates

### C.1 Main Test Structure

```cpp
// File: tests/validation/test_marangoni_microchannel.cu

#include "physics/thermal_lbm.h"
#include "physics/fluid_lbm.h"
#include "io/vtk_writer.h"
#include <iostream>

// Physical parameters (Case 1)
const int L_ref = 256;
const int NX = 2 * L_ref;
const int NY = L_ref;
const int NZ = 1;

const float rho = 1.0f;
const float mu = 0.2f;
const float nu = mu / rho;
const float sigma_ref = 0.025f;
const float sigma_t = -5.0e-4f;
const float kappa = 0.2f;
const float T_ref = 10.0f;
const float T_h = 20.0f;
const float T_0 = 4.0f;

int main() {
    // 1. Initialize phase field
    float* phase;
    cudaMalloc(&phase, NX*NY*NZ*sizeof(float));
    initializePhaseField<<<...>>>(phase, NX, NY, NZ);

    // 2. Set up thermal solver
    float alpha = kappa / (rho * 1.0f);  // cp = 1
    ThermalLBM thermal(NX, NY, NZ, alpha);
    thermal.initialize(T_ref);

    // 3. Set up fluid solver
    FluidLBM fluid(NX, NY, NZ, nu, rho,
                   BoundaryType::PERIODIC,
                   BoundaryType::WALL,
                   BoundaryType::PERIODIC);
    fluid.initialize();

    // 4. Time loop
    const int NUM_STEPS = 10000;
    for (int step = 0; step < NUM_STEPS; step++) {
        // Thermal step
        thermal.collisionBGK(fluid.getVelocityX(),
                            fluid.getVelocityY(),
                            fluid.getVelocityZ());
        thermal.streaming();
        applyThermalBCs(thermal, T_ref, T_h, T_0, L_ref);

        // Compute Marangoni force
        computeMarangoniForce(thermal.getTemperature(), phase,
                             fx, fy, fz, sigma_t, NX, NY, NZ);

        // Fluid step
        fluid.setBodyForce(fx, fy, fz);
        fluid.collisionBGK();
        fluid.streaming();
        fluid.applyBoundaryConditions();

        // Monitor convergence
        if (step % 100 == 0) {
            float max_u = computeMaxVelocity(fluid.getVelocityX(),
                                            fluid.getVelocityY(),
                                            fluid.getVelocityZ(),
                                            NX*NY*NZ);
            printf("Step %d: max_u = %e\n", step, max_u);
        }
    }

    // 5. Validate
    float L2_T = computeL2Error_Temperature(thermal.getTemperature(),
                                            analytical_T, NX*NY*NZ);
    float L2_U = computeL2Error_Velocity(fluid.getVelocityX(),
                                         fluid.getVelocityY(),
                                         analytical_ux, analytical_uy,
                                         NX*NY*NZ);

    printf("Validation results:\n");
    printf("  L2_T = %.6f (target < 0.01)\n", L2_T);
    printf("  L2_U = %.6f (target < 0.05)\n", L2_U);

    bool pass = (L2_T < 0.01f) && (L2_U < 0.05f);
    printf("Test %s\n", pass ? "PASSED" : "FAILED");

    return pass ? 0 : 1;
}
```

### C.2 Marangoni Force Kernel

```cpp
// File: src/physics/marangoni/marangoni_force.cu

__global__ void computeMarangoniForce(
    const float* T,          // Temperature field
    const float* phase,      // Phase field (0=gas, 1=liquid)
    float* fx, float* fy, float* fz,  // Output force
    float sigma_t,           // dσ/dT
    float dx,                // Grid spacing
    int nx, int ny, int nz)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i >= nx || j >= ny || k >= nz) return;

    int idx = i + j*nx + k*nx*ny;

    // Only compute force at interface (0.4 < phase < 0.6)
    float phi = phase[idx];
    if (phi < 0.4f || phi > 0.6f) {
        fx[idx] = fy[idx] = fz[idx] = 0.0f;
        return;
    }

    // Compute temperature gradient (central difference)
    float dTdx = 0.0f, dTdy = 0.0f, dTdz = 0.0f;

    if (i > 0 && i < nx-1) {
        int idx_xp = (i+1) + j*nx + k*nx*ny;
        int idx_xm = (i-1) + j*nx + k*nx*ny;
        dTdx = (T[idx_xp] - T[idx_xm]) / (2.0f * dx);
    }

    if (j > 0 && j < ny-1) {
        int idx_yp = i + (j+1)*nx + k*nx*ny;
        int idx_ym = i + (j-1)*nx + k*nx*ny;
        dTdy = (T[idx_yp] - T[idx_ym]) / (2.0f * dx);
    }

    if (k > 0 && k < nz-1) {
        int idx_zp = i + j*nx + (k+1)*nx*ny;
        int idx_zm = i + j*nx + (k-1)*nx*ny;
        dTdz = (T[idx_zp] - T[idx_zm]) / (2.0f * dx);
    }

    // For horizontal interface, normal is (0, 1, 0)
    // Tangential gradient: ∇_s T = ∇T - (∇T·n)n = (dT/dx, 0, dT/dz)
    fx[idx] = sigma_t * dTdx;
    fy[idx] = 0.0f;  // No normal component
    fz[idx] = sigma_t * dTdz;  // Zero for 2D case (nz=1)
}
```

---

**End of Design Document**

**Total Estimated Implementation Time:** 3-4 days
**Complexity Level:** Medium
**Dependencies:** ThermalLBM, FluidLBM (existing), Marangoni force (new)
**Risk Level:** Low-Medium (well-understood physics, reference implementation exists)
