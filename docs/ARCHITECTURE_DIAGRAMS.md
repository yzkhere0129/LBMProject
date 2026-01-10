# LBM-CUDA Platform Architecture Diagrams

**Date:** 2026-01-06

---

## 1. Module Hierarchy

```
┌─────────────────────────────────────────────────────────────────┐
│                     MultiphysicsSolver                          │
│                  (Orchestration & Coupling)                     │
└────────────────────┬────────────────────────────────────────────┘
                     │
         ┌───────────┼───────────┬────────────┬──────────────┐
         │           │           │            │              │
         ▼           ▼           ▼            ▼              ▼
    ┌────────┐  ┌────────┐  ┌────────┐  ┌──────────┐  ┌──────────┐
    │Thermal │  │ Fluid  │  │  VOF   │  │  Laser   │  │  Force   │
    │  LBM   │  │  LBM   │  │ Solver │  │  Source  │  │Accumulat.│
    │ (D3Q7) │  │(D3Q19) │  │        │  │          │  │          │
    └────┬───┘  └────┬───┘  └───┬────┘  └────┬─────┘  └─────┬────┘
         │           │          │            │              │
         ▼           ▼          ▼            ▼              ▼
    ┌────────┐  ┌────────┐  ┌────────┐  ┌─────────────────────────┐
    │ Phase  │  │Boundary│  │Interface│  │   Boundary Conditions   │
    │ Change │  │ Nodes  │  │Tracking │  │  ┌─────────────────────┐│
    │        │  │(no-slip│  │(normal, │  │  │ SurfaceTension      ││
    │        │  │ walls) │  │curvature│  │  │ MarangoniEffect     ││
    └────────┘  └────────┘  └─────────┘  │  │ RecoilPressure      ││
                                          │  └─────────────────────┘│
                                          └─────────────────────────┘
```

---

## 2. Data Ownership

```
┌───────────────────────────────────────────────────────────────────┐
│                         Device Memory                             │
└───────────────────────────────────────────────────────────────────┘

ThermalLBM owns:                    FluidLBM owns:
┌─────────────────┐                ┌──────────────────┐
│ d_g_src         │ (D3Q7)         │ d_f_src          │ (D3Q19)
│ d_g_dst         │                │ d_f_dst          │
│ d_temperature   │ ◄─────┐        │ d_rho            │
└─────────────────┘        │        │ d_ux, d_uy, d_uz │ ◄───┐
                           │        │ d_pressure       │     │
PhaseChangeSolver owns:    │        └──────────────────┘     │
┌─────────────────┐        │                                 │
│ d_liquid_frac   │ ◄──────┤        VOFSolver owns:          │
│ d_liquid_prev   │        │        ┌──────────────────┐     │
└─────────────────┘        │        │ d_fill_level     │     │
                           │        │ d_cell_flags     │     │
ForceAccumulator owns:     │        │ d_interface_norm │     │
┌─────────────────┐        │        │ d_curvature      │     │
│ d_force_x       │ ◄──────┼────────│ d_fill_level_tmp │     │
│ d_force_y       │        │        └──────────────────┘     │
│ d_force_z       │        │                                 │
└─────────────────┘        │                                 │
                           │                                 │
        READ-ONLY ACCESS via const getters                   │
        ───────────────────────────────────────────────────  │
        getTemperature() ───────────────────────────────────┘
        getLiquidFraction() ─────────────────────────────────┘
        getVelocityX/Y/Z() ──────────────────────────────────┘
        getFillLevel() ──────────────────────────────────────┘
```

---

## 3. Coupling Data Flow (One Timestep)

```
STEP 1: LASER → THERMAL
┌──────────┐    computeVolumetricHeatSource()    ┌──────────┐
│  Laser   │  ──────────────────────────────────> │ Thermal  │
│  Source  │         Q [W/m³]                     │   LBM    │
└──────────┘                                      └────┬─────┘
                                                       │
                                                       │ addHeatSource(Q, dt)
                                                       ▼
STEP 2: THERMAL → PHASE CHANGE                   ┌────────────┐
┌──────────┐    updateLiquidFraction()           │ Temperature│
│ Thermal  │  ───────────────────────────────>   │   Field    │
│   LBM    │         T [K]                        └─────┬──────┘
└──────────┘                                            │
      │                                                 │
      │ T [K]                                           │
      ▼                                                 │
┌───────────────┐                                       │
│ PhaseChange   │                                       │
│    Solver     │                                       │
└───────┬───────┘                                       │
        │ liquid_fraction [0-1]                         │
        ▼                                               │
                                                        │
STEP 3: THERMAL → EVAPORATION → VOF                    │
┌──────────┐  computeEvaporationMassFlux()             │
│ Thermal  │  ────────────────────────>  ┌──────────┐  │
│   LBM    │     J_evap [kg/(m²·s)]      │   VOF    │  │
└──────────┘                              │  Solver  │  │
                                          └─────┬────┘  │
                                                │       │
                                     applyEvaporationMassLoss()
                                                │       │
                                                ▼       │
STEP 4: VOF → INTERFACE PROPERTIES              │       │
┌──────────┐    reconstructInterface()          │       │
│   VOF    │  ─────────────────────────>  ┌────────────┴────┐
│  Solver  │                              │ interface_normal│
└──────────┘                              │   curvature     │
                                          └──────┬──────────┘
                                                 │
                                                 │
STEP 5: FORCES (THERMAL + VOF → FLUID)           │
┌──────────┐                                     │
│ Thermal  │  ──┐                                │
│   LBM    │    │ T [K]                          │
└──────────┘    │                                │
                │                                │
┌──────────┐    │                                │
│   VOF    │  ──┼────────────────────────────────┘
│  Solver  │    │ fill_level, normal, curvature
└──────────┘    │
                │
                ▼
        ┌───────────────────┐
        │ ForceAccumulator  │
        │    reset()        │
        │      ↓            │
        │ Marangoni         │ ◄─── computeForce(T, fill, normal)
        │ SurfaceTension    │ ◄─── computeForce(curvature, fill)
        │ RecoilPressure    │ ◄─── computeForce(P_sat, fill)
        │ Buoyancy          │ ◄─── computeForce(T, T_ref, beta)
        │      ↓            │
        │ convertToLattice()│      F_lattice = F_phys * (dt²/dx)
        │      ↓            │
        │ limitForcesByCFL()│      Prevent divergence
        └────────┬──────────┘
                 │ fx, fy, fz [lattice units]
                 ▼
        ┌───────────────────┐
        │    FluidLBM       │
        │  collisionBGK()   │      Guo forcing scheme
        │  streaming()      │
        │  computeMacro()   │
        └────────┬──────────┘
                 │ ux, uy, uz [lattice units]
                 ▼

STEP 6: FLUID → VOF (ADVECTION)
┌──────────┐  convertVelocityToPhysical()   ┌──────────┐
│  Fluid   │  ──────────────────────────>   │   VOF    │
│   LBM    │    v_phys = v_latt * (dx/dt)   │  Solver  │
└──────────┘                                 └─────┬────┘
                                                   │
                                    advectFillLevel(v_phys, dt)
                                                   │
                                                   ▼
                                          ┌────────────────┐
                                          │  Upwind Advec. │
                                          │       ↓        │
                                          │  Compression   │
                                          │       ↓        │
                                          │ Reconstruction │
                                          └────────────────┘
```

---

## 4. Unit Conversion Points

```
┌───────────────────────────────────────────────────────────────────┐
│                    PHYSICAL UNITS [SI]                            │
│                  (meters, seconds, kilograms)                     │
└───────────────────────────────┬───────────────────────────────────┘
                                │
                    ┌───────────┴───────────┐
                    │   Boundary Layer      │
                    │  (Unit Conversion)    │
                    └───────────┬───────────┘
                                │
┌───────────────────────────────┴───────────────────────────────────┐
│                    LATTICE UNITS [dimensionless]                  │
│                                                                   │
│  ThermalLBM:                                                      │
│    alpha_lattice = alpha_phys * dt / dx²                         │
│    tau = alpha_lattice / cs² + 0.5                               │
│    omega = 1 / tau                                               │
│                                                                   │
│  FluidLBM:                                                        │
│    nu_lattice = nu_phys * dt / dx²                               │
│    tau = nu_lattice / cs² + 0.5                                  │
│    omega = 1 / tau                                               │
│                                                                   │
│  Force Conversion:                                                │
│    F_lattice = F_phys * (dt² / dx)                               │
│                                                                   │
│  Velocity Conversion:                                             │
│    v_phys = v_lattice * (dx / dt)                                │
└───────────────────────────────────────────────────────────────────┘
                                │
                    ┌───────────┴───────────┐
                    │   Boundary Layer      │
                    │  (Unit Conversion)    │
                    └───────────┬───────────┘
                                │
┌───────────────────────────────┴───────────────────────────────────┐
│                    PHYSICAL UNITS [SI]                            │
│                  (Output for visualization)                       │
└───────────────────────────────────────────────────────────────────┘
```

---

## 5. Operator Splitting (LBM Timestep)

```
┌─────────────────────────────────────────────────────────────────┐
│                     THERMAL LBM (D3Q7)                          │
└─────────────────────────────────────────────────────────────────┘

   STAGE 1: Collision
   ┌────────────────────────────────────────┐
   │  for each cell:                        │
   │    T = Σ g_i                          │
   │    g_eq = equilibrium(T, u)           │
   │    g_new = g - omega * (g - g_eq)     │
   └────────────────┬───────────────────────┘
                    │ g_src → g_dst (in-place)
                    ▼
   STAGE 2: Streaming
   ┌────────────────────────────────────────┐
   │  for each cell:                        │
   │    for each direction i:               │
   │      target = current + c_i            │
   │      g_dst[target][i] = g_src[cell][i]│
   │  (bounce-back at boundaries)           │
   └────────────────┬───────────────────────┘
                    │ swap(g_src, g_dst)
                    ▼
   STAGE 3: Heat Source
   ┌────────────────────────────────────────┐
   │  dT = Q * dt / (rho * cp)             │
   │  for each direction i:                 │
   │    g[i] += w[i] * dT                  │
   └────────────────┬───────────────────────┘
                    │
                    ▼
   STAGE 4: Boundary Conditions
   ┌────────────────────────────────────────┐
   │  Top:    radiation BC                  │
   │  Bottom: substrate cooling             │
   │  Sides:  adiabatic (already in stream) │
   └────────────────┬───────────────────────┘
                    │
                    ▼
   STAGE 5: Compute Temperature
   ┌────────────────────────────────────────┐
   │  T = Σ g_i                            │
   │  T = clamp(T, T_min, T_max)           │
   └────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                     FLUID LBM (D3Q19)                           │
└─────────────────────────────────────────────────────────────────┘

   STAGE 1: Collision (with Guo forcing)
   ┌────────────────────────────────────────┐
   │  rho = Σ f_i                          │
   │  u* = Σ c_i * f_i / rho               │
   │  u = u* + 0.5 * F / rho               │
   │  f_eq = equilibrium(rho, u)           │
   │  F_term = (1 - omega/2) * force_term  │
   │  f_new = f - omega*(f - f_eq) + F_term│
   └────────────────┬───────────────────────┘
                    │ f_src → f_dst
                    ▼
   STAGE 2: Streaming
   ┌────────────────────────────────────────┐
   │  for each cell:                        │
   │    for each direction i:               │
   │      target = current + e_i            │
   │      f_dst[target][i] = f_src[cell][i]│
   │  (periodic or bounce-back)             │
   └────────────────┬───────────────────────┘
                    │ swap(f_src, f_dst)
                    ▼
   STAGE 3: Boundary Conditions
   ┌────────────────────────────────────────┐
   │  Walls: bounce-back at wall nodes      │
   │  Periodic: handled in streaming        │
   └────────────────┬───────────────────────┘
                    │
                    ▼
   STAGE 4: Compute Macroscopic
   ┌────────────────────────────────────────┐
   │  rho = Σ f_i                          │
   │  u = Σ c_i * f_i / rho                │
   │  p = cs² * (rho - rho0)               │
   │  (enforce u=0 at walls)                │
   └────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                     VOF SOLVER                                  │
└─────────────────────────────────────────────────────────────────┘

   STAGE 1: Upwind Advection
   ┌────────────────────────────────────────┐
   │  for each cell:                        │
   │    df/dt = -u·∇f (upwind scheme)      │
   │    f_new = f + dt * df/dt             │
   └────────────────┬───────────────────────┘
                    │ fill_level → fill_level_tmp
                    ▼
   STAGE 2: Interface Compression (Olsson-Kreiss)
   ┌────────────────────────────────────────┐
   │  epsilon = C * |u|_max * dx            │
   │  n = -∇f / |∇f|                        │
   │  div_flux = ∇·(epsilon·f·(1-f)·n)     │
   │  f_new = f + dt * div_flux             │
   └────────────────┬───────────────────────┘
                    │ fill_level_tmp → fill_level
                    ▼
   STAGE 3: Interface Reconstruction
   ┌────────────────────────────────────────┐
   │  n = -∇f / |∇f|                        │
   │  kappa = ∇·n                           │
   │  classify cells (gas/liquid/interface) │
   └────────────────────────────────────────┘
```

---

## 6. Stability Mechanisms

```
┌─────────────────────────────────────────────────────────────────┐
│                  TIMESTEP CONSTRAINTS                           │
└─────────────────────────────────────────────────────────────────┘

    ThermalLBM (D3Q7):
    ┌────────────────────────────────────────┐
    │  omega < 1.95 (stability limit)        │
    │  tau > 0.505                           │
    │  CFL_thermal = (alpha*dt/dx²)/cs² < 0.5│
    │                                        │
    │  VALIDATION:                           │
    │  - if omega >= 1.95: THROW EXCEPTION   │
    │  - if omega >= 1.85: WARNING           │
    │  - NO SILENT CLAMPING                  │
    └────────────────────────────────────────┘

    FluidLBM (D3Q19):
    ┌────────────────────────────────────────┐
    │  omega < 2.0 (von Neumann)             │
    │  tau > 0.51 (practical limit)          │
    │  CFL_advection = v_max*dt/dx < 0.5     │
    │                                        │
    │  VALIDATION:                           │
    │  - if tau < 0.51: CLAMP + WARNING      │
    └────────────────────────────────────────┘

    VOFSolver:
    ┌────────────────────────────────────────┐
    │  CFL_advection = v_max*dt/dx < 0.5     │
    │  CFL_compression = epsilon*dt/dx < 0.5 │
    │                                        │
    │  VALIDATION:                           │
    │  - Check CFL before advection          │
    │  - Warn if v_max*dt/dx > 0.5          │
    └────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                  FORCE CFL LIMITER                              │
└─────────────────────────────────────────────────────────────────┘

    Tier 1: Hard CFL Limit
    ┌────────────────────────────────────────┐
    │  v_new = v_current + F                 │
    │  CFL = v_new * dt / dx                 │
    │  if CFL > CFL_max:                     │
    │    scale = CFL_max / CFL               │
    │    F *= scale                          │
    └────────────────────────────────────────┘
                    │
                    ▼
    Tier 2: Gradual Scaling
    ┌────────────────────────────────────────┐
    │  v_ramp = ramp_factor * v_target       │
    │  if v_new > v_target:                  │
    │    Hard limit to v_target              │
    │  elif v_new > v_ramp:                  │
    │    Smooth ramp-down                    │
    │  else:                                 │
    │    No scaling (full force)             │
    └────────────────────────────────────────┘
                    │
                    ▼
    Tier 3: Adaptive Region-Based (Keyhole Mode)
    ┌────────────────────────────────────────┐
    │  Interface (0.01<f<0.99):              │
    │    v_target = 0.5 (high for recoil)    │
    │    if force is z-dominant:             │
    │      v_target *= 1.5 (recoil boost)    │
    │                                        │
    │  Bulk liquid (f>0.99, liq_frac>0.5):  │
    │    v_target = 0.3 (Marangoni)          │
    │                                        │
    │  Solid (f>0.99, liq_frac<0.01):       │
    │    F = 0 (no flow in solid)            │
    │                                        │
    │  Gas (f<0.01):                         │
    │    v_target = 0.1 (minimal)            │
    └────────────────────────────────────────┘
```

---

## 7. Memory Layout (Structure of Arrays)

```
┌─────────────────────────────────────────────────────────────────┐
│              THERMAL LBM (D3Q7) - SoA Layout                    │
└─────────────────────────────────────────────────────────────────┘

d_g_src:  [num_cells * 7]
┌─────────────────────────────────────────────────────────────────┐
│ g[0][0], g[0][1], ..., g[0][6], g[1][0], ..., g[N-1][6]       │
│ ├───────────────────────┘                                       │
│ │ Cell 0: 7 distributions (D3Q7)                               │
│ └─────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
         ↓ Collision (in-place)
         ↓ Streaming (to d_g_dst)
         ↓ swap(d_g_src, d_g_dst)

d_temperature: [num_cells * 1]
┌─────────────────────────────────────────────────────────────────┐
│ T[0], T[1], T[2], ..., T[N-1]                                  │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│              FLUID LBM (D3Q19) - SoA Layout                     │
└─────────────────────────────────────────────────────────────────┘

d_f_src:  [num_cells * 19]
┌─────────────────────────────────────────────────────────────────┐
│ f[0][0], ..., f[0][18], f[1][0], ..., f[N-1][18]              │
│ ├─────────────────────────┘                                     │
│ │ Cell 0: 19 distributions (D3Q19)                             │
└─────────────────────────────────────────────────────────────────┘

d_ux, d_uy, d_uz: [num_cells * 1] each
┌─────────────────────────────────────────────────────────────────┐
│ ux[0], ux[1], ..., ux[N-1]                                     │
│ uy[0], uy[1], ..., uy[N-1]                                     │
│ uz[0], uz[1], ..., uz[N-1]                                     │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│              VOF SOLVER - Scalar Fields                         │
└─────────────────────────────────────────────────────────────────┘

d_fill_level: [num_cells * 1]
┌─────────────────────────────────────────────────────────────────┐
│ f[0], f[1], f[2], ..., f[N-1]     (0=gas, 1=liquid)           │
└─────────────────────────────────────────────────────────────────┘

d_interface_normal: [num_cells * 3]
┌─────────────────────────────────────────────────────────────────┐
│ nx[0], ny[0], nz[0], nx[1], ny[1], nz[1], ..., nz[N-1]        │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│         INDEXING: i + nx * (j + ny * k)                         │
└─────────────────────────────────────────────────────────────────┘

Example: 3D domain (nx=32, ny=32, nz=32)
  Cell at (i=5, j=10, k=15):
    idx = 5 + 32*(10 + 32*15) = 5 + 32*490 = 15685

  For D3Q19 distribution at direction q=7:
    f[idx + q*num_cells] = f[15685 + 7*32768]
```

---

## 8. Test Hierarchy

```
┌─────────────────────────────────────────────────────────────────┐
│                         TEST SUITE                              │
└─────────────────────────────────────────────────────────────────┘

Unit Tests (17+)
├── VOF
│   ├── test_vof_advection_rotation.cu           ✅ PASS
│   ├── test_vof_mass_conservation.cu            ✅ PASS
│   ├── test_vof_interface_compression.cu        ✅ PASS
│   ├── test_vof_curvature_sphere.cu             ✅ PASS
│   └── test_vof_curvature_cylinder.cu           ✅ PASS
│
├── Thermal
│   ├── test_gaussian_diffusion.cu               ✅ PASS
│   ├── test_thermal_walberla_match.cu           ✅ PASS (2.01%)
│   ├── test_thermal_timestep_convergence.cu     ✅ PASS
│   └── test_omega_clamping_fix.cu               ✅ PASS
│
└── Materials
    ├── test_materials.cu                        ✅ PASS
    └── test_material_interpolation.cu           ✅ PASS

Integration Tests (10+)
├── test_thermal_vof_coupling.cu                 ✅ PASS
├── test_fluid_vof_coupling.cu                   ✅ PASS
├── test_multiphysics_orchestration.cu           ✅ PASS
└── test_force_accumulator.cu                    ✅ PASS

Validation Tests (12+)
├── Convergence
│   ├── test_grid_convergence.cu                 ✅ Order 2.00
│   ├── test_timestep_convergence.cu             ✅ Order 1.0
│   └── test_energy_conservation.cu              ✅ 1.19% error
│
├── Stability
│   ├── test_cfl_stability.cu                    ✅ PASS
│   ├── test_force_divergence.cu                 ✅ PASS
│   └── test_divergence_free.cu                  ✅ PASS
│
└── Physics
    ├── test_stefan_problem.cu                   ✅ PASS
    ├── test_marangoni_velocity.cu               ✅ PASS
    ├── test_evaporation_hertz_knudsen.cu        ✅ PASS
    └── test_laser_melting_senior.cu             ✅ PASS

Benchmark Tests
├── test_3d_heat_diffusion_senior.cu            ✅ PASS
└── test_keyhole_formation_senior.cu            ✅ PASS

┌─────────────────────────────────────────────────────────────────┐
│              VALIDATION METRICS                                 │
└─────────────────────────────────────────────────────────────────┘

  waLBerla Comparison:     2.01% error  (target: <5%)    ✅
  Grid Convergence:        Order 2.00   (target: ~2.0)   ✅
  Timestep Convergence:    Order 1.0    (expected: 1.0)  ✅
  Energy Conservation:     1.19% error  (excellent)      ✅
  VOF Mass Conservation:   3.3% error   (with compress.) ✅
```

---

## Document Information

**Version:** 1.0
**Last Updated:** 2026-01-06
**Related Documents:**
- Full review: `ARCHITECTURAL_REVIEW_POST_BUGFIXES.md`
- Summary: `ARCHITECTURE_REVIEW_SUMMARY.md`
- Bug fixes: `OMEGA_CLAMPING_BUG_FIX.md`, `VOF_INTERFACE_COMPRESSION.md`
