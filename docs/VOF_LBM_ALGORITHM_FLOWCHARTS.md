# VOF+LBM Algorithm Flowcharts and Data Flow Diagrams

**Project:** LBMProject - CUDA-based Lattice Boltzmann Method Framework
**Document Type:** Visual Algorithm Analysis
**Date:** 2025-12-17

---

## Table of Contents

1. [Main Time Stepping Loop](#1-main-time-stepping-loop)
2. [VOF Advection Detailed Flow](#2-vof-advection-detailed-flow)
3. [Interface Reconstruction Pipeline](#3-interface-reconstruction-pipeline)
4. [Force Accumulation and Application](#4-force-accumulation-and-application)
5. [LBM Collision-Streaming Cycle](#5-lbm-collision-streaming-cycle)
6. [Data Dependency Graph](#6-data-dependency-graph)
7. [Kernel Execution Timeline](#7-kernel-execution-timeline)

---

## 1. Main Time Stepping Loop

### Overall Multiphysics Algorithm

```
┌──────────────────────────────────────────────────────────────────────┐
│                    MULTIPHYSICS TIME STEP LOOP                       │
│                    (MultiphysicsSolver::step())                      │
└──────────────────────────────────────────────────────────────────────┘
                                  │
                                  ↓
        ┌─────────────────────────────────────────────────┐
        │  PHASE 1: THERMAL SOLUTION                      │
        │  ────────────────────────                       │
        │                                                  │
        │  1.1 Compute Laser Heat Source                  │
        │      Q_laser(x,y,z) = η·I(x,y)·α·exp(-α·z)     │
        │      [computeLaserHeatSourceKernel]             │
        │                                                  │
        │  1.2 Thermal LBM Step (D3Q7)                    │
        │      • Collision (BGK): f_i ← f_i + ω(f_eq-f_i)│
        │      • Streaming: f_i(x,t+dt) ← f_i(x-c_i,t)   │
        │      • Boundaries: Dirichlet, Neumann           │
        │      [ThermalLBM::step()]                       │
        │                                                  │
        │  Output: T(x,y,z,t) [temperature field]        │
        └─────────────────┬───────────────────────────────┘
                          │
                          ↓
        ┌─────────────────────────────────────────────────┐
        │  PHASE 2: PHASE CHANGE                          │
        │  ──────────────────                             │
        │                                                  │
        │  2.1 Compute Liquid Fraction                    │
        │      fl = { 0           if T < T_solidus        │
        │           { (T-Ts)/(Tl-Ts) if Ts ≤ T ≤ Tl      │
        │           { 1           if T > T_liquidus       │
        │                                                  │
        │  2.2 Compute Phase Change Rate                  │
        │      dfl/dt = ∂fl/∂T · ∂T/∂t                   │
        │                                                  │
        │  2.3 Compute Evaporation Flux                   │
        │      J_evap = { 0.01·M·P_sat/√(2πMRT)          │
        │                 if T > T_boil                   │
        │                                                  │
        │  [PhaseChange::computePhaseChange()]            │
        │                                                  │
        │  Output: fl, dfl/dt, J_evap                     │
        └─────────────────┬───────────────────────────────┘
                          │
                          ↓
        ┌─────────────────────────────────────────────────┐
        │  PHASE 3: VOF ADVECTION & INTERFACE TRACKING    │
        │  ─────────────────────────────────────────      │
        │                                                  │
        │  3.1 Determine VOF Subcycling                   │
        │      n_substeps = max(1, ⌈2·v_max·dt/dx⌉)      │
        │      dt_vof = dt / n_substeps                   │
        │                                                  │
        │  3.2 Convert Velocity to Physical Units         │
        │      u_phys = u_lattice × (dx/dt)               │
        │      [convertVelocityToPhysicalUnitsKernel]     │
        │                                                  │
        │  FOR sub = 1 to n_substeps:                     │
        │    │                                             │
        │    3.3 Advect Fill Level (upwind)               │
        │        ∂f/∂t + ∇·(f·u) = 0                      │
        │        [advectFillLevelUpwindKernel]            │
        │        • CFL check: v·dt_vof/dx < 0.5           │
        │        • Periodic boundaries                     │
        │        • Clamp to [0,1]                         │
        │    │                                             │
        │    3.4 Reconstruct Interface Normals            │
        │        n = -∇f / |∇f|                           │
        │        [reconstructInterfaceKernel]             │
        │    │                                             │
        │    3.5 Compute Curvature                        │
        │        κ = ∇·n                                  │
        │        [computeCurvatureKernel]                 │
        │    │                                             │
        │    3.6 Update Cell Flags                        │
        │        GAS (f<0.01), INTERFACE, LIQUID (f>0.99) │
        │        [convertCellsKernel]                     │
        │    │                                             │
        │  END FOR                                         │
        │                                                  │
        │  3.7 Apply Evaporation Mass Loss                │
        │      df = -J_evap · dt / (ρ·dx)                │
        │      [applyEvaporationMassLossKernel]           │
        │      • Limited to 2% per step                   │
        │                                                  │
        │  3.8 Apply Solidification Shrinkage             │
        │      df = β · dfl/dt · dt                       │
        │      [applySolidificationShrinkageKernel]       │
        │      • Only at interface (0.01 < f < 0.99)      │
        │      • Only during solidification (dfl/dt < 0)  │
        │                                                  │
        │  Output: f(x,y,z,t), n(x,y,z,t), κ(x,y,z,t)   │
        └─────────────────┬───────────────────────────────┘
                          │
                          ↓
        ┌─────────────────────────────────────────────────┐
        │  PHASE 4: FORCE ACCUMULATION                    │
        │  ────────────────────────                       │
        │                                                  │
        │  4.1 Zero Force Arrays                          │
        │      F_x = F_y = F_z = 0                        │
        │      [zeroForceKernelLocal]                     │
        │                                                  │
        │  4.2 Add Surface Tension (CSF)                  │
        │      F_σ = σ·κ·∇f  [N/m³]                       │
        │      [computeCSFForceKernel]                    │
        │      • Only at interface (0.01 < f < 0.99)      │
        │                                                  │
        │  4.3 Add Marangoni Force                        │
        │      F_M = (dσ/dT)·∇_s T·|∇f|  [N/m³]          │
        │      [addMarangoniForceKernel]                  │
        │      • Hybrid interface detection                │
        │      • Gradient limiting: |∇T| < 5e8 K/m        │
        │      • Tangential projection: ∇_s T = ∇T-(n·∇T)n│
        │                                                  │
        │  4.4 Add Recoil Pressure                        │
        │      F_R = P_recoil·n·|∇f|  [N/m³]             │
        │      [addRecoilPressureKernel]                  │
        │      • P_recoil ∝ exp((T-T_boil)/ΔT)           │
        │                                                  │
        │  4.5 Convert to Lattice Units                   │
        │      F_lattice = F_phys × (dt²/dx)              │
        │      [convertForceToLatticeUnitsKernel]         │
        │                                                  │
        │  4.6 CFL-based Force Limiting                   │
        │      IF v_predicted > v_max THEN                │
        │         F ← F × (v_max / v_predicted)           │
        │      [limitForcesByCFL_AdaptiveKernel]          │
        │      • Region-based: interface, bulk, solid      │
        │      • v_target_interface = 0.5 (~10 m/s)       │
        │      • v_target_bulk = 0.3 (~6 m/s)             │
        │                                                  │
        │  Output: F_x, F_y, F_z [lattice units]          │
        └─────────────────┬───────────────────────────────┘
                          │
                          ↓
        ┌─────────────────────────────────────────────────┐
        │  PHASE 5: FLUID LBM STEP                        │
        │  ────────────────────────                       │
        │                                                  │
        │  5.1 Collision with Forcing (BGK + Guo)         │
        │      f_i^* = f_i + ω(f_i^eq - f_i)              │
        │      f_i^post = f_i^* + (1-ω/2)·S_i(F)          │
        │      [fluidBGKCollisionKernelWithForce]         │
        │      • Macroscopic: ρ, u from f_i               │
        │      • Equilibrium: f_eq(ρ,u)                   │
        │      • Force term: Guo scheme                    │
        │                                                  │
        │  5.2 Streaming                                   │
        │      f_i(x+c_i, t+dt) ← f_i^post(x,t)           │
        │      [streamD3Q19Kernel]                        │
        │      • Pull scheme: read from neighbors          │
        │                                                  │
        │  5.3 Boundary Conditions                         │
        │      • Periodic: wrap-around                     │
        │      • Wall: bounce-back or velocity BC          │
        │      [applyBoundaryConditions]                   │
        │                                                  │
        │  5.4 Compute Macroscopic Quantities             │
        │      ρ = Σ f_i                                  │
        │      u = (Σ c_i·f_i + F·dt/2) / ρ               │
        │      [computeMacroscopicKernel]                 │
        │                                                  │
        │  Output: ρ, u, v, w [lattice units]             │
        └─────────────────┬───────────────────────────────┘
                          │
                          ↓
        ┌─────────────────────────────────────────────────┐
        │  PHASE 6: DIAGNOSTICS & OUTPUT                  │
        │  ──────────────────────────                     │
        │                                                  │
        │  6.1 Compute Derived Quantities                 │
        │      • Pressure: p = c_s²(ρ - ρ_0)             │
        │      • Velocity magnitude: |u|                   │
        │      • Temperature gradients                     │
        │      • VOF mass: M = Σ f_i                      │
        │                                                  │
        │  6.2 Check Stability Criteria                   │
        │      • NaN/Inf check                             │
        │      • CFL violations                            │
        │      • Mass conservation                         │
        │                                                  │
        │  6.3 Write Output (if output step)              │
        │      • VTK files: T, u, v, w, f, κ              │
        │      • Time series: T_max, v_max, M_VOF         │
        │                                                  │
        │  6.4 Update Time                                 │
        │      t ← t + dt                                 │
        │      step ← step + 1                            │
        │                                                  │
        └──────────────────┬──────────────────────────────┘
                           │
                           ↓
                     ┌─────────────┐
                     │  NEXT STEP  │
                     └─────────────┘
```

---

## 2. VOF Advection Detailed Flow

### Upwind Advection Algorithm

```
┌────────────────────────────────────────────────────────────────────┐
│  advectFillLevelUpwindKernel                                       │
│  (First-order donor-cell scheme)                                   │
└────────────────────────────────────────────────────────────────────┘
                              │
                              ↓
        ┌──────────────────────────────────────────┐
        │  INPUT DATA (per cell i,j,k)             │
        │  ───────────────────────                 │
        │  • f[i,j,k]: current fill level          │
        │  • u[i,j,k]: x-velocity [m/s]            │
        │  • v[i,j,k]: y-velocity [m/s]            │
        │  • w[i,j,k]: z-velocity [m/s]            │
        │  • dt: time step [s]                     │
        │  • dx: lattice spacing [m]               │
        └──────────────┬───────────────────────────┘
                       │
                       ↓
        ┌──────────────────────────────────────────┐
        │  STEP 1: Determine Upwind Neighbors      │
        │  ───────────────────────────────         │
        │                                           │
        │  X-direction:                             │
        │    IF u ≥ 0 THEN                         │
        │       i_up = (i > 0) ? i-1 : nx-1        │  ← Periodic BC
        │    ELSE                                   │
        │       i_up = (i < nx-1) ? i+1 : 0        │
        │                                           │
        │  Y-direction:                             │
        │    IF v ≥ 0 THEN                         │
        │       j_up = (j > 0) ? j-1 : ny-1        │
        │    ELSE                                   │
        │       j_up = (j < ny-1) ? j+1 : 0        │
        │                                           │
        │  Z-direction:                             │
        │    IF w ≥ 0 THEN                         │
        │       k_up = (k > 0) ? k-1 : nz-1        │
        │    ELSE                                   │
        │       k_up = (k < nz-1) ? k+1 : 0        │
        │                                           │
        └──────────────┬───────────────────────────┘
                       │
                       ↓
        ┌──────────────────────────────────────────┐
        │  STEP 2: Compute Upwind Derivatives      │
        │  ───────────────────────────────         │
        │                                           │
        │  Advection equation: ∂f/∂t = -u·∂f/∂x    │
        │                                           │
        │  X-direction derivative:                  │
        │    IF u ≥ 0 THEN                         │
        │       dfdt_x = -u·(f[i,j,k]-f[i_up,j,k])/dx  │
        │    ELSE                                   │
        │       dfdt_x = -u·(f[i_up,j,k]-f[i,j,k])/dx  │
        │                                           │
        │  Y-direction derivative:                  │
        │    IF v ≥ 0 THEN                         │
        │       dfdt_y = -v·(f[i,j,k]-f[i,j_up,k])/dx  │
        │    ELSE                                   │
        │       dfdt_y = -v·(f[i,j_up,k]-f[i,j,k])/dx  │
        │                                           │
        │  Z-direction derivative:                  │
        │    IF w ≥ 0 THEN                         │
        │       dfdt_z = -w·(f[i,j,k]-f[i,j,k_up])/dx  │
        │    ELSE                                   │
        │       dfdt_z = -w·(f[i,j,k_up]-f[i,j,k])/dx  │
        │                                           │
        │  Total rate of change:                    │
        │    dfdt = dfdt_x + dfdt_y + dfdt_z        │
        │                                           │
        └──────────────┬───────────────────────────┘
                       │
                       ↓
        ┌──────────────────────────────────────────┐
        │  STEP 3: Time Integration (Forward Euler)│
        │  ─────────────────────────────────────   │
        │                                           │
        │  f_new = f[i,j,k] + dt × dfdt             │
        │                                           │
        └──────────────┬───────────────────────────┘
                       │
                       ↓
        ┌──────────────────────────────────────────┐
        │  STEP 4: Post-Processing                 │
        │  ────────────────────                    │
        │                                           │
        │  4.1 Flush Denormals:                    │
        │      IF f_new < 1e-6 THEN f_new = 0      │
        │                                           │
        │  4.2 Clamp to Physical Bounds:            │
        │      f_new = max(0, min(1, f_new))        │
        │                                           │
        └──────────────┬───────────────────────────┘
                       │
                       ↓
        ┌──────────────────────────────────────────┐
        │  OUTPUT                                   │
        │  ──────                                   │
        │  fill_level_new[i,j,k] = f_new            │
        │                                           │
        └───────────────────────────────────────────┘


STABILITY ANALYSIS:
───────────────────

CFL Condition: v_max · dt / dx < 0.5

Example:
  v_max = 10 m/s (Marangoni flow)
  dx = 2e-6 m
  dt = 1e-7 s

  CFL = 10 × 1e-7 / 2e-6 = 0.5  ← Borderline, requires subcycling

Subcycling Strategy:
  n_substeps = max(1, ceil(2 × v_max × dt / dx))
  dt_vof = dt / n_substeps

  → n_substeps = 2
  → dt_vof = 5e-8 s
  → CFL = 0.25  ✓ Stable
```

---

## 3. Interface Reconstruction Pipeline

### Normal and Curvature Computation

```
┌────────────────────────────────────────────────────────────────────┐
│  INTERFACE GEOMETRY PIPELINE                                       │
└────────────────────────────────────────────────────────────────────┘
                              │
                              ↓
        ┌──────────────────────────────────────────┐
        │  STEP 1: Compute Fill Level Gradient     │
        │  (reconstructInterfaceKernel)            │
        │  ────────────────────────────────         │
        │                                           │
        │  Using central differences:               │
        │                                           │
        │  ∂f/∂x = [f(i+1,j,k) - f(i-1,j,k)] / 2dx │
        │  ∂f/∂y = [f(i,j+1,k) - f(i,j-1,k)] / 2dx │
        │  ∂f/∂z = [f(i,j,k+1) - f(i,j,k-1)] / 2dx │
        │                                           │
        │  Gradient vector:                         │
        │    ∇f = (grad_x, grad_y, grad_z)         │
        │                                           │
        │  Magnitude:                               │
        │    |∇f| = √(grad_x² + grad_y² + grad_z²) │
        │                                           │
        └──────────────┬───────────────────────────┘
                       │
                       ↓
        ┌──────────────────────────────────────────┐
        │  STEP 2: Compute Interface Normal        │
        │  ────────────────────────────             │
        │                                           │
        │  Definition:                              │
        │    n = -∇f / |∇f|                         │
        │                                           │
        │  Sign convention:                         │
        │    • Normal points from LIQUID to GAS     │
        │    • Negative gradient (f decreases)      │
        │                                           │
        │  IF |∇f| > 1e-8 THEN                     │
        │    n.x = -grad_x / |∇f|                  │
        │    n.y = -grad_y / |∇f|                  │
        │    n.z = -grad_z / |∇f|                  │
        │  ELSE                                     │
        │    n = (0, 0, 0)  ← Bulk fluid/gas       │
        │                                           │
        │  Properties:                              │
        │    • Unit vector: |n| = 1                 │
        │    • Only non-zero near interface         │
        │                                           │
        └──────────────┬───────────────────────────┘
                       │
                       ↓
        ┌──────────────────────────────────────────┐
        │  STEP 3: Compute Curvature               │
        │  (computeCurvatureKernel)                │
        │  ─────────────────────                   │
        │                                           │
        │  Definition:                              │
        │    κ = ∇·n = ∂n_x/∂x + ∂n_y/∂y + ∂n_z/∂z│
        │                                           │
        │  ONLY for interface cells (0.01 < f < 0.99)│
        │                                           │
        │  Compute normal divergence:               │
        │    ∂n_x/∂x = [n_x(i+1) - n_x(i-1)] / 2dx │
        │    ∂n_y/∂y = [n_y(j+1) - n_y(j-1)] / 2dx │
        │    ∂n_z/∂z = [n_z(k+1) - n_z(k-1)] / 2dx │
        │                                           │
        │  Curvature:                               │
        │    κ = ∂n_x/∂x + ∂n_y/∂y + ∂n_z/∂z       │
        │                                           │
        │  Physical meaning:                        │
        │    • Sphere (R): κ = 2/R                  │
        │    • Cylinder (R): κ = 1/R                │
        │    • Plane: κ = 0                         │
        │    • Sign: κ > 0 → convex (liquid bulges) │
        │             κ < 0 → concave (liquid dip)  │
        │                                           │
        └──────────────┬───────────────────────────┘
                       │
                       ↓
        ┌──────────────────────────────────────────┐
        │  OUTPUT FIELDS                            │
        │  ─────────────                            │
        │                                           │
        │  • interface_normal[i,j,k]: float3        │
        │      Components: (n_x, n_y, n_z)          │
        │      Magnitude: |n| = 1 at interface      │
        │                 |n| = 0 in bulk           │
        │                                           │
        │  • curvature[i,j,k]: float                │
        │      Units: [1/m]                         │
        │      Non-zero only at interface           │
        │                                           │
        │  Used by:                                 │
        │    ✓ Surface tension (CSF)                │
        │    ✓ Marangoni force                      │
        │    ✓ Recoil pressure                      │
        │    ✓ Contact angle BC                     │
        │                                           │
        └───────────────────────────────────────────┘


ACCURACY VALIDATION:
────────────────────

Sphere Test (R = 20 cells):
  Analytical: κ_exact = 2/R = 0.100
  Numerical:  κ_computed = 0.098
  Error:      2% ✓

Cylinder Test (R = 16 cells):
  Analytical: κ_exact = 1/R = 0.0625
  Numerical:  κ_computed = 0.068
  Error:      8.8% ✓

Resolution Requirements:
  Good accuracy: 10+ cells per radius
  Acceptable:    5-10 cells per radius
  Poor:          < 5 cells per radius
```

---

## 4. Force Accumulation and Application

### CSF and Marangoni Force Computation

```
┌────────────────────────────────────────────────────────────────────┐
│  FORCE ACCUMULATION PIPELINE                                       │
│  (All forces computed in physical units [N/m³])                    │
└────────────────────────────────────────────────────────────────────┘
                              │
                              ↓
        ┌──────────────────────────────────────────┐
        │  FORCE 1: Surface Tension (CSF Model)    │
        │  ─────────────────────────────────       │
        │                                           │
        │  Physical Model:                          │
        │    Surface force: τ = σ·κ·n  [N/m²]     │
        │    Volumetric: F = σ·κ·∇f  [N/m³]       │
        │                                           │
        │  Algorithm:                               │
        │    IF 0.01 < f < 0.99 THEN               │
        │       κ = curvature[i,j,k]               │
        │       ∇f = central_difference(f)          │
        │       F_x = σ × κ × ∇f_x                 │
        │       F_y = σ × κ × ∇f_y                 │
        │       F_z = σ × κ × ∇f_z                 │
        │    ELSE                                   │
        │       F = 0  (bulk liquid/gas)            │
        │                                           │
        │  Material property:                       │
        │    σ = 1.5 N/m (Ti-6Al-4V liquid)        │
        │                                           │
        │  Output: F_σ(x,y,z)                      │
        └──────────────┬───────────────────────────┘
                       │
                       ↓
        ┌──────────────────────────────────────────┐
        │  FORCE 2: Marangoni Effect               │
        │  ──────────────────────                  │
        │                                           │
        │  Physical Model:                          │
        │    ∇σ = (dσ/dT)·∇T  (surface gradient)   │
        │    F_M = (dσ/dT)·∇_s T·|∇f|  [N/m³]     │
        │                                           │
        │  Algorithm:                               │
        │    2.1 Interface Detection (HYBRID):      │
        │        is_vof = (0.001 < f < 0.999)       │
        │        is_melt = (T > T_melt) AND         │
        │                 (k ≥ nz-10) AND           │
        │                 (T_above < T - 50K)       │
        │        IF NOT (is_vof OR is_melt) RETURN  │
        │                                           │
        │    2.2 Compute Temperature Gradient:      │
        │        ∇T = central_difference(T)         │
        │        |∇T| = √(∇T_x² + ∇T_y² + ∇T_z²)  │
        │                                           │
        │    2.3 Gradient Limiting:                 │
        │        IF |∇T| > 5e8 K/m THEN            │
        │           ∇T ← ∇T × (5e8 / |∇T|)         │
        │                                           │
        │    2.4 Tangential Projection:             │
        │        n = interface_normal[i,j,k]        │
        │        n·∇T = n_x·∇T_x + n_y·∇T_y + n_z·∇T_z │
        │        ∇_s T_x = ∇T_x - (n·∇T)·n_x       │
        │        ∇_s T_y = ∇T_y - (n·∇T)·n_y       │
        │        ∇_s T_z = ∇T_z - (n·∇T)·n_z       │
        │                                           │
        │    2.5 Compute Fill Gradient Magnitude:   │
        │        ∇f = central_difference(f)         │
        │        |∇f| = √(∇f_x² + ∇f_y² + ∇f_z²)  │
        │                                           │
        │    2.6 Marangoni Force:                   │
        │        coeff = (dσ/dT) × |∇f|            │
        │        F_M_x = coeff × ∇_s T_x            │
        │        F_M_y = coeff × ∇_s T_y            │
        │        F_M_z = coeff × ∇_s T_z            │
        │                                           │
        │  Material property:                       │
        │    dσ/dT = -3.5e-4 N/(m·K)               │
        │                                           │
        │  Output: F_M(x,y,z)                      │
        └──────────────┬───────────────────────────┘
                       │
                       ↓
        ┌──────────────────────────────────────────┐
        │  FORCE 3: Recoil Pressure                │
        │  ─────────────────────                   │
        │                                           │
        │  Physical Model:                          │
        │    P_recoil = P_sat × exp((T-T_boil)/ΔT) │
        │    F_R = P_recoil · n · |∇f|  [N/m³]    │
        │                                           │
        │  Algorithm:                               │
        │    IF T > T_boil AND (0.01 < f < 0.99)   │
        │       P_sat = 101325 Pa                   │
        │       ΔT = 100 K                          │
        │       P_r = P_sat·exp((T-T_boil)/ΔT)     │
        │       n = interface_normal[i,j,k]         │
        │       |∇f| = magnitude(central_diff(f))   │
        │       F_R_x = P_r × n_x × |∇f|           │
        │       F_R_y = P_r × n_y × |∇f|           │
        │       F_R_z = P_r × n_z × |∇f|           │
        │    ELSE                                   │
        │       F_R = 0                             │
        │                                           │
        │  Output: F_R(x,y,z)                      │
        └──────────────┬───────────────────────────┘
                       │
                       ↓
        ┌──────────────────────────────────────────┐
        │  STEP 4: Total Force Accumulation        │
        │  ─────────────────────────────            │
        │                                           │
        │  F_total_x = F_σ_x + F_M_x + F_R_x       │
        │  F_total_y = F_σ_y + F_M_y + F_R_y       │
        │  F_total_z = F_σ_z + F_M_z + F_R_z       │
        │                                           │
        └──────────────┬───────────────────────────┘
                       │
                       ↓
        ┌──────────────────────────────────────────┐
        │  STEP 5: Unit Conversion (Phys → Lattice)│
        │  ──────────────────────────────────────   │
        │                                           │
        │  Formula:                                 │
        │    F_lattice = F_phys × (dt² / dx)        │
        │                                           │
        │  Reasoning:                               │
        │    LBM: dv = F_lattice × dt_lattice = F   │
        │         (dt_lattice = 1 in LBM units)     │
        │    Physics: dv = F_phys/ρ × dt_phys       │
        │                                           │
        │  Conversion factor:                       │
        │    factor = dt² / dx                      │
        │                                           │
        │  Example:                                 │
        │    dt = 1e-7 s, dx = 2e-6 m              │
        │    factor = (1e-7)² / 2e-6 = 5e-9 m/s²   │
        │                                           │
        └──────────────┬───────────────────────────┘
                       │
                       ↓
        ┌──────────────────────────────────────────┐
        │  STEP 6: CFL-based Force Limiting        │
        │  ─────────────────────────────────        │
        │                                           │
        │  Problem: Strong forces → v explosion     │
        │                                           │
        │  Solution: Adaptive limiter               │
        │                                           │
        │  6.1 Classify Cell:                       │
        │      IF fill < 0.01: v_target = 0.1       │
        │      IF fill > 0.99 AND solid: v_t = 0    │
        │      IF fill > 0.99 AND liquid: v_t = 0.3 │
        │      IF 0.01 < fill < 0.99: v_t = 0.5     │
        │                                           │
        │  6.2 Predict Velocity:                    │
        │      v_new = v_current + F_lattice        │
        │      |v_new| = √(v_new_x² + v_new_y² + v_new_z²) │
        │                                           │
        │  6.3 Compute Scale:                       │
        │      IF |v_new| > v_target THEN           │
        │         scale = v_target / |v_new|        │
        │         F ← F × scale                     │
        │                                           │
        │  6.4 Special: Recoil Boost                │
        │      IF |F_z| / |F| > 0.8 THEN            │
        │         v_target ← v_target × 1.5         │
        │      (Allow stronger z-forces for recoil) │
        │                                           │
        └──────────────┬───────────────────────────┘
                       │
                       ↓
        ┌──────────────────────────────────────────┐
        │  OUTPUT: Limited Lattice Forces          │
        │  ───────────────────────────────          │
        │                                           │
        │  F_x[i,j,k] [lattice units]              │
        │  F_y[i,j,k] [lattice units]              │
        │  F_z[i,j,k] [lattice units]              │
        │                                           │
        │  Ready for LBM collision step             │
        │                                           │
        └───────────────────────────────────────────┘
```

---

## 5. LBM Collision-Streaming Cycle

### D3Q19 Lattice Boltzmann Algorithm

```
┌────────────────────────────────────────────────────────────────────┐
│  FLUID LBM STEP (D3Q19 with Guo Forcing)                           │
└────────────────────────────────────────────────────────────────────┘
                              │
                              ↓
        ┌──────────────────────────────────────────┐
        │  INPUT DATA                               │
        │  ──────────                               │
        │  • f_i[q,idx]: distribution functions     │
        │      (q = 0..18, idx = 0..num_cells-1)   │
        │  • F_x, F_y, F_z: body forces [lattice]   │
        │  • ω: relaxation frequency                │
        │  • τ: relaxation time                     │
        │                                           │
        └──────────────┬───────────────────────────┘
                       │
                       ↓
        ┌──────────────────────────────────────────┐
        │  STEP 1: Compute Macroscopic Quantities  │
        │  ───────────────────────────────────      │
        │                                           │
        │  Density:                                 │
        │    ρ = Σ_{i=0}^{18} f_i                  │
        │                                           │
        │  Momentum (raw):                          │
        │    ρu_raw = Σ_{i=0}^{18} c_i · f_i       │
        │                                           │
        │  Velocity (force-corrected):              │
        │    u = (ρu_raw + F·dt/2) / ρ             │
        │    v = (ρv_raw + F_y·dt/2) / ρ           │
        │    w = (ρw_raw + F_z·dt/2) / ρ           │
        │                                           │
        │  Velocity squared:                        │
        │    u² = u_x² + u_y² + u_z²               │
        │                                           │
        └──────────────┬───────────────────────────┘
                       │
                       ↓
        ┌──────────────────────────────────────────┐
        │  STEP 2: Compute Equilibrium Distribution│
        │  ─────────────────────────────────────   │
        │                                           │
        │  D3Q19 Equilibrium:                       │
        │                                           │
        │  f_i^eq = w_i · ρ · [                     │
        │      1 + (c_i·u) / c_s²                  │
        │        + (c_i·u)² / (2c_s⁴)              │
        │        - u² / (2c_s²)                    │
        │  ]                                        │
        │                                           │
        │  Where:                                   │
        │    c_s² = 1/3 (speed of sound squared)    │
        │    w_i = lattice weights:                 │
        │      w_0 = 1/3    (rest particle)         │
        │      w_1-6 = 1/18 (face directions)       │
        │      w_7-18 = 1/36 (edge directions)      │
        │                                           │
        │    c_i = lattice velocities:              │
        │      c_0 = (0,0,0)                        │
        │      c_1-6 = (±1,0,0), (0,±1,0), (0,0,±1) │
        │      c_7-18 = (±1,±1,0), (±1,0,±1), (0,±1,±1) │
        │                                           │
        └──────────────┬───────────────────────────┘
                       │
                       ↓
        ┌──────────────────────────────────────────┐
        │  STEP 3: Compute Force Term (Guo Scheme) │
        │  ────────────────────────────────────    │
        │                                           │
        │  Guo forcing term:                        │
        │                                           │
        │  S_i = w_i · (1 - ω/2) · [               │
        │      (c_i - u) · F / c_s²                │
        │    + (c_i · u)(c_i · F) / c_s⁴           │
        │  ]                                        │
        │                                           │
        │  Physical interpretation:                 │
        │    • First term: direct forcing           │
        │    • Second term: momentum coupling       │
        │    • (1-ω/2): force interpolation factor │
        │                                           │
        │  Advantages of Guo scheme:                │
        │    ✓ 2nd order accurate in time           │
        │    ✓ Correct momentum conservation        │
        │    ✓ Galilean invariance                  │
        │                                           │
        └──────────────┬───────────────────────────┘
                       │
                       ↓
        ┌──────────────────────────────────────────┐
        │  STEP 4: BGK Collision with Forcing      │
        │  ───────────────────────────────────      │
        │                                           │
        │  Two-stage collision:                     │
        │                                           │
        │  4.1 BGK relaxation:                      │
        │      f_i^* = f_i + ω(f_i^eq - f_i)       │
        │                                           │
        │  4.2 Force application:                   │
        │      f_i^post = f_i^* + S_i               │
        │                                           │
        │  Combined (single-step):                  │
        │      f_i^post = f_i + ω(f_i^eq - f_i) + S_i │
        │                                           │
        │  Where:                                   │
        │    ω = 1/τ = relaxation frequency         │
        │    τ = ν/c_s² + 0.5 (from viscosity)     │
        │                                           │
        │  Stability constraint:                    │
        │    τ > 0.5  ⟺  ω < 2                     │
        │                                           │
        └──────────────┬───────────────────────────┘
                       │
                       ↓
        ┌──────────────────────────────────────────┐
        │  STEP 5: Streaming (Pull Scheme)         │
        │  ────────────────────────────             │
        │                                           │
        │  Propagation step:                        │
        │    f_i(x, t+dt) ← f_i^post(x - c_i, t)   │
        │                                           │
        │  Implementation (pull):                   │
        │    FOR each direction i:                  │
        │      opposite_dir = opposite[i]           │
        │      neighbor_x = x - c_i                 │
        │      f_dst[i,x] = f_src[i, neighbor_x]    │
        │                                           │
        │  Memory layout optimization:              │
        │    • SoA: f[direction][cell]              │
        │    • Perfect coalescing when reading f_src│
        │    • Warp divergence in neighbor lookup   │
        │                                           │
        │  Boundary handling:                       │
        │    • Periodic: wrap indices                │
        │    • Wall: bounce-back (opposite dir)     │
        │    • Velocity BC: Zou-He or equilibrium   │
        │                                           │
        └──────────────┬───────────────────────────┘
                       │
                       ↓
        ┌──────────────────────────────────────────┐
        │  STEP 6: Swap Buffers                    │
        │  ─────────────────                       │
        │                                           │
        │  Double buffering:                        │
        │    tmp = f_src                            │
        │    f_src = f_dst                          │
        │    f_dst = tmp                            │
        │                                           │
        │  Avoids race conditions in streaming      │
        │                                           │
        └──────────────┬───────────────────────────┘
                       │
                       ↓
        ┌──────────────────────────────────────────┐
        │  OUTPUT                                   │
        │  ──────                                   │
        │                                           │
        │  Updated distribution functions: f_i      │
        │  Macroscopic fields: ρ, u, v, w          │
        │  Pressure: p = c_s²(ρ - ρ_0)             │
        │                                           │
        │  Ready for next time step                 │
        │                                           │
        └───────────────────────────────────────────┘


PERFORMANCE CHARACTERISTICS:
────────────────────────────

Memory Access Pattern (Streaming):
  • f_src reads: 19 directions × num_cells
  • f_dst writes: 19 directions × num_cells
  • Bandwidth-bound operation

Arithmetic Intensity (Collision):
  • Equilibrium: ~50 FLOPs per cell
  • Force term: ~30 FLOPs per cell
  • Total: ~80 FLOPs per cell × 19 directions
  • Compute-bound for small domains

Occupancy:
  • Block size: 8×8×8 = 512 threads
  • Registers per thread: ~32
  • Shared memory: 0 (could optimize with caching)
  • Theoretical occupancy: ~75%
```

---

## 6. Data Dependency Graph

### Field Dependencies Across Physics Modules

```
TIME STEP N:
────────────

┌─────────────┐
│ Temperature │  T^n
│   Field     │
└──────┬──────┘
       │
       ├──────────────────────────────────────┐
       │                                       │
       ↓                                       ↓
┌──────────────┐                    ┌─────────────────┐
│ Phase Change │                    │ Marangoni Force │
│              │                    │                 │
│ • fl^n       │                    │ F_M = (dσ/dT)·  │
│ • dfl/dt^n   │                    │       ∇_s T·|∇f|│
│ • J_evap^n   │                    │                 │
└──────┬───────┘                    └────────┬────────┘
       │                                     │
       │                                     │
       ↓                                     │
┌─────────────┐                              │
│ VOF Solver  │                              │
│             │                              │
│ Inputs:     │←─────────────┐               │
│  • u^n      │  Velocity    │               │
│  • J_evap^n │  from LBM    │               │
│  • dfl/dt^n │              │               │
│             │              │               │
│ Process:    │              │               │
│  1. Advect  │              │               │
│  2. Reconstruct             │               │
│  3. Curvature│              │               │
│             │              │               │
│ Outputs:    │              │               │
│  • f^{n+1}  │              │               │
│  • n^{n+1}  │              │               │
│  • κ^{n+1}  │              │               │
└──────┬──────┘              │               │
       │                     │               │
       │                     │               │
       ├─────────────────────┴───────────────┤
       │                                     │
       ↓                                     ↓
┌──────────────┐                   ┌────────────────┐
│Surface Tension                   │ Recoil Pressure│
│              │                   │                │
│ F_σ = σκ∇f   │                   │ F_R = P_r·n·|∇f│
└──────┬───────┘                   └────────┬───────┘
       │                                    │
       └────────────┬───────────────────────┘
                    │
                    ↓
         ┌──────────────────┐
         │ Force Accumulator │
         │                   │
         │ F_total = F_σ +   │
         │           F_M +   │
         │           F_R     │
         │                   │
         │ • Unit conversion │
         │ • CFL limiting    │
         └─────────┬─────────┘
                   │
                   ↓
         ┌─────────────────┐
         │  Fluid LBM      │
         │                 │
         │ Inputs:         │
         │  • f_i^n        │
         │  • F^n          │
         │                 │
         │ Process:        │
         │  • Collision    │
         │  • Streaming    │
         │                 │
         │ Outputs:        │
         │  • f_i^{n+1}    │
         │  • u^{n+1}      │
         │  • ρ^{n+1}      │
         └─────────┬───────┘
                   │
                   ↓
         ┌─────────────────┐
         │ Thermal LBM     │
         │                 │
         │ Inputs:         │
         │  • g_i^n        │
         │  • Q_laser^n    │
         │  • u^{n+1}      │ ← Convection
         │                 │
         │ Process:        │
         │  • Collision    │
         │  • Streaming    │
         │                 │
         │ Outputs:        │
         │  • T^{n+1}      │
         └─────────────────┘


DEPENDENCY SUMMARY:
──────────────────

Sequential Dependencies (MUST be in order):
  1. Thermal^n → Phase Change^n
  2. Phase Change^n → VOF Mass Loss^n
  3. Fluid^n → VOF Advection^{n+1}
  4. VOF^{n+1} + Thermal^n → Forces^{n+1}
  5. Forces^{n+1} → Fluid^{n+1}
  6. Fluid^{n+1} → Thermal^{n+1} (convection)

Parallel Opportunities:
  • Surface tension + Marangoni + Recoil (independent force kernels)
  • VOF subcycles can be parallelized across multiple streams
  • Fluid + Thermal collision steps (different lattices, no data sharing)

Critical Path (longest sequential chain):
  Thermal → Phase → VOF → Forces → Fluid
  ≈ 3ms + 1ms + 2ms + 2ms + 5ms = 13ms per step
```

---

## 7. Kernel Execution Timeline

### GPU Kernel Launch Sequence (Single Time Step)

```
TIME (ms) →
0         2         4         6         8        10        12        14
│─────────│─────────│─────────│─────────│─────────│─────────│─────────│
│                                                                       │
│ ┌─────────────────────────────────────────────────────────────────┐ │
│ │ PHASE 1: THERMAL LBM (~3ms)                                     │ │
│ │ ───────────────────────                                         │ │
│ │ ┌─────────────┐  ┌──────────────┐  ┌──────────────┐           │ │
│ │ │ Laser Heat  │→ │ Thermal Coll.│→ │ Thermal Stream│           │ │
│ │ │ Source      │  │ D3Q7 BGK     │  │ + Boundaries  │           │ │
│ │ │ 0.5ms       │  │ 1.5ms        │  │ 1.0ms         │           │ │
│ │ └─────────────┘  └──────────────┘  └──────────────┘           │ │
│ └─────────────────────────────────────────────────────────────────┘ │
│         │                                                             │
│         ↓                                                             │
│ ┌─────────────────────────────────────────────────────────────────┐ │
│ │ PHASE 2: PHASE CHANGE (~1ms)                                    │ │
│ │ ────────────────────────                                        │ │
│ │ ┌────────────┐  ┌───────────┐  ┌──────────────┐               │ │
│ │ │ Liquid     │→ │ Phase     │→ │ Evaporation  │               │ │
│ │ │ Fraction   │  │ Rate      │  │ Flux         │               │ │
│ │ │ 0.3ms      │  │ 0.4ms     │  │ 0.3ms        │               │ │
│ │ └────────────┘  └───────────┘  └──────────────┘               │ │
│ └─────────────────────────────────────────────────────────────────┘ │
│         │                                                             │
│         ↓                                                             │
│ ┌─────────────────────────────────────────────────────────────────┐ │
│ │ PHASE 3: VOF ADVECTION & INTERFACE (~3ms total)                 │ │
│ │ ──────────────────────────────────────                          │ │
│ │                                                                  │ │
│ │ ┌─────────────┐  ┌──────────────────────────────────────────┐  │ │
│ │ │ Convert Vel.│→ │ VOF SUBCYCLE LOOP (n=2 iterations)       │  │ │
│ │ │ Lattice→Phys│  │                                           │  │ │
│ │ │ 0.2ms       │  │ ┌──────────┐  ┌─────────┐  ┌──────────┐ │  │ │
│ │ └─────────────┘  │ │ Advect   │→ │Reconstruct→│ Curvature│ │  │ │
│ │                  │ │ Upwind   │  │ Interface │  │ κ=∇·n    │ │  │ │
│ │                  │ │ 0.6ms    │  │ 0.4ms    │  │ 0.3ms    │ │  │ │
│ │                  │ └──────────┘  └─────────┘  └──────────┘ │  │ │
│ │                  │ [Repeat for iteration 2]                  │  │ │
│ │                  └──────────────────────────────────────────┘  │ │
│ │                              │                                   │ │
│ │                              ↓                                   │ │
│ │                  ┌───────────────┐  ┌──────────────┐           │ │
│ │                  │ Evap. Mass    │→ │ Solid. Shrink│           │ │
│ │                  │ Loss          │  │              │           │ │
│ │                  │ 0.3ms         │  │ 0.2ms        │           │ │
│ │                  └───────────────┘  └──────────────┘           │ │
│ └─────────────────────────────────────────────────────────────────┘ │
│         │                                                             │
│         ↓                                                             │
│ ┌─────────────────────────────────────────────────────────────────┐ │
│ │ PHASE 4: FORCE ACCUMULATION (~2ms)                              │ │
│ │ ──────────────────────────────                                  │ │
│ │ ┌──────┐                                                         │ │
│ │ │ Zero │                                                         │ │
│ │ │Forces│                                                         │ │
│ │ │0.1ms │                                                         │ │
│ │ └───┬──┘                                                         │ │
│ │     │                                                             │ │
│ │     ├────────┬──────────┬────────────┐                          │ │
│ │     ↓        ↓          ↓            ↓                          │ │
│ │ ┌────────┐ ┌──────────┐ ┌──────────┐                           │ │
│ │ │Surface │ │Marangoni │ │ Recoil   │  ← PARALLEL EXECUTION     │ │
│ │ │Tension │ │ Force    │ │ Pressure │                           │ │
│ │ │ CSF    │ │ ∇_s T    │ │ P_r·n    │                           │ │
│ │ │ 0.3ms  │ │ 0.4ms    │ │ 0.3ms    │  (max = 0.4ms)            │ │
│ │ └────────┘ └──────────┘ └──────────┘                           │ │
│ │     │        │          │                                        │ │
│ │     └────────┴──────────┴────────────┐                          │ │
│ │                                       ↓                          │ │
│ │                          ┌────────────────────┐                  │ │
│ │                          │ Unit Conversion    │                  │ │
│ │                          │ Phys → Lattice     │                  │ │
│ │                          │ 0.2ms              │                  │ │
│ │                          └─────────┬──────────┘                  │ │
│ │                                    ↓                             │ │
│ │                          ┌──────────────────┐                    │ │
│ │                          │ CFL Force Limiter│                    │ │
│ │                          │ Adaptive Region  │                    │ │
│ │                          │ 0.5ms            │                    │ │
│ │                          └──────────────────┘                    │ │
│ └─────────────────────────────────────────────────────────────────┘ │
│         │                                                             │
│         ↓                                                             │
│ ┌─────────────────────────────────────────────────────────────────┐ │
│ │ PHASE 5: FLUID LBM (~5ms)                                       │ │
│ │ ─────────────────────                                           │ │
│ │ ┌──────────────┐  ┌───────────┐  ┌──────────────┐             │ │
│ │ │ Collision    │→ │ Streaming │→ │ Boundaries   │             │ │
│ │ │ BGK + Guo    │  │ D3Q19     │  │ + Macro Comp │             │ │
│ │ │ 2.5ms        │  │ 2.0ms     │  │ 0.5ms        │             │ │
│ │ └──────────────┘  └───────────┘  └──────────────┘             │ │
│ └─────────────────────────────────────────────────────────────────┘ │
│                                                                       │
│───────────────────────────────────────────────────────────────────────│
  TOTAL: ~14ms per time step (64³ grid, RTX 3060)


OPTIMIZATION OPPORTUNITIES:
───────────────────────────

1. Kernel Fusion:
   • VOF: Advect + Reconstruct + Curvature → 1 kernel
   • Savings: 2 kernel launches, 30% faster

2. Asynchronous Streams:
   • Stream 1: Thermal LBM
   • Stream 2: Fluid LBM collision
   • Overlap where possible

3. Force Accumulation Already Parallel:
   • CSF, Marangoni, Recoil run in parallel
   • Max time = 0.4ms (Marangoni)

4. Persistent Kernels:
   • Launch once, process multiple time steps
   • Reduce kernel launch overhead

5. Shared Memory:
   • Stencil operations (advection, reconstruction)
   • Cache 10×10×10 tiles with halo
   • Expected speedup: 2x for stencil kernels
```

---

## Conclusion

This document provides visual flowcharts and detailed algorithm flows for the VOF+LBM coupling implementation. Key insights:

1. **Clear Phase Separation:** Each physics module operates in well-defined phases with explicit data dependencies.

2. **Efficient Coupling:** VOF subcycling allows different CFL constraints while maintaining overall accuracy.

3. **Parallel Opportunities:** Force accumulation kernels can run in parallel; future work could overlap thermal and fluid LBM.

4. **Performance Bottlenecks:** Fluid LBM collision dominates runtime (35%); optimization should focus here.

5. **Data Flow:** One-way coupling for advection, two-way for forces ensures stable numerical behavior.

For implementation details and mathematical formulations, see the companion document:
`/home/yzk/LBMProject/docs/VOF_LBM_ALGORITHM_ANALYSIS.md`

---

**Document Version:** 1.0
**Author:** LBM-CUDA Chief Architect
**Date:** 2025-12-17
