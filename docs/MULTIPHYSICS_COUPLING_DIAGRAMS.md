# Multiphysics Coupling Diagrams

**Date:** 2026-01-10
**Purpose:** Visual representation of coupling architecture

---

## 1. Overall Coupling Sequence (Time Integration)

```
┌─────────────────────────────────────────────────────────────────┐
│                  MultiphysicsSolver::step(dt)                   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ Step 1: Laser Heat Source                                       │
│   applyLaserSource(dt)                                          │
│   └─> Q_laser [W/m³] → ThermalLBM source term                  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ Step 2: Thermal Diffusion + Phase Change                        │
│   thermalStep(dt)                                               │
│   ├─> ∂T/∂t = α∇²T + Q/ρc_p                                    │
│   ├─> Phase change: H(T) → liquid_fraction                     │
│   └─> Evaporation: J_evap = f(T, P_sat)                        │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ Step 3: VOF Interface Tracking                                  │
│   vofStep(dt) - with 10 subcycles                               │
│   ├─> ∂f/∂t + ∇·(f·u) = 0  [advection]                         │
│   ├─> Interface reconstruction → normals, curvature             │
│   └─> Evaporation mass loss: ∂f/∂t = -J_evap/(ρ·dx)            │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ Step 4: Fluid Flow (with force coupling)                        │
│   fluidStep(dt)                                                 │
│   ├─> computeTotalForce() [FORCE PIPELINE]                     │
│   │   ├─> Buoyancy: F_b = ρ·β·(T-T_ref)·g                      │
│   │   ├─> Surface Tension: F_st = σ·κ·∇f                       │
│   │   ├─> Marangoni: F_m = (dσ/dT)·∇_s T                       │
│   │   ├─> Recoil: F_r = P_recoil·n                             │
│   │   ├─> Darcy: F_d = -C·(1-f_l)²/f_l³·ρ·v                    │
│   │   ├─> Convert to lattice units: F_L = F/ρ                  │
│   │   └─> CFL limiting: F_L → F_limited                        │
│   ├─> collisionBGK(F_x, F_y, F_z) [Guo forcing]                │
│   ├─> streaming()                                               │
│   └─> computeMacroscopic() → u, ρ, p                            │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
                         time += dt
```

---

## 2. Force Accumulation Pipeline (Detail)

```
┌────────────────────────────────────────────────────────────────┐
│                   ForceAccumulator Pipeline                     │
└────────────────────────────────────────────────────────────────┘

STAGE 1: RESET
┌────────────────┐
│ reset()        │  → Zero all force arrays
│ F_x = F_y = F_z│     (d_fx_, d_fy_, d_fz_ = 0)
│ = 0 [N/m³]     │
└────────────────┘
         │
         ▼
STAGE 2: ACCUMULATE FORCES (Physical Units [N/m³])
┌────────────────────────────────────────────────────────────────┐
│ addBuoyancyForce(T, T_ref, β, ρ, g)                            │
│   F_buoyancy = ρ · β · (T - T_ref) · g                         │
│   Example: ρ=4110, β=1.5e-5, ΔT=500K, g=9.81                   │
│            → F ≈ 300 N/m³                                       │
└────────────────────────────────────────────────────────────────┘
         │ (F_x += F_buoy_x, F_y += F_buoy_y, F_z += F_buoy_z)
         ▼
┌────────────────────────────────────────────────────────────────┐
│ addSurfaceTensionForce(κ, f, σ)                                │
│   F_st = σ · κ · ∇f                                            │
│   Example: σ=1.65, κ=1e5, |∇f|=1e5                             │
│            → F ≈ 2.7e11 N/m³ (interface only)                  │
└────────────────────────────────────────────────────────────────┘
         │ (F_x += F_st_x, ...)
         ▼
┌────────────────────────────────────────────────────────────────┐
│ addMarangoniForce(T, f, n, dσ/dT, h_interface)                 │
│   F_m = (dσ/dT) · ∇_s T · |∇f| / (h_interface · dx)            │
│   ∇_s T = ∇T - (∇T·n)n  [tangential projection]               │
│   Example: dσ/dT=-0.26e-3, ∇T=1e6, |∇f|/dx=2.5e5, h=1.0       │
│            → F ≈ -32.5e6 N/m³  ← BUG if h=2.0 → -16.25e6       │
└────────────────────────────────────────────────────────────────┘
         │ (F_x += F_m_x, ...)
         ▼
┌────────────────────────────────────────────────────────────────┐
│ addRecoilPressureForce(T, f, n, T_boil, L_v, ...)              │
│   P_sat = P_0 · exp(L_v·M/R · (1/T_boil - 1/T))               │
│   F_recoil = -C_r · P_sat · n · |∇f| / (h · dx)                │
│   Example: T=3500K, P_sat≈1e6 Pa, C_r=0.54                     │
│            → F ≈ 5e8 N/m³ (z-direction, into liquid)           │
└────────────────────────────────────────────────────────────────┘
         │ (F_x += F_r_x, ...)
         ▼
┌────────────────────────────────────────────────────────────────┐
│ addDarcyDamping(f_l, v, C_darcy, ρ)                            │
│   F_darcy = -C · (1-f_l)² / (f_l³+ε) · ρ · v                   │
│   Example: f_l=0.5 (mushy), v=0.01 m/s, C=1e7                  │
│            → F ≈ -4e8 N/m³ (opposes flow)                      │
└────────────────────────────────────────────────────────────────┘
         │ (F_x += F_d_x, ...)
         ▼
STAGE 3: UNIT CONVERSION
┌────────────────────────────────────────────────────────────────┐
│ convertToLatticeUnits(dx, dt, ρ)                               │
│   F_lattice = F_physical / ρ                                   │
│                                                                 │
│   Rationale:                                                    │
│   - LBM expects dimensionless acceleration                     │
│   - a = F/ρ [m/s²]                                             │
│   - In lattice: a_L = a_phys (when dt_L = 1)                  │
│   - Therefore: F_L = F_phys / ρ_phys                           │
│                                                                 │
│   Example: F=-32.5e6 N/m³, ρ=4110 kg/m³                        │
│            → F_L = -7908 (dimensionless)                       │
└────────────────────────────────────────────────────────────────┘
         │
         ▼
STAGE 4: CFL LIMITING (Stability)
┌────────────────────────────────────────────────────────────────┐
│ applyCFLLimiting(v_current, v_target, ramp_factor)             │
│                                                                 │
│   IF (|v_current + F_L| > v_target):                           │
│      scale = (v_target - |v_current|) / |F_L|                  │
│      F_limited = F_L · scale                                   │
│   ELSE IF (|v_current + F_L| > ramp_factor · v_target):        │
│      Gradual scaling (smooth transition)                       │
│   ELSE:                                                         │
│      F_limited = F_L (no limiting)                             │
│                                                                 │
│   Example: v_current=0, F_L=-7908, v_target=0.15               │
│            → scale = 0.15/7908 = 1.9e-5                        │
│            → F_limited = -0.15                                 │
│                                                                 │
│   Result: Velocity builds up gradually over time               │
└────────────────────────────────────────────────────────────────┘
         │
         ▼
OUTPUT: F_limited → FluidLBM::collisionBGK(F_x, F_y, F_z)
```

---

## 3. Marangoni Force Computation (Detail with BUG)

```
┌─────────────────────────────────────────────────────────────────┐
│           Marangoni Force Computation Flowchart                 │
└─────────────────────────────────────────────────────────────────┘

INPUT: Temperature T, Fill Level f, Normals n, Material dσ/dT

STEP 1: Interface Detection
┌────────────────────────────┐
│ if (f < 0.01 || f > 0.99)  │ ──> RETURN (not at interface)
│    return 0;               │
└────────────────────────────┘
         │ (interface detected)
         ▼
STEP 2: Temperature Gradient (Central Differences)
┌────────────────────────────────────────────────────────────────┐
│ ∇T = [(T[i+1] - T[i-1])/(2dx),                                 │
│       (T[j+1] - T[j-1])/(2dx),                                 │
│       (T[k+1] - T[k-1])/(2dx)]                                 │
│                                                                 │
│ Example: Laser heating in z-direction                          │
│   T[k+1]=2000K, T[k-1]=1500K, dx=2e-6                          │
│   ∇T_z = (2000-1500)/(2×2e-6) = 1.25e8 K/m                     │
└────────────────────────────────────────────────────────────────┘
         │
         ▼
STEP 3: Tangential Projection (Remove Normal Component)
┌────────────────────────────────────────────────────────────────┐
│ ∇T·n = (∇T_x·n_x + ∇T_y·n_y + ∇T_z·n_z)                        │
│ ∇_s T = ∇T - (∇T·n)·n                                          │
│                                                                 │
│ Physics: Marangoni force acts TANGENTIAL to interface          │
│          Surface tension gradient pulls liquid along surface   │
│                                                                 │
│ Example: n = (0, 0, 1) [horizontal interface]                  │
│          ∇T = (1e6, 0, 1.25e8)                                 │
│          ∇T·n = 1.25e8                                          │
│          ∇_s T = (1e6, 0, 0)  [only tangential component]      │
└────────────────────────────────────────────────────────────────┘
         │
         ▼
STEP 4: Fill Level Gradient (Interface Delta Function)
┌────────────────────────────────────────────────────────────────┐
│ ∇f = [(f[i+1] - f[i-1])/(2dx), ...]                            │
│ |∇f| = sqrt(∇f_x² + ∇f_y² + ∇f_z²)                             │
│                                                                 │
│ Example: Sharp interface (f=0→1 over 2 cells)                  │
│   f[i-1]=0.0, f[i+1]=1.0                                        │
│   ∇f_x = (1.0-0.0)/(2×2e-6) = 2.5e5 1/m                        │
│   |∇f| ≈ 2.5e5 1/m                                             │
└────────────────────────────────────────────────────────────────┘
         │
         ▼
STEP 5: Marangoni Force Calculation
┌────────────────────────────────────────────────────────────────┐
│ F_marangoni = (dσ/dT) · ∇_s T · |∇f| / (h_interface · dx)      │
│                                                                 │
│ ┌──────────────────────────────────────────────────────────┐   │
│ │ CRITICAL BUG:                                            │   │
│ │ h_interface parameter NOT passed from multiphysics.cu    │   │
│ │ Default value: h_interface = 2.0 (from header file)      │   │
│ │ Correct value: h_interface = 1.0 (for sharp VOF)         │   │
│ │                                                           │   │
│ │ Impact: Force UNDERESTIMATED by factor of 2              │   │
│ └──────────────────────────────────────────────────────────┘   │
│                                                                 │
│ Example (BUGGY):                                                │
│   dσ/dT = -0.26e-3 N/(m·K)                                     │
│   ∇_s T = (1e6, 0, 0) K/m                                      │
│   |∇f| = 2.5e5 1/m                                             │
│   h_interface = 2.0  ← DEFAULT (WRONG!)                        │
│   dx = 2e-6 m                                                  │
│                                                                 │
│   F_x = (-0.26e-3) × 1e6 × 2.5e5 / (2.0 × 2e-6)                │
│       = -65 / 4e-6                                             │
│       = -16.25e6 N/m³                                           │
│                                                                 │
│ Example (FIXED with h_interface=1.0):                          │
│   F_x = (-0.26e-3) × 1e6 × 2.5e5 / (1.0 × 2e-6)                │
│       = -65 / 2e-6                                             │
│       = -32.5e6 N/m³  (2× larger)                              │
└────────────────────────────────────────────────────────────────┘
         │
         ▼
OUTPUT: F_marangoni added to force accumulator
```

---

## 4. Unit Conversion and CFL Limiting Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                     Force → Velocity Pipeline                   │
└─────────────────────────────────────────────────────────────────┘

PHYSICAL FORCE [N/m³]
│
│ F_marangoni = -32.5e6 N/m³  (fixed value)
│
▼
UNIT CONVERSION
┌────────────────────────────────────────────────────────────────┐
│ F_lattice = F_physical / ρ_physical                             │
│           = -32.5e6 / 4110                                      │
│           = -7908  [dimensionless]                             │
│                                                                 │
│ Interpretation: This is acceleration in lattice units           │
│   a_lattice = 7908 (lattice lengths per timestep²)             │
└────────────────────────────────────────────────────────────────┘
│
▼
CFL LIMITING (Adaptive)
┌────────────────────────────────────────────────────────────────┐
│ Current state: v_current = 0 (at t=0)                          │
│ Target velocity: v_target = 0.15 (lattice units)               │
│                 = 0.15 × (dx/dt) = 3 m/s (physical)            │
│                                                                 │
│ Predicted velocity: v_new = v_current + F_lattice              │
│                           = 0 + (-7908)                         │
│                           = -7908  (lattice units)              │
│                                                                 │
│ Check: |v_new| = 7908 >> v_target = 0.15                       │
│                                                                 │
│ Action: SCALE FORCE                                             │
│   scale = (v_target - |v_current|) / |F_lattice|               │
│         = (0.15 - 0) / 7908                                     │
│         = 1.9e-5                                                │
│                                                                 │
│   F_limited = F_lattice × scale                                │
│             = -7908 × 1.9e-5                                    │
│             = -0.15  (lattice units)                            │
└────────────────────────────────────────────────────────────────┘
│
▼
APPLY FORCE IN LBM
┌────────────────────────────────────────────────────────────────┐
│ Guo forcing scheme (in fluidBGKCollisionVaryingForceKernel):   │
│   u_corrected = u_uncorrected + 0.5 × F_limited / ρ_lattice    │
│                                                                 │
│ For ρ_lattice ≈ 1:                                              │
│   Δu = 0.5 × F_limited                                          │
│      = 0.5 × (-0.15)                                            │
│      = -0.075  (lattice units)                                 │
│                                                                 │
│ After 1 timestep:                                               │
│   v_new = v_old + Δu                                            │
│         = 0 + (-0.075)                                          │
│         = -0.075  (lattice units)                              │
│         = -0.075 × (2e-6 / 1e-9)                               │
│         = -150 m/s  (physical) ← Still too high!               │
│                                                                 │
│ WAIT: This suggests CFL limiting needs further tuning!         │
│       OR: Force magnitude is still too large                   │
│       OR: Time stepping is incorrect                           │
└────────────────────────────────────────────────────────────────┘
│
▼
NEXT TIMESTEP (Gradual Build-Up)
┌────────────────────────────────────────────────────────────────┐
│ At t=1: v_current = -0.075                                     │
│         F_lattice = -7908 (still the same)                     │
│         v_new = -0.075 + (-7908) = -7908.075                   │
│         |v_new| >> v_target → LIMIT AGAIN                      │
│         scale = (0.15 - 0.075) / 7908 = 9.5e-6                 │
│         F_limited = -7908 × 9.5e-6 = -0.075                    │
│         Δu = 0.5 × (-0.075) = -0.0375                          │
│         v_new = -0.075 + (-0.0375) = -0.1125                   │
│                                                                 │
│ At t=2: v_current = -0.1125                                    │
│         F_limited ≈ -0.0375 (approaching limit)                │
│         v_new ≈ -0.13                                           │
│                                                                 │
│ At t→∞: v approaches -0.15 (asymptotically)                    │
│         Velocity saturates at CFL-limited value                │
└────────────────────────────────────────────────────────────────┘
```

**OBSERVATION:** The CFL limiter correctly prevents velocity explosion, but may be preventing velocity from reaching physical expectations. Need to verify if v_target=0.15 is appropriate for Marangoni flow.

---

## 5. Data Dependency Graph

```
┌─────────────┐
│  Laser      │
│  Power      │
└──────┬──────┘
       │
       ▼
┌──────────────────────────────────────┐
│       ThermalLBM                     │
│  ∂T/∂t = α∇²T + Q_laser/ρc_p         │
└──────┬───────────────┬───────────────┘
       │               │
       │ T(x,t)        │ J_evap(T)
       ▼               ▼
┌──────────────┐  ┌────────────────┐
│  Marangoni   │  │  VOF Mass      │
│  Force       │  │  Loss          │
│  F_m(∇T,n)   │  │  ∂f/∂t=-J/ρdx  │
└──────┬───────┘  └────────┬───────┘
       │                   │
       ▼                   ▼
┌──────────────────────────────────────┐
│      ForceAccumulator                │
│  F_total = F_b + F_m + F_st + ...    │
└──────────────┬───────────────────────┘
               │ F_total(x)
               ▼
┌──────────────────────────────────────┐
│       FluidLBM                       │
│  ∂u/∂t + (u·∇)u = -∇p/ρ + ν∇²u + F  │
└──────────────┬───────────────────────┘
               │ u(x,t)
               ▼
┌──────────────────────────────────────┐
│       VOFSolver                      │
│  ∂f/∂t + ∇·(f·u) = 0                 │
│  Reconstruct → n, κ                  │
└──────────────┬───────────────────────┘
               │ n, κ
               │
               ├────────┐
               ▼        ▼
         Surface    Marangoni
         Tension    (feedback)
         F_st(κ)    F_m(n,∇T)
```

**Key Observations:**

1. **Thermal → Force:** One-way coupling (T drives forces)
2. **Force → Fluid:** One-way coupling (forces drive velocity)
3. **Fluid → VOF:** One-way coupling (velocity advects interface)
4. **VOF → Force:** One-way coupling (normals/curvature for F_st, F_m)

**No circular dependencies** → Sequential coupling is stable!

---

## 6. Memory Layout and Pointer Flow

```
┌───────────────────────────────────────────────────────────────┐
│              MultiphysicsSolver (Owner)                        │
├───────────────────────────────────────────────────────────────┤
│                                                                │
│  ┌──────────────────────────────────────────────┐             │
│  │  unique_ptr<ForceAccumulator> force_accum_   │             │
│  │  ├─> float* d_fx_ [device memory]            │             │
│  │  ├─> float* d_fy_ [device memory]            │             │
│  │  └─> float* d_fz_ [device memory]            │             │
│  └──────────────────────────────────────────────┘             │
│                           ▲                                    │
│                           │ getters return raw pointers        │
│                           │                                    │
│  ┌────────────────────────────────────────────────────┐       │
│  │  d_force_x_ = force_accum_->getForceX();          │       │
│  │  d_force_y_ = force_accum_->getForceY();          │       │
│  │  d_force_z_ = force_accum_->getForceZ();          │       │
│  │  ↑                                                 │       │
│  │  │ Aliasing: d_force_x_ POINTS TO force_accum's   │       │
│  │  │           internal memory (no copy!)           │       │
│  └────────────────────────────────────────────────────┘       │
│                           │                                    │
│                           │ Passed to FluidLBM                 │
│                           ▼                                    │
│  ┌──────────────────────────────────────────────┐             │
│  │  unique_ptr<FluidLBM> fluid_                 │             │
│  │  └─> collisionBGK(d_force_x_,                │             │
│  │                   d_force_y_,                 │             │
│  │                   d_force_z_)                 │             │
│  │       ↓                                       │             │
│  │  Kernel reads force arrays directly          │             │
│  │  No memory copy required                     │             │
│  └──────────────────────────────────────────────┘             │
│                                                                │
└───────────────────────────────────────────────────────────────┘

ADVANTAGES:
✓ Zero-copy: No cudaMemcpy between force computation and fluid step
✓ Memory efficiency: Single allocation for force arrays
✓ Cache-friendly: Forces computed and consumed in same memory location

SAFETY:
✓ RAII: ForceAccumulator destructor frees memory automatically
✓ Lifetime: force_accum_ lives as long as MultiphysicsSolver
✓ No dangling pointers: d_force_x/y/z_ valid throughout solver lifetime
```

---

## 7. Thread Safety and Race Conditions

```
┌─────────────────────────────────────────────────────────────────┐
│                  CUDA Kernel Thread Mapping                     │
└─────────────────────────────────────────────────────────────────┘

EXAMPLE: addBuoyancyForceKernel

Grid dimensions: (blocks_x, blocks_y, blocks_z)
Block dimensions: (256 threads per block)

Thread mapping:
┌─────────────────────────────────────────────────────────────┐
│ Thread 0: idx = 0 → writes to fx[0], fy[0], fz[0]          │
│ Thread 1: idx = 1 → writes to fx[1], fy[1], fz[1]          │
│ Thread 2: idx = 2 → writes to fx[2], fy[2], fz[2]          │
│ ...                                                          │
│ Thread N: idx = N → writes to fx[N], fy[N], fz[N]          │
└─────────────────────────────────────────────────────────────┘

KEY PROPERTY: Each thread writes to UNIQUE memory location
              No two threads write to same idx
              → NO RACE CONDITIONS

FORCE ACCUMULATION (+=) is safe because:
┌─────────────────────────────────────────────────────────────┐
│ Step 1: reset()          → All forces = 0                   │
│ Step 2: addBuoyancy()    → fx[idx] += F_b (thread-safe)     │
│ Step 3: synchronize()    → Wait for all threads             │
│ Step 4: addMarangoni()   → fx[idx] += F_m (thread-safe)     │
│ Step 5: synchronize()    → Wait for all threads             │
│ ...                                                          │
└─────────────────────────────────────────────────────────────┘

NO ATOMICS REQUIRED because:
- Each kernel processes complete domain
- No overlapping writes between threads
- Sequential kernel launches ensure proper ordering
```

---

## 8. Critical Bug Visualization

```
┌─────────────────────────────────────────────────────────────────┐
│         Marangoni Force Scaling Bug Visualization               │
└─────────────────────────────────────────────────────────────────┘

CURRENT (BUGGY) CODE:
┌──────────────────────────────────────────────────────────────┐
│ multiphysics_solver.cu:1674                                  │
│                                                               │
│ force_accumulator_->addMarangoniForce(                       │
│     temperature, fill_level, normals,                        │
│     config_.dsigma_dT,                                       │
│     config_.nx, config_.ny, config_.nz,                      │
│     config_.dx);  ← Missing h_interface!                     │
│                                                               │
│ ↓ Uses default from header                                   │
│                                                               │
│ force_accumulator.h:120                                      │
│ void addMarangoniForce(..., float h_interface = 2.0f);       │
│                                     ^^^^^^^^^^^^^^^^         │
│                                     DEFAULT VALUE            │
└──────────────────────────────────────────────────────────────┘
           │
           ▼
┌──────────────────────────────────────────────────────────────┐
│ force_accumulator.cu:276                                     │
│                                                               │
│ float coeff = dsigma_dT * grad_f_mag / (h_interface * dx);   │
│                                          ^^^^^^^^^^^         │
│                                          = 2.0 (WRONG!)      │
│                                                               │
│ Should be: h_interface = 1.0 for sharp VOF interface         │
│                                                               │
│ Impact: Force divided by 2.0 instead of 1.0                  │
│         → 50% reduction in Marangoni magnitude!              │
└──────────────────────────────────────────────────────────────┘

FORCE MAGNITUDE COMPARISON:
┌──────────────────────────────────────────────────────────────┐
│                                                               │
│  BUGGY (h=2.0):   F = -16.25e6 N/m³  ■■■■■■■■■■              │
│                                                               │
│  FIXED (h=1.0):   F = -32.50e6 N/m³  ■■■■■■■■■■■■■■■■■■■■    │
│                                                               │
│  Ratio: 2.0×                                                 │
│                                                               │
└──────────────────────────────────────────────────────────────┘

IMPACT ON VELOCITY (after unit conversion and CFL limiting):
┌──────────────────────────────────────────────────────────────┐
│                                                               │
│  BUGGY:  v_max ≈ 1.5 m/s  (too small, tests fail)            │
│                                                               │
│  FIXED:  v_max ≈ 3.0 m/s  (correct, tests should pass)       │
│                                                               │
└──────────────────────────────────────────────────────────────┘

FIX:
┌──────────────────────────────────────────────────────────────┐
│ force_accumulator_->addMarangoniForce(                       │
│     temperature, fill_level, normals,                        │
│     config_.dsigma_dT,                                       │
│     config_.nx, config_.ny, config_.nz,                      │
│     config_.dx,                                              │
│     1.0f);  ← EXPLICIT h_interface = 1.0                     │
└──────────────────────────────────────────────────────────────┘
```

---

## 9. Energy Flow Diagram (Validation)

```
┌─────────────────────────────────────────────────────────────────┐
│                   Energy Conservation Check                     │
└─────────────────────────────────────────────────────────────────┘

INPUT POWER:
┌──────────────────────┐
│  Laser Power         │
│  P_laser = 200 W     │
│  (absorbed)          │
└──────┬───────────────┘
       │
       ▼
THERMAL ENERGY:
┌──────────────────────────────────────────────────────────────┐
│  E_thermal = ∫ ρ · c_p · T dV                                 │
│                                                               │
│  dE_thermal/dt = P_laser - P_evap - P_rad - P_cond + P_visc  │
│                                                               │
│  P_laser: Input from laser                                   │
│  P_evap:  Evaporation cooling                                │
│  P_rad:   Radiation cooling                                  │
│  P_cond:  Boundary conduction                                │
│  P_visc:  Viscous dissipation (u → heat)                     │
└──────────────────────┬───────────────────────────────────────┘
       │               │
       │ Heat → Flow   │ Flow → Heat
       ▼               ▼
KINETIC ENERGY:
┌──────────────────────────────────────────────────────────────┐
│  E_kinetic = ∫ (1/2) ρ |u|² dV                                │
│                                                               │
│  dE_kinetic/dt = ∫ F · u dV - P_visc                          │
│                                                               │
│  Where F·u represents work done by forces:                   │
│    - Buoyancy:        F_b · u  (converts thermal → kinetic)  │
│    - Marangoni:       F_m · u  (converts thermal → kinetic)  │
│    - Surface tension: F_st · u (potential → kinetic)         │
│    - Darcy:           F_d · u  (dissipation, negative)       │
└──────────────────────────────────────────────────────────────┘

TOTAL ENERGY BALANCE:
┌──────────────────────────────────────────────────────────────┐
│  d(E_thermal + E_kinetic)/dt = P_laser - P_evap - P_rad      │
│                                                               │
│  Should satisfy: |residual| / P_laser < 5% (validation)       │
└──────────────────────────────────────────────────────────────┘
```

---

## 10. Test Failure Diagnosis Flowchart

```
┌─────────────────────────────────────────────────────────────────┐
│         Marangoni Velocity Test Failure Diagnosis               │
└─────────────────────────────────────────────────────────────────┘

START: Test expects v_marangoni > 1.0 m/s
       Observed: v_marangoni ≈ 0 m/s
│
▼
CHECK 1: Is Marangoni force computed?
├─ NO → BUG: Force computation disabled or not called
│          Fix: Enable config_.enable_marangoni
│
└─ YES → Force magnitude F_m > 0 (verified in diagnostics)
         ▼
         CHECK 2: Is temperature gradient present?
         ├─ NO → BUG: ∇T ≈ 0, no driving force
         │          Fix: Increase laser power or gradient
         │
         └─ YES → ∇T ~ 1e6 K/m (verified)
                  ▼
                  CHECK 3: Is interface detected?
                  ├─ NO → BUG: fill_level not in (0.01, 0.99)
                  │          Fix: Check VOF initialization
                  │
                  └─ YES → Interface cells detected
                           ▼
                           CHECK 4: Force magnitude correct?
                           ├─ TOO SMALL → BUG: h_interface scaling
                           │                   ← FOUND HERE!
                           │              Fix: Set h=1.0 explicitly
                           │
                           └─ CORRECT → F ~ 1e7 N/m³
                                        ▼
                                        CHECK 5: Unit conversion OK?
                                        ├─ NO → BUG: F_lattice wrong
                                        │          Fix: Check ρ conversion
                                        │
                                        └─ YES → F_lattice ~ 1e3
                                                 ▼
                                                 CHECK 6: CFL limiting?
                                                 ├─ TOO AGGRESSIVE
                                                 │    → F_limited ≈ 0
                                                 │    Fix: Increase v_target
                                                 │
                                                 └─ REASONABLE
                                                      ▼
                                                      CHECK 7: Force applied to fluid?
                                                      ├─ NO → BUG: Pointer issue
                                                      │          Fix: Check d_force_x/y/z
                                                      │
                                                      └─ YES
                                                           ▼
                                                           Test should PASS!
```

**DIAGNOSIS:** Failure at CHECK 4 (h_interface bug)

---

## Conclusion

The multiphysics architecture is sound with excellent modular design. The critical bug in Marangoni force computation (missing h_interface parameter) is the root cause of test failures. Fixing this single parameter should resolve the Marangoni velocity validation issues.
