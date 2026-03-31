# LBM vs OpenFOAM Spot Melting Benchmark — Status Report

**Date**: 2026-03-31  
**Branch**: `benchmark/conduction-316L`  
**Codebase**: `/home/yzk/LBMProject`

---

## 1. Project Overview

CUDA-based D3Q19/D3Q7 LBM solver for metal AM (LPBF) simulation. Architecture:
- D3Q19 fluid (BGK/EDM collision, TRT stabilization)
- D3Q7 thermal (BGK collision, ESM phase change — Jiaung 2001)
- VOF free surface tracking (PLIC/TVD)
- Marangoni surface forces (CSF method via VOF interface)
- Darcy mushy-zone damping (semi-implicit Carman-Kozeny)
- MultiphysicsSolver orchestrates all modules

## 2. Benchmark Configuration (Both Solvers)

| Parameter | Value |
|-----------|-------|
| Material | 316L, **constant** properties (ρ=7900, cp=700, k=20) |
| Laser | P=150W, η=0.35, r₀=25μm, stationary spot |
| Schedule | ON: 0–50μs, OFF: 50–100μs |
| Domain | 200×200×100 μm (dx=2μm) |
| Phase change | T_sol=1650K, T_liq=1700K, L=260kJ/kg |
| Marangoni | dσ/dT = **+1.0e-4** N/(m·K) — inward flow → downward jet |
| Viscosity | μ=0.005 Pa·s (ν=6.33e-7 m²/s, Pr=0.175) |
| Darcy | Carman-Kozeny Cu=1e6, ε=1e-3 |
| BCs | Dirichlet 300K on 5 faces, adiabatic top surface |

## 3. Validation Results

### 3.1 Pure Conduction (no flow) — PASS ✅

![conduction validation](contours/solidus_validation.png)

LBM and OpenFOAM solidus isotherms are **near-identical** at all 4 time snapshots. This confirms:
- ESM (enthalpy source method) is physically equivalent to OpenFOAM's solidificationMeltingSource
- Heat source injection is correct
- Thermal diffusion is accurate

### 3.2 Marangoni Convection — FAIL ❌

![marangoni validation](contours/marangoni_validation.png)

**OpenFOAM (dark gray)**: Deep V-shaped melt pool, depth reaching ~50μm at 50μs and ~60μm at 75μs. Classic positive-dσ/dT Marangoni pattern — inward surface flow drives a strong downward jet at the center.

**LBM (gold)**: Shallow semicircular pool with slight irregularity, depth only ~20-30μm. Nearly identical to the pure-conduction case (blue dotted). The Marangoni force has **negligible effect** on the melt pool shape.

*Note: Contours are extracted via `matplotlib.contour()` on 2D temperature fields (marching squares), not from point-sorted CSVs. This guarantees topology-correct smooth curves for any shape including deep V-pools.*

### 3.3 Quantitative Depth Comparison

| Time | OF Marangoni depth | LBM Marangoni depth | LBM conduction depth | LBM deepening |
|------|-------------------|--------------------|--------------------|--------------|
| 25μs | ~30 μm | ~18 μm | 16 μm | +12% |
| 50μs | ~50 μm | ~30 μm | 24 μm | +25% |
| 60μs | ~57 μm | ~32 μm | 26 μm | +23% |
| 75μs | ~60 μm | ~38 μm | 30 μm | +27% |

LBM Marangoni produces only a ~25% deepening vs conduction, while OpenFOAM shows ~100% deepening. **The LBM Marangoni force is ~4× too weak.**

## 4. Diagnosed Root Causes

### 4.1 Marangoni Force Implementation Mismatch

**OpenFOAM approach** (correct for single-phase-with-wall):
- Marangoni is applied as a **velocity boundary condition** at the top wall:
  ```
  u_face = u_internal + (dσ/dT / μ) · ∇_s T · δ
  ```
  where δ is the cell-to-face distance. This directly sets the surface velocity.

**LBM approach** (CSF body force — designed for free surfaces):
- Marangoni is applied as a **volumetric body force** via the CSF (Continuum Surface Force) model:
  ```
  F = dσ/dT · ∇_s T · |∇f|
  ```
  where `|∇f|` is the VOF gradient magnitude that localizes the force to the interface.
- Requires a VOF interface with proper gradient. We created a thin 2-cell gas buffer, but:
  - The interface is a single-cell step (fill: 1.0 → 0.5 → 0.0), creating a very narrow `|∇f|` zone
  - The CSF force is smeared over ~1 cell, while OpenFOAM's BC acts directly at the surface
  - The force magnitude depends on the VOF gradient resolution, which is poor at dx=2μm

### 4.2 Velocity Evidence

| Metric | LBM | Expected (OF-scale) |
|--------|-----|-------------------|
| v_max at 50μs | 2.88 m/s | ~10-15 m/s |
| Max Marangoni force | ~6e-3 LU | Should be ~10× higher |
| CFL limiter triggered | No (0% reduction) | — |

The LBM flow velocity is ~4× too low, consistent with the ~4× insufficient melt pool deepening.

### 4.3 Additional Concerns

1. **Darcy over-damping?** Cu=1e6 with the LBM semi-implicit Darcy formulation (`u = m/(ρ+K/2)`) may damp velocities more aggressively than OpenFOAM's diagonal-source Darcy. Needs quantitative comparison.

2. **CFL force limiter**: While it reported 0% reduction, the `ForceAccumulator` has a `fl > 0.999` gate for Marangoni that restricts it to fully-liquid cells only. If the interface cell (fill=0.5) has fl<1 due to being partially solid, the force is zeroed there.

3. **Thermal omega=1.0**: At omega=1.0, collision fully relaxes to equilibrium each step. This kills all non-equilibrium modes including the advection term `v·∇T`. The thermal advection coupling may be severely under-resolved, meaning the downward jet doesn't efficiently transport heat downward.

## 5. Key Files

| File | Description |
|------|-------------|
| `apps/benchmark_conduction_316L.cu` | Pure conduction benchmark (standalone, lbm_thermal_only) |
| `apps/benchmark_spot_melt_316L.cu` | Marangoni benchmark (MultiphysicsSolver, full physics) |
| `singlepointbench/contours/plot_marangoni_validation.py` | Comparison plotting script |
| `singlepointbench/contours/marangoni_validation.png` | Current comparison figure |
| `openfoam/spot_melting_marangoni/` | OpenFOAM case (complete, already run) |
| `src/physics/force_accumulator.cu` | Marangoni kernel (CSF formulation, line ~278) |
| `src/physics/multiphysics/multiphysics_solver.cu` | Step orchestration, force pipeline |
| `src/physics/thermal/thermal_lbm.cu` | D3Q7 thermal solver + ESM (T_MAX raised to 1e6) |

## 6. Technical Debt / Bugs Found During Benchmark

1. **T_MAX hard clamp** in `computeTemperatureKernel`: Was 50000K, raised to 1e6. The old value destroyed g↔T consistency at high temperatures.

2. **BGK source term instability**: Adding `g_q += w_q·dT` with dT/T > 10% and omega > 1 causes exponential blowup. Root cause: over-relaxation amplifies non-equilibrium perturbations from the source injection. Fix: tau=1.0 (omega=1.0) eliminates over-relaxation.

3. **`addHeatSourceKernel` CE correction**: The `source_correction = 1.0` comment claims "no correction needed for scalar transport" citing Li et al. (2013). However, Chapman-Enskog analysis confirms the correction IS needed for the non-equilibrium modes (though the zeroth moment is exact). At omega=1.0 this is moot, but at omega>1 it contributes to instability.

## 7. Recommended Next Steps

### Priority 1: Fix Marangoni force magnitude
- **Option A**: Replace CSF body force with a direct Marangoni boundary condition (like OpenFOAM). Apply `τ = dσ/dT · ∇_s T` as a stress BC at the top metal surface using the Inamuro specular method (already validated in the 1D Marangoni return flow benchmark).
- **Option B**: Improve the VOF interface for CSF — use a smoother fill-level transition (tanh over 3-4 cells) to create a broader `|∇f|` zone for better force localization.

### Priority 2: Verify Darcy equivalence
- Compare LBM semi-implicit Darcy damping vs OpenFOAM's diagonal-source Darcy at the same Cu, ε values with a simple 1D solidification-with-flow test.

### Priority 3: Fix thermal advection at omega=1.0
- tau_thermal=1.0 means the thermal collision completely relaxes to equilibrium. The first-order non-equilibrium part (which carries the heat flux and advection) is zeroed each step. Consider using tau_thermal=0.8 (omega=1.25) for the Marangoni benchmark, but this requires the equilibrium-forced heat injection to avoid the BGK instability.

---

*Report generated for external AI consultation. All source code, simulation data, and OpenFOAM reference cases are in the repository.*
