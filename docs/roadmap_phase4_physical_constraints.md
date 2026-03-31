# Phase 4 Technical Roadmap: Physical Constraints for Hybrid LBM-FDM

**Date**: 2026-04-01  
**Baseline**: Hybrid D3Q19 FluidLBM + FDM ThermalSolver (SUPERBEE TVD), commit `479953e`  
**Target**: Eliminate +32% depth overshoot by adding missing physics, not tuning parameters

---

## Item 1: Evaporation Cooling & Recoil Pressure

**Problem**: T_max = 80,000 K (no physical limit) → ∇T 6× too steep → Marangoni 6× too strong

### Mathematical Model

At the free surface where T > T_boil, the Hertz-Knudsen-Langmuir evaporation flux:

```
J_evap = α_evap · P_sat(T) / √(2π · R · T / M)
```

where:
- α_evap = 0.18 (evaporation coefficient, 316L)
- M = 0.0558 kg/mol (molar mass)
- R = 8.314 J/(mol·K)
- P_sat(T) = P_ref · exp[(L_vap·M/R) · (1/T_boil - 1/T)] (Clausius-Clapeyron)

The energy sink at the surface (per unit area):

```
q_evap = J_evap · L_vap    [W/m²]
```

where L_vap = 7.45×10⁶ J/kg for 316L.

### FDM Integration

In `ThermalFDM::applyEvaporationCooling()` — applied AFTER the advDiff step, at the z_max surface:

```
dT_evap = -q_evap · dt / (ρ · cp · dx)    [K per step]
T_surface -= dT_evap
```

This is a **surface-only** correction — no kernel changes to `fdmAdvDiffKernel`. Implemented as a separate boundary kernel on the z_max face, similar to the existing `fdmRadiationBCKernel` pattern.

### Recoil Pressure (FluidLBM side)

The evaporation recoil pressure normal to the surface:

```
P_recoil = 0.54 · P_sat(T)    [Pa]    (Knight 1979, factor 0.54)
```

Applied as a body force in the FluidLBM via the existing `ForceAccumulator::addRecoilPressureForce()` path. This is already implemented in the codebase — just needs `config.enable_recoil_pressure = true`. The force pushes the surface inward, deepening the keyhole.

### Expected Impact

With evaporation cooling, T_max should saturate at ~3500-4000 K (self-regulating: as T rises above T_boil, q_evap grows exponentially and clamps T). This reduces ∇T by ~20×, bringing Marangoni velocities from ~100 m/s down to ~5-15 m/s — matching OpenFOAM's range.

### Implementation Effort

- `ThermalFDM::applyEvaporationCooling()`: 1 new CUDA kernel (~50 lines), following the existing `ThermalLBM` implementation pattern
- Config: set `enable_recoil_pressure = true`, `enable_evaporation_mass_loss = true` (existing flags)
- Estimated: **1-2 days**

---

## Item 2: Inamuro BC Mach-Number Limiter

**Problem**: delta_f from Inamuro stress BC produces u_LU > 1.0 at the surface, bypassing all FluidLBM internal velocity clamps (U_MAX=0.3).

### Proposed Solution

Replace the fixed `DELTA_F_MAX = 0.04` constant with a **local Mach-aware limiter**:

```cuda
// In applyMarangoniStressBCKernel:
float u_increment = 2.0f * sqrtf(delta_fx*delta_fx + delta_fy*delta_fy);
float Ma_local = u_increment / CS;  // CS = 1/√3 for D3Q19
float Ma_max = 0.2f;  // target surface Mach number

if (Ma_local > Ma_max) {
    float scale = Ma_max / Ma_local;
    delta_fx *= scale;
    delta_fy *= scale;
}
```

This ensures the surface velocity increment per step stays below Ma=0.2, regardless of the temperature gradient magnitude. Combined with Item 1 (evaporation limiting T and thus ∇T), the limiter should rarely activate in practice.

### Alternative: Implicit Stress Coupling

For stronger consistency, the Marangoni stress could be treated semi-implicitly:

```
u_surface^{n+1} = u_surface^n + (dsigma_dT/mu) · ∇_s T · dt
```

by computing the stress from the current T field and applying it as a velocity Dirichlet BC (Zou-He) rather than a distribution perturbation (Inamuro). This avoids the distribution-level instability entirely but requires implementing Zou-He velocity BC at the z_max wall.

### Implementation Effort

- Ma-aware limiter: **2 hours** (modify existing kernel)
- Zou-He alternative: **1 day** (new BC kernel + validation against 1D return flow)

---

## Item 3: Smagorinsky LES / Effective Viscosity

**Problem**: τ_fluid = 0.55 is dangerously close to the τ=0.5 singularity. Numerical noise at small scales amplifies into velocity spikes.

### Smagorinsky Model

The subgrid-scale eddy viscosity:

```
ν_sgs = (C_s · Δ)² · |S̄|
```

where:
- C_s = 0.1-0.2 (Smagorinsky constant, typical 0.1 for wall-bounded flows)
- Δ = dx (filter width = grid spacing)
- |S̄| = √(2·S_ij·S_ij) (strain rate magnitude from resolved velocity field)

The strain rate tensor from the D3Q19 non-equilibrium stress:

```
S_ij = -(ω / 2·ρ·cs²) · Σ_q (c_qi·c_qj · f_q^neq)
```

This is available from the collision kernel's existing computation (the non-equilibrium part `f - f_eq` is already computed during BGK/TRT collision).

### Effective Relaxation Time

The total kinematic viscosity becomes:

```
ν_eff = ν_phys + ν_sgs
τ_eff = 0.5 + 3·ν_eff·dt/dx²
```

Since ν_sgs ≥ 0, τ_eff ≥ τ_phys. In high-shear regions (Marangoni jet), |S̄| is large → ν_sgs provides additional damping → τ_eff moves safely away from 0.5.

### LBM Integration

The D3Q19 EDM collision kernel already supports per-cell ω (through the Darcy coefficient mechanism which modifies the effective τ). The Smagorinsky ν_sgs can be:

1. Computed in a pre-collision kernel from the current non-equilibrium stress
2. Added to the base ν_LU to get ν_eff
3. Passed to the collision kernel as a per-cell relaxation parameter

This is the standard LBM-LES approach (Hou et al., 1996; Teixeira, 1998).

### Expected Impact

At the Marangoni jet center where |S̄| ~ |∂v_x/∂z| ~ 10 m/s / 10μm = 10⁶ s⁻¹:

```
ν_sgs = (0.1 × 2e-6)² × 10⁶ = 4×10⁻⁸ m²/s
```

This is 0.06× ν_phys = 6.33×10⁻⁷ — small compared to molecular viscosity at this resolution. The Smagorinsky model acts as a stabilizer rather than a significant physics modifier. At coarser resolutions (dx=5-10μm), ν_sgs would be more significant.

### Implementation Effort

- Strain-rate-from-neq kernel: **1 day** (standard LBM-LES, well-documented)
- Per-cell τ in collision: already supported via Darcy mechanism
- Validation: lid-driven cavity at Re=1000 with LES vs DNS — **1 day**
- Total: **2-3 days**

---

## Priority & Dependency

```
Item 1 (Evaporation) ──────► Eliminates T_max overshoot
         │                    (root cause of +32% depth error)
         │
Item 2 (Inamuro Ma cap) ──► Prevents velocity spikes
         │                    (secondary stabilizer, less critical with Item 1)
         │
Item 3 (Smagorinsky) ─────► Stabilizes low-τ regime
                              (tertiary, mostly for robustness at coarse grids)
```

**Recommended execution order**: 1 → 2 → 3

Item 1 alone should close 80%+ of the depth gap with OpenFOAM. Items 2 and 3 are stabilization layers for production robustness.

---

## Verification Protocol

After each item, rerun the Marangoni spot melting benchmark and compare:

| Metric | Current | Target (Item 1) | OpenFOAM |
|--------|---------|-----------------|----------|
| T_max at 50μs | 80,000 K | 3,500-4,000 K | 12,000 K |
| v_max at 50μs | 108 m/s | 10-20 m/s | ~15 m/s |
| Depth at 50μs | 66 μm | 45-55 μm | 50 μm |
| Depth at 75μs | 88 μm | 55-65 μm | 60 μm |
| Pool shape | Deep U | V-shaped | V-shaped |
