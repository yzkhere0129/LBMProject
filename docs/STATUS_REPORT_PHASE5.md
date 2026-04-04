# Hybrid LBM-FDM LPBF Simulation Platform — Full Development Report

**Project**: CUDA-based Lattice Boltzmann + Finite Difference hybrid solver for metal additive manufacturing  
**Date**: 2026-04-02  
**Branch**: `benchmark/conduction-316L` (16 commits, 33 source files, +6773 lines)  
**Developer**: Yzk + Claude Opus 4.6

---

## Executive Summary

Over the course of Phase 1–5 development, we built a hybrid LBM-FDM platform capable of simulating laser powder bed fusion (LPBF) processes including keyhole formation. The platform couples a D3Q19 fluid LBM solver with a WENO5 explicit FDM thermal solver, connected by an abstract `IThermalSolver` interface enabling runtime solver selection.

**Key milestone**: First successful simulation of keyhole deep penetration (66μm) in 316L stainless steel at P=150W, v=800mm/s, with physically self-consistent temperature (T_max≈5000K), evaporative mass loss, and recoil-pressure-driven surface deformation.

**Key limitation**: D3Q19 incompressible LBM violates the Mach number constraint (Ma>>1) at the keyhole tip, producing non-physical velocity magnitudes (500–1500 m/s reported vs ~30 m/s physical) and excessive spatter.

---

## 1. Platform Architecture

### 1.1 Module Hierarchy

```
MultiphysicsSolver (orchestrator, ~3000 LOC)
  │
  ├── IThermalSolver (abstract interface, 28 virtual methods)
  │   ├── ThermalLBM (D3Q7 BGK, legacy, for regression)
  │   └── ThermalFDM (WENO5 advection + central-diff diffusion + ESM)
  │         ├── fdmAdvDiffKernel (fused advection-diffusion, VOF-aware)
  │         ├── fdmESMKernel (enthalpy source method for phase change)
  │         ├── fdmEvaporationCoolingKernel (HKL model at surface)
  │         ├── fdmEvapCoolingFromFluxKernel (pre-computed J path)
  │         ├── fdmGasWipeKernel (gas cell temperature reset)
  │         └── weno5.cuh (Jiang-Shu 1996 left/right reconstruction)
  │
  ├── FluidLBM (D3Q19, ~2600 LOC)
  │   ├── fluidBGKCollisionEDMKernel + Smagorinsky LES (Hou 1996)
  │   ├── fluidTRTCollisionEDMKernel + Smagorinsky LES
  │   ├── applyMarangoniStressBCKernel (Inamuro specular, flat wall)
  │   └── smagorinsky_les.cuh (exact algebraic τ_eff formula)
  │
  ├── VOFSolver (PLIC/TVD interface reconstruction + advection)
  │   └── applyEvaporationMassLossKernel (VOF mass sink from J_evap)
  │
  ├── ForceAccumulator (all body forces in physical units → lattice conversion)
  │   ├── addRecoilPressureForceKernel (F = P_recoil × ∇f, Clausius-Clapeyron)
  │   ├── addMarangoniForceKernel (CSF, ×4 compensation for |∇f| deficit)
  │   ├── addSurfaceTensionForceKernel (CSF curvature)
  │   ├── computeDarcyCoefficientKernel (Carman-Kozeny, K=D×dt)
  │   └── convertToLatticeUnits + CFL limiting
  │
  ├── LaserSource (moving Gaussian heat source)
  │   ├── computeLaserHeatSourceKernel (|∂f/∂z| projection + Fresnel)
  │   ├── Dynamic power normalization (P_actual → P_target=η×P)
  │   └── RayTracingLaser (geometric multi-bounce, available but not active)
  │
  └── PhaseChangeSolver (enthalpy method, shared by LBM and FDM)
```

### 1.2 Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| FDM replaces D3Q7 for thermal | D3Q7 TVD limiter kills advection at Ma>0.25 (Pr=0.175 deadlock) |
| WENO5 over SUPERBEE | 5th-order reduces numerical diffusion from 15% to <1% (but physical α dominates) |
| Smagorinsky LES (Hou 1996 algebraic) | Stabilizes τ_f→0.5 at small dt without lagging error |
| IThermalSolver interface | Runtime solver selection preserves backward compatibility |
| Inamuro stress BC for Marangoni (flat wall) | Eliminates CSF 4× force deficit at flat surfaces |
| CSF with ×4 multiplier for Marangoni (VOF) | Compensates discrete |∇f| integral on 2-cell interface |
| F = P_recoil × ∇f (not × n̂) | VOF normals were zero at flat interfaces; ∇f computed directly |
| Dynamic laser power normalization | Ensures exact 52.5W absorption regardless of VOF discretization |
| α_evap = 0.04 (not 0.18) | Allows T to overshoot T_boil so P_recoil > P_Laplace for keyhole |

---

## 2. Development Timeline & Validation

### Phase 1: Pure Conduction Benchmark

**Goal**: Validate thermal diffusion + ESM phase change against OpenFOAM.

**Setup**: 316L, P=150W, r₀=25μm, stationary spot, 50μs ON + 50μs OFF. No fluid, no VOF.

**Result**: LBM solidus isotherm (T=1650K) matches OpenFOAM within the line width at all 4 snapshots (25, 50, 60, 75μs). This proved the ESM enthalpy-porosity formulation is physically equivalent to OpenFOAM's `solidificationMeltingSource`.

**Bug found**: `computeTemperatureKernel` had T_MAX=50000K hard clamp that broke distribution↔temperature consistency at high laser power. Fixed by raising to 1e6.

**Bug found**: BGK source term instability — `g_q += w_q×dT` with dT/T > 10% and ω > 1 causes exponential blowup. Root cause: over-relaxation amplifies non-equilibrium from large source injection. Fix: tau=1.0 (ω=1.0) for the conduction benchmark.

### Phase 2: Marangoni Spot Melt (D3Q7 LBM Thermal)

**Goal**: Add Marangoni surface flow and compare melt pool shape to OpenFOAM.

**Result**: Melt pool was "wide and shallow" — nearly identical to pure conduction despite v_max=62 m/s reported.

**Root cause diagnosed**: The D3Q7 `computeThermalEquilibrium` has a TVD limiter `MAX_ADVECTION=0.9` that clips `cu/cs²` at ±0.9. Since D3Q7 cs²=1/4, this triggers when u_LU > 0.225 (v_phys > 3.3 m/s). **93% of thermal advection was being silently killed.**

**Bug found**: `computeDarcyCoefficientKernel` had `K = D×ρ×dt` instead of `K = D×dt`. Extra ρ=7900 factor caused 7900× over-damping. The Darcy "防弹衣" was the second major cause of shallow pools.

**Inamuro stress BC implemented**: Specular reflection with stress injection at z_max wall, bypassing CSF entirely for flat-wall Marangoni. Validated against 1D return flow (L2=0.989%).

### Phase 3: Hybrid LBM-FDM Architecture

**Goal**: Replace D3Q7 thermal with explicit FDM to eliminate the Pr deadlock.

**Key insight**: For 316L (Pr=0.175), no single dt simultaneously satisfies τ_thermal ∈ [0.6,1.0] and Ma < 0.3 and τ_fluid > 0.52. FDM thermal decouples from the Ma constraint entirely.

**Implementation**:
- `IThermalSolver` abstract interface (28 pure virtual methods)
- `ThermalFDM` class with double-buffered T arrays
- SUPERBEE TVD advection (later upgraded to WENO5)
- ESM phase change operates directly on macroscopic T (no distributions)
- Auto-subcycling: combined 6·Fo + CFL ≤ 1

**Validation**:
- Pure conduction: FDM matches OpenFOAM-validated LBM benchmark (PASS)
- Forced convection (v=5m/s): SUPERBEE gives 23% peak attenuation (15% numerical + 8% physical)
- WENO5 upgrade: 24% → negligible improvement, confirming physical diffusion dominates

**Result**: Marangoni melt pool deepened from 40μm (D3Q7) to 66μm (FDM), overshooting OpenFOAM's 50μm by +32%.

### Phase 4: Smagorinsky LES + Evaporation

**Goal**: Stabilize low-τ fluid LBM and constrain T_max.

**Smagorinsky LES**: Hou (1996) exact algebraic formula τ_eff = 0.5×(τ₀ + √(τ₀² + 18×Cs²×Q_mag/ρ)). Eliminates lagging error critical at τ₀≈0.507. Integrated into both BGK+EDM and TRT+EDM collision kernels. Cs configurable via command line.

**Evaporation cooling**: Hertz-Knudsen-Langmuir model at surface. α_evap reduced from 0.18 to 0.04 to allow T > T_boil (necessary for P_recoil > P_Laplace).

**Parameter sweep**: 12 cases (Cs=[0.05,0.10,0.15,0.20] × dt=[10,15,20]ns). Finding: Cs has negligible effect; dt is the only dominant variable. Width is invariant at 84μm across all cases (thermal diffusion dominated).

**OpenFOAM comparison** (natural HKL evaporation, no hard cap):
- T_max: LBM 10,894K vs OF 12,482K (−13%)
- Depth (50μs): LBM 40μm vs OF 50μm (−20%)
- Width (50μs): LBM 84μm vs OF ~68μm (+24%)

### Phase 5: Full LPBF Keyhole Simulation

**Goal**: Moving laser, VOF free surface, recoil pressure, evaporation mass loss.

**Domain**: 300×150×150μm (150×75×75 cells), 50 metal + 25 gas buffer layers.

**Physics enabled**: VOF PLIC advection, recoil pressure (F=P×∇f), evaporation cooling + mass loss, Marangoni CSF (×4 compensation), surface tension CSF, Darcy mushy zone, Smagorinsky LES, FDM WENO5 thermal, moving Gaussian laser with |∂f/∂z| cos θ projection + Fresnel.

**Bugs found and fixed during Phase 5**:

1. **Energy black hole**: Laser deposited on gas-side interface cells (f<0.5) that were immediately wiped to 600K by gas-reset kernel. Fix: f≥0.5 mask on laser deposition.

2. **F_z(recoil) = 0**: Recoil force used VOF normals (`n` from `reconstructInterface`) which were zero/undefined at flat interfaces. Fix: use `∇f` computed by central differences directly.

3. **Missing cos θ projection**: Laser energy was deposited proportional to |∇f| (full magnitude) instead of |∂f/∂z| (z-component only). This would cause vertical keyhole walls to absorb the same energy as flat surfaces. Fix: Q = q×|∂f/∂z|×Fresnel(θ).

4. **P_recoil < P_Laplace**: At α_evap=0.18, evaporation clamped T at T_boil=3200K where P_recoil=55kPa < P_Laplace=70kPa (σ/r₀). No keyhole could form. Fix: α_evap=0.04 allows T≈4500K where P_recoil=600kPa >> P_Laplace.

5. **VOF ghost refill**: `enable_vof_mass_correction=true` was re-adding evaporated mass back into the domain, causing solidified voids to spontaneously fill with phantom metal. Fix: disabled for evaporation-enabled simulations.

6. **computeEvaporationMassFlux stub**: ThermalFDM had an empty implementation. The recoil-enabled path in MultiphysicsSolver called it, got zero flux, and no mass was ever removed. Fix: implemented full HKL mass flux kernel.

**Final result**: Keyhole forms and reaches 66μm depth at quasi-steady state. T_max = 4500–5500K. Frame-to-frame evolution is continuous. Moving laser leaves a visible (but excessively disrupted) weld track.

---

## 3. Current Keyhole Results

### 3.1 Parameters

| Parameter | Value |
|-----------|-------|
| Material | 316L (constant properties: ρ=7900, cp=700, k=20, μ=0.005) |
| Laser | P=150W, η=0.35, r₀=25μm, v_scan=800 mm/s |
| Grid | dx=2μm, dt=20ns, 150×75×75 cells |
| T_vaporization | 3200K (α_evap=0.04 for overheating allowance) |
| Recoil | C_r=0.54 (Knight 1979), multiplier=1.0 |
| Marangoni | dσ/dT=+1e-4, CSF ×4 compensation |
| Smagorinsky | Cs=0.20 |
| VOF | Mass correction OFF, subcycles=1 |

### 3.2 Quantitative Metrics (t=100μs)

| Metric | Value | Assessment |
|--------|-------|------------|
| Keyhole depth | 66 μm | ✅ Quasi-steady, physically plausible |
| T_max | 4700–5500 K | ⚠️ ~30% above real 316L (~3500K at keyhole bottom) |
| v_max (reported) | 500–1500 m/s | ❌ Non-physical (should be ~30 m/s) |
| P_absorbed | 52.5 W (normalized) | ✅ Exact energy conservation |
| VOF mass | Decreasing (~0.07%/step at T=5000K) | ✅ Physical evaporation |
| Energy balance | ~11% diagnostic error | ⚠️ Diagnostic reads pre-normalization value |
| Wall time | 158s for 100μs (5000 steps) | ✅ Production-usable |
| Stability | Zero crashes in all Phase 5 runs | ✅ Robust |

### 3.3 Qualitative Observations

**What works**:
- Keyhole forms and deepens under recoil pressure
- Laser track moves across domain correctly
- VOF interface deforms (not flat)
- Temperature self-regulates via evaporation
- Mass is lost through evaporation (VOF decreases)
- Frame-to-frame evolution is smooth and continuous

**What doesn't work well**:
- Excessive spatter: liquid metal is ejected too violently, leaving large voids behind the laser instead of a smooth solidified weld bead
- Velocity field is non-physical (Ma >> 1 at keyhole tip)
- Weld track morphology is too disrupted compared to experimental observations

---

## 4. Known Issues — Detailed Analysis

### 4.1 Mach Number Violation (FUNDAMENTAL)

**The core issue**: At dx=2μm, dt=20ns, the lattice speed of sound is c_s = dx/(dt×√3) = 57.7 m/s. The recoil-driven flow at the keyhole tip reaches 30–50 m/s physically, giving Ma = 0.5–0.9. But the D3Q19 equilibrium distribution becomes negative for Ma > 0.82 (u_LU > √(2/3)), and compressibility errors scale as Ma². The solver survives via the U_MAX=10 LU safety clamp and Smagorinsky LES, but the velocity field is quantitatively unreliable.

**Impact**: 
- Velocity magnitude is 10–30× physical → liquid is ejected too violently
- Pressure field has spurious acoustic fluctuations
- Galilean invariance of viscous stress is lost

**Cannot be fixed by**: Parameter tuning, LES, WENO5, or any scheme upgrade to the D3Q19 lattice. The issue is the low-order (second-order) truncation of the Maxwell-Boltzmann equilibrium.

**Possible solutions** (future work):
- Compressible LBM (D3Q27 with extended equilibrium, limited to Ma~0.7)
- Hybrid LBM-FVM: use FVM for the high-Ma keyhole region, LBM for the bulk
- Accept the Ma limitation and use the platform for conduction-mode welding (Ma<0.3) where it is validated

### 4.2 Excessive Spatter Behind Laser

**Symptom**: At t>50μs, the region behind the laser shows large voids (f=0) where metal has been blown away, instead of a smooth solidified weld bead.

**Root cause**: The recoil pressure (physical ~300 kPa at T=4500K) accelerates liquid metal. In the LBM, this acceleration produces v >> c_s (Ma violation). The liquid is ejected at non-physical speed, creating oversized spatter that leaves permanent surface defects.

**This is NOT caused by**:
- Evaporative mass loss (only 0.07%/step, accounts for <1μm of material)
- VOF mass correction (disabled)
- Laser power (correctly normalized to 52.5W)

**Mitigation attempted**: recoil_force_multiplier reduced from 8→1. Improved but did not eliminate.

### 4.3 Temperature Over-Prediction

**Symptom**: T_max = 4500–5500K vs expected ~3500–4000K.

**Root cause**: α_evap was reduced from 0.18 to 0.04 to allow P_recoil > P_Laplace. This is a fundamental trade-off:
- α_evap=0.18 → T_max≈3200K → P_recoil=55kPa < P_Laplace=70kPa → **no keyhole**
- α_evap=0.04 → T_max≈5000K → P_recoil=600kPa >> P_Laplace → **keyhole forms**

In a properly resolved implicit solver (like OpenFOAM), this trade-off doesn't exist because the pressure-velocity coupling handles the recoil-Laplace balance naturally at each implicit iteration.

---

## 5. Complete Bug Fix Registry

| # | Bug | Location | Root Cause | Fix | Phase |
|---|-----|----------|-----------|-----|-------|
| 1 | T_MAX=50000 hard clamp | thermal_lbm.cu | Broke g↔T consistency at high T | Raised to 1e6 | 1 |
| 2 | BGK source instability at ω>1 | thermal_lbm.cu | Over-relaxation amplifies non-eq | tau=1.0 for benchmarks | 1 |
| 3 | Darcy K had spurious ρ factor | force_accumulator.cu | K=D×ρ×dt → 7900× over-damping | K=D×dt | 2 |
| 4 | D3Q7 TVD limiter killed advection | lattice_d3q7.cu | cu/cs²>0.9 clips at u>3.3m/s | Replaced D3Q7 with FDM | 3 |
| 5 | addHeatSource CE correction | thermal_lbm.cu | source_correction=1.0 incorrect | Mitigated by FDM (no CE needed) | 3 |
| 6 | Laser energy on gas-side cells | multiphysics_solver.cu | f<0.5 cells wiped→energy black hole | f≥0.5 mask | 5 |
| 7 | Missing cos θ laser projection | multiphysics_solver.cu | |∇f| instead of |∂f/∂z| | Q=q×|∂f/∂z|×Fresnel | 5 |
| 8 | Recoil F_z = 0 | force_accumulator.cu | Used VOF normals (zero at flat surface) | F=P×∇f directly | 5 |
| 9 | P_recoil < P_Laplace | thermal_fdm.cu | α_evap=0.18 clamped T at T_boil | α_evap=0.04 | 5 |
| 10 | VOF ghost mass refill | benchmark_keyhole.cu | Mass correction re-added evaporated mass | Disabled for evap runs | 5 |
| 11 | computeEvaporationMassFlux stub | thermal_fdm.cu | FDM had empty implementation | Implemented HKL kernel | 5 |
| 12 | Smagorinsky Cs hardcoded | fluid_lbm.cu | 0.1f in kernel call | Configurable member + cmdline | 4 |

---

## 6. File Inventory

### New Files (this branch, 11 source + 3 scripts + 3 benchmarks + 2 docs)

| File | LOC | Purpose |
|------|-----|---------|
| `include/physics/thermal_solver_interface.h` | 120 | Abstract interface for LBM/FDM thermal selection |
| `include/physics/thermal_fdm.h` | 130 | FDM thermal solver class declaration |
| `src/physics/thermal/thermal_fdm.cu` | 780 | FDM kernels: advDiff, ESM, heatSource, BCs, evaporation |
| `src/physics/thermal/weno5.cuh` | 100 | WENO5 left/right face reconstruction (Jiang-Shu 1996) |
| `src/physics/fluid/smagorinsky_les.cuh` | 110 | Smagorinsky LES via Hou (1996) exact algebraic formula |
| `apps/benchmark_conduction_316L.cu` | 560 | Pure conduction benchmark (validated vs OpenFOAM) |
| `apps/benchmark_spot_melt_316L.cu` | 420 | Marangoni spot melt benchmark |
| `apps/benchmark_keyhole_316L.cu` | 240 | Full LPBF keyhole benchmark |
| `apps/diag_marangoni_isolated.cu` | 215 | Diagnostic: frozen-T + probes |
| `apps/test_thermal_fdm.cu` | 270 | FDM validation (conduction + convection) |
| `singlepointbench/contours/plot_marangoni_validation.py` | 195 | Cross-platform contour comparison |
| `singlepointbench/contours/plot_validation.py` | 147 | Conduction validation plots |
| `build/run_sweep.py` | 120 | Automated Cs×dt parameter sweep |
| `docs/roadmap_phase4_physical_constraints.md` | 120 | Phase 4 roadmap |
| `docs/STATUS_REPORT_PHASE5.md` | this file | |

### Modified Files (7 core library files)

| File | Changes | Key Modifications |
|------|---------|-------------------|
| `fluid_lbm.cu` | +112 lines | Smagorinsky in BGK+TRT kernels, Inamuro stress BC, U_MAX=10 |
| `fluid_lbm.h` | +21 lines | applyMarangoniStressBC, cs_smag member |
| `force_accumulator.cu` | +7/−7 lines | Darcy K=D×dt, recoil F=P×∇f |
| `multiphysics_solver.cu` | +80/−15 lines | FDM routing, laser |∂f/∂z|+Fresnel, CSF ×4, evap cooling, power norm |
| `multiphysics_solver.h` | +15 lines | IThermalSolver type, marangoni_csf_multiplier, use_fdm_thermal |
| `thermal_lbm.cu` | +1/−1 line | T_MAX 50000→1e6 |
| `thermal_lbm.h` | +2 lines | `: public IThermalSolver` inheritance |

---

## 7. Performance

| Configuration | Cells | Steps | Wall Time | Cells/sec |
|---------------|-------|-------|-----------|-----------|
| Pure conduction (100³×50) | 500K | 723 | 1.8s | 200M/s |
| Marangoni spot melt (100²×50) | 500K | 2500 | 40s | 31M/s |
| Keyhole full physics (150×75×75) | 843K | 5000 | 158s | 27M/s |
| Keyhole 200μs | 843K | 10000 | ~320s | 26M/s |

GPU: NVIDIA RTX 3050 Laptop (4GB VRAM, SM86). All runs fit in VRAM.

---

## 8. Recommended Next Steps

### 8.1 Short-Term (1–2 weeks)

1. **Diagnose spatter**: Add velocity field visualization (quiver plot on cross-sections) to identify where Ma>0.3 and map the velocity error spatial distribution.

2. **Recoil force capping**: Instead of a global multiplier, cap the per-cell recoil force to produce v_increment < 0.1 LU per step (prevents Ma violation at the source).

3. **Extended weld track**: Run 400μs to capture full single-track with solidification, enabling weld bead morphology comparison.

### 8.2 Medium-Term (1–2 months)

4. **Quantitative validation**: Compare keyhole depth vs Cunningham et al. (2019) synchrotron X-ray data for 316L at matched power/speed.

5. **Powder bed**: Integrate with existing powder bed generator (`generate_powder_bed.py`) for realistic LPBF geometry.

6. **Multi-GPU**: Current domain (843K cells) fits on a single GPU. Production runs (2M+ cells) need multi-GPU via domain decomposition.

### 8.3 Long-Term (3+ months)

7. **Compressible fluid solver**: Replace D3Q19 with either D3Q27 compressible LBM or a hybrid LBM-FVM approach for the keyhole region.

8. **AMR**: Adaptive mesh refinement at the keyhole tip (dx=0.5μm) with coarser mesh (dx=4μm) in the bulk.

---

## 9. Lessons Learned

1. **Low-Pr metals are LBM's Achilles heel**: The Prandtl number deadlock (Pr=0.175 for steel) makes it impossible to simultaneously satisfy thermal stability and Ma constraints with a single D3Q7 lattice. The FDM hybrid solved this cleanly.

2. **Never trust "standard" LBM source terms**: The `source_correction=1.0` in `addHeatSourceKernel` was a documentation-validated "fix" that was actually wrong. The CE correction does affect non-equilibrium modes even if the zeroth moment is exact.

3. **Every diagnostic number must be verified independently**: The "v_max=62 m/s" from Phase 2 was actually 4.34 LU (Ma=7.5) due to unit confusion in `getMaxVelocity()`.

4. **VOF mass correction is dangerous with evaporation**: A conservation mechanism designed for purely advective interface tracking will fight against physical mass loss, creating phantom refills.

5. **P_recoil vs P_Laplace is the keyhole gatekeeper**: Unless recoil pressure exceeds the surface tension restoring force, no amount of numerical effort will produce a keyhole. This is a fundamental physics threshold, not a numerical parameter.

6. **Iterative debugging on GPU is expensive**: Each keyhole run takes 2–5 minutes. A rigorous pre-computation check (dimensional analysis, force balance estimation) before running would have saved many cycles.
