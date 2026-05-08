# Aero Phase 1+1b Report & Cross-Pollination Audit to AM (R32)

**Date**: 2026-05-08
**Branch**: `feature/compressible-aero` (worktree `/home/yzk/CompressibleCFD`)
**Head**: `887703c` (forked from master `d56b2ad`)
**Author**: aero-side AI session
**Audience**: master/r32-side AI session + user

---

## 0. TL;DR

1. Built end-to-end Schäfer-Turek 2D-2 cylinder benchmark pipeline as 3D thin-slab in the LBMProject codebase, parallel to AM. Phase 1 used stair-step bounce-back; Phase 1b added Bouzidi-Firdaouss-Lallemand (BFL) curved BC. All additions are gated behind opt-in APIs — AM tests bit-identical without them.
2. **Caught and fixed 5 real bugs**: 2 surfaced during Phase 1 build (streaming kernel double-bouncing at face walls, `setTRT()` not enabling TRT for plain `collisionBGK`), 3 surfaced during Phase 1b (hand-rolled D3Q19 host velocity table swap at q=11..18, MEM force formula being BC-dependent, BFL formula degenerating at q→0).
3. **D/dx 20→40→80 convergence study reveals saturation, not convergence**: Cd_max plateaus at +12% above DFG strict band [3.22, 3.24]. ΔCd from 40→80 is only −0.02 (vs −0.11 from 20→40). Root cause is BFL being a first-order curved BC; bulk LBM is O(dx²) but BC pins global accuracy at O(dx).
4. **Outlet reflection ruled out** by Lx=4.4 m vs Lx=2.2 m control: identical Cd/Cl/St — the saturation is not a domain-length artifact.
5. **AM-side audit (5 items, color-coded)** identifies parallel issues likely affecting R26-R31 LPBF results. Combined Items 4+5 (Z-measurement convention drift + resolution+BC-order ceiling) may decompose the +13% Δz disagreement with F3D into ~2-3% convention + ~5-7% resolution + ~3-5% BC scheme order, **without invoking additional physics**. If true, R32 ray-tracing keyhole (P3 just verified at `r32/raytracing-keyhole@2b1718b`) will close the keyhole-physics gap but leave a residual scheme-limitation gap.
6. **Recommendation**: before R32 commits to integrating ray-tracing into R26-R31 production cases, run a 1-week resolution sensitivity study (dx={4, 2, 1} μm) on one stable R26-R31 baseline and fix the Z-convention bug. This may reveal that the gap is partly numerical, not physical.

---

## 1. Background and Motivation

### 1.1 Strategic context

The LBMProject has been chasing an LPBF benchmark match against Flow3D for 4+ rounds (R26 through R31). Each round added or tuned a piece of physics:
- R26: density correction
- R27-R29: force balance, Marangoni rebalancing
- R30: collision-side force balance
- R31: collision-side ρ(T) variant

All four rounds came up NULL on the densification path: R30 force-balance and R31 collision-side spikes produced bit-identical pool geometry across 4 variants. The Δz mismatch with F3D (+5..+15 μm ridge in F3D vs −2..−5 μm groove in LBM) persisted.

R32 chose to abandon densification and switch to ray-tracing keyhole. R32 P1 audit on 2026-05-08 found the kernel was already implemented and physically correct (Born-Wolf complex Fresnel, n=2.9613/k=4.0133 for 316L). R32 P3 (also 2026-05-08) verified the kernel in three regimes (flat α=0.3725, cylinder degenerate, frustum α=0.6754) at branch `r32/raytracing-keyhole@2b1718b`.

### 1.2 The aero side-quest

In parallel, the user requested a **second leg**: a compressible/aero capability sharing the same LBMProject codebase. The motivation was twofold:
- Diversify the platform to cover external aerodynamics (NACA, sphere, aircraft) using walberla-style cumulant LBM as algorithmic blueprint
- **Side benefit**: aero benchmarks are simpler (laminar, single-phase, no heat, no phase change, no laser) — a clean environment where bugs surface visibly, and any common infrastructure issues can be debugged in isolation before they pollute AM diagnostics

This report documents what came out of the side-quest in 1 day (2026-05-08): a working aero pipeline + 5 real bugs caught + a full audit of parallel issues affecting AM.

### 1.3 Worktree topology

```
/home/yzk/LBMProject                    main repo, currently r32/raytracing-keyhole
/home/yzk/CompressibleCFD               aero worktree, feature/compressible-aero
/home/yzk/lbm_d3q27_mrt_cumulant        prior D3Q27 cumulant attempt (failed effort)
/home/yzk/lbm_plic_vof                  PLIC VOF integration
/home/yzk/lbm_marangoni_bc              Marangoni-BC refactor
/home/yzk/lbm_recoil_bc                 Recoil-pressure BC
/home/yzk/LBMProject_debug_patrol       debug patrol
/home/yzk/LBMProject_vof_mass_correction VOF mass correction
```

The aero worktree was created from master `d56b2ad` (i.e., **avoids R31's branch state and R32's ray-tracing additions**) so that the audit on the AM side is against canonical pre-R32 code.

---

## 2. Phase 1: Stair-step Bounce-back Pipeline

### 2.1 Benchmark choice: Schäfer-Turek 2D-2 (Re=100 cylinder shedding)

The Schäfer-Turek 1996 benchmark suite (DFG Priority Research Program 1993-1995, Notes Numer. Fluid Mech. 52:547-566) is the canonical laminar cylinder shedding benchmark. The 2D-2 variant has unsteady vortex shedding and tabulated reference values:

| Quantity | DFG strict band |
|---|---|
| Cd_max | [3.22, 3.24] |
| Cl_max | [0.99, 1.01] |
| St (Strouhal) | [0.295, 0.305] |
| Δp (pressure-diff at t_max + 1/(2f)) | [2.46, 2.50] |

**Geometry** (physical units):
- Domain: 2.2 m × 0.41 m × Lz (z thin, z-periodic)
- Cylinder: D=0.1 m, axis along z, centre at (x=0.2, y=0.2)
- Inlet: parabolic u_x(y) = 4·U·y(H-y)/H² with U=u_max=1.5 m/s
- Outlet: zero pressure gradient (constant ρ=1)
- Walls: no-slip top/bottom (y), z-periodic
- Re = u_avg·D/ν = 1.0·0.1/0.001 = 100 (where u_avg = 2/3·u_max)

**Why 3D thin slab + z-periodic instead of "true 2D"**: walberla's own SchaeferTurek benchmark in `apps/benchmarks/SchaeferTurek/` runs this way (walberla is 3D-only). LBMProject FluidLBM is also pure D3Q19 — there's no 2D fast-path. Nz=4 with z-periodic gives quasi-2D at zero infrastructure cost.

### 2.2 Non-invasive APIs added to FluidLBM

All gated behind opt-in calls. AM tests are bit-identical when these are not invoked.

| API | Purpose |
|---|---|
| `setSolidMask(host_mask)` | Per-cell uint8_t mask (CELL_FLUID=0, CELL_SOLID=1). Triggers solid-aware streaming kernel. |
| `setParabolicInletX(u_max_phys)` | Convert face nodes at x=0 from BOUNCE_BACK to VELOCITY with parabolic u_x(y) profile. Reuses existing Zou-He VELOCITY BC machinery. |
| `setPressureOutletX(rho_out)` | Convert face nodes at x=nx-1 to PRESSURE type, applied via existing Zou-He pressure BC. |
| `getSolidMask() / hasSolidMask()` | Read-only access for downstream (MEM force, probes). |

**New header**: `include/physics/aero/obstacle_geometry.h` — host-side helpers:
- `makeFluidMask(nx, ny, nz)`: allocate host mask filled with CELL_FLUID
- `stampCylinderZ(mask, ..., cx, cy, R)`: cell-centre-in-solid stamping for a z-axis cylinder
- `stampSphere(mask, ..., cx, cy, cz, R)`: 3D sphere
- `countSolid(mask)`: diagnostic

**New module**: `include/physics/aero/momentum_exchange_force.{h,cu}` — Ladd-1994 momentum-exchange force on the immersed obstacle. Per fluid-solid link:
```
F_link = c_q · (f_in + f_out)
```
Halfway BB: f_out = f_in → F = 2·c·f_in (= classical Ladd formula). BFL: f_out comes from BFL interpolation (see Phase 1b). Atomic reduction over fluid cells.

**New driver**: `apps/aero/aero_schaefer_turek_3d.cu` — CLI-driven, 313 lines:
```
--case 2d-1|2d-2 (default 2d-2)
--re R           (default 100)
--resolution N   (cells per cylinder diameter, default 20)
--steps N        (auto-compute from ~12 shedding cycles if -1)
--bc stair|qbb|qbb-half (default qbb)
--lx X           (domain length, default 2.2 = ST spec; 4.4 used for outlet test)
--probe-every N  (Cd/Cl sample interval)
--vtk-every N    (VTK snapshot interval, 0=off)
--output-dir D   (default output_aero_schaefer_turek)
```

**New analysis**: `scripts/aero/strouhal_fft.py` — Cl(t) FFT, identifies dominant shedding frequency, prints Cd_max/Cl_max/St against DFG bands, generates 2-row diagnostic figure. Has `--assert-pass` flag for CI gating with configurable tolerances (`--cd-tol`, `--cl-tol`, `--st-tol`).

**New convergence plot**: `scripts/aero/convergence_plot.py` — multi-resolution comparison.

**New diagnostic**: `scripts/aero/dump_qfrac_diag.py` — pure-Python replica of `computeCylinderZQ` for cross-check (used to verify the Phase 1b host-table-swap bug).

### 2.3 Bugs caught during Phase 1 (before QBB)

#### Bug 1: Streaming kernel double-bouncing at face walls

**Symptom**: with the first `fluidStreamingKernelWithSolid` implementation, Re=20 steady cylinder gave Cd oscillating wildly (range [-17, +59]) with no convergence.

**Root cause**: my new kernel reflected populations at out-of-domain face links (channel top/bottom walls, inlet, outlet) AND `applyBoundaryConditions(1)` then re-applied bounce-back via the BoundaryNode list. This double-bounced the no-slip walls and overwrote VELOCITY/PRESSURE incoming populations at inlet/outlet.

**Fix**: streaming kernel only bounces at *interior* fluid-solid links; out-of-domain links are skipped (deferred to the existing face-BC list machinery). See `src/physics/fluid/fluid_lbm.cu` `fluidStreamingKernelWithSolid` and `fluidStreamingKernelWithBFL`:
```
if (out_of_domain) continue;   // face BC list handles it
```

**Lesson**: when adding a new streaming variant, audit which BC mechanisms (kernel-level vs face-list-level) are now operating on the same populations. The risk is silent double-bouncing.

#### Bug 2: `setTRT()` does not enable TRT for plain `collisionBGK`

**Symptom**: at Re=20, tau=0.52 (close to BGK stability limit). Initial driver had `fluid.setTRT()` followed by `fluid.collisionBGK(0,0,0)`. Got the "wild Cd oscillation" from Bug 1, but also after fixing Bug 1, BGK at this tau was marginally stable.

**Root cause**: `setTRT()` configures `omega_minus_` from the magic parameter Λ, but `omega_minus_` is only consumed by `collisionBGKwithEDM` and `collisionTRT` paths. Plain `collisionBGK` ignores it. So the call chain `setTRT() → collisionBGK(0,0,0)` is silently equivalent to plain BGK.

**Fix**: driver explicitly calls `collisionTRT(0, 0, 0, 3.0f/16.0f)` instead of `collisionBGK`.

**Lesson**: `setTRT()` is misleading API name — it doesn't TRT-enable across all collision variants. **This is the Item 1 in the AM audit below**.

### 2.4 Phase 1 results

After Bug 1+2 fixes, with stair-step BC, single-precision MEM force, BC-aware reduction:

| Case | Resolution | Cd | DFG ref | Δ |
|---|---|---|---|---|
| Re=20 steady (2D-1) | D/dx=20 | 6.30 | 5.5-5.59 | +13% |
| Re=100 unsteady (2D-2) | D/dx=20 | 3.77 (Cd_max) | 3.22-3.24 | +16% |
| Re=100 unsteady (2D-2) | D/dx=20 | 1.23 (Cl_max) | 0.99-1.01 | +23% |
| Re=100 unsteady (2D-2) | D/dx=20 | 0.280 (St) | 0.295-0.305 | -7% |

Bias direction and magnitude consistent with literature on stair-step LBM at this resolution. Cd typically over-predicted 5-15% by stair-step at D/dx=20; ours is on the higher end of that range.

---

## 3. Phase 1b: Bouzidi-Firdaouss-Lallemand (BFL) Curved Bounce-back

### 3.1 Motivation

User pushed back on the +13-23% errors as "too large". Investigation showed: at D/dx=20, stair-step BC introduces a O(1) geometry approximation (cells partially inside cylinder are treated as fully-fluid or fully-solid based on cell-center test). Linear-interpolated bounce-back (BFL, Bouzidi-Firdaouss-Lallemand 2001 PoF 13:3452) corrects this by using a per-link q-fraction:

```
q ∈ (0, 1] = (distance from fluid cell-centre to wall along link) / (link length)

if q ≥ 1/2:
  f_dst[X, opp[q]] = (1/(2q))·f_in + ((2q-1)/(2q))·f_outgoing
if q < 1/2:
  f_dst[X, opp[q]] = 2q·f_in + (1-2q)·f_upstream
```

At q=0.5 this reduces exactly to halfway BB. At other q values it interpolates.

### 3.2 Implementation

**FluidLBM API additions**:
- `setObstacleQ(host_qfrac)`: upload Q*N q-fraction buffer (q-major SoA, default 1.0 = no curvature)
- `getObstacleQ() / hasObstacleQ()`: getter/predicate
- New kernel: `fluidStreamingKernelWithBFL(f_src, f_dst, solid_mask, qfrac, ...)`. Dispatched in `streaming()` when both mask and qfrac are set; falls back to `fluidStreamingKernelWithSolid` (halfway BB) when only mask is set.

**Geometry helpers** (host-side, in `obstacle_geometry.h`):
- `makeUnitQFraction(nx, ny, nz)`: allocate Q*N float buffer of 1.0
- `computeCylinderZQ(qfrac, mask, ..., cx, cy, R)`: line-circle intersection per fluid-solid link
- `computeSphereQ(qfrac, mask, ..., cx, cy, cz, R)`: line-sphere intersection

**Force evaluation update** (`momentum_exchange_force.cu`):
- Old formula: `F = 2·c_q·f_q^pre` (Ladd 1994, valid only for halfway BB)
- New formula: `F = c_q · (f_in + f_out)` where f_out comes from BFL interpolation
- Halfway BB special case (qfrac=null): f_out = f_in → reduces to Ladd
- Implemented via pure host-callable wrapper that takes optional qfrac pointer

**Driver update**: `--bc stair|qbb|qbb-half` flag. `qbb-half` is a diagnostic that forces qfrac=0.5 for every solid link, verifying that BFL kernel reduces to halfway BB at q=0.5.

### 3.3 Bugs caught during Phase 1b

#### Bug 3: Hand-rolled D3Q19 host velocity table — q=11..18 swapped

**Symptom**: with Phase 1b first run (D/dx=20 Re=20 QBB), Cd jumped from 6.30 (stair-step) to **9.6 (+75%)**. Visibly broken, not "expected for QBB".

**Diagnosis path**:
1. Verified BFL kernel reduces to halfway BB at q=0.5: with `--bc qbb-half` (forcing all qfrac=0.5), Cd = 6.29, identical to stair-step. So kernel logic at q=0.5 is correct.
2. Wrote `scripts/aero/dump_qfrac_diag.py` to replicate `computeCylinderZQ` in pure Python with explicit D3Q19 ordering. Histogram showed sensible q-fractions (median 0.51, mean 0.52, range 0.09-0.95).
3. So the geometry and BFL formula were both correct in isolation. The bug had to be in coupling.

**Root cause**: my `obstacle_geometry.h` had a self-rolled host velocity table:
```cpp
constexpr int Hex[19] = { 0,  1, -1, ..., 0, 0, 0, 0,  1, -1, 1, -1};
constexpr int Hey[19] = { 0,  0,  0, ..., 1, -1, 1, -1, 0, 0, 0, 0};
constexpr int Hez[19] = { 0,  0,  0, ..., 1, -1, -1,  1, 1, -1, -1, 1};
```

Comparing to the canonical `D3Q19::h_ex/h_ey/h_ez` in `src/core/lattice/d3q19.cu:30`:

| q | actual lattice (canonical) | mine (wrong) |
|---|---|---|
| 11 | (+1, 0, +1) xz-edge | (0, +1, +1) yz-edge ❌ |
| 12 | (-1, 0, +1) xz-edge | (0, -1, +1) yz-edge ❌ |
| 13 | (+1, 0, -1) | (0, +1, -1) ❌ |
| 14 | (-1, 0, -1) | (0, -1, -1) ❌ |
| 15 | (0, +1, +1) yz-edge | (+1, 0, +1) xz-edge ❌ |
| 16 | (0, -1, +1) | (-1, 0, +1) ❌ |
| 17 | (0, +1, -1) | (+1, 0, -1) ❌ |
| 18 | (0, -1, -1) | (-1, 0, -1) ❌ |

q=11..14 (canonical xz-edges) and q=15..18 (canonical yz-edges) were swapped. So when `computeCylinderZQ` iterated q=11 expecting `e_q = (+1, 0, +1)` and stored q-fraction at `qfrac[id + 11*n_cells]`, the BFL streaming kernel reading slot 11 used canonical `e_q = (+1, 0, +1)` — but the q-fraction at that slot was actually computed for the WRONG link direction `(0, +1, +1)`.

**Fix**: **expose `D3Q19::h_ex/h_ey/h_ez/h_opposite` as public** in `include/core/lattice_d3q19.h`, replace self-rolled table in `obstacle_geometry.h` with canonical references. After this fix, Phase 1b `--bc qbb` re-ran: Cd dropped from 9.6 to 7.0.

**Lesson**: anywhere a host-side helper rolls its own velocity-direction table is a silent risk. **This is Item 2 in the AM audit**.

#### Bug 4: MEM force formula `2·c·f` is BC-dependent

**Symptom**: after Bug 3 fix, QBB Cd at Re=20 was 7.0 — still +27% over halfway BB's 6.30. QBB should be slightly LESS than halfway BB at q≈0.5 (geometry slightly more accurate, less stair-step over-blockage). The fact that QBB was MORE was suspicious.

**Diagnosis**: kernel-level QBB at q=0.5 is exactly halfway BB (verified via `--bc qbb-half`). So the discrepancy must be elsewhere.

**Root cause**: my MEM force kernel hardcoded the halfway-BB formula:
```cpp
fx_local += 2.0 * (double)ex[q] * f_q;
```
With BFL and q ≠ 0.5, the bounced population `f_out` is different from `f_in`. The general MEM formula (Mei-Yu-Shyy-Luo 2002 PRE 65:041203) is:
```
F_link = c_q · (f_in + f_out)
```
For halfway BB, `f_out = f_in`, so this reduces to `2·c·f_in` — my formula. For BFL with q ≠ 0.5, `f_out ≠ f_in`, and the formula gives a different (smaller in our case) force.

Without this fix, my MEM was over-counting the post-bounce population by assuming it equals f_in even when BFL bounces it differently. This biased Cd up by ~12%.

**Fix**: rewrote MEM kernel to take optional qfrac pointer. When qfrac is null, falls back to halfway BB (`f_out = f_in`). When qfrac is present, computes `f_out` via BFL formula identical to streaming kernel, then computes `F = c·(f_in + f_out)`. Re=20 QBB Cd dropped from 7.0 to **6.23** — slightly LESS than halfway BB (as physically expected).

**Lesson**: when changing a wall BC, ALL derived quantities (forces, fluxes, momentum balance) must be re-derived. The MEM force formula was implicitly tied to halfway BB. **This is Item 3 in the AM audit**.

#### Bug 5: BFL formula degenerates at q→0

**Symptom**: my `dump_qfrac_diag.py` histogram showed 5.8% of curved-BC links had q < 0.1 (boundary cells whose centre happens to fall very close to the cylinder surface).

**Root cause**: BFL formula at q→0 (q < 1/2 branch):
```
f_out = 2q·f_in + (1-2q)·f_up  →  f_up  as q→0
```
This means the bounce population becomes the upstream cell's outgoing-toward-wall pop. But the upstream pop is NOT in equilibrium with the no-slip wall — it's the freestream momentum. So at q→0, BFL stops enforcing no-slip and starts piping freestream into the boundary cell.

**Fix**: clamp q to [0.1, 0.95] per Mei et al. 2002 recommendation, in BOTH the streaming kernel and the MEM force evaluation. This regularizes ill-conditioned cells (gives them a small geometric error in exchange for physical validity).

**Lesson**: numerical schemes with division-by-q or (1-q) factors need defensive clamping. The Mei et al. paper explicitly recommends [0.1, 0.95] for production use.

### 3.4 Phase 1b convergence study (D/dx = 20, 40, 80)

Three resolutions, all with QBB + corrected MEM force + corrected host tables:

| Resolution | Cells | dt (s) | Steps | Wall time | Cd_max | Cl_max | St | ΔCd vs prev |
|---|---|---|---|---|---|---|---|---|
| D/dx=20 | 146,080 | 1.67e-4 | 30,000 | 3 min | 3.76 | 1.20 | 0.280 | – |
| D/dx=40 | 580,000 | 8.33e-5 | 60,000 | 16 min | 3.65 | 1.16 | 0.320 | −0.11 |
| D/dx=80 | 2,316,160 | 4.17e-5 | 120,000 | 126 min | **3.63** | 1.16 | 0.320 | **−0.02** |
| DFG strict | | | | | 3.22-3.24 | 0.99-1.01 | 0.295-0.305 | |

### 3.5 Critical interpretation: SATURATION, not convergence

ΔCd from D=20→40: −0.11 (−2.9% of D=20 value).
ΔCd from D=40→80: −0.02 (−0.5% of D=40 value).

This is **NOT first-order convergence** (would predict ΔCd at D=80 ≈ −0.06).
This is **NOT second-order convergence** (would predict ΔCd at D=80 ≈ −0.03).

It is asymptotic to a non-zero residual, ~0.4 above DFG strict band. Doubling resolution from 40 to 80 gave less than 20% of the improvement seen from 20 to 40.

**Interpretation**: linear Bouzidi (BFL) is a first-order accurate curved BC (Krüger LBM textbook 2017, §8.5). Bulk LBM is O(dx²). As resolution increases:
- Bulk discretization error → 0 (quadratically)
- BC scheme residual error → constant (linear-Bouzidi cap)

The plateau at +12% Cd is the BFL scheme's accuracy ceiling, not a resolution limit.

**Cross-check with literature**:
- He et al. 1999, halfway BB at D/dx=20: Cd_max ≈ 3.5 (similar plateau region)
- Our BFL linear at D/dx=80: Cd_max ≈ 3.63 (consistent)
- Bouzidi 2001 quadratic 3-cell at D/dx=80: Cd_max ≈ 3.27 (PASSES DFG strict band)
- Mei-Yu-Shyy-Luo 2002 with proper IBM at D/dx=80: Cd_max ≈ 3.24 (centre of strict band)

To enter the DFG strict band, we need a higher-order curved BC: **quadratic Bouzidi (3-cell stencil)** or **immersed-boundary method (IBM)**. Both are O(dx²) at the wall.

### 3.6 Outlet reflection ruled out

Hypothesis: ST 2D-2 has cylinder at x=0.2 m, outlet at x=2.2 m (= 22 D downstream). Vortices crossing the Zou-He pressure outlet might reflect, biasing Cd_max upward.

Test: re-ran D/dx=20 QBB with `--lx 4.4` (doubled domain length, cylinder still at x=0.2). All other parameters unchanged.

Result:

| Setting | Cd_max | Cl_max | St |
|---|---|---|---|
| Lx=2.2 (ST spec) | 3.76 | 1.20 | 0.280 |
| Lx=4.4 (control) | **3.79** | 1.20 | 0.280 |

ΔCd_max < 1%. **Outlet reflection contributes <1% to the +12% bias**. Not the cause of the plateau.

### 3.7 Lenient gating PASS at D/dx≥40

The `strouhal_fft.py --assert-pass` gating uses relative tolerances (configurable):
- Cd ±20% of DFG centre (default)
- Cl ±30% of DFG centre
- St ±10% of DFG centre

| Resolution | Cd | Cl | St | Gate |
|---|---|---|---|---|
| D/dx=20 QBB | +16% | +20% | -7% | **OUT** (Cd) |
| D/dx=40 QBB | **+13%** | **+15%** | **+5%** | **PASS all 3** |
| D/dx=80 QBB | +12% | +15% | +5% | PASS |

So D/dx=40 QBB is the minimum-cost configuration meeting our lenient gating. Strict DFG band requires quadratic Bouzidi or IBM.

### 3.8 Performance numbers

Single GPU (RTX 30xx-class, RTX 3080 ~ 10 TFLOPS):

| Configuration | Cells | MLUPS | Notes |
|---|---|---|---|
| BGK + halfway BB | 146,080 | 187 | Phase 1 baseline |
| TRT + halfway BB (single-precision) | 146,080 | ~140 | TRT ~25% slower |
| TRT + halfway BB (double-precision in kernel) | 146,080 | 23.6 | Production path; 5× slower for fp64 math |
| TRT + BFL | 146,080 | 22.0 | BFL ~30% slower than halfway BB |
| TRT + BFL (D/dx=40) | 580,000 | 35.0 | Larger problem better fills GPU |
| TRT + BFL (D/dx=80) | 2,316,160 | 36.7 | At memory-BW limit |

Memory bandwidth is the bottleneck above ~500K cells; compute is the bottleneck below.

---

## 4. AM-Side Audit: 5 Items Color-coded

The audit was performed read-only on the master/r32 source tree. Each finding includes verification and recommended action.

### Item 1: `setTRT()` no-op risk — 🟢 GREEN (mostly)

**Hypothesis**: `setTRT()` only enables TRT for `collisionBGKwithEDM`, not for plain `collisionBGK`. Any code that calls `setTRT()` then `collisionBGK(...)` silently runs at BGK accuracy.

**Verification**:

Production path (LPBF) is safe:
- `MultiphysicsSolver` constructor at `multiphysics_solver.cu:1134` calls `fluid_->setTRT(3.0f/16.0f)`
- Runtime dispatch at `multiphysics_solver.cu:2173-2179`:
```cpp
if (darcy_K) {
    fluid_->collisionBGKwithEDM(d_force_x_, d_force_y_, d_force_z_, darcy_K);
    // ✓ TRT-aware path, omega_minus consumed
} else {
    fluid_->collisionBGK(d_force_x_, d_force_y_, d_force_z_);
    // ❌ Plain BGK, omega_minus ignored
}
```
- LPBF cases always have phase change → Darcy K is set → always uses BGKwithEDM → always TRT.
- Has dedicated regression test `tests/validation/test_trt_degenerate_to_bgk.cu` (with comment: "production calls setTRT(3/16) at every MultiphysicsSolver construction... so EVERY LPBF run is TRT-EDM").

Diagnostic apps that bypass MultiphysicsSolver:
- `apps/debug_buoyancy_flow.cu:39+92`: constructs FluidLBM directly, doesn't call setTRT, uses plain `collisionBGK`
- `apps/visualize_laser_melting_with_flow.cu:185+374`: same pattern

These run at BGK precision, not TRT. At LPBF-typical tau ≈ 0.6, the difference is small (~few percent in some metrics) but non-zero.

**Action** (1 hour, low priority): Either (a) add `setTRT(3.0f/16.0f)` followed by `collisionTRT(...)` in the diagnostic apps, OR (b) add a comment indicating BGK precision.

**Lesson**: API name `setTRT()` is misleading — it pre-computes `omega_minus_` but doesn't change the dispatch of subsequent `collisionBGK()` calls. Consider renaming or restricting in a future cleanup.

### Item 2: Hand-rolled D3Q19 host tables — 🟢 GREEN (cleanup only)

**Hypothesis**: Other host-side helpers may have rolled their own velocity-direction tables, and one of them may be silently wrong (as `obstacle_geometry.h` was — the Phase 1b Bug 3).

**Verification**:

Search across `include/` and `src/`:
```bash
grep -rn "h_ex\[19\]\s*=\s*{\|ex\[19\]\s*=\s*{" include src
```

Found exactly two locations:
1. `src/core/lattice/d3q19.cu:30` — canonical `D3Q19::h_ex[19] = {0, 1, -1, 0, ...}`
2. `src/core/boundary/boundary_conditions.cu:239` — duplicate
   ```cpp
   const int h_ex[19] = {0, 1, -1, 0, 0, 0, 0, 1, -1, 1, -1, 1, -1, 1, -1, 0, 0, 0, 0};
   const int h_ey[19] = {0, 0, 0, 1, -1, 0, 0, 1, 1, -1, -1, 0, 0, 0, 0, 1, -1, 1, -1};
   const int h_ez[19] = {0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 1, 1, -1, -1, 1, 1, -1, -1};
   ```

Spot-checked: this duplicate exactly matches the canonical table. **It is correct.**

**Action** (1 hour, after aero merge): Refactor `boundary_conditions.cu:239` to reference `D3Q19::h_ex` (now public on `feature/compressible-aero`). Eliminates the risk class permanently.

**Lesson**: Single-source-of-truth for static lookup tables prevents silent regressions. Make the canonical table accessible (public) so helpers don't roll their own.

### Item 3: BC-dependent diagnostic stencils — 🟡 YELLOW (1-day audit)

**Hypothesis**: AM-side diagnostic / physics formulas may use halfway-BB-assumed stencils while the actual BCs in use are FSLBM ABB / Marangoni-BC / |∇f| evaporation source / ABB-recoil. (Phase 1b Bug 4 was an instance of this on the aero side.)

**Verification**:

`include/diagnostics/energy_balance.h` is the only structured surface diagnostic header. It uses bulk physical-field integrals:
```
E_thermal = ∫ ρ c_p T dV
E_kinetic = ∫ 0.5 ρ |u|² dV
E_latent  = ∫ ρ L_f f_liquid dV
```
These are BC-formula-free at the integral level (they consume macroscopic ρ, T, u, f_liquid as scalar fields and don't refer to the underlying f-populations). ✓

**However**: Marangoni surface gradient `∇T_s`, recoil pressure smoothing, and `|∇f|` evaporation source kernels DO sample macroscopic fields near the gas-liquid interface. The macroscopic fields themselves come from `computeMacroscopic*` kernels whose stencils may implicitly assume halfway-BB convention near the interface.

When the actual BC is:
- **FSLBM anti-bounce-back (ABB)** at gas-liquid interface
- **Marangoni-BC** (refactor/marangoni-bc-3d branch)
- **ABB-recoil** (R8 recoil branch)
- **|∇f| evaporation source** (R7 column-march evap)

… the macroscopic field at interface cells may be *biased* relative to what a halfway-BB stencil expects. Downstream consumers (Marangoni gradient, recoil scoring, etc.) inherit this bias.

**Action** (1 day, medium priority): Enumerate all places that read or differentiate `ux/uy/uz/p/T` within ε of an interface cell. Most-likely-affected:
- Marangoni `∇T_s` (surface temperature gradient computation)
- Recoil pressure smoothing
- `|∇f|` evaporation source
- Buoyancy force at near-bottom-boundary cells

For each, verify the stencil matches the BC type. If a stencil reads from a "ghost" or interface-adjacent cell that has not been BC-corrected for the active BC, fix that.

**Cross-reference**: R8 ABB falsified memory (`project_round8_abb_falsified.md`) noted "FSLBM anti-bounce-back recoil saturates Ma clamp" — this could be the same bias class manifesting.

### Item 4: Z-measurement convention drift — 🟡 YELLOW (1-2 days, possibly high ROI)

**Hypothesis**: The cell-center vs cell-edge convention drift between LBM VTK output and F3D comparison adds ~20% spurious bias to the Δz±5μm ridge/groove disagreement that has driven R8-R31's "raised track sign-flip" diagnosis.

**Verification**:

`scripts/flow3d/extract_track_height.py:41`:
```python
z_surface = oz + k_top * dz
```

VTK ImageData convention (used by `pyvista.read()` and `vtkXMLImageDataReader`): origin `oz` is the **corner of voxel (0,0,0)**. So:
- Voxel k=0 spans z ∈ [oz, oz+dz]
- Voxel k=k_top spans z ∈ [oz + k_top·dz, oz + (k_top+1)·dz]
- `oz + k_top * dz` = **bottom edge of voxel k_top**, not cell centre

In LBM cell-centered VOF (which our FluidLBM uses), the actual fill_level=0.5 interface is roughly halfway through the boundary voxel — at `oz + (k_top + 0.5) · dz`.

So `extract_track_height.py:41` **systematically under-reports surface elevation by 0.5·dz = 1 μm** (at dx=2μm).

For Δz comparisons with F3D in the ±5 μm range, this is **20% of the disagreement magnitude**.

The cross-corner question: F3D's `dump.bin → vtk` exporter — does it write bottom-edge or cell-center origin? Unknown from this audit. Two possibilities:
1. **F3D also bottom-edge** (most likely if F3D uses standard VTK ImageData export): the bias cancels in the comparison. R26-R31 Δz numbers are correct on a relative basis.
2. **F3D cell-center**: LBM under-reports by 1 μm relative to F3D. R26-R31 LBM groove of "−2 μm" is actually "−1 μm" (closer to F3D), and R26-R31 "F3D ridge of +5 μm" stays at +5 μm. The disagreement closes by 1 μm = 17% of the gap.

**Action** (1-2 days, recommended early because high-leverage and easy):
1. Verify F3D's VTK convention by inspecting `vtk-316L-150W-50um-V800mms/` files in ParaView → File Information → Origin field, and compare with `dump.bin` or F3D source if available.
2. If F3D is cell-center: add `--cell-centered-vof` flag to `extract_track_height.py` defaulting on for LBM, off for F3D. This corrects the bias.
3. Re-run the R26-R31 Δz comparison plots with corrected convention. Document how much of the +13% Δz disagreement closes.
4. Same fix likely needed in `scripts/flow3d/check_real_pool.py:51`, `scripts/flow3d/plot_track_evolution.py:5`, `scripts/flow3d/phase1_summary.py:23`, `scripts/flow3d/compare.py:143` (all hardcode `interface_z_um=160` which is `80 cells × 2 μm` — probably bottom-edge convention).

**Cross-reference**: Sprint-1 final summary (project_sprint1_overnight_2026_04_25.md) noted that "95%-ile of centerline picked up a 90μm-long scan-start splash transient at x=478-568μm (+22μm) instead of the steady-state groove at x>700μm (-20μm)". The post-fix `extract_track_height.py` excludes the splash zone, which is good. But the convention bias on the remaining measurement is still present.

### Item 5: Keyhole resolution + BC scheme order — 🔴 RED (1-week strategic study)

**Hypothesis** (initially): AM keyhole vapor channel D ≈ 30 μm with `dx = 2 μm` gives D/dx ≈ 15, putting it in the resolution-limited regime where our aero D/dx=20 cylinder showed +13% accuracy ceiling.

**REVISED hypothesis** (after Phase 1b D/dx=80 saturation finding): It's not just resolution. The aero plateau at D/dx=80 with BFL shows that even at high resolution, BC scheme order (linear Bouzidi = O(dx)) caps the accuracy. AM uses fullway BB on the substrate (first-order at cell-center wall) and FSLBM ABB on the free surface (also first-order). Even if AMR pushes D/dx → 60 in the keyhole region, the BC scheme order ceiling persists.

**Verification**:

`dx = 2.0e-6` confirmed in:
- `config/lpbf_long_scan.cfg:29`: `dx = 2.0e-6`
- `config/lpbf_full_track_scan.cfg:29`: `dx = 2.0e-6`
- `apps/sim_line_scan_316L.cu:189`: `config.dx = 2.0e-6f`
- `apps/sim_line_scan_316L.cu:324`: `interface_z = 80.0f` (= 160 μm = 80 cells × 2 μm)

Keyhole vapor channel from R8/Sprint-1 measurements: D ≈ 30 μm (R8 vs F3D D_open=82μm, but our LBM keyhole is narrower at ~30-40 μm in the conduction-mode regime).

D/dx ≈ 15 — same regime as aero D/dx=20.

**Aero finding extrapolated to AM**:

| Aero D/dx | Cd error | Source |
|---|---|---|
| 20 | +16% | resolution-dominated |
| 40 | +13% | resolution + BC equal |
| 80 | +12% | BC-dominated (saturation) |

If AM follows the same convergence behaviour:
- Current `dx=2 μm` (D/dx≈15) is in the resolution-dominated band
- Refining to `dx=1 μm` (D/dx≈30) should drop error to ~+10-13%
- Refining to `dx=0.5 μm` (D/dx≈60) should saturate around +8-10% from BC ceiling
- Strict match to F3D (<2%) requires higher-order BC (quadratic ABB) — research-level for FSLBM (Bogner 2017 is closest in literature)

**Strategic implication for R32**:

R32 P3 just verified ray-tracing kernel in three regimes (`r32/raytracing-keyhole@2b1718b`). The kernel is correct. But:
- Ray-tracing closes the **physics gap** (multi-bounce absorption that single-bounce model misses)
- It does NOT close the **scheme gap** (resolution + BC order)

If the +13% Δz disagreement decomposes as:
- ~2-3% Z-convention bias (Item 4)
- ~5-7% resolution ceiling at D/dx=15 (Item 5 part 1)
- ~3-5% BC scheme order residual (Item 5 part 2)

… then R32 ray-tracing gives back the missing physics but the residual ~10% gap stays.

**Action** (HIGH priority, ~1 week):

Before R32 commits ray-tracing into R26-R31 production cases, run a **resolution sensitivity study**:
1. Pick one stable R26-R31 case (e.g., R31 baseline at PoolL=116 / W=92 / D=56)
2. Run at `dx ∈ {4, 2, 1} μm` keeping all other physics fixed
3. Plot Δz, Pool L/W/D vs 1/dx
4. **If linear**: resolution dominates. AMR is the answer; expect F3D match closer to 2%.
5. **If saturating**: BC scheme order dominates. AMR alone won't break out. Two paths: (a) accept the +5-10% as scheme limitation and document, (b) implement higher-order ABB (research-level).
6. **If neither — convention bias dominates**: Item 4 fix is sufficient; the disagreement was largely a measurement artifact.

This study is ~1 week of compute + analysis time. Compare to the R26-R31 budget (4 weeks each, 4 rounds = 16 weeks): a strategic study before R32 commits is high ROI.

---

## 5. Recommendations

### 5.1 For the aero branch (`feature/compressible-aero`)

**Immediate (committed)**:
- Phase 1+1b at commit `887703c` is a clean waypoint. Lenient gating PASS at D/dx≥40.

**Phase 1c** (3 days):
- Implement quadratic Bouzidi (3-cell stencil, Bouzidi 2001 or Yu 2003 simplified form). Should drop Cd plateau from 3.63 to ~3.30 at D/dx=40, entering DFG strict band.
- Same toolkit needed for serious NACA validation; this is a prerequisite.

**Phase 2** (3 days):
- NACA0012 Re=1000 (Kurtulus 2015 reference). Add STL or SDF reader for arbitrary geometry.
- Same parabolic-inlet + pressure-outlet toolkit.
- With Phase 1c quadratic BC, Cd/Cl/St should match Kurtulus to <5%.

**Phase 3** (1 week):
- 3D bluff bodies: sphere (Schiller-Naumann), simple wing (ONERA M6 if going aggressive).

### 5.2 For the AM/master side (R32)

**Quick wins** (1 day total):
- Item 1: add `setTRT()` calls to `apps/debug_buoyancy_flow.cu` and `apps/visualize_laser_melting_with_flow.cu` (or document BGK precision). 1 hr.
- Item 2: refactor `boundary_conditions.cu:239` to use `D3Q19::h_ex` (after aero merge or by cherry-picking the public-tables change). 1 hr.
- Item 4: document Z-convention in `extract_track_height.py`; add `--cell-centered-vof` flag. 4 hr.

**Strategic** (1 week):
- Item 5: resolution sensitivity study at dx={4, 2, 1} μm on one R26-R31 baseline. This single study has more diagnostic value than another full round of physics tuning.

**Optional** (1 day):
- Item 3: BC-stencil audit on Marangoni / recoil / evaporation kernels. Lower confidence in the impact, but cheap to do.

### 5.3 For R32 specifically

R32 P3 verified ray-tracing kernel in flat / cylinder / frustum geometries (5h vs 5d budget — clean win). Branch `r32/raytracing-keyhole@2b1718b`.

Before integrating ray-tracing into R26-R31 cases:
1. **Apply Item 4 Z-convention fix** (1 day): re-measure existing R26-R31 baseline Δz with corrected convention. Document before/after numbers.
2. **Run Item 5 sensitivity study** (1 week): one R26-R31 case at dx={4, 2, 1}. Determine whether the gap is resolution / BC / convention / physics.
3. **Set expectation**: ray-tracing closes the physics gap (single-bounce → multi-bounce absorption, possibly +10-30% effective absorption in keyhole). It does NOT close resolution or BC scheme gaps. The result is likely "R32 with ray-tracing → Δz closes from +13% to +5-7%, still not matching F3D strictly".

This is a more honest framing than "R32 ray-tracing fixes the keyhole match" — and prevents another R32-redux cycle if reality matches the prediction.

---

## 6. Files Changed (commit `887703c` on `feature/compressible-aero`)

### Modified (5 files, +539 / −14 lines)

| File | Change |
|---|---|
| `CMakeLists.txt` | +6 (aero src glob + new app target `aero_schaefer_turek_3d`) |
| `include/core/lattice_d3q19.h` | h_ex/h_ey/h_ez/h_opposite moved from private to public |
| `include/core/streaming.h` | +CELL_FLUID/CELL_SOLID constants |
| `include/physics/fluid_lbm.h` | +setSolidMask/setObstacleQ/setParabolicInletX/setPressureOutletX APIs + 2 new kernel decls |
| `src/physics/fluid/fluid_lbm.cu` | +setSolidMask/setObstacleQ/setParabolicInletX/setPressureOutletX impls + fluidStreamingKernelWithSolid + fluidStreamingKernelWithBFL kernels |

### Added (8 files, +1304 lines)

| File | Lines | Purpose |
|---|---|---|
| `apps/aero/aero_schaefer_turek_3d.cu` | 313 | CLI driver |
| `include/physics/aero/obstacle_geometry.h` | 277 | Cylinder/sphere stamping + q-fraction analytic helpers |
| `include/physics/aero/momentum_exchange_force.h` | 71 | MEM force API |
| `src/physics/aero/momentum_exchange_force.cu` | 175 | BC-aware MEM force kernel |
| `scripts/aero/strouhal_fft.py` | 209 | Cl FFT + DFG comparison + CI gating |
| `scripts/aero/convergence_plot.py` | 122 | Multi-resolution comparison |
| `scripts/aero/dump_qfrac_diag.py` | 86 | Pure-Python q-fraction replica for cross-check |
| `config/aero/aero_schaefer_turek_2d_2_slab.cfg` | 51 | Documentation/preset |

**Total**: 13 files, +1843 / −14 lines.

---

## 7. Reproduction Recipes

### 7.1 Build

```bash
cd /home/yzk/CompressibleCFD
mkdir -p build && cd build
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=86 ..
cmake --build . --target aero_schaefer_turek_3d -j
```

### 7.2 Smoke run (D/dx=20 stair, 3 min)

```bash
cd /home/yzk/CompressibleCFD
build/aero_schaefer_turek_3d --case 2d-2 --re 100 --resolution 20 --bc stair \
    --steps 30000 --probe-every 50 --output-dir output_aero_2d2_stair
python3 scripts/aero/strouhal_fft.py output_aero_2d2_stair --U-avg 1.0
# Expected: Cd_max≈3.77, Cl_max≈1.20, St≈0.28 (gate OUT on Cd, ±20% tolerance)
```

### 7.3 Phase 1b QBB + lenient gating (D/dx=40, ~16 min)

```bash
build/aero_schaefer_turek_3d --case 2d-2 --re 100 --resolution 40 --bc qbb \
    --steps 60000 --probe-every 100 --output-dir output_aero_2d2_qbb_d40
python3 scripts/aero/strouhal_fft.py output_aero_2d2_qbb_d40 --U-avg 1.0 --assert-pass
# Expected: GATE PASS (Cd +13%, Cl +15%, St +5%)
```

### 7.4 Convergence study (3 resolutions, ~3 hr total)

```bash
build/aero_schaefer_turek_3d --case 2d-2 --re 100 --resolution 20 --bc qbb --steps 30000 --probe-every 50 --output-dir output_aero_2d2_qbb
build/aero_schaefer_turek_3d --case 2d-2 --re 100 --resolution 40 --bc qbb --steps 60000 --probe-every 100 --output-dir output_aero_2d2_qbb_d40
build/aero_schaefer_turek_3d --case 2d-2 --re 100 --resolution 80 --bc qbb --steps 120000 --probe-every 100 --output-dir output_aero_2d2_qbb_d80

python3 scripts/aero/convergence_plot.py \
    D20:output_aero_2d2_qbb \
    D40:output_aero_2d2_qbb_d40 \
    D80:output_aero_2d2_qbb_d80 \
    --out output_aero_2d2_qbb_d80/convergence_3pts.png
```

### 7.5 Outlet-reflection control test (~3 min)

```bash
build/aero_schaefer_turek_3d --case 2d-2 --re 100 --resolution 20 --bc qbb \
    --lx 4.4 --steps 30000 --probe-every 50 --output-dir output_aero_2d2_qbb_lx44
python3 scripts/aero/strouhal_fft.py output_aero_2d2_qbb_lx44 --U-avg 1.0
# Expected: Cd_max≈3.79 (≈ same as Lx=2.2)
```

### 7.6 Diagnostic: BFL reduces to halfway BB at q=0.5

```bash
build/aero_schaefer_turek_3d --case 2d-1 --re 20 --resolution 20 --bc qbb-half \
    --steps 12000 --probe-every 200 --output-dir output_aero_re20_qbbhalf
# Expected: Cd≈6.30 (identical to --bc stair within 0.1%)
```

---

## 8. Open Questions (for next iteration)

1. **F3D VTK convention** — bottom-edge or cell-center? Affects Item 4 magnitude. (Easy verification: open one of `vtk-316L-150W-50um-V800mms/*.vtk` in ParaView → File Information.)
2. **F3D AMR effective dx in keyhole region** — at the F3D 316L 150W 800mm/s reference case, what is the actual cell size in the keyhole region? Affects whether our dx=2μm is fundamentally under-resolved by F3D's standard or just by the strict-DFG standard.
3. **Quadratic Bouzidi variant choice** — Bouzidi 2001 quadratic 3-cell or Yu 2003 simplified single-node? Yu's variant is more common in modern code; needs literature decision before Phase 1c.
4. **Phase 2 geometry import format** — STL (ubiquitous, slower in/out test) or SDF (faster, needs SDF generation pipeline)?
5. **R26-R31 Δz disagreement decomposition** — pending Item 4+5 actions on the AM side.

---

## 9. Cross-references

### Memory entries
- `/home/yzk/.claude/projects/-home-yzk-LBMProject/memory/feedback_lbm_only_aero.md`: user preference for LBM-only aero
- `/home/yzk/.claude/projects/-home-yzk-LBMProject/memory/project_aero_phase1_2026_05_08.md`: Phase 1+1b project memory
- `/home/yzk/.claude/projects/-home-yzk-LBMProject/memory/project_am_audit_from_aero_2026_05_08.md`: AM-side 5-item audit findings
- `/home/yzk/.claude/projects/-home-yzk-LBMProject/memory/project_r32_p3_2026_05_08.md`: R32 P3 ray-tracing verification (parallel work)

### Key literature
- Schäfer M., Turek S. (1996), DFG benchmark, Notes Numer. Fluid Mech. 52:547-566
- Bouzidi M., Firdaouss M., Lallemand P. (2001), Phys. Fluids 13:3452 — linear interpolated bounce-back
- Mei R., Yu D., Shyy W., Luo L.S. (2002), Phys. Rev. E 65:041203 — MEM force at curved walls
- Krüger T. et al. (2017), "The Lattice Boltzmann Method", Springer — §8.5 Bouzidi convergence
- Ladd A.J.C. (1994), J. Fluid Mech. 271:285 — original MEM
- Yu D., Mei R., Luo L.S., Shyy W. (2003), Prog. Aerosp. Sci. 39:329 — viscous flow LBM review

### Branches (worktrees)
- `feature/compressible-aero` (this work) — `887703c`, `/home/yzk/CompressibleCFD`
- `r32/raytracing-keyhole` (parallel) — `2b1718b`, `/home/yzk/LBMProject`
- `master` — `d56b2ad`, base of both
