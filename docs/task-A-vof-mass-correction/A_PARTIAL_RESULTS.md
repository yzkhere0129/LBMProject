# Task A — Partial Validation Results (this machine)

**Date:** 2026-04-26
**Hardware constraint:** RTX 3050 Laptop (4 GB VRAM, 95 W TGP), ~12 step/min on
this Phase-2 config. Full Phase-2 (10000 steps) needs ~14 hours; Phase-4
(25000 steps) ~35 hours. Validation moved to a faster machine.

## What was completed

| Task | Status |
|---|---|
| Build unblocked (3 stale CMake targets, 1 missing FluidLBM setter) | ✅ commit `1934854` |
| A1 v_z-weighted mass correction algorithm | ✅ commit `cc90a26` |
| 5 latent bugs fixed (zero-div guard, scratch buffer cache, dedupe TVD/PLIC) | ✅ commit `cc90a26` |
| 8 GTest unit tests for A1 (7 active, 1 disabled) | ✅ pass |
| TRT-degenerate-to-BGK anchor test | ✅ pass |
| 7-criteria acceptance script | ✅ commit `dc7903f` |
| Phase-2 reproduction (sim_linescan_phase2 to step 10000) | ⏸️ **stopped at step 3510 (35.1%)** |
| Phase-4 (2 ms validation) | ❌ **not feasible on this hardware** |
| 75 phase-change validation tests | ❌ **not run** (would compete with Phase-2 for GPU) |

## A1 algorithm verification — engineering ✅

Algorithm engaging correctly through 3510 steps. 7 logged correction events,
all on the A1-vz path (W ≫ 0, never fell to uniform-additive fallback):

```
Step    ΔM         W           applied   notes
~0      -27.5      3.9e-2      -27.5     pre-melt-pool, tiny W, near-fallback
~500    -12.5      2.6e+3      -11.0     melt pool igniting
~1000    +7.5      1.6e+4       +6.5     pool established, A1 fully active
~1500   +10.0      2.1e+4       +8.5     pool growing, ΔM tiny relative
~2000    +2.0      2.4e+4       +1.5
~2500    +3.0      2.7e+4       +2.5
~3000    +3.0      2.8e+4       +3.0
```

`applied` slightly less than `ΔM` is the known clamp-saturation effect (A1
single-pass, no iterative redistribution — the disabled `DISABLED_ClampOverflow
Redistributed` test). At realistic LPBF Δm magnitudes (0.0001-0.0004% of total)
this leak is far below the 1.0% mass-drift gate.

## Acceptance criteria — early data

Trajectory comparison Phase-1 (no correction) vs Phase-2 (NEW A1):

| t [μs] | Centerline Δh 95%ile (μm) | Ridge -100 (μm) | Ridge -200 (μm) | \|ΔM/M₀\| |
|---|---|---|---|---|
| **Phase-1 (full reference run)** | | | | |
| 100 | NO ZONE | +8 | +8 | +0.193 % |
| 200 | NO ZONE | +8 | +10 | +0.227 % |
| 300 | NO ZONE | +8 | +10 | +0.218 % |
| 400 | **-16** (n=10, very noisy) | **+10** | **+12** | +0.165 % |
| 800 | **-14** (n=169) | +10 | +10 | -0.180 % |
| **Phase-2 (NEW A1, partial)** | | | | |
| 100 | NO ZONE | +8 | +8 | **+0.0001 %** |
| 200 | NO ZONE | **+14** ⚠️ | +8 | **-0.0000 %** |
| 400 | **NOT RUN** | | | |
| 800 | **NOT RUN** | | | |

## What the partial data tells us

### 1. Mass conservation — massive win
A1 is keeping mass tight to **0.0001 %** vs Phase-1's +0.227 % at the same
time step — about **10⁴× tighter**. This was expected: the new helper runs
the correction every advection step, the per-step Δm is tiny, and W ≫ 0
means the correction lands on real flow rather than the broken multiplicative
scale. **Criterion 4 (mass drift < 1.0 %) is comfortably passed.**

### 2. Side ridge at −100 μm — early warning ⚠️
At t = 200 μs Phase-2 shows **+14 μm** at the −100 μm offset, vs Phase-1's
+8 μm at the same frame. The brief's hard reject line is +15 μm; we're
**1 μm below the reject line and trending up**.

This is consistent with the math expert's pre-implementation warning:

> "max(v_z, 0) is directionally wrong for LPBF. The trailing groove is fed
> by lateral, surface-tangent capillary back-flow … Curvature weighting is
> not what you want here: high-κ regions are the ridge tops and pinned
> splash crests — exactly the cells Phase-2 over-fed."

Pure v_z weighting at t = 200 μs concentrates upward-flowing cells **near
the active laser zone** (recoil-driven uplift jets). The -100 μm offset is
close enough to that zone to absorb extra mass. As the pool stabilises
(t > 400 μs in Phase-1 trajectory) the upward cells re-distribute over the
trailing capillary back-flow region, but that transition is exactly when
we lost data.

### 3. Centerline Δh — undecidable from this data
The trailing-zone measurement first becomes statistically meaningful at
t ≥ 400 μs (Phase-1 first non-zero sample at step 5000, n = 10 cells, value
-16 μm; settles to -14 μm at step 10000 with n = 169 cells). **Phase-2
stopped at step 3510, still in the "NO ZONE" regime for the centerline
metric.** Cannot project from t ≤ 200 μs whether A1 will improve, regress,
or match Phase-1.

### 4. v_z @ -150 μm — both Phase-1 and Phase-2 building up identically
Phase-1: 0 → +0.005 → +0.005 → +0.129 m/s over t = 100 → 800 μs.
Phase-2: 0 → +0.011 m/s over t = 100 → 200 μs (matches Phase-1 trajectory
shape). Recirculation only establishes after the laser walks far enough
(post t = 400 μs).

## Verdict on this machine

**Cannot reach merge protocol from partial data.** The brief's pass conditions
require either Phase-2 final + Phase-4 final, or a clean reject signal. The
+14 μm ridge at t = 200 μs is a yellow flag, not a red one.

**Engineering correctness is high-confidence:**
- All commits clean, build green, unit tests + TRT anchor pass
- A1 algorithm verified active, mass conservation working as designed
- Real risk: A1 may fail criterion 3 (side ridges) at t = 800 μs if the
  ridge keeps growing past +15 μm. Phase-1 ridges peaked at +12 μm
  intermediate then settled to +10 μm; A1 may follow the same pattern
  but starting higher.

## Hand-off to faster machine

Re-running Phase-2 + Phase-4 on a faster GPU (RTX 30/40 series with full
TGP) is the next step. Expected wall time on RTX 3090/4080 class hardware:
Phase-2 ~30-60 min, Phase-4 ~75-150 min.

### Quick reproduction recipe

```bash
cd /path/to/LBMProject_vof_mass_correction
git checkout feature/vof-mass-correction-destination

# Build (verifies all 3 commits land clean)
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target sim_linescan_phase2 sim_linescan_phase4 \
                            test_trt_degenerate_to_bgk \
                            test_vof_mass_correction_weighted -j

# Sanity tests (< 1 min)
./build/tests/test_vof_mass_correction_weighted   # 7/7 active pass
./build/tests/validation/test_trt_degenerate_to_bgk  # 1/1 pass

# Phase-2 (~30 min on RTX 3090)
rm -rf output_phase2 && mkdir output_phase2
stdbuf -oL ./build/sim_linescan_phase2 > output_phase2/run.log 2>&1

# Phase-2 verdict
python3 scripts/diagnostics/check_acceptance.py \
    output_phase2/line_scan_010000.vtk \
    output_phase2/line_scan_000000.vtk

# Phase-4 (~90 min on RTX 3090)
rm -rf output_phase4 && mkdir output_phase4
stdbuf -oL ./build/sim_linescan_phase4 > output_phase4/run.log 2>&1
python3 scripts/diagnostics/check_acceptance.py \
    output_phase4/line_scan_025000.vtk \
    output_phase4/line_scan_000000.vtk
```

### Decision branches after reproduction

#### If all 7 criteria pass on Phase-2 + Phase-4
→ Squash-merge to `benchmark/conduction-316L`. Brief's merge protocol.

#### If centerline Δh ≥ -10 μm passes but ridge -100 μm > +15 μm at t = 800/2000 μs
→ A1 is over-feeding near-laser cells. Implement **A2 (trailing-band mask)**
on top: in `applyMassCorrectionInline`, mask out cells with
`x_lattice > x_laser - margin`. Margin ~50 μm. Need to plumb laser_x as a
new argument to the helper. Estimated ~30 lines of code change.

#### If centerline Δh stays at Phase-1's -14 μm or worse
→ A1 isn't redistributing mass to the centerline at all. **A3 (A1 + A2 mask)**
or **A4 (evap source reconstruction)** are the next layers. The math
expert's recommended weight `max(-n·v, 0)·|∇f|` (interface-flux) replaces
`max(v_z, 0)` and would target *only* genuine inflow into the liquid
phase, not arbitrary upward-moving cells.

#### If algorithm hits NaN / instability before t = 500 μs
→ A1 fundamental mismatch. Pivot to A4 (evap source) or commit a
`lessons_learned.md` per brief's reject protocol.

## Files / commits to inspect

```
# A1 algorithm
include/physics/vof_solver.h          (new overload signature, helper decl)
src/physics/vof/vof_solver.cu         (new kernels + helper, lines ~1245-1450)

# Tests
tests/unit/vof/test_vof_mass_correction_weighted.cu
tests/validation/test_trt_degenerate_to_bgk.cu

# Diagnostics
scripts/diagnostics/check_acceptance.py
scripts/diagnostics/locate_phase2_mass.py

# Commits (chronological)
1934854 fix(build): unblock build — setUseGuoForcing, stale CMake targets, TRT accessor
cc90a26 feat(vof): A1 v_z-weighted mass correction + bug fixes (Task A)
dc7903f feat(diag): Task-A 7-criteria acceptance check script
```
