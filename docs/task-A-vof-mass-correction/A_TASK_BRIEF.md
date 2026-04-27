# Task A — VOF Mass-Correction Destination Algorithm

**Worktree:** `/home/yzk/LBMProject_vof_mass_correction/`
**Branch:** `feature/vof-mass-correction-destination`
**Forked from:** `03180f1 test(trt): anchor test that TRT(ω⁻=ω⁺) ≡ BGK within FP32 round-off` on `benchmark/conduction-316L`

## What broke

`enforceGlobalMassConservationKernel` at `src/physics/vof/vof_solver.cu:1227`:

```cuda
__global__ void enforceGlobalMassConservationKernel(
    float* fill_level, float scale_factor, int num_cells)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_cells) return;
    float f_old = fill_level[idx];
    float f_new = f_old * scale_factor;
    fill_level[idx] = fminf(1.0f, fmaxf(0.0f, f_new));
}
```

**Uniform multiplicative scaling.** When the sim has lost mass (typical for
LPBF: evaporation removes metal), every cell with f > 0 gets multiplied
by the same scale_factor > 1. Cells already at f=1 are clamped (no
change), so the "extra" mass concentrates in **partially-filled cells
only** — i.e., the VOF interface.

The interface in an LPBF simulation has two qualitatively different
populations:
1. **Trailing centerline cells** (the legitimate refill site — surface
   under capillary pressure, supposed to fill toward the center groove)
2. **Scan-start splash deposit + side ridges** (mass already over-deposited
   there from initial keyhole punching; these cells absorb the redistribution
   and grow taller, making the U-shape worse)

Phase-2 of the night protocol verified this. Mass correction enabled →
centerline Δh degraded from -18.9 μm (Phase-1) to **-28.9 μm at t=400 μs**.
Aborted run.

## What "good" would do

Distribute reclaimed mass preferentially to **cells whose v_z > 0 at the
free surface AND are in the trailing zone (x < laser_x)**. This matches
the physical intent of mass conservation: replenish the cells that are
trying to fill (capillary back-flow), not the cells already over-pinned.

## Concrete algorithm options

A1. **v_z-weighted scale**: `f_new = f_old + Δm * w(idx)` where
    `w(idx) ∝ max(v_z(idx,k_top), 0)` for surface cells, normalised
    such that Σw = 1. Conservative; safe at fl=1 limit; needs `velocity_z`
    pointer plumbed into VOFSolver.

A2. **Trailing-band mask**: scale only cells with `x < x_laser - margin`
    AND `0.05 < f < 0.95`. Simpler but needs laser position injection
    each step.

A3. **Combined**: A1 with A2 mask.

A4. **Reconstruct from evap source**: instead of correcting after-the-fact,
    track the geometric origin of evap losses and reinject. Most physical
    but most invasive.

Recommend starting with A1 (least invasive change to algorithm; validates
the principle) and benchmarking against Phase-2 result.

## Files of interest

- `src/physics/vof/vof_solver.cu` — kernel + host method
- `include/physics/vof_solver.h` — VOFSolver class
- `src/physics/multiphysics/multiphysics_solver.cu` — caller (currently
  not enabled by default; was on for Phase-2 only via
  `apps/sim_linescan_phase2.cu` `enable_vof_mass_correction = true`)

## Reproduction harness

To compare a new algorithm against the Phase-2 result:

```
cd /home/yzk/LBMProject_vof_mass_correction
# After modifying enforceGlobalMassConservationKernel:
cmake --build build --target sim_linescan_phase2 -j 4
rm -rf output_phase2
mkdir output_phase2
stdbuf -oL ./build/sim_linescan_phase2 > output_phase2/run.log 2>&1
# wait ~33 min
python3 scripts/flow3d/phase1_summary.py output_phase2/line_scan_005000.vtk
python3 scripts/flow3d/check_mass_conservation.py \
    output_phase2/line_scan_000000.vtk output_phase2/line_scan_005000.vtk
```

## Acceptance Criteria — STRICT (revised 2026-04-26 by main session)

The original "improves over Phase-1" wording was too permissive: any
≥1 μm gain at t=800 μs would technically satisfy it. Below is the
hardened version.

### Pass conditions — ALL must hold

| # | Metric                              | Threshold |
|---|-------------------------------------|-----------|
| 1 | Centerline Δh @ t = 800 μs (95%ile) | **≥ -10 μm** (Phase-1 baseline -16; F3D target -1; midpoint as merge bar) |
| 2 | Centerline Δh @ t = 2 ms            | **≥ -10 μm** (steady state must not regress vs t=800 μs) |
| 3 | Side-ridge Δh @ x_offset = -100, -200 μm | **both in [+3, +10] μm range** (must include F3D mean +4 and max +7.5) |
| 4 | Total mass drift over t_total       | **\|ΔM/M₀\| < 1.0 %** (absolute, not relative to Phase-1) |
| 5 | TRT degenerate-to-BGK anchor test   | PASS (`./build/tests/validation/test_trt_degenerate_to_bgk`) |
| 6 | All 75 phase-change validation tests | PASS (no regression) |
| 7 | v_z @ -150 μm at t = 800 μs         | **> +0.10 m/s** (Phase-1 had +0.196 — recirculation must not die) |

### Reject conditions — ANY ONE triggers abort

- Tried 3 different destination algorithms (e.g., v_z-weighted,
  trailing-band masked, evap-source reconstruction), best Δh improvement
  over Phase-1 (-16) is **< 3 μm**
- Any algorithm produces NaN or instability before t < 500 μs in the
  Phase-2 reproduction harness
- Any algorithm pushes side ridges above **+15 μm** at any trailing-zone
  offset (physically tearing the surface)
- Mass drift exceeds **3.0 %** (worse than baseline Phase-1's -2.55 %)

### Merge protocol

- **All 7 pass conditions met** → squash + rebase to `benchmark/conduction-316L`,
  with a commit message describing the algorithm + measured improvements
- **Reject condition triggered** → DO NOT merge code. Instead:
  - Commit a `lessons_learned.md` to `docs/task-A-vof-mass-correction/`
    describing each algorithm tried, its failure mode, and what data
    pointed at the next physical bottleneck (likely Bogner FSLBM or
    sharp-interface VOF rewrite — both Sprint-level)
  - Cherry-pick *only* `docs/task-A-vof-mass-correction/` to main
  - Delete the worktree branch

### Validation harness — required before any merge

```
# 1. Short test (33 min) — Phase-2 reproduction
cd /home/yzk/LBMProject_vof_mass_correction
cmake --build build --target sim_linescan_phase2 -j 4
rm -rf output_phase2 && mkdir output_phase2
stdbuf -oL ./build/sim_linescan_phase2 > output_phase2/run.log 2>&1
python3 scripts/flow3d/phase1_summary.py output_phase2/line_scan_010000.vtk
python3 scripts/flow3d/check_mass_conservation.py \
    output_phase2/line_scan_000000.vtk output_phase2/line_scan_010000.vtk

# 2. Full test (~85 min) — Phase-4 2 ms validation
cmake --build build --target sim_linescan_phase4 -j 4
rm -rf output_phase4 && mkdir output_phase4
stdbuf -oL ./build/sim_linescan_phase4 > output_phase4/run.log 2>&1
python3 scripts/flow3d/phase1_summary.py output_phase4/line_scan_025000.vtk

# 3. Regression suite
ctest --test-dir build -L "validation;phase_change" --timeout 120
./build/tests/validation/test_trt_degenerate_to_bgk
```

Comparison reference data already in repo:
- `output_line_scan/line_scan_010000.vtk`           — pre-fix baseline (-20 μm)
- `output_phase1/line_scan_010000.vtk`              — Phase-1 best so far (-16 μm)
- `vtk-316L-150W-50um-V800mms/150WV800mms-50um_99.vtk` — F3D ground truth (-1 μm)
