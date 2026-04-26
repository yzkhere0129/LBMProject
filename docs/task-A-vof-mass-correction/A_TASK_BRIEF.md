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

Target: at t=400 μs, centerline Δh ≥ Phase-1's -18.9 μm AND mass-loss
rate within 0.5 % of Phase-1's -1.66 % per 800 μs.

## Don't merge until

- New algorithm passes the reproduction comparison above
- TRT-degenerate-to-BGK anchor test still passes
  (`./build/tests/validation/test_trt_degenerate_to_bgk`)
- No regression on existing 75 phase-change validation tests
- Centerline Δh improves over both Phase-1 (-16) and Phase-2 (-28.9) at
  t=800 μs
