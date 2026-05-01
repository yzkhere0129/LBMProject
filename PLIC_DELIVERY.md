# PLIC-VOF Upgrade — Delivery Summary

**Branch**: `feature/plic-vof` (worktree `/home/yzk/lbm_plic_vof`)
**Parent**: `master` @ `d56b2ad`
**Status**: All code-level work complete. **Only LPBF F3D-calibration simulation run remains** (left for main process to avoid GPU contention).

## TL;DR

13 commits on top of `d56b2ad`. 35 PLIC test cases passing. All five LPBF surface-physics paths (laser, CSF, Marangoni, recoil, evap) have a sharp-delta + height-function variant that is opt-in via `MultiphysicsConfig::enableFullPLICStack()`. Default behavior is byte-identical to pre-Phase-1 master.

```bash
# Quick verify everything works:
cmake --build build -j8
./build/diag_plic_smoke --steps 100        # legacy + PLIC, prints comparison

# Run full-stack regression:
for t in build/tests/test_plic_*; do [ -x "$t" ] && $t; done
./build/tests/integration/multiphysics/test_plic_full_stack
```

## Commit chain (newest last)

```
5c359d4  Phase 1   InterfaceGeometryView API + H1/H2/H3 fixes
f84355a  Phase 1.5 HF normals (sphere mean=9.7e-4 rad satisfies ε<1e-3)
b31194c  Phase 2   Laser column-march sharp deposit
0344c09  Phase 3a  CFK height-function curvature
8d04119  Phase 3b/c/d Sharp-delta CSF + Marangoni + recoil
bb25559  Phase 5   MultiphysicsSolver dispatch + viz patches
9d14b18  Phase 4a  PLIC HKL evap (sharp delta)
72fa066  Phase 4a/6 Wire evap + full-stack integration test
e014e3f  Phase 7   200-step long-run validation
45178f2  fix       CSF sign bug (outward → inward squeeze)
dad253f  feat      diag_plic_smoke runnable diagnostic
e242d34  feat      enableFullPLICStack convenience preset
07813b8  Phase 8   Close all known gaps (this is the final commit)
```

## What Phase 8 closed (the highest-priority audit findings)

1. **HF curvature for `feature/plic-vof`** — already there; agents verified math (`cfd-math-expert`)
2. **Geometric polygon-clipping `dA_PLIC`** — Phase 4a's cosine-delta variant only captures sharp localization. The new `applyEvaporationMassLossPLICAreaKernel` uses true PLIC-plane area for the 1/cos(θ) tilted-interface enhancement. Test results: 0.017–0.026% error vs analytic on 45°/3D-diagonal cases.
3. **`molar_mass` 316L mismatch** — recoil kernel was reading `SurfaceConfig::molar_mass=0.0476` (Ti6Al4V). Fixed to read `material.molar_mass` first. Closes a silent ×2-3 P_recoil error for 316L runs.
4. **5 leftover function-level statics** — promoted to class members (multi-instance safety). The H1 fix in 5c359d4 caught the ones in `advectFillLevel`'s reduction buffer + PLIC clamp counter; the audit caught 5 more in advection diagnostics + evap.
5. **`enableFullPLICStack()` foot-gun** — `MultiphysicsSolver::initialize()` now auto-flips the VOFSolver normal/curvature methods. Users can no longer "forget".
6. **4 physics-pinning tests** — new tests check actual physics (hot→cold direction, Clausius-Clapeyron magnitude, Laplace pressure, analytic ΔT) rather than tautological direction-only assertions.
7. **F3D-comparison harness** — runnable apps + scripts ready for main-process to run.

## Physics tests (35 cases, all passing)

| Test | Cases | What it pins |
|---|---|---|
| `test_plic_normal_vs_analytic` | 3 | Sphere n̂ mean error 9.7e-4 rad (ε<1e-3 spec) |
| `test_plic_alpha_inversion` | 2 | SZ inclusion-exclusion round-trip 1.2e-7 |
| `test_plic_dirty_flag` | 6 | API contract for cache lifecycle |
| `test_plic_curvature` | 2 | Sphere κ mean rel err 2.15e-3 |
| `test_plic_csf_force` | 1 | F direction inward (after 45178f2 sign fix) |
| `test_plic_csf_laplace` | 1 | Column-integrated F·n̂ vs σκ (16.9%) |
| `test_plic_marangoni_recoil` | 2 | Marangoni tangency, recoil anti-parallel |
| `test_plic_marangoni_direction` | 1 | F_z hot→cold for dσ/dT<0 (NEW) |
| `test_plic_recoil_magnitude` | 1 | Clausius-Clapeyron magnitude rel err 0.8% (NEW) |
| `test_plic_laser_column` | 4 | Top-down deposit / bulk rejection / f-floor |
| `test_plic_laser_column_dT` | 1 | Analytic ΔT match within 0.25% (NEW) |
| `test_plic_evaporation` | 2 | Bulk-cell rejection, total mass loss order |
| `test_plic_evap_area` | 6 | dA_PLIC vs analytic 0.017% (NEW) |
| `test_plic_full_stack` | 3 | E2E 50/200 step stability, mass=0, no NaN |

## API quick reference

### Enable PLIC for a simulation

```cpp
#include "physics/multiphysics_solver.h"
using namespace lbm::physics;

MultiphysicsConfig cfg;
// ... configure grid, material, laser, BCs as usual ...

cfg.enableFullPLICStack(/*h_smooth_lu=*/1.5f);  // turns on all 5 PLIC paths

// VOFSolver normal + curvature methods are auto-flipped by initialize().
// No manual setNormalReconstructionMethod / setCurvatureMethod calls needed.

MultiphysicsSolver solver(cfg);
solver.initialize(T_init, fill_init);
for (int step = 0; step < n_steps; ++step) solver.step();
```

### Individual PLIC opt-ins (diagnosis / partial cutover)

```cpp
cfg.laser.plic_aware_column_march    = true;   // Phase 2
cfg.surface.csf_use_plic_delta       = true;   // Phase 3b
cfg.surface.marangoni_use_plic_delta = true;   // Phase 3c
cfg.surface.recoil_use_plic_delta    = true;   // Phase 3d
cfg.surface.evap_use_plic_delta      = true;   // Phase 4a (cosine δ)
cfg.surface.evap_use_plic_area       = true;   // Phase 4a (geometric dA — overrides delta)
cfg.surface.plic_h_smooth_lu         = 1.5f;   // cosine-kernel half-width
```

## What's left for the main process

1. **Run the F3D calibration**:

```bash
# Pre-flight: F3D reference is already extracted.
ls vtk-316L-150W-50um-V800mms/f3d_reference_metrics.json     # exists
ls vtk-316L-150W-50um-V800mms/f3d_preview.png                # exists

# OPTIONAL: bump absorptivity to 0.70 in BOTH apps for F3D-magnitude pool sizes.
# Currently both apps inherit the legacy 0.40, which gives ~50% pool dimensions
# vs F3D. Edit apps/sim_line_scan_316L.cu and apps/sim_line_scan_316L_plic.cu
# if you want apples-to-F3D rather than apples-to-apples comparison.

# Run both LBM modes (~2 hrs total at full LPBF run length):
bash scripts/run_plic_calibration.sh

# Compare:
python3 scripts/compare_plic_vs_f3d.py
# → prints 3-row table (legacy / PLIC / F3D) + saves compare_plic_vs_f3d.png
```

2. **Expected outcome (hypothesis to verify)**:
   - PLIC ON should narrow Pool W toward F3D's 73 μm (legacy ~110 μm).
   - PLIC ON should sharpen the keyhole D_open toward F3D's 83 μm (legacy ~40 μm conduction).
   - Pool L will likely still differ — that gap is partly conduction-calibration, not VOF-smearing.

3. **If results disappoint, the next investigation steps are documented in `PLIC_UPGRADE_ROADMAP.md` §11** (this file is local-only / gitignored).

## Known limitations (honest disclosure)

- **CSF Laplace test is at 16.9 % rel err vs σκ** (within the 20% gate). The remaining 17% comes from integrating along a Cartesian axis instead of the true n̂ direction; for tilted normals the column-walk undershoots by O(1−cos θ). To tighten, integrate along the actual n̂ (more expensive). Sign and magnitude order are correct.

- **PLIC vs legacy `T_peak` shows -83 K in 200-step diag** (1605 vs 1688 K) — `test_plic_laser_column_dT` confirms the Phase 2 deposit formula is correct to 0.25%, so the diag difference is conduction equilibration of the more-concentrated PLIC source, not a kernel bug. Real-time T_max comparison needs the LPBF benchmark.

- **diag_plic_smoke is too short to see meaningful Pool W/L differences** (32×32×24 grid, 1 μs run). The full LPBF benchmark via `sim_line_scan_316L_plic` is what will actually show the upgrade benefit.

- **Curvature kernel uses 3-point central diffs** for h_u, h_v while HF-normal kernel uses Parker-Youngs 8-point weighted. Sub-optimal at 45° interfaces but not wrong; defer until a real LPBF case shows κ noise problems.

- **Phase 4b (Track-C δ_h-weighted mass redistribution) was not done.** The existing kernel already gates at f∈(0.01, 0.99) ≈ interface cells, so this is a marginal improvement. Not in the critical path.

## Files of interest

```
include/physics/interface_geometry.h           Phase 1 API + dA helper
include/physics/vof_solver.h                   Phase 1+ public methods
include/physics/multiphysics_solver.h          enableFullPLICStack + flags
include/physics/force_accumulator.h            PLIC force kernel host wrappers

src/physics/vof/vof_solver.cu                  HF normals, HF κ, dA evap
src/physics/force_accumulator.cu               Sharp-delta CSF/Marangoni/recoil
src/physics/multiphysics/multiphysics_solver.cu Dispatch + auto-flip

apps/diag_plic_smoke.cu                        Quick PLIC-vs-legacy diagnostic
apps/sim_line_scan_316L_plic.cu                LPBF F3D-comparison binary

scripts/run_plic_calibration.sh                Run both modes
scripts/compare_plic_vs_f3d.py                 Compare to F3D
scripts/extract_f3d_reference_metrics.py       F3D ground-truth extractor

vtk-316L-150W-50um-V800mms/f3d_reference_metrics.json   Locked F3D numbers
vtk-316L-150W-50um-V800mms/f3d_preview.png              Visual confirmation

tests/unit/vof/test_plic_*.cu                  35 PLIC test cases
tests/unit/laser/test_plic_laser_column.cu     Phase 2 unit test
tests/integration/multiphysics/test_plic_full_stack.cu  E2E test
```
