# Debug Patrol Live Log

**Started**: 2026-04-27 21:50
**Branch**: debug/patrol-2026-04-27 (off benchmark/conduction-316L @ 36d62b0)

## Mission scope

User mandate: "现在的代码库我认为应该是漏洞百出的，再多的纠错也不为过" — assume
the codebase is full of latent bugs, do exhaustive debug patrol during the
5060-runtime wait. Continuous, parallel, multi-subagent.

## Already-applied audit fixes (already on main)

From overnight code-audit pass 1 + 2 (90 findings total):
- BUG-3: run_simulation.cu thermal dt/dx (5-line patch)
- F-02: thermal_lbm.cu computeTemperatureKernel const-cast write (UB removed)
- F-04: FluidLBM/VOFSolver copy ctor delete (rule of five)
- F-06: multiphysics_solver.cu evap mass_loss_scale=1.0 round-trip removed (~190 MB host transfer/step saved)
- F-08: getMaxMetalVelocity / getMaxTemperature CUDA_CHECK
- F-12: int*int*int overflow widened
- F-13: vof_solver compression non-periodic BC warning
- F-01: verified false alarm via `apps/diag_evap_material_readback`

Plus separately:
- BUG-1 (recoil_pressure.cu Ti6Al4V constants) verified DOES NOT bite production
  (multiphysics_solver.cu reads correct material; dead code only)

## Open audit findings (not yet patched)

### HIGH (from pass 1) — PATCHED
- **F-05 DONE** (commit cbfb0a2): d_fill_raw + d_fill_tmp hoisted to class members
- **F-07 DONE** (commit 7fbdfd7): 198 MB D→H replaced with GPU reductions (3 new kernels)
- **F-16 DONE** (commit f09aa55): reduceSum result buffer pre-allocated in RayTracingLaser
- F-10: zeroForceKernelLocal duplicate of force_accumulator.cu kernel
- F-11: D3Q7 lattice weights {0.25f, 0.125f, ...} duplicated in 8 kernels

### HIGH (from pass 2)
- F-03: 19/28 multiphysics integration tests are EXPECT_TRUE(true) stubs
- F-04: test_vof_mass_correction built but no add_test (CTest skips it)
- F-05: RayleighTaylorGerrisExact has no TIMEOUT (CI hang risk)
- F-07: validate() thermal diffusivity check is vacuous (defaults always positive)

### Many MEDIUM/LOW items

90 total - 7 fixed - ~9 above HIGH = ~74 more items to triage.

## Patrol queue (priority order)

1. DONE — F-05, F-07, F-16 perf fixes (this session)
2. DONE — F3-02 d_block_max statics hoisted (previous session, commit 9a6061b)
3. DONE — 12 stub tests replaced (previous session)
4. ⏳ Fix F-04 add_test missing for vof_mass_correction
5. ⏳ MEDIUM cleanups (F-10 zeroForceKernelLocal dedupe, etc.)

---

## F-05 / F-07 / F-16 Perf Fixes — 2026-04-27

### Timing baseline (benchmark_keyhole_316L, 100 steps, full multiphysics)
- **BEFORE** (main repo @ benchmark/conduction-316L): 147.8 s
- **AFTER** (patrol branch, F-05 + F-07 + F-16 + prior F3-02): 97.7 s
- **Speedup**: 34% on this benchmark (ray_tracing disabled, so F-16 not exercised)

### F-05: vofStep malloc/free
- Files: `include/physics/multiphysics_solver.h`, `src/physics/multiphysics/multiphysics_solver.cu`
- Added `d_vof_fill_raw_` + `d_vof_fill_tmp_` as class members (nullptr in ctor init list)
- Allocated in `allocateMemory()`, freed in `freeMemory()`
- Per-step block in `vofStep()` removed 2 × cudaMalloc/Free calls
- Commit: cbfb0a2

### F-07: 198 MB D→H diagnostic copies
- File: `src/physics/multiphysics/multiphysics_solver.cu`
- Added 3 new kernels: `findMaxVecMagnitudeKernel`, `findMaxAbsZKernel`, `findMaxVecMagnitudeIdxKernel`
- Both `if (enable_cfl_diag)` blocks (before + after CFL) now use GPU reductions
- Single `sizeof(float)` copy per reduction; 3-component index transfer is 3×4 bytes
- Reuses existing `d_energy_temp_` (double-size scratch) as float reduction target
- Diagnostic output (values + format) is identical to before
- Commit: 7fbdfd7

### F-16: reduceSum per-call malloc
- Files: `include/physics/ray_tracing_laser.h`, `src/physics/laser/ray_tracing_laser.cu`
- Added `d_reduce_result_ (CudaBuffer<float>, size 1)` to class private members
- Initialised in constructor member-init list
- `reduceSum()` changed from `static` to non-static; uses `d_reduce_result_.get()`
- Removes 2 × cudaMalloc/cudaFree per `traceAndDeposit()` call
- Commit: f09aa55

### Test results
- `test_vof_mass_correction_flux`: 9/9 PASS
- `test_thermal_lbm`: 8/8 PASS (1 disabled)

## Currently active subagents

(filled as launched)

---

## Audit Pass 3 — Completed 2026-04-27

**File**: `docs/debug-patrol/audit-pass3-findings.md`  
**Total findings**: 37 (3 CRITICAL, 8 HIGH, 15 MEDIUM, 11 LOW)

**Scope covered**:
1. vof_solver.cu Track-C additions (~900 new lines) — 9 findings
2. sprint-history overnight_audit test files — 3 findings  
3. apps/sim_linescan_phase1-5 + S3A1/S3A3/dawn3 — 7 findings
4. include/utils/cuda_memory.h CudaBuffer<T> — 3 findings
5. include/core/unit_converter.h — 2 findings
6. src/config/simulation_config.cpp — 2 findings
7. MultiphysicsSolver BC ordering — 3 findings
8. Cross-cutting — 8 findings

**New critical items for immediate attention**:
- F3-02: Static local GPU pointers in advectFillLevel() — survive VOFSolver destruction
- F3-15: Phase-3 kinematic_viscosity=0.0167 likely unit mismatch (τ→freeze instead of τ→0.55)
- F3-36: applyMassCorrectionInline casts long long interface_count to int — wraps for large domains

**Pass 3 adds 37 new findings to the triage queue.**  
Running total: 90 (pass1+2) + 37 (pass3) = 127 findings cataloged.

---

## Numerical-accuracy audit on collision kernels — Completed 2026-04-27

**File**: `docs/debug-patrol/numerical-audit-collision-kernels.md` (~660 lines)
**Scope**: line-by-line vs textbook (Kupershtokh 2009, Latt 2006, Hou 1996,
Ginzburg 2008, Guo 2002) review of 7 areas in `src/physics/fluid/fluid_lbm.cu`.

**Verdict**: 2 confirmed bugs, 3 potential, 5 style. Codebase is mostly numerically
sound — equilibrium, regularized projection, TRT decomposition, Smagorinsky algebraic
formula, opposite table all verified correct by independent derivation.

**Confirmed bugs (P0)**:
- **N-1.2** [BUG]: `fluidBGKCollisionEDMKernel` (line 1455) and
  `computeMacroscopicSemiImplicitDarcyEDMKernel` (line 2060) use
  DIFFERENT formulas for `u_phys` in mushy zone. ~40% relative
  discrepancy at typical K_LU. Each step's collision-time u write is
  overwritten with the inconsistent post-stream value, then fed back to
  next step's force build. Could explain Sprint-1 raised-track sign-flip.
  Patch: `:2060-2062` use `inv_denom` not `inv_rho` for F half-shift.
- **N-7.1** [BUG]: `applyCFLLimitingKernel` is discontinuous at
  `v_new = v_target` boundary (~20% scale jump), contradicting its
  comment claim "smooth exponential damping ... avoids discontinuous
  force jump". Spatial banding likely in v ~ v_target regime.
  Patch: switch regime decision to `v_current` instead of `v_new`.

**Potential bugs (P1)**:
- **N-1.1**: Sprint-1 hybrid `Δu = F/(ρ+0.5K)` deviates from textbook
  Kupershtokh EDM. Internally consistent but compounds with semi-implicit
  Darcy in a non-published way. Document or revert.
- **N-2.2**: `m/(ρ+0.5K)` form ≠ clean Crank-Nicolson Darcy; matches an
  ω-modulated decay rate. Subtle calibration concern, not a bug.
- **N-3.2**: Λ-preserving LES drives ω⁻ → 1.85 under heavy turbulence
  with Λ=3/16. No instability but anti-symmetric channel becomes
  ~over-relaxed.

**Verified clean (NOT bugs despite suspicion)**:
- Equilibrium f_eq formula (constants 1, 3, 4.5, -1.5 all exact in FP32).
- D3Q19 lattice tables (`opposite[]` self-inverse, weights sum to 1,
  2nd-moment isotropy `Σ w c c = cs² I` to FP32 precision).
- TRT neq-split formulation matches `f^+/f^-` decomposition exactly.
- Regularized 2nd-order Hermite projection round-trips Π exactly.
- Hou 1996 algebraic Smagorinsky derivation matches code exactly.
- Guo source term S_q expansion (1986-1992) matches Guo 2002 Eq. 20.

**Action**: 2 confirmed bugs ready for fix → see OUTBOX entry.
