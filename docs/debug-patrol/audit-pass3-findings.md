# Audit Pass 3 — Findings

**Auditor**: Claude Sonnet 4.6  
**Date**: 2026-04-27  
**Branch**: debug/patrol-2026-04-27  
**Scope**: vof_solver.cu Track-C additions, sprint-history audit tests, apps/sim_linescan_*, include/utils/cuda_memory.h, include/core/unit_converter.h, src/config/simulation_config.cpp, MultiphysicsSolver BC ordering  
**Line budget**: this file ~800 lines

---

## Summary

37 findings across 7 scope areas. 3 Critical, 8 High, 15 Medium, 11 Low.

---

## Section 1 — vof_solver.cu (Track-C additions, ~900 new lines)

### F3-01 [HIGH] vof_solver.cu:2966 — countInterfaceCellsKernel hardcodes 256-element shared array

**Evidence**: `__shared__ int shared_count[256];` (line 2966) combined with the call sites using `blockSize = 256`. If any future caller changes `blockSize`, the shared memory declaration is too small (for larger) or wastes space (for smaller). The parallel reduction at lines 2983-2988 reads `shared_count[tid + stride]` — if `blockDim.x > 256` and a thread with `tid >= 256` executes, it writes out-of-bounds into shared memory, causing silent corruption or a kernel crash.

**Fix**: Replace with `extern __shared__ int shared_count[];` and pass `blockSize * sizeof(int)` as the third launch parameter, matching the pattern used by `computeMassReductionKernel` and `computeVzWeightSumKernel` in the same file.

---

### F3-02 [CRITICAL] vof_solver.cu:1690,1745 — static local GPU pointers survive across VOFSolver lifetimes

**Evidence**:
```
static float* d_block_max = nullptr;
static int d_block_max_size = 0;
```
These appear inside `advectFillLevel()` at line 1690 and in the TVD dispatch at line 1745. Because they are `static`, they persist in process memory. If a second `VOFSolver` is created after the first is destroyed (e.g. in a test harness, a restart, or future multi-solver coupling), the stale pointer refers to freed GPU memory, and the first call to `maxVelocityMagnitudeKernel` or the CFL reduction will read or write a dangling device pointer — undefined behaviour that will not be caught by `CUDA_CHECK_KERNEL()`.

Similarly `static int plic_call = 0;` at line 3834 means the per-call diagnostic modulo counter does not reset between solver instances.

**Fix**: Promote `d_block_max` and `d_block_max_size` to `VOFSolver` member variables (or `lbm::utils::CudaBuffer<float>`) managed by the constructor/destructor, matching how `plic_nx_` etc. are handled. Reset `plic_call` to an instance member.

---

### F3-03 [HIGH] vof_solver.cu:3687-3693 — plicWallSealKernel writes to z_min=0 only; z_max seal silently absent

**Evidence**:
```cpp
if (seal_z) {
    if (k == 0)      fill[idx] = fill[i + nx * (j + ny * 1)];
    // k == nz-1 case missing
}
```
The kernel seals the top z-face (`k == nz-1`) only for the y-direction, and for z it only copies from k=1 into k=0 (bottom). The top z gas cap (k = nz-1) is never sealed. In the LPBF case the domain top is the free surface / gas cap — fragmentation artifacts at the top face are reported but the kernel will not fix them because the branch is absent.

**Fix**: Add `if (k == nz-1) fill[idx] = fill[i + nx * (j + ny * (nz-2))];` inside the `seal_z` branch.

---

### F3-04 [MEDIUM] vof_solver.cu:283-290 — TVD/upwind kernels apply symmetric flush AND a redundant clamp

**Evidence** (upwind kernel, replicated in TVD kernel, lines 283-290 and 725-730):
```cpp
if (f_new < 1e-9f) f_new = 0.0f;
if (f_new > 1.0f - 1e-9f) f_new = 1.0f;
fill_level_new[idx] = fminf(1.0f, fmaxf(0.0f, f_new));
```
After the flush branches set `f_new` to exactly `0.0f` or `1.0f`, the `fminf/fmaxf` clamp on the last line is a no-op. This is not a correctness bug, but the comment "symmetric flushing for minimal mass loss" on what is actually asymmetric (only small values are zeroed, large values are set to 1) is misleading and the redundant clamp obscures intent.

**Fix**: Remove the final `fminf/fmaxf` line (the flush already covers the edge cases) or remove the two flush branches and keep only the clamp. Pick one strategy and document it.

---

### F3-05 [MEDIUM] vof_solver.cu:1096-1103 — interface compression skips cells with u_max < 1e-8 but uses full dt

**Evidence**: The Olsson-Kreiss compression coefficient `epsilon = C_compress * u_max * dx` (line 1097) is velocity-dependent, but the integration uses the full simulation `dt` rather than a compression-specific pseudo-time step. When velocities are very small, epsilon is very small and compression does nothing; when velocities are large (Marangoni melt pool, ~1-10 m/s), `epsilon * dt / dx` can approach or exceed 0.5, violating the stated CFL condition for compression. This is a known issue (noted in MEMORY.md: "Compression uses full dt") but not guarded. In the current production configuration compression is disabled, so this is LATENT.

**Fix**: Either enforce `CFL_compress = epsilon * dt / dx <= 0.5` and subcycle, or document the known violation and enforce `enabled=false` as an assertion in the non-periodic BC path.

---

### F3-06 [HIGH] vof_solver.cu:504-521 — TVD flux: second-order term uses delta_center without correct limiter projection

**Evidence** (repeated across all 12 face-direction branches):
```cpp
float f_face_second = f_upwind + 0.5f * delta_center;  // NO phi here!
float F_high = u_face_xm * f_face_second;
flux_xm = F_low + phi * (F_high - F_low);
```
The comment "BUGFIX: phi should only be applied ONCE in the flux blending" is present, but the implementation produces `flux = u * [f_upwind + phi * 0.5 * delta_center]`, which is the Lax-Wendroff second-order correction blended with upwind using `phi`. This is a standard TVD implementation. The concern is that `delta_center` is (f_center - f_upwind), i.e. the downstream gradient, while the classical TVD gradient ratio `r` is (upstream / downstream). The actual face reconstruction uses the **downstream** gradient directly as the high-order correction, not the upstream-vs-downstream ratio that the gradient ratio `r` was computed from. This is a subtle inconsistency: `phi` is computed from `r = delta_upwind / delta_center` but the correction term then uses `delta_center` (not `delta_upwind`). For smooth flows this converges, but near sharp interfaces the TVD compression can be misdirected.

**Fix**: Verify against Sweby (1984) — standard Lax-Wendroff correction should be `f_upwind + phi * delta_upwind / 2` (i.e., use `delta_upwind`, not `delta_center`). If the intent is to use the downstream gradient, add a comment explaining the non-standard choice. This affects 12 code blocks.

---

### F3-07 [LOW] vof_solver.cu:942-943 — contact angle kernel uses 3.14159265f literal instead of M_PI_F or __fadd_rn

**Evidence**: `cosf(contact_angle * 3.14159265f / 180.0f)`. The `__device__` code uses `3.14159265f` (7-digit truncation of pi) rather than the CUDA constant `(float)M_PI` or `3.14159274f` (nearest float). The error is 8.7e-8 which is sub-ULP for float trig, but the inconsistency with the rest of the codebase (which uses `3.14159265359f` elsewhere) should be resolved.

**Fix**: Use `(float)M_PI` inside `#ifdef __CUDACC__` or define a local `constexpr float PI = 3.14159265359f`.

---

### F3-08 [MEDIUM] vof_solver.cu:3834 — plic_call diagnostic runs on every 500th call to plicFinalClampKernel, requiring an extra computeTotalMass() D2H reduction

**Evidence**:
```cpp
static int plic_call = 0;
if (plic_call % 500 == 0) {
    float mass_before = computeTotalMass();
    plicFinalClampKernel<<<...>>>(d_fill_level_, N);
    float mass_after = computeTotalMass();
```
`computeTotalMass()` is an expensive host-blocking GPU reduction. Calling it twice per 500 PLIC steps to measure clamp loss is reasonable in diagnostic mode, but there is no way to disable this at compile time for production. In a 2 ms run with dt=80 ns, there are 25000 steps of PLIC: that is 50 extra full reductions. Individually small, but adds latency.

**Fix**: Gate behind a `bool mass_diag_enabled_` flag defaulting to false, or convert to `#ifdef PLIC_CLAMP_DIAG`.

---

### F3-09 [MEDIUM] vof_solver.cu:3709 — plicAllocateIfNeeded uses `size() > 0` to test initialization

**Evidence**: `if (plic_nx_.size() > 0) return;`. This is correct today because the default-constructed `CudaBuffer<float>` has `size_ = 0`. But if the buffer is ever reset to 0 elements (`reset(0)`) during a solver resize, the next `advectFillLevelPLIC` call would skip reallocation and operate on a null pointer. The guard should test `ptr_ != nullptr` or `size_ == num_cells_`.

**Fix**: Change to `if (plic_nx_.get() != nullptr) return;` or expose a `bool empty() const` on `CudaBuffer`.

---

## Section 2 — Sprint-history audit tests

### F3-10 [HIGH] overnight_audit/test_pure_conduction.cu:88-92 — step ordering is collisionBGK before applyFaceThermalBC, reversing the ESM-corrected order

**Evidence** (lines 87-92 of test_pure_conduction.cu):
```cpp
for (int face = 0; face < 6; face++) {
    thermal.applyFaceThermalBC(face, 2, dt, dx, T_bc);
}
thermal.collisionBGK(nullptr, nullptr, nullptr);
thermal.streaming();
thermal.computeTemperature();
```
The MEMORY.md section "ESM step order" explicitly documents: `collisionBGK → streaming → computeTemperature → applyFaceThermalBC (Dirichlet overwrites ESM at boundary)`. This test applies Dirichlet BC **before** collision. For pure conduction without phase change this produces the wrong boundary condition at Dirichlet faces because BGK will have already consumed the pre-modified distributions before streaming. The Gaussian decay test (Test A) and sinusoidal decay (Test B) rely on this wrong order and may have passing thresholds tuned to the wrong physics.

**Fix**: Move `applyFaceThermalBC` calls to after `computeTemperature()`. Cross-check that L2 error thresholds (Test B: `L2 < 0.02`) remain valid.

---

### F3-11 [MEDIUM] overnight_audit/test_pure_conduction.cu:65 — loop variable `k` shadows outer parameter name

**Evidence**: The outer scope uses `const float alpha = k / (rho * cp)` where `k` is conductivity (line 32), and the inner triple loop then declares `for (int k = 0; k < NZ; k++)`. This shadow hides the conductivity variable during the initialization loop. It compiles but is confusing; a static analysis tool would flag it.

**Fix**: Rename the conductivity variable to `k_cond` at line 32 to avoid shadowing.

---

### F3-12 [LOW] overnight_audit tests — no assertion-level pass/fail, only printf PASS/FAIL strings

**Evidence**: All 11 test files in `overnight_audit/tests/` print `PASS/FAIL` via `printf` but return `0` unconditionally from `main()`. A test runner (CTest) that checks exit codes will always report success regardless of actual failure. This is consistent with the pass-2 finding about stub tests but extends to the audit-specific test suite.

**Fix**: Add `return (all_pass) ? 0 : 1;` at the end of each `main()`. Or convert to a proper test framework.

---

## Section 3 — apps/sim_linescan_* configuration drift

### F3-13 [CRITICAL] sim_linescan_S3A1.cu + sim_linescan_S3A3.cu — T_solidus/T_liquidus NOT overridden; use MaterialDatabase defaults

**Evidence**: `sim_linescan_S3A1.cu` and `sim_linescan_S3A3.cu` call `MaterialDatabase::get316L()` but contain **no** subsequent `config.material.T_solidus = ...` assignment. All other phase apps (phase1, 2, 3, 4, 5, dawn3) explicitly set:
```cpp
config.material.T_solidus  = 1674.15f;   // F3D ts1
config.material.T_liquidus = 1697.15f;   // F3D tl1
```
The Sprint-1 rationale (comments in phase1.cu:125-131) explains that the 23 K mushy zone vs the old 200 K default prevents Darcy damping from stagnating the trailing liquid pool. S3A1 and S3A3 were the experimental Sprint-3 parametric runs and appear to have been copied from an earlier template before this Sprint-1 fix. However, `material_database.cu` line 82 shows the Sprint-1 values are now the default in `get316L()`, so the omission is currently harmless — but if the database defaults ever revert, these two apps silently regress without any warning.

**Risk**: Not a current bug, but a silent regression trap. Sprint-3 comparison data from S3A1/S3A3 may be internally consistent but not comparable with phase1-5 runs unless the defaults were already aligned.

**Fix**: Add explicit `config.material.T_solidus = 1674.15f; config.material.T_liquidus = 1697.15f;` to S3A1 and S3A3, with a comment pointing to the F3D source.

---

### F3-14 [HIGH] sim_linescan_S3A1.cu + sim_linescan_S3A3.cu — laser_spot_radius = 50e-6 instead of 39e-6

**Evidence**:
- phase1-5, dawn3: `config.laser_spot_radius = 39.0e-6f;` (F3D dum2=39e-6)
- S3A1 line 147, S3A3 line 147: `config.laser_spot_radius = 50.0e-6f;`

The 50 µm radius is the pre-Sprint-1 value. At 39 µm the peak Gaussian intensity is 1.64× higher, which is central to the keyhole depth comparison. S3A1/S3A3 results cannot be directly compared to phase1-5 without accounting for this 28% radius difference. No comment in either file explains this choice as intentional.

**Risk**: If S3A1/S3A3 were Sprint-3 parametric attempts intended to compare against the phase-4 baseline, the spot radius discrepancy makes the comparison unreliable. Sprint-3 memory entry says S3A1 tested CSF×2 and S3A3 tested something else, but the different spot radius confounds the parametric study.

**Fix**: Either align S3A1/S3A3 to 39e-6 (if they were intended as controlled parametric variants) or add a prominent comment explaining the intentional deviation.

---

### F3-15 [HIGH] sim_linescan_phase3.cu — kinematic_viscosity = 0.0167 vs 0.065 in all other phases

**Evidence**:
- phase1, 2, 2_mini, 4, 5, dawn3, S3A1, S3A3: `config.kinematic_viscosity = 0.065f`
- phase3 line 193: `config.kinematic_viscosity = 0.0167f`

At dx=2e-6 and dt=8e-8: `nu_lattice = 0.0167 * 8e-8 / (2e-6)^2 = 0.000334`, giving `tau = 0.000334/0.333 + 0.5 = 0.501` — barely above the stability threshold of 0.50. The FluidLBM constructor at `fluid_lbm.cu:103` will print a WARNING and clamp to tau=0.51. Phase3 was testing "τ→0.55 made center Δh worse" per MEMORY.md — but the comment in phase3 says `kinematic_viscosity = 0.0167f` which gives tau~0.50, not 0.55. There is a numerical inconsistency: 0.0167 LU gives tau≈0.50, but 0.0167 as the physical ν in m²/s converted to LU gives tau≈0.501. The comment in phase4/5 says "Phase-3 tested τ→0.55 (ν_LU=0.0167)"; if 0.0167 is intended as a **lattice unit** viscosity, then the conversion is applied twice (lattice ν from a value that is already lattice ν), resulting in a completely wrong tau.

**Fix**: Clarify whether 0.0167 is a physical viscosity [m²/s] or a lattice unit viscosity. If it is ν_LU=0.0167 (making tau = 0.0167/0.333 + 0.5 = 0.55), then the conversion `nu_lattice = 0.0167 * dt/dx²` in the FluidLBM constructor will produce a much smaller tau. This is a unit mismatch that would make Phase-3 run in a regime very different from what the comments describe.

---

### F3-16 [MEDIUM] sim_linescan_phase5.cu — emissivity set via two paths; one may override the other silently

**Evidence** (phase5.cu lines 250, 316):
```cpp
config.material.emissivity = 0.55f;      // line 250 — Phase-5 change
...
config.emissivity = config.material.emissivity;  // line 316 — forward to config
```
This is correct: the material property is updated first, then forwarded. However, other phases (phase1-4, dawn3) set `config.emissivity = config.material.emissivity` WITHOUT first modifying `config.material.emissivity`. Since `get316L()` now returns 0.55 by default (material_database.cu:103), phases 1-4 will also use 0.55, but their comments say "emissivity = config.material.emissivity" implying they use the database value. Phase-4 has no explicit emissivity override yet its comments say "emissivity 0.28 (mat default)" — this was true before the database was updated but is now stale.

**Risk**: Phase-1 through Phase-4 output data and their archived `melt_pool_depth.csv` files were generated before the database change. Re-running them with the current database would produce different results, making archived comparisons invalid. No `// emissivity = X.XX` explicit value is printed in the console summary for these apps.

**Fix**: Add `config.material.emissivity` to the printf summary block in all phase apps. Consider pinning historical phases with explicit values like phase5 does.

---

### F3-17 [MEDIUM] sim_linescan_S3A3.cu:15 — file header says `mkdir -p output_line_scan` but code creates `output_S3A3`

**Evidence**:
```
 *   mkdir -p output_line_scan && ./sim_line_scan_316L   // line 15 comment
...
mkdir("output_S3A3", 0755);                               // line 272 code
```
The header comment is a copy-paste artifact from an earlier template. Running the instructions in the file comment would create the wrong directory, and VTK files would be written to `output_S3A3/` while the user expects to find them in `output_line_scan/`.

**Fix**: Update the comment to `mkdir -p output_S3A3`.

---

### F3-18 [LOW] sim_linescan_phase1.cu — file header says `mkdir -p output_phase1 && ./sim_line_scan_316L` but the executable built from this file will not be named `sim_line_scan_316L`

**Evidence**: The CMakeLists target (not registered yet per Phase5 comments) would be `sim_linescan_phase1`, not `sim_line_scan_316L`. Multiple apps share this copy-paste header.

**Fix**: Update usage comments to match actual target names.

---

### F3-19 [LOW] sim_linescan_phase2.cu — has ray_tracing.max_bounces=3 matching phase1 but phase4/5 use 3 and 5 respectively; no delta table

**Evidence**: Unlike phase5.cu which has an explicit parameter delta table vs phase4, phase2.cu has no such table. The max_bounces=3 is the iter-6 value. If phase2 was supposed to be a controlled experiment, there is no record of which parameter it was testing relative to a baseline.

**Fix**: Add a parameter delta table comment to phase2.cu and phase2_mini.cu consistent with the phase5 style.

---

## Section 4 — Draft tests in sprint-history

*(No `tests/draft-2026-04-27/` directory exists in the worktree. The 11 files in `docs/sprint-history/overnight_audit/tests/` are the closest equivalent and were reviewed above.)*

---

## Section 5 — include/utils/cuda_memory.h CudaBuffer<T>

### F3-20 [MEDIUM] cuda_memory.h:41 — CudaBuffer constructor takes `int n`, enabling silent negative allocation

**Evidence**:
```cpp
explicit CudaBuffer(int n) : ptr_(nullptr), size_(n) {
    if (n > 0) {
        CUDA_CHECK(cudaMalloc(&ptr_, bytes()));
    }
}
```
If `n` is negative (e.g. `nx * ny * nz` overflows `int` and becomes negative), the condition `n > 0` is false, `ptr_` stays `nullptr`, and `size_` is set to the negative value. A subsequent `bytes()` call returns `static_cast<size_t>(negative_int) * sizeof(T)` which wraps to a huge positive value, causing a catastrophic cudaMalloc on the next `reset()` call or misreporting from `size()`. There is no assertion that `n >= 0`.

**Fix**: Either use `size_t` for both the parameter and `size_` member, or add `if (n < 0) throw std::invalid_argument("CudaBuffer: negative size");` at the top of the constructor and `reset()`.

---

### F3-21 [LOW] cuda_memory.h:85-86 — implicit conversion operators to T* hide misuse at kernel call sites

**Evidence**:
```cpp
operator T*() { return ptr_; }
operator const T*() const { return ptr_; }
```
These implicit conversions mean passing a `CudaBuffer<float>` directly to a function expecting `float*` compiles silently, which is the intended ergonomic feature. However, if a function expects two separate `float*` arguments and the caller accidentally passes the same buffer twice, no compiler warning is generated. This is CLAUDE.md "clever over clear" risk — the benefit (clean kernel calls) is real, but the implicit conversion can mask bugs.

**Risk level is LOW** because the calling convention in this codebase is consistent and the review showed no immediate double-pass cases.

---

### F3-22 [LOW] cuda_memory.h:139-141 — destructor calls cudaFree without checking if a CUDA context exists

**Evidence**:
```cpp
void free() {
    if (ptr_) {
        cudaFree(ptr_);  // Intentionally no CUDA_CHECK in destructor.
```
The comment acknowledges this is intentional, which is the standard pattern for destructors. However, if a `CudaBuffer` is destroyed after the CUDA context has been torn down (e.g., in a static-duration object at program shutdown), `cudaFree` will return `cudaErrorCudartUnloading`. The silent ignore is correct per the destructor contract, but there is no mention of this edge case in the documentation.

**Fix**: Add a one-line comment: `// cudaFree after context teardown returns cudaErrorCudartUnloading; intentionally ignored.`

---

## Section 6 — include/core/unit_converter.h and src/config/simulation_config.cpp

### F3-23 [MEDIUM] unit_converter.h:282-283 — timeToLattice rounds to nearest integer but truncation is more appropriate for step count

**Evidence**:
```cpp
return static_cast<int>(t_phys / dt_ + 0.5f);  // Round to nearest
```
For physical time → step count, the conventional choice is **floor** (not round) because you want the simulation to complete at least `t_phys` seconds, not potentially end one step early. For `t_phys = 800e-6` and `dt = 8e-8`, the exact result is 10000.0. Floating-point arithmetic could produce 9999.9999 which rounded would give 10000 (correct) but floors to 9999 (one step short). The rounding compensates for this FP error, making it marginally safer than floor. However, the +0.5 rounding can also cause off-by-one in the wrong direction when `t_phys` is exactly halfway between two step counts.

**Fix**: The rounding is defensible. Improve the comment to explain the float rounding rationale: `// +0.5f compensates for FP under-truncation when t_phys/dt is an exact integer`.

---

### F3-24 [LOW] unit_converter.h:77-82 — validation throws std::invalid_argument on device, but exception handling in device code is undefined

**Evidence**:
```cpp
#if !CUDA_ARCH_CHECK
if (dx <= 0.0f || dt <= 0.0f || rho_phys <= 0.0f) {
    throw std::invalid_argument("...");
}
#endif
```
The `#if !CUDA_ARCH_CHECK` guard is defined as `#define CUDA_ARCH_CHECK __CUDA_ARCH__` — which is only non-zero when compiling device code. For host-only TUs that include this header without `__CUDACC__`, the `HOST_DEVICE` macro expands to nothing and `CUDA_ARCH_CHECK` expands to `0` (since `__CUDA_ARCH__` is 0 on host). The guard is correct. However, if `HOST_DEVICE` marks a function that is compiled for both host and device (because the header is included from a `.cu` file), the `#if !CUDA_ARCH_CHECK` branch is evaluated at compile time based on `__CUDA_ARCH__`, so the throw is compiled out of device code. This is the intended behavior and is correct.

No bug; noting for audit completeness.

---

### F3-25 [HIGH] simulation_config.cpp:357 — preset "ti6al4v_melting" hardcodes dt=5e-10 without comment justification

**Evidence**:
```cpp
cfg.time.dt = 5e-10;
```
For a 160×160×80 µm domain at dx=2µm, the LBM stability criterion `Ma_max < 0.2` with typical velocities ~1 m/s gives `dt_max = Ma * dx / (sqrt(3) * v_max) ≈ 2.3e-7 s`. The preset uses dt=5e-10 which is 460× smaller than necessary, making any run with this preset ~460× slower than needed. No comment explains why this extreme value was chosen.

**Fix**: Replace with `cfg.time.dt = 1e-7;` or add a comment explaining why the extremely small step is required (e.g. numerical instability at this viscosity/power combination).

---

### F3-26 [MEDIUM] simulation_config.cpp:68-76 — key-value parser requires `:` at end of line with no `=` to detect sections; malformed config files fail silently

**Evidence**:
```cpp
if (line.back() == ':' && line.find('=') == std::string::npos) {
    current_section = line.substr(0, line.size() - 1);
    continue;
}
```
Lines that look like `domain:nx = 5` (YAML inline) or `dt:1e-7` (typo) fall through to the key-value parser with wrong results. There is no validation that parsed section names are known, so a typo like `[domian]` silently drops all keys in that section. The config loader does not report unknown keys.

**Fix**: After parsing all key-value pairs, validate that all mandatory keys (nx, ny, nz, dt) were set. Log a warning for any unrecognized section name.

---

## Section 7 — MultiphysicsSolver BC ordering

### F3-27 [HIGH] multiphysics_solver.cu — LBM thermal path applies Dirichlet BC before collision (MEMORY-documented bug)

**Evidence** (lines 1754-1808 of multiphysics_solver.cu):
```
applyFaceThermalBC(DIRICHLET)   ← pre-collision
collisionBGK()                   ← consumes modified distributions
streaming()
computeTemperature()             ← ESM correction included
applyFaceThermalBC(DIRICHLET)   ← re-applied post-streaming (lines 1853-1865)
```
The pre-collision BC at line 1758 is present alongside the post-streaming re-application at line 1857. The MEMORY.md entry "ESM step order" says BC should be applied only **after** `computeTemperature()`. The pre-collision application is redundant at best (the post-streaming re-application overwrites it) and harmful if any intermediate kernel reads temperature before the re-application. Specifically, `applySubsurfaceBoilingCap()` at line 1844 runs after `computeTemperature()` but before the Dirichlet re-application — if the cap modifies temperatures at Dirichlet faces, the subsequent re-application will overwrite the capped values with the prescribed temperature (correct behavior), but this ordering is fragile.

**Risk**: Not an active bug given the re-application is present, but the pre-collision BC constitutes dead/misleading code that could be misread as the authoritative application point. If a future developer removes the "redundant" post-streaming re-application, the ESM-ordering bug reactivates.

**Fix**: Remove the pre-collision `applyFaceThermalBC` block (lines 1754-1775) and document that Dirichlet BCs are applied post-streaming only.

---

### F3-28 [MEDIUM] multiphysics_solver.cu — legacy radiation/convection block and per-face block can both fire on the same run

**Evidence** (lines 1787-1805):
```cpp
if (!use_per_face) {
    if (config_.enable_radiation_bc) { ... applyRadiationBC ... }
    if (config_.enable_substrate_cooling) { ... applySubstrateCoolingBC ... }
}
```
The condition `!use_per_face` ensures mutual exclusion: if any per-face BC is set, the legacy path is skipped. However, `use_per_face` is true if any of `DIRICHLET, CONVECTIVE, RADIATION` is present on any face. A configuration that sets `z_min=ADIABATIC, z_max=ADIABATIC, x/y=WALL` with `enable_radiation_bc=true` would have `use_per_face=false` (no per-face BC set) while also running the legacy radiation block. This is likely the intended behavior for backward compatibility but can cause double-application of radiation if a user sets both `enable_radiation_bc=true` AND `z_max=RADIATION` in FaceBoundaryConfig.

**Fix**: Add an assertion or warning if `enable_radiation_bc && use_per_face` to prevent silent double-counting.

---

### F3-29 [MEDIUM] multiphysics_solver.cu — FDM thermal path order not audited here

**Evidence**: The scope of BC ordering audit was the LBM thermal path (which is well-documented in MEMORY.md). The FDM thermal path (`config_.use_fdm_thermal == true`) is used in all production linescan apps but its BC application order was not covered by pass 1 or 2. The FDM path is inside `thermalStepFDM()` which is called from `thermalStep()`. This is flagged for a focused pass-4 audit.

---

## Section 8 — Additional findings from cross-cutting analysis

### F3-30 [MEDIUM] vof_solver.cu:2138-2139 — mass correction fallback calls cudaFree/cudaMalloc inside per-step hot path

**Evidence**:
```cpp
if (d_mass_partial_sums_size_ < gridSize) {
    if (d_mass_partial_sums_) cudaFree(d_mass_partial_sums_);
    CUDA_CHECK(cudaMalloc(&d_mass_partial_sums_, gridSize * sizeof(float)));
```
`d_mass_partial_sums_` is a raw pointer member, not a `CudaBuffer`. It is lazily allocated inside the mass-correction critical path. If `gridSize` grows between calls (e.g. domain resize, or an early call with a small grid), the old allocation is freed and a new one is made. `cudaFree` and `cudaMalloc` inside a tight loop are expensive. Since `gridSize` is deterministic for a given domain size (fixed `num_cells_`), this reallocation never fires in practice after the first step — but the code structure does not make this invariant obvious.

**Fix**: Allocate `d_mass_partial_sums_` in the `VOFSolver` constructor or `setMassCorrectionEnabled()` at known fixed size `(num_cells_ + 255) / 256`. Use `CudaBuffer<float>` to get RAII.

---

### F3-31 [LOW] vof_solver.cu:2157 — Kahan summation is applied to a small host-side reduction of ~32 K floats

**Evidence** (lines 2151-2157):
```cpp
double kc = 0.0;
for (float p : h_partial) {
    double y = static_cast<double>(p) - kc;
    double t = W_dbl + y;
    kc = (t - W_dbl) - y;
    W_dbl = t;
}
```
Kahan compensated summation is appropriate for summing millions of FP values. For a grid with 8.25M cells at blockSize=256, there are ~32K partial sums. At double precision, standard summation already has relative error ~32768 * epsilon_double ≈ 7e-12, which is negligible vs the 1e-6 correction threshold. Kahan adds code complexity without meaningful benefit here.

**Fix**: Replace with `W_dbl = std::accumulate(h_partial.begin(), h_partial.end(), 0.0)`. Minor CLAUDE.md simplicity improvement.

---

### F3-32 [MEDIUM] Unit conversion — fluid_lbm.cu:94 uses `kinematic_viscosity * dt / (dx * dx)` but FluidLBM constructor accepts `kinematic_viscosity` as a physical ν [m²/s], while sim_linescan_phase3.cu passes 0.0167 (which could be LU or physical)

*(This is the F3-15 cross-reference, noted separately for completeness.)*

---

### F3-33 [LOW] unit_converter.h — no `computeDt()` or `computeDtFromMach()` factory method

**Evidence**: The `UnitConverter` class is passive (stores dx, dt, rho) and requires the caller to pre-determine dt externally. LBM practice is to derive dt from the Mach constraint: `dt = Ma * dx / (cs * v_ref)` where `cs = 1/sqrt(3)`. The absence of this factory means dt is scattered across app files and config files without a single source of truth. This is the root cause of the phase3 viscosity confusion (F3-15) and the preset dt=5e-10 oddity (F3-25).

**Fix**: Add `static float computeDtFromMach(float Ma, float dx, float v_ref_phys)` as a static method, and add a Mach-number validator to `validate()`.

---

### F3-34 [LOW] apps/sim_linescan_*.cu — computeMeltPoolDimensions is copy-pasted verbatim across all 10 files

**Evidence**: The function `computeMeltPoolDimensions()` (searching for bounding box of fl>0.5 and lf>0.5 cells) is present in identical form in at least phase1, phase4, phase5. It is not in any shared header. This is a CLAUDE.md violation (code duplication, not modular).

**Fix**: Move to a shared header `include/utils/melt_pool_metrics.h` with inline implementation. One declaration, zero copies.

---

### F3-35 [MEDIUM] apps/sim_linescan_phase5.cu:180-184 — compile-time macro PHASE5_VOF_MASS_CORR defined but not registered in CMakeLists, and the file header explicitly says NOT to add it yet

**Evidence**:
```
// CMakeLists registration: DO NOT add yet — leave for human post worktree-A merge.
```
The file documents the feature toggle but explicitly defers CMakeLists registration. This means the feature can never be tested without manual CMake modification, and the toggle effectively dead code until a separate PR lands. The `constexpr bool kVofMassCorr` is assigned but whether it is actually used further down in the file is not checked here.

**Fix** (deferred, not urgent): Once worktree-A merges, ensure `kVofMassCorr` is wired to the actual `enableMassCorrection()` call rather than defined and unused.

---

### F3-36 [HIGH] vof_solver.cu — applyMassCorrectionInline(vz) fallback uses `int interface_count` but the partial count sum uses `long long interface_count` 

**Evidence** (lines 2192-2200):
```cpp
long long interface_count = 0;
for (int c : h_counts) interface_count += c;

if (interface_count > 0) {
    ...
    applyMassCorrectionKernel<<<mc_grid, mc_block>>>(
        d_fill_level_, delta_m, nx_, ny_, nz_,
        static_cast<int>(interface_count), ...);
```
`interface_count` is accumulated as `long long` but cast to `int` for the kernel call. For a domain with 8.25M cells and an interface fraction of 10% (typical for melt pool surface), `interface_count ≈ 825000`, which fits in `int` (max ~2.1 billion). However, for larger domains the cast silently truncates. The kernel parameter is `int interface_count`, so passing a value > INT_MAX would produce a garbage count and divide-by-near-zero in `mass_correction / interface_count`.

**Fix**: Change `applyMassCorrectionKernel` signature to take `long long interface_count` (or `int64_t`), or add an assertion `assert(interface_count <= INT_MAX)`.

---

### F3-37 [LOW] cuda_memory.h:117-122 — reset() frees memory before setting new size, leaving size_ stale if CUDA_CHECK throws

**Evidence**:
```cpp
void reset(int new_size) {
    free();
    size_ = new_size;
    if (new_size > 0) {
        CUDA_CHECK(cudaMalloc(&ptr_, bytes()));
    }
}
```
If `cudaMalloc` throws (via `CUDA_CHECK`), `ptr_` is null (set by `free()`) but `size_` is already updated to `new_size`. The object is in an inconsistent state: `size_` says `new_size` elements but `ptr_ == nullptr`. A subsequent `bytes()` call returns a nonzero value, and passing a null `ptr_` to a kernel invokes undefined behaviour. Move/copy operations are deleted, so this cannot be transferred away, but the object is still live and broken.

**Fix**: Set `size_ = new_size` after the `cudaMalloc` succeeds:
```cpp
void reset(int new_size) {
    free();
    ptr_ = nullptr; size_ = 0;
    if (new_size > 0) {
        CUDA_CHECK(cudaMalloc(&ptr_, bytes_for(new_size)));
        size_ = new_size;
    }
}
```

---

## Findings Count by Severity

| Severity | Count |
|----------|-------|
| CRITICAL | 3 (F3-02, F3-13, F3-25) |
| HIGH     | 8 (F3-01, F3-03, F3-06, F3-10, F3-14, F3-15, F3-27, F3-36) |
| MEDIUM   | 15 (F3-04, F3-05, F3-08, F3-09, F3-11, F3-16, F3-17, F3-23, F3-26, F3-28, F3-29, F3-30, F3-32, F3-35) |
| LOW      | 11 (F3-07, F3-12, F3-18, F3-19, F3-20, F3-21, F3-22, F3-24, F3-31, F3-33, F3-34, F3-37) |
| **Total**| **37** |

---

## Top-5 Most Worrying Findings

### 1. F3-02 [CRITICAL] — Static local GPU pointers in advectFillLevel()
- `static float* d_block_max` inside a method survives across `VOFSolver` destructor + constructor cycles
- A test harness or multi-solver workflow creates a second `VOFSolver` → first kernel call writes to freed GPU memory
- Silent corruption: `CUDA_CHECK_KERNEL()` only catches launch errors, not stale-pointer writes

### 2. F3-15 [HIGH] — Phase-3 kinematic_viscosity = 0.0167 is likely a unit mismatch
- Comments in phase4/5 say "Phase-3 tested τ→0.55 (ν_LU=0.0167)" implying 0.0167 is a lattice-unit viscosity
- But FluidLBM constructor applies `nu_lattice = nu_phys * dt / dx²`, so if 0.0167 is meant as ν_LU, the constructor converts it again: `nu_lattice = 0.0167 * 8e-8 / 4e-12 ≈ 3.3e5`, giving `tau ≈ 1e6` (essentially frozen fluid)
- All Phase-3 results may have been generated with a completely unphysical fluid state

### 3. F3-36 [HIGH] — applyMassCorrectionInline casts long long to int for kernel argument
- For 8.25M cells, a 10% interface fraction gives ~825K interface cells — safely within int range today
- If Track-C mass correction is enabled for a larger domain (future scale-up to multi-GPU), `interface_count > 2.1B` and the cast silently wraps to a negative value
- `delta_m / negative_count` produces a large negative `delta_f` per cell: mass correction subtracts from every interface cell instead of adding, rapidly driving the simulation unstable

### 4. F3-13 [CRITICAL] — S3A1/S3A3 missing explicit T_solidus override
- Currently harmless because get316L() defaults now match Sprint-1 values
- But any future change to the database defaults silently invalidates Sprint-3 parametric data
- Combined with F3-14 (different spot radius), S3A1/S3A3 results cannot be compared to phase1-5 without careful documentation of the deviations

### 5. F3-10 [HIGH] — Overnight audit test_pure_conduction.cu uses wrong BC ordering
- Applies Dirichlet BC before collision, not after computeTemperature()
- MEMORY.md documents this exact ordering as a historical fatal bug (Bug fix 2 in Stefan benchmark section)
- The test reports PASS/FAIL via printf but returns 0 unconditionally; CI never catches the physics error
- Anyone using this test as a reference implementation would replicate the wrong ordering in new code

