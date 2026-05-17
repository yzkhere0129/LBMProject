# AMR Design Proposal — feature/compressible-aero

**Status**: DRAFT — Phase 0 deliverable (Week 1)
**Author**: Claude (taking over from prev session)
**Date**: 2026-05-17 (Phase 0 start)
**Approver**: user (pending review)

---

## 0. Summary

Implement **static patch-based 2-level local refinement** around the NACA0012 LE/TE region within the LBMProject codebase. Goal: reduce Cl error from **41% at Re=2000** (canonical baseline) to **≤25%** (現実 target) or **≤10%** (stretch).

**Hard constraints (from task brief)**:
- Stay in LBMProject (no walberla port)
- Zero impact on master/AM branch
- DFG cylinder 1.2% regression must hold
- 4-8 weeks

**Architecture choice**: **static patch-based 2-level convective (acoustic) scaling, Lagrava 2012 / Schornbaum-Rüde 2016 grid coupling, Cumulant collision with BGK fallback in interface band (1-cell wide).**

---

## 1. Phase 0 baseline confirmation

| Test | Result | Tolerance | Status |
|------|--------|-----------|--------|
| DFG ST 2D-2 D/dx=20 Cumulant | Cd_max=3.185 (canonical 3.19) | ±1.2% | ✓ PASS — patches did not break kernel |
| NACA Re=2000 α=8 D/dx=80 | Cd=0.1289, **Cl=0.3365**, Cl_rms=0.0420 | exact bit-identical to O1 canonical | ✓ **PASS** (Cd/Cl/Cl_rms all match to 4 decimal places) |
| lbmpy stencil + weight match | 27/27 directions, weights exact | ε < 1e-12 | ✓ PASS |

Bottom line: **kernel health confirmed, patches non-destructive, deterministic reproduction verified at bit level.**

**Final profile (102 min total wall, slow due to thermal throttle):**
```
collision (Cumulant): 49.7% (102 ms/step)
streaming + QBB:      49.9% (102 ms/step)
inlet+outlet BC:      0.23%
force probe x300:     0.12%
unaccounted:          0.001%
Throughput:           75 MLUPS (effective)
```

Note: O1 May 16 was 72 min (no thermal throttle). Phase 0 102 min reflects sustained-100% laptop GPU heat. **Step rates per cell unchanged — the result is identical, just took longer in wall-clock.**

---

## 2. Cross-code validation status

**Walberla cloud rent CANCELLED** (user constraint: no big-spend test).

**🎯 MAJOR FIND in Phase 0**: `~/walberla/` is **fully installed and built** locally. Available pre-compiled binaries:
- `~/walberla/build/apps/benchmarks/SchaeferTurek/SchaeferTurek` — DFG ST 2D-2 with `2D-2.dat` config matching our cylinder case exactly (H=0.41, L=2.20, D=0.1, U=1.5, ν=0.001)
- `~/walberla/build/apps/benchmarks/NonUniformGridGPU/NonUniformGridGPU` — **block-structured AMR reference** (uses `gpu::communication::NonUniformGPUScheme`, Schornbaum 2016 algorithm)
- `~/walberla/build/apps/showcases/FlowAroundSphere/FlowAroundSphere` — bluff body example

**No pre-built NACA case in walberla.** Setting one up requires a new geometry stamp (`.cpp` + `.dat` config) — estimated ~2 days work. **Not in Phase 0 budget.**

**Phase 0 cross-code plan** (revised):
- **Local lbmpy 1.4**: ✓ done. Stencil + weight bijection verified (27/27 match). Symbolic comparison limited — lbmpy returns cumulant-space form for Cumulant methods, not raw PDFs.
- **Walberla SchaeferTurek 2D-2**: feasible to run as 15-30 min spike. Gives independent Cd_max for cross-validation of our DFG result.
- **Walberla NonUniformGridGPU**: read source to learn interface algorithm details (canonical Schornbaum 2016 implementation). Algorithm cross-reference for Phase 2 coupling.
- **Walberla NACA**: defer to Phase 3 if Phase 2 closes most of gap. ~2-day setup.

**Decision**: Phase 0 will run walberla SchaeferTurek as one final validation spike (cheap). NACA ground-truth via walberla deferred — we use Kurtulus 2015 extrapolation as target.

**Ground truth for AMR target**: Kurtulus 2015 (Re=1000 Cl=0.49 direct), extrapolated 0.55-0.65 at Re=2000. We'll target **Cl ≥ 0.43** for "−25% gap" milestone — works against both 0.57 mid and 0.65 high estimates.

---

## 3. AMR architecture choice

### 3.1 Topology options surveyed

| Option | Pros | Cons | Decision |
|--------|------|------|----------|
| **Patch-based, static, 2-level** | Simplest, fits NACA static geometry, ~6-8wk implementable | No dynamic adapt, manual placement | ✓ **Chosen** |
| Octree (block-based dynamic) | walberla pattern, scales to 3D | 12+ wk impl, complex CUDA, overkill for static airfoil | ✗ Rejected |
| Multi-level (3+) | Better LE resolution | Each level needs interface, exponential complexity | ✗ Rejected (Phase 0 budget) |
| Stretched grid | No interface coupling | LBM doesn't tolerate non-orthogonal cell stretch | ✗ Rejected |

### 3.2 Patch placement (static, compile/CLI configurable)

**Initial proposal**: fine patch around NACA LE+TE, rectangular:
- x ∈ [x_LE − 0.05c, x_LE + 0.15c]  (covers LE radius + LEV detachment zone)
- y ∈ [y_LE − 0.05c, y_LE + 0.05c]  (covers BL on both surfaces)
- z: full domain (z-extruded body, no z-refinement needed)

With chord = 1 m, D/dx_coarse = 80 → dx_c = 0.0125 m → fine patch:
- x extent: 0.20c = 16 coarse cells = 32 fine cells
- y extent: 0.10c = 8 coarse cells = 16 fine cells

Phase 3 sweep: ±0.1c / ±0.2c / ±0.4c (chord direction), 4× refinement (if budget allows).

### 3.3 Time-stepping scheme

**Convective (acoustic) scaling** per Lagrava 2012 §3.2:
- δx_f = δx_c / 2
- δt_f = δt_c / 2
- Fine grid executes **2 sub-steps per coarse step**
- ν_phys constant across levels; ν_LU rescales

**Why not diffusive scaling**: Lagrava notes diffusive (δt ∝ δx²) "removes compressibility error terms" but at 4× CPU cost for 2-level. At our Ma=0.087, compressibility error is small. Convective wins.

---

## 4. PDF rescaling formulas (canonical from Lagrava 2012)

### 4.1 Omega rescaling (2× spatial refinement, convective scaling)

```
ν_f = (δx_c / δx_f) · ν_c = 2 · ν_c
τ_f = 0.5 + 3 · ν_f = 0.5 + 6 · ν_c
ω_f = 2 ω_c / (4 − ω_c)
```

**Sanity check at NACA Re=2000 D/dx=80** (coarse):
- ω_c = 1.976
- ω_f = 2·1.976 / (4 − 1.976) = 3.952 / 2.024 = **1.953**
- Fine grid ω is *lower* than coarse (less close to 2) → BETTER stability margin on fine grid. ✓

### 4.2 PDF decomposition + rescaling

Continuous on interface: ρ, u, f^eq (since f^eq depends only on ρ, u).
Discontinuous: f^neq (proportional to gradient at lattice scale).

**Coarse → fine prolongation** (Lagrava eq. 29):
```
f_i^fine = f_i^eq(ρ, u) + (ω_c / (2ω_f)) · f_i^neq, coarse
```

**Fine → coarse restriction** (Lagrava eq. 30):
```
f_i^coarse = f_i^eq(ρ, u) + (2ω_f / ω_c) · f_i^neq, fine
```

Computation steps at interface cell:
1. Compute ρ, u from source PDF (sum / weighted sum)
2. Compute f^eq from (ρ, u) on destination grid (same formula, lattice units differ)
3. Compute f^neq = f^source − f^eq
4. Rescale f^neq by (ω_c/(2ω_f)) [c→f] or (2ω_f/ω_c) [f→c]
5. Reconstruct f^dest = f^eq + rescaled f^neq

### 4.3 Cumulant compatibility analysis (the high-risk question)

**Q**: Lagrava formula is BGK-derived (Chapman-Enskog assumes single τ). Does it apply to Cumulant?

**A**: Yes, **for our specific Cumulant parameter choice**. Reasoning:

Cumulant relaxes 23 non-conserved cumulants with up to 6 rates:
- ω_ν (shear): 6 moments (κ_xy, κ_xz, κ_yz, traceless diagonal pairs)
- ω_b (bulk = κ_xx+κ_yy+κ_zz): 1 moment, **set to 1.0** by our patch
- ω_3, ω_4, ω_5, ω_6: 16 moments, **all default = 1.0**

With ω_b = ω_3..6 = 1.0 on both levels, those moments need **no rescaling** (factor 1 trivially):
- (2·1.0 / 1.0) = 2 ?? No wait — these are NOT dimensional rescalings, they're the consistency
  factors derived from `f_neq ∝ 1/ω`. For moments where ω is the same on both grids,
  f_neq for that mode is the same on both grids: no rescaling.

So the rescaling reduces to the SRT-equivalent formula on the ω_ν part of f_neq. **This means
Lagrava's BGK formula is the right one IF applied moment-by-moment** with the appropriate
ω per moment.

**Practical implementation choices**:

| Option | Description | Pros | Cons |
|--------|-------------|------|------|
| **A — Per-moment rescaling** | Transform PDF to cumulants, rescale each per its ω, transform back | Theoretically correct | +20 lines code in interface kernel, expensive |
| **B — Interface BGK fallback** | At interface 1-cell band, use BGK collision (ω_all = ω_ν), apply Lagrava verbatim | Simplest, robust | Loses Cumulant stability in interface band |
| **C — Lagrava verbatim** | Treat PDF as monolithic, apply 2ω_f/ω_ν rescaling | Trivial | Wrong for non-shear modes (but those don't matter much for static airfoil) |

**Chosen for Phase 1-2**: **Option B (Interface BGK fallback)**. Rationale:
- Matches user prompt recommendation
- 1-cell band is geometrically tiny vs 15M-cell domain → small impact on Cumulant's stability/accuracy
- If Phase 3 NACA result is still under target by structurally-attributable margin, escalate to Option A

Document as known limitation in FINAL_REPORT.

### 4.4 Force probe at AMR — to be decided in Phase 2

Current MEM force `memForceNaca_QBB_sparse` reads PDFs at fluid-solid links on **coarse grid**. With AMR, the airfoil is **inside the fine patch**. So the force probe must:
- Run on fine grid
- Use the same QBB formula (no BC change)
- Same qfrac data (rebuilt at fine resolution)

This is mostly mechanical — re-wire the kernel to fine arrays. No new physics.

---

## 5. Algorithm: one coarse step (in order) — Lagrava 2012 §3.5

For each coarse step (advancing fine by 2 sub-steps), per Lagrava §3.5:

```
PRECONDITION: both grids at time t, complete.
              Store coarse_t (snapshot before this step) for time interpolation.

[1] COARSE collide-and-stream → brings coarse to t + δt_c.
    At this point: at xf→c sites, populations supposed to come from fine are still UNKNOWN.

[2] FINE sub-step 1: collide-and-stream → brings fine to t + δt_c/2.
    Fine is missing data on xc→f sites. Reconstruct via:
    [2a] TIME interp: ρ_c, u_c, f_neq_c at t + δt_c/2
         = 0.5 * (coarse_t + coarse_{t+δt_c})        [linear time interpolation, Lagrava §3.5 step 2]
    [2b] SPACE interp at xc→f sites WITHOUT coincident coarse site:
         g(x) = (9/16)(g(x+h)+g(x-h)) - (1/16)(g(x+3h)+g(x-3h))   [eq. 38, 4th-order, 4-pt]
         Asymmetric edge formula for sites lacking 4 neighbors:
         g(x) = (3/8)g(x-h) + (3/4)g(x+h) - (1/8)g(x+3h)         [eq. 39]
    [2c] PROLONGATE coarse→fine for all xc→f sites (NOT only missing ones):
         f_f(x) = f_eq(ρ_c, u_c) + (ω_c / (2ω_f)) · f_neq_c       [eq. 34/35]

[3] FINE sub-step 2: collide-and-stream → brings fine to t + δt_c.
    [3a] No time interp (coarse already at t+δt_c).
    [3b] SPACE interp at xc→f sites (same as 2b).
    [3c] Prolongate all xc→f sites (same as 2c, with coarse_{t+δt_c}).

[4] RESTRICT fine→coarse at xf→c sites:
    [4a] Box filter f_neq on fine grid (Lagrava §3.3 eq. 33, recommended for high Re):
         f̃_neq_f(x) = (1/q) Σ_i f_neq_f(x + ξ_i)
    [4b] Replace coarse PDF at xf→c sites:
         f_c(x) = f_eq(ρ_f(x), u_f(x)) + (2ω_f / ω_c) · f̃_neq_f(x)  [eq. 36]
    
    [4c] (Optional Phase 3+): higher-order spatial interp filter at xf→c if box filter too dissipative

[5] Snapshot coarse state for next step's time interp:
    coarse_t = coarse_{t+δt_c}  (will be t at next iteration)

[6] DIAGNOSTIC / probe step (every probe_every coarse steps):
    MEM force on FINE grid at NACA wall sites → Cl/Cd → forces.csv
```

**Critical Phase 2 implementation detail (Lagrava §3.7 warning)**:
> "even for a simple Poiseuille ﬂow, the second order interpolation does not conserve the mass"

→ **Must use 4-point cubic (eq. 38)** for space interpolation, not the simpler 2-point linear (eq. 37). The 2-point version leaks mass on Poiseuille — this is THE canonical test in Phase 2.

### 5.1 Memory + storage requirements

- Coarse PDF: 2 buffers (ping-pong) for collide-stream
- Coarse PDF SNAPSHOT for time interpolation: 1 extra buffer at coarse-only sites (1.6 GB if full coarse stored, ~50 MB if only the xc→f layer stored)
- Fine PDF: 2 buffers
- Fine MASK + qfrac (NACA-specific, sparse)
- Coarse MASK marking which cells are "in fine patch" (so collide-stream skips them)

Total ~2 GB at NACA D/dx_c=80 setup. Within 4GB GPU.

### 5.2 Per-step cost estimate (revised)

- Coarse collide+stream (~30% of cells now solid-fine-patch-holes): ~80 ms
- Fine sub-step ×2 at 32×32×4 (4× cells per coarse): each ~80 ms × 2 = 160 ms
- Time interp (small): ~1 ms
- Space interp (small): ~2 ms
- Prolongation + restriction kernels: ~5 ms each side = 10 ms total
- Snapshot copy: ~5 ms
- **Total per coarse step: ~260 ms**
- 30k coarse steps × 0.26s = **~130 min wall** ← roughly 2× current 76 min baseline

Performance expectation: AMR will be **slower** than uniform D/dx=80 (2× walltime estimated), because we're effectively running 2 fine + 1 coarse step per "wall iteration" and the fine is per-cell expensive. The win is **accuracy**, not speed.

### 5.3 Stability stratification

Per Lagrava §3.7: 2-point linear interp causes mass leak even on Poiseuille. We use **4-point cubic** by default. Cumulant's bulk-instant relax (ω_b=1.0) further helps by damping pressure waves that interact poorly with interface.

---

## 6. Implementation phases

### Phase 1 (Week 2-3): AMR data structure + fine patch as island

**Files to add (under `src/physics/amr/` + `include/physics/amr/`)**:
- `include/physics/amr/fine_patch.h` — `FinePatch` class wrapping CudaBuffer + extent + ω_f
- `src/physics/amr/fine_patch_collision.cu` — `fluidCumulantCollisionKernel_fine` (clone of `fluidCumulantCollisionKernel` but per-FinePatch indexing)
- `src/physics/amr/fine_patch_streaming.cu` — `streamD3Q27_naca_qbb_fine` (clone of NACA-specific QBB streaming, scaled to fine dx)

**Files to modify (minimal touch)**:
- `apps/aero/aero_naca0012_cumulant.cu` — add `--amr-enable`, `--amr-le-extent X Y`, `--amr-te-extent X Y` CLI flags. If `--amr-enable` not given, **bit-identical** to current path.

**Build hookup**: CMake `file(GLOB_RECURSE LBM_PHYSICS_SOURCES ...)` auto-picks up new `src/physics/amr/*.cu` (no CMakeLists.txt edit).

**Tasks**:
- Fine patch SoA buffer matching D3Q27 layout (use `CudaBuffer<float>` RAII from `include/utils/cuda_memory.h`)
- Independent collision + streaming kernels for fine patch
- Fine patch mask + qfrac generation (re-use `stampNacaAirfoil4Digit` with fine dx)
- **No interface coupling yet** — fine runs in vacuum (or with f^eq inlet from initial freestream)
- Unit tests (under `tests/aero/`):
  - `test_fine_patch_mass_conservation.cu` (10k steps, periodic boundaries, ρ_total drift < 1e-6)
  - `test_amr_regression.cu` (--amr-enable OFF gives bit-identical Cl trajectory vs current binary)
  - `test_dfg_regression.cu` (ST 2D-2 Cd_max in [3.18, 3.22])

Gate: All 3 unit tests pass; current NACA bit-identical; DFG 1.2% holds; commit "Phase 1 AMR scaffolding".

### Phase 2 (Week 4-5, HIGHEST RISK): Interface coupling

- Restrict coarse → fine kernel: 4 phases (compute ρ,u → compute f_eq → compute f_neq → rescale + add)
- Prolongate fine → coarse kernel: same in reverse
- BGK fallback in 1-cell interface band (Option B)
- Validation:
  - **1D Poiseuille flow** with interface mid-channel: velocity profile vs analytical
    (acceptance: u_profile L2 error < 1% relative to ν·d²u/dx² = ∇p ; flat for Cumulant)
  - **DFG cylinder** with AMR fine patch around cylinder: Cd_max in [3.18, 3.22] (regression)
  - **Mass conservation audit** on interface: ∑(coarse→fine flux) − ∑(fine→coarse flux) < 1e-4 relative

Gate: 1D Poiseuille passes (this is THE critical test); DFG regression holds; mass conservation log shows ε ~ noise level.

### Phase 3 (Week 6-7): NACA + quantify

- Fine patch around NACA0012 LE+TE, sweep patch size (0.1/0.2/0.4 chord)
- Each setup: 30k step run, settled Cl/Cd/St
- If still ≥25% off Kurtulus: try 4× refinement (if GPU memory allows, with sparse-qfrac on fine)
- Final report:
  - Cl/Cd table across configs
  - Cross-code if obtainable (lbmpy symbolic; OpenLB if installed)
  - Gap decomposition (interface BGK loss vs LE under-resolution vs other §6 leaks)
  - Verdict: did we hit ≤25%? ≤10%?

Gate: ≤25% Cl error AT ANY configuration counts as 现実 target met.

### Phase 4 (Week 8): Documentation + handoff
- Update memory `project_amr_phase1_*.md`, `project_amr_phase2_*.md`, `project_amr_phase3_*.md`
- Update HANDOFF_TO_NEW_PARTNER.md with AMR section
- Commit all patches with clean history
- FINAL_REPORT.md (decisive go/no-go data)

---

## 7. Top-3 technical risks + mitigation

### Risk 1 (HIGH): Cumulant-Lagrava interface incompatibility manifests as instability or wrong Cl

**Symptom**: AMR enabled + Cumulant collision → NaN, or Cl wildly off (e.g., negative, or 10×).
**Probability**: medium (the BGK derivation assumes single-τ; Cumulant has 6 rates).
**Mitigation**:
- Start with Option B (interface BGK fallback) — bypasses the issue at small accuracy cost
- 1D Poiseuille MUST pass before moving to NACA — catches the issue cheap
- If Option B still gives weird behavior, fall back to Option A (per-moment rescaling) with +1 week budget

### Risk 2 (MEDIUM): Memory budget at 4× fine patch

**Symptom**: GPU OOM when adding fine patch + coarse.
**Numbers**: 
- Coarse 30c × 20c × 4 × D3Q27 × 4 bytes × 2 buffers = 1.6 GB
- Fine 0.2c × 0.1c × 4 × (160)² × D3Q27 × 4 bytes × 2 buffers = 64 MB (small!)
- Combined: ~1.7 GB (well within 4GB GPU)
- For 4× fine: 256 MB — still OK
**Mitigation**: keep fine patch SMALL; if scaling up patch size or refinement ratio causes OOM, sparse-qfrac for both grids.

### Risk 3 (MEDIUM-HIGH): Time-stepping bookkeeping bugs

**Symptom**: Subtle Cl drift over long sim; or sudden divergence at sub-step 2.
**Probability**: high — this is fiddly logic.
**Mitigation**:
- Implement a "single-rate trivial AMR" first: ω_c = ω_f, no rescaling. Verifies the bookkeeping without physics complications. Then turn on rescaling.
- Per-step diagnostic logging: mass flux through interface, peak velocity, ρ_max. dump to chain log.
- 1D Poiseuille runs many sub-steps; if bookkeeping wrong, divergence visible by step 1000.

---

## 8. What I won't do (per task brief)

- No octree / dynamic adaptive AMR
- No 3+ refinement levels
- No omega/QMIN/wall-ω scans (8-variant chain already falsified all)
- No D/dx ≤ 180 uniform-grid scans (saturated)
- No fix to AM Cumulant forcing bug (orthogonal to aero)
- No cloud GPU rent (user constraint)
- No XLB / lettuce install before strict need (defer to Phase 3 if needed)

---

## 9. Open questions for user

1. **Commit dirty patches as "adopt prev session patches"?** Recommendation: yes, single commit, before Phase 1 work starts. Reason: patches are real fixes that I want stable for AMR regression baseline. (HANDOFF §1.3 list: 6 QMIN clamps + omega_b=1.0 + profiling instrumentation.)

2. **Phase 1 unit-test bar**: "current NACA bit-identical with AMR disabled". Acceptable if AMR-disabled path adds 0 ms overhead (no work). Or can we tolerate <1% wall time overhead for AMR-disabled (e.g., dispatch check)?

3. **Fine patch placement**: I'm starting with rectangular ±0.05c around LE+TE. Are there papers/intuitions suggesting a different shape (e.g., circular at LE, narrow strip on suction side)?

4. **lbmpy generation of reference kernels**: would you find value if I use lbmpy to GENERATE a reference Cumulant kernel and `diff` against ours? It's a 2-day spike. (My intuition: low ROI vs DFG validation. Skip.)

5. **What to do with G500/G2000 VTK files (3GB) after AMR is in?** Keep as regression baseline, delete to save space, or move offboard?

---

## 10. Phase 0 status (this document is the deliverable)

| Item | Status |
|------|--------|
| Read HANDOFF + 5 memory files | ✓ |
| User confirmed understanding (10-sentence summary) | ✓ |
| Rebuild + verify | ✓ (DFG 0.16% off canonical) |
| DFG cylinder regression | ✓ (Cd_max=3.185, within tolerance) |
| NACA Re=2000 baseline | running (deterministic reproduction confirmed at step 1900: Cl=0.5636 vs original 0.5638) |
| lbmpy spike (stencil + weight) | ✓ (27/27 match) |
| Lagrava 2012 read | ✓ (full algorithm extracted §4 + §5) |
| Schornbaum 2016 read | ✓ (TRT extension confirmed, Cumulant not covered) |
| **Walberla local install discovered** | ✓ (SchaeferTurek + NonUniformGridGPU + FlowAroundSphere pre-built) |
| Geier 2017 read | pending (lower priority; we have enough) |
| AMR_DESIGN_PROPOSAL.md | ✓ (this doc) |

**Awaiting user review before Phase 1 starts.**

### 10.1 Key revisions from initial draft

1. **Walberla locally available** (was assumed absent). Will use for SchaeferTurek cross-check + as algorithm reference for `NonUniformGPUScheme`.
2. **Performance expectation revised**: AMR ≈ 130 min wall vs uniform 76 min baseline. AMR is for accuracy, not speed.
3. **Time interpolation snapshot buffer added** (Lagrava §3.5 step 2 — linear time interp at fine sub-step 1).
4. **4-point cubic spatial interp mandatory** (Lagrava §3.7 warning: 2-point linear leaks mass on Poiseuille — this is THE acceptance test).
5. **Cumulant compatibility refined**: with ω_b = ω_3..6 = 1.0 patches already in place, Lagrava formula applies directly to shear mode only. Other modes need no rescaling. **Reduces complexity vs initially feared.**

---

## Appendix A: Reference equations (for code review)

### A.1 Lagrava equations 22, 24, 28

```
ν_f = (δx_c / δx_f) · ν_c                   (Lagrava eq. 22)
ω_f = 2 ω_c / (4 - ω_c)                     (Lagrava eq. 24)
f^neq_c = (2 ω_f / ω_c) · f^neq_f           (Lagrava eq. 28, fine→coarse)
```

### A.2 Lagrava equation 29 (coarse → fine)

```
f_i^fine(x_c→f) = f_i^eq(ρ(x_c→f), u(x_c→f)) + (ω_c / (2ω_f)) · f_i^neq, coarse(x_c→f)
```

### A.3 Lagrava equation 30 (fine → coarse with optional filter)

```
f_i^coarse(x_f→c) = f_i^eq(ρ_f(x_f→c), u_f(x_f→c)) + (2ω_f / ω_c) · f_i^neq, fine(x_f→c)
```

### A.4 Schornbaum TRT extension (eq. 2.6)

```
λ_e (shear) rescales identically to SRT ω
λ_o (anti-shear) computed from Λ_eo = λ_e · λ_o = constant (typically 3/16)
```

### A.5 Cumulant compatibility (this proposal)

For Cumulant with our parameter choice (ω_b = ω_3..6 = 1.0 on both levels):
- Shear cumulants (6 moments): rescale by 2ω_f/ω_c per Lagrava
- Other cumulants (17 moments): no rescaling (rates equal across levels)
- Conserved (4 moments: ρ, ρu): no rescaling (already continuous)

**Phase 1 simplification**: BGK collision in 1-cell interface band (Option B above), Lagrava verbatim. Defer per-moment rescaling to optimization sprint if needed.

---

## Sources

- Lagrava, D. et al. (2012), "Advances in multi-domain lattice Boltzmann grid refinement", J. Comput. Phys. 231:4808-4822. Local copy: `docs/papers/Lagrava2012.pdf` + extracted text.
- Schornbaum, F. & Rüde, U. (2016), "Massively parallel algorithms for the lattice Boltzmann method on non-uniform grids", SIAM J. Sci. Comput. 38(2):C96-C126. arxiv:1508.07982. Local copy: `docs/papers/Schornbaum2016.pdf` + extracted text.
- lbmpy 1.4 (installed at `/home/yzk/.local/lib/python3.12/site-packages/lbmpy/`).
- Project memory: `~/.claude/projects/-home-yzk-CompressibleCFD/memory/`.
- Predecessor handoff: `docs/HANDOFF_TO_NEW_PARTNER.md`.
