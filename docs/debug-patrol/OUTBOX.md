# Patrol OUTBOX (patrol → main findings ready for review)

Items the patrol has fixed/found that main should consider for production.

## Performance fixes F-05 / F-07 / F-16 — 2026-04-27

Three cherry-pick-ready commits. Each is independent and safe to apply.

| Fix | Commit | Files changed | Speedup |
|-----|--------|---------------|---------|
| F-05 hoist vofStep buffers | cbfb0a2 | multiphysics_solver.h + .cu | ~2 malloc/free per step |
| F-07 GPU reductions in CFL diag | 7fbdfd7 | multiphysics_solver.cu | ~198 MB D→H per diag → ~12 bytes |
| F-16 pre-alloc reduceSum scratch | f09aa55 | ray_tracing_laser.h + .cu | ~2 malloc/free per laser step |

**Measured combined impact** (benchmark_keyhole_316L, 100 steps, full multiphysics, ray-tracing OFF):
- Before: 147.8 s → After: 97.7 s — **34% speedup**
- Note: F-16 not exercised (ray_tracing disabled in this config). F3-02 (prior commit 9a6061b) also contributed.

Tests passing on patrol branch: `test_vof_mass_correction_flux` 9/9, `test_thermal_lbm` 8/8.

Cherry-pick commands:
```bash
git cherry-pick cbfb0a2  # F-05
git cherry-pick 7fbdfd7  # F-07
git cherry-pick f09aa55  # F-16
```

## Audit Pass 3 — 2026-04-27

**37 findings** in `docs/debug-patrol/audit-pass3-findings.md`.

### Action required — CRITICAL

| ID | File | Issue |
|----|------|-------|
| F3-02 | src/physics/vof/vof_solver.cu:1690,1745 | `static float* d_block_max` survives VOFSolver lifetime — dangling GPU ptr on second construction |
| F3-13 | apps/sim_linescan_S3A1.cu + S3A3.cu | No T_solidus/T_liquidus override — silent regression trap if MaterialDatabase defaults change |
| F3-25 | src/config/simulation_config.cpp:357 | Preset dt=5e-10 is 460× too small; no justification comment |

### Action required — HIGH (top 3)

| ID | File | Issue |
|----|------|-------|
| F3-15 | apps/sim_linescan_phase3.cu:193 | kinematic_viscosity=0.0167 likely meant as LU value; constructor applies dt/dx² conversion a second time → τ≈∞ |
| F3-36 | src/physics/vof/vof_solver.cu:2200 | long long interface_count cast to int in kernel arg; wraps for future large domains |
| F3-10 | docs/sprint-history/overnight_audit/tests/test_pure_conduction.cu:87-92 | BC applied before collision, not after computeTemperature() — documented ESM ordering bug |

All 37 findings cataloged in `docs/debug-patrol/audit-pass3-findings.md`.

---

## 2026-04-27 main-session reviews of pass 3 findings

### F3-15 — FALSE ALARM (audit miss)

The `kinematic_viscosity = 0.0167f` in Phase-3 was NOT a unit mismatch.
Audit missed the `MultiphysicsSolver` layer's lattice→physical conversion:

```
config.kinematic_viscosity (LU)           [user sets in app]
    ↓ multiphysics_solver.cu:951-952  nu_phys = nu_LU × dx²/dt
nu_physical (m²/s)                        [passed to FluidLBM]
    ↓ fluid_lbm.cu:94             nu_lattice = nu_phys × dt/dx²
nu_lattice = original LU value            ✓ end-to-end correct
```

Phase-3 ran at correct τ=0.55 as intended. Its conclusion "low ν hurts
center Δh" is VALID. **Do NOT cherry-pick or act on F3-15.**

### F3-02 — FIXED in patrol

Commit `9a6061b` hoists the two `static float* d_block_max` from
`advectFillLevelPLIC`/`advectFillLevelTVD` into `VOFSolver` class members
with destructor cleanup. Cherry-pick to `benchmark/conduction-316L`:
```bash
git cherry-pick 9a6061b
```

### F3-13, F3-25 — confirmed real but LOW priority

- F3-13: Sprint-3 apps S3A1/S3A3 don't override T_solidus/T_liquidus.
  Currently latent (defaults match Sprint-1). Document as known limitation.
- F3-25: ti6al4v_melting preset dt=5e-10 too small. Fix in `simulation_config.cpp`
  but path is `@deprecated` per its own header — cleanup only, not blocker.

---

## Numerical-accuracy audit on collision kernels — 2026-04-27

**Doc**: `docs/debug-patrol/numerical-audit-collision-kernels.md`
**Total**: 2 confirmed bugs, 3 potential, 5 style (across 7 audit areas)
**State**: investigation only, NO code changes — fixes need design call by main.

### 2 confirmed bugs ready for fix

| ID | File | Issue |
|---|---|---|
| **N-1.2** | `src/physics/fluid/fluid_lbm.cu:2060-2062` vs `:1455` (and parallel sites in TRT-EDM, Reg-EDM, Reg-Guo kernels) | EDM collision and post-stream macroscopic kernels disagree on `u_phys` formula in mushy zone. Collision: `(m+0.5F)/(ρ+0.5K)`. Macro: `m/(ρ+0.5K) + 0.5F/ρ`. Algebraic difference `0.5·F·0.5K/(ρ(ρ+0.5K))` ≈ ~40% relative error at K_LU=10 with F_LU=1e-3. Each step the macro overwrites with the inconsistent value, feeding next-step force build. Likely contributor to Sprint-1 raised-track sign-flip (groove vs ridge). |
| **N-7.1** | `src/physics/force_accumulator.cu:701-761` (gradual) and `:766-860` (adaptive) | CFL limiter is discontinuous at `v_new = v_target` boundary when `v_current > v_ramp` (~18% scale jump in test case `v_target=1, ramp=0.8, v_current=0.9, f_mag=0.1`). Contradicts comment claim of "smooth exponential damping ... avoids discontinuous force jump". Causes spatial force banding in v ~ v_target regime — concern for recoil hot spots and melt boil cells. |

### Suggested patches (sketches in audit doc, NOT applied)

**N-1.2 (option a — match collision)**:
```c
// fluid_lbm.cu:2060-2062, in computeMacroscopicSemiImplicitDarcyEDMKernel
u_x = u_bare_x + 0.5f * force_x[id] * inv_denom;  // was: * inv_rho
u_y = u_bare_y + 0.5f * force_y[id] * inv_denom;
u_z = u_bare_z + 0.5f * force_z[id] * inv_denom;
```
**Verification test** to add: one-step mushy run (K_LU=1, F_LU=1e-3),
assert `|u_collision - u_macro| < 1e-7` per component.

**N-7.1**: switch regime decision to `v_current` instead of `v_new`.
See §7 "Suggested fix" in the audit doc for sketch.

### Potential bugs (P1) — main needs design call

* **N-1.1**: Sprint-1 hybrid `Δu = F/(ρ+0.5K)` is not standard Kupershtokh
  EDM. Internally consistent, limits OK, but in `K → ∞` overcompensates by
  factor `ρ/(ρ+0.5K)` vs clean Crank-Nicolson Darcy. Decide whether to
  keep + document or revise.
* **N-2.2**: `m/(ρ+0.5K)` form ≠ clean CN Darcy; matches an ω-modulated
  decay rate. Subtle calibration-coupled, not a bug.
* **N-3.2**: TRT Λ-preserving LES drives ω⁻ → 1.85 with Λ=3/16 under
  heavy turbulence (τ_eff=5). No instability but anti-symmetric mode
  becomes ~over-relaxed; watch for boundary errors.

### Verified-clean (NO bugs despite suspicion)

* Equilibrium f_eq formula (constants 1, 3, 4.5, -1.5 all exact in FP32)
* D3Q19 `opposite[]` self-inverse; weights sum to 1; 2nd-moment isotropy
  `Σ w c c = cs² I` to FP32 precision
* TRT neq-split formulation matches `f^+/f^-` decomposition exactly
* Regularized 2nd-order Hermite projection round-trips Π exactly
  (verified by python)
* Hou 1996 algebraic Smagorinsky derivation matches code exactly
* Guo 2002 source term S_q expansion in Reg-Guo kernel correct

The two confirmed bugs are *coupling* errors — each individual kernel is
correct in isolation, but the handoff between them is inconsistent. This
pattern slips through unit tests because each kernel passes its own.

---

## 🚨 N-1.2 PRODUCTION-IMPACT BUG FIXED

Commit `a210acf` on patrol, cherry-picked as `9195c60` on
`benchmark/conduction-316L`. Pushed to origin.

### Why this matters NOW for 5060 work

The macro kernel's u_phys mismatch corrupts Marangoni/recoil/Darcy force
builds in the **mushy zone** every step. Track-C iter-4 result on Phase-2
(centerline -6μm) was achieved DESPITE this bug. Phase-4 result
(centerline -4μm at t=2ms) was DESPITE this bug.

After fix, the trailing-zone mass balance gets correct velocity input.
Predict: another 1-3 μm centerline improvement and possibly clean
ridge/groove sign (the bug systematically biased trailing flow).

### Decision for 5060 currently running Phase-4-corrected

| 5060 status | Recommendation |
|---|---|
| Already mid-run | Let finish; the result is the "buggy + ny=110" baseline; rerun on fixed version for comparison |
| Not yet started | `git pull && rebuild`; tonight's runs use the fixed kernel |
| Already finished | Compare output to fixed-rerun |

The fix is one-line in `src/physics/fluid/fluid_lbm.cu` line 2060
(inv_rho → inv_denom). Build is identical structure; full rebuild needed
because of inline functions but only ~1 file dirty.

