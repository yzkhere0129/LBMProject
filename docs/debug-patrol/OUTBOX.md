# Patrol OUTBOX (patrol → main findings ready for review)

Items the patrol has fixed/found that main should consider for production.

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
