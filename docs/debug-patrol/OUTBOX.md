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
