# Debug Patrol Communication Protocol

**Worktree path**: `/home/yzk/LBMProject_debug_patrol`
**Branch**: `debug/patrol-2026-04-27`
**Started**: 2026-04-27

## Three-way comm

```
                         origin (GitHub)
                              │
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
      benchmark/         debug/patrol     feature/vof-mass-
      conduction-316L    -2026-04-27      correction-destination
      (production HEAD)  (debug + audit)  (worktree A, frozen)
              │               │
              ▼               ▼
       Main session       Patrol session
       (3050)             (this worktree)
       Phase-4/5 runs     Audit fixes,
       Track-C iter-4     bug hunts,
       physics            test repairs
```

## File-system contracts

In this worktree (`/home/yzk/LBMProject_debug_patrol`):

| File | Owner | Purpose |
|---|---|---|
| `docs/debug-patrol/LIVE_LOG.md` | patrol | running log of issues found, fixes attempted, status |
| `docs/debug-patrol/INBOX.md` | main → patrol | items requested for investigation |
| `docs/debug-patrol/OUTBOX.md` | patrol → main | findings ready to merge to main branch |
| `docs/debug-patrol/COMM_PROTOCOL.md` | shared | this file |

## Workflow

### Patrol (this worktree) → Main (benchmark/conduction-316L)

1. Patrol finds + fixes bug
2. `git commit` in patrol worktree
3. `git push origin debug/patrol-2026-04-27`
4. Append summary to `OUTBOX.md`
5. Main can `git fetch && git log debug/patrol-2026-04-27` to inspect
6. Main cherry-picks or merges as appropriate

### Main → Patrol (request)

1. Main appends to `INBOX.md` with specific question/area
2. Commits + pushes to `benchmark/conduction-316L`
3. Patrol does `git fetch && git rebase origin/benchmark/conduction-316L` periodically
4. Picks items off INBOX

### Conflict avoidance

Patrol worktree NEVER:
- Modifies `apps/sim_linescan_phase{1..5}*.cu` (those are main's production sims)
- Modifies `tests/validation/test_low_tau_collision_sweep.cu` or
  `test_trt_omega_minus_arithmetic.cu` (recently landed, stable)
- Touches `output_*` directories

Patrol worktree CAN:
- Audit + fix any `src/` or `include/` file
- Replace stub tests (`tests/integration/multiphysics/test_*.cu` with `EXPECT_TRUE(true)`)
- Add new tests in `tests/regression/` or `tests/unit/`
- Add diag tools in `scripts/diagnostics/`
- Document findings in `docs/`

## Status checking

From main session (`/home/yzk/LBMProject`):
```bash
# See what patrol has done
cd /home/yzk/LBMProject  # or any worktree path
git fetch origin
git log origin/debug/patrol-2026-04-27 --not benchmark/conduction-316L --oneline

# See live findings
cat /home/yzk/LBMProject_debug_patrol/docs/debug-patrol/LIVE_LOG.md
cat /home/yzk/LBMProject_debug_patrol/docs/debug-patrol/OUTBOX.md
```
