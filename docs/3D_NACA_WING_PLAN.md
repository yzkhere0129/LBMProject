# 3D NACA Wing Demo — Design + Feasibility Analysis

**Date**: 2026-05-21
**Goal**: Move beyond quasi-2D (nz=4) to a true 3D finite-span wing to capture wingtip vortex and span-wise pressure variation.

This is the natural T2.1 progression after STL D1-D4. We do the math BEFORE writing any code to see what's achievable on the 4 GB GPU.

---

## 1. Reference setup (target)

| | |
|---|---|
| Geometry | NACA0012, chord c=1 m, span s=4c (aspect ratio AR=4) |
| Flow | Re=2000, α=8°, U∞=1 m/s |
| Domain (chord units) | x=30c, y=20c, **z=8c** (4c span + 2c margin each side) |
| Reference (NACA in current code) | quasi-2D: nz=4, no span-wise variation |

For comparison, real F-18 has wing AR ~3.5 (variable-sweep moot in clean config). So AR=4 is realistic.

---

## 2. Memory budget at various resolutions

Memory per cell ≈ **27 PDFs × 4 B × 2 (src+dst) = 216 B/cell** (D3Q27 with double-buffering).
Plus solid mask, rho/u temp fields, sparse qfrac — call it +20% for total.

| D/dx | nx (=30·D) | ny (=20·D) | nz (=8·D) | n_cells | mem (PDF×2 + 20%) |
|---|---|---|---|---|---|
| 20 | 600 | 400 | 160 | 38.4 M | **9.96 GB** ❌ |
| 30 | 900 | 600 | 240 | 130 M | 33.6 GB ❌ |
| 40 | 1200 | 800 | 320 | 307 M | 79.7 GB ❌ |

**Verdict**: Even at the coarsest D/dx=20 (LE radius = 0.25 cells — way under-resolved), a uniform 3D NACA wing won't fit. **Single-GPU 4GB cannot do this.**

---

## 3. Where the budget allows simulation

Reducing **span**:

| span | nz | nz·D/dx=20 | n_cells (D/dx=20, x=30c) | mem |
|---|---|---|---|---|
| 4c (target) | 160 | | 38.4 M | 9.96 GB ❌ |
| 2c | 80 | | 19.2 M | 4.98 GB ❌ |
| 1c (half) | 40 | | 9.6 M | **2.49 GB** ✓ |
| 0.5c | 20 | | 4.8 M | **1.24 GB** ✓ |

So at most a **1c-span wing** at D/dx=20 fits in 4 GB. LE radius 0.25 cells — terrible, but the BL on the wing surface would still get **20 cells across chord** which is acceptable for transition-flow Cd/Cl.

Reducing **domain length/width**:

If we reduce x=15c (instead of 30c) and y=10c (instead of 20c), we save 4×. Then:

| D/dx | nz·c=2c span | n_cells (15c×10c×2c) | mem |
|---|---|---|---|
| 40 | 80 | 38.4 M | 9.96 GB ❌ |
| **30** | **60** | **16.2 M** | **4.20 GB** marginal ❌ |
| 20 | 40 | 4.8 M | **1.24 GB** ✓ |

Tight domain at D/dx=20 — useful for proof-of-concept.

---

## 4. Recommended demo configuration

**Phase 3.1 (this code)**: half-span 0.5c wing at D/dx=30, with span-wise periodic BC (so geometry is "infinite span" but with full 3D collision operator). Tests whether 3D code path works without trying to capture wingtip.

| | |
|---|---|
| Domain | 15c × 10c × **0.5c** |
| D/dx | 30 |
| Resolution per chord | 30 cells across c (LE radius ~0.4 cells, still rough) |
| z BC | periodic |
| Geometry | NACA0012 stamped at every z slab (effectively quasi-2D but 3D solver runs) |
| n_cells | ~6.8 M |
| Memory | ~1.5 GB |
| Wall time (extrap from D/dx=80 quasi-2D 65 min, ×8 cells, ÷2 in xy = 8/2/4=1× → ~60-80 min) | OK |

**Goal**: validate that running with nz > 4 doesn't break anything. Cd/Cl should match the existing quasi-2D NACA result within FP32 noise (since geometry is the same span-wise).

**Phase 3.2 (next)**: finite-span 1c wing at D/dx=20, no-slip wing tip BC.

| | |
|---|---|
| Domain | 15c × 10c × **2c** (1c wing + 0.5c above/below) |
| D/dx | 20 |
| n_cells | ~4.8 M |
| Memory | ~1.25 GB |
| Geometry | NACA0012 stamped only z ∈ [0.5c, 1.5c] (the wing); rest fluid |
| z BC | wall or periodic-with-margin |

**Goal**: see wingtip vortex roll-up; expect spanwise lift distribution (elliptic-ish).

---

## 5. What needs to change in the code

| Module | Change | Effort |
|---|---|---|
| `apps/aero/aero_naca0012_cumulant.cu` | Allow nz > 4. Currently nz is hardcoded(?) or set via CLI. Add `--lz-over-c Z` flag | 1 hour |
| `obstacle_geometry.h::stampNacaAirfoil4Digit` | Span-limit option: only stamp z ∈ [z_lo, z_hi] (for finite wing). Currently stamps all z. | 1 hour |
| Force probe | Already 3D (`memForceNaca_QBB_sparse` sums over all z) — no change. Just need to interpret per-span force if we want sectional Cl(z). | 1 day for spanwise breakdown |
| Z boundary | Currently periodic via `applyZPeriodicBC` (or whatever) — check. Add wall-BC option for non-periodic. | 2 hours |
| Stretched grid? | NO — uniform Cartesian only. Future work. | — |

Total: **~1-2 days of code** to be able to run 3D NACA wing on existing infrastructure.

---

## 6. Critical limitations going in

1. **LE resolution will be terrible**. D/dx=20 → LE radius (0.0158c) = 0.32 cells. Even worse than current D/dx=80 quasi-2D (which is at 1.27 cells). Expect Cl deficit > 50% (worse than current 18.5%).
2. **No AMR for 3D** — single-level patch AMR would still help LE, but the existing AMR code is set up for the LE-only patch and we have no spanwise refinement strategy.
3. **Domain truncation**: x=15c, y=10c is tight. Far-field BC errors may inflate drag estimate.
4. **Time-to-settled**: similar to 2D (~65 min for 30k steps at quasi-2D D/dx=80). With ~5 M cells at D/dx=20 we'd be roughly the same wall-time.

---

## 7. ROI assessment

- **Pro**: Truly 3D simulation, captures wingtip vortex, real aero physics (instead of 2D approximation).
- **Pro**: Establishes that the code handles nz > 4 cleanly — needed for any future user STL with 3D body.
- **Con**: LE resolution drops to 0.3-0.4 cells at D/dx=20-30 → quantitative Cd/Cl will be even worse than current 18.5% gap.
- **Con**: Likely will NOT validate well vs literature — but visual flow patterns (vortex roll-up) can still be impressive.

**Recommendation**: **Do Phase 3.1 (periodic-span) as a proof-of-concept**, ~1 day code + 1 hour run. Then decide whether Phase 3.2 (finite-span) is worth the LE-resolution penalty.

---

## 8. Path beyond 4 GB

For meaningful 3D NACA wing aero (target Cl gap < 25%), the realistic path is:

1. **Multi-GPU MPI** — splits domain across multiple GPUs. 4× A100 80 GB ≈ 320 GB, enables D/dx=80 3D wing.
2. **Cell-stretched grid (non-uniform Δx)** — requires fundamentally rewriting LBM streaming. Out of scope.
3. **Octree AMR (multi-level)** — extends our single-level Phase 2 to nested levels. ~3 weeks new code.

None of these are in scope for the 4 GB single-GPU constraint we're working under.

---

## 9. Next step: code change list

If you want to proceed with Phase 3.1:

- [ ] Add `--lz-over-c Z` CLI flag (currently hardcoded inside driver?)
- [ ] Verify `stampNacaAirfoil4Digit` handles arbitrary nz correctly (suspicion: yes, it's a 3D loop)
- [ ] Add `--bc-z periodic|wall` selector for z faces
- [ ] Run Phase 3.1 (periodic z, all z slabs have airfoil): expect Cd/Cl == quasi-2D within FP32 noise
- [ ] If passes, Phase 3.2 (span-limited stamp)
- [ ] Output: spanwise Cl(z) distribution plot

— 2026-05-21 Claude / 3D NACA feasibility analysis
