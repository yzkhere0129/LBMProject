# D3Q27 Single-Node QBB Bouzidi BC for NACA — Paper Design

**Goal:** Eliminate stair-step discretization artifact that flips Cl sign for rotated NACA airfoil. Reuse the proven D3Q19 single-node QBB approach (Phase 1c, lbmpy formula).

**No new physics** — same algorithm, larger lattice (Q=19 → Q=27).

## Reference: existing D3Q19 implementation

`src/physics/fluid/fluid_lbm.cu:2655` — `fluidStreamingKernelWithSingleNodeQBB`. ~80 lines. Algorithm:

For each fluid cell:
1. Compute local moments (ρ, u_x, u_y, u_z) from the 19 local PDFs
2. For each q direction: stream OR apply QBB
   - If dst is fluid: standard push, `f_dst[dst, q] = f_in`
   - If dst is solid: lbmpy single-node QBB formula:
     ```
     t1 = (f_in - f_out) + (f_in + f_out − ω · feq_sym) / (1 − ω)
     t2 = qf · (f_in + f_out) / (1 + qf)
     result = ((1 − qf) / (1 + qf)) · 0.5 · t1 + t2
     f_dst[id, opp_q] = result
     ```
   - `feq_sym = f_eq(q, ρ, u) + f_eq(opp_q, ρ, u)`
   - `qf` = per-link q-fraction in (0, 1], clamped to [0.05, 0.95]

**Key observation:** the formula is INDEPENDENT of the lattice — only depends on (f_in, f_out, ω, qf, feq_sym). So extending Q=19 → Q=27 requires:
- Loop over Q=27 instead of Q=19
- Use ex27/ey27/ez27 instead of ex/ey/ez
- Use D3Q27::computeEquilibrium (already implemented in `core/lattice_d3q27.h`)
- Allocate qfrac with 27 entries per cell

## Files to add / modify

### NEW FILES

**`src/physics/cumulant/streaming_d3q27_qbb.cu`** (~120 lines)
Contains:
- `__global__ void streamD3Q27_naca_qbb(...)` — D3Q27 version of the streaming kernel with single-node QBB
- Same signature pattern as `streamD3Q27_naca` plus `const float* qfrac` and `float omega`

### MODIFIED FILES

**`include/physics/aero/obstacle_geometry.h`** (~80 lines added)
- `inline std::vector<float> makeNacaQFraction(nx, ny, nz, dx, mask, LE, chord, thick%, alpha_rad)`:
  - Loop over fluid cells
  - For each q ∈ {1..26}, if dst is solid, compute q-fraction via bisection / sub-sampling
  - Return host vector, sized n_cells × 27

  Implementation skeleton:
  ```cpp
  for each fluid cell (i, j, k):
    for each q ∈ {1..26}:
      compute dst = (i+ex27[q], j+ey27[q], k+ez27[q])
      if dst out-of-grid OR mask[dst] != SOLID: qfrac[id + q*n_cells] = 1.0; continue
      // Find smallest t ∈ (0, 1] where the segment from cell-center to dst-center
      // first enters the airfoil
      P0 = cell center (world coords)
      D  = link direction × dx (world)
      // Transform to airfoil-frame
      For t in 64 sub-samples between (0, 1]:
        P(t) = P0 + t*D
        (xr, yr) = R(-α)(P(t) − LE)
        if 0 ≤ xr/chord ≤ 1 AND |yr| ≤ y_t(xr/chord)*chord:
          qfrac = t  (could refine via bisection between t-1/N and t)
          break
  ```

**`apps/aero/aero_naca0012_cumulant.cu`** (~30 lines changed)
- Add `--bc stair|qbb-snode` flag (defaults to `stair` for backward-compat)
- If `--bc qbb-snode`:
  - After mask is built, call `makeNacaQFraction` to build qfrac
  - Allocate `d_qfrac` on device, copy qfrac
  - In the time loop, call `streamD3Q27_naca_qbb` (instead of `streamD3Q27_naca`)
- Same MEM kernel works (it just reads f, no qfrac dependency for halfway BB MEM)
- Optionally add a separate `memForceNaca_QBB` that uses qfrac for the BFL force formula. PHASE 2.

### NO MODIFICATION needed

- D3Q27 lattice tables — already correct
- Cumulant collision — orthogonal to BC choice
- Inlet/outlet/Y-walls — orthogonal

## Estimated effort

- streaming kernel D3Q27 QBB:           1–2 hours (mechanical port from D3Q19)
- makeNacaQFraction:                    2–3 hours (sub-sampling + transcendental find)
- Driver wiring + smoke compile:        1 hour
- **Total paper-side work**:            **4–6 hours of code, no sim runs**

## Correctness sanity checks (paper-side, before any sim)

1. **Compile clean** — no missing symbols, no D3Q19 leakage into D3Q27 code
2. **qfrac values for α=0** — should be {0.5, dx-dependent, etc.} with vertical mirror symmetry across the (untilted) chord. Print a few and inspect by hand.
3. **qfrac values for α=+8°** — should NOT have the cancellation property. Mirror-pair cells should give qfrac that match the curved-surface offset, not the lattice-cell offset.
4. **At α=0, with QBB**: simulation should give Cl=0 (same as stair-step at α=0)
5. **At α=+8°, with QBB**: hopefully Cl > 0 with sign matching Kurtulus. CRITICAL ACCEPTANCE TEST.

## Ask-user gates

These are simulation runs that consume GPU. **Do not run without user approval:**
- Smoke run α=0 with QBB (1000 steps, ~1 min)
- Smoke run α=+8° with QBB (16000 steps, ~5 min)
- Production run for Kurtulus comparison (if smoke succeeds)

## Hypothesis confirmation criteria

If QBB gives:
- α=0: Cl ≈ 0 ✓
- α=+8°: Cl > 0 (positive, magnitude in [0.3, 0.6])
- α=-8°: Cl < 0 (anti-symmetric)

Then the stair-step hypothesis is CONFIRMED. Bug = pure discretization artifact, not a code error. Phase D unblocked.

If QBB still gives wrong sign:
- Bug is deeper than stair-step — possibly in cumulant or streaming for a more subtle reason
- Need to recurse: try walberla / OpenLB control on same mesh
