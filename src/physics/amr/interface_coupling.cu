/**
 * @file interface_coupling.cu
 * @brief Coarse↔fine PDF coupling kernels per Lagrava 2012.
 *
 * Phase 2 minimum-viable approach (this file, Phase 2.2/2.3/2.4):
 *   - prolongation (coarse → fine boundary cells): nearest-coarse-cell
 *     value, no spatial interp. Simple, dissipative.
 *   - restriction (fine interior → coarse cells in patch): unweighted
 *     average over the 8 fine cells covering each coarse cell.
 *
 * Algorithm follows Lagrava 2012 eq. 29/30/36 with the simplification
 * "no spatial interp" (i.e., the surrounding coarse cells are assumed
 * to vary smoothly over one coarse cell width — the 0th-order approx).
 *
 * Upgrade path (later Phase 2 commits):
 *   - bilinear spatial interp (9/3/3/1 weights, cell-center 2× convention)
 *   - box-filter restriction over q lattice dirs (Lagrava eq. 33)
 *   - linear time interp at fine sub-step 1 (requires coarse snapshot)
 *
 * Sign of the rescaling factors:
 *   f_neq^fine = (ω_c / (2·ω_f)) · f_neq^coarse   (prolongation)
 *   f_neq^coarse = (2·ω_f / ω_c) · f_neq^fine     (restriction)
 *
 * Reference: docs/papers/Lagrava2012.txt + AMR_DESIGN_PROPOSAL.md §4.
 */

#include "core/lattice_d3q27.h"

namespace lbm {
namespace physics {
namespace amr {

using lbm::core::ex27;
using lbm::core::ey27;
using lbm::core::ez27;

// ============================================================================
// Lagrava 2012 §3.6 cubic (Catmull-Rom) interpolation along refinement interface
// ----------------------------------------------------------------------------
// At t=0.5 reproduces Lagrava eq. shown in §3.6: weights 9/16, -1/16.
// For cell-center 2× refinement, fine boundary cells sit at offset ±0.25 from
// the containing coarse cell center, so we evaluate the cubic at t=0.25 or t=0.75.
//
// Convention: t ∈ [0, 1] is the position of the interpolated point between
// stencil indices 0 and 1 of a 4-cell stencil (-1, 0, 1, 2).
//
// Catmull-Rom weights:
//   w(-1) = -0.5 t (1-t)(2-t)
//   w(0)  =  (1+t)(1-t)(2-t) / 2
//   w(1)  =  (1+t) t (2-t) / 2
//   w(2)  = -0.5 (1+t) t (1-t)
// Sum = 1 for all t ∈ [0,1].
__device__ inline void catmull_rom_weights_t(float t, float w[4]) {
    const float one_p_t = 1.0f + t;
    const float one_m_t = 1.0f - t;
    const float two_m_t = 2.0f - t;
    w[0] = -0.5f * t * one_m_t * two_m_t;       // x_{-1}
    w[1] =  0.5f * one_p_t * one_m_t * two_m_t; // x_0
    w[2] =  0.5f * one_p_t * t * two_m_t;       // x_1
    w[3] = -0.5f * one_p_t * t * one_m_t;       // x_2
}

// Build 1D cubic stencil offsets (di) and weights (w) for a fine boundary
// cell in quadrant q ∈ {0,1} of its containing coarse cell.
//   q=0 → fine at coarse offset -0.25  → t=0.75, stencil = {-2,-1, 0,+1}
//   q=1 → fine at coarse offset +0.25  → t=0.25, stencil = {-1, 0,+1,+2}
__device__ inline void cubic_1d_q(int q, int di_out[4], float w_out[4]) {
    if (q == 0) {
        di_out[0] = -2; di_out[1] = -1; di_out[2] = 0; di_out[3] = +1;
        catmull_rom_weights_t(0.75f, w_out);
    } else {
        di_out[0] = -1; di_out[1] =  0; di_out[2] = +1; di_out[3] = +2;
        catmull_rom_weights_t(0.25f, w_out);
    }
}

/**
 * @brief Prolongation kernel: 2D bicubic Catmull-Rom over 16 coarse cells.
 *
 * Replaces the 4-cell bilinear (9/3/3/1) of earlier Phase 2-B. Per Lagrava
 * 2012 §3.6, bilinear interpolation is locally 2nd-order which gives a
 * globally O(1) pressure jump at the coarse-fine interface — incompatible
 * with LBM's 2nd-order global accuracy. Cubic (Catmull-Rom) at t=0.25 or
 * t=0.75 brings the interpolation error in line with LBM.
 *
 * Stencil (per quadrant q={0,1} in each direction):
 *   q=0 (fine at coarse offset -0.25): cells {-2, -1, 0, +1}, t=0.75
 *   q=1 (fine at coarse offset +0.25): cells {-1, 0, +1, +2}, t=0.25
 * 2D weights = outer product of 1D x-weights and 1D y-weights (16 total).
 *
 * Solid-aware: any of the 16 cells that's solid → weight zeroed,
 * remaining weights renormalized so Σw=1 (preserves mass).
 * If all 16 solid → skip cell (return).
 *
 * Rescaling (Lagrava 2012 eq. 29): same as bilinear,
 *   f_fine = f_eq(ρ_avg, u_avg) + (ω_c / 2ω_f) · f_neq_avg
 *
 * @param f_coarse   Coarse PDF (SoA).
 * @param f_fine     Fine PDF (SoA, written at boundary cells only).
 * @param solid_c    Coarse solid mask (for solid-aware weight zeroing).
 */
__global__ void prolongateBoundaryFineFromCoarse(
    const float* __restrict__ f_coarse,
    float*       __restrict__ f_fine,
    const unsigned char* __restrict__ solid_c,
    int i_lo, int j_lo, int k_lo,
    int refine,
    int nx_c, int ny_c, int nz_c,
    int nx_f, int ny_f, int nz_f,
    float omega_c, float omega_f)
{
    using lbm::core::D3Q27;

    const int i_f = blockIdx.x * blockDim.x + threadIdx.x;
    const int j_f = blockIdx.y * blockDim.y + threadIdx.y;
    const int k_f = blockIdx.z * blockDim.z + threadIdx.z;
    if (i_f >= nx_f || j_f >= ny_f || k_f >= nz_f) return;

    const bool on_x_edge = (i_f == 0 || i_f == nx_f - 1);
    const bool on_y_edge = (j_f == 0 || j_f == ny_f - 1);
    if (!on_x_edge && !on_y_edge) return;

    const int i_c = i_lo + i_f / refine;
    const int j_c = j_lo + j_f / refine;
    const int k_c = k_lo + k_f / refine;

    if (i_c < 0 || i_c >= nx_c || j_c < 0 || j_c >= ny_c ||
        k_c < 0 || k_c >= nz_c) return;

    // Quadrant within containing coarse cell.
    const int qx = i_f % refine;
    const int qy = j_f % refine;

    // Build 4×4 bicubic stencil offsets + weights (outer product of 1D cubics).
    int  dix[4], djy[4];
    float wx[4], wy[4];
    cubic_1d_q(qx, dix, wx);
    cubic_1d_q(qy, djy, wy);

    auto clamp_idx = [](int v, int v_max) -> int {
        return (v < 0) ? 0 : ((v >= v_max) ? (v_max - 1) : v);
    };

    const int n_cells_c = nx_c * ny_c * nz_c;
    const int n_cells_f = nx_f * ny_f * nz_f;

    auto coarse_id = [&](int ic, int jc, int kc) -> int {
        return ic + jc * nx_c + kc * nx_c * ny_c;
    };

    // 16 stencil cells with clamped indices + outer-product weights.
    int   ids[16];
    float ws[16];
    #pragma unroll
    for (int j_s = 0; j_s < 4; ++j_s) {
        #pragma unroll
        for (int i_s = 0; i_s < 4; ++i_s) {
            const int ic = clamp_idx(i_c + dix[i_s], nx_c);
            const int jc = clamp_idx(j_c + djy[j_s], ny_c);
            const int c = i_s + 4 * j_s;
            ids[c] = coarse_id(ic, jc, k_c);
            ws[c]  = wx[i_s] * wy[j_s];
        }
    }

    // Solid-aware: zero weights on solid cells, renormalize.
    // Note: bicubic weights include NEGATIVE values (Catmull-Rom). Zeroing
    // a negative weight changes Σ|w| but the sign-preserving renormalization
    // wsum keeps the partition-of-unity property (Σ wsum_renorm = 1).
    if (solid_c) {
        float wsum = 0.0f;
        #pragma unroll
        for (int c = 0; c < 16; ++c) {
            if (solid_c[ids[c]] != 0) ws[c] = 0.0f;
            wsum += ws[c];
        }
        // wsum can be near zero if mostly-symmetric solid pattern → fallback.
        if (!isfinite(wsum) || fabsf(wsum) < 1e-3f) return;
        const float inv_wsum = 1.0f / wsum;
        #pragma unroll
        for (int c = 0; c < 16; ++c) ws[c] *= inv_wsum;
    }

    // Per-cell CE decomp (same structure as bilinear, but 16-cell loop).
    // Per-cell decomp avoids the nonlinear f_eq coupling that NaN'd the
    // earlier "bilinear sum of f, decomp with f_eq(sum)" approach.
    float rho_avg = 0.0f, ux_avg = 0.0f, uy_avg = 0.0f, uz_avg = 0.0f;
    float f_neq_avg[27] = {0.0f};

    #pragma unroll 4
    for (int c = 0; c < 16; ++c) {
        if (ws[c] == 0.0f) continue;
        float f_loc[27];
        float r = 0.0f, mx = 0.0f, my = 0.0f, mz = 0.0f;
        #pragma unroll
        for (int q = 0; q < 27; ++q) {
            const float v = f_coarse[ids[c] + q * n_cells_c];
            f_loc[q] = v;
            r += v;
            mx += ex27[q] * v;
            my += ey27[q] * v;
            mz += ez27[q] * v;
        }
        const float r_safe = fmaxf(r, 1e-6f);
        const float u_x = mx / r_safe;
        const float u_y = my / r_safe;
        const float u_z = mz / r_safe;
        rho_avg += ws[c] * r;
        ux_avg  += ws[c] * u_x;
        uy_avg  += ws[c] * u_y;
        uz_avg  += ws[c] * u_z;
        #pragma unroll
        for (int q = 0; q < 27; ++q) {
            const float feq_q = D3Q27::computeEquilibrium(q, r, u_x, u_y, u_z);
            f_neq_avg[q] += ws[c] * (f_loc[q] - feq_q);
        }
    }

    // Safety: if interpolated ρ is unphysical (Catmull-Rom's negative weights
    // can amplify shocks), fall back to the containing cell.
    if (!isfinite(rho_avg) || rho_avg < 0.5f) {
        const int id_c = coarse_id(i_c, j_c, k_c);
        rho_avg = 0.0f;
        ux_avg = uy_avg = uz_avg = 0.0f;
        for (int q = 0; q < 27; ++q) f_neq_avg[q] = 0.0f;
        float mx = 0.0f, my = 0.0f, mz = 0.0f;
        float f_loc[27];
        #pragma unroll
        for (int q = 0; q < 27; ++q) {
            const float v = f_coarse[id_c + q * n_cells_c];
            f_loc[q] = v;
            rho_avg += v;
            mx += ex27[q] * v;
            my += ey27[q] * v;
            mz += ez27[q] * v;
        }
        const float r_safe = fmaxf(rho_avg, 1e-6f);
        ux_avg = mx / r_safe; uy_avg = my / r_safe; uz_avg = mz / r_safe;
        #pragma unroll
        for (int q = 0; q < 27; ++q) {
            const float feq_q = D3Q27::computeEquilibrium(q, rho_avg, ux_avg, uy_avg, uz_avg);
            f_neq_avg[q] = f_loc[q] - feq_q;
        }
    }

    // f_fine = f_eq(ρ_avg, u_avg) + (ω_c / 2ω_f) · f_neq_avg
    const float scale = omega_c / (2.0f * omega_f);
    const int id_f = i_f + j_f * nx_f + k_f * nx_f * ny_f;
    #pragma unroll
    for (int q = 0; q < 27; ++q) {
        const float feq_q = D3Q27::computeEquilibrium(q, rho_avg, ux_avg, uy_avg, uz_avg);
        f_fine[id_f + q * n_cells_f] = feq_q + scale * f_neq_avg[q];
    }
}

/**
 * @brief Restriction kernel: write coarse cell PDF from average of 8
 *        fine cells covering it (in patch INTERIOR only — 1-cell margin
 *        from patch boundary acts as buffer per Lagrava overlap convention).
 *
 * For each coarse cell (i_c, j_c, k_c) inside the patch interior (NOT
 * on the patch boundary), computes (ρ_f, u_f) and f_neq_f averaged
 * over the 8 fine cells covering it, then writes coarse PDF =
 * f_eq(ρ_avg, u_avg) + (2·ω_f / ω_c) · f_neq_avg.
 *
 * SOLID-AWARE (Phase 2.4 fix 2026-05-18): if ANY of the 8 fine cells
 * is solid, the coarse cell is NOT restricted (keeps its coarse-step
 * value). Reason: fine solid cells have no valid PDF (streaming skips
 * them); their stale init values would contaminate the average and
 * produce a wildly wrong coarse PDF, observed as Cd→4.4 (30× too large)
 * vs Cl moving correctly toward literature. Skipping near-wall coarse
 * cells leaves the force probe reading clean coarse-step values for
 * cells right at the airfoil while still propagating fine information
 * to coarse cells one layer further out.
 *
 * Also skips coarse cells that are themselves solid (defensive).
 *
 * @param f_fine     Fine PDF (SoA).
 * @param f_coarse   Coarse PDF buffer (SoA, written at patch-interior cells).
 * @param solid_c    Coarse solid mask (defensive).
 * @param solid_f    Fine solid mask (used to skip restriction near walls).
 * @param i_lo,j_lo,k_lo  Patch lower corner in coarse cell indices.
 * @param i_hi,j_hi,k_hi  Patch upper corner (exclusive).
 * @param refine          Refinement factor (= 2).
 */
__global__ void restrictFineToCoarsePatch(
    const float* __restrict__ f_fine,
    float*       __restrict__ f_coarse,
    const unsigned char* __restrict__ solid_c,
    const unsigned char* __restrict__ solid_f,
    int i_lo, int j_lo, int k_lo,
    int i_hi, int j_hi, int k_hi,
    int refine,
    int nx_c, int ny_c, int nz_c,
    int nx_f, int ny_f, int nz_f,
    float omega_c, float omega_f)
{
    using lbm::core::D3Q27;

    // Launch over coarse cells in patch INTERIOR (1-cell margin on each side).
    const int i_c = blockIdx.x * blockDim.x + threadIdx.x + (i_lo + 1);
    const int j_c = blockIdx.y * blockDim.y + threadIdx.y + (j_lo + 1);
    const int k_c = blockIdx.z * blockDim.z + threadIdx.z + k_lo;

    if (i_c >= i_hi - 1 || j_c >= j_hi - 1 || k_c >= k_hi) return;

    const int id_c = i_c + j_c * nx_c + k_c * nx_c * ny_c;
    const int n_cells_c = nx_c * ny_c * nz_c;
    const int n_cells_f = nx_f * ny_f * nz_f;

    // Defensive: skip if coarse cell is solid (force probe doesn't use these,
    // but writing junk could affect adjacent cells via next collision).
    if (solid_c && solid_c[id_c] != 0) return;

    // Fine cells covering coarse (i_c, j_c, k_c):
    // 2 × 2 × 2 fine cells starting at fine indices (2·(i_c-i_lo), 2·(j_c-j_lo), 2·(k_c-k_lo)).
    const int i_f_lo = refine * (i_c - i_lo);
    const int j_f_lo = refine * (j_c - j_lo);
    const int k_f_lo = refine * (k_c - k_lo);

    // First pass: check if ALL 8 fine cells are fluid (none solid).
    // If any is solid, skip restriction for this coarse cell — keep coarse value.
    for (int dk = 0; dk < refine; ++dk)
    for (int dj = 0; dj < refine; ++dj)
    for (int di = 0; di < refine; ++di) {
        const int i_f = i_f_lo + di;
        const int j_f = j_f_lo + dj;
        const int k_f = k_f_lo + dk;
        if (i_f >= nx_f || j_f >= ny_f || k_f >= nz_f) return;  // partial coverage = skip
        const int id_f = i_f + j_f * nx_f + k_f * nx_f * ny_f;
        if (solid_f[id_f] != 0) return;  // any solid fine cell → skip restriction
    }

    // Accumulate sum of PDFs over 8 fine cells (all confirmed fluid).
    float f_sum[27] = {0.0f};
    int n_sum = 0;
    for (int dk = 0; dk < refine; ++dk)
    for (int dj = 0; dj < refine; ++dj)
    for (int di = 0; di < refine; ++di) {
        const int i_f = i_f_lo + di;
        const int j_f = j_f_lo + dj;
        const int k_f = k_f_lo + dk;
        const int id_f = i_f + j_f * nx_f + k_f * nx_f * ny_f;
        #pragma unroll
        for (int q = 0; q < 27; ++q) {
            f_sum[q] += f_fine[id_f + q * n_cells_f];
        }
        ++n_sum;
    }
    if (n_sum == 0) return;
    const float inv_n = 1.0f / (float)n_sum;
    #pragma unroll
    for (int q = 0; q < 27; ++q) {
        f_sum[q] *= inv_n;
    }

    // f_sum now holds the average fine PDF. Compute (ρ, u) from it.
    float rho = 0.0f, mx = 0.0f, my = 0.0f, mz = 0.0f;
    #pragma unroll
    for (int q = 0; q < 27; ++q) {
        rho += f_sum[q];
        mx  += ex27[q] * f_sum[q];
        my  += ey27[q] * f_sum[q];
        mz  += ez27[q] * f_sum[q];
    }
    const float rho_safe = fmaxf(rho, 1e-12f);
    const float ux = mx / rho_safe;
    const float uy = my / rho_safe;
    const float uz = mz / rho_safe;

    // f_neq_avg = f_sum - f_eq(ρ_avg, u_avg)
    // Rescale: f_neq^coarse = (2·ω_f / ω_c) · f_neq_avg.
    // Write coarse PDF = f_eq + f_neq_coarse.
    const float scale = (2.0f * omega_f) / omega_c;
    #pragma unroll
    for (int q = 0; q < 27; ++q) {
        const float feq_q = D3Q27::computeEquilibrium(q, rho, ux, uy, uz);
        const float f_neq_avg = f_sum[q] - feq_q;
        const float f_neq_c = scale * f_neq_avg;
        f_coarse[id_c + q * n_cells_c] = feq_q + f_neq_c;
    }
}

// =====================================================================
// Phase 2.5: time interpolation
// =====================================================================

/**
 * @brief Snapshot a band of coarse cells into a compact buffer.
 *
 * Band spans [i_lo-1, i_lo-1+band_nx) × [j_lo-1, j_lo-1+band_ny) × [k_lo, k_lo+band_nz)
 * in coarse cell indices.
 */
__global__ void snapshotCoarseBand(
    const float* __restrict__ f_coarse,
    float*       __restrict__ f_snap_band,
    int i_lo, int j_lo, int k_lo,
    int band_nx, int band_ny, int band_nz,
    int nx_c, int ny_c, int nz_c)
{
    const int bi = blockIdx.x * blockDim.x + threadIdx.x;
    const int bj = blockIdx.y * blockDim.y + threadIdx.y;
    const int bk = blockIdx.z * blockDim.z + threadIdx.z;
    if (bi >= band_nx || bj >= band_ny || bk >= band_nz) return;

    const int i_c = bi + i_lo - 1;
    const int j_c = bj + j_lo - 1;
    const int k_c = bk + k_lo;
    if (i_c < 0 || i_c >= nx_c || j_c < 0 || j_c >= ny_c ||
        k_c < 0 || k_c >= nz_c) return;

    const int id_c = i_c + j_c * nx_c + k_c * nx_c * ny_c;
    const int n_cells_c = nx_c * ny_c * nz_c;

    const int band_id = bi + bj * band_nx + bk * band_nx * band_ny;
    const int n_band = band_nx * band_ny * band_nz;

    #pragma unroll
    for (int q = 0; q < 27; ++q) {
        f_snap_band[band_id + q * n_band] = f_coarse[id_c + q * n_cells_c];
    }
}

/**
 * @brief Bicubic prolongation with linear TIME interp (Phase 2.5).
 *
 * Same 16-cell Catmull-Rom stencil as the non-time-interp variant, but each
 * stencil cell's PDF is read as 0.5·(f_coarse_cur + f_snap_band) — linear
 * temporal interpolation at the fine sub-step 1 midpoint t + δtc/2.
 */
__global__ void prolongateBoundaryFineWithTimeInterp(
    const float* __restrict__ f_coarse_cur,
    const float* __restrict__ f_snap_band,
    float*       __restrict__ f_fine,
    const unsigned char* __restrict__ solid_c,
    int i_lo, int j_lo, int k_lo,
    int band_nx, int band_ny, int band_nz,
    int refine,
    int nx_c, int ny_c, int nz_c,
    int nx_f, int ny_f, int nz_f,
    float omega_c, float omega_f)
{
    using lbm::core::D3Q27;

    const int i_f = blockIdx.x * blockDim.x + threadIdx.x;
    const int j_f = blockIdx.y * blockDim.y + threadIdx.y;
    const int k_f = blockIdx.z * blockDim.z + threadIdx.z;
    if (i_f >= nx_f || j_f >= ny_f || k_f >= nz_f) return;

    const bool on_x_edge = (i_f == 0 || i_f == nx_f - 1);
    const bool on_y_edge = (j_f == 0 || j_f == ny_f - 1);
    if (!on_x_edge && !on_y_edge) return;

    const int i_c = i_lo + i_f / refine;
    const int j_c = j_lo + j_f / refine;
    const int k_c = k_lo + k_f / refine;

    if (i_c < 0 || i_c >= nx_c || j_c < 0 || j_c >= ny_c ||
        k_c < 0 || k_c >= nz_c) return;

    const int qx = i_f % refine;
    const int qy = j_f % refine;

    int  dix[4], djy[4];
    float wx[4], wy[4];
    cubic_1d_q(qx, dix, wx);
    cubic_1d_q(qy, djy, wy);

    auto clamp_idx = [](int v, int v_max) -> int {
        return (v < 0) ? 0 : ((v >= v_max) ? (v_max - 1) : v);
    };

    const int n_cells_c = nx_c * ny_c * nz_c;
    const int n_cells_f = nx_f * ny_f * nz_f;
    const int n_band = band_nx * band_ny * band_nz;

    auto coarse_id = [&](int ic, int jc, int kc) -> int {
        return ic + jc * nx_c + kc * nx_c * ny_c;
    };
    auto band_id_of = [&](int ic, int jc, int kc) -> int {
        const int bi = ic - (i_lo - 1);
        const int bj = jc - (j_lo - 1);
        const int bk = kc - k_lo;
        if (bi < 0 || bi >= band_nx || bj < 0 || bj >= band_ny ||
            bk < 0 || bk >= band_nz) return -1;
        return bi + bj * band_nx + bk * band_nx * band_ny;
    };

    int   ids_c[16], ids_b[16];
    float ws[16];
    #pragma unroll
    for (int j_s = 0; j_s < 4; ++j_s) {
        #pragma unroll
        for (int i_s = 0; i_s < 4; ++i_s) {
            const int ic = clamp_idx(i_c + dix[i_s], nx_c);
            const int jc = clamp_idx(j_c + djy[j_s], ny_c);
            const int c = i_s + 4 * j_s;
            ids_c[c] = coarse_id(ic, jc, k_c);
            ids_b[c] = band_id_of(ic, jc, k_c);
            ws[c]    = wx[i_s] * wy[j_s];
        }
    }

    if (solid_c) {
        float wsum = 0.0f;
        #pragma unroll
        for (int c = 0; c < 16; ++c) {
            if (solid_c[ids_c[c]] != 0) ws[c] = 0.0f;
            wsum += ws[c];
        }
        if (!isfinite(wsum) || fabsf(wsum) < 1e-3f) return;
        const float inv_wsum = 1.0f / wsum;
        #pragma unroll
        for (int c = 0; c < 16; ++c) ws[c] *= inv_wsum;
    }

    auto read_pdf = [&](int q, int id_c, int id_b) -> float {
        const float v_cur = f_coarse_cur[id_c + q * n_cells_c];
        const float v_snap = (id_b >= 0) ? f_snap_band[id_b + q * n_band] : v_cur;
        return 0.5f * (v_cur + v_snap);
    };

    float rho_avg = 0.0f, ux_avg = 0.0f, uy_avg = 0.0f, uz_avg = 0.0f;
    float f_neq_avg[27] = {0.0f};

    #pragma unroll 4
    for (int c = 0; c < 16; ++c) {
        if (ws[c] == 0.0f) continue;
        float f_loc[27];
        float r = 0.0f, mx = 0.0f, my = 0.0f, mz = 0.0f;
        #pragma unroll
        for (int q = 0; q < 27; ++q) {
            const float v = read_pdf(q, ids_c[c], ids_b[c]);
            f_loc[q] = v;
            r += v;
            mx += ex27[q] * v;
            my += ey27[q] * v;
            mz += ez27[q] * v;
        }
        const float r_safe = fmaxf(r, 1e-6f);
        const float u_x = mx / r_safe;
        const float u_y = my / r_safe;
        const float u_z = mz / r_safe;
        rho_avg += ws[c] * r;
        ux_avg  += ws[c] * u_x;
        uy_avg  += ws[c] * u_y;
        uz_avg  += ws[c] * u_z;
        #pragma unroll
        for (int q = 0; q < 27; ++q) {
            const float feq_q = D3Q27::computeEquilibrium(q, r, u_x, u_y, u_z);
            f_neq_avg[q] += ws[c] * (f_loc[q] - feq_q);
        }
    }

    if (!isfinite(rho_avg) || rho_avg < 0.5f) {
        const int id_c_ctr = coarse_id(i_c, j_c, k_c);
        const int id_b_ctr = band_id_of(i_c, j_c, k_c);
        rho_avg = 0.0f;
        ux_avg = uy_avg = uz_avg = 0.0f;
        for (int q = 0; q < 27; ++q) f_neq_avg[q] = 0.0f;
        float mx = 0.0f, my = 0.0f, mz = 0.0f;
        float f_loc[27];
        #pragma unroll
        for (int q = 0; q < 27; ++q) {
            const float v = read_pdf(q, id_c_ctr, id_b_ctr);
            f_loc[q] = v;
            rho_avg += v;
            mx += ex27[q] * v;
            my += ey27[q] * v;
            mz += ez27[q] * v;
        }
        const float r_safe = fmaxf(rho_avg, 1e-6f);
        ux_avg = mx / r_safe; uy_avg = my / r_safe; uz_avg = mz / r_safe;
        #pragma unroll
        for (int q = 0; q < 27; ++q) {
            const float feq_q = D3Q27::computeEquilibrium(q, rho_avg, ux_avg, uy_avg, uz_avg);
            f_neq_avg[q] = f_loc[q] - feq_q;
        }
    }

    const float scale = omega_c / (2.0f * omega_f);
    const int id_f = i_f + j_f * nx_f + k_f * nx_f * ny_f;
    #pragma unroll
    for (int q = 0; q < 27; ++q) {
        const float feq_q = D3Q27::computeEquilibrium(q, rho_avg, ux_avg, uy_avg, uz_avg);
        f_fine[id_f + q * n_cells_f] = feq_q + scale * f_neq_avg[q];
    }
}

} // namespace amr
} // namespace physics
} // namespace lbm
