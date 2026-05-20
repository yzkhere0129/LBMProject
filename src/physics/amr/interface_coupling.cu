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
// R1b: Lagrava 1D-along-interface cubic helpers.
// ----------------------------------------------------------------------------
// Catmull-Rom 4-point cubic Lagrange interpolation; w sums to 1 ∀ t∈[0,1].
//   w[0] applies to stencil cell at offset -1 from interp interval start
//   w[1] applies to offset  0
//   w[2] applies to offset +1
//   w[3] applies to offset +2
// At t=0.5, weights are {-1/16, 9/16, 9/16, -1/16} — Lagrava 2012 §3.6.
__device__ inline void catmull_rom_weights_t(float t, float w[4]) {
    const float one_p_t = 1.0f + t;
    const float one_m_t = 1.0f - t;
    const float two_m_t = 2.0f - t;
    w[0] = -0.5f * t * one_m_t * two_m_t;
    w[1] =  0.5f * one_p_t * one_m_t * two_m_t;
    w[2] =  0.5f * one_p_t * t * two_m_t;
    w[3] = -0.5f * one_p_t * t * one_m_t;
}

// 1D cubic stencil offsets + weights for a fine cell at coarse-frame offset
// ±0.25 (cell-center 2× refinement, qx∈{0,1}).
//   q=0 → fine offset -0.25 (left half of containing) → stencil {-2,-1,0,+1}, t=0.75
//   q=1 → fine offset +0.25 (right half)              → stencil {-1, 0,+1,+2}, t=0.25
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
 * @brief R1b prolongation: 1D Catmull-Rom along interface tangent +
 *        1D linear in normal direction. Bilinear fallback at corners.
 *
 * Per Lagrava 2012 §3.6: linear (locally 2nd-order) interpolation creates
 * O(1) pressure jump at the coarse-fine interface, incompatible with LBM's
 * 2nd-order global accuracy. Cubic restores compatibility.
 *
 * R1 (2026-05-20 daf4836) tried 2D outer-product bicubic which overshot
 * in both directions and required spatial monotonicity-switch fallback
 * — that switch was itself a perturbation source that broke Cl
 * (Cl_rms 27× larger than bilinear). Reverted in 24473e2.
 *
 * R1b applies cubic ONLY in the tangential direction (along the patch
 * edge). The normal direction uses simple linear interpolation (2 cells,
 * weights 3/4 + 1/4). Negative cubic weights only appear in 1 direction
 * so overshoot is naturally bounded. Stencil = 8 cells (2 normal × 4
 * tangent) instead of 16 (full 2D bicubic) or 4 (bilinear).
 *
 * At PATCH CORNERS (cells on both x- and y-edge), there is no unique
 * "tangent direction" — fall back to 4-cell bilinear. Corner cells are
 * a tiny minority (4 cells per (nx-2)+(ny-2)) so this has negligible
 * impact on global accuracy.
 *
 * ρ-clamp limiter (no spatial scheme switching, unlike R1):
 *   ρ_avg is post-hoc clamped to [ρ_min, ρ_max] of the stencil.
 *   The SAME kernel + formula runs for every cell — the clamp is a
 *   sanity bound, not a scheme switch.
 *
 * @param f_coarse   Coarse PDF (SoA).
 * @param f_fine     Fine PDF (SoA, written at boundary cells only).
 * @param solid_c    Coarse solid mask (for fallback decision).
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

    const int qx = i_f % refine;
    const int qy = j_f % refine;

    auto clamp_idx = [](int v, int v_max) -> int {
        return (v < 0) ? 0 : ((v >= v_max) ? (v_max - 1) : v);
    };

    const int n_cells_c = nx_c * ny_c * nz_c;
    const int n_cells_f = nx_f * ny_f * nz_f;

    auto coarse_id = [&](int ic, int jc, int kc) -> int {
        return ic + jc * nx_c + kc * nx_c * ny_c;
    };

    // -----------------------------------------------------------------------
    // Build stencil (up to 8 cells):
    //   corner (both edges):   4-cell bilinear        (n=4)
    //   x-edge (i_f extreme):  cubic in y × linear in x  (n=8)
    //   y-edge (j_f extreme):  cubic in x × linear in y  (n=8)
    // -----------------------------------------------------------------------
    int   ids[8];
    float ws[8];
    int   n_stencil = 0;

    if (on_x_edge && on_y_edge) {
        // Corner cell: 4-cell bilinear.
        const int di_neigh = (qx == 0) ? -1 : +1;
        const int dj_neigh = (qy == 0) ? -1 : +1;
        const int i_c_n = clamp_idx(i_c + di_neigh, nx_c);
        const int j_c_n = clamp_idx(j_c + dj_neigh, ny_c);
        ids[0] = coarse_id(i_c,   j_c,   k_c); ws[0] = 9.0f / 16.0f;
        ids[1] = coarse_id(i_c_n, j_c,   k_c); ws[1] = 3.0f / 16.0f;
        ids[2] = coarse_id(i_c,   j_c_n, k_c); ws[2] = 3.0f / 16.0f;
        ids[3] = coarse_id(i_c_n, j_c_n, k_c); ws[3] = 1.0f / 16.0f;
        n_stencil = 4;
    } else if (on_x_edge) {
        // X-edge: tangent along Y → 4-cell cubic in Y. Normal along X → 2-cell linear.
        int   dj_cubic[4];
        float wy_cubic[4];
        cubic_1d_q(qy, dj_cubic, wy_cubic);
        const int di_neigh_x = (qx == 0) ? -1 : +1;
        // wx_lin[0] for the containing coarse cell (offset 0), wx_lin[1] for neighbor.
        const float wx_lin[2] = {0.75f, 0.25f};
        const int   dix_lin[2] = {0, di_neigh_x};
        #pragma unroll
        for (int jj = 0; jj < 4; ++jj) {
            #pragma unroll
            for (int ii = 0; ii < 2; ++ii) {
                const int ic = clamp_idx(i_c + dix_lin[ii], nx_c);
                const int jc = clamp_idx(j_c + dj_cubic[jj], ny_c);
                const int c = ii + 2 * jj;
                ids[c] = coarse_id(ic, jc, k_c);
                ws[c]  = wx_lin[ii] * wy_cubic[jj];
            }
        }
        n_stencil = 8;
    } else {
        // Y-edge: tangent along X, normal along Y.
        int   di_cubic[4];
        float wx_cubic[4];
        cubic_1d_q(qx, di_cubic, wx_cubic);
        const int dj_neigh_y = (qy == 0) ? -1 : +1;
        const float wy_lin[2] = {0.75f, 0.25f};
        const int   djy_lin[2] = {0, dj_neigh_y};
        #pragma unroll
        for (int jj = 0; jj < 2; ++jj) {
            #pragma unroll
            for (int ii = 0; ii < 4; ++ii) {
                const int ic = clamp_idx(i_c + di_cubic[ii], nx_c);
                const int jc = clamp_idx(j_c + djy_lin[jj], ny_c);
                const int c = ii + 4 * jj;
                ids[c] = coarse_id(ic, jc, k_c);
                ws[c]  = wx_cubic[ii] * wy_lin[jj];
            }
        }
        n_stencil = 8;
    }

    // Solid-aware: zero weight on solid cells, renormalize remainder.
    if (solid_c) {
        float wsum = 0.0f;
        for (int c = 0; c < n_stencil; ++c) {
            if (solid_c[ids[c]] != 0) ws[c] = 0.0f;
            wsum += ws[c];
        }
        if (!isfinite(wsum) || fabsf(wsum) < 1e-3f) return;
        const float inv_wsum = 1.0f / wsum;
        for (int c = 0; c < n_stencil; ++c) ws[c] *= inv_wsum;
    }

    // Per-cell CE decomp + track ρ stencil min/max for clamp limiter.
    float rho_avg = 0.0f, ux_avg = 0.0f, uy_avg = 0.0f, uz_avg = 0.0f;
    float f_neq_avg[27] = {0.0f};
    float rho_min = 1e30f, rho_max = -1e30f;

    for (int c = 0; c < n_stencil; ++c) {
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

        rho_min = fminf(rho_min, r);
        rho_max = fmaxf(rho_max, r);

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

    // ρ-clamp limiter: bound rho_avg to stencil range. f_neq_avg unmodified
    // (sub-grid noise in f_neq is much smaller than ρ-overshoot in our flows).
    // This is a POST-HOC sanity bound — the SAME kernel runs for every cell.
    // No spatial scheme switching (which was the R1 instability).
    if (isfinite(rho_min) && isfinite(rho_max) && rho_min <= rho_max) {
        rho_avg = fmaxf(rho_min, fminf(rho_max, rho_avg));
    }

    // Catastrophic guard: if rho_avg is still unphysical, fall back to
    // containing cell (rare; only if all stencil cells had bad data).
    if (!isfinite(rho_avg) || rho_avg < 0.5f) {
        const int id_ctr = coarse_id(i_c, j_c, k_c);
        rho_avg = 0.0f; ux_avg = uy_avg = uz_avg = 0.0f;
        for (int q = 0; q < 27; ++q) f_neq_avg[q] = 0.0f;
        float mx = 0.0f, my = 0.0f, mz = 0.0f;
        float f_loc[27];
        #pragma unroll
        for (int q = 0; q < 27; ++q) {
            const float v = f_coarse[id_ctr + q * n_cells_c];
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
 * @brief R1b time-interp prolongation: 1D cubic along interface +
 *        1D linear normal, with linear TIME interp at sub-step 1.
 *
 * Same structure as the non-time-interp variant — see comments there.
 * Each stencil cell's PDF is the 0.5·(f_coarse_cur + f_snap_band)
 * time-average evaluated at fine sub-step 1 (t + δt_c/2).
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

    // Build stencil (4 corner, 8 edge) — identical layout to non-time variant.
    int   ids_c[8], ids_b[8];
    float ws[8];
    int   n_stencil = 0;

    if (on_x_edge && on_y_edge) {
        const int di_neigh = (qx == 0) ? -1 : +1;
        const int dj_neigh = (qy == 0) ? -1 : +1;
        const int i_c_n = clamp_idx(i_c + di_neigh, nx_c);
        const int j_c_n = clamp_idx(j_c + dj_neigh, ny_c);
        ids_c[0] = coarse_id(i_c,   j_c,   k_c); ids_b[0] = band_id_of(i_c,   j_c,   k_c); ws[0] = 9.0f/16.0f;
        ids_c[1] = coarse_id(i_c_n, j_c,   k_c); ids_b[1] = band_id_of(i_c_n, j_c,   k_c); ws[1] = 3.0f/16.0f;
        ids_c[2] = coarse_id(i_c,   j_c_n, k_c); ids_b[2] = band_id_of(i_c,   j_c_n, k_c); ws[2] = 3.0f/16.0f;
        ids_c[3] = coarse_id(i_c_n, j_c_n, k_c); ids_b[3] = band_id_of(i_c_n, j_c_n, k_c); ws[3] = 1.0f/16.0f;
        n_stencil = 4;
    } else if (on_x_edge) {
        int   dj_cubic[4];
        float wy_cubic[4];
        cubic_1d_q(qy, dj_cubic, wy_cubic);
        const int di_neigh_x = (qx == 0) ? -1 : +1;
        const float wx_lin[2] = {0.75f, 0.25f};
        const int   dix_lin[2] = {0, di_neigh_x};
        #pragma unroll
        for (int jj = 0; jj < 4; ++jj) {
            #pragma unroll
            for (int ii = 0; ii < 2; ++ii) {
                const int ic = clamp_idx(i_c + dix_lin[ii], nx_c);
                const int jc = clamp_idx(j_c + dj_cubic[jj], ny_c);
                const int c = ii + 2 * jj;
                ids_c[c] = coarse_id(ic, jc, k_c);
                ids_b[c] = band_id_of(ic, jc, k_c);
                ws[c]    = wx_lin[ii] * wy_cubic[jj];
            }
        }
        n_stencil = 8;
    } else {
        int   di_cubic[4];
        float wx_cubic[4];
        cubic_1d_q(qx, di_cubic, wx_cubic);
        const int dj_neigh_y = (qy == 0) ? -1 : +1;
        const float wy_lin[2] = {0.75f, 0.25f};
        const int   djy_lin[2] = {0, dj_neigh_y};
        #pragma unroll
        for (int jj = 0; jj < 2; ++jj) {
            #pragma unroll
            for (int ii = 0; ii < 4; ++ii) {
                const int ic = clamp_idx(i_c + di_cubic[ii], nx_c);
                const int jc = clamp_idx(j_c + djy_lin[jj], ny_c);
                const int c = ii + 4 * jj;
                ids_c[c] = coarse_id(ic, jc, k_c);
                ids_b[c] = band_id_of(ic, jc, k_c);
                ws[c]    = wx_cubic[ii] * wy_lin[jj];
            }
        }
        n_stencil = 8;
    }

    if (solid_c) {
        float wsum = 0.0f;
        for (int c = 0; c < n_stencil; ++c) {
            if (solid_c[ids_c[c]] != 0) ws[c] = 0.0f;
            wsum += ws[c];
        }
        if (!isfinite(wsum) || fabsf(wsum) < 1e-3f) return;
        const float inv_wsum = 1.0f / wsum;
        for (int c = 0; c < n_stencil; ++c) ws[c] *= inv_wsum;
    }

    // Time-averaged PDF read.
    auto read_pdf = [&](int q, int id_c, int id_b) -> float {
        const float v_cur = f_coarse_cur[id_c + q * n_cells_c];
        const float v_snap = (id_b >= 0) ? f_snap_band[id_b + q * n_band] : v_cur;
        return 0.5f * (v_cur + v_snap);
    };

    float rho_avg = 0.0f, ux_avg = 0.0f, uy_avg = 0.0f, uz_avg = 0.0f;
    float f_neq_avg[27] = {0.0f};
    float rho_min = 1e30f, rho_max = -1e30f;

    for (int c = 0; c < n_stencil; ++c) {
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

        rho_min = fminf(rho_min, r);
        rho_max = fmaxf(rho_max, r);

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

    if (isfinite(rho_min) && isfinite(rho_max) && rho_min <= rho_max) {
        rho_avg = fmaxf(rho_min, fminf(rho_max, rho_avg));
    }

    if (!isfinite(rho_avg) || rho_avg < 0.5f) {
        const int id_c_ctr = coarse_id(i_c, j_c, k_c);
        const int id_b_ctr = band_id_of(i_c, j_c, k_c);
        rho_avg = 0.0f; ux_avg = uy_avg = uz_avg = 0.0f;
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
