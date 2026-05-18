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

/**
 * @brief Prolongation kernel: bilinear (9/3/3/1) interp over 4 coarse cells
 *        with ρ-guard fallback to nearest-coarse when bilinear stencil
 *        yields an unphysical density.
 *
 * For cell-center 2× refinement, each boundary fine cell sits in one of
 * 4 quadrants of its containing coarse cell. Bilinear stencil uses the
 * containing cell (weight 9/16) + 2 face-neighbors (3/16 each) +
 * diagonal neighbor (1/16), with direction of neighbors set by the
 * fine cell's quadrant.
 *
 * Fallback: if bilinear ρ_avg < 0.5 (sanity floor, baseline ρ=1) OR
 * any of the 4 stencil cells is solid, use only the containing coarse
 * cell (weight 1.0, w_others = 0).
 *
 * Rescaling (Lagrava 2012 eq. 29):
 *   f_fine = f_eq(ρ_bilin, u_bilin) + (ω_c / 2ω_f) · f_neq_bilin
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

    // Quadrant direction within containing coarse cell.
    const int qx = i_f % refine;
    const int qy = j_f % refine;
    const int di_neigh = (qx == 0) ? -1 : +1;
    const int dj_neigh = (qy == 0) ? -1 : +1;

    auto clamp_idx = [](int v, int v_max) -> int {
        return (v < 0) ? 0 : ((v >= v_max) ? (v_max - 1) : v);
    };
    const int i_c_n = clamp_idx(i_c + di_neigh, nx_c);
    const int j_c_n = clamp_idx(j_c + dj_neigh, ny_c);

    const int n_cells_c = nx_c * ny_c * nz_c;
    const int n_cells_f = nx_f * ny_f * nz_f;

    auto coarse_id = [&](int ic, int jc, int kc) -> int {
        return ic + jc * nx_c + kc * nx_c * ny_c;
    };
    const int id_aa = coarse_id(i_c,    j_c,    k_c);  // 9/16
    const int id_ba = coarse_id(i_c_n,  j_c,    k_c);  // 3/16
    const int id_ab = coarse_id(i_c,    j_c_n,  k_c);  // 3/16
    const int id_bb = coarse_id(i_c_n,  j_c_n,  k_c);  // 1/16

    // Decide bilinear vs nearest based on solid presence in stencil.
    bool use_bilinear = true;
    if (solid_c) {
        if (solid_c[id_aa] != 0 || solid_c[id_ba] != 0 ||
            solid_c[id_ab] != 0 || solid_c[id_bb] != 0) {
            use_bilinear = false;
        }
    }

    float w_aa, w_ba, w_ab, w_bb;
    if (use_bilinear) {
        w_aa = 9.0f / 16.0f;
        w_ba = 3.0f / 16.0f;
        w_ab = 3.0f / 16.0f;
        w_bb = 1.0f / 16.0f;
    } else {
        w_aa = 1.0f; w_ba = 0.0f; w_ab = 0.0f; w_bb = 0.0f;
    }

    // Per-cell decomposition: compute ρ_i, u_i, f_eq_i, f_neq_i for each of
    // the 4 stencil cells, then BILINEAR the (ρ, u) and f_neq_i separately.
    // This is correct under Chapman-Enskog (each cell has its own consistent
    // CE expansion); the earlier approach of "bilinear sum f_cell, then
    // decompose with f_eq(sum_ρ, sum_u)" couples f_eq nonlinearly with sum
    // and introduces spurious f_neq components that quickly diverge.
    const int ids[4] = {id_aa, id_ba, id_ab, id_bb};
    const float ws[4] = {w_aa, w_ba, w_ab, w_bb};

    float rho_avg = 0.0f, ux_avg = 0.0f, uy_avg = 0.0f, uz_avg = 0.0f;
    float f_neq_avg[27] = {0.0f};

    #pragma unroll
    for (int c = 0; c < 4; ++c) {
        if (ws[c] <= 0.0f) continue;
        // Load PDF, compute moments.
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

    // Safety: if averaged ρ is unphysical, fall back to containing cell.
    if (!isfinite(rho_avg) || rho_avg < 0.5f) {
        rho_avg = 0.0f;
        ux_avg = uy_avg = uz_avg = 0.0f;
        for (int q = 0; q < 27; ++q) f_neq_avg[q] = 0.0f;
        float mx = 0.0f, my = 0.0f, mz = 0.0f;
        float f_loc[27];
        #pragma unroll
        for (int q = 0; q < 27; ++q) {
            const float v = f_coarse[id_aa + q * n_cells_c];
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
 * @brief Bilinear prolongation with linear TIME interp.
 *
 * Same structure as prolongateBoundaryFineFromCoarse but reads each
 * stencil cell from BOTH f_coarse_cur and f_snap_band, averages 0.5,
 * then runs per-cell CE decomposition + bilinear average.
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
    const int di_neigh = (qx == 0) ? -1 : +1;
    const int dj_neigh = (qy == 0) ? -1 : +1;

    auto clamp_idx = [](int v, int v_max) -> int {
        return (v < 0) ? 0 : ((v >= v_max) ? (v_max - 1) : v);
    };
    const int i_c_n = clamp_idx(i_c + di_neigh, nx_c);
    const int j_c_n = clamp_idx(j_c + dj_neigh, ny_c);

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

    const int id_aa_c = coarse_id(i_c,    j_c,    k_c);
    const int id_ba_c = coarse_id(i_c_n,  j_c,    k_c);
    const int id_ab_c = coarse_id(i_c,    j_c_n,  k_c);
    const int id_bb_c = coarse_id(i_c_n,  j_c_n,  k_c);
    const int id_aa_b = band_id_of(i_c,    j_c,    k_c);
    const int id_ba_b = band_id_of(i_c_n,  j_c,    k_c);
    const int id_ab_b = band_id_of(i_c,    j_c_n,  k_c);
    const int id_bb_b = band_id_of(i_c_n,  j_c_n,  k_c);

    bool use_bilinear = true;
    if (solid_c) {
        if (solid_c[id_aa_c] != 0 || solid_c[id_ba_c] != 0 ||
            solid_c[id_ab_c] != 0 || solid_c[id_bb_c] != 0) {
            use_bilinear = false;
        }
    }

    float w_aa, w_ba, w_ab, w_bb;
    if (use_bilinear) {
        w_aa = 9.0f / 16.0f;
        w_ba = 3.0f / 16.0f;
        w_ab = 3.0f / 16.0f;
        w_bb = 1.0f / 16.0f;
    } else {
        w_aa = 1.0f; w_ba = 0.0f; w_ab = 0.0f; w_bb = 0.0f;
    }

    // Helper: read PDF f_q at coarse stencil position, time-averaged.
    auto read_pdf = [&](int q, int id_c, int id_b) -> float {
        const float v_cur = f_coarse_cur[id_c + q * n_cells_c];
        const float v_snap = (id_b >= 0) ? f_snap_band[id_b + q * n_band] : v_cur;
        return 0.5f * (v_cur + v_snap);
    };

    const int ids_c[4] = {id_aa_c, id_ba_c, id_ab_c, id_bb_c};
    const int ids_b[4] = {id_aa_b, id_ba_b, id_ab_b, id_bb_b};
    const float ws[4] = {w_aa, w_ba, w_ab, w_bb};

    float rho_avg = 0.0f, ux_avg = 0.0f, uy_avg = 0.0f, uz_avg = 0.0f;
    float f_neq_avg[27] = {0.0f};

    #pragma unroll
    for (int c = 0; c < 4; ++c) {
        if (ws[c] <= 0.0f) continue;
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
        rho_avg = 0.0f;
        ux_avg = uy_avg = uz_avg = 0.0f;
        for (int q = 0; q < 27; ++q) f_neq_avg[q] = 0.0f;
        float mx = 0.0f, my = 0.0f, mz = 0.0f;
        float f_loc[27];
        #pragma unroll
        for (int q = 0; q < 27; ++q) {
            const float v = read_pdf(q, id_aa_c, id_aa_b);
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
