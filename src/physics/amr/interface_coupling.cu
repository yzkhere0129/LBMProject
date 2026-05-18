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
 * @brief Prolongation kernel: write boundary fine cells from nearest coarse cell.
 *
 * Each fine cell at the boundary of the patch (i_f == 0 or nx_f-1, etc.)
 * gets its 27 PDFs OVERWRITTEN with the prolongated value from the
 * coarse cell that contains it. Interior fine cells are untouched.
 *
 * Rescaling: f_fine = f_eq(ρ_c, u_c) + (ω_c / (2·ω_f)) · (f_coarse - f_eq).
 *
 * @param f_coarse   Coarse PDF (SoA, n_cells_c = nx_c*ny_c*nz_c).
 * @param f_fine     Fine PDF buffer to write into (SoA, n_cells_f).
 * @param i_lo,j_lo,k_lo  Patch lower corner in coarse cell indices.
 * @param refine     Refinement factor (= 2).
 * @param nx_c,ny_c,nz_c  Coarse domain extents.
 * @param nx_f,ny_f,nz_f  Fine patch extents.
 * @param omega_c, omega_f  Shear relaxation rates (Lagrava eq. 24).
 */
__global__ void prolongateBoundaryFineFromCoarse(
    const float* __restrict__ f_coarse,
    float*       __restrict__ f_fine,
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

    // Only process boundary fine cells (1-cell layer at each edge).
    const bool on_x_edge = (i_f == 0 || i_f == nx_f - 1);
    const bool on_y_edge = (j_f == 0 || j_f == ny_f - 1);
    // z is periodic; skip z edges (no prolongation needed for periodic).
    if (!on_x_edge && !on_y_edge) return;

    // Map fine cell → containing coarse cell.
    const int i_c = i_lo + i_f / refine;
    const int j_c = j_lo + j_f / refine;
    const int k_c = k_lo + k_f / refine;

    // Clamp to coarse domain (should be inside since patch is interior).
    if (i_c < 0 || i_c >= nx_c || j_c < 0 || j_c >= ny_c ||
        k_c < 0 || k_c >= nz_c) return;

    const int id_c = i_c + j_c * nx_c + k_c * nx_c * ny_c;
    const int n_cells_c = nx_c * ny_c * nz_c;

    const int id_f = i_f + j_f * nx_f + k_f * nx_f * ny_f;
    const int n_cells_f = nx_f * ny_f * nz_f;

    // Load coarse PDF, compute (ρ, u) and f_neq.
    float f_c[27];
    float rho = 0.0f, mx = 0.0f, my = 0.0f, mz = 0.0f;
    #pragma unroll
    for (int q = 0; q < 27; ++q) {
        const float v = f_coarse[id_c + q * n_cells_c];
        f_c[q] = v;
        rho += v;
        mx  += ex27[q] * v;
        my  += ey27[q] * v;
        mz  += ez27[q] * v;
    }
    const float rho_safe = fmaxf(rho, 1e-12f);
    const float ux = mx / rho_safe;
    const float uy = my / rho_safe;
    const float uz = mz / rho_safe;

    // f_neq^coarse = f^coarse - f^eq(ρ_c, u_c).
    // Rescale: f_neq^fine = (ω_c / (2·ω_f)) · f_neq^coarse.
    const float scale = omega_c / (2.0f * omega_f);
    #pragma unroll
    for (int q = 0; q < 27; ++q) {
        const float feq_q = D3Q27::computeEquilibrium(q, rho, ux, uy, uz);
        const float f_neq_c = f_c[q] - feq_q;
        const float f_neq_f = scale * f_neq_c;
        f_fine[id_f + q * n_cells_f] = feq_q + f_neq_f;
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

} // namespace amr
} // namespace physics
} // namespace lbm
