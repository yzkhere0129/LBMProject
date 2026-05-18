/**
 * @file fine_patch_streaming.cu
 * @brief D3Q27 QBB streaming + MEM force probe for the fine patch.
 *
 * Phase 1: verbatim clone of physics::cumulant::streamD3Q27_naca_qbb_sparse
 * and the inline MEM force kernel from aero_naca0012_cumulant.cu. Distinct
 * symbol names ensure the fine path is independent.
 *
 * Edge BC behavior (Phase 1 islanded fine patch):
 *   Z: periodic    (same as coarse)
 *   Y: free-slip mirror via y_mirror27_qbb    (same as coarse)
 *   X: bounce-back via opposite27             (same as coarse, will be replaced by ghost-layer prolongation in Phase 2)
 *
 * TODO (post Phase 2): if behavior remains identical to coarse, dedup
 * with src/physics/cumulant/streaming_d3q27_qbb.cu by sharing helpers.
 *
 * Reference: docs/AMR_DESIGN_PROPOSAL.md §6 Phase 1.3.
 */

#include "physics/cumulant/cumulant_d3q27.h"
#include "core/lattice_d3q27.h"

namespace lbm {
namespace physics {
namespace amr {

// =====================================================================
// File-local constants and helpers (TODO: move to shared header for dedup)
// =====================================================================

// Y-mirror reflection table for free-slip walls. Flips c_y, preserves c_x, c_z.
// Verified hand-by-hand 2026-05-16 against the D3Q27 stencil — see
// HANDOFF_TO_NEW_PARTNER.md §4.1 for derivation.
__constant__ int y_mirror27_qbb_fine[27] = {
    0, 1, 2, 4, 3, 5, 6,           // rest + faces
    9, 10, 7, 8,                    // xy edges (q=7..10)
    11, 12, 13, 14,                 // xz edges (q=11..14)
    16, 15, 18, 17,                 // yz edges (q=15..18)
    23, 24, 26, 25, 19, 20, 22, 21  // corners (q=19..26)
};

/**
 * @brief Sparse-CSR lookup for the q-fraction along a given link.
 *
 * @param id      Cell linear index.
 * @param Q_opp   Direction index (1..26) pointing FROM cell INTO solid.
 * @return  qfrac in [0, 1] if found; 0.5 (halfway-BB fallback) otherwise.
 */
__device__ inline float lookup_sparse_qfrac_fine(
    int id, unsigned char Q_opp,
    const int* __restrict__ offset,
    const unsigned char* __restrict__ link_q,
    const float* __restrict__ link_val)
{
    int start = offset[id];
    int end   = offset[id + 1];
    #pragma unroll 4
    for (int e = start; e < end; ++e) {
        if (link_q[e] == Q_opp) return link_val[e];
    }
    return 0.5f;
}

// =====================================================================
// Fine-patch streaming with single-node QBB on NACA wall
// =====================================================================

__global__ void fineStreamD3Q27_naca_qbb_sparse(
    const float* __restrict__ f_src,
    float* __restrict__ f_dst,
    const unsigned char* __restrict__ solid_mask,
    const int*           __restrict__ qf_offset,
    const unsigned char* __restrict__ qf_link_q,
    const float*         __restrict__ qf_link_val,
    int nx, int ny, int nz,
    float omega)
{
    using lbm::core::ex27;
    using lbm::core::ey27;
    using lbm::core::ez27;
    using lbm::core::opposite27;
    using lbm::core::D3Q27;

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int idy = blockIdx.y * blockDim.y + threadIdx.y;
    const int idz = blockIdx.z * blockDim.z + threadIdx.z;
    if (idx >= nx || idy >= ny || idz >= nz) return;

    const int id = idx + idy * nx + idz * nx * ny;
    const int n_cells = nx * ny * nz;
    if (solid_mask[id] != 0) return;

    // Local moments
    float rho = 0.0f, mx = 0.0f, my = 0.0f, mz = 0.0f;
    #pragma unroll
    for (int q = 0; q < 27; ++q) {
        const float fq = f_src[id + q * n_cells];
        rho += fq;
        mx += ex27[q] * fq;
        my += ey27[q] * fq;
        mz += ez27[q] * fq;
    }
    const float rho_safe = fmaxf(rho, 1e-12f);
    const float ux = mx / rho_safe;
    const float uy = my / rho_safe;
    const float uz = mz / rho_safe;

    const float inv_one_minus_omega = 1.0f / (1.0f - omega);
    // Same relaxed clamp as coarse (debug 2026-05-15 patch).
    constexpr float QMIN = 1e-3f;
    constexpr float QMAX = 0.999f;

    for (int Q = 0; Q < 27; ++Q) {
        int src_x = idx - ex27[Q];
        int src_y = idy - ey27[Q];
        int src_z = idz - ez27[Q];

        // Z periodic
        if (src_z < 0)   src_z += nz;
        if (src_z >= nz) src_z -= nz;

        // X bounce-back / Y free-slip mirror at edges
        // (Phase 1 isolation: this is the wrong BC for a true interior
        //  patch but stable for the islanded test. Phase 2 replaces with
        //  ghost-layer prolongation from coarse.)
        bool oob_x = (src_x < 0 || src_x >= nx);
        bool oob_y = (src_y < 0 || src_y >= ny);
        if (oob_x || oob_y) {
            const int reflected_q = oob_y ? y_mirror27_qbb_fine[Q]
                                          : opposite27[Q];
            f_dst[id + Q * n_cells] = f_src[id + reflected_q * n_cells];
            continue;
        }

        const int src_id = src_x + src_y * nx + src_z * nx * ny;
        if (solid_mask[src_id] == 0) {
            // Pull from fluid neighbour.
            f_dst[id + Q * n_cells] = f_src[src_id + Q * n_cells];
            continue;
        }

        // Source is solid: single-node QBB (lbmpy formula).
        const int Q_opp = opposite27[Q];
        float qf = lookup_sparse_qfrac_fine(
            id, static_cast<unsigned char>(Q_opp),
            qf_offset, qf_link_q, qf_link_val);
        if (qf > QMAX) qf = QMAX;
        if (qf < QMIN) qf = QMIN;

        const float f_in  = f_src[id + Q_opp * n_cells];
        const float f_out = f_src[id + Q     * n_cells];

        const float feq_a   = D3Q27::computeEquilibrium(Q,     rho, ux, uy, uz);
        const float feq_b   = D3Q27::computeEquilibrium(Q_opp, rho, ux, uy, uz);
        const float feq_sym = feq_a + feq_b;

        const float t1 = (f_in - f_out)
                       + (f_in + f_out - omega * feq_sym) * inv_one_minus_omega;
        const float t2 = qf * (f_in + f_out) / (1.0f + qf);
        const float result = ((1.0f - qf) / (1.0f + qf)) * 0.5f * t1 + t2;

        f_dst[id + Q * n_cells] = result;
    }
}

// =====================================================================
// Fine-patch MEM force probe at NACA wall (sparse QBB variant)
// =====================================================================
// Verbatim clone of memForceNaca_QBB_sparse from the NACA driver.
// Same formula: F_link = c_q · (f_in + f_out_new), where f_out_new is
// reconstructed from local moments + qfrac (same single-node QBB
// algorithm as the streaming kernel).

__global__ void fineMemForceNaca_QBB_sparse(
    const float* __restrict__ f,
    const unsigned char* __restrict__ solid_mask,
    const int*           __restrict__ qf_offset,
    const unsigned char* __restrict__ qf_link_q,
    const float*         __restrict__ qf_link_val,
    float omega,
    int nx, int ny, int nz,
    double* Fx_acc, double* Fy_acc, double* Fz_acc)
{
    using lbm::core::ex27;
    using lbm::core::ey27;
    using lbm::core::ez27;
    using lbm::core::opposite27;
    using lbm::core::D3Q27;

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int idy = blockIdx.y * blockDim.y + threadIdx.y;
    const int idz = blockIdx.z * blockDim.z + threadIdx.z;
    if (idx >= nx || idy >= ny || idz >= nz) return;

    const int id = idx + idy * nx + idz * nx * ny;
    const int n_cells = nx * ny * nz;
    if (solid_mask[id] != 0) return;

    // Local moments
    float rho = 0.0f, mx = 0.0f, my = 0.0f, mz = 0.0f;
    #pragma unroll
    for (int q = 0; q < 27; ++q) {
        const float fq = f[id + q * n_cells];
        rho += fq;
        mx += ex27[q] * fq;
        my += ey27[q] * fq;
        mz += ez27[q] * fq;
    }
    const float rho_safe = fmaxf(rho, 1e-12f);
    const float ux = mx / rho_safe;
    const float uy = my / rho_safe;
    const float uz = mz / rho_safe;

    const float inv_one_minus_omega = 1.0f / (1.0f - omega);
    constexpr float QMIN_F = 1e-3f;
    constexpr float QMAX_F = 0.999f;

    double fx_local = 0.0, fy_local = 0.0, fz_local = 0.0;
    for (int q = 1; q < 27; ++q) {
        int dst_x = idx + ex27[q];
        int dst_y = idy + ey27[q];
        int dst_z = idz + ez27[q];
        if (dst_z < 0)  dst_z += nz;
        if (dst_z >= nz) dst_z -= nz;
        if (dst_x < 0 || dst_x >= nx || dst_y < 0 || dst_y >= ny) continue;
        const int dst_id = dst_x + dst_y * nx + dst_z * nx * ny;
        if (solid_mask[dst_id] == 0) continue;

        const int q_opp = opposite27[q];
        float qf = lookup_sparse_qfrac_fine(
            id, static_cast<unsigned char>(q),
            qf_offset, qf_link_q, qf_link_val);
        if (qf > QMAX_F) qf = QMAX_F;
        if (qf < QMIN_F) qf = QMIN_F;

        const float f_in  = f[id + q     * n_cells];
        const float f_out = f[id + q_opp * n_cells];

        const float feq_a   = D3Q27::computeEquilibrium(q,     rho, ux, uy, uz);
        const float feq_b   = D3Q27::computeEquilibrium(q_opp, rho, ux, uy, uz);
        const float feq_sym = feq_a + feq_b;

        const float t1 = (f_in - f_out)
                       + (f_in + f_out - omega * feq_sym) * inv_one_minus_omega;
        const float t2 = qf * (f_in + f_out) / (1.0f + qf);
        const float f_out_new = ((1.0f - qf) / (1.0f + qf)) * 0.5f * t1 + t2;

        const double sum = (double)f_in + (double)f_out_new;
        fx_local += (double)ex27[q] * sum;
        fy_local += (double)ey27[q] * sum;
        fz_local += (double)ez27[q] * sum;
    }
    if (fx_local != 0.0) atomicAdd(Fx_acc, fx_local);
    if (fy_local != 0.0) atomicAdd(Fy_acc, fy_local);
    if (fz_local != 0.0) atomicAdd(Fz_acc, fz_local);
}

} // namespace amr
} // namespace physics
} // namespace lbm
