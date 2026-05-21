/**
 * @file streaming_d3q27_qbb.cu
 * @brief D3Q27 PULL streaming + single-node second-order QBB (lbmpy formula).
 *
 * Mechanical port of fluid_lbm.cu's `fluidStreamingKernelWithSingleNodeQBB`
 * from D3Q19 to D3Q27. The lbmpy formula is lattice-independent (only
 * depends on f_in, f_out, ω, qf, feq_sym), so the only differences from the
 * D3Q19 version are:
 *   - Q=27 (uses ex27/ey27/ez27/opposite27)
 *   - D3Q27::computeEquilibrium for feq_sym
 *   - free-slip Y boundary mirror (vs configurable in fluid_lbm.cu version)
 *
 * Mapping PUSH ⇄ PULL for QBB:
 *   PUSH:  f_in  = f_src[X, q_push]      (pop heading INTO solid)
 *          f_out = f_src[X, opp(q_push)]
 *          dst   = f_dst[X, opp(q_push)]
 *   PULL:  We compute f_dst[X, Q] for every Q. If src=X-e_Q is solid:
 *          q_push  = opp(Q)              (push direction toward solid)
 *          f_in    = f_src[X, opp(Q)]    (pop heading toward solid)
 *          f_out   = f_src[X, Q]         (pop heading away from solid)
 *          qf used = qfrac[X, opp(Q)]    (qfrac stores the link-to-solid frac)
 *          dst     = f_dst[X, Q]         ✓
 */

#include "physics/cumulant/streaming_d3q27_qbb.h"
#include "core/lattice_d3q27.h"

namespace lbm {
namespace physics {
namespace cumulant {

// y-mirror table for D3Q27 free-slip walls: flip cy, preserve cx and cz.
// Same table as the driver's `y_mirror27` in aero_naca0012_cumulant.cu;
// renamed here to avoid __constant__ symbol clash if both TUs end up linked.
__constant__ int y_mirror27_qbb[27] = {
    0, 1, 2, 4, 3, 5, 6,            // rest + faces (q=0..6)
    9, 10, 7, 8,                    // xy edges (q=7..10)
    11, 12, 13, 14,                 // xz edges (q=11..14, no cy component)
    16, 15, 18, 17,                 // yz edges (q=15..18)
    23, 24, 26, 25, 19, 20, 22, 21  // corners (q=19..26)
};

__global__ void streamD3Q27_naca_qbb(
    const float* __restrict__ f_src,
    float* __restrict__ f_dst,
    const unsigned char* __restrict__ solid_mask,
    const float* __restrict__ qfrac,
    int nx, int ny, int nz,
    float omega,
    bool z_wall)
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
    if (solid_mask[id] != 0) return;  // skip solid cells

    // Phase 1: local moments at this cell, needed for f_eq in QBB formula.
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
    // QMIN/QMAX relaxed 0.05/0.95 → 1e-3/0.999 (debug 2026-05-15):
    // original clamp was systematic geometric error at sub-cell LE/TE.
    constexpr float QMIN = 1e-3f;
    constexpr float QMAX = 0.999f;

    // Phase 2: PULL-stream. For each Q, compute f_dst[id, Q].
    for (int Q = 0; Q < 27; ++Q) {
        int src_x = idx - ex27[Q];
        int src_y = idy - ey27[Q];
        int src_z = idz - ez27[Q];

        // z: periodic by default; full bounce-back if `z_wall` flag is set.
        // (Phase 3.2 finite-span wing demos need no-slip at z=0/z=nz-1.)
        bool oob_z = false;
        if (src_z < 0 || src_z >= nz) {
            if (z_wall) {
                oob_z = true;  // handle below with opposite27 reflection
            } else {
                if (src_z < 0)   src_z += nz;
                if (src_z >= nz) src_z -= nz;
            }
        }

        // x: halfway BB (driver overwrites i=0/nx-1 with inlet/outlet);
        // y: free-slip mirror (preserves tangential u_x, flips u_y);
        // z (if wall): halfway BB (opposite27).
        bool oob_x = (src_x < 0 || src_x >= nx);
        bool oob_y = (src_y < 0 || src_y >= ny);
        if (oob_x || oob_y || oob_z) {
            // Priority: y mirror takes precedence (preserves tangent at top/bottom),
            // else opposite27 for x/z walls.
            const int reflected_q = oob_y ? y_mirror27_qbb[Q]
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

        // Source is solid: single-node QBB. The link from this cell toward
        // the solid is in direction opp(Q); qfrac[X, opp(Q)] holds the wall
        // distance fraction along that link.
        const int Q_opp = opposite27[Q];
        float qf = qfrac[id + Q_opp * n_cells];
        if (qf > QMAX) qf = QMAX;
        if (qf < QMIN) qf = QMIN;

        const float f_in  = f_src[id + Q_opp * n_cells];  // pop into solid
        const float f_out = f_src[id + Q     * n_cells];  // pop away from solid

        // f_eq is symmetric in (Q, Q_opp) for the +/- pair, so feq_sym is
        // the same regardless of which we call f_eq_q vs f_eq_opp.
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

// ---------------------------------------------------------------------------
// Sparse-qfrac variant. Identical algorithm, qfrac access via CSR-like lookup.
// ---------------------------------------------------------------------------

__device__ inline float lookup_sparse_qfrac(
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
    return 0.5f;  // shouldn't happen for solid-side link, fall back to halfway
}

__global__ void streamD3Q27_naca_qbb_sparse(
    const float* __restrict__ f_src,
    float* __restrict__ f_dst,
    const unsigned char* __restrict__ solid_mask,
    const int*           __restrict__ qf_offset,
    const unsigned char* __restrict__ qf_link_q,
    const float*         __restrict__ qf_link_val,
    int nx, int ny, int nz,
    float omega,
    bool z_wall)
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
    // Relaxed clamp (debug 2026-05-15): 0.05/0.95 → 1e-3/0.999.
    constexpr float QMIN = 1e-3f;
    constexpr float QMAX = 0.999f;

    for (int Q = 0; Q < 27; ++Q) {
        int src_x = idx - ex27[Q];
        int src_y = idy - ey27[Q];
        int src_z = idz - ez27[Q];

        bool oob_z = false;
        if (src_z < 0 || src_z >= nz) {
            if (z_wall) { oob_z = true; }
            else {
                if (src_z < 0)   src_z += nz;
                if (src_z >= nz) src_z -= nz;
            }
        }

        bool oob_x = (src_x < 0 || src_x >= nx);
        bool oob_y = (src_y < 0 || src_y >= ny);
        if (oob_x || oob_y || oob_z) {
            const int reflected_q = oob_y ? y_mirror27_qbb[Q]
                                          : opposite27[Q];
            f_dst[id + Q * n_cells] = f_src[id + reflected_q * n_cells];
            continue;
        }

        const int src_id = src_x + src_y * nx + src_z * nx * ny;
        if (solid_mask[src_id] == 0) {
            f_dst[id + Q * n_cells] = f_src[src_id + Q * n_cells];
            continue;
        }

        const int Q_opp = opposite27[Q];
        float qf = lookup_sparse_qfrac(
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

} // namespace cumulant
} // namespace physics
} // namespace lbm
