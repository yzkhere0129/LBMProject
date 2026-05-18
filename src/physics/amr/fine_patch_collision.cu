/**
 * @file fine_patch_collision.cu
 * @brief D3Q27 Cumulant collision kernel — fine-patch variant.
 *
 * Phase 1: verbatim clone of physics::cumulant::fluidCumulantCollisionKernel.
 * Distinct symbol name ensures the fine-patch code path is isolated from
 * the coarse driver. Allows independent optimization / modification of
 * fine-only behavior (e.g., BGK fallback in interface band — Phase 2)
 * without risk to the verified coarse kernel.
 *
 * TODO (post Phase 2): if the fine kernel diverges from the coarse kernel
 * (e.g., BGK fallback), keep separate. Else, deduplicate by extracting
 * the body to a shared `__device__ __forceinline__` helper.
 *
 * Reference: docs/AMR_DESIGN_PROPOSAL.md §6 Phase 1.
 */

#include "physics/cumulant/cumulant_d3q27.h"

#define MIDX(p, q, r) ((p) + 3 * (q) + 9 * (r))

namespace lbm {
namespace physics {
namespace amr {

/**
 * @brief Cumulant collision kernel on the fine patch (verbatim clone).
 *
 * Identical math to physics::cumulant::fluidCumulantCollisionKernel.
 * Operates on the fine-patch PDF buffer (n_cells = nx*ny*nz in fine
 * units).
 *
 * Phase 1: launched in-place (f_src == f_dst) since collision is local.
 * Phase 2: caller may choose BGK-equivalent omega override at interface
 * cells via a mask (planned).
 */
__global__ void fineCumulantCollisionKernel(
    const float* f_src, float* f_dst,
    float* rho_out, float* ux_out, float* uy_out, float* uz_out,
    int nx, int ny, int nz,
    float omega_nu, float omega_b,
    float omega_3, float omega_4, float omega_5, float omega_6)
{
    using lbm::physics::cumulant::computeRawMoments27;
    using lbm::physics::cumulant::shiftToCentralMoments27;
    using lbm::physics::cumulant::centralToCumulants27;
    using lbm::physics::cumulant::relaxCumulants27;
    using lbm::physics::cumulant::cumulantsToCentralMoments27;
    using lbm::physics::cumulant::centralToRawMoments27;
    using lbm::physics::cumulant::rawMomentsToPDFs27;

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int idy = blockIdx.y * blockDim.y + threadIdx.y;
    const int idz = blockIdx.z * blockDim.z + threadIdx.z;
    if (idx >= nx || idy >= ny || idz >= nz) return;

    const int id = idx + idy * nx + idz * nx * ny;
    const int n_cells = nx * ny * nz;

    // Load 27 PDFs into local register array
    float f[27];
    #pragma unroll
    for (int q = 0; q < 27; ++q) {
        f[q] = f_src[id + q * n_cells];
    }

    // Stage 2a: raw moments
    float m[27];
    computeRawMoments27(f, m);

    // Stage 2b: extract macroscopic + central moments
    const float rho = m[MIDX(0, 0, 0)];
    const float inv_rho = (rho > 1e-12f) ? (1.0f / rho) : 0.0f;
    const float ux = m[MIDX(1, 0, 0)] * inv_rho;
    const float uy = m[MIDX(0, 1, 0)] * inv_rho;
    const float uz = m[MIDX(0, 0, 1)] * inv_rho;

    float kappa[27];
    shiftToCentralMoments27(m, ux, uy, uz, kappa);

    // Stage 2c: central moments → cumulants
    float K[27];
    centralToCumulants27(kappa, rho, K);

    // Stage 2d: relax in cumulant space
    relaxCumulants27(K, rho, omega_nu, omega_b, omega_3, omega_4, omega_5, omega_6);

    // Stage 2e: inverse transforms
    cumulantsToCentralMoments27(K, rho, kappa);
    centralToRawMoments27(kappa, ux, uy, uz, m);
    rawMomentsToPDFs27(m, f);

    // Write back
    #pragma unroll
    for (int q = 0; q < 27; ++q) {
        f_dst[id + q * n_cells] = f[q];
    }

    if (rho_out) rho_out[id] = rho;
    if (ux_out)  ux_out[id]  = ux;
    if (uy_out)  uy_out[id]  = uy;
    if (uz_out)  uz_out[id]  = uz;
}

#undef MIDX

} // namespace amr
} // namespace physics
} // namespace lbm
