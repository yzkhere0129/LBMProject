/**
 * @file streaming_d3q27_qbb.h
 * @brief D3Q27 PULL-style streaming with single-node second-order QBB
 *        (lbmpy / walberla formula) on solid neighbours.
 *
 * Same algorithm as `fluidStreamingKernelWithSingleNodeQBB` (D3Q19) in
 * src/physics/fluid/fluid_lbm.cu, ported to D3Q27. See
 * docs/bouzidi_d3q27_design.md for the design rationale (eliminating the
 * stair-step Cl-sign artifact for rotated NACA airfoil).
 *
 * This kernel REPLACES `streamD3Q27_naca` in the NACA driver when
 * curved-BC mode is selected via `--bc qbb-snode`.
 *
 * Conventions (must match the driver):
 *   - X face (i=0, i=nx-1): halfway BB (driver overwrites with inlet/outlet).
 *   - Y face (j=0, j=ny-1): free-slip mirror (preserve tangential u_x, flip u_y).
 *   - Z face: periodic.
 *   - Obstacle (solid_mask != 0): single-node QBB using qfrac[X, opp(Q)] for
 *     each PULL-direction Q whose source X-e_Q is solid.
 */

#pragma once

#include <cuda_runtime.h>

namespace lbm {
namespace physics {
namespace cumulant {

/**
 * @brief PULL-style D3Q27 streaming kernel with curved-wall single-node QBB.
 *
 * @param f_src       Input populations (q-major SoA, 27·n_cells floats)
 * @param f_dst       Output populations (q-major SoA, 27·n_cells floats)
 * @param solid_mask  Cell flag, 0 = fluid, non-zero = solid
 * @param qfrac       Per-cell-per-link q-fraction (n_cells*27 floats), only
 *                    indices (id, q) with X+e_q solid are read; others may
 *                    be 1.0 sentinel.
 * @param nx,ny,nz    Domain extents
 * @param omega       Relaxation rate ω = 1/τ (BGK-style; passed through to
 *                    lbmpy formula). For Cumulant, use the same shear ω as
 *                    the collision (ω_ν).
 *
 * Launch: one thread per cell (3D grid).
 * Solid cells skip work (early return).
 */
__global__ void streamD3Q27_naca_qbb(
    const float* __restrict__ f_src,
    float* __restrict__ f_dst,
    const unsigned char* __restrict__ solid_mask,
    const float* __restrict__ qfrac,
    int nx, int ny, int nz,
    float omega);

} // namespace cumulant
} // namespace physics
} // namespace lbm
