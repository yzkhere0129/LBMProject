/**
 * @file ib_spread_interp.h
 * @brief Spreading & interpolation kernels for Immersed Boundary LBM.
 *
 * Roma-Peskin 3-cell δ_h kernel (compact support 1.5·Δx per dimension).
 * 3D δ_h = (1/Δx³) · φ(rx) · φ(ry) · φ(rz),  r = (x_cell − X_marker) / Δx.
 *
 * Two GPU kernels:
 *   - interpolateVelocityKernel:  u_L(X_k) = Σ_cells u(x_c) · φ_x · φ_y · φ_z
 *   - spreadForceKernel:          f(x_c) += Σ_k F_L(X_k) · (1/Δx³) · φ · ds_k
 *
 * Spreading uses atomicAdd (race condition between markers writing to same
 * Eulerian cell). Marker count for NACA D/c=40 thin-slab nz=8: ~664 → ~5.4e4
 * atomicAdds per IB step (negligible cost).
 *
 * The φ template works for both float and double; GPU kernels use float for
 * production, host-side unit tests use double for analytical-grade verification
 * of partition of unity, conservation, and moment preservation.
 *
 * References:
 *   Roma A.M., Peskin C.S., Berger M.J. (1999). "An adaptive version of the
 *     immersed boundary method." J. Comput. Phys. 153, 509-534.
 *   Wu J. & Shu C. (2009). JCP 228, 1963 — IB-LBM framework.
 */

#pragma once

#include <cuda_runtime.h>

namespace lbm {
namespace physics {
namespace aero {

/**
 * @brief Roma-Peskin 3-cell 1D shape function.
 *
 * Compact support |r| ≤ 1.5. Continuous at branch boundary |r|=0.5 and
 * at the support edges. Partition-of-unity property:
 *     Σ_{i} φ(r − i) = 1   for any real r.
 *
 * @tparam T  float or double (kernels use float; tests use double).
 * @param r   Normalized distance (x_cell − X_marker) / Δx.
 * @return    φ(r) ∈ [0, 2/3].
 */
template <typename T>
__host__ __device__ inline T phi_rp3(T r) {
    const T ar = (r < T(0)) ? -r : r;
    if (ar > T(1.5)) return T(0);
    if (ar <= T(0.5)) {
        const T s = T(1) - T(3) * r * r;
        // s should be ≥ 0.25 in this branch; guard against tiny negative
        // from FP roundoff at exactly |r|=0.5
        return (T(1) / T(3)) * (T(1) + ((s > T(0)) ? sqrt(s) : T(0)));
    }
    const T t = T(1) - ar;        // t ∈ [-0.5, 0.5)
    const T s = T(1) - T(3) * t * t;
    return (T(1) / T(6)) * (T(5) - T(3) * ar - ((s > T(0)) ? sqrt(s) : T(0)));
}

/**
 * @brief Interpolate Eulerian velocity field to Lagrangian markers (GPU).
 *
 * For each marker k:
 *   u_L_x[k] = Σ_{(i,j,kk) in 3³ stencil} u_x(i,j,kk) · φ(rx)·φ(ry)·φ(rz)
 *   (similarly for u_L_y, u_L_z)
 *
 * Scan stencil: i ∈ {floor(xL/dx) − 1, floor(xL/dx), floor(xL/dx) + 1}
 * (and analogous for j, k). This 3-cell scan captures all nonzero φ values
 * because the support has width 3 (in cell units) for any marker position.
 *
 * @param d_ux,uy,uz       Eulerian velocity field (n_cells = nx·ny·nz floats)
 * @param nx, ny, nz       Domain extents
 * @param dx               Lattice spacing [m]
 * @param periodic_z       1 if z-direction is periodic, 0 if walls
 * @param d_xL,yL,zL       Lagrangian marker positions (n_markers floats each)
 * @param n_markers        Number of markers
 * @param d_uL_x,y,z       Output interpolated velocity (n_markers floats each)
 *
 * Launch: 1 thread per marker.
 */
__global__ void interpolateVelocityKernel(
    const float* __restrict__ d_ux,
    const float* __restrict__ d_uy,
    const float* __restrict__ d_uz,
    int nx, int ny, int nz, float dx,
    int periodic_z,
    const float* __restrict__ d_xL,
    const float* __restrict__ d_yL,
    const float* __restrict__ d_zL,
    int n_markers,
    float* __restrict__ d_uL_x,
    float* __restrict__ d_uL_y,
    float* __restrict__ d_uL_z);

/**
 * @brief Spread Lagrangian force to Eulerian field (GPU, atomicAdd).
 *
 * For each marker k, atomic-add to 3³ Eulerian cells:
 *   ΔF(i,j,kk) = F_L[k] · (1/Δx³) · φ(rx)·φ(ry)·φ(rz) · ds[k]
 *
 * The (1/Δx³) factor is the δ_h normalisation; ds[k] is the arc-length
 * element associated with marker k. Caller must zero F_field before launch.
 *
 * @param d_xL,yL,zL,ds    Marker positions + arc-length elements
 * @param d_FL_x,y,z       Lagrangian force per marker
 * @param n_markers
 * @param nx,ny,nz,dx
 * @param periodic_z
 * @param d_F_x,y,z_field  Eulerian force field — atomically updated
 *
 * Launch: 1 thread per marker. Each thread does 27 atomicAdds × 3 components
 * = 81 atomicAdds.
 */
__global__ void spreadForceKernel(
    const float* __restrict__ d_xL,
    const float* __restrict__ d_yL,
    const float* __restrict__ d_zL,
    const float* __restrict__ d_ds,
    const float* __restrict__ d_FL_x,
    const float* __restrict__ d_FL_y,
    const float* __restrict__ d_FL_z,
    int n_markers,
    int nx, int ny, int nz, float dx,
    int periodic_z,
    float* d_F_x_field,
    float* d_F_y_field,
    float* d_F_z_field);

} // namespace aero
} // namespace physics
} // namespace lbm
