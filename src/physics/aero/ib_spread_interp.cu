/**
 * @file ib_spread_interp.cu
 * @brief Implementation of IB-LBM spreading & interpolation kernels.
 */

#include "physics/aero/ib_spread_interp.h"

#include <cmath>

namespace lbm {
namespace physics {
namespace aero {

namespace {

// Reflect z index into [0, nz) for periodic z. Markers may sit anywhere in z;
// for thin-slab nz=8 with markers at (k+0.5)·dx the wrap distance is at most 1.
__device__ inline int wrap_z(int k, int nz) {
    if (k < 0) k += nz;
    if (k >= nz) k -= nz;
    return k;
}

} // anonymous

// ---------------------------------------------------------------------------
// interpolateVelocityKernel
// ---------------------------------------------------------------------------
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
    float* __restrict__ d_uL_z)
{
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= n_markers) return;

    const float xL = d_xL[k];
    const float yL = d_yL[k];
    const float zL = d_zL[k];

    const float xnorm = xL / dx;
    const float ynorm = yL / dx;
    const float znorm = zL / dx;
    const int i_c = static_cast<int>(floorf(xnorm));
    const int j_c = static_cast<int>(floorf(ynorm));
    const int k_c = static_cast<int>(floorf(znorm));

    float uL_x = 0.0f, uL_y = 0.0f, uL_z = 0.0f;

    #pragma unroll
    for (int di = -1; di <= 1; ++di) {
        const int ii = i_c + di;
        if (ii < 0 || ii >= nx) continue;
        const float xc = (float(ii) + 0.5f) * dx;
        const float wx = phi_rp3<float>((xc - xL) / dx);
        if (wx == 0.0f) continue;

        #pragma unroll
        for (int dj = -1; dj <= 1; ++dj) {
            const int jj = j_c + dj;
            if (jj < 0 || jj >= ny) continue;
            const float yc = (float(jj) + 0.5f) * dx;
            const float wy = phi_rp3<float>((yc - yL) / dx);
            if (wy == 0.0f) continue;

            #pragma unroll
            for (int dkk = -1; dkk <= 1; ++dkk) {
                int kk = k_c + dkk;
                if (periodic_z) {
                    kk = wrap_z(kk, nz);
                } else {
                    if (kk < 0 || kk >= nz) continue;
                }
                const float zc = (float(k_c + dkk) + 0.5f) * dx;  // unwrapped for r
                const float wz = phi_rp3<float>((zc - zL) / dx);
                if (wz == 0.0f) continue;

                const int id = ii + jj * nx + kk * nx * ny;
                const float w = wx * wy * wz;
                uL_x += d_ux[id] * w;
                uL_y += d_uy[id] * w;
                uL_z += d_uz[id] * w;
            }
        }
    }

    d_uL_x[k] = uL_x;
    d_uL_y[k] = uL_y;
    d_uL_z[k] = uL_z;
}

// ---------------------------------------------------------------------------
// spreadForceKernel
// ---------------------------------------------------------------------------
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
    float* d_F_z_field)
{
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= n_markers) return;

    const float xL = d_xL[k];
    const float yL = d_yL[k];
    const float zL = d_zL[k];
    const float ds = d_ds[k];
    const float FLx = d_FL_x[k];
    const float FLy = d_FL_y[k];
    const float FLz = d_FL_z[k];

    const int i_c = static_cast<int>(floorf(xL / dx));
    const int j_c = static_cast<int>(floorf(yL / dx));
    const int k_c = static_cast<int>(floorf(zL / dx));

    const float ds_over_dx3 = ds / (dx * dx * dx);

    #pragma unroll
    for (int di = -1; di <= 1; ++di) {
        const int ii = i_c + di;
        if (ii < 0 || ii >= nx) continue;
        const float xc = (float(ii) + 0.5f) * dx;
        const float wx = phi_rp3<float>((xc - xL) / dx);
        if (wx == 0.0f) continue;

        #pragma unroll
        for (int dj = -1; dj <= 1; ++dj) {
            const int jj = j_c + dj;
            if (jj < 0 || jj >= ny) continue;
            const float yc = (float(jj) + 0.5f) * dx;
            const float wy = phi_rp3<float>((yc - yL) / dx);
            if (wy == 0.0f) continue;

            #pragma unroll
            for (int dkk = -1; dkk <= 1; ++dkk) {
                int kk = k_c + dkk;
                if (periodic_z) {
                    kk = wrap_z(kk, nz);
                } else {
                    if (kk < 0 || kk >= nz) continue;
                }
                const float zc = (float(k_c + dkk) + 0.5f) * dx;
                const float wz = phi_rp3<float>((zc - zL) / dx);
                if (wz == 0.0f) continue;

                const int id = ii + jj * nx + kk * nx * ny;
                const float w = wx * wy * wz * ds_over_dx3;
                atomicAdd(&d_F_x_field[id], FLx * w);
                atomicAdd(&d_F_y_field[id], FLy * w);
                atomicAdd(&d_F_z_field[id], FLz * w);
            }
        }
    }
}

} // namespace aero
} // namespace physics
} // namespace lbm
