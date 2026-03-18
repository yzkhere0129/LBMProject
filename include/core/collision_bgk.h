/**
 * @file collision_bgk.h
 * @brief BGK (Bhatnagar-Gross-Krook) collision operator for LBM
 *
 * The BGK collision operator is the simplest and most widely used collision
 * model in LBM. It uses a single relaxation time to relax the distribution
 * functions towards their equilibrium values.
 *
 * All methods are inlined here; no separate .cu is needed.
 */

#pragma once

#include <cuda_runtime.h>
#include "core/lattice_d3q19.h"

namespace lbm {
namespace core {

/**
 * @brief BGK collision operator implementation
 *
 * The BGK collision operator follows the equation:
 * f_i^*(x,t) = f_i(x,t) - omega * (f_i(x,t) - f_i^eq(x,t))
 * where omega = dt/tau is the relaxation parameter
 */
class BGKCollision {
public:
    __host__ __device__ static float computeOmega(float nu, float dt = 1.0f) {
        float tau = nu / D3Q19::CS2 + 0.5f * dt;
        return dt / tau;
    }

    __host__ __device__ static float computeViscosity(float omega, float dt = 1.0f) {
        float tau = dt / omega;
        return D3Q19::CS2 * (tau - 0.5f * dt);
    }

    __host__ __device__ static float computeTau(float omega, float dt = 1.0f) {
        return dt / omega;
    }

    __host__ __device__ static float collide(float f, float feq, float omega) {
        return f + omega * (feq - f);
    }

    __host__ __device__ static void collideNode(
        const float* f_in, float* f_out,
        float rho, float ux, float uy, float uz, float omega)
    {
        #pragma unroll
        for (int q = 0; q < D3Q19::Q; ++q) {
            float feq = D3Q19::computeEquilibrium(q, rho, ux, uy, uz);
            f_out[q] = collide(f_in[q], feq, omega);
        }
    }

    __host__ __device__ static bool isStable(float omega) {
        return (omega > 0.5f && omega < 1.9f);
    }

    __host__ __device__ static float maxStableVelocity(float /*omega*/) {
        const float MA_MAX = 0.1f;
        return MA_MAX * D3Q19::CS;
    }
};

/**
 * @brief CUDA kernel for BGK collision on a 3D domain
 */
__global__ inline void bgkCollisionKernel(
    const float* __restrict__ f_src,
    float* __restrict__ f_dst,
    float* __restrict__ rho,
    float* __restrict__ ux,
    float* __restrict__ uy,
    float* __restrict__ uz,
    float omega,
    int nx, int ny, int nz)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    int idz = blockIdx.z * blockDim.z + threadIdx.z;

    if (idx >= nx || idy >= ny || idz >= nz) return;

    int id = idx + idy * nx + idz * nx * ny;
    int n_cells = nx * ny * nz;

    float f[D3Q19::Q];
    #pragma unroll
    for (int q = 0; q < D3Q19::Q; ++q) {
        f[q] = f_src[id + q * n_cells];
    }

    float m_rho = D3Q19::computeDensity(f);
    float m_ux, m_uy, m_uz;
    D3Q19::computeVelocity(f, m_rho, m_ux, m_uy, m_uz);

    rho[id] = m_rho;
    ux[id]  = m_ux;
    uy[id]  = m_uy;
    uz[id]  = m_uz;

    #pragma unroll
    for (int q = 0; q < D3Q19::Q; ++q) {
        float feq = D3Q19::computeEquilibrium(q, m_rho, m_ux, m_uy, m_uz);
        f_dst[id + q * n_cells] = BGKCollision::collide(f[q], feq, omega);
    }
}

/**
 * @brief CUDA kernel for fused BGK collision and streaming
 */
__global__ inline void bgkCollisionStreamKernel(
    const float* __restrict__ f_src,
    float* __restrict__ f_dst,
    float* __restrict__ rho,
    float* __restrict__ ux,
    float* __restrict__ uy,
    float* __restrict__ uz,
    float omega,
    int nx, int ny, int nz)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    int idz = blockIdx.z * blockDim.z + threadIdx.z;

    if (idx >= nx || idy >= ny || idz >= nz) return;

    int id = idx + idy * nx + idz * nx * ny;
    int n_cells = nx * ny * nz;

    float f[D3Q19::Q];
    #pragma unroll
    for (int q = 0; q < D3Q19::Q; ++q) {
        f[q] = f_src[id + q * n_cells];
    }

    float m_rho = D3Q19::computeDensity(f);
    float m_ux, m_uy, m_uz;
    D3Q19::computeVelocity(f, m_rho, m_ux, m_uy, m_uz);

    rho[id] = m_rho;
    ux[id]  = m_ux;
    uy[id]  = m_uy;
    uz[id]  = m_uz;

    #pragma unroll
    for (int q = 0; q < D3Q19::Q; ++q) {
        float feq = D3Q19::computeEquilibrium(q, m_rho, m_ux, m_uy, m_uz);
        float f_post = BGKCollision::collide(f[q], feq, omega);
        int neighbor_id = D3Q19::getNeighborIndex(idx, idy, idz, q, nx, ny, nz);
        f_dst[neighbor_id + q * n_cells] = f_post;
    }
}

} // namespace core
} // namespace lbm
