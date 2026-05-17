/**
 * @file momentum_exchange_force.cu
 * @brief MEM force kernel for halfway-bounce-back obstacles
 */

#include "physics/aero/momentum_exchange_force.h"
#include "core/lattice_d3q19.h"
#include "core/streaming.h"
#include "utils/cuda_check.h"

namespace lbm {
namespace physics {
namespace aero {

using core::D3Q19;
using core::Streaming;
using core::ex;
using core::ey;
using core::ez;
using core::opposite;

namespace {
__device__ __forceinline__ float bflFOut(
    float f_in, float f_outgoing, float f_up_or_in, float qf_clamped)
{
    // q ≥ 0.5
    if (qf_clamped >= 0.5f) {
        const float inv_2q = 0.5f / qf_clamped;
        return inv_2q * f_in + (2.0f * qf_clamped - 1.0f) * inv_2q * f_outgoing;
    }
    // q < 0.5: linear blend with upstream same-direction population.
    return 2.0f * qf_clamped * f_in + (1.0f - 2.0f * qf_clamped) * f_up_or_in;
}
} // anonymous

namespace {

__global__ void memForceKernel(
    const float* __restrict__ f,
    const unsigned char* __restrict__ solid_mask,
    const float* __restrict__ qfrac,            // may be nullptr
    int nx, int ny, int nz,
    int periodic_x, int periodic_y, int periodic_z,
    double* Fx_acc, double* Fy_acc, double* Fz_acc)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int idy = blockIdx.y * blockDim.y + threadIdx.y;
    const int idz = blockIdx.z * blockDim.z + threadIdx.z;
    if (idx >= nx || idy >= ny || idz >= nz) return;

    const int id = idx + idy * nx + idz * nx * ny;
    const int n_cells = nx * ny * nz;

    if (solid_mask[id] != Streaming::CELL_FLUID) return;

    double fx_local = 0.0, fy_local = 0.0, fz_local = 0.0;

    for (int q = 1; q < D3Q19::Q; ++q) {
        int dst_x = idx + ex[q];
        int dst_y = idy + ey[q];
        int dst_z = idz + ez[q];

        bool out_of_domain = false;
        if (periodic_x) { if (dst_x < 0) dst_x += nx; if (dst_x >= nx) dst_x -= nx; }
        else if (dst_x < 0 || dst_x >= nx) out_of_domain = true;
        if (periodic_y) { if (dst_y < 0) dst_y += ny; if (dst_y >= ny) dst_y -= ny; }
        else if (dst_y < 0 || dst_y >= ny) out_of_domain = true;
        if (periodic_z) { if (dst_z < 0) dst_z += nz; if (dst_z >= nz) dst_z -= nz; }
        else if (dst_z < 0 || dst_z >= nz) out_of_domain = true;

        if (out_of_domain) continue;
        const int dst_id = dst_x + dst_y * nx + dst_z * nx * ny;
        if (solid_mask[dst_id] == Streaming::CELL_FLUID) continue;

        const float f_in = f[id + q * n_cells];   // about to push into wall
        float f_out;

        if (qfrac == nullptr) {
            // Halfway BB
            f_out = f_in;
        } else {
            const int q_opp = opposite[q];
            float qf = qfrac[id + q * n_cells];
            // Relaxed clamp (debug 2026-05-15): 0.1/0.95 → 1e-3/0.999.
            const float QMIN = 1e-3f, QMAX = 0.999f;
            if (qf > QMAX) qf = QMAX;
            if (qf < QMIN) qf = QMIN;

            float f_outgoing = 0.0f, f_up = 0.0f;
            if (qf >= 0.5f) {
                f_outgoing = f[id + q_opp * n_cells];
                f_out = bflFOut(f_in, f_outgoing, /*unused*/0.0f, qf);
            } else {
                int up_x = idx - ex[q];
                int up_y = idy - ey[q];
                int up_z = idz - ez[q];
                bool up_oob = false;
                if (periodic_x) { if (up_x<0) up_x+=nx; if (up_x>=nx) up_x-=nx; }
                else if (up_x<0 || up_x>=nx) up_oob = true;
                if (periodic_y) { if (up_y<0) up_y+=ny; if (up_y>=ny) up_y-=ny; }
                else if (up_y<0 || up_y>=ny) up_oob = true;
                if (periodic_z) { if (up_z<0) up_z+=nz; if (up_z>=nz) up_z-=nz; }
                else if (up_z<0 || up_z>=nz) up_oob = true;

                if (!up_oob) {
                    const int up_id = up_x + up_y * nx + up_z * nx * ny;
                    if (solid_mask[up_id] == Streaming::CELL_FLUID) {
                        f_up = f[up_id + q * n_cells];
                        f_out = bflFOut(f_in, /*unused*/0.0f, f_up, qf);
                    } else {
                        // Fallback: halfway BB
                        f_out = f_in;
                    }
                } else {
                    f_out = f_in;
                }
            }
        }

        // Generalised MEM: F_link = c_q · (f_in + f_out)
        const double sum = (double)f_in + (double)f_out;
        fx_local += (double)ex[q] * sum;
        fy_local += (double)ey[q] * sum;
        fz_local += (double)ez[q] * sum;
    }

    if (fx_local != 0.0) atomicAdd(Fx_acc, fx_local);
    if (fy_local != 0.0) atomicAdd(Fy_acc, fy_local);
    if (fz_local != 0.0) atomicAdd(Fz_acc, fz_local);
}

} // anonymous namespace

// Single-node QBB MEM kernel — mirror of fluidStreamingKernelWithSingleNodeQBB
// for force evaluation. Per fluid cell with solid neighbour:
//   1. Compute local moments rho, ux, uy, uz
//   2. For each direction q to solid: compute QBB f_out using local equilibrium
//   3. F_link = c · (f_in + f_out_new), accumulate
namespace {
__global__ void memForceQBBKernel(
    const float* __restrict__ f,
    const unsigned char* __restrict__ solid_mask,
    const float* __restrict__ qfrac,
    float omega,
    int nx, int ny, int nz,
    int periodic_x, int periodic_y, int periodic_z,
    double* Fx_acc, double* Fy_acc, double* Fz_acc)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int idy = blockIdx.y * blockDim.y + threadIdx.y;
    const int idz = blockIdx.z * blockDim.z + threadIdx.z;
    if (idx >= nx || idy >= ny || idz >= nz) return;

    const int id = idx + idy * nx + idz * nx * ny;
    const int n_cells = nx * ny * nz;

    if (solid_mask[id] != Streaming::CELL_FLUID) return;

    // Local moments
    float rho_local = 0.0f, mx = 0.0f, my = 0.0f, mz = 0.0f;
    #pragma unroll
    for (int q = 0; q < D3Q19::Q; ++q) {
        float f_q = f[id + q * n_cells];
        rho_local += f_q;
        mx += ex[q] * f_q;
        my += ey[q] * f_q;
        mz += ez[q] * f_q;
    }
    const float rho_safe = fmaxf(rho_local, 1e-12f);
    const float ux = mx / rho_safe;
    const float uy = my / rho_safe;
    const float uz = mz / rho_safe;

    const float inv_one_minus_omega = 1.0f / (1.0f - omega);

    double fx_local = 0.0, fy_local = 0.0, fz_local = 0.0;
    for (int q = 1; q < D3Q19::Q; ++q) {
        int dst_x = idx + ex[q];
        int dst_y = idy + ey[q];
        int dst_z = idz + ez[q];

        bool out_of_domain = false;
        if (periodic_x) { if (dst_x < 0) dst_x += nx; if (dst_x >= nx) dst_x -= nx; }
        else if (dst_x < 0 || dst_x >= nx) out_of_domain = true;
        if (periodic_y) { if (dst_y < 0) dst_y += ny; if (dst_y >= ny) dst_y -= ny; }
        else if (dst_y < 0 || dst_y >= ny) out_of_domain = true;
        if (periodic_z) { if (dst_z < 0) dst_z += nz; if (dst_z >= nz) dst_z -= nz; }
        else if (dst_z < 0 || dst_z >= nz) out_of_domain = true;

        if (out_of_domain) continue;
        const int dst_id = dst_x + dst_y * nx + dst_z * nx * ny;
        if (solid_mask[dst_id] == Streaming::CELL_FLUID) continue;

        const int q_opp = opposite[q];
        float qf = qfrac[id + q * n_cells];
        // Relaxed clamp (debug 2026-05-15): 0.05/0.95 → 1e-3/0.999.
        const float QMIN = 1e-3f, QMAX = 0.999f;
        if (qf > QMAX) qf = QMAX;
        if (qf < QMIN) qf = QMIN;

        const float f_in = f[id + q * n_cells];
        const float f_out = f[id + q_opp * n_cells];

        const float feq_q   = D3Q19::computeEquilibrium(q,   rho_local, ux, uy, uz);
        const float feq_opp = D3Q19::computeEquilibrium(q_opp, rho_local, ux, uy, uz);
        const float feq_sym = feq_q + feq_opp;

        const float t1 = (f_in - f_out)
                       + (f_in + f_out - omega * feq_sym) * inv_one_minus_omega;
        const float t2 = qf * (f_in + f_out) / (1.0f + qf);
        const float f_out_new = ((1.0f - qf) / (1.0f + qf)) * 0.5f * t1 + t2;

        const double sum = (double)f_in + (double)f_out_new;
        fx_local += (double)ex[q] * sum;
        fy_local += (double)ey[q] * sum;
        fz_local += (double)ez[q] * sum;
    }

    if (fx_local != 0.0) atomicAdd(Fx_acc, fx_local);
    if (fy_local != 0.0) atomicAdd(Fy_acc, fy_local);
    if (fz_local != 0.0) atomicAdd(Fz_acc, fz_local);
}
} // anonymous

// Quad-Bouzidi MEM kernel — mirror of fluidStreamingKernelWithQuadBouzidi.
namespace {
__global__ void memForceQuadBouzidiKernel(
    const float* __restrict__ f,
    const unsigned char* __restrict__ solid_mask,
    const float* __restrict__ qfrac,
    int nx, int ny, int nz,
    int periodic_x, int periodic_y, int periodic_z,
    double* Fx_acc, double* Fy_acc, double* Fz_acc)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int idy = blockIdx.y * blockDim.y + threadIdx.y;
    const int idz = blockIdx.z * blockDim.z + threadIdx.z;
    if (idx >= nx || idy >= ny || idz >= nz) return;

    const int id = idx + idy * nx + idz * nx * ny;
    const int n_cells = nx * ny * nz;

    if (solid_mask[id] != Streaming::CELL_FLUID) return;

    double fx_local = 0.0, fy_local = 0.0, fz_local = 0.0;
    for (int q = 1; q < D3Q19::Q; ++q) {
        int dst_x = idx + ex[q];
        int dst_y = idy + ey[q];
        int dst_z = idz + ez[q];

        bool out_of_domain = false;
        if (periodic_x) { if (dst_x < 0) dst_x += nx; if (dst_x >= nx) dst_x -= nx; }
        else if (dst_x < 0 || dst_x >= nx) out_of_domain = true;
        if (periodic_y) { if (dst_y < 0) dst_y += ny; if (dst_y >= ny) dst_y -= ny; }
        else if (dst_y < 0 || dst_y >= ny) out_of_domain = true;
        if (periodic_z) { if (dst_z < 0) dst_z += nz; if (dst_z >= nz) dst_z -= nz; }
        else if (dst_z < 0 || dst_z >= nz) out_of_domain = true;
        if (out_of_domain) continue;

        const int dst_id = dst_x + dst_y * nx + dst_z * nx * ny;
        if (solid_mask[dst_id] == Streaming::CELL_FLUID) continue;

        const int q_opp = opposite[q];
        float qf = qfrac[id + q * n_cells];
        // Relaxed clamp (debug 2026-05-15): 0.05/0.95 → 1e-3/0.999.
        const float QMIN = 1e-3f, QMAX = 0.999f;
        if (qf > QMAX) qf = QMAX;
        if (qf < QMIN) qf = QMIN;

        const float f_in = f[id + q * n_cells];

        // Lookup helper (same as streaming kernel)
        auto lookup_upstream = [&] (int steps, int& up_id) -> bool {
            int up_x = idx - steps * ex[q];
            int up_y = idy - steps * ey[q];
            int up_z = idz - steps * ez[q];
            if (periodic_x) { if (up_x < 0) up_x += nx; if (up_x >= nx) up_x -= nx; }
            else if (up_x < 0 || up_x >= nx) return false;
            if (periodic_y) { if (up_y < 0) up_y += ny; if (up_y >= ny) up_y -= ny; }
            else if (up_y < 0 || up_y >= ny) return false;
            if (periodic_z) { if (up_z < 0) up_z += nz; if (up_z >= nz) up_z -= nz; }
            else if (up_z < 0 || up_z >= nz) return false;
            up_id = up_x + up_y * nx + up_z * nx * ny;
            return solid_mask[up_id] == Streaming::CELL_FLUID;
        };

        float f_out_new;
        if (qf <= 0.5f) {
            int up1_id = -1, up2_id = -1;
            const bool have_up1 = lookup_upstream(1, up1_id);
            const bool have_up2 = lookup_upstream(2, up2_id);
            if (have_up1 && have_up2) {
                const float f_up1 = f[up1_id + q * n_cells];
                const float f_up2 = f[up2_id + q * n_cells];
                f_out_new =   qf * (1.0f + 2.0f * qf) * f_in
                            + (1.0f - 2.0f * qf) * (1.0f + 2.0f * qf) * f_up1
                            + (-qf * (1.0f - 2.0f * qf)) * f_up2;
            } else if (have_up1) {
                const float f_up1 = f[up1_id + q * n_cells];
                f_out_new = 2.0f * qf * f_in + (1.0f - 2.0f * qf) * f_up1;
            } else {
                f_out_new = f_in;
            }
        } else {
            int up1_id = -1;
            const bool have_up1 = lookup_upstream(1, up1_id);
            if (have_up1) {
                const float f_out = f[id + q_opp * n_cells];
                const float f_up_opp = f[up1_id + q_opp * n_cells];
                f_out_new =   (1.0f / (qf * (1.0f + 2.0f * qf))) * f_in
                            + ((2.0f * qf - 1.0f) / qf) * f_out
                            + (-(2.0f * qf - 1.0f) / (2.0f * qf + 1.0f)) * f_up_opp;
            } else {
                const float f_out = f[id + q_opp * n_cells];
                const float inv_2q = 0.5f / qf;
                f_out_new = inv_2q * f_in + (2.0f * qf - 1.0f) * inv_2q * f_out;
            }
        }

        const double sum = (double)f_in + (double)f_out_new;
        fx_local += (double)ex[q] * sum;
        fy_local += (double)ey[q] * sum;
        fz_local += (double)ez[q] * sum;
    }

    if (fx_local != 0.0) atomicAdd(Fx_acc, fx_local);
    if (fy_local != 0.0) atomicAdd(Fy_acc, fy_local);
    if (fz_local != 0.0) atomicAdd(Fz_acc, fz_local);
}
} // anonymous

void computeObstacleForceLU_QuadBouzidi(
    const float* d_f,
    const unsigned char* d_solid_mask,
    const float* d_qfrac,
    int nx, int ny, int nz,
    int periodic_x, int periodic_y, int periodic_z,
    float& Fx, float& Fy, float& Fz)
{
    double *d_Fx = nullptr, *d_Fy = nullptr, *d_Fz = nullptr;
    CUDA_CHECK(cudaMalloc(&d_Fx, sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_Fy, sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_Fz, sizeof(double)));
    const double zero = 0.0;
    CUDA_CHECK(cudaMemcpy(d_Fx, &zero, sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_Fy, &zero, sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_Fz, &zero, sizeof(double), cudaMemcpyHostToDevice));

    dim3 block(8, 8, 8);
    dim3 grid((nx + block.x - 1) / block.x,
              (ny + block.y - 1) / block.y,
              (nz + block.z - 1) / block.z);

    memForceQuadBouzidiKernel<<<grid, block>>>(
        d_f, d_solid_mask, d_qfrac, nx, ny, nz,
        periodic_x, periodic_y, periodic_z,
        d_Fx, d_Fy, d_Fz);
    CUDA_CHECK_KERNEL();
    CUDA_CHECK(cudaDeviceSynchronize());

    double h_Fx, h_Fy, h_Fz;
    CUDA_CHECK(cudaMemcpy(&h_Fx, d_Fx, sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&h_Fy, d_Fy, sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&h_Fz, d_Fz, sizeof(double), cudaMemcpyDeviceToHost));
    cudaFree(d_Fx); cudaFree(d_Fy); cudaFree(d_Fz);

    Fx = (float)h_Fx;
    Fy = (float)h_Fy;
    Fz = (float)h_Fz;
}

void computeObstacleForceLU_SingleNodeQBB(
    const float* d_f,
    const unsigned char* d_solid_mask,
    const float* d_qfrac,
    float omega,
    int nx, int ny, int nz,
    int periodic_x, int periodic_y, int periodic_z,
    float& Fx, float& Fy, float& Fz)
{
    double *d_Fx = nullptr, *d_Fy = nullptr, *d_Fz = nullptr;
    CUDA_CHECK(cudaMalloc(&d_Fx, sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_Fy, sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_Fz, sizeof(double)));
    const double zero = 0.0;
    CUDA_CHECK(cudaMemcpy(d_Fx, &zero, sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_Fy, &zero, sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_Fz, &zero, sizeof(double), cudaMemcpyHostToDevice));

    dim3 block(8, 8, 8);
    dim3 grid((nx + block.x - 1) / block.x,
              (ny + block.y - 1) / block.y,
              (nz + block.z - 1) / block.z);

    memForceQBBKernel<<<grid, block>>>(
        d_f, d_solid_mask, d_qfrac, omega, nx, ny, nz,
        periodic_x, periodic_y, periodic_z,
        d_Fx, d_Fy, d_Fz);
    CUDA_CHECK_KERNEL();
    CUDA_CHECK(cudaDeviceSynchronize());

    double h_Fx, h_Fy, h_Fz;
    CUDA_CHECK(cudaMemcpy(&h_Fx, d_Fx, sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&h_Fy, d_Fy, sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&h_Fz, d_Fz, sizeof(double), cudaMemcpyDeviceToHost));
    cudaFree(d_Fx); cudaFree(d_Fy); cudaFree(d_Fz);

    Fx = (float)h_Fx;
    Fy = (float)h_Fy;
    Fz = (float)h_Fz;
}

void computeObstacleForceLU(
    const float* d_f,
    const unsigned char* d_solid_mask,
    const float* d_qfrac,
    int nx, int ny, int nz,
    int periodic_x, int periodic_y, int periodic_z,
    float& Fx, float& Fy, float& Fz)
{
    double *d_Fx = nullptr, *d_Fy = nullptr, *d_Fz = nullptr;
    CUDA_CHECK(cudaMalloc(&d_Fx, sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_Fy, sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_Fz, sizeof(double)));
    const double zero = 0.0;
    CUDA_CHECK(cudaMemcpy(d_Fx, &zero, sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_Fy, &zero, sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_Fz, &zero, sizeof(double), cudaMemcpyHostToDevice));

    dim3 block(8, 8, 8);
    dim3 grid((nx + block.x - 1) / block.x,
              (ny + block.y - 1) / block.y,
              (nz + block.z - 1) / block.z);

    memForceKernel<<<grid, block>>>(
        d_f, d_solid_mask, d_qfrac, nx, ny, nz,
        periodic_x, periodic_y, periodic_z,
        d_Fx, d_Fy, d_Fz);
    CUDA_CHECK_KERNEL();
    CUDA_CHECK(cudaDeviceSynchronize());

    double h_Fx, h_Fy, h_Fz;
    CUDA_CHECK(cudaMemcpy(&h_Fx, d_Fx, sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&h_Fy, d_Fy, sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&h_Fz, d_Fz, sizeof(double), cudaMemcpyDeviceToHost));
    cudaFree(d_Fx); cudaFree(d_Fy); cudaFree(d_Fz);

    Fx = (float)h_Fx;
    Fy = (float)h_Fy;
    Fz = (float)h_Fz;
}

} // namespace aero
} // namespace physics
} // namespace lbm
