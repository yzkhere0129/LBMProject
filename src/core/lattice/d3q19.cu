/**
 * @file d3q19.cu
 * @brief Implementation of D3Q19 lattice structure
 */

#include "core/lattice_d3q19.h"
#include <cuda.h>
#include <iostream>
#include <stdexcept>
#include <string>
#include "utils/cuda_check.h"

namespace lbm {
namespace core {

// Device constant memory definitions (global, not class members)
__constant__ int ex[D3Q19::Q];
__constant__ int ey[D3Q19::Q];
__constant__ int ez[D3Q19::Q];
__constant__ float w[D3Q19::Q];
__constant__ double w_double[D3Q19::Q];  // Double-precision weights for high-accuracy TRT
__constant__ int opposite[D3Q19::Q];

// Static member initialization
bool D3Q19::initialized = false;

// Host-side lattice constants
// Velocity directions: rest, faces (6), edges (12)
// Standard D3Q19 ordering matching test reference
const int D3Q19::h_ex[19] = {
    0, 1, -1, 0, 0, 0, 0,
    1, -1, 1, -1, 1, -1, 1, -1, 0, 0, 0, 0
};

const int D3Q19::h_ey[19] = {
    0, 0, 0, 1, -1, 0, 0,
    1, 1, -1, -1, 0, 0, 0, 0, 1, -1, 1, -1
};

const int D3Q19::h_ez[19] = {
    0, 0, 0, 0, 0, 1, -1,
    0, 0, 0, 0, 1, 1, -1, -1, 1, 1, -1, -1
};

// Weights for equilibrium distribution (float - for BGK compatibility)
const float D3Q19::h_w[19] = {
    1.0f/3.0f,     // rest (w0)
    1.0f/18.0f, 1.0f/18.0f, 1.0f/18.0f, 1.0f/18.0f, 1.0f/18.0f, 1.0f/18.0f,  // faces (w1)
    1.0f/36.0f, 1.0f/36.0f, 1.0f/36.0f, 1.0f/36.0f, 1.0f/36.0f, 1.0f/36.0f,  // edges (w2)
    1.0f/36.0f, 1.0f/36.0f, 1.0f/36.0f, 1.0f/36.0f, 1.0f/36.0f, 1.0f/36.0f
};

// Double-precision weights for high-accuracy TRT kernels
// These provide ~15 decimal digits vs 7 for float, critical for long-time integrations
const double D3Q19::h_w_double[19] = {
    1.0/3.0,     // rest (w0) - EXACT: 0.333333333333333315...
    1.0/18.0, 1.0/18.0, 1.0/18.0, 1.0/18.0, 1.0/18.0, 1.0/18.0,  // faces (w1) - EXACT: 0.055555555555555552...
    1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0,  // edges (w2) - EXACT: 0.027777777777777776...
    1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0
};

// Opposite direction indices (for bounce-back)
const int D3Q19::h_opposite[19] = {
    0,   // 0: rest (0,0,0) -> rest (0,0,0)
    2,   // 1: (+1,0,0) -> (-1,0,0)
    1,   // 2: (-1,0,0) -> (+1,0,0)
    4,   // 3: (0,+1,0) -> (0,-1,0)
    3,   // 4: (0,-1,0) -> (0,+1,0)
    6,   // 5: (0,0,+1) -> (0,0,-1)
    5,   // 6: (0,0,-1) -> (0,0,+1)
    10,  // 7: (+1,+1,0) -> (-1,-1,0)
    9,   // 8: (-1,+1,0) -> (+1,-1,0)
    8,   // 9: (+1,-1,0) -> (-1,+1,0)
    7,   // 10: (-1,-1,0) -> (+1,+1,0)
    14,  // 11: (+1,0,+1) -> (-1,0,-1)
    13,  // 12: (-1,0,+1) -> (+1,0,-1)
    12,  // 13: (+1,0,-1) -> (-1,0,+1)
    11,  // 14: (-1,0,-1) -> (+1,0,+1)
    18,  // 15: (0,+1,+1) -> (0,-1,-1)
    17,  // 16: (0,-1,+1) -> (0,+1,-1)
    16,  // 17: (0,+1,-1) -> (0,-1,+1)
    15   // 18: (0,-1,-1) -> (0,+1,+1)
};

void D3Q19::initializeDevice() {
    if (initialized) return;

    cudaError_t err;

    // Helper lambda to check CUDA errors and throw on failure
    auto checkCudaError = [](cudaError_t err, const char* operation) {
        if (err != cudaSuccess) {
            throw std::runtime_error(
                std::string("D3Q19 initialization failed - ") + operation + ": " +
                cudaGetErrorString(err)
            );
        }
    };

    // Copy lattice constants to device constant memory
    err = cudaMemcpyToSymbol(ex, h_ex, Q * sizeof(int));
    checkCudaError(err, "Failed to copy ex to device");

    err = cudaMemcpyToSymbol(ey, h_ey, Q * sizeof(int));
    checkCudaError(err, "Failed to copy ey to device");

    err = cudaMemcpyToSymbol(ez, h_ez, Q * sizeof(int));
    checkCudaError(err, "Failed to copy ez to device");

    err = cudaMemcpyToSymbol(w, h_w, Q * sizeof(float));
    checkCudaError(err, "Failed to copy w to device");

    err = cudaMemcpyToSymbol(w_double, h_w_double, Q * sizeof(double));
    checkCudaError(err, "Failed to copy w_double to device");

    err = cudaMemcpyToSymbol(opposite, h_opposite, Q * sizeof(int));
    checkCudaError(err, "Failed to copy opposite to device");

    initialized = true;
}

__host__ __device__ float D3Q19::getVelocityMagnitudeSquared(int q) {
#ifdef __CUDA_ARCH__
    return ::lbm::core::ex[q]*::lbm::core::ex[q] +
           ::lbm::core::ey[q]*::lbm::core::ey[q] +
           ::lbm::core::ez[q]*::lbm::core::ez[q];
#else
    return h_ex[q]*h_ex[q] + h_ey[q]*h_ey[q] + h_ez[q]*h_ez[q];
#endif
}

__host__ __device__ float D3Q19::computeEquilibrium(
    int q, float rho, float ux, float uy, float uz) {

#ifdef __CUDA_ARCH__
    // Device code - use constant memory
    // Compressible form: f_eq = w * rho * (1 + 3*(c·u) + 4.5*(c·u)² - 1.5*u²)
    float eu = ::lbm::core::ex[q]*ux + ::lbm::core::ey[q]*uy + ::lbm::core::ez[q]*uz;
    float u2 = ux*ux + uy*uy + uz*uz;
    return ::lbm::core::w[q] * rho * (1.0f + 3.0f*eu + 4.5f*eu*eu - 1.5f*u2);
#else
    // Host code - use host arrays
    float eu = h_ex[q]*ux + h_ey[q]*uy + h_ez[q]*uz;
    float u2 = ux*ux + uy*uy + uz*uz;
    return h_w[q] * rho * (1.0f + 3.0f*eu + 4.5f*eu*eu - 1.5f*u2);
#endif
}

// Double-precision equilibrium computation for high-accuracy kernels
__device__ __forceinline__ double computeEquilibriumDouble(
    int q, double rho, double ux, double uy, double uz) {
    // Device-only function - use constant memory arrays with explicit casts
    double eu = (double)::lbm::core::ex[q] * ux +
                (double)::lbm::core::ey[q] * uy +
                (double)::lbm::core::ez[q] * uz;
    double u2 = ux*ux + uy*uy + uz*uz;
    return (double)::lbm::core::w[q] * rho * (1.0 + 3.0*eu + 4.5*eu*eu - 1.5*u2);
}

__host__ __device__ float D3Q19::computeDensity(const float* f) {
    float rho = 0.0f;
    #pragma unroll
    for (int q = 0; q < Q; ++q) {
        rho += f[q];
    }
    return rho;
}

__host__ __device__ void D3Q19::computeVelocity(
    const float* f, float rho, float& ux, float& uy, float& uz) {

    ux = 0.0f;
    uy = 0.0f;
    uz = 0.0f;

#ifdef __CUDA_ARCH__
    // Device code
    #pragma unroll
    for (int q = 0; q < Q; ++q) {
        ux += f[q] * ::lbm::core::ex[q];
        uy += f[q] * ::lbm::core::ey[q];
        uz += f[q] * ::lbm::core::ez[q];
    }
#else
    // Host code
    for (int q = 0; q < Q; ++q) {
        ux += f[q] * h_ex[q];
        uy += f[q] * h_ey[q];
        uz += f[q] * h_ez[q];
    }
#endif

    // Protect against division by zero
    // For LBM, density should never be zero in physical regions
    // If rho is near-zero, set velocity to zero instead of dividing
    constexpr float RHO_EPSILON = 1e-8f;
    if (rho > RHO_EPSILON) {
        float inv_rho = 1.0f / rho;
        ux *= inv_rho;
        uy *= inv_rho;
        uz *= inv_rho;
    }
    // else: velocities remain zero (already initialized)
}

__host__ __device__ int D3Q19::getNeighborIndex(
    int x, int y, int z, int q, int nx, int ny, int nz) {

#ifdef __CUDA_ARCH__
    // Device code - find neighbor in direction q with periodic boundary conditions
    int x_dst = (x + ::lbm::core::ex[q] + nx) % nx;
    int y_dst = (y + ::lbm::core::ey[q] + ny) % ny;
    int z_dst = (z + ::lbm::core::ez[q] + nz) % nz;
#else
    // Host code
    int x_dst = (x + h_ex[q] + nx) % nx;
    int y_dst = (y + h_ey[q] + ny) % ny;
    int z_dst = (z + h_ez[q] + nz) % nz;
#endif

    return x_dst + y_dst * nx + z_dst * nx * ny;
}

bool D3Q19::isInitialized() {
    return initialized;
}

} // namespace core
} // namespace lbm