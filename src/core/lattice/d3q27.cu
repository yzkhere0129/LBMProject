/**
 * @file d3q27.cu
 * @brief D3Q27 lattice constants and kernels
 */

#include "core/lattice_d3q27.h"
#include <stdexcept>
#include <string>

namespace lbm {
namespace core {

// Device constant memory arrays
__constant__ int ex27[27];
__constant__ int ey27[27];
__constant__ int ez27[27];
__constant__ float w27[27];
__constant__ double w_double27[27];
__constant__ int opposite27[27];

bool D3Q27::initialized = false;

// ---------------------------------------------------------------------------
// Velocity table (Q=27)
//   q=0           : rest (0,0,0)
//   q=1..6        : faces (same ordering as D3Q19 q=1..6)
//   q=7..18       : edges (same ordering as D3Q19 q=7..18)
//   q=19..26      : 8 corners (±1,±1,±1)  — NEW vs D3Q19
//
// Corner enumeration with opposite pairs:
//   q=19 (+1,+1,+1)  ↔  q=20 (-1,-1,-1)
//   q=21 (+1,+1,-1)  ↔  q=22 (-1,-1,+1)
//   q=23 (+1,-1,+1)  ↔  q=24 (-1,+1,-1)
//   q=25 (-1,+1,+1)  ↔  q=26 (+1,-1,-1)
// ---------------------------------------------------------------------------

const int D3Q27::h_ex[27] = {
    // 0  1   2  3  4  5  6   <-- rest, faces
       0, 1, -1, 0, 0, 0, 0,
    // 7  8   9 10 11 12 13 14 15 16 17 18   <-- edges (D3Q19 ordering)
       1,-1,  1,-1, 1,-1, 1,-1, 0, 0, 0, 0,
    // 19 20 21 22 23 24 25 26   <-- corners
       1,-1, 1,-1, 1,-1,-1, 1
};

const int D3Q27::h_ey[27] = {
       0, 0,  0, 1,-1, 0, 0,
       1, 1, -1,-1, 0, 0, 0, 0, 1,-1, 1,-1,
       1,-1, 1,-1,-1, 1, 1,-1
};

const int D3Q27::h_ez[27] = {
       0, 0,  0, 0, 0, 1,-1,
       0, 0,  0, 0, 1, 1,-1,-1, 1, 1,-1,-1,
       1,-1,-1, 1, 1,-1, 1,-1
};

// Weights:  w_rest = 8/27, w_face = 2/27, w_edge = 1/54, w_corner = 1/216
// Sum check: 8/27 + 6·2/27 + 12·1/54 + 8·1/216
//          = 8/27 + 12/27 + 6/27 + 1/27 = 27/27 = 1  ✓
// Second-moment check: c_s² = 1/3  (same as D3Q19)
const float D3Q27::h_w[27] = {
    8.0f/27.0f,
    2.0f/27.0f, 2.0f/27.0f, 2.0f/27.0f, 2.0f/27.0f, 2.0f/27.0f, 2.0f/27.0f,
    1.0f/54.0f, 1.0f/54.0f, 1.0f/54.0f, 1.0f/54.0f,
    1.0f/54.0f, 1.0f/54.0f, 1.0f/54.0f, 1.0f/54.0f,
    1.0f/54.0f, 1.0f/54.0f, 1.0f/54.0f, 1.0f/54.0f,
    1.0f/216.0f, 1.0f/216.0f, 1.0f/216.0f, 1.0f/216.0f,
    1.0f/216.0f, 1.0f/216.0f, 1.0f/216.0f, 1.0f/216.0f
};

const double D3Q27::h_w_double[27] = {
    8.0/27.0,
    2.0/27.0, 2.0/27.0, 2.0/27.0, 2.0/27.0, 2.0/27.0, 2.0/27.0,
    1.0/54.0, 1.0/54.0, 1.0/54.0, 1.0/54.0,
    1.0/54.0, 1.0/54.0, 1.0/54.0, 1.0/54.0,
    1.0/54.0, 1.0/54.0, 1.0/54.0, 1.0/54.0,
    1.0/216.0, 1.0/216.0, 1.0/216.0, 1.0/216.0,
    1.0/216.0, 1.0/216.0, 1.0/216.0, 1.0/216.0
};

// Opposite indices: bounce-back pairing.
const int D3Q27::h_opposite[27] = {
    0,                       // 0 rest
    2,  1,  4,  3,  6,  5,   // 1↔2, 3↔4, 5↔6 (faces)
    10, 9,  8,  7,           // 7↔10, 8↔9   (xy edges)
    14, 13, 12, 11,          // 11↔14, 12↔13 (xz edges)
    18, 17, 16, 15,          // 15↔18, 16↔17 (yz edges)
    20, 19, 22, 21, 24, 23, 26, 25  // 19↔20, 21↔22, 23↔24, 25↔26 (corners)
};

void D3Q27::initializeDevice() {
    if (initialized) return;

    auto check = [](cudaError_t err, const char* op) {
        if (err != cudaSuccess) {
            throw std::runtime_error(std::string("D3Q27 init - ") + op + ": "
                                     + cudaGetErrorString(err));
        }
    };

    check(cudaMemcpyToSymbol(ex27, h_ex, Q * sizeof(int)), "ex27");
    check(cudaMemcpyToSymbol(ey27, h_ey, Q * sizeof(int)), "ey27");
    check(cudaMemcpyToSymbol(ez27, h_ez, Q * sizeof(int)), "ez27");
    check(cudaMemcpyToSymbol(w27, h_w, Q * sizeof(float)), "w27");
    check(cudaMemcpyToSymbol(w_double27, h_w_double, Q * sizeof(double)), "w_double27");
    check(cudaMemcpyToSymbol(opposite27, h_opposite, Q * sizeof(int)), "opposite27");

    initialized = true;
}

__host__ __device__ float D3Q27::getVelocityMagnitudeSquared(int q) {
#ifdef __CUDA_ARCH__
    return ::lbm::core::ex27[q] * ::lbm::core::ex27[q]
         + ::lbm::core::ey27[q] * ::lbm::core::ey27[q]
         + ::lbm::core::ez27[q] * ::lbm::core::ez27[q];
#else
    return h_ex[q] * h_ex[q] + h_ey[q] * h_ey[q] + h_ez[q] * h_ez[q];
#endif
}

__host__ __device__ float D3Q27::computeEquilibrium(
    int q, float rho, float ux, float uy, float uz)
{
#ifdef __CUDA_ARCH__
    const float cx = ::lbm::core::ex27[q];
    const float cy = ::lbm::core::ey27[q];
    const float cz = ::lbm::core::ez27[q];
    const float wq = ::lbm::core::w27[q];
#else
    const float cx = h_ex[q];
    const float cy = h_ey[q];
    const float cz = h_ez[q];
    const float wq = h_w[q];
#endif
    // Compressible 2nd-order Hermite equilibrium:
    //   f_eq = w · ρ · (1 + 3(c·u) + 9/2 (c·u)² − 3/2 |u|²)
    const float cu = cx * ux + cy * uy + cz * uz;
    const float u2 = ux * ux + uy * uy + uz * uz;
    return wq * rho * (1.0f + 3.0f * cu + 4.5f * cu * cu - 1.5f * u2);
}

__host__ __device__ float D3Q27::computeDensity(const float* f) {
    float r = 0.0f;
    #pragma unroll
    for (int q = 0; q < Q; ++q) r += f[q];
    return r;
}

__host__ __device__ void D3Q27::computeVelocity(
    const float* f, float rho, float& ux, float& uy, float& uz)
{
    float mx = 0.0f, my = 0.0f, mz = 0.0f;
#ifdef __CUDA_ARCH__
    #pragma unroll
    for (int q = 0; q < Q; ++q) {
        mx += ::lbm::core::ex27[q] * f[q];
        my += ::lbm::core::ey27[q] * f[q];
        mz += ::lbm::core::ez27[q] * f[q];
    }
#else
    for (int q = 0; q < Q; ++q) {
        mx += h_ex[q] * f[q];
        my += h_ey[q] * f[q];
        mz += h_ez[q] * f[q];
    }
#endif
    const float inv_rho = 1.0f / rho;
    ux = mx * inv_rho;
    uy = my * inv_rho;
    uz = mz * inv_rho;
}

bool D3Q27::isInitialized() { return initialized; }

} // namespace core
} // namespace lbm
