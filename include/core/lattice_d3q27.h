/**
 * @file lattice_d3q27.h
 * @brief D3Q27 lattice structure for 3D Lattice Boltzmann Method
 *
 * D3Q27 is the 3D 27-velocity lattice required for Cumulant LBM with full
 * Galilean correction (Geier-Schoenherr-Pasquali 2017). Adds 8 corner
 * directions to the D3Q19 face+edge stencil:
 *   - 1 rest:   (0,0,0)
 *   - 6 faces:  (±1,0,0), (0,±1,0), (0,0,±1)
 *   - 12 edges: (±1,±1,0), (±1,0,±1), (0,±1,±1)
 *   - 8 corners: (±1,±1,±1)
 *
 * Weights:
 *   w_rest   = 8/27
 *   w_face   = 2/27
 *   w_edge   = 1/54
 *   w_corner = 1/216
 *
 * Speed of sound c_s² = 1/3 (same as D3Q19).
 */

#pragma once

#include <cuda_runtime.h>
#include <cmath>

namespace lbm {
namespace core {

// Forward declaration of device constant memory arrays
// Defined in d3q27.cu. Names suffixed with "27" to avoid clash with D3Q19's
// ex/ey/ez/w/opposite which live in the same lbm::core namespace.
extern __constant__ int ex27[27];
extern __constant__ int ey27[27];
extern __constant__ int ez27[27];
extern __constant__ float w27[27];
extern __constant__ double w_double27[27];
extern __constant__ int opposite27[27];

/**
 * @brief D3Q27 lattice constants and structure
 */
class D3Q27 {
public:
    static constexpr int Q = 27;
    static constexpr float CS2 = 1.0f / 3.0f;
    static constexpr float CS = 0.57735026919f;

    /// Initialize lattice constants on device. Must be called before use.
    static void initializeDevice();

    /// Velocity magnitude squared for direction q (0..26).
    __host__ __device__ static float getVelocityMagnitudeSquared(int q);

    /// Standard 2nd-order Hermite equilibrium:
    ///   f_eq_q = w_q · ρ · [1 + 3(c·u) + 9/2·(c·u)² − 3/2·|u|²]
    __host__ __device__ static float computeEquilibrium(
        int q, float rho, float ux, float uy, float uz);

    /// Density = sum of all 27 populations.
    __host__ __device__ static float computeDensity(const float* f);

    /// Velocity from populations + density (no force correction).
    __host__ __device__ static void computeVelocity(
        const float* f, float rho, float& ux, float& uy, float& uz);

    static bool isInitialized();

    // Host-side lattice tables. Public for host-side helpers (geometry,
    // BC, post-processing).
    static const int h_ex[Q];
    static const int h_ey[Q];
    static const int h_ez[Q];
    static const int h_opposite[Q];
    static const float h_w[Q];
    static const double h_w_double[Q];

private:
    static bool initialized;
};

} // namespace core
} // namespace lbm
