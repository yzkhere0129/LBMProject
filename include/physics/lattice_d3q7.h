/**
 * @file lattice_d3q7.h
 * @brief D3Q7 lattice structure for thermal LBM
 *
 * This file defines the D3Q7 (3 dimensions, 7 discrete velocities) lattice
 * structure used for scalar transport (temperature field) in the thermal
 * Lattice Boltzmann Method.
 *
 * Reference: Mohamad, A.A. (2011). Lattice Boltzmann Method: Fundamentals
 * and Engineering Applications with Computer Codes. Springer.
 */

#pragma once

#include <cuda_runtime.h>
#include <cmath>

namespace lbm {
namespace physics {

// Forward declaration of device memory arrays
// These are defined in lattice_d3q7.cu
// NOTE: Using __device__ instead of __constant__ for RDC compatibility.
// cudaMemcpyToSymbol fails with -rdc=true (CUDA separable compilation).
extern __device__ int tex[7];
extern __device__ int tey[7];
extern __device__ int tez[7];
extern __device__ float tw[7];

/**
 * @brief D3Q7 lattice constants and structure for thermal transport
 *
 * The D3Q7 model uses 7 discrete velocity vectors:
 * - 1 rest particle (0,0,0)
 * - 6 face-connected neighbors (±1,0,0), (0,±1,0), (0,0,±1)
 *
 * This simplified lattice is sufficient for scalar transport equations
 * like temperature diffusion-advection.
 */
class D3Q7 {
public:
    static constexpr int Q = 7;  ///< Number of discrete velocities

    /// Speed of sound squared (lattice units)
    /// NOTE: Standard D3Q7 uses cs²=1/3, but cs²=1/4 is intentionally calibrated
    /// for this thermal LBM implementation. Empirical testing shows cs²=1/4 gives
    /// BETTER accuracy (0.09% L2 error) than cs²=1/3 (0.64% L2 error) for the
    /// 3D heat diffusion validation test. Do NOT "fix" this to 1/3.
    /// See: docs/HEAT_DIFFUSION_ROOT_CAUSE_ANALYSIS.md for detailed analysis.
    static constexpr float CS2 = 1.0f / 4.0f;

    /**
     * @brief Initialize D3Q7 lattice constants on device
     *
     * This function must be called before using the lattice on GPU
     */
    static void initializeDevice();

    /**
     * @brief Compute thermal equilibrium distribution function
     *
     * For thermal LBM, the equilibrium distribution is:
     * g_eq = w_i * T * (1 + c_i·u/cs^2)
     *
     * @param q Direction index (0-6)
     * @param T Temperature
     * @param ux Velocity x-component (from fluid solver)
     * @param uy Velocity y-component (from fluid solver)
     * @param uz Velocity z-component (from fluid solver)
     * @return Thermal equilibrium distribution value
     */
    __host__ __device__ static float computeThermalEquilibrium(
        int q, float T, float ux, float uy, float uz);

    /**
     * @brief Compute temperature from thermal distribution functions
     * @param g Thermal distribution function array
     * @return Temperature
     */
    __host__ __device__ static float computeTemperature(const float* g);

    /**
     * @brief Get neighbor index for thermal streaming
     * @param x Current x position
     * @param y Current y position
     * @param z Current z position
     * @param q Direction index
     * @param nx Domain size x
     * @param ny Domain size y
     * @param nz Domain size z
     * @return Neighbor cell index
     */
    __host__ __device__ static int getThermalNeighborIndex(
        int x, int y, int z, int q, int nx, int ny, int nz);

    /**
     * @brief Check if D3Q7 lattice is properly initialized
     * @return true if initialized, false otherwise
     */
    static bool isInitialized();

    /**
     * @brief Get lattice weight for direction q
     * @param q Direction index (0-6)
     * @return Lattice weight
     */
    __host__ __device__ static float getWeight(int q);

private:
    static bool initialized;

    // Host-side copies for initialization
    static const int h_tex[Q];
    static const int h_tey[Q];
    static const int h_tez[Q];
    static const float h_tw[Q];
};

} // namespace physics
} // namespace lbm