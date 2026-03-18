/**
 * @file lattice_d3q19.h
 * @brief D3Q19 lattice structure for 3D Lattice Boltzmann Method
 *
 * This file defines the D3Q19 (3 dimensions, 19 discrete velocities) lattice
 * structure used in the Lattice Boltzmann Method for simulating fluid flow
 * in three dimensions.
 */

#pragma once

#include <cuda_runtime.h>
#include <cmath>

namespace lbm {
namespace core {

// Forward declaration of device constant memory arrays
// These are defined in d3q19.cu
extern __constant__ int ex[19];
extern __constant__ int ey[19];
extern __constant__ int ez[19];
extern __constant__ float w[19];
extern __constant__ double w_double[19];  // Double-precision weights for high-accuracy TRT
extern __constant__ int opposite[19];

/**
 * @brief D3Q19 lattice constants and structure
 *
 * The D3Q19 model uses 19 discrete velocity vectors:
 * - 1 rest particle (0,0,0)
 * - 6 face-connected neighbors (±1,0,0), (0,±1,0), (0,0,±1)
 * - 12 edge-connected neighbors (±1,±1,0), (±1,0,±1), (0,±1,±1)
 */
class D3Q19 {
public:
    static constexpr int Q = 19;  ///< Number of discrete velocities
    static constexpr float CS2 = 1.0f / 3.0f;  ///< Speed of sound squared (lattice units)
    static constexpr float CS = 0.57735026919f;  ///< Speed of sound (sqrt(1/3))

    // Velocity directions (stored in device constant memory for fast access)
    // Note: These are declared outside the class in the .cu file
    // as __constant__ memory cannot be class members

    /**
     * @brief Initialize lattice constants on device
     *
     * This function must be called before using the lattice on GPU
     */
    static void initializeDevice();

    /**
     * @brief Get velocity magnitude squared for direction q
     * @param q Direction index (0-18)
     * @return Velocity magnitude squared
     */
    __host__ __device__ static float getVelocityMagnitudeSquared(int q);

    /**
     * @brief Compute equilibrium distribution function
     * @param q Direction index
     * @param rho Density
     * @param ux Velocity x-component
     * @param uy Velocity y-component
     * @param uz Velocity z-component
     * @return Equilibrium distribution value
     */
    __host__ __device__ static float computeEquilibrium(
        int q, float rho, float ux, float uy, float uz);

    /**
     * @brief Compute density from distribution functions
     * @param f Distribution function array
     * @return Density
     */
    __host__ __device__ static float computeDensity(const float* f);

    /**
     * @brief Compute velocity from distribution functions
     * @param f Distribution function array
     * @param rho Density (pre-computed)
     * @param ux Output velocity x-component
     * @param uy Output velocity y-component
     * @param uz Output velocity z-component
     */
    __host__ __device__ static void computeVelocity(
        const float* f, float rho, float& ux, float& uy, float& uz);

    /**
     * @brief Get neighbor index for streaming
     * @param x Current x position
     * @param y Current y position
     * @param z Current z position
     * @param q Direction index
     * @param nx Domain size x
     * @param ny Domain size y
     * @param nz Domain size z
     * @return Neighbor cell index
     */
    __host__ __device__ static int getNeighborIndex(
        int x, int y, int z, int q, int nx, int ny, int nz);

    /**
     * @brief Check if lattice is properly initialized
     * @return true if initialized, false otherwise
     */
    static bool isInitialized();

private:
    static bool initialized;

    // Host-side copies for initialization
    static const int h_ex[Q];
    static const int h_ey[Q];
    static const int h_ez[Q];
    static const float h_w[Q];
    static const double h_w_double[Q];  // Double-precision weights for TRT
    static const int h_opposite[Q];
};

} // namespace core
} // namespace lbm