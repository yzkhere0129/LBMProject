/**
 * @file boundary_conditions.h
 * @brief Boundary condition implementations for LBM
 *
 * Provides various boundary conditions including bounce-back, velocity,
 * pressure, and periodic boundaries.
 */

#pragma once

#include <cuda_runtime.h>
#include "core/lattice_d3q19.h"

namespace lbm {
namespace core {

/**
 * @brief Boundary condition types
 */
enum class BoundaryType {
    NONE,           ///< No boundary (fluid node)
    BOUNCE_BACK,    ///< No-slip wall (bounce-back)
    VELOCITY,       ///< Prescribed velocity
    PRESSURE,       ///< Prescribed pressure
    PERIODIC,       ///< Periodic boundary
    SYMMETRY,       ///< Symmetry/free-slip
    OUTFLOW         ///< Outflow/zero-gradient
};

/**
 * @brief Boundary condition implementations
 */
class BoundaryConditions {
public:
    /**
     * @brief Apply bounce-back boundary condition
     *
     * Implements no-slip wall by reversing distribution functions
     *
     * @param f Distribution functions at boundary node
     * @param q Direction index
     * @return Modified distribution function
     */
    __host__ __device__ static float bounceBack(const float* f, int q);

    /**
     * @brief Apply bounce-back to a full node
     *
     * @param f_in Input distributions
     * @param f_out Output distributions
     */
    __host__ __device__ static void bounceBackNode(const float* f_in, float* f_out);

    /**
     * @brief Apply velocity boundary condition (Zou-He)
     *
     * @param f Distribution functions
     * @param rho Density at boundary
     * @param u_wall Wall velocity components
     * @param normal Normal direction (0=x, 1=y, 2=z)
     * @param sign Normal sign (+1 or -1)
     */
    __device__ static void velocityBoundaryZouHe(
        float* f, float rho,
        float ux_wall, float uy_wall, float uz_wall,
        int normal, int sign);

    /**
     * @brief Apply pressure boundary condition (Zou-He)
     *
     * @param f Distribution functions
     * @param p_boundary Prescribed pressure
     * @param normal Normal direction (0=x, 1=y, 2=z)
     * @param sign Normal sign (+1 or -1)
     */
    __device__ static void pressureBoundaryZouHe(
        float* f, float p_boundary,
        int normal, int sign);

    /**
     * @brief Check if a direction points into the domain
     *
     * @param q Direction index
     * @param boundary_type Type of boundary
     * @return true if pointing inward
     */
    __host__ __device__ static bool isIncomingDirection(int q, unsigned int boundary_type);

    /**
     * @brief Apply halfway bounce-back for moving walls
     *
     * @param f Distribution functions
     * @param q Direction index
     * @param u_wall Wall velocity
     * @return Modified distribution
     */
    __device__ static float movingWallBounceBack(
        const float* f, int q,
        float ux_wall, float uy_wall, float uz_wall);
};

/**
 * @brief Boundary node information structure
 */
struct BoundaryNode {
    int x, y, z;              ///< Position
    BoundaryType type;        ///< Boundary type
    float ux, uy, uz;        ///< Velocity (for velocity BC)
    float pressure;          ///< Pressure (for pressure BC)
    unsigned int directions; ///< Boundary direction flags
};

/**
 * @brief CUDA kernel for applying bounce-back boundaries
 *
 * @param f Distribution functions
 * @param boundary_nodes Array of boundary nodes
 * @param n_boundary Number of boundary nodes
 * @param nx Domain size x
 * @param ny Domain size y
 * @param nz Domain size z
 */
__global__ void applyBounceBackKernel(
    float* f,
    const BoundaryNode* boundary_nodes,
    int n_boundary,
    int nx, int ny, int nz);

/**
 * @brief CUDA kernel for applying velocity boundaries
 *
 * @param f Distribution functions
 * @param rho Density field
 * @param boundary_nodes Array of boundary nodes
 * @param n_boundary Number of boundary nodes
 * @param nx Domain size x
 * @param ny Domain size y
 * @param nz Domain size z
 */
__global__ void applyVelocityBoundaryKernel(
    float* f,
    const float* rho,
    const BoundaryNode* boundary_nodes,
    int n_boundary,
    int nx, int ny, int nz);

/**
 * @brief CUDA kernel for applying all boundary conditions
 *
 * @param f Distribution functions
 * @param rho Density field
 * @param boundary_nodes Array of boundary nodes
 * @param n_boundary Number of boundary nodes
 * @param nx Domain size x
 * @param ny Domain size y
 * @param nz Domain size z
 */
__global__ void applyBoundaryConditionsKernel(
    float* f,
    const float* rho,
    const BoundaryNode* boundary_nodes,
    int n_boundary,
    int nx, int ny, int nz);

} // namespace core
} // namespace lbm