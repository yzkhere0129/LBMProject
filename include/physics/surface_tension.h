/**
 * @file surface_tension.h
 * @brief Surface tension force computation using CSF model
 *
 * This file implements the Continuum Surface Force (CSF) model for computing
 * surface tension effects in VOF simulations. The CSF model converts the
 * interfacial force into a body force that can be added to the Navier-Stokes
 * equations.
 *
 * Physical model:
 * - Surface tension force: F = σ * κ * n * δ_s
 * - σ: surface tension coefficient [N/m]
 * - κ: interface curvature [1/m]
 * - n: interface normal (unit vector)
 * - δ_s: interface delta function (approximated by |∇f|)
 *
 * For VOF, the force is smoothed over the interface region:
 * - F = σ * κ * ∇f
 *
 * Reference:
 * - Brackbill, J. U., Kothe, D. B., & Zemach, C. (1992). A continuum method
 *   for modeling surface tension. Journal of Computational Physics, 100(2), 335-354.
 */

#pragma once

#include <cuda_runtime.h>

namespace lbm {
namespace physics {

/**
 * @brief Surface tension solver using CSF model
 *
 * This class computes surface tension forces for VOF-based free surface
 * simulations. The CSF model converts the sharp interface force into a
 * volumetric force distributed over the interface region.
 *
 * The force is computed as:
 * F_surface = σ * κ * ∇f
 *
 * where:
 * - σ is the surface tension coefficient
 * - κ is the interface curvature
 * - ∇f is the fill level gradient
 */
class SurfaceTension {
public:
    /**
     * @brief Constructor
     * @param nx Domain size in x-direction
     * @param ny Domain size in y-direction
     * @param nz Domain size in z-direction
     * @param surface_tension_coeff Surface tension coefficient σ [N/m]
     * @param dx Lattice spacing [m]
     */
    SurfaceTension(int nx, int ny, int nz, float surface_tension_coeff, float dx = 1.0f);

    /**
     * @brief Destructor
     */
    ~SurfaceTension();

    /**
     * @brief Compute CSF surface tension force
     * @param fill_level Device array of fill level field (0-1)
     * @param curvature Device array of interface curvature [1/m]
     * @param force_x Output: force x-component [N/m³]
     * @param force_y Output: force y-component [N/m³]
     * @param force_z Output: force z-component [N/m³]
     * @note Force is volumetric (per unit volume), ready for LBM body force
     */
    void computeCSFForce(const float* fill_level,
                         const float* curvature,
                         float* force_x,
                         float* force_y,
                         float* force_z) const;

    /**
     * @brief Compute CSF force and add to existing force field
     * @param fill_level Device array of fill level field (0-1)
     * @param curvature Device array of interface curvature [1/m]
     * @param force_x Input/Output: force x-component [N/m³]
     * @param force_y Input/Output: force y-component [N/m³]
     * @param force_z Input/Output: force z-component [N/m³]
     */
    void addCSFForce(const float* fill_level,
                     const float* curvature,
                     float* force_x,
                     float* force_y,
                     float* force_z) const;

    /**
     * @brief Set surface tension coefficient
     * @param sigma Surface tension coefficient [N/m]
     */
    void setSurfaceTension(float sigma) { sigma_ = sigma; }

    /**
     * @brief Get surface tension coefficient
     * @return Surface tension coefficient [N/m]
     */
    float getSurfaceTension() const { return sigma_; }

    /**
     * @brief Compute Laplace pressure jump for validation
     * @param curvature Mean curvature [1/m]
     * @return Pressure jump ΔP = σ * κ [Pa]
     */
    float computeLaplacePressure(float curvature) const {
        return sigma_ * curvature;
    }

    /**
     * @brief Get domain dimensions
     */
    int getNx() const { return nx_; }
    int getNy() const { return ny_; }
    int getNz() const { return nz_; }

private:
    // Domain dimensions
    int nx_, ny_, nz_;
    int num_cells_;

    // Physical parameters
    float sigma_;  ///< Surface tension coefficient [N/m]
    float dx_;     ///< Lattice spacing [m]
};

// CUDA kernels for surface tension

/**
 * @brief CUDA kernel for CSF surface tension force
 * @note F = σ * κ * ∇f
 */
__global__ void computeCSFForceKernel(
    const float* fill_level,
    const float* curvature,
    float* force_x,
    float* force_y,
    float* force_z,
    float sigma,
    float dx,
    int nx, int ny, int nz);

/**
 * @brief CUDA kernel for adding CSF force to existing force field
 */
__global__ void addCSFForceKernel(
    const float* fill_level,
    const float* curvature,
    float* force_x,
    float* force_y,
    float* force_z,
    float sigma,
    float dx,
    int nx, int ny, int nz);

} // namespace physics
} // namespace lbm
