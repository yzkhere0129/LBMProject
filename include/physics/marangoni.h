/**
 * @file marangoni.h
 * @brief Marangoni effect (thermocapillary flow) computation
 *
 * This file implements the Marangoni effect, which is the mass transfer along
 * an interface due to surface tension gradients. In laser melting, temperature
 * gradients at the melt pool surface create surface tension gradients that
 * drive strong convective flows.
 *
 * Physical model:
 * - Surface tension varies with temperature: σ(T) = σ₀ + dσ/dT * (T - T₀)
 * - Marangoni stress (tangential): τ = dσ/dT * ∇_s T
 * - Body force equivalent: F = τ * δ_s / h
 *
 * For VOF implementation:
 * - F_Marangoni = (dσ/dT) * ∇_s T * |∇f| / h
 * - ∇_s T = (I - n⊗n) · ∇T (tangential temperature gradient)
 * - h is interface thickness (typically 1-2 grid cells)
 *
 * References:
 * - Hu, H., & Argyropoulos, S. A. (1996). Mathematical modelling of
 *   solidification and melting: a review. Modelling and Simulation in
 *   Materials Science and Engineering, 4(4), 371.
 * - Khairallah, S. A., et al. (2016). Laser powder-bed fusion additive
 *   manufacturing: Physics of complex melt flow and formation mechanisms
 *   of pores, spatter, and denudation zones. Acta Materialia, 108, 36-45.
 */

#pragma once

#include <cuda_runtime.h>

namespace lbm {
namespace physics {

/**
 * @brief Marangoni effect solver for thermocapillary flows
 *
 * This class computes the Marangoni force arising from temperature-dependent
 * surface tension. The force drives fluid flow tangent to the interface from
 * hot to cold regions (for most metals where dσ/dT < 0).
 *
 * The force is computed as:
 * F_Marangoni = (dσ/dT) * ∇_s T * |∇f| / h
 *
 * where:
 * - dσ/dT is the temperature coefficient of surface tension [N/(m·K)]
 * - ∇_s T is the surface tangential temperature gradient
 * - |∇f| approximates the interface delta function
 * - h is the interface thickness
 */
class MarangoniEffect {
public:
    /**
     * @brief Constructor
     * @param nx Domain size in x-direction
     * @param ny Domain size in y-direction
     * @param nz Domain size in z-direction
     * @param dsigma_dT Temperature coefficient of surface tension dσ/dT [N/(m·K)]
     * @param dx Lattice spacing [m]
     * @param interface_thickness Interface thickness h [lattice units]
     * @param max_gradient_limit Maximum physical temperature gradient [K/m] (default: 5e8)
     * @param T_melt Melting temperature [K] (default: 1923K for Ti6Al4V)
     * @param T_boil Boiling temperature [K] (default: 3560K for Ti6Al4V)
     * @param interface_cutoff_min Minimum fill fraction for interface detection (default: 0.001)
     * @param interface_cutoff_max Maximum fill fraction for interface detection (default: 0.999)
     */
    MarangoniEffect(int nx, int ny, int nz,
                    float dsigma_dT,
                    float dx = 1.0f,
                    float interface_thickness = 2.0f,
                    float max_gradient_limit = 5.0e8f,
                    float T_melt = 1923.0f,
                    float T_boil = 3560.0f,
                    float interface_cutoff_min = 0.001f,
                    float interface_cutoff_max = 0.999f);

    /**
     * @brief Destructor
     */
    ~MarangoniEffect();

    /**
     * @brief Compute Marangoni force
     * @param temperature Device array of temperature field [K]
     * @param fill_level Device array of fill level field (0-1)
     * @param interface_normal Device array of interface normals
     * @param force_x Output: force x-component [N/m³]
     * @param force_y Output: force y-component [N/m³]
     * @param force_z Output: force z-component [N/m³]
     */
    void computeMarangoniForce(const float* temperature,
                               const float* fill_level,
                               const float3* interface_normal,
                               float* force_x,
                               float* force_y,
                               float* force_z) const;

    /**
     * @brief Compute Marangoni force and add to existing force field
     * @param temperature Device array of temperature field [K]
     * @param fill_level Device array of fill level field (0-1)
     * @param interface_normal Device array of interface normals
     * @param force_x Input/Output: force x-component [N/m³]
     * @param force_y Input/Output: force y-component [N/m³]
     * @param force_z Input/Output: force z-component [N/m³]
     */
    void addMarangoniForce(const float* temperature,
                           const float* fill_level,
                           const float3* interface_normal,
                           float* force_x,
                           float* force_y,
                           float* force_z) const;

    /**
     * @brief Set temperature coefficient of surface tension
     * @param dsigma_dT Temperature coefficient [N/(m·K)]
     */
    void setDSigmaDT(float dsigma_dT) { dsigma_dT_ = dsigma_dT; }

    /**
     * @brief Get temperature coefficient
     * @return dσ/dT [N/(m·K)]
     */
    float getDSigmaDT() const { return dsigma_dT_; }

    /**
     * @brief Set interface thickness
     * @param h Interface thickness [lattice units]
     */
    void setInterfaceThickness(float h) { h_interface_ = h; }

    /**
     * @brief Get interface thickness
     * @return Interface thickness [lattice units]
     */
    float getInterfaceThickness() const { return h_interface_; }

    /**
     * @brief Compute characteristic Marangoni velocity for validation
     * @param delta_T Temperature difference across interface [K]
     * @param viscosity Dynamic viscosity [Pa·s]
     * @return Characteristic velocity v ~ (dσ/dT * ΔT) / μ [m/s]
     */
    float computeMarangoniVelocity(float delta_T, float viscosity) const;

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
    float dsigma_dT_;              ///< Temperature coefficient of surface tension [N/(m·K)]
    float dx_;                     ///< Lattice spacing [m]
    float h_interface_;            ///< Interface thickness [lattice units]
    float max_gradient_limit_;     ///< Maximum physical temperature gradient [K/m]
    float T_melt_;                 ///< Melting temperature [K]
    float T_boil_;                 ///< Boiling temperature [K]
    float interface_cutoff_min_;   ///< Minimum fill fraction for interface detection
    float interface_cutoff_max_;   ///< Maximum fill fraction for interface detection
};

// CUDA kernels for Marangoni effect

/**
 * @brief CUDA kernel for Marangoni force computation
 * @note F = (dσ/dT) * ∇_s T * |∇f| [N/m³]
 */
__global__ void computeMarangoniForceKernel(
    const float* temperature,
    const float* fill_level,
    const float3* interface_normal,
    float* force_x,
    float* force_y,
    float* force_z,
    float dsigma_dT,
    float dx,
    float h_interface,
    float max_gradient_limit,
    float interface_cutoff_min,
    float interface_cutoff_max,
    int nx, int ny, int nz);

/**
 * @brief CUDA kernel for adding Marangoni force to existing force field
 */
__global__ void addMarangoniForceKernel(
    const float* temperature,
    const float* fill_level,
    const float3* interface_normal,
    float* force_x,
    float* force_y,
    float* force_z,
    float dsigma_dT,
    float dx,
    float h_interface,
    float max_gradient_limit,
    float T_melt,
    float interface_cutoff_min,
    float interface_cutoff_max,
    int nx, int ny, int nz);

} // namespace physics
} // namespace lbm
