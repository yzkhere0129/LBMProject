/**
 * @file recoil_pressure.h
 * @brief Recoil pressure computation for evaporating metal surfaces
 *
 * This module computes the recoil pressure exerted on a liquid metal
 * surface due to rapid evaporation. The recoil pressure is a critical
 * driver of keyhole formation in high-power laser welding and LPBF.
 *
 * Physical model:
 *
 * The recoil pressure arises from momentum conservation during evaporation.
 * As metal vapor leaves the surface at high velocity, the reaction force
 * pushes the liquid surface downward.
 *
 *   P_recoil = C_r * p_sat(T)
 *
 * where:
 *   - C_r = 0.54 (from kinetic theory, Knight 1979)
 *   - p_sat(T) = saturation pressure from Clausius-Clapeyron
 *
 * The recoil force is converted to a volumetric body force for LBM:
 *
 *   F_recoil = P_recoil * n * |grad(f)| / h_interface
 *
 * where:
 *   - n: interface normal (pointing from liquid into gas)
 *   - |grad(f)|: magnitude of VOF gradient (interface delta function)
 *   - h_interface: interface thickness for smoothing [cells]
 *
 * Key physics:
 * - Recoil pressure acts normal to surface, directed INTO the liquid
 * - At boiling point (p_sat ~ 1 atm), P_recoil ~ 54 kPa
 * - At T > T_boil, P_recoil grows exponentially (Clausius-Clapeyron)
 * - P_recoil ~ 1 MPa triggers keyhole formation
 *
 * References:
 * - Knight, C. J. (1979): AIAA Journal 17(5), 519-523
 * - Khairallah, S. A. et al. (2016): Acta Materialia 108, 36-45
 * - Tan, W. et al. (2013): Computational Materials Science 77, 188-195
 */

#pragma once

#include <cuda_runtime.h>

namespace lbm {
namespace physics {

/**
 * @brief Recoil pressure configuration
 */
struct RecoilPressureConfig {
    float coefficient = 0.54f;          ///< C_r coefficient (0.54-0.56 typical)
    float smoothing_width = 2.0f;       ///< Interface smoothing [cells]
    float max_pressure = 1e8f;          ///< Numerical limiter [Pa]
    float fill_level_threshold = 0.01f; ///< Min |grad(f)| for force application

    RecoilPressureConfig() = default;
};

/**
 * @brief Recoil pressure model for evaporating surfaces
 *
 * Computes the recoil pressure force that drives keyhole formation
 * in high-intensity laser processing.
 *
 * Typical values for Ti6Al4V:
 *   - At T = 3560 K (boiling): P_r ~ 54 kPa
 *   - At T = 4000 K: P_r ~ 500 kPa
 *   - At T = 4500 K: P_r ~ 5 MPa (keyhole regime)
 *
 * The force direction is always INTO the liquid (along -n where n
 * points from liquid to gas), causing surface depression.
 */
class RecoilPressure {
public:
    /**
     * @brief Constructor with configuration
     * @param config Recoil pressure configuration
     * @param dx Lattice spacing [m]
     */
    RecoilPressure(const RecoilPressureConfig& config, float dx);

    /**
     * @brief Constructor with direct parameters
     * @param recoil_coefficient C_r coefficient (0.54 typical)
     * @param smoothing_width Interface smoothing width [cells]
     * @param dx Lattice spacing [m]
     */
    RecoilPressure(float recoil_coefficient = 0.54f,
                   float smoothing_width = 2.0f,
                   float dx = 1e-6f);

    /**
     * @brief Destructor
     */
    ~RecoilPressure() = default;

    // ========================================================================
    // Device-callable computation methods
    // ========================================================================

    /**
     * @brief Compute recoil pressure from saturation pressure
     * @param p_sat Saturation pressure [Pa]
     * @return Recoil pressure [Pa]
     */
    __host__ __device__ float computePressure(float p_sat) const {
        float P_r = C_r_ * p_sat;
        return fminf(P_r, P_max_);  // Limiter for stability
    }

    /**
     * @brief Compute recoil force magnitude at a point
     * @param p_sat Saturation pressure [Pa]
     * @param grad_f_mag Magnitude of VOF gradient |grad(f)|
     * @return Force magnitude [N/m3]
     */
    __host__ __device__ float computeForceMagnitude(float p_sat, float grad_f_mag) const {
        if (grad_f_mag < f_threshold_) {
            return 0.0f;  // Not at interface
        }
        float P_r = computePressure(p_sat);
        // F = P_r * |grad(f)| / h (volumetric force)
        return P_r * grad_f_mag / (h_interface_ * dx_);
    }

    // ========================================================================
    // Field computation methods (CUDA kernel wrappers)
    // ========================================================================

    /**
     * @brief Compute recoil pressure field
     * @param saturation_pressure p_sat(T) field [Pa]
     * @param recoil_pressure Output: P_recoil field [Pa]
     * @param nx, ny, nz Grid dimensions
     */
    void computePressureField(
        const float* saturation_pressure,
        float* recoil_pressure,
        int nx, int ny, int nz) const;

    /**
     * @brief Compute recoil force field (volumetric force)
     * @param saturation_pressure p_sat(T) field [Pa]
     * @param fill_level VOF fill level (0-1)
     * @param interface_normal Surface normals (pointing liquid->gas)
     * @param force_x, force_y, force_z Output forces [N/m3]
     * @param nx, ny, nz Grid dimensions
     *
     * Force direction: INTO the liquid (along -n)
     * Force magnitude: P_recoil * |grad(f)| / h_interface
     */
    void computeForceField(
        const float* saturation_pressure,
        const float* fill_level,
        const float3* interface_normal,
        float* force_x,
        float* force_y,
        float* force_z,
        int nx, int ny, int nz) const;

    /**
     * @brief Add recoil force to existing force arrays
     *
     * Convenience method for integration with MultiphysicsSolver::computeTotalForce()
     */
    void addForceField(
        const float* saturation_pressure,
        const float* fill_level,
        const float3* interface_normal,
        float* force_x,
        float* force_y,
        float* force_z,
        int nx, int ny, int nz) const;

    /**
     * @brief Add recoil force directly from temperature field
     *
     * Computes P_sat from temperature using Clausius-Clapeyron and applies
     * recoil force in a single pass. This is the preferred method for
     * MultiphysicsSolver integration.
     *
     * @param temperature Temperature field [K]
     * @param fill_level VOF fill level (0-1)
     * @param interface_normal Surface normals (pointing liquid->gas)
     * @param force_x, force_y, force_z Force arrays to add to [N/m3]
     * @param nx, ny, nz Grid dimensions
     */
    void addForceFromTemperature(
        const float* temperature,
        const float* fill_level,
        const float3* interface_normal,
        float* force_x,
        float* force_y,
        float* force_z,
        int nx, int ny, int nz) const;

    // ========================================================================
    // Parameter access and modification
    // ========================================================================

    float getRecoilCoefficient() const { return C_r_; }
    void setRecoilCoefficient(float C_r) { C_r_ = C_r; }

    float getSmoothingWidth() const { return h_interface_; }
    void setSmoothingWidth(float h) { h_interface_ = h; }

    float getMaxPressure() const { return P_max_; }
    void setMaxPressure(float P_max) { P_max_ = P_max; }

    float getLatticeSpacing() const { return dx_; }

private:
    float C_r_;             ///< Recoil coefficient (0.54 typical)
    float h_interface_;     ///< Interface thickness [cells]
    float dx_;              ///< Lattice spacing [m]
    float P_max_;           ///< Maximum pressure limiter [Pa]
    float f_threshold_;     ///< Minimum |grad(f)| threshold
};

// ============================================================================
// CUDA Kernels
// ============================================================================

/**
 * @brief CUDA kernel for recoil pressure computation
 */
__global__ void computeRecoilPressureKernel(
    const float* __restrict__ saturation_pressure,
    float* __restrict__ recoil_pressure,
    float C_r,
    float P_max,
    int num_cells);

/**
 * @brief CUDA kernel for recoil force computation
 *
 * Computes volumetric force from recoil pressure:
 *   F = P_r * (-n) * |grad(f)| / h
 *
 * where -n ensures force points INTO the liquid
 */
__global__ void computeRecoilForceKernel(
    const float* __restrict__ saturation_pressure,
    const float* __restrict__ fill_level,
    const float3* __restrict__ interface_normal,
    float* __restrict__ force_x,
    float* __restrict__ force_y,
    float* __restrict__ force_z,
    float C_r,
    float h_interface,
    float dx,
    float P_max,
    float f_threshold,
    int nx, int ny, int nz);

/**
 * @brief CUDA kernel for adding recoil force to existing arrays
 */
__global__ void addRecoilForceKernel(
    const float* __restrict__ saturation_pressure,
    const float* __restrict__ fill_level,
    const float3* __restrict__ interface_normal,
    float* __restrict__ force_x,
    float* __restrict__ force_y,
    float* __restrict__ force_z,
    float C_r,
    float h_interface,
    float dx,
    float P_max,
    float f_threshold,
    int nx, int ny, int nz);

/**
 * @brief CUDA device function to compute VOF gradient magnitude
 *
 * Used to identify interface cells and compute force distribution
 */
__device__ inline float computeGradFMagnitude(
    const float* fill_level,
    int i, int j, int k,
    int nx, int ny, int nz,
    float dx)
{
    // Neighbor indices with clamping
    int im = max(0, i - 1);
    int ip = min(nx - 1, i + 1);
    int jm = max(0, j - 1);
    int jp = min(ny - 1, j + 1);
    int km = max(0, k - 1);
    int kp = min(nz - 1, k + 1);

    // Fetch neighbor values
    float f_xm = fill_level[im + nx * (j + ny * k)];
    float f_xp = fill_level[ip + nx * (j + ny * k)];
    float f_ym = fill_level[i + nx * (jm + ny * k)];
    float f_yp = fill_level[i + nx * (jp + ny * k)];
    float f_zm = fill_level[i + nx * (j + ny * km)];
    float f_zp = fill_level[i + nx * (j + ny * kp)];

    // Central difference gradients
    float grad_x = (f_xp - f_xm) / (2.0f * dx);
    float grad_y = (f_yp - f_ym) / (2.0f * dx);
    float grad_z = (f_zp - f_zm) / (2.0f * dx);

    return sqrtf(grad_x * grad_x + grad_y * grad_y + grad_z * grad_z);
}

} // namespace physics
} // namespace lbm
