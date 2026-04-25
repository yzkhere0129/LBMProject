/**
 * @file ray_tracing_laser.h
 * @brief Geometric ray tracing heat source for LPBF powder bed simulation
 *
 * Replaces the downward-projected Beer-Lambert model with ray tracing that
 * captures the "trap effect" — multiple reflections between powder particles
 * that boost effective absorptivity from ~0.35 to ~0.7.
 *
 * Algorithm per timestep:
 *   1. Discretize Gaussian beam into N rays (inverse-CDF + golden angle)
 *   2. Trace each ray through the LBM grid using 3D-DDA (Amanatides-Woo)
 *   3. At VOF interface hits: absorb (Fresnel), reflect, deposit via atomicAdd
 *   4. Terminate on: boundary escape / energy exhaustion / bounce limit
 *
 * References:
 *   - Amanatides & Woo (1987), "A Fast Voxel Traversal Algorithm for Ray Tracing"
 *   - Khairallah et al. (2016), "Laser powder-bed fusion additive manufacturing:
 *     Physics of complex melt flow and formation mechanisms of pores, spatter,
 *     and denudation zones", Acta Materialia 108, 36-45.
 */

#pragma once

#include <cuda_runtime.h>
#include "physics/laser_source.h"
#include "utils/cuda_memory.h"

namespace lbm {
namespace physics {

// ============================================================================
// Ray data structure (POD, usable on both host and device)
// ============================================================================

struct Ray {
    float3 pos;      ///< Current position [m]
    float3 dir;      ///< Normalized propagation direction
    float  energy;   ///< Remaining power carried by this ray [W]
    int    bounces;  ///< Number of reflections completed
    int    active;   ///< 1 = still tracing, 0 = terminated (int for GPU friendliness)
};

// ============================================================================
// Configuration
// ============================================================================

struct RayTracingConfig {
    bool  enabled            = false;   ///< Use ray tracing instead of Beer-Lambert
    int   num_rays           = 2048;    ///< Rays per timestep (power-of-2 preferred)
    int   max_bounces        = 5;       ///< Maximum reflections before termination
    int   max_dda_steps      = 500;     ///< Safety limit on DDA iterations per bounce
    float absorptivity       = 0.35f;   ///< Base absorptivity α₀ (constant Fresnel)
    bool  use_fresnel        = false;   ///< Angle-dependent Fresnel model (Phase 2)
    float fresnel_n_refract  = 2.9613f; ///< Real part n  (316L @ 1064nm Mills)
    float fresnel_k_extinct  = 4.0133f; ///< Imaginary part k (316L @ 1064nm Mills) — Sprint-1: needed for metal Fresnel; without k, single-bounce α is wildly off and keyhole cannot form
    float energy_cutoff      = 0.01f;   ///< Terminate when energy < cutoff × initial
    float spawn_margin_cells = 2.0f;    ///< Spawn height above domain top [cells]
    int   normal_smoothing   = 0;       ///< fill_level smoothing passes (0 = use raw)
    float cutoff_radii       = 3.0f;    ///< Gaussian beam cutoff [spot radii]
};

// ============================================================================
// RayTracingLaser class
// ============================================================================

class RayTracingLaser {
public:
    /**
     * @brief Construct ray tracing laser module
     * @param config  Ray tracing parameters
     * @param nx,ny,nz  Grid dimensions
     * @param dx  Lattice spacing [m]
     */
    RayTracingLaser(const RayTracingConfig& config, int nx, int ny, int nz, float dx);
    ~RayTracingLaser() = default;

    // Move-only (CudaBuffer members)
    RayTracingLaser(const RayTracingLaser&) = delete;
    RayTracingLaser& operator=(const RayTracingLaser&) = delete;
    RayTracingLaser(RayTracingLaser&&) = default;
    RayTracingLaser& operator=(RayTracingLaser&&) = default;

    /**
     * @brief Trace rays and deposit heat into d_heat_source
     *
     * @param d_fill_level   VOF fill level [device, num_cells] (nullptr → treat as vacuum)
     * @param d_normals      Interface normals from VOF [device, num_cells] (nullptr → compute internally)
     * @param laser          Current laser state (position, power, spot radius)
     * @param d_heat_source  Output volumetric heat [W/m³] — accumulated via atomicAdd
     *                       Caller MUST zero this array before calling.
     */
    void traceAndDeposit(const float* d_fill_level,
                         const float3* d_normals,
                         const LaserSource& laser,
                         float* d_heat_source);

    // ----- Diagnostics -----

    /// Total power deposited in last traceAndDeposit() call [W]
    float getDepositedPower() const { return deposited_power_; }

    /// Total power that escaped the domain [W]
    float getEscapedPower() const { return escaped_power_; }

    /// Total input power of last call [W]
    float getInputPower() const { return input_power_; }

    /// Relative energy conservation error: |dep+esc-input|/input
    float getEnergyError() const;

    /// Number of rays used
    int getNumRays() const { return config_.num_rays; }

private:
    RayTracingConfig config_;
    int nx_, ny_, nz_;
    float dx_;

    // Device buffers
    lbm::utils::CudaBuffer<Ray>   d_rays_;
    lbm::utils::CudaBuffer<float> d_deposited_;   ///< Per-ray deposited energy [W]
    lbm::utils::CudaBuffer<float> d_escaped_;      ///< Per-ray escaped energy [W]
    lbm::utils::CudaBuffer<float> d_smoothed_fill_;///< Smoothed fill_level (optional)

    // Diagnostic cache (host)
    float deposited_power_ = 0.0f;
    float escaped_power_   = 0.0f;
    float input_power_     = 0.0f;

    /// Sum a device array of n floats (CUB or fallback reduction)
    static float reduceSum(const float* d_array, int n);
};

// ============================================================================
// CUDA kernel forward declarations
// ============================================================================

/**
 * @brief Initialize rays with Gaussian beam profile
 *
 * Uses inverse-CDF radial sampling + golden-angle azimuthal distribution.
 * Each ray carries equal power: P_ray = P_total * F_cutoff / N
 */
__global__ void initializeRaysKernel(
    Ray* rays,
    int num_rays,
    float laser_x,       ///< Beam center X [m]
    float laser_y,       ///< Beam center Y [m]
    float spot_radius,   ///< w₀ [m]
    float total_power,   ///< P [W]
    float spawn_z,       ///< Ray origin Z [m]
    float cutoff_radii   ///< Beam cutoff in spot radii
);

/**
 * @brief Trace rays through VOF field using 3D-DDA, deposit energy at interfaces
 *
 * Each thread traces one ray independently. On interface hit (f transitions
 * from <0.5 to >=0.5), absorbs energy, reflects, and continues until terminated.
 */
__global__ void traceRaysKernel(
    Ray* rays,
    const float* d_fill_level,
    const float3* d_normals,
    float* d_heat_source,
    float* d_deposited,
    float* d_escaped,
    float absorptivity,
    bool use_fresnel,
    float fresnel_n,
    int max_bounces,
    int max_dda_steps,
    float energy_cutoff,
    int nx, int ny, int nz,
    float dx
);

} // namespace physics
} // namespace lbm
