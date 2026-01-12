/**
 * @file force_accumulator.cu
 * @brief Implementation of ForceAccumulator class
 */

#include "physics/force_accumulator.h"
#include "physics/marangoni.h"
#include "physics/surface_tension.h"
#include "physics/recoil_pressure.h"
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <stdexcept>
#include <cmath>
#include "utils/cuda_check.h"

namespace lbm {
namespace physics {

// ============================================================================
// CUDA Kernels
// ============================================================================

/**
 * @brief Zero out force arrays
 */
__global__ void zeroForceKernel(float* fx, float* fy, float* fz, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    fx[idx] = 0.0f;
    fy[idx] = 0.0f;
    fz[idx] = 0.0f;
}

/**
 * @brief Add buoyancy force (Boussinesq approximation)
 * F_buoyancy = ρ₀ · β · (T - T_ref) · g [N/m³]
 */
__global__ void addBuoyancyForceKernel(
    const float* temperature,
    const float* liquid_fraction,
    float* fx, float* fy, float* fz,
    float T_ref, float beta, float rho,
    float gx, float gy, float gz,
    bool use_liquid_fraction,
    int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    float T = temperature[idx];

    // NaN protection
    if (isnan(T) || isinf(T)) {
        return;
    }

    // Optional liquid fraction masking (only apply buoyancy to liquid)
    float mask = 1.0f;
    if (use_liquid_fraction) {
        mask = liquid_fraction[idx];  // 0=solid, 1=liquid
    }

    float dT = T - T_ref;
    float factor = rho * beta * dT * mask;

    // Accumulate force
    fx[idx] += factor * gx;
    fy[idx] += factor * gy;
    fz[idx] += factor * gz;
}

/**
 * @brief Add Darcy damping (velocity-dependent)
 * F_darcy = -C · (1 - f_l)² / (f_l³ + ε) · ρ · v [N/m³]
 *
 * Note: Velocity must be in physical units [m/s] for correct force magnitude
 *
 * CRITICAL FIX (2025-12-03): Added damping factor clamping to prevent numerical
 * instability in mushy zone. The raw Carman-Kozeny formula can produce extreme
 * values (>1e11) in solid regions, causing velocity oscillations and incorrect
 * mushy zone behavior.
 */
__global__ void addDarcyDampingKernel(
    const float* liquid_fraction,
    const float* vx_lattice, const float* vy_lattice, const float* vz_lattice,
    float* fx, float* fy, float* fz,
    float darcy_coeff, float rho, float dx, float dt,
    int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    float fl = liquid_fraction[idx];

    // Carman-Kozeny model: F_darcy = -C·(1 - fl)²/(fl³ + ε)·ρ·v
    //
    // Physical meaning:
    // - fl = 0 (solid): Maximum damping, velocity → 0
    // - fl = 0.5 (mushy): Moderate damping, velocity reduced
    // - fl = 1 (liquid): No damping, free flow
    //
    // The epsilon prevents division by zero in pure solid (fl=0)
    const float eps = 1e-3f;  // Increased from 1e-4 for better numerical stability

    float damping_factor = -darcy_coeff * (1.0f - fl) * (1.0f - fl) / (fl * fl * fl + eps);

    // CRITICAL FIX: Clamp damping factor to prevent numerical instability
    //
    // Without clamping, solid regions (fl~0) produce damping_factor ~ -1e11,
    // which causes:
    // 1. Velocity oscillations (overcompensation)
    // 2. Incorrect mushy zone behavior (v_mushy > v_liquid)
    // 3. Numerical instability near phase interfaces
    //
    // Physical reasoning for limit:
    // - Maximum acceleration from damping should not exceed gravitational forces
    // - For Ti6Al4V: g = 9.81 m/s², buoyancy ~ 100 N/m³
    // - Typical velocity: v ~ 0.01 m/s
    // - Max damping: F_max ~ 1e8 N/m³ → damping_factor_max ~ 1e10 s⁻¹
    //
    // This gives physically reasonable damping while still enforcing v_solid ≈ 0
    const float max_damping_factor = 1e9f;  // [s⁻¹]
    damping_factor = fmaxf(damping_factor, -max_damping_factor);

    // Convert velocity from lattice units to physical units [m/s]
    float v_phys_conv = dx / dt;
    float vx = vx_lattice[idx] * v_phys_conv;
    float vy = vy_lattice[idx] * v_phys_conv;
    float vz = vz_lattice[idx] * v_phys_conv;

    // Damping force in physical units [N/m³]
    // Note: rho factor for proper physical units (momentum per volume per time)
    fx[idx] += damping_factor * rho * vx;
    fy[idx] += damping_factor * rho * vy;
    fz[idx] += damping_factor * rho * vz;
}

/**
 * @brief Add surface tension force (CSF model)
 * F_st = σ · κ · ∇f [N/m³]
 */
__global__ void addSurfaceTensionForceKernel(
    const float* curvature,
    const float* fill_level,
    float* fx, float* fy, float* fz,
    float sigma, float dx,
    int nx, int ny, int nz)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i >= nx || j >= ny || k >= nz) return;

    int idx = i + nx * (j + ny * k);

    // Only apply at interface (where fill level gradient is significant)
    float f = fill_level[idx];
    if (f <= 0.01f || f >= 0.99f) {
        return;  // Not at interface
    }

    // Compute fill level gradient (central differences)
    float grad_fx = 0.0f, grad_fy = 0.0f, grad_fz = 0.0f;

    if (i > 0 && i < nx - 1) {
        int idx_xp = (i + 1) + nx * (j + ny * k);
        int idx_xm = (i - 1) + nx * (j + ny * k);
        grad_fx = (fill_level[idx_xp] - fill_level[idx_xm]) / (2.0f * dx);
    }

    if (j > 0 && j < ny - 1) {
        int idx_yp = i + nx * ((j + 1) + ny * k);
        int idx_ym = i + nx * ((j - 1) + ny * k);
        grad_fy = (fill_level[idx_yp] - fill_level[idx_ym]) / (2.0f * dx);
    }

    if (k > 0 && k < nz - 1) {
        int idx_zp = i + nx * (j + ny * (k + 1));
        int idx_zm = i + nx * (j + ny * (k - 1));
        grad_fz = (fill_level[idx_zp] - fill_level[idx_zm]) / (2.0f * dx);
    }

    // CSF force: F = σ · κ · ∇f
    float kappa = curvature[idx];
    float force_mag = sigma * kappa;

    fx[idx] += force_mag * grad_fx;
    fy[idx] += force_mag * grad_fy;
    fz[idx] += force_mag * grad_fz;
}

/**
 * @brief Add Marangoni force (thermocapillary)
 * F_m = (dσ/dT) · ∇_s T · |∇f| / h [N/m³]
 * where |∇f|/h acts as interface delta function
 */
__global__ void addMarangoniForceKernel(
    const float* temperature,
    const float* fill_level,
    const float3* normals,
    float* fx, float* fy, float* fz,
    float dsigma_dT, float dx, float h_interface,
    int nx, int ny, int nz)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i >= nx || j >= ny || k >= nz) return;

    int idx = i + nx * (j + ny * k);

    // FIX (2026-01-12): Use interface normal magnitude for interface detection
    // This fixes the bug where fill_level=1.0 (fully liquid) excluded ALL cells
    //
    // Previous approach: f in [0.01, 0.99] - fails when domain is fully liquid
    // New approach: Use |n| > threshold - works regardless of fill_level
    //
    // Interface normal is computed by VOF solver and only has significant
    // magnitude at actual interface cells
    float3 n = normals[idx];
    float n_mag = sqrtf(n.x*n.x + n.y*n.y + n.z*n.z);

    // Skip non-interface cells (normal magnitude too small)
    // Threshold 0.01 filters out numerical noise in bulk regions
    if (n_mag < 0.01f) {
        return;
    }

    // Also check fill_level as secondary filter (keep for VOF consistency)
    float f = fill_level[idx];
    if (f <= 0.001f || f >= 0.999f) {
        // Even with significant normal, skip pure gas/liquid cells
        // Use wider cutoffs (0.001, 0.999) to capture more interface
        return;
    }

    // Compute temperature gradient (central differences)
    float grad_T_x = 0.0f, grad_T_y = 0.0f, grad_T_z = 0.0f;

    if (i > 0 && i < nx - 1) {
        int idx_xp = (i + 1) + nx * (j + ny * k);
        int idx_xm = (i - 1) + nx * (j + ny * k);
        grad_T_x = (temperature[idx_xp] - temperature[idx_xm]) / (2.0f * dx);
    }

    if (j > 0 && j < ny - 1) {
        int idx_yp = i + nx * ((j + 1) + ny * k);
        int idx_ym = i + nx * ((j - 1) + ny * k);
        grad_T_y = (temperature[idx_yp] - temperature[idx_ym]) / (2.0f * dx);
    }

    if (k > 0 && k < nz - 1) {
        int idx_zp = i + nx * (j + ny * (k + 1));
        int idx_zm = i + nx * (j + ny * (k - 1));
        grad_T_z = (temperature[idx_zp] - temperature[idx_zm]) / (2.0f * dx);
    }

    // Compute surface-tangential gradient: ∇_s T = ∇T - (∇T · n) n
    float grad_T_dot_n = grad_T_x * n.x + grad_T_y * n.y + grad_T_z * n.z;
    float grad_T_s_x = grad_T_x - grad_T_dot_n * n.x;
    float grad_T_s_y = grad_T_y - grad_T_dot_n * n.y;
    float grad_T_s_z = grad_T_z - grad_T_dot_n * n.z;

    // Compute fill level gradient magnitude (interface delta function)
    // Use one-sided differences at boundaries to avoid zero gradients
    float grad_f_x = 0.0f, grad_f_y = 0.0f, grad_f_z = 0.0f;

    if (i == 0) {
        // Forward difference at left boundary
        int idx_xp = (i + 1) + nx * (j + ny * k);
        grad_f_x = (fill_level[idx_xp] - f) / dx;
    } else if (i == nx - 1) {
        // Backward difference at right boundary
        int idx_xm = (i - 1) + nx * (j + ny * k);
        grad_f_x = (f - fill_level[idx_xm]) / dx;
    } else {
        // Central difference in interior
        int idx_xp = (i + 1) + nx * (j + ny * k);
        int idx_xm = (i - 1) + nx * (j + ny * k);
        grad_f_x = (fill_level[idx_xp] - fill_level[idx_xm]) / (2.0f * dx);
    }

    if (j == 0) {
        // Forward difference at bottom boundary
        int idx_yp = i + nx * ((j + 1) + ny * k);
        grad_f_y = (fill_level[idx_yp] - f) / dx;
    } else if (j == ny - 1) {
        // Backward difference at top boundary
        int idx_ym = i + nx * ((j - 1) + ny * k);
        grad_f_y = (f - fill_level[idx_ym]) / dx;
    } else {
        // Central difference in interior
        int idx_yp = i + nx * ((j + 1) + ny * k);
        int idx_ym = i + nx * ((j - 1) + ny * k);
        grad_f_y = (fill_level[idx_yp] - fill_level[idx_ym]) / (2.0f * dx);
    }

    if (k == 0) {
        // Forward difference at front boundary
        int idx_zp = i + nx * (j + ny * (k + 1));
        grad_f_z = (fill_level[idx_zp] - f) / dx;
    } else if (k == nz - 1) {
        // Backward difference at back boundary
        int idx_zm = i + nx * (j + ny * (k - 1));
        grad_f_z = (f - fill_level[idx_zm]) / dx;
    } else {
        // Central difference in interior
        int idx_zp = i + nx * (j + ny * (k + 1));
        int idx_zm = i + nx * (j + ny * (k - 1));
        grad_f_z = (fill_level[idx_zp] - fill_level[idx_zm]) / (2.0f * dx);
    }

    float grad_f_mag = sqrtf(grad_f_x*grad_f_x + grad_f_y*grad_f_y + grad_f_z*grad_f_z);

    if (grad_f_mag < 1e-12f) {
        return;  // No interface
    }

    // Marangoni force (CSF formulation): F = (dσ/dT) · ∇_s T · |∇f| / h
    //
    // Physical derivation:
    //   Surface stress: τ_s = (dσ/dT) · ∇_s T  [N/m²]
    //   CSF converts surface force to volumetric: F = τ_s · δ(interface)  [N/m³]
    //   where δ(interface) ≈ |∇f| / h (interface delta function)
    //   and h is the interface thickness [lattice units, dimensionless]
    //
    // Units: [N/(m·K)] · [K/m] · [1/m] / [dimensionless] = [N/m³]  ✓
    //
    // FIX (2026-01-12): Restored h_interface division for proper CSF normalization
    // The raw |∇f| in sharp interfaces can be very large (|∇f| ~ Δf/dx ~ 1e5 1/m)
    // Division by h_interface (typically 2-4 lattice cells) provides proper normalization
    // so that ∫ δ(interface) dy ≈ 1
    //
    // CRITICAL: h_interface should match the actual interface thickness in lattice units
    // For sharp test interfaces: h=2-3 cells
    // For smooth VOF interfaces: h=4-6 cells
    float coeff = dsigma_dT * grad_f_mag / h_interface;

    fx[idx] += coeff * grad_T_s_x;
    fy[idx] += coeff * grad_T_s_y;
    fz[idx] += coeff * grad_T_s_z;
}

/**
 * @brief Add recoil pressure force (evaporation-driven)
 * F_recoil = P_recoil · n · |∇f| / h [N/m³]
 * where P_recoil = C_r · P_sat(T)
 */
__global__ void addRecoilPressureForceKernel(
    const float* temperature,
    const float* fill_level,
    const float3* normals,
    float* fx, float* fy, float* fz,
    float T_boil, float L_v, float M, float P_atm,
    float C_r, float smoothing_width, float max_pressure,
    float dx,
    int nx, int ny, int nz)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i >= nx || j >= ny || k >= nz) return;

    int idx = i + nx * (j + ny * k);

    // Only apply at interface
    float f = fill_level[idx];
    if (f <= 0.01f || f >= 0.99f) {
        return;  // Not at interface
    }

    float T = temperature[idx];

    // Only apply recoil pressure above activation temperature
    const float T_activation = T_boil - 500.0f;  // Start 500K below boiling
    if (T < T_activation) {
        return;
    }

    // Compute saturation pressure using Clausius-Clapeyron
    // P_sat = P_0 · exp(L_v · M / R · (1/T_boil - 1/T))
    const float R_gas = 8.314f;  // J/(mol·K)
    float exponent = (L_v * M / R_gas) * (1.0f / T_boil - 1.0f / T);

    // Clamp exponent for stability
    exponent = fminf(exponent, 50.0f);  // exp(50) ~ 5e21
    exponent = fmaxf(exponent, -50.0f);

    float P_sat = P_atm * expf(exponent);

    // Recoil pressure with limiter
    float P_recoil = C_r * P_sat;
    P_recoil = fminf(P_recoil, max_pressure);

    // Get interface normal (points from liquid to gas)
    float3 n = normals[idx];

    // Compute fill level gradient magnitude (interface delta function)
    // Use one-sided differences at boundaries to avoid zero gradients
    float grad_f_x = 0.0f, grad_f_y = 0.0f, grad_f_z = 0.0f;

    if (i == 0) {
        // Forward difference at left boundary
        int idx_xp = (i + 1) + nx * (j + ny * k);
        grad_f_x = (fill_level[idx_xp] - f) / dx;
    } else if (i == nx - 1) {
        // Backward difference at right boundary
        int idx_xm = (i - 1) + nx * (j + ny * k);
        grad_f_x = (f - fill_level[idx_xm]) / dx;
    } else {
        // Central difference in interior
        int idx_xp = (i + 1) + nx * (j + ny * k);
        int idx_xm = (i - 1) + nx * (j + ny * k);
        grad_f_x = (fill_level[idx_xp] - fill_level[idx_xm]) / (2.0f * dx);
    }

    if (j == 0) {
        // Forward difference at bottom boundary
        int idx_yp = i + nx * ((j + 1) + ny * k);
        grad_f_y = (fill_level[idx_yp] - f) / dx;
    } else if (j == ny - 1) {
        // Backward difference at top boundary
        int idx_ym = i + nx * ((j - 1) + ny * k);
        grad_f_y = (f - fill_level[idx_ym]) / dx;
    } else {
        // Central difference in interior
        int idx_yp = i + nx * ((j + 1) + ny * k);
        int idx_ym = i + nx * ((j - 1) + ny * k);
        grad_f_y = (fill_level[idx_yp] - fill_level[idx_ym]) / (2.0f * dx);
    }

    if (k == 0) {
        // Forward difference at front boundary
        int idx_zp = i + nx * (j + ny * (k + 1));
        grad_f_z = (fill_level[idx_zp] - f) / dx;
    } else if (k == nz - 1) {
        // Backward difference at back boundary
        int idx_zm = i + nx * (j + ny * (k - 1));
        grad_f_z = (f - fill_level[idx_zm]) / dx;
    } else {
        // Central difference in interior
        int idx_zp = i + nx * (j + ny * (k + 1));
        int idx_zm = i + nx * (j + ny * (k - 1));
        grad_f_z = (fill_level[idx_zp] - fill_level[idx_zm]) / (2.0f * dx);
    }

    float grad_f_mag = sqrtf(grad_f_x*grad_f_x + grad_f_y*grad_f_y + grad_f_z*grad_f_z);

    if (grad_f_mag < 1e-12f) {
        return;  // No interface
    }

    // Recoil force: F = P_recoil · n · |∇f| / h (volumetric force)
    // Direction: INTO liquid (along -n, since n points liquid->gas)
    float coeff = P_recoil * grad_f_mag / (smoothing_width * dx);

    fx[idx] += -coeff * n.x;  // Negative: force pushes INTO liquid
    fy[idx] += -coeff * n.y;
    fz[idx] += -coeff * n.z;
}

/**
 * @brief Convert forces from physical units to lattice units
 * F_lattice = F_physical · (dt² / dx) [dimensionless]
 */
__global__ void convertToLatticeUnitsKernel(
    float* fx, float* fy, float* fz,
    float conversion_factor, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    fx[idx] *= conversion_factor;
    fy[idx] *= conversion_factor;
    fz[idx] *= conversion_factor;
}

/**
 * @brief Gradual CFL limiter for force limiting
 */
__global__ void applyCFLLimitingKernel(
    float* fx, float* fy, float* fz,
    const float* ux, const float* uy, const float* uz,
    float v_target, float ramp_factor, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    // Current velocity [lattice units]
    float vx = ux[idx];
    float vy = uy[idx];
    float vz = uz[idx];
    float v_current = sqrtf(vx*vx + vy*vy + vz*vz);

    // Force components [lattice units]
    float f_x = fx[idx];
    float f_y = fy[idx];
    float f_z = fz[idx];
    float f_mag = sqrtf(f_x*f_x + f_y*f_y + f_z*f_z);

    if (f_mag < 1e-12f) return;  // No force

    // Predicted velocity: v_new = v + F (in lattice units, dt=1)
    float v_new_x = vx + f_x;
    float v_new_y = vy + f_y;
    float v_new_z = vz + f_z;
    float v_new = sqrtf(v_new_x*v_new_x + v_new_y*v_new_y + v_new_z*v_new_z);

    // Ramping threshold
    float v_ramp = ramp_factor * v_target;

    float scale = 1.0f;

    if (v_new > v_target) {
        // Hard limit: scale to reach exactly v_target
        float delta_v_allowed = fmaxf(0.0f, v_target - v_current);
        scale = delta_v_allowed / (f_mag + 1e-12f);
        scale = fminf(scale, 1.0f);
    }
    else if (v_new > v_ramp) {
        // Gradual scaling zone
        float excess_ratio = (v_new - v_ramp) / (v_target - v_ramp + 1e-12f);
        float scale_at_target = (v_target - v_current) / (f_mag + 1e-12f);
        scale_at_target = fminf(scale_at_target, 1.0f);

        scale = 1.0f - excess_ratio * (1.0f - scale_at_target);
        scale = fmaxf(scale, 0.01f);  // Never zero out completely
    }

    // Apply scaling
    fx[idx] = f_x * scale;
    fy[idx] = f_y * scale;
    fz[idx] = f_z * scale;
}

/**
 * @brief Adaptive CFL limiter with region-based velocity targets
 */
__global__ void applyCFLLimitingAdaptiveKernel(
    float* fx, float* fy, float* fz,
    const float* ux, const float* uy, const float* uz,
    const float* fill_level, const float* liquid_fraction,
    float v_target_interface, float v_target_bulk,
    float interface_lo, float interface_hi,
    float recoil_boost_factor, float ramp_factor,
    int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    float fill = fill_level[idx];
    float liq_frac = liquid_fraction[idx];

    // Current velocity [lattice units]
    float vx = ux[idx];
    float vy = uy[idx];
    float vz = uz[idx];
    float v_current = sqrtf(vx*vx + vy*vy + vz*vz);

    // Force components [lattice units]
    float f_x = fx[idx];
    float f_y = fy[idx];
    float f_z = fz[idx];
    float f_mag = sqrtf(f_x*f_x + f_y*f_y + f_z*f_z);

    if (f_mag < 1e-12f) return;

    // Region classification
    float v_target;
    bool is_interface = (fill > interface_lo) && (fill < interface_hi);
    bool is_gas = (fill <= interface_lo);
    bool is_solid = (fill >= interface_hi) && (liq_frac < 0.01f);
    bool is_mushy = (fill >= interface_hi) && (liq_frac >= 0.01f) && (liq_frac < 0.5f);

    if (is_solid) {
        // Solid: zero force
        fx[idx] = 0.0f;
        fy[idx] = 0.0f;
        fz[idx] = 0.0f;
        return;
    }
    else if (is_gas) {
        v_target = 0.1f;
    }
    else if (is_interface) {
        v_target = v_target_interface;

        // Check for recoil signature (z-dominant force)
        float fz_ratio = fabsf(f_z) / (f_mag + 1e-12f);
        if (fz_ratio > 0.7f) {
            v_target *= recoil_boost_factor;
        }
    }
    else if (is_mushy) {
        v_target = v_target_bulk * liq_frac;
    }
    else {  // bulk liquid
        v_target = v_target_bulk;
    }

    // Gradual scaling (same as basic CFL limiter)
    float v_new_x = vx + f_x;
    float v_new_y = vy + f_y;
    float v_new_z = vz + f_z;
    float v_new = sqrtf(v_new_x*v_new_x + v_new_y*v_new_y + v_new_z*v_new_z);

    float v_ramp = ramp_factor * v_target;
    float scale = 1.0f;

    if (v_new > v_target) {
        float delta_v_allowed = fmaxf(0.0f, v_target - v_current);
        scale = delta_v_allowed / (f_mag + 1e-12f);
        scale = fminf(scale, 1.0f);
    }
    else if (v_new > v_ramp) {
        float excess_ratio = (v_new - v_ramp) / (v_target - v_ramp + 1e-12f);
        float scale_at_target = (v_target - v_current) / (f_mag + 1e-12f);
        scale_at_target = fminf(scale_at_target, 1.0f);

        scale = 1.0f - excess_ratio * (1.0f - scale_at_target);
        scale = fmaxf(scale, 0.01f);
    }

    fx[idx] = f_x * scale;
    fy[idx] = f_y * scale;
    fz[idx] = f_z * scale;
}

/**
 * @brief Compute force magnitude for diagnostics
 */
__global__ void computeForceMagnitudeKernel(
    const float* fx, const float* fy, const float* fz,
    float* f_mag, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    float fx_val = fx[idx];
    float fy_val = fy[idx];
    float fz_val = fz[idx];
    f_mag[idx] = sqrtf(fx_val*fx_val + fy_val*fy_val + fz_val*fz_val);
}

// ============================================================================
// ForceAccumulator Implementation
// ============================================================================

ForceAccumulator::ForceAccumulator(int nx, int ny, int nz)
    : nx_(nx), ny_(ny), nz_(nz), num_cells_(nx * ny * nz),
      d_fx_(nullptr), d_fy_(nullptr), d_fz_(nullptr)
{
    allocateMemory();
}

ForceAccumulator::~ForceAccumulator() {
    freeMemory();
}

void ForceAccumulator::allocateMemory() {
    size_t bytes = num_cells_ * sizeof(float);

    // Clear any previous CUDA errors before allocation
    cudaGetLastError();

    cudaError_t err;

    err = cudaMalloc(&d_fx_, bytes);
    if (err != cudaSuccess) {
        throw std::runtime_error("ForceAccumulator: Failed to allocate d_fx: " +
                                std::string(cudaGetErrorString(err)));
    }

    err = cudaMalloc(&d_fy_, bytes);
    if (err != cudaSuccess) {
        throw std::runtime_error("ForceAccumulator: Failed to allocate d_fy: " +
                                std::string(cudaGetErrorString(err)));
    }

    err = cudaMalloc(&d_fz_, bytes);
    if (err != cudaSuccess) {
        throw std::runtime_error("ForceAccumulator: Failed to allocate d_fz: " +
                                std::string(cudaGetErrorString(err)));
    }

    // Initialize to zero
    reset();
}

void ForceAccumulator::freeMemory() {
    if (d_fx_) cudaFree(d_fx_);
    if (d_fy_) cudaFree(d_fy_);
    if (d_fz_) cudaFree(d_fz_);

    d_fx_ = nullptr;
    d_fy_ = nullptr;
    d_fz_ = nullptr;
}

void ForceAccumulator::reset() {
    int threads = 256;
    int blocks = (num_cells_ + threads - 1) / threads;

    zeroForceKernel<<<blocks, threads>>>(d_fx_, d_fy_, d_fz_, num_cells_);
    CUDA_CHECK_KERNEL();
    CUDA_CHECK(cudaDeviceSynchronize());

    // Reset diagnostics
    buoyancy_mag_ = 0.0f;
    darcy_mag_ = 0.0f;
    surface_tension_mag_ = 0.0f;
    marangoni_mag_ = 0.0f;
    recoil_mag_ = 0.0f;
}

void ForceAccumulator::addBuoyancyForce(
    const float* temperature, float T_ref, float beta,
    float rho, float gx, float gy, float gz,
    const float* liquid_fraction)
{
    int threads = 256;
    int blocks = (num_cells_ + threads - 1) / threads;

    bool use_lf = (liquid_fraction != nullptr);

    addBuoyancyForceKernel<<<blocks, threads>>>(
        temperature, liquid_fraction,
        d_fx_, d_fy_, d_fz_,
        T_ref, beta, rho, gx, gy, gz,
        use_lf, num_cells_);
    CUDA_CHECK_KERNEL();

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("ForceAccumulator::addBuoyancyForce: Kernel launch failed: " +
                                std::string(cudaGetErrorString(err)));
    }

    CUDA_CHECK(cudaDeviceSynchronize());

    // Update diagnostic
    buoyancy_mag_ = getMaxForceMagnitude();
}

void ForceAccumulator::addDarcyDamping(
    const float* liquid_fraction, const float* vx, const float* vy, const float* vz,
    float darcy_coeff, float dx, float dt, float rho)
{
    int threads = 256;
    int blocks = (num_cells_ + threads - 1) / threads;

    addDarcyDampingKernel<<<blocks, threads>>>(
        liquid_fraction, vx, vy, vz,
        d_fx_, d_fy_, d_fz_,
        darcy_coeff, rho, dx, dt,
        num_cells_);
    CUDA_CHECK_KERNEL();

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("ForceAccumulator::addDarcyDamping: Kernel launch failed: " +
                                std::string(cudaGetErrorString(err)));
    }

    CUDA_CHECK(cudaDeviceSynchronize());

    // Update diagnostic
    darcy_mag_ = getMaxForceMagnitude();
}

void ForceAccumulator::addSurfaceTensionForce(
    const float* curvature, const float* fill_level,
    float sigma, int nx, int ny, int nz, float dx)
{
    dim3 threads(8, 8, 8);
    dim3 blocks((nx + threads.x - 1) / threads.x,
                (ny + threads.y - 1) / threads.y,
                (nz + threads.z - 1) / threads.z);

    addSurfaceTensionForceKernel<<<blocks, threads>>>(
        curvature, fill_level,
        d_fx_, d_fy_, d_fz_,
        sigma, dx, nx, ny, nz);
    CUDA_CHECK_KERNEL();

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("ForceAccumulator::addSurfaceTensionForce: Kernel launch failed: " +
                                std::string(cudaGetErrorString(err)));
    }

    CUDA_CHECK(cudaDeviceSynchronize());

    // Update diagnostic
    surface_tension_mag_ = getMaxForceMagnitude();
}

void ForceAccumulator::addMarangoniForce(
    const float* temperature, const float* fill_level,
    const float3* normals, float dsigma_dT,
    int nx, int ny, int nz, float dx, float h_interface)
{
    // FIX (2026-01-12): Restored h_interface usage for proper CSF normalization
    // The parameter is essential for normalizing the interface delta function

    dim3 threads(8, 8, 8);
    dim3 blocks((nx + threads.x - 1) / threads.x,
                (ny + threads.y - 1) / threads.y,
                (nz + threads.z - 1) / threads.z);

    addMarangoniForceKernel<<<blocks, threads>>>(
        temperature, fill_level, normals,
        d_fx_, d_fy_, d_fz_,
        dsigma_dT, dx, h_interface,
        nx, ny, nz);
    CUDA_CHECK_KERNEL();

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("ForceAccumulator::addMarangoniForce: Kernel launch failed: " +
                                std::string(cudaGetErrorString(err)));
    }

    CUDA_CHECK(cudaDeviceSynchronize());

    // Update diagnostic
    marangoni_mag_ = getMaxForceMagnitude();
}

void ForceAccumulator::addRecoilPressureForce(
    const float* temperature, const float* fill_level,
    const float3* normals, float T_boil, float L_v,
    float M, float P_atm, float C_r,
    float smoothing_width, float max_pressure,
    int nx, int ny, int nz, float dx)
{
    dim3 threads(8, 8, 8);
    dim3 blocks((nx + threads.x - 1) / threads.x,
                (ny + threads.y - 1) / threads.y,
                (nz + threads.z - 1) / threads.z);

    addRecoilPressureForceKernel<<<blocks, threads>>>(
        temperature, fill_level, normals,
        d_fx_, d_fy_, d_fz_,
        T_boil, L_v, M, P_atm, C_r,
        smoothing_width, max_pressure, dx,
        nx, ny, nz);
    CUDA_CHECK_KERNEL();

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("ForceAccumulator::addRecoilPressureForce: Kernel launch failed: " +
                                std::string(cudaGetErrorString(err)));
    }

    CUDA_CHECK(cudaDeviceSynchronize());

    // Update diagnostic
    recoil_mag_ = getMaxForceMagnitude();
}

void ForceAccumulator::convertToLatticeUnits(float dx, float dt, float rho) {
    // CRITICAL FIX: Corrected force conversion for Guo forcing scheme
    //
    // The Guo forcing scheme in fluidBGKCollisionVaryingForceKernel applies force as:
    //   u_corrected = u_uncorrected + 0.5 * F / ρ_lattice
    //
    // Where F is body force and ρ_lattice ≈ 1 in standard LBM.
    //
    // Starting from physical force density F_phys [N/m³] = [kg/(m²·s²)]:
    //   - Acceleration: a = F_phys / ρ_phys [m/s²]
    //   - In lattice: F_lattice should produce the same acceleration
    //   - Since ρ_lattice ≈ 1, we have: a_lattice = F_lattice / 1 = F_lattice
    //   - Physical acceleration → lattice: a_lattice = a_phys × (dt / (dx/dt))² = a_phys × dt² / dx²
    //
    // WAIT: This gives F_lattice = a_phys × dt² / dx² = (F_phys / ρ_phys) × dt² / dx²
    //
    // But empirically, LBM forces should be scaled as:
    //   F_lattice = F_phys / ρ_phys [acceleration in physical units]
    //
    // This works because LBM collision happens every lattice timestep (dt_lattice = 1),
    // and the force directly represents acceleration when ρ_lattice = 1.
    //
    // Evidence: Buoyancy of 120 N/m³ ÷ 4110 kg/m³ = 0.029 m/s² ≈ reasonable lattice force
    float conversion_factor = 1.0f / rho;

    int threads = 256;
    int blocks = (num_cells_ + threads - 1) / threads;

    convertToLatticeUnitsKernel<<<blocks, threads>>>(
        d_fx_, d_fy_, d_fz_, conversion_factor, num_cells_);
    CUDA_CHECK_KERNEL();

    CUDA_CHECK(cudaDeviceSynchronize());
}

void ForceAccumulator::applyCFLLimiting(
    const float* vx, const float* vy, const float* vz,
    float dx, float dt, float v_target, float ramp_factor)
{
    int threads = 256;
    int blocks = (num_cells_ + threads - 1) / threads;

    applyCFLLimitingKernel<<<blocks, threads>>>(
        d_fx_, d_fy_, d_fz_,
        vx, vy, vz,
        v_target, ramp_factor, num_cells_);
    CUDA_CHECK_KERNEL();

    CUDA_CHECK(cudaDeviceSynchronize());
}

void ForceAccumulator::applyCFLLimitingAdaptive(
    const float* vx, const float* vy, const float* vz,
    const float* fill_level, const float* liquid_fraction,
    float dx, float dt, float v_target_interface,
    float v_target_bulk, float interface_lo,
    float interface_hi, float recoil_boost_factor,
    float ramp_factor)
{
    int threads = 256;
    int blocks = (num_cells_ + threads - 1) / threads;

    applyCFLLimitingAdaptiveKernel<<<blocks, threads>>>(
        d_fx_, d_fy_, d_fz_,
        vx, vy, vz,
        fill_level, liquid_fraction,
        v_target_interface, v_target_bulk,
        interface_lo, interface_hi,
        recoil_boost_factor, ramp_factor,
        num_cells_);
    CUDA_CHECK_KERNEL();

    CUDA_CHECK(cudaDeviceSynchronize());
}

float ForceAccumulator::getMaxForceMagnitude() const {
    // Allocate temporary array for force magnitude
    float* d_f_mag;
    cudaError_t err = cudaMalloc(&d_f_mag, num_cells_ * sizeof(float));
    if (err != cudaSuccess) {
        throw std::runtime_error("ForceAccumulator::getMaxForceMagnitude: Failed to allocate d_f_mag: " +
                                std::string(cudaGetErrorString(err)));
    }

    int threads = 256;
    int blocks = (num_cells_ + threads - 1) / threads;

    computeForceMagnitudeKernel<<<blocks, threads>>>(
        d_fx_, d_fy_, d_fz_, d_f_mag, num_cells_);
    CUDA_CHECK_KERNEL();

    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy to host and find max
    std::vector<float> h_f_mag(num_cells_);
    cudaMemcpy(h_f_mag.data(), d_f_mag, num_cells_ * sizeof(float),
               cudaMemcpyDeviceToHost);

    CUDA_CHECK(cudaFree(d_f_mag));

    return *std::max_element(h_f_mag.begin(), h_f_mag.end());
}

void ForceAccumulator::getForceBreakdown(
    float& buoyancy_mag, float& darcy_mag,
    float& surface_tension_mag, float& marangoni_mag,
    float& recoil_mag) const
{
    buoyancy_mag = buoyancy_mag_;
    darcy_mag = darcy_mag_;
    surface_tension_mag = surface_tension_mag_;
    marangoni_mag = marangoni_mag_;
    recoil_mag = recoil_mag_;
}

void ForceAccumulator::printForceBreakdown() const {
    std::cout << "\n=== Force Breakdown ===\n";
    std::cout << "  Buoyancy:        " << buoyancy_mag_ << " N/m³\n";
    std::cout << "  Darcy damping:   " << darcy_mag_ << " N/m³\n";
    std::cout << "  Surface tension: " << surface_tension_mag_ << " N/m³\n";
    std::cout << "  Marangoni:       " << marangoni_mag_ << " N/m³\n";
    std::cout << "  Recoil pressure: " << recoil_mag_ << " N/m³\n";
    std::cout << "  Total (max):     " << getMaxForceMagnitude() << " N/m³\n";
    std::cout << "=======================\n" << std::flush;
}

} // namespace physics
} // namespace lbm
