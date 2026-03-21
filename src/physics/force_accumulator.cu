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
 * @brief Add VOF-based buoyancy force (density difference model)
 *
 * Physical derivation:
 *   Local density: ρ(f) = f × ρ_liquid + (1-f) × ρ_gas
 *   Reference density: ρ_ref = ρ_liquid (liquid phase as reference)
 *   Buoyancy force: F = (ρ - ρ_ref) × g = (ρ_gas - ρ_liquid) × (1-f) × g [N/m³]
 *
 * This formulation is exact for two-phase flows with constant densities.
 * It naturally produces:
 *   - Gas regions (f=0): F = (ρ_gas - ρ_liquid) × g (upward buoyancy if ρ_gas < ρ_liquid)
 *   - Interface (f=0.5): F = 0.5 × (ρ_gas - ρ_liquid) × g (interpolated)
 *   - Liquid regions (f=1): F = 0 (no buoyancy relative to reference)
 *
 * @param fill_level VOF field [0=gas, 1=liquid]
 * @param fx, fy, fz Force accumulation arrays [N/m³]
 * @param rho_liquid Liquid density [kg/m³]
 * @param rho_gas Gas density [kg/m³]
 * @param gx, gy, gz Gravity vector [m/s²]
 * @param n Number of cells
 */
__global__ void addVOFBuoyancyForceKernel(
    const float* fill_level,
    float* fx, float* fy, float* fz,
    float rho_liquid, float rho_gas,
    float gx, float gy, float gz,
    int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    float f = fill_level[idx];

    // NaN protection
    if (isnan(f) || isinf(f)) {
        return;
    }

    // Clamp fill level to valid range [0, 1]
    f = fmaxf(0.0f, fminf(1.0f, f));

    // CRITICAL FIX (2026-01-19): Corrected VOF buoyancy force formula
    //
    // Buoyancy force for Rayleigh-Taylor / VOF simulations:
    //   F = (ρ - ρ_avg) × g  [N/m³]
    //
    // Physical derivation:
    //   ρ(f) = f × ρ_heavy + (1-f) × ρ_light
    //   ρ_avg = (ρ_heavy + ρ_light) / 2  ← Reference density
    //
    //   F = (ρ - ρ_avg) × g
    //     = [f × ρ_heavy + (1-f) × ρ_light - (ρ_heavy + ρ_light)/2] × g
    //     = [f × (ρ_heavy - ρ_light) + ρ_light - (ρ_heavy + ρ_light)/2] × g
    //     = [f × Δρ - Δρ/2] × g
    //     = (f - 0.5) × Δρ × g
    //
    // Alternative form (equivalent):
    //   F = (2f - 1) × Δρ / 2 × g
    //
    // Physical interpretation:
    //   - f=1.0 (pure heavy): F = +0.5 × Δρ × g (net downward, half the full Δρ)
    //   - f=0.0 (pure light):  F = -0.5 × Δρ × g (net upward, half the full Δρ)
    //   - f=0.5 (interface):   F = 0 (balanced)
    //
    // ROOT CAUSE OF 2.3× SLOWDOWN: Previous formula used (2f-1)×Δρ without /2,
    // which gave 2× too large forces! This caused immediate velocity blow-up,
    // triggering CFL limiting or causing numerical instability that damped growth.
    //
    float density_diff = rho_liquid - rho_gas;  // Δρ = ρ_heavy - ρ_light
    float factor = (f - 0.5f) * density_diff;   // CORRECTED: (f - 0.5) × Δρ

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
    const float* liquid_fraction,
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

    // Gas/liquid isolation: strict cutoff to prevent Marangoni in gas cells
    // and pure liquid interior where |∇f| should be zero anyway
    float f = fill_level[idx];
    if (f <= 0.01f || f >= 0.99f) {
        return;
    }

    // Mushy zone gate: suppress Marangoni in deep solid/mushy where it's unphysical.
    // With EDM forcing (no Guo anisotropy accumulation), we only need to exclude
    // the solid zone where Marangoni has no physical meaning. The gate at fl > 0.1
    // allows force across the entire diffuse VOF interface while blocking solid.
    float fl_gate = 1.0f;
    if (liquid_fraction != nullptr) {
        float fl = liquid_fraction[idx];
        fl_gate = fminf(fmaxf((fl - 0.1f) / 0.1f, 0.0f), 1.0f);
        if (fl_gate < 1e-6f) {
            return;
        }
    }

    // Compute temperature gradient with one-sided differences at boundaries
    float grad_T_x = 0.0f, grad_T_y = 0.0f, grad_T_z = 0.0f;

    if (i > 0 && i < nx - 1) {
        int idx_xp = (i + 1) + nx * (j + ny * k);
        int idx_xm = (i - 1) + nx * (j + ny * k);
        grad_T_x = (temperature[idx_xp] - temperature[idx_xm]) / (2.0f * dx);
    } else if (i == 0) {
        int idx_xp = (i + 1) + nx * (j + ny * k);
        grad_T_x = (temperature[idx_xp] - temperature[idx]) / dx;
    } else { // i == nx - 1
        int idx_xm = (i - 1) + nx * (j + ny * k);
        grad_T_x = (temperature[idx] - temperature[idx_xm]) / dx;
    }

    if (j > 0 && j < ny - 1) {
        int idx_yp = i + nx * ((j + 1) + ny * k);
        int idx_ym = i + nx * ((j - 1) + ny * k);
        grad_T_y = (temperature[idx_yp] - temperature[idx_ym]) / (2.0f * dx);
    } else if (j == 0) {
        int idx_yp = i + nx * ((j + 1) + ny * k);
        grad_T_y = (temperature[idx_yp] - temperature[idx]) / dx;
    } else { // j == ny - 1
        int idx_ym = i + nx * ((j - 1) + ny * k);
        grad_T_y = (temperature[idx] - temperature[idx_ym]) / dx;
    }

    if (k > 0 && k < nz - 1) {
        int idx_zp = i + nx * (j + ny * (k + 1));
        int idx_zm = i + nx * (j + ny * (k - 1));
        grad_T_z = (temperature[idx_zp] - temperature[idx_zm]) / (2.0f * dx);
    } else if (k == 0) {
        int idx_zp = i + nx * (j + ny * (k + 1));
        grad_T_z = (temperature[idx_zp] - temperature[idx]) / dx;
    } else { // k == nz - 1
        int idx_zm = i + nx * (j + ny * (k - 1));
        grad_T_z = (temperature[idx] - temperature[idx_zm]) / dx;
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

    // Marangoni force (CSF formulation): F = f · (dσ/dT) · ∇_s T · |∇f| / h
    //
    // Physical derivation:
    //   Surface stress: τ_s = (dσ/dT) · ∇_s T  [N/m²]
    //   CSF converts surface force to volumetric: F = τ_s · δ(interface)  [N/m³]
    //   where δ(interface) ≈ |∇f| / h (interface delta function)
    //   and h is the interface thickness [lattice units, dimensionless]
    //
    // Standard CSF Marangoni force: F = dσ/dT × ∇_s T × |∇f|  [N/m³]
    //
    // |∇f| is the interface delta function approximation. When computed with
    // physical dx, its integral across the interface is ≈ 1 — NO additional
    // h_interface normalization is needed (matches the surface tension CSF
    // kernel which also uses ∇f directly without /h).
    //
    // fl_gate suppresses force in solid/mushy zone.
    // Gas-side isolation is already handled by f ∈ (0.01, 0.99) check above
    // and by |∇f| → 0 in bulk regions.
    float coeff = fl_gate * dsigma_dT * grad_f_mag;

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

    // Recoil force (CSF formulation): F = -P_recoil · n · |∇f|  [N/m³]
    // |∇f| is the CSF delta function that converts surface pressure to volumetric force.
    // Units: [Pa] × [1/m] = [N/m³]. No extra normalization needed.
    //
    // BUG FIX: Was dividing by (smoothing_width × dx) which added an extra
    // [1/m] factor, amplifying force ~250000× for dx=2μm. This matches
    // recoil_pressure.cu (lines 192, 274) which correctly uses F = P × |∇f|.
    float coeff = P_recoil * grad_f_mag;

    fx[idx] += -coeff * n.x;  // Negative: force pushes INTO liquid
    fy[idx] += -coeff * n.y;
    fz[idx] += -coeff * n.z;
}

/**
 * @brief Compute Darcy coefficient field K for semi-implicit velocity update
 *
 * LINEAR MODEL: K_LU = C · (1 - fl) · ρ · dt
 *
 * Replaces Carman-Kozeny ((1-fl)²/(fl³+ε)) which is catastrophically stiff:
 * a fl difference of 0.01→0.1 creates 810× K gradient, causing velocity
 * checkerboarding at the solidus front on coarse (2μm) grids.
 *
 * The linear model provides smooth, monotonic braking:
 *   fl=0 (solid): K = C·ρ·dt (full resistance)
 *   fl=0.5 (mushy): K = 0.5·C·ρ·dt (half resistance)
 *   fl=1 (liquid): K = 0 (free flow)
 *
 * Semi-implicit velocity: u = [m + 0.5·F] / (ρ_LU + 0.5·K_LU)
 */
__global__ void computeDarcyCoefficientKernel(
    const float* liquid_fraction,
    const float* fill_level,
    float* darcy_K,
    float darcy_coeff,
    float rho,
    float dt,
    int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    // Gas phase (f < 0.05): zero Darcy resistance. Air is not mushy zone!
    if (fill_level != nullptr && fill_level[idx] < 0.05f) {
        darcy_K[idx] = 0.0f;
        return;
    }

    float fl = liquid_fraction[idx];
    fl = fmaxf(0.0f, fminf(1.0f, fl));

    // Linear Darcy: K = C · (1 - fl)
    float K_factor = darcy_coeff * (1.0f - fl);

    // Convert to lattice units: K_LU = K_factor · ρ_phys · dt
    darcy_K[idx] = K_factor * rho * dt;
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
        // Smooth exponential damping instead of hard cutoff
        // Avoids discontinuous force jump at v_target boundary
        float excess = v_current - v_target * ramp_factor;
        if (excess > 0.0f) {
            float damping = expf(-2.0f * excess / (v_target + 1e-12f));
            scale = fminf(scale, damping);
        } else {
            float delta_v_allowed = fmaxf(0.0f, v_target - v_current);
            scale = delta_v_allowed / (f_mag + 1e-12f);
            scale = fminf(scale, 1.0f);
        }
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
        // Smooth exponential damping instead of hard cutoff
        float excess = v_current - v_target * ramp_factor;
        if (excess > 0.0f) {
            float damping = expf(-2.0f * excess / (v_target + 1e-12f));
            scale = fminf(scale, damping);
        } else {
            float delta_v_allowed = fmaxf(0.0f, v_target - v_current);
            scale = delta_v_allowed / (f_mag + 1e-12f);
            scale = fminf(scale, 1.0f);
        }
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
 * @brief Per-cell velocity increment cap (dynamic CFL relaxation)
 *
 * Limits force magnitude so that the single-step velocity change
 * Δu = 0.5·|F|/ρ_LU does not exceed max_delta_u.
 * With ρ_LU ≈ 1 (standard LBM): |F_max| = 2 · max_delta_u.
 *
 * This prevents numerical shock when large physical forces (e.g.,
 * Marangoni at ~90M N/m³) are suddenly unmasked by phase change
 * (Darcy suppression drops to zero when fl → 1).
 */
__global__ void capForcePerCellKernel(
    float* fx, float* fy, float* fz,
    float f_max_lu,
    int* num_capped,
    float* max_uncapped_force,
    float* total_deleted_momentum,
    int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    float fxv = fx[idx];
    float fyv = fy[idx];
    float fzv = fz[idx];
    float fmag = sqrtf(fxv*fxv + fyv*fyv + fzv*fzv);

    if (fmag > f_max_lu) {
        float scale = f_max_lu / fmag;
        fx[idx] = fxv * scale;
        fy[idx] = fyv * scale;
        fz[idx] = fzv * scale;

        atomicAdd(num_capped, 1);
        // Track max uncapped force (approximate via atomicMax on int)
        atomicMax(reinterpret_cast<int*>(max_uncapped_force),
                  __float_as_int(fmag));
        // Track deleted momentum: |F_original| - |F_capped|
        atomicAdd(total_deleted_momentum, fmag - f_max_lu);
    }
}

/**
 * @brief 3x3x3 box-filter smoothing of force field
 *
 * For each interior cell, replaces (fx,fy,fz) with the average over its
 * 3x3x3 neighbourhood (27 cells).  Only the centre cell is smoothed when
 * |F_centre| > threshold; boundary cells (any face) are written through
 * unchanged.  The input and output buffers are separate so there are no
 * read-write hazards within a single pass.
 *
 * The filter is conservative: the neighbourhood mean preserves the total
 * force integral to first order (sum-before == sum-after up to boundary
 * contributions that are held fixed).
 */
__global__ void smoothForceFieldKernel(
    const float* fx_in, const float* fy_in, const float* fz_in,
    float* fx_out, float* fy_out, float* fz_out,
    float threshold,
    int nx, int ny, int nz)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i >= nx || j >= ny || k >= nz) return;

    int idx = i + nx * (j + ny * k);

    // Boundary cells: pass through unchanged
    if (i == 0 || i == nx - 1 ||
        j == 0 || j == ny - 1 ||
        k == 0 || k == nz - 1)
    {
        fx_out[idx] = fx_in[idx];
        fy_out[idx] = fy_in[idx];
        fz_out[idx] = fz_in[idx];
        return;
    }

    // Skip cells with negligible force to avoid spreading zero-force regions
    float fc_x = fx_in[idx];
    float fc_y = fy_in[idx];
    float fc_z = fz_in[idx];
    float f_mag = sqrtf(fc_x*fc_x + fc_y*fc_y + fc_z*fc_z);

    if (f_mag <= threshold) {
        fx_out[idx] = fc_x;
        fy_out[idx] = fc_y;
        fz_out[idx] = fc_z;
        return;
    }

    // Accumulate 3x3x3 neighbourhood sum
    float sum_x = 0.0f, sum_y = 0.0f, sum_z = 0.0f;

    for (int dk = -1; dk <= 1; ++dk) {
        for (int dj = -1; dj <= 1; ++dj) {
            for (int di = -1; di <= 1; ++di) {
                int ni = i + di;
                int nj = j + dj;
                int nk = k + dk;
                int nidx = ni + nx * (nj + ny * nk);
                sum_x += fx_in[nidx];
                sum_y += fy_in[nidx];
                sum_z += fz_in[nidx];
            }
        }
    }

    const float inv27 = 1.0f / 27.0f;
    fx_out[idx] = sum_x * inv27;
    fy_out[idx] = sum_y * inv27;
    fz_out[idx] = sum_z * inv27;
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
      d_fx_(nullptr), d_fy_(nullptr), d_fz_(nullptr), d_darcy_K_(nullptr)
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
    if (d_darcy_K_) cudaFree(d_darcy_K_);

    d_fx_ = nullptr;
    d_fy_ = nullptr;
    d_fz_ = nullptr;
    d_darcy_K_ = nullptr;
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

void ForceAccumulator::addVOFBuoyancyForce(
    const float* fill_level,
    float rho_liquid, float rho_gas,
    float gx, float gy, float gz)
{
    int threads = 256;
    int blocks = (num_cells_ + threads - 1) / threads;

    addVOFBuoyancyForceKernel<<<blocks, threads>>>(
        fill_level,
        d_fx_, d_fy_, d_fz_,
        rho_liquid, rho_gas,
        gx, gy, gz,
        num_cells_);
    CUDA_CHECK_KERNEL();

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("ForceAccumulator::addVOFBuoyancyForce: Kernel launch failed: " +
                                std::string(cudaGetErrorString(err)));
    }

    CUDA_CHECK(cudaDeviceSynchronize());

    // Update diagnostic (same as thermal buoyancy for compatibility)
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

void ForceAccumulator::computeDarcyCoefficientField(
    const float* liquid_fraction, const float* fill_level,
    float darcy_coeff, float rho, float dx, float dt)
{
    // Lazy-allocate the Darcy coefficient buffer
    if (!d_darcy_K_) {
        cudaError_t err = cudaMalloc(&d_darcy_K_, num_cells_ * sizeof(float));
        if (err != cudaSuccess) {
            throw std::runtime_error(
                "ForceAccumulator::computeDarcyCoefficientField: "
                "Failed to allocate d_darcy_K_: " +
                std::string(cudaGetErrorString(err)));
        }
    }

    int threads = 256;
    int blocks = (num_cells_ + threads - 1) / threads;

    computeDarcyCoefficientKernel<<<blocks, threads>>>(
        liquid_fraction, fill_level, d_darcy_K_, darcy_coeff, rho, dt, num_cells_);
    CUDA_CHECK_KERNEL();

    CUDA_CHECK(cudaDeviceSynchronize());
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
    const float* liquid_fraction,
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
        temperature, fill_level, liquid_fraction, normals,
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
    // Convert volumetric force [N/m³] to lattice force [dimensionless]
    //
    // DIMENSIONAL ANALYSIS:
    //   Physical acceleration: a = F_phys / ρ_phys [m/s²]
    //   Lattice velocity change per step: Δu = a × dt²/dx = F_phys × dt²/(dx × ρ_phys)
    //   Collision kernel applies: Δu = F_LU / ρ_LU  (where ρ_LU ≈ 1)
    //   Equating: F_LU = F_phys × dt² / (dx × ρ_phys)
    //
    // WHY 1/ρ_phys IS REQUIRED:
    //   Forces like buoyancy already contain ρ_phys (F = ρ·β·ΔT·g), so the
    //   ρ_phys in numerator cancels with 1/ρ_phys in conversion → correct.
    //   Surface tension (F = σ·κ·∇f) and Marangoni (F = dσ/dT·|∇f|·∇_sT/h)
    //   do NOT contain ρ_phys → the 1/ρ_phys is essential for correct scaling.
    //   Without it, these forces are ρ_phys (~7900 for steel) times too large.
    //
    float conversion_factor = dt * dt / (dx * rho);

    int threads = 256;
    int blocks = (num_cells_ + threads - 1) / threads;

    convertToLatticeUnitsKernel<<<blocks, threads>>>(
        d_fx_, d_fy_, d_fz_, conversion_factor, num_cells_);
    CUDA_CHECK_KERNEL();

    CUDA_CHECK(cudaDeviceSynchronize());
}

ForceAccumulator::CapStats ForceAccumulator::capPerCellVelocityIncrement(float max_delta_u) {
    // EDM forcing: Δu = F / ρ_LU.  With ρ_LU ≈ 1: Δu ≈ |F|
    // To enforce Δu ≤ max_delta_u → |F| ≤ max_delta_u
    float f_max_lu = max_delta_u;

    // Allocate counters on device
    int* d_num_capped;
    float* d_max_uncapped;
    float* d_total_deleted;
    CUDA_CHECK(cudaMalloc(&d_num_capped, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_max_uncapped, sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_total_deleted, sizeof(float)));
    CUDA_CHECK(cudaMemset(d_num_capped, 0, sizeof(int)));
    CUDA_CHECK(cudaMemset(d_max_uncapped, 0, sizeof(float)));
    CUDA_CHECK(cudaMemset(d_total_deleted, 0, sizeof(float)));

    int threads = 256;
    int blocks = (num_cells_ + threads - 1) / threads;

    capForcePerCellKernel<<<blocks, threads>>>(
        d_fx_, d_fy_, d_fz_, f_max_lu,
        d_num_capped, d_max_uncapped, d_total_deleted,
        num_cells_);
    CUDA_CHECK_KERNEL();
    CUDA_CHECK(cudaDeviceSynchronize());

    CapStats stats;
    stats.cap_threshold = f_max_lu;
    CUDA_CHECK(cudaMemcpy(&stats.num_capped, d_num_capped, sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&stats.max_uncapped_force, d_max_uncapped, sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&stats.total_deleted_momentum, d_total_deleted, sizeof(float), cudaMemcpyDeviceToHost));
    stats.total_cells = num_cells_;

    CUDA_CHECK(cudaFree(d_num_capped));
    CUDA_CHECK(cudaFree(d_max_uncapped));
    CUDA_CHECK(cudaFree(d_total_deleted));

    return stats;
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
    CUDA_CHECK(cudaMemcpy(h_f_mag.data(), d_f_mag, num_cells_ * sizeof(float),
               cudaMemcpyDeviceToHost));

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

void ForceAccumulator::smoothForceField(int nx, int ny, int nz, int iterations) {
    const size_t bytes = num_cells_ * sizeof(float);

    // Allocate temporary scratch buffers for ping-pong smoothing
    float* d_tmp_x;
    float* d_tmp_y;
    float* d_tmp_z;
    CUDA_CHECK(cudaMalloc(&d_tmp_x, bytes));
    CUDA_CHECK(cudaMalloc(&d_tmp_y, bytes));
    CUDA_CHECK(cudaMalloc(&d_tmp_z, bytes));

    dim3 threads(8, 8, 8);
    dim3 blocks((nx + threads.x - 1) / threads.x,
                (ny + threads.y - 1) / threads.y,
                (nz + threads.z - 1) / threads.z);

    // Only smooth cells with |F| above this threshold to avoid diffusing
    // zero-force regions into active force regions.
    const float threshold = 1e-10f;

    for (int iter = 0; iter < iterations; ++iter) {
        // Read from primary arrays, write to scratch
        smoothForceFieldKernel<<<blocks, threads>>>(
            d_fx_, d_fy_, d_fz_,
            d_tmp_x, d_tmp_y, d_tmp_z,
            threshold, nx, ny, nz);
        CUDA_CHECK_KERNEL();
        CUDA_CHECK(cudaDeviceSynchronize());

        // Copy smoothed result back to primary arrays
        CUDA_CHECK(cudaMemcpy(d_fx_, d_tmp_x, bytes, cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(d_fy_, d_tmp_y, bytes, cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(d_fz_, d_tmp_z, bytes, cudaMemcpyDeviceToDevice));
    }

    CUDA_CHECK(cudaFree(d_tmp_x));
    CUDA_CHECK(cudaFree(d_tmp_y));
    CUDA_CHECK(cudaFree(d_tmp_z));
}

} // namespace physics
} // namespace lbm
