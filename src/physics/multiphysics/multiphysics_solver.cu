/**
 * @file multiphysics_solver.cu
 * @brief Implementation of MultiphysicsSolver
 */

#include "physics/multiphysics_solver.h"
#include <cuda_runtime.h>
#include <iostream>
#include <iomanip>
#include <vector>
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include "utils/cuda_check.h"

namespace lbm {
namespace physics {

// ============================================================================
// CUDA Kernels
// ============================================================================

/**
 * @brief Find maximum value in array (reduction)
 */
__global__ void findMaxKernel(const float* data, float* result, int n) {
    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Load data into shared memory
    sdata[tid] = (idx < n) ? data[idx] : -INFINITY;
    __syncthreads();

    // Reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }

    // Write result for this block
    if (tid == 0) {
        atomicMax((int*)result, __float_as_int(sdata[0]));
    }
}

/**
 * @brief Compute velocity magnitude
 */
__global__ void computeVelocityMagnitudeKernel(
    const float* ux, const float* uy, const float* uz,
    float* vmag, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    float u = ux[idx];
    float v = uy[idx];
    float w = uz[idx];
    vmag[idx] = sqrtf(u*u + v*v + w*w);
}

/**
 * @brief Freeze velocity in solid region (T < T_solidus)
 *
 * Prevents LBM acoustic shockwaves from advecting the VOF free surface
 * in the cold solid ahead of the melt pool. The Darcy penalty only
 * DAMPS velocity; this kernel ZEROES it, making the solid truly rigid.
 */
static __global__ void freezeSolidVelocityKernel(
    float* __restrict__ vx,
    float* __restrict__ vy,
    float* __restrict__ vz,
    const float* __restrict__ temperature,
    const float* __restrict__ fill_level,
    float T_solidus, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    // Only freeze METAL cells that are below solidus.
    // Gas (f < 0.05) is always free to move — air is not solid!
    if (fill_level != nullptr && fill_level[idx] < 0.05f) return;
    if (temperature[idx] < T_solidus) {
        vx[idx] = 0.0f;
        vy[idx] = 0.0f;
        vz[idx] = 0.0f;
    }
}

/**
 * @brief Mask forces in pure-solid cells; do NOT scale forces in mushy zone.
 *
 * Cold powder (fl ≈ 0) must feel no surface tension or Marangoni — without
 * this, the cold-powder layer creeps under spurious CSF forces.
 *
 * BUG FIX (Sprint-1, 2026-04-25): old version multiplied by fl, which
 * compounded with the Marangoni kernel's own fl-gate (force_accumulator.cu:326)
 * and gave 0.7 × 0.7 ≈ 0.49 attenuation on partial-melt cells (fl=0.7).
 * That structurally suppressed Marangoni at the very interface where it should
 * be strongest, contributing 10–25 % melt-pool width error vs Flow3D.
 *
 * Each force kernel is now responsible for its own mushy-zone treatment
 * (Marangoni: ramp 0.1→0.2 inside the kernel; CSF: ∇f naturally limits to
 * interface; buoyancy: weighted by fl internally). This mask becomes a
 * pure cold-powder cutoff at fl = 0.05.
 */
static __global__ void maskForceByLiquidFractionKernel(
    float* __restrict__ fx, float* __restrict__ fy, float* __restrict__ fz,
    const float* __restrict__ liquid_fraction, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float fl = liquid_fraction[idx];
    // Step cutoff: only kill forces in the (effectively) cold-solid phase.
    // Above 0.05 fl, leave forces untouched — let each kernel's own gate decide.
    float gate = (fl >= 0.05f) ? 1.0f : 0.0f;
    fx[idx] *= gate;
    fy[idx] *= gate;
    fz[idx] *= gate;
}

/**
 * @brief Convert velocity from lattice units to physical units [m/s]
 *
 * LBM velocity is dimensionless (lattice units), typically O(0.01-0.1)
 * Physical velocity: v_phys = v_lattice * (dx / dt)
 *
 * This conversion is required for VOF advection, which expects [m/s]
 */
__global__ void convertVelocityToPhysicalUnitsKernel(
    const float* ux_lattice, const float* uy_lattice, const float* uz_lattice,
    float* ux_physical, float* uy_physical, float* uz_physical,
    float conversion_factor, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    ux_physical[idx] = ux_lattice[idx] * conversion_factor;
    uy_physical[idx] = uy_lattice[idx] * conversion_factor;
    uz_physical[idx] = uz_lattice[idx] * conversion_factor;
}

/**
 * @brief Check for NaN or Inf
 */
__global__ void checkNaNKernel(const float* data, int* has_nan, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    if (isnan(data[idx]) || isinf(data[idx])) {
        atomicAdd(has_nan, 1);
    }
}

/**
 * @brief 27-point (Moore neighborhood) isotropic smoothing kernel
 *
 * Full 3×3×3 stencil with isotropic weights:
 *   Center (1):      w=8/64
 *   Face neighbors (6):  w=4/64 each
 *   Edge neighbors (12): w=2/64 each
 *   Corner neighbors (8): w=1/64 each
 *   Total: 8 + 6×4 + 12×2 + 8×1 = 64/64 = 1 (normalized)
 *
 * This weight distribution recovers the isotropic Laplacian to O(h²),
 * compensating for D3Q7's directional bias in thermal diffusion.
 */
static __global__ void smoothField27Kernel(
    const float* __restrict__ src,
    float* __restrict__ dst,
    int nx, int ny, int nz)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    if (i >= nx || j >= ny || k >= nz) return;

    int idx = i + nx * (j + ny * k);

    // Boundary cells: pass through (zero-gradient implicitly)
    if (i == 0 || i == nx-1 || j == 0 || j == ny-1 || k == 0 || k == nz-1) {
        dst[idx] = src[idx];
        return;
    }

    // 27-point isotropic stencil
    float sum = 0.0f;

    // Center: weight 8
    sum += 8.0f * src[idx];

    // 6 face neighbors: weight 4 each
    sum += 4.0f * (src[(i-1) + nx*(j   + ny*k    )] +
                   src[(i+1) + nx*(j   + ny*k    )] +
                   src[i     + nx*((j-1)+ ny*k   )] +
                   src[i     + nx*((j+1)+ ny*k   )] +
                   src[i     + nx*(j    + ny*(k-1))] +
                   src[i     + nx*(j    + ny*(k+1))]);

    // 12 edge neighbors: weight 2 each
    sum += 2.0f * (src[(i-1) + nx*((j-1)+ ny*k    )] +
                   src[(i+1) + nx*((j-1)+ ny*k    )] +
                   src[(i-1) + nx*((j+1)+ ny*k    )] +
                   src[(i+1) + nx*((j+1)+ ny*k    )] +
                   src[(i-1) + nx*(j    + ny*(k-1))] +
                   src[(i+1) + nx*(j    + ny*(k-1))] +
                   src[(i-1) + nx*(j    + ny*(k+1))] +
                   src[(i+1) + nx*(j    + ny*(k+1))] +
                   src[i     + nx*((j-1)+ ny*(k-1))] +
                   src[i     + nx*((j+1)+ ny*(k-1))] +
                   src[i     + nx*((j-1)+ ny*(k+1))] +
                   src[i     + nx*((j+1)+ ny*(k+1))]);

    // 8 corner neighbors: weight 1 each
    sum += 1.0f * (src[(i-1) + nx*((j-1)+ ny*(k-1))] +
                   src[(i+1) + nx*((j-1)+ ny*(k-1))] +
                   src[(i-1) + nx*((j+1)+ ny*(k-1))] +
                   src[(i+1) + nx*((j+1)+ ny*(k-1))] +
                   src[(i-1) + nx*((j-1)+ ny*(k+1))] +
                   src[(i+1) + nx*((j-1)+ ny*(k+1))] +
                   src[(i-1) + nx*((j+1)+ ny*(k+1))] +
                   src[(i+1) + nx*((j+1)+ ny*(k+1))]);

    dst[idx] = sum / 64.0f;
}

/**
 * @brief Zero Darcy coefficient in gas cells (VOF fill < threshold)
 *
 * Without this, K = C·(1-fl) applies maximum penalty in the gas phase
 * (fl=0 in gas → K=C), turning the inert atmosphere into concrete and
 * trapping Marangoni flow sub-surface.
 */
static __global__ void darcyZeroInGasKernel(
    float* __restrict__ darcy_K,
    const float* __restrict__ fill_level,
    float threshold, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    if (fill_level[idx] < threshold)
        darcy_K[idx] = 0.0f;
}

/**
 * @brief Darcy under-relaxation kernel: K_new = α·K_computed + (1-α)·K_old
 */
static __global__ void darcyUnderRelaxKernel(
    float* __restrict__ K_current,
    const float* __restrict__ K_prev,
    float alpha, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    K_current[idx] = alpha * K_current[idx] + (1.0f - alpha) * K_prev[idx];
}

// TODO: These local kernels duplicate functionality in force_accumulator.cu.
// They exist because ForceAccumulator's versions are not linkable from this TU.
// Remove once ForceAccumulator exposes these via public API or shared header.

/**
 * @brief Zero out force array (local copy -- see force_accumulator.cu)
 */
static __global__ void zeroForceKernelLocal(float* fx, float* fy, float* fz, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    fx[idx] = 0.0f;
    fy[idx] = 0.0f;
    fz[idx] = 0.0f;
}

/**
 * @brief Compute force magnitude for diagnostics (local copy -- see force_accumulator.cu)
 */
static __global__ void computeForceMagnitudeKernelLocal(
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

/**
 * @brief Compute volumetric laser heat source
 * @param d_heat_source Output: volumetric heat source [W/m³]
 * @param laser Laser source object
 * @param nx, ny, nz Grid dimensions
 * @param dx Lattice spacing [m]
 * @param z_surface Z-coordinate of surface [lattice units]
 */
__global__ void computeLaserHeatSourceKernel(
    float* d_heat_source,
    const float* fill_level,
    const float* temperature,    // for plasma shielding cutoff
    LaserSource laser,
    int nx, int ny, int nz,
    float dx,
    float z_surface)
{
    // R7 COLUMN-MARCH (OpenFOAM laserMeltFoam updateFLB.H alignment):
    // Thread mapping: one thread per (i, j) column, marching top-down in Z.
    // Each metal-side VOF cell absorbs its own fraction `f` of the remaining
    // beam (`laserFraction`), with carry-over until the beam is depleted.
    //
    //   laserFraction = 1.0
    //   for k = nz-1 downto 0:
    //     if f[k] > 0.01:
    //       absorbed  = min(laserFraction, f[k])
    //       Q_vol[k]  = q_surface * absorbed / dx    [W/m³]
    //       laserFraction -= absorbed
    //       if laserFraction <= 0: break
    //
    // This is EXACTLY conservative: Σ (Q_vol·dx) = q_surface·(1 - remainder).
    // No |∂f/∂z| singularity, no discrete integral deficit — the absorption
    // is exactly ∫α(z)dz integrated cell-by-cell.
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k_init = blockIdx.z * blockDim.z + threadIdx.z;

    if (i >= nx || j >= ny || k_init >= nz) return;
    // 3-D grid repurposed: only the k=0 slice does real column work.
    // Caller MUST cudaMemset the buffer to zero before launching this kernel
    // (MultiphysicsSolver::applyLaserSource does so at line ~1595).
    if (k_init != 0) return;

    float x = i * dx;
    float y = j * dx;

    // Surface flux q_surface = η · (2P/πr₀²) · exp(-2r²/r₀²)     [W/m²]
    // = absorptivity × computeIntensity(x,y)
    float q_surface = laser.absorptivity * laser.computeIntensity(x, y);
    if (q_surface <= 0.0f) return;

    if (fill_level == nullptr) {
        // Legacy path: no VOF. Deposit at z_surface band, 1/dx volumetric factor.
        int k_surf = (int)z_surface;
        if (k_surf >= 0 && k_surf < nz) {
            int idx = i + nx * (j + ny * k_surf);
            d_heat_source[idx] = q_surface / dx;
        }
        return;
    }

    // Column-march top-down (+Z is "up" / laser comes from above).
    float laserFraction = 1.0f;
    for (int k = nz - 1; k >= 0; --k) {
        int idx = i + nx * (j + ny * k);
        float f = fill_level[idx];

        // Skip gas / near-vacuum cells; they don't absorb.
        if (f < 0.01f) continue;

        // Optional plasma shield: if this (metal-interface) cell is already
        // vaporizing, it shields deeper cells. Linear ramp 3300–3800 K.
        float shield = 1.0f;
        if (temperature != nullptr) {
            float T_local = temperature[idx];
            if (T_local > 3800.0f) shield = 0.0f;
            else if (T_local > 3300.0f)
                shield = 1.0f - (T_local - 3300.0f) / 500.0f;
        }

        float absorbed = fminf(laserFraction, f);
        // Volumetric source: q_surface [W/m²] × absorbed-fraction / dx = W/m³
        d_heat_source[idx] = q_surface * absorbed * shield / dx;

        laserFraction -= absorbed;  // always deplete (shielded energy is "lost")
        if (laserFraction <= 1e-6f) break;
    }
}

/**
 * @brief Convert volumetric force [N/m³] to lattice units
 * F_lattice = F_physical * (dt² / dx)
 *
 * This is the correct conversion for FluidLBM which works in lattice units.
 * The Guo forcing scheme internally handles the density division.
 */
__global__ void convertForceToLatticeUnitsKernel(
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
 * @brief CFL-based force limiter to prevent numerical divergence
 *
 * STABILITY FIX: Limits forces such that resulting velocities don't violate CFL condition
 *
 * Problem: Marangoni forces can cause velocity to explode from 1 m/s → 600 km/s in a few timesteps
 * Root cause: Strong temperature gradients (up to 1e6 K/m) create large surface tension forces
 *             that accelerate flow faster than diffusion can stabilize it (CFL >> 1)
 *
 * Solution: Predict velocity after force application and scale back forces if CFL would exceed limit
 *
 * Physical justification: Real materials cannot accelerate arbitrarily fast due to:
 *   - Viscous dissipation (always present)
 *   - Inertia (momentum diffusion timescale)
 *   - Interface dynamics (finite interface response time)
 *
 * This limiter enforces these physical constraints numerically.
 *
 * @param fx, fy, fz Force components [lattice units] - will be modified in-place
 * @param ux, uy, uz Current velocity components [lattice units]
 * @param dt Timestep [physical units, seconds]
 * @param dx Lattice spacing [physical units, meters]
 * @param max_CFL Maximum allowed CFL number (typically 0.3-0.5)
 * @param n Number of cells
 */
__global__ void limitForcesByCFL_kernel(
    float* fx, float* fy, float* fz,
    const float* ux, const float* uy, const float* uz,
    float dt, float dx, float max_CFL,
    int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    // Current velocity [lattice units: dimensionless, O(0.1)]
    float vx = ux[idx];
    float vy = uy[idx];
    float vz = uz[idx];

    // Forces are already in lattice units after convertForceToLatticeUnitsKernel
    // In lattice units: dv = F * dt_lattice = F * 1 = F (since dt_lattice=1)
    // So the velocity update is simply: v_new = v_old + F
    float v_new_x = vx + fx[idx];
    float v_new_y = vy + fy[idx];
    float v_new_z = vz + fz[idx];

    float v_new_mag_lattice = sqrtf(v_new_x*v_new_x + v_new_y*v_new_y + v_new_z*v_new_z);

    // Convert to physical velocity for CFL check
    // v_phys = v_lattice * (dx / dt)
    float v_new_mag_phys = v_new_mag_lattice * (dx / dt);

    // Compute CFL number
    float CFL = v_new_mag_phys * dt / dx;

    // If CFL exceeds limit, scale back forces
    if (CFL > max_CFL) {
        float scale = max_CFL / CFL;

        fx[idx] *= scale;
        fy[idx] *= scale;
        fz[idx] *= scale;
    }
}

/**
 * @brief Improved CFL limiter with gradual force scaling for better surface deformation
 *
 * This kernel provides a more nuanced approach to force limiting that allows
 * stronger deformation while maintaining numerical stability.
 *
 * Key improvements over the basic limiter:
 * 1. Uses target lattice velocity instead of CFL number (clearer semantics)
 * 2. Gradual scaling with smooth transition (no sudden force cutoff)
 * 3. Allows higher forces when velocity is still low (important for initial deformation)
 * 4. Considers force direction relative to current velocity
 *
 * Physics:
 *   - At t=0, velocity is ~0, so forces can be larger
 *   - As velocity builds up, gradually reduce force to maintain stability
 *   - Smooth scaling prevents discontinuous acceleration
 *
 * @param fx, fy, fz Force components [lattice units] - will be modified in-place
 * @param ux, uy, uz Current velocity components [lattice units]
 * @param v_target Target maximum lattice velocity (typically 0.1-0.15)
 * @param ramp_factor Fraction of v_target where scaling begins (0.5-0.9)
 * @param n Number of cells
 */
__global__ void limitForcesGradual_kernel(
    float* fx, float* fy, float* fz,
    const float* ux, const float* uy, const float* uz,
    float v_target,
    float ramp_factor,
    int n)
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

    if (f_mag < 1e-12f) return;  // No force to limit

    // Predicted velocity after force application: v_new = v + F (in lattice units)
    float v_new_x = vx + f_x;
    float v_new_y = vy + f_y;
    float v_new_z = vz + f_z;
    float v_new = sqrtf(v_new_x*v_new_x + v_new_y*v_new_y + v_new_z*v_new_z);

    // Compute scaling based on where we are relative to target
    // v_ramp = ramp_factor * v_target is where we start limiting
    float v_ramp = ramp_factor * v_target;

    float scale = 1.0f;

    if (v_new > v_target) {
        // Hard limit: cannot exceed v_target
        // Scale force to reach exactly v_target
        // v_new = |v + s*F| = v_target
        // Approximate: s = (v_target - v_current) / f_mag (if moving in force direction)
        float delta_v_allowed = fmaxf(0.0f, v_target - v_current);
        scale = delta_v_allowed / (f_mag + 1e-12f);
        scale = fminf(scale, 1.0f);  // Never amplify forces
    }
    else if (v_new > v_ramp) {
        // Gradual scaling zone: smooth transition from 1.0 to reduced value
        // Linear interpolation: scale = 1 - (v_new - v_ramp) / (v_target - v_ramp) * (1 - scale_min)
        // But simpler: just use the ratio
        float excess_ratio = (v_new - v_ramp) / (v_target - v_ramp + 1e-12f);
        float scale_at_target = (v_target - v_current) / (f_mag + 1e-12f);
        scale_at_target = fminf(scale_at_target, 1.0f);

        // Smooth interpolation between 1.0 and scale_at_target
        scale = 1.0f - excess_ratio * (1.0f - scale_at_target);
        scale = fmaxf(scale, 0.01f);  // Never completely zero out force
    }
    // else: v_new <= v_ramp, no scaling needed (scale = 1.0)

    // Apply scaling
    fx[idx] = f_x * scale;
    fy[idx] = f_y * scale;
    fz[idx] = f_z * scale;
}

/**
 * @brief Adaptive CFL limiter with region-based velocity targets for keyhole simulation
 *
 * This kernel implements a more sophisticated CFL limiting strategy that allows
 * different velocity limits in different regions of the domain:
 *
 * Region classification (based on VOF fill level and liquid fraction):
 *   1. Interface cells (0.01 < fill < 0.99): Highest velocity allowed
 *      - Strong recoil pressure needs high velocity tolerance
 *      - v_target = v_target_interface (default 0.5 = ~10 m/s)
 *
 *   2. Bulk liquid (fill > 0.99, liquid_fraction > 0.5): Moderate velocity
 *      - Marangoni and convection flows
 *      - v_target = v_target_bulk (default 0.3 = ~6 m/s)
 *
 *   3. Solid region (fill > 0.99, liquid_fraction < 0.01): Zero velocity
 *      - Force must be zeroed out
 *      - v_target = 0
 *
 *   4. Gas region (fill < 0.01): Minimal forcing
 *      - Gas cells should have minimal forces
 *      - v_target = 0.1 (small, for numerical stability)
 *
 * Special handling for recoil pressure:
 *   - If force is predominantly in z-direction (recoil signature)
 *   - And cell is at interface
 *   - Allow extra boost factor (default 1.5x) to v_target
 *
 * @param fx, fy, fz Force components [lattice units] - modified in-place
 * @param ux, uy, uz Current velocity components [lattice units]
 * @param fill_level VOF fill level field [0-1]
 * @param liquid_fraction Phase field [0=solid, 1=liquid]
 * @param v_target_interface Target velocity for interface cells
 * @param v_target_bulk Target velocity for bulk liquid cells
 * @param interface_lo Lower fill level threshold for interface (default 0.01)
 * @param interface_hi Upper fill level threshold for interface (default 0.99)
 * @param recoil_boost_factor Extra allowance for z-dominant forces (default 1.5)
 * @param ramp_factor Gradual ramp factor (fraction of v_target where scaling begins)
 * @param n Number of cells
 */
__global__ void limitForcesByCFL_AdaptiveKernel(
    float* fx, float* fy, float* fz,
    const float* ux, const float* uy, const float* uz,
    const float* fill_level,
    const float* liquid_fraction,
    float v_target_interface,
    float v_target_bulk,
    float interface_lo,
    float interface_hi,
    float recoil_boost_factor,
    float ramp_factor,
    int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    // Read fill level and liquid fraction
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

    if (f_mag < 1e-12f) return;  // No force to limit

    // ========================================================================
    // Region classification and target velocity assignment
    // ========================================================================
    float v_target;

    bool is_interface = (fill > interface_lo) && (fill < interface_hi);
    bool is_gas = (fill <= interface_lo);
    bool is_solid = (fill >= interface_hi) && (liq_frac < 0.01f);
    bool is_mushy = (fill >= interface_hi) && (liq_frac >= 0.01f) && (liq_frac < 0.5f);
    // Bulk liquid: fill >= interface_hi && liq_frac >= 0.5 (default case in else)

    if (is_solid) {
        // Solid region: force must be zero
        fx[idx] = 0.0f;
        fy[idx] = 0.0f;
        fz[idx] = 0.0f;
        return;
    }
    else if (is_gas) {
        // Gas region: minimal velocity allowed
        v_target = 0.1f;
    }
    else if (is_interface) {
        // Interface region: high velocity for recoil pressure
        v_target = v_target_interface;

        // Check for recoil pressure signature (z-dominant force)
        // Recoil pressure acts primarily in negative z direction (into the material)
        float fz_ratio = fabsf(f_z) / (f_mag + 1e-12f);
        if (fz_ratio > 0.7f) {
            // Force is predominantly z-directed (recoil pressure signature)
            // Allow extra boost for keyhole formation
            v_target *= recoil_boost_factor;
        }
    }
    else if (is_mushy) {
        // Mushy zone: reduced velocity (partial Darcy damping)
        v_target = v_target_bulk * liq_frac;  // Scale with liquid fraction
    }
    else {  // is_bulk_liquid
        // Bulk liquid: moderate velocity
        v_target = v_target_bulk;
    }

    // ========================================================================
    // Gradual force scaling (same algorithm as limitForcesGradual_kernel)
    // ========================================================================

    // Predicted velocity after force application
    float v_new_x = vx + f_x;
    float v_new_y = vy + f_y;
    float v_new_z = vz + f_z;
    float v_new = sqrtf(v_new_x*v_new_x + v_new_y*v_new_y + v_new_z*v_new_z);

    // Ramping threshold
    float v_ramp = ramp_factor * v_target;

    float scale = 1.0f;

    if (v_new > v_target) {
        // Hard limit: cannot exceed v_target
        float delta_v_allowed = fmaxf(0.0f, v_target - v_current);
        scale = delta_v_allowed / (f_mag + 1e-12f);
        scale = fminf(scale, 1.0f);  // Never amplify forces
    }
    else if (v_new > v_ramp) {
        // Gradual scaling zone
        float excess_ratio = (v_new - v_ramp) / (v_target - v_ramp + 1e-12f);
        float scale_at_target = (v_target - v_current) / (f_mag + 1e-12f);
        scale_at_target = fminf(scale_at_target, 1.0f);

        // Smooth interpolation between 1.0 and scale_at_target
        scale = 1.0f - excess_ratio * (1.0f - scale_at_target);
        scale = fmaxf(scale, 0.01f);  // Never completely zero out force
    }
    // else: v_new <= v_ramp, no scaling needed (scale = 1.0)

    // Apply scaling
    fx[idx] = f_x * scale;
    fy[idx] = f_y * scale;
    fz[idx] = f_z * scale;
}

// ============================================================================
// MultiphysicsConfig::validate()
// ============================================================================

void MultiphysicsConfig::validate() const {
    // ---- Grid validation ----
    if (nx <= 0 || ny <= 0 || nz <= 0) {
        throw std::runtime_error("MultiphysicsConfig: grid dimensions must be positive "
            "(nx=" + std::to_string(nx) + ", ny=" + std::to_string(ny) +
            ", nz=" + std::to_string(nz) + ")");
    }
    if (dx <= 0.0f) {
        throw std::runtime_error("MultiphysicsConfig: dx must be positive (dx=" +
            std::to_string(dx) + ")");
    }
    if (dt <= 0.0f) {
        throw std::runtime_error("MultiphysicsConfig: dt must be positive (dt=" +
            std::to_string(dt) + ")");
    }

    // ---- LBM stability: fluid tau ----
    // kinematic_viscosity is in LATTICE units (historical convention)
    float tau_fluid = kinematic_viscosity / (1.0f / 3.0f) + 0.5f;
    if (tau_fluid < 0.51f) {
        fprintf(stderr, "WARNING: tau_fluid=%.4f < 0.51 -- LBM is likely UNSTABLE "
                "(nu_lattice=%.4f)\n", tau_fluid, kinematic_viscosity);
    }
    if (tau_fluid > 5.0f) {
        fprintf(stderr, "WARNING: tau_fluid=%.2f > 5.0 -- very diffusive, "
                "accuracy may be poor\n", tau_fluid);
    }

    // ---- LBM stability: thermal tau ----
    if (enable_thermal && dx > 0.0f && dt > 0.0f) {
        float alpha_lattice = thermal_diffusivity * dt / (dx * dx);
        // D3Q7 thermal LBM: tau_T = alpha_lattice / (1/4) + 0.5
        float tau_thermal = alpha_lattice / 0.25f + 0.5f;
        if (tau_thermal < 0.51f) {
            fprintf(stderr, "WARNING: tau_thermal=%.4f < 0.51 -- thermal LBM may be UNSTABLE "
                    "(alpha_lattice=%.4e)\n", tau_thermal, alpha_lattice);
        }
    }

    // ---- CFL limit ----
    if (cfl_limit > 0.577f) {
        fprintf(stderr, "WARNING: cfl_limit=%.3f exceeds LBM stability bound 1/sqrt(3)=0.577\n",
                cfl_limit);
    }

    // ---- Physics consistency ----
    if (enable_phase_change && !enable_thermal) {
        throw std::runtime_error("MultiphysicsConfig: enable_phase_change requires enable_thermal");
    }
    if (enable_evaporation_mass_loss && (!enable_vof || !enable_thermal)) {
        throw std::runtime_error("MultiphysicsConfig: enable_evaporation_mass_loss "
                "requires both enable_vof and enable_thermal");
    }
    // Marangoni without VOF uses Inamuro stress BC at z_max wall (no normals needed).
    // Only the CSF path (with VOF) requires interface normals.
    if (enable_recoil_pressure && (!enable_vof || !enable_thermal)) {
        throw std::runtime_error("MultiphysicsConfig: enable_recoil_pressure "
                "requires both enable_vof and enable_thermal");
    }
    if (enable_solidification_shrinkage && !enable_phase_change) {
        throw std::runtime_error("MultiphysicsConfig: enable_solidification_shrinkage "
                "requires enable_phase_change");
    }
    if (enable_thermal_advection && (!enable_thermal || !enable_fluid)) {
        throw std::runtime_error("MultiphysicsConfig: enable_thermal_advection "
                "requires both enable_thermal and enable_fluid");
    }
    if (enable_vof_advection && (!enable_vof || !enable_fluid)) {
        throw std::runtime_error("MultiphysicsConfig: enable_vof_advection "
                "requires both enable_vof and enable_fluid");
    }
    if (enable_surface_tension && !enable_vof) {
        throw std::runtime_error("MultiphysicsConfig: enable_surface_tension "
                "requires enable_vof");
    }

    // ---- Parameter ranges ----
    if (laser_absorptivity < 0.0f || laser_absorptivity > 1.0f) {
        throw std::runtime_error("MultiphysicsConfig: laser_absorptivity must be in [0,1] "
            "(got " + std::to_string(laser_absorptivity) + ")");
    }
    if (emissivity < 0.0f || emissivity > 1.0f) {
        throw std::runtime_error("MultiphysicsConfig: emissivity must be in [0,1] "
            "(got " + std::to_string(emissivity) + ")");
    }
    if (enable_surface_tension && surface_tension_coeff <= 0.0f) {
        throw std::runtime_error("MultiphysicsConfig: surface_tension_coeff must be > 0 "
            "when surface tension is enabled (got " +
            std::to_string(surface_tension_coeff) + ")");
    }
    if (density <= 0.0f) {
        throw std::runtime_error("MultiphysicsConfig: density must be positive "
            "(got " + std::to_string(density) + ")");
    }
    if (vof_subcycles < 1) {
        throw std::runtime_error("MultiphysicsConfig: vof_subcycles must be >= 1 "
            "(got " + std::to_string(vof_subcycles) + ")");
    }

    // ---- Core parameter ranges ----
    if (fluid.kinematic_viscosity <= 0.0f) {
        throw std::runtime_error("MultiphysicsConfig: kinematic_viscosity must be positive "
            "(got " + std::to_string(fluid.kinematic_viscosity) + ")");
    }
    if (physics.enable_thermal && thermal.thermal_diffusivity <= 0.0f) {
        throw std::runtime_error("MultiphysicsConfig: thermal_diffusivity must be positive "
            "when thermal enabled (got " + std::to_string(thermal.thermal_diffusivity) + ")");
    }
    if (physics.enable_laser && laser.spot_radius <= 0.0f) {
        throw std::runtime_error("MultiphysicsConfig: laser spot_radius must be positive "
            "(got " + std::to_string(laser.spot_radius) + ")");
    }
    if (physics.enable_laser && laser.penetration_depth <= 0.0f) {
        throw std::runtime_error("MultiphysicsConfig: laser penetration_depth must be positive "
            "(got " + std::to_string(laser.penetration_depth) + ")");
    }
    if (numerics.cfl_velocity_target <= 0.0f) {
        throw std::runtime_error("MultiphysicsConfig: cfl_velocity_target must be positive "
            "(got " + std::to_string(numerics.cfl_velocity_target) + ")");
    }

    // ---- Material validation ----
    if (!material.validate()) {
        throw std::runtime_error("MultiphysicsConfig: material properties failed validation "
                "(check densities, specific heats, and temperatures)");
    }
}

// ============================================================================
// Constructor / Destructor
// ============================================================================

MultiphysicsSolver::MultiphysicsSolver(const MultiphysicsConfig& config)
    : config_(config),
      unit_converter_(config.dx, config.dt, config.density),
      d_force_x_(nullptr),
      d_force_y_(nullptr),
      d_force_z_(nullptr),
      d_temperature_static_(nullptr),
      d_liquid_fraction_static_(nullptr),
      d_velocity_physical_x_(nullptr),
      d_velocity_physical_y_(nullptr),
      d_velocity_physical_z_(nullptr),
      d_evap_mass_flux_(nullptr),
      current_time_(0.0f),
      interface_z_(static_cast<float>(config.nz - 1)),  // Default: surface at top
      initial_mass_(0.0f),
      initial_temperature_(300.0f),  // Default: room temperature
      previous_thermal_energy_(0.0f),
      previous_time_(0.0f),
      current_step_(0),
      d_energy_temp_(nullptr),
      energy_output_interval_(default_energy_interval_),
      time_last_computed_(-1.0f),  // Initialize to negative to force first computation
      laser_energy_accumulated_(0.0)
{
    // Ensure density consistency: use material's liquid density
    config_.fluid.density = config_.material.rho_liquid;

    // Validate configuration (comprehensive checks for stability, consistency, ranges)
    config_.validate();

    // Print derived lattice parameters for unit-conversion verification
    {
        float tau_fluid = config_.kinematic_viscosity / (1.0f / 3.0f) + 0.5f;
        float tau_thermal = 0.0f;
        if (config_.enable_thermal && config_.dx > 0.0f && config_.dt > 0.0f) {
            float alpha_lattice = config_.thermal_diffusivity * config_.dt / (config_.dx * config_.dx);
            tau_thermal = alpha_lattice / 0.25f + 0.5f;
        }
        printf("LBM Parameters: tau_fluid=%.3f, tau_thermal=%.3f, dx=%.2e m, dt=%.2e s\n",
               tau_fluid, tau_thermal, config_.dx, config_.dt);
    }

    // ============================================================
    // BUG 3 FIX (Solution 1): Adaptive diagnostic interval
    // ============================================================
    // Problem: Fixed diagnostic_interval=10 causes different physical
    //          time intervals for different dt, amplifying noise
    // Solution: Scale interval to maintain ~5 μs between diagnostics
    //
    // Examples:
    //   dt=0.20μs → interval=25 → Δt=5.0μs
    //   dt=0.10μs → interval=50 → Δt=5.0μs
    //   dt=0.05μs → interval=100 → Δt=5.0μs
    // ============================================================
    const float TARGET_DIAGNOSTIC_INTERVAL_SECONDS = 0.5e-6f;  // 0.5 μs (R6: high resolution for energy balance)
    diagnostic_interval_ = std::max(1, static_cast<int>(
        TARGET_DIAGNOSTIC_INTERVAL_SECONDS / config_.dt
    ));

    // Reserve space for energy history (time-averaged dE/dt)
    energy_history_.reserve(ENERGY_HISTORY_SIZE);
    time_history_.reserve(ENERGY_HISTORY_SIZE);

    // Allocate device memory
    allocateMemory();

    // Initialize physics modules based on flags

    // Thermal solver: runtime selection between D3Q7 LBM and explicit FDM
    if (config_.enable_thermal) {
        if (config_.use_fdm_thermal) {
            thermal_ = std::make_unique<ThermalFDM>(
                config_.nx, config_.ny, config_.nz,
                config_.material,
                config_.thermal_diffusivity,
                config_.enable_phase_change,
                config_.dt, config_.dx
            );
        } else {
            thermal_ = std::make_unique<ThermalLBM>(
                config_.nx, config_.ny, config_.nz,
                config_.material,
                config_.thermal_diffusivity,
                config_.enable_phase_change,
                config_.dt, config_.dx
            );
        }
        thermal_->setEmissivity(config_.emissivity);
    } else {
        // Allocate static temperature field
        int num_cells = config_.nx * config_.ny * config_.nz;
        CUDA_CHECK(cudaMalloc(&d_temperature_static_, num_cells * sizeof(float)));
    }

    // Fluid solver (required)
    if (config_.enable_fluid) {
        // BUG FIX (2026-01-25): Use config.kinematic_viscosity, not material property
        // config.kinematic_viscosity is in LATTICE UNITS (dimensionless)
        // FluidLBM expects PHYSICAL viscosity [m²/s]
        // Conversion: nu_physical = nu_lattice * dx² / dt
        //
        // Previous bug: Was computing nu_physical from material properties, which gave
        // tau=0.512 (UNSTABLE). Config allowed overriding to tau=0.65 (STABLE), but
        // this override was being ignored!
        float nu_lattice = config_.kinematic_viscosity;  // Lattice units (dimensionless)
        float nu_physical = nu_lattice * (config_.dx * config_.dx) / config_.dt;  // Convert to m²/s

        // Verify tau for stability
        float tau_check = 0.5f + 3.0f * nu_lattice;
        if (tau_check < 0.55f) {
            std::cerr << "WARNING: FluidLBM tau=" << tau_check << " < 0.55 (UNSTABLE!)\n";
            std::cerr << "         Config kinematic_viscosity=" << nu_lattice << " (lattice)\n";
        }

        fluid_ = std::make_unique<FluidLBM>(
            config_.nx, config_.ny, config_.nz,
            nu_physical,             // Physical kinematic viscosity [m²/s] converted from config
            config_.material.rho_liquid,  // Liquid density [kg/m³]
            config_.boundaries.fluidBCX(),  // Per-face -> per-axis fluid BC
            config_.boundaries.fluidBCY(),
            config_.boundaries.fluidBCZ(),
            config_.dt,              // Time step for unit conversion
            config_.dx               // Lattice spacing for unit conversion
        );

        // Enable TRT collision to eliminate checkerboard instability at low tau.
        // BGK at tau < 0.6 produces odd-even decoupling (ω > 1.67 overshoots
        // anti-symmetric modes). TRT uses a separate ω- for these modes.
        // Λ = 3/16 gives optimal wall boundary accuracy.
        fluid_->setTRT(3.0f / 16.0f);
    }

    // VOF solver (required for interface tracking)
    if (config_.enable_vof) {
        vof_ = std::make_unique<VOFSolver>(
            config_.nx, config_.ny, config_.nz,
            config_.dx,
            config_.boundaries.vofBCX(),
            config_.boundaries.vofBCY(),
            config_.boundaries.vofBCZ()
        );

        // PLIC geometric advection: sharp interface (1 cell width),
        // exactly mass-conservative. Replaces default UPWIND which is
        // first-order diffusive and smears the interface over 15+ cells.
        vof_->setAdvectionScheme(VOFAdvectionScheme::PLIC);

        // Enable mass conservation correction to prevent 90% mass loss
        // This redistributes numerical mass loss back to interface cells
        if (config_.enable_vof_mass_correction) {
            vof_->setMassConservationCorrection(true, config_.vof_mass_correction_damping);
            vof_->setMassCorrectionUseFluxWeight(config_.vof_mass_correction_use_flux_weight);

            // Track-C Gate 2: set substrate height once at init (fixed geometry).
            // Gate 1 (laser x) is updated per-step in step() via setMassCorrectionLaserX().
            if (config_.mass_correction_use_track_c &&
                config_.mass_correction_z_substrate_lu >= 0.0f) {
                vof_->setMassCorrectionZSubstrate(config_.mass_correction_z_substrate_lu,
                                                  config_.mass_correction_z_offset_lu);
            }

            const char* track_name = "A-vz";
            if (config_.vof_mass_correction_use_flux_weight) {
                track_name = config_.mass_correction_use_track_c ? "C-flux+gates" : "B-flux";
            }
            std::cout << "  VOF mass correction: ENABLED (damping="
                      << config_.vof_mass_correction_damping
                      << ", track=" << track_name
                      << ")" << std::endl;
        }
    }

    // Surface tension (optional - add in Step 3)
    if (config_.enable_surface_tension) {
        surface_tension_ = std::make_unique<SurfaceTension>(
            config_.nx, config_.ny, config_.nz,
            config_.surface_tension_coeff,
            config_.dx
        );
    }

    // Marangoni effect (required)
    if (config_.enable_marangoni) {
        marangoni_ = std::make_unique<MarangoniEffect>(
            config_.nx, config_.ny, config_.nz,
            config_.dsigma_dT,
            config_.dx,
            2.0f  // Interface thickness = 2 cells
        );
    }

    // Laser source (optional - add in Step 4)
    if (config_.enable_laser) {
        laser_ = std::make_unique<LaserSource>(
            config_.laser_power,
            config_.laser_spot_radius,
            config_.laser_absorptivity,
            config_.laser_penetration_depth
        );

        // Position laser: use config parameters if specified, otherwise auto-center
        float x_start = (config_.laser_start_x >= 0.0f)
            ? config_.laser_start_x
            : (config_.nx * config_.dx * 0.5f);  // Auto: domain center
        float y_start = (config_.laser_start_y >= 0.0f)
            ? config_.laser_start_y
            : (config_.ny * config_.dx * 0.5f);  // Auto: domain center
        float z_surface = 0.0f;  // At surface (actual z set in kernel)

        laser_->setPosition(x_start, y_start, z_surface);

        // Set scan velocity from config
        laser_->setScanVelocity(config_.laser_scan_vx, config_.laser_scan_vy);

        // Ray tracing laser (optional - replaces Beer-Lambert projection)
        if (config_.ray_tracing.enabled) {
            ray_tracing_laser_ = std::make_unique<RayTracingLaser>(
                config_.ray_tracing,
                config_.nx, config_.ny, config_.nz, config_.dx
            );
        }
    }

    // Recoil pressure is applied via ForceAccumulator::addRecoilPressureForce (see below).
    // The standalone RecoilPressure class path (dead in production) was removed.

    // ============================================================
    // Initialize ForceAccumulator (robust force pipeline)
    // ============================================================
    force_accumulator_ = std::make_unique<ForceAccumulator>(
        config_.nx, config_.ny, config_.nz);

    // Point legacy force arrays to ForceAccumulator's internal arrays
    // This avoids unnecessary memory allocation and copying
    d_force_x_ = force_accumulator_->getFx();
    d_force_y_ = force_accumulator_->getFy();
    d_force_z_ = force_accumulator_->getFz();

    std::cout << "MultiphysicsSolver initialized:" << std::endl;
    std::cout << "  Grid: " << config_.nx << " x " << config_.ny
              << " x " << config_.nz << std::endl;
    std::cout << "  dx = " << config_.dx * 1e6 << " um" << std::endl;
    std::cout << "  dt = " << config_.dt * 1e6 << " us" << std::endl;
    std::cout << "  Diagnostic interval: " << diagnostic_interval_
              << " steps (" << (diagnostic_interval_ * config_.dt * 1e6) << " us)" << std::endl;
    std::cout << "  CFL limiter:" << std::endl;
    if (config_.cfl_use_adaptive) {
        std::cout << "    Mode: ADAPTIVE REGION-BASED (keyhole)" << std::endl;
        std::cout << "    v_target_interface = " << config_.cfl_v_target_interface << " (lattice) = "
                  << (config_.cfl_v_target_interface * config_.dx / config_.dt) << " m/s" << std::endl;
        std::cout << "    v_target_bulk = " << config_.cfl_v_target_bulk << " (lattice) = "
                  << (config_.cfl_v_target_bulk * config_.dx / config_.dt) << " m/s" << std::endl;
        std::cout << "    recoil_boost = " << config_.cfl_recoil_boost_factor << "x" << std::endl;
        std::cout << "    interface: " << config_.cfl_interface_threshold_lo << " < fill < "
                  << config_.cfl_interface_threshold_hi << std::endl;
    } else if (config_.cfl_use_gradual_scaling) {
        std::cout << "    Mode: GRADUAL SCALING" << std::endl;
        std::cout << "    v_target = " << config_.cfl_velocity_target << " (lattice) = "
                  << (config_.cfl_velocity_target * config_.dx / config_.dt) << " m/s" << std::endl;
        std::cout << "    ramp_factor = " << config_.cfl_force_ramp_factor << std::endl;
    } else {
        std::cout << "    Mode: HARD CFL LIMIT" << std::endl;
        std::cout << "    CFL_max = " << config_.cfl_limit << std::endl;
    }
    std::cout << "  Physics modules:" << std::endl;
    std::cout << "    Thermal:         " << (config_.enable_thermal ? "ON" : "OFF") << std::endl;
    std::cout << "    Fluid:           " << (config_.enable_fluid ? "ON" : "OFF") << std::endl;
    std::cout << "    VOF:             " << (config_.enable_vof ? "ON" : "OFF") << std::endl;
    std::cout << "    Surface Tension: " << (config_.enable_surface_tension ? "ON" : "OFF") << std::endl;
    std::cout << "    Marangoni:       " << (config_.enable_marangoni ? "ON" : "OFF") << std::endl;
    std::cout << "    Darcy Damping:   " << (config_.enable_darcy ? "ON" : "OFF") << std::endl;
    std::cout << "    Buoyancy:        " << (config_.enable_buoyancy ? "ON" : "OFF") << std::endl;
    std::cout << "    Evap Mass Loss:  " << (config_.enable_evaporation_mass_loss ? "ON" : "OFF") << std::endl;
    std::cout << "    Recoil Pressure: " << (config_.enable_recoil_pressure ? "ON" : "OFF") << std::endl;
    if (config_.enable_recoil_pressure) {
        std::cout << "      C_r = " << config_.recoil_coefficient << std::endl;
        std::cout << "      P_max = " << config_.recoil_max_pressure * 1e-6 << " MPa" << std::endl;
    }
    if (config_.enable_buoyancy) {
        std::cout << "      β = " << config_.thermal_expansion_coeff << " K⁻¹" << std::endl;
        std::cout << "      g = (" << config_.gravity_x << ", "
                  << config_.gravity_y << ", " << config_.gravity_z << ") m/s²" << std::endl;
        std::cout << "      T_ref = " << config_.reference_temperature << " K" << std::endl;
    }
    std::cout << "    Laser:           " << (config_.enable_laser ? "ON" : "OFF") << std::endl;
    if (config_.enable_laser && laser_) {
        std::cout << "      Power = " << config_.laser_power << " W" << std::endl;
        std::cout << "      Spot radius = " << config_.laser_spot_radius * 1e6 << " um" << std::endl;
        std::cout << "      Absorptivity = " << config_.laser_absorptivity << std::endl;
        std::cout << "      Config position: (" << config_.laser_start_x * 1e6 << ", "
                  << config_.laser_start_y * 1e6 << ") um" << std::endl;
        std::cout << "      Actual position: (" << laser_->x0 * 1e6 << ", "
                  << laser_->y0 * 1e6 << ") um" << std::endl;
        std::cout << "      Cell index: (" << (laser_->x0 / config_.dx) << ", "
                  << (laser_->y0 / config_.dx) << ")" << std::endl;
    }
}

MultiphysicsSolver::~MultiphysicsSolver() {
    freeMemory();
}

// ============================================================================
// Memory Management
// ============================================================================

void MultiphysicsSolver::allocateMemory() {
    int num_cells = config_.nx * config_.ny * config_.nz;

    // NOTE: Force arrays are now managed by ForceAccumulator
    // No need to allocate d_force_x_, d_force_y_, d_force_z_ here

    // Allocate static liquid fraction field
    CUDA_CHECK(cudaMalloc(&d_liquid_fraction_static_, num_cells * sizeof(float)));

    // Allocate physical velocity buffers for VOF advection
    // VOF expects velocity in [m/s], but LBM outputs lattice units
    CUDA_CHECK(cudaMalloc(&d_velocity_physical_x_, num_cells * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_velocity_physical_y_, num_cells * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_velocity_physical_z_, num_cells * sizeof(float)));

    // Allocate energy reduction temporary (single double for global reduction)
    CUDA_CHECK(cudaMalloc(&d_energy_temp_, sizeof(double)));

    // Allocate evaporation mass flux buffer (for VOF-thermal coupling)
    CUDA_CHECK(cudaMalloc(&d_evap_mass_flux_, num_cells * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_evap_mass_flux_, 0, num_cells * sizeof(float)));

    // Allocate hydrodynamic smoothing buffers (mushy-zone stabilization)
    CUDA_CHECK(cudaMalloc(&d_fl_smoothed_, num_cells * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_T_smoothed_, num_cells * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_darcy_K_prev_, num_cells * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_fl_smoothed_, 0, num_cells * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_T_smoothed_, 0, num_cells * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_darcy_K_prev_, 0, num_cells * sizeof(float)));

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("Failed to allocate device memory: " +
                               std::string(cudaGetErrorString(err)));
    }

    // NOTE: Force arrays (d_force_x/y/z_) now point to ForceAccumulator's internal arrays
    // ForceAccumulator handles initialization via reset() called before each step
    // No need to cudaMemset them here

    // Initialize liquid fraction to 1.0 (fully liquid) by default
    std::vector<float> h_lf(num_cells, 1.0f);
    cudaMemcpy(d_liquid_fraction_static_, h_lf.data(),
              num_cells * sizeof(float), cudaMemcpyHostToDevice);
}

void MultiphysicsSolver::freeMemory() {
    // NOTE: Force arrays are now managed by ForceAccumulator destructor
    if (d_temperature_static_) cudaFree(d_temperature_static_);
    if (d_liquid_fraction_static_) cudaFree(d_liquid_fraction_static_);
    if (d_velocity_physical_x_) cudaFree(d_velocity_physical_x_);
    if (d_velocity_physical_y_) cudaFree(d_velocity_physical_y_);
    if (d_velocity_physical_z_) cudaFree(d_velocity_physical_z_);
    if (d_evap_mass_flux_) cudaFree(d_evap_mass_flux_);
    if (d_energy_temp_) cudaFree(d_energy_temp_);
    if (d_fl_smoothed_) cudaFree(d_fl_smoothed_);
    if (d_T_smoothed_) cudaFree(d_T_smoothed_);
    if (d_darcy_K_prev_) cudaFree(d_darcy_K_prev_);
}

// ============================================================================
// Initialization
// ============================================================================

void MultiphysicsSolver::initialize(float initial_temperature,
                                    float interface_height)
{
    int num_cells = config_.nx * config_.ny * config_.nz;

    // Store initial temperature for energy reference
    initial_temperature_ = initial_temperature;

    // Initialize temperature
    if (config_.enable_thermal && thermal_) {
        thermal_->initialize(initial_temperature);
    } else if (d_temperature_static_) {
        // Set uniform temperature in static field
        std::vector<float> h_temp(num_cells, initial_temperature);
        cudaMemcpy(d_temperature_static_, h_temp.data(),
                  num_cells * sizeof(float), cudaMemcpyHostToDevice);
    }

    // Initialize fluid (zero velocity)
    // CRITICAL: Use lattice density = 1.0, not physical density!
    // LBM operates in dimensionless units; physical density is only used
    // for unit conversions (e.g., in force conversion), not for distribution functions.
    if (fluid_) {
        fluid_->initialize(1.0f, 0.0f, 0.0f, 0.0f);
    }

    // Initialize VOF with planar interface
    if (vof_) {
        std::vector<float> h_fill(num_cells);
        int z_interface = static_cast<int>(interface_height * config_.nz);

        for (int k = 0; k < config_.nz; ++k) {
            for (int j = 0; j < config_.ny; ++j) {
                for (int i = 0; i < config_.nx; ++i) {
                    int idx = i + config_.nx * (j + config_.ny * k);

                    // Smooth interface with tanh profile
                    float z_dist = k - z_interface;
                    h_fill[idx] = 0.5f * (1.0f - tanhf(z_dist / 2.0f));
                    h_fill[idx] = std::max(0.0f, std::min(1.0f, h_fill[idx]));
                }
            }
        }

        vof_->initialize(h_fill.data());
        vof_->reconstructInterface();

        // Store initial mass for conservation check
        initial_mass_ = vof_->computeTotalMass();
    }

    current_time_ = 0.0f;

    // Store interface z position for laser surface detection
    // interface_z_ is in lattice units (cell index)
    // CRITICAL FIX: Clamp to valid range [0, nz-1] to prevent out-of-bounds
    // When VOF is disabled, interface_height may be 1.0, which maps to z=nz (out of bounds)
    // For laser melting from top, we want z = nz-1 (top cell)
    float z_interface_unclamped = interface_height * config_.nz;
    interface_z_ = fminf(z_interface_unclamped, static_cast<float>(config_.nz - 1));

    // Warn if clamping occurred
    if (z_interface_unclamped > config_.nz - 1) {
        std::cout << "  [WARNING] interface_height=" << interface_height
                  << " maps to z=" << z_interface_unclamped << " > nz-1=" << (config_.nz-1)
                  << ". Clamping to z=" << interface_z_ << " (top surface).\n";
    }

    // Register output fields now that all sub-solvers are initialized
    registerOutputFields();

    std::cout << "Multiphysics solver initialized:" << std::endl;
    std::cout << "  Initial temperature: " << initial_temperature << " K" << std::endl;
    std::cout << "  Interface height: " << interface_height << " (z = "
              << static_cast<int>(interface_z_) << ")" << std::endl;
    if (vof_) {
        std::cout << "  Initial mass: " << initial_mass_ << std::endl;
    }
}

void MultiphysicsSolver::initialize(const float* temperature_field,
                                    const float* fill_level_field)
{
    int num_cells = config_.nx * config_.ny * config_.nz;

    // Compute average temperature as reference for energy calculations
    float sum_T = 0.0f;
    for (int i = 0; i < num_cells; ++i) {
        sum_T += temperature_field[i];
    }
    initial_temperature_ = sum_T / num_cells;

    // Initialize temperature
    if (config_.enable_thermal && thermal_) {
        thermal_->initialize(temperature_field);
    } else if (d_temperature_static_) {
        cudaMemcpy(d_temperature_static_, temperature_field,
                  num_cells * sizeof(float), cudaMemcpyHostToDevice);
    }

    // Initialize fluid (zero velocity)
    // CRITICAL: Use lattice density = 1.0, not physical density!
    if (fluid_) {
        fluid_->initialize(1.0f, 0.0f, 0.0f, 0.0f);
    }

    // Initialize VOF
    if (vof_) {
        vof_->initialize(fill_level_field);
        vof_->reconstructInterface();
        initial_mass_ = vof_->computeTotalMass();
    }

    current_time_ = 0.0f;

    // Compute interface_z_ from the actual fill_level field
    // Find the highest z where fill_level transitions from >0.5 to <0.5
    // (the powder/gas boundary)
    {
        float z_max_interface = 0.0f;
        for (int k = 0; k < config_.nz; ++k) {
            for (int j = 0; j < config_.ny; ++j) {
                for (int i = 0; i < config_.nx; ++i) {
                    int idx = i + config_.nx * (j + config_.ny * k);
                    if (fill_level_field[idx] > 0.5f) {
                        z_max_interface = fmaxf(z_max_interface, static_cast<float>(k));
                    }
                }
            }
        }
        interface_z_ = z_max_interface + 1.0f;  // Just above the highest metal cell
        std::cout << "  Interface z (from fill_level): " << interface_z_
                  << " cells (" << interface_z_ * config_.dx * 1e6f << " μm)" << std::endl;
    }

    // Register output fields now that all sub-solvers are initialized
    registerOutputFields();
}

void MultiphysicsSolver::registerOutputFields() {
    field_registry_.clear();

    // Temperature (from thermal solver or static field)
    const float* temp_ptr = getTemperature();
    if (temp_ptr) {
        field_registry_.registerScalar("temperature", temp_ptr);
    }

    // Liquid fraction (requires thermal solver with phase change)
    const float* lf_ptr = getLiquidFraction();
    if (lf_ptr) {
        field_registry_.registerScalar("liquid_fraction", lf_ptr);
    }

    // Velocity vector field (requires fluid solver)
    if (fluid_) {
        field_registry_.registerVector("velocity",
                                       fluid_->getVelocityX(),
                                       fluid_->getVelocityY(),
                                       fluid_->getVelocityZ());
        // Pressure
        const float* p_ptr = fluid_->getPressure();
        if (p_ptr) {
            field_registry_.registerScalar("pressure", p_ptr);
        }
    }

    // VOF fields (requires VOF solver)
    if (vof_) {
        field_registry_.registerScalar("fill_level", vof_->getFillLevel());
        const float* curv_ptr = vof_->getCurvature();
        if (curv_ptr) {
            field_registry_.registerScalar("curvature", curv_ptr);
        }
    }

    auto names = field_registry_.getFieldNames();
    std::cout << "  Registered " << names.size() << " output fields:";
    for (const auto& n : names) {
        std::cout << " " << n;
    }
    std::cout << std::endl;
}

void MultiphysicsSolver::setStaticTemperature(const float* temperature_field) {
    if (d_temperature_static_) {
        int num_cells = config_.nx * config_.ny * config_.nz;
        cudaMemcpy(d_temperature_static_, temperature_field,
                  num_cells * sizeof(float), cudaMemcpyDeviceToDevice);
    }
}

void MultiphysicsSolver::setStaticLiquidFraction(const float* liquid_fraction_field) {
    if (d_liquid_fraction_static_) {
        int num_cells = config_.nx * config_.ny * config_.nz;
        cudaMemcpy(d_liquid_fraction_static_, liquid_fraction_field,
                  num_cells * sizeof(float), cudaMemcpyDeviceToDevice);
    }
}

// ============================================================================
// Time Integration
// ============================================================================

void MultiphysicsSolver::step(float dt) {
    if (dt == 0.0f) {
        dt = config_.dt;
    }

    // Snapshot T at step start for bisection ESM enthalpy inversion.
    // Must precede any T-modifying operation (laser, heat source, diffusion).
    if (config_.enable_thermal && thermal_) {
        thermal_->storePreviousTemperature();
    }

    // Step 1: Apply laser source (if enabled)
    if (config_.enable_laser && laser_) {
        applyLaserSource(dt);
    }

    // Step 2: Thermal diffusion (if enabled)
    if (config_.enable_thermal && thermal_) {
        thermalStep(dt);
    }

    // ========================================================================
    // Step 3: Fluid flow
    // ========================================================================
    // CRITICAL FIX: Moved BEFORE VOF advection so VOF sees updated velocities
    //
    // Previous bug: VOF was reading velocities from previous timestep
    //   Order was: Laser → Thermal → VOF → Fluid
    //   VOF would read old velocities, resulting in v_max=0.0 in diagnostics
    //
    // Correct order: Laser → Thermal → Fluid → VOF
    //   VOF now reads current timestep velocities
    //   This is physically correct: velocity drives interface advection
    // ========================================================================
    if (config_.enable_fluid && fluid_) {
        fluidStep(dt);
    }

    // Step 4: VOF interface management
    if (vof_) {
        // Track-C Gate 1: push current laser x position into VOF before advection.
        // laser_->x0 is the physical x in [m] after updatePosition() ran in Step 1.
        // Convert to lattice units: lu = x_m / dx.
        if (laser_ && config_.mass_correction_use_track_c) {
            float laser_x_lu = laser_->x0 / config_.dx;
            vof_->setMassCorrectionLaserX(laser_x_lu,
                                          config_.mass_correction_trailing_margin_lu);
        }

        if (config_.enable_vof_advection) {
            // Full VOF with advection
            vofStep(dt);
        } else {
            // Only reconstruct interface (Step 1: no advection)
            vof_->reconstructInterface();
            vof_->applyBoundaryConditions(1, 10.0f);
        }
    }

    // ========================================================================
    // EVAPORATION MASS LOSS: Remove material due to evaporation
    // ========================================================================
    // CRITICAL FIX: Moved out of vofStep() so it works even when fluid is disabled
    // Physics: Evaporation removes material from the surface
    //   dm/dt = -J_evap * A_interface    [kg/s]
    //   df/dt = -J_evap / (rho * dx)     [1/s]
    //
    // This couples the thermal solver's evaporation calculation to VOF:
    // - ThermalLBM computes J_evap using Hertz-Knudsen-Langmuir model
    // - VOFSolver applies mass loss to fill_level field
    //
    // Requirements: enable_thermal && enable_evaporation_mass_loss && vof_
    // ========================================================================
    if (config_.enable_evaporation_mass_loss && config_.enable_thermal && thermal_ && vof_) {
        // Compute evaporation mass flux from thermal solver
        // Pass VOF fill_level to compute evaporation only at actual interface
        thermal_->computeEvaporationMassFlux(d_evap_mass_flux_, vof_->getFillLevel());

        // CRITICAL FIX (2026-01-25): Apply evaporation COOLING to thermal field
        // Previously, mass was removed but latent heat wasn't - causing T > 3000K
        // Physics: Q_cool = J_evap * L_vap [W/m²] removes heat at interface
        thermal_->applyEvaporationCooling(d_evap_mass_flux_, vof_->getFillLevel(),
                                          dt, config_.dx,
                                          config_.evap_cooling_factor);

        // VOF mass loss: apply full physical HKL mass flux.
        // F-06 (code-audit pass 1, 2026-04-27): the prior block multiplied the
        // device array by a constexpr 1.0f via a GPU→CPU→GPU round-trip — a
        // ~190 MB×3 host transfer per step on the 8M-cell production grid.
        // The scale was a no-op; removed. To re-introduce a non-1.0 scale,
        // do it in-kernel rather than round-tripping through host.
        vof_->applyEvaporationMassLoss(d_evap_mass_flux_,
                                       config_.material.rho_liquid, dt);

        // Diagnostic: Print evaporation info every 100 steps
        if (current_step_ % 100 == 0) {
            int num_cells = config_.nx * config_.ny * config_.nz;
            std::vector<float> h_J_evap(num_cells);
            CUDA_CHECK(cudaMemcpy(h_J_evap.data(), d_evap_mass_flux_, num_cells * sizeof(float), cudaMemcpyDeviceToHost));

            float max_J = 0.0f;
            int evap_cells = 0;
            for (int i = 0; i < num_cells; ++i) {
                if (h_J_evap[i] > 1e-10f) {
                    evap_cells++;
                    max_J = std::max(max_J, h_J_evap[i]);
                }
            }

            if (evap_cells > 0) {
                printf("[EVAPORATION] Step %d: active_cells=%d, max_J=%.4e kg/(m^2*s)\n",
                       current_step_, evap_cells, max_J);
            }
        }
    }

    current_time_ += dt;
    current_step_++;

    // ============================================================
    // Store liquid fraction for next step's rate calculation
    // ============================================================
    // This must happen AFTER all physics updates for the current step
    // so that df_l/dt = (fl_current - fl_previous) / dt is correct
    if (config_.enable_solidification_shrinkage && thermal_ && thermal_->hasPhaseChange()) {
        auto phase_solver = thermal_->getPhaseChangeSolver();
        if (phase_solver) {
            phase_solver->storePreviousLiquidFraction();
        }
    }

    // ============================================================
    // Week 1 Monday: Energy Balance Diagnostics
    // ============================================================
    if (current_step_ % diagnostic_interval_ == 0) {
        printEnergyBalance();
    }
}

void MultiphysicsSolver::applyLaserSource(float dt) {
    if (!laser_ || !thermal_) return;

    // STABILITY FIX: Configurable laser shutoff time
    // Negative shutoff time means laser is always on
    if (config_.laser_shutoff_time >= 0.0f && current_time_ > config_.laser_shutoff_time) {
        // Laser is OFF - no heat source applied
        return;
    }

    // Update laser position based on scan velocity
    laser_->updatePosition(dt);

    // Allocate temporary device memory for heat source (RAII guard for exception safety)
    struct CudaFreeGuard { float* p; ~CudaFreeGuard() { if (p) cudaFree(p); } };
    int num_cells = config_.nx * config_.ny * config_.nz;
    float* d_heat_source = nullptr;
    CUDA_CHECK(cudaMalloc(&d_heat_source, num_cells * sizeof(float)));
    CudaFreeGuard guard{d_heat_source};
    CUDA_CHECK(cudaMemset(d_heat_source, 0, num_cells * sizeof(float)));

    if (config_.ray_tracing.enabled && ray_tracing_laser_) {
        // ============================================================
        // Ray Tracing path: geometric multi-reflection in powder bed
        // ============================================================

        // Ensure VOF interface normals are fresh (normally updated in Step 4,
        // but ray tracing runs in Step 1 and needs current normals)
        if (vof_) {
            vof_->reconstructInterface();
        }

        ray_tracing_laser_->traceAndDeposit(
            vof_ ? vof_->getFillLevel() : nullptr,
            vof_ ? vof_->getInterfaceNormals() : nullptr,
            *laser_,
            d_heat_source
        );

        // Diagnostic: print energy balance periodically
        if (current_step_ % diagnostic_interval_ == 0) {
            printf("[RayTrace] Step %d: deposited=%.3f W, escaped=%.3f W, "
                   "input=%.3f W, error=%.4e\n",
                   current_step_,
                   ray_tracing_laser_->getDepositedPower(),
                   ray_tracing_laser_->getEscapedPower(),
                   ray_tracing_laser_->getInputPower(),
                   ray_tracing_laser_->getEnergyError());
        }
    } else {
        // ============================================================
        // Beer-Lambert path: volumetric Gaussian projection (original)
        // ============================================================
        dim3 threads(8, 8, 8);
        dim3 blocks(
            (config_.nx + threads.x - 1) / threads.x,
            (config_.ny + threads.y - 1) / threads.y,
            (config_.nz + threads.z - 1) / threads.z
        );

        float z_surface = interface_z_;
        const float* fill_ptr = vof_ ? vof_->getFillLevel() : nullptr;
        const float* T_ptr = thermal_ ? thermal_->getTemperature() : nullptr;
        computeLaserHeatSourceKernel<<<blocks, threads>>>(
            d_heat_source,
            fill_ptr,
            T_ptr,
            *laser_,
            config_.nx, config_.ny, config_.nz,
            config_.dx,
            z_surface
        );
        CUDA_CHECK_KERNEL();
        CUDA_CHECK(cudaDeviceSynchronize());

        // ============================================================
        // DYNAMIC POWER NORMALIZATION: force exact energy conservation
        //
        // The VOF |∂f/∂z| interface kernel has a discrete integral deficit
        // (~50% for 2-cell interface). Instead of guessing a static multiplier,
        // compute the actual deposited power and rescale to match η×P exactly.
        //
        // P_actual = Σ(Q_vol × dx³), P_target = η × P_laser
        // scale = P_target / P_actual
        // Q_vol *= scale
        // ============================================================
        if (fill_ptr != nullptr) {
            // Host-side reduction (fast enough: called once per step, num_cells ~ 800K)
            std::vector<float> h_Q(num_cells);
            CUDA_CHECK(cudaMemcpy(h_Q.data(), d_heat_source,
                                  num_cells * sizeof(float), cudaMemcpyDeviceToHost));

            double P_actual = 0.0;
            float dV = config_.dx * config_.dx * config_.dx;
            for (int i = 0; i < num_cells; i++) {
                P_actual += static_cast<double>(h_Q[i]) * dV;
            }

            float P_target = config_.laser_absorptivity * config_.laser_power;

            if (P_actual > 1e-6) {
                float scale = static_cast<float>(P_target / P_actual);
                // Apply scale on device
                int bs = 256, gs = (num_cells + bs - 1) / bs;
                // Simple scale kernel (inline lambda not supported in CUDA, use thrust or manual)
                for (int i = 0; i < num_cells; i++) h_Q[i] *= scale;
                CUDA_CHECK(cudaMemcpy(d_heat_source, h_Q.data(),
                                      num_cells * sizeof(float), cudaMemcpyHostToDevice));

                // R6: Accumulate laser energy at P_target (fill gate is in the laser
                // kernel itself, so normalization already accounts for it)
                laser_energy_accumulated_ += static_cast<double>(P_target) * dt;

                if (current_step_ % diagnostic_interval_ == 0) {
                    printf("[LaserNorm] P_actual=%.2f W, P_target=%.2f W, scale=%.3f\n",
                           (float)P_actual, P_target, scale);
                }
            }
        }
    }

    thermal_->addHeatSource(d_heat_source, dt);

    // R6: Accumulate laser energy for energy balance diagnostic.
    // For Beer-Lambert with VOF: already accumulated in normalization block above.
    // For ray tracing: read actual deposited power (Fresnel multi-bounce can
    //   give much higher effective absorption than config_.laser_absorptivity).
    //   Sprint-1 fix (2026-04-25): was using laser_absorptivity*P_laser, which
    //   reported 60 W vs the actual RT-deposited 102-108 W → ENERGY R6 balance
    //   reported 70-80 % error when it was really ~10 %.
    // For Beer-Lambert without VOF: accumulate at the absorptivity-scaled value.
    if (config_.ray_tracing.enabled && ray_tracing_laser_) {
        float P_deposited = ray_tracing_laser_->getDepositedPower();
        laser_energy_accumulated_ += static_cast<double>(P_deposited) * dt;
    } else if (!vof_) {
        float P_deposited = config_.laser_absorptivity * config_.laser_power;
        laser_energy_accumulated_ += static_cast<double>(P_deposited) * dt;
    }
}

void MultiphysicsSolver::thermalStep(float dt) {
    if (!thermal_) return;

    // Set VOF fill_level for ESM gas masking (prevents phase change in gas)
    if (vof_) {
        thermal_->setVOFFillLevel(vof_->getFillLevel());
    }

    // NO T cap: let physics self-regulate via evaporation cooling.
    // Skip the cap entirely so temperature can reach its natural equilibrium
    // where laser input balances evaporation + conduction losses.
    thermal_->setSkipTemperatureCap(true);

    // Thermal-fluid coupling: pass velocity for advection term v*nabla(T)
    const float* ux = config_.enable_thermal_advection && fluid_ ? fluid_->getVelocityX() : nullptr;
    const float* uy = config_.enable_thermal_advection && fluid_ ? fluid_->getVelocityY() : nullptr;
    const float* uz = config_.enable_thermal_advection && fluid_ ? fluid_->getVelocityZ() : nullptr;

    // ============================================================
    // CRITICAL BUFFER MANAGEMENT: Apply BCs BEFORE streaming
    // ============================================================
    // Order: BCs -> collision -> streaming -> computeTemperature
    // BCs modify d_g_src; collision processes those modifications;
    // streaming propagates them. Applying BCs AFTER streaming would
    // write to the wrong buffer and the effects would be lost.
    // ============================================================

    if (current_step_ == 0) {
        printf("[THERMAL BC] Per-face boundary conditions:\n");
        const char* bc_names[] = {"PERIODIC", "ADIABATIC", "DIRICHLET", "CONVECTIVE", "RADIATION"};
        for (int f = 0; f < 6; ++f) {
            const char* face_names[] = {"x_min", "x_max", "y_min", "y_max", "z_min", "z_max"};
            int bc = static_cast<int>(config_.boundaries.thermalBCForFace(f));
            printf("  %s: %s\n", face_names[f], bc_names[bc]);
        }
    }

    // ============================================================
    // PER-FACE THERMAL BOUNDARY CONDITIONS
    // ============================================================
    // Dispatch the correct BC type to each of the 6 faces using the
    // FaceBoundaryConfig. This replaces the old monolithic approach
    // where boundary_type was a single integer for all faces, and
    // radiation/convection were separate boolean flags.
    //
    // Legacy path: if boundary_type != 0 and no face has non-PERIODIC
    // thermal BC, fall back to legacy applyBoundaryConditions() call.
    // ============================================================

    bool use_per_face = config_.boundaries.hasAnyThermalBC(ThermalBCType::ADIABATIC) ||
                        config_.boundaries.hasAnyThermalBC(ThermalBCType::DIRICHLET) ||
                        config_.boundaries.hasAnyThermalBC(ThermalBCType::CONVECTIVE) ||
                        config_.boundaries.hasAnyThermalBC(ThermalBCType::RADIATION);

    if (use_per_face) {
        // New per-face thermal BC path
        for (int face = 0; face < 6; ++face) {
            ThermalBCType bc = config_.boundaries.thermalBCForFace(face);
            thermal_->applyFaceThermalBC(
                face,
                static_cast<int>(bc),
                dt,
                config_.dx,
                config_.boundaries.dirichlet_temperature,
                config_.boundaries.convective_h,
                config_.boundaries.convective_T_inf,
                config_.boundaries.radiation_emissivity,
                config_.boundaries.radiation_T_ambient
            );
        }
    } else if (config_.boundary_type == 1 || config_.boundary_type == 2) {
        // Legacy path: uniform BC on all 6 faces
        thermal_->applyBoundaryConditions(
            config_.boundary_type,
            config_.substrate_temperature
        );
    }

    // ============================================================
    // Legacy radiation/convection flags (backward compatibility)
    // ============================================================
    // If the user set enable_radiation_bc or enable_substrate_cooling
    // but did NOT configure per-face boundaries, apply the old-style
    // top-surface radiation and bottom-surface convection.
    // When per-face BCs are active, these flags are redundant because
    // the same physics is expressed through FaceBoundaryConfig.
    // ============================================================
    if (!use_per_face) {
        if (config_.enable_radiation_bc) {
            thermal_->applyRadiationBC(
                dt,
                config_.dx,
                config_.emissivity,
                config_.ambient_temperature
            );
        }

        if (config_.enable_substrate_cooling) {
            thermal_->applySubstrateCoolingBC(
                dt,
                config_.dx,
                config_.substrate_h_conv,
                config_.substrate_temperature
            );
        }
    }

    // BGK collision (processes BC modifications from above)
    thermal_->collisionBGK(ux, uy, uz);

    // Streaming (propagates distributions with BC effects)
    thermal_->streaming();

    // Compute temperature from distribution functions
    thermal_->computeTemperature();

    // Evaporation cooling — HKL energy sink at the free surface.
    // Only applied here when enable_evaporation_mass_loss is NOT set
    // (i.e., VOF mass loss is disabled). When mass loss IS enabled, evaporation
    // cooling is already applied in thermalStep() via the J_evap path (line 1504),
    // which correctly restricts cooling to interface cells. Applying it again here
    // would double-count the energy removal.
    if (thermal_ && !config_.enable_evaporation_mass_loss) {
        thermal_->applyEvaporationCooling(nullptr, nullptr,
                                          config_.dt, config_.dx, 1.0f);
    }

    // ============================================================
    // SUB-SURFACE BOILING CAP
    // ============================================================
    // Evaporation cooling only acts on interface cells (f ∈ [0.01, 0.99]).
    // Bulk liquid cells (f >= 0.99) can overheat beyond T_boil due to
    // thermal conduction from the laser hotspot. Cap their temperature
    // at T_boil + 50K (configurable overshoot) and track the removed
    // energy as volumetric boiling latent heat for energy conservation.
    // ============================================================
    if (thermal_) {
        // Sprint-1 (2026-04-25): raised overshoot from 50 K to 1500 K.
        // The 50 K cap effectively pinned bulk T at T_boil+50 K = 3140 K, which
        // broadcast laterally and forced LBM Tmax 3650 K vs Flow3D 4262 K peak
        // (-14 %). Real LPBF keyhole bottoms can superheat to 4500-5500 K
        // (Khairallah 2016, Flow3D peak 4262 K). The aggressive cap was
        // for stability when ESM evap cooling was less robust; with R7
        // implicit Newton evap cooling now in production it can be relaxed.
        thermal_->applySubsurfaceBoilingCap(config_.material.T_vaporization, 1500.0f);
    }

    // ============================================================
    // RE-APPLY DIRICHLET BCs AFTER STREAMING
    // ============================================================
    // Streaming can perturb Dirichlet values via bounce-back.
    // Re-applying ensures exact enforcement (standard LBM pattern).
    // ============================================================
    if (use_per_face) {
        for (int face = 0; face < 6; ++face) {
            ThermalBCType bc = config_.boundaries.thermalBCForFace(face);
            if (bc == ThermalBCType::DIRICHLET) {
                thermal_->applyFaceThermalBC(
                    face,
                    static_cast<int>(ThermalBCType::DIRICHLET),
                    dt, config_.dx,
                    config_.boundaries.dirichlet_temperature,
                    0.0f, 0.0f, 0.0f, 0.0f  // unused params
                );
            }
        }
    } else if (config_.boundary_type == 1 || config_.boundary_type == 2) {
        thermal_->applyBoundaryConditions(
            config_.boundary_type,
            config_.substrate_temperature
        );
    }
}

void MultiphysicsSolver::vofStep(float dt) {
    if (!vof_ || !fluid_) return;

    int num_cells = config_.nx * config_.ny * config_.nz;

    // ==================================================================
    // CRITICAL FIX: Convert velocity from lattice units to physical [m/s]
    // ==================================================================
    // LBM outputs velocity in lattice units (dimensionless, ~0.01-0.1)
    // VOF advection expects velocity in physical units [m/s]
    //
    // Without this conversion, VOF advection is ~20x too slow!
    // ==================================================================
    const float velocity_conversion = config_.dx / config_.dt;

    int threads = 256;
    int blocks = (num_cells + threads - 1) / threads;

    convertVelocityToPhysicalUnitsKernel<<<blocks, threads>>>(
        fluid_->getVelocityX(),
        fluid_->getVelocityY(),
        fluid_->getVelocityZ(),
        d_velocity_physical_x_,
        d_velocity_physical_y_,
        d_velocity_physical_z_,
        velocity_conversion,
        num_cells
    );
    CUDA_CHECK_KERNEL();
    CUDA_CHECK(cudaDeviceSynchronize());

    // Diagnostic: print velocity conversion info on first call
    static bool first_call = true;
    static int diag_count = 0;
    if (first_call) {
        first_call = false;
        std::cout << "[VOF UNIT FIX] Velocity unit conversion enabled:\n";
        std::cout << "  dx = " << config_.dx << " m\n";
        std::cout << "  dt = " << config_.dt << " s\n";
        std::cout << "  conversion factor = " << velocity_conversion << " m/s per lattice unit\n";
    }

    // DIAGNOSTIC: Verify converted velocity buffers are non-zero
    // BUG FIX: Was sampling only top layer (gas phase → zero velocity).
    // Now samples FULL domain to see the actual Marangoni flow at the interface.
    if (diag_count % 500 == 0 && diag_count < 5000) {
        std::vector<float> h_ux_lattice(num_cells), h_ux_phys(num_cells);
        CUDA_CHECK(cudaMemcpy(h_ux_lattice.data(), fluid_->getVelocityX(),
                              num_cells * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_ux_phys.data(), d_velocity_physical_x_,
                              num_cells * sizeof(float), cudaMemcpyDeviceToHost));

        float v_max_lattice = 0.0f, v_max_phys = 0.0f;
        for (int i = 0; i < num_cells; ++i) {
            v_max_lattice = std::max(v_max_lattice, std::abs(h_ux_lattice[i]));
            v_max_phys = std::max(v_max_phys, std::abs(h_ux_phys[i]));
        }

        printf("[VELOCITY CONVERSION CHECK] Call %d:\n", diag_count);
        printf("  Lattice velocity (LBM):  v_max = %.6f (dimensionless)\n", v_max_lattice);
        printf("  Physical velocity (VOF): v_max = %.6f m/s (%.2f mm/s)\n", v_max_phys, v_max_phys * 1000);
        printf("  Expected conversion:     %.6f * %.2f = %.6f m/s\n",
               v_max_lattice, velocity_conversion, v_max_lattice * velocity_conversion);

        if (v_max_phys < 1e-10f && v_max_lattice > 1e-6f) {
            printf("  ERROR: Conversion kernel FAILED - physical velocity is zero despite non-zero lattice velocity!\n");
        } else if (std::abs(v_max_phys - v_max_lattice * velocity_conversion) > 1e-4f) {
            printf("  WARNING: Conversion mismatch detected!\n");
        } else {
            printf("  OK: Conversion successful\n");
        }
    }
    diag_count++;

    // ========================================================================
    // DIAGNOSTIC: Track fill_level changes to debug static surface issue
    // ========================================================================
    static int vof_diag_count = 0;
    static float prev_mass = -1.0f;
    static float prev_z_centroid = -1.0f;
    std::vector<float> h_fill_before;

    bool do_diagnostic = (vof_diag_count % 100 == 0);
    if (do_diagnostic) {
        h_fill_before.resize(num_cells);
        cudaMemcpy(h_fill_before.data(), vof_->getFillLevel(),
                   num_cells * sizeof(float), cudaMemcpyDeviceToHost);
    }

    // SOLIDUS VELOCITY BRAKE: Darcy-only (no hard freeze)
    // FLOW-3D approach: Darcy K = C·(1-fl)·ρ·dt handles solid braking smoothly.
    // Hard velocity zeroing (freezeSolidVelocityKernel) REMOVED — it created
    // artificial rigid boundaries that pinned the contact line and caused balling.

    // Subcycling for VOF stability
    float dt_sub = dt / config_.vof_subcycles;

    for (int i = 0; i < config_.vof_subcycles; ++i) {
        vof_->advectFillLevel(d_velocity_physical_x_,
                              d_velocity_physical_y_,
                              d_velocity_physical_z_,
                              dt_sub);
    }

    // REMOVED: Evaporation logic moved to main step() function (line ~957)
    // so it works even when fluid is disabled

    // ========================================================================
    // SOLIDIFICATION SHRINKAGE: Apply volume change due to solidification
    // ========================================================================
    // Physics: Metal contracts during solidification due to density change
    //   df/dt = -beta * df_l/dt
    // where:
    //   beta = (rho_solid - rho_liquid) / rho_solid = 1 - rho_l/rho_s
    //   df_l/dt = rate of liquid fraction change (negative during solidification)
    //
    // This couples the thermal solver's phase change to VOF:
    // - PhaseChangeSolver computes df_l/dt from temperature evolution
    // - VOFSolver applies volume shrinkage to fill_level field
    //
    // Requirements: enable_solidification_shrinkage && thermal with phase change
    // ========================================================================
    if (config_.enable_solidification_shrinkage && thermal_ && thermal_->hasPhaseChange()) {
        // Get phase change solver from thermal
        auto phase_solver = thermal_->getPhaseChangeSolver();
        if (phase_solver) {
            // Compute liquid fraction rate
            phase_solver->computeLiquidFractionRate(config_.dt);

            // Get rate field and material properties
            const float* dfl_dt = phase_solver->getLiquidFractionRate();
            float beta = config_.material.getShrinkageFactor();

            // Apply to VOF
            vof_->applySolidificationShrinkage(dfl_dt, beta, config_.dx, config_.dt);

            // ============================================================
            // ENHANCED DIAGNOSTIC: Solidification shrinkage rate analysis
            // ============================================================
            // Print detailed info every 100 steps for debugging cavity issues
            if (current_step_ % 100 == 0) {
                // Sample dfl_dt field to find max solidification rate
                std::vector<float> h_dfl_dt(num_cells);
                CUDA_CHECK(cudaMemcpy(h_dfl_dt.data(), dfl_dt, num_cells * sizeof(float), cudaMemcpyDeviceToHost));

                float max_solidification_rate = 0.0f;  // Most negative value
                float max_melting_rate = 0.0f;         // Most positive value
                int solidifying_cells = 0;
                int melting_cells = 0;
                int max_rate_idx = 0;
                float sum_rate = 0.0f;

                for (int i = 0; i < num_cells; ++i) {
                    float rate = h_dfl_dt[i];
                    if (rate < -1e-10f) {
                        solidifying_cells++;
                        sum_rate += rate;
                        if (rate < max_solidification_rate) {
                            max_solidification_rate = rate;
                            max_rate_idx = i;
                        }
                    } else if (rate > 1e-10f) {
                        melting_cells++;
                        if (rate > max_melting_rate) {
                            max_melting_rate = rate;
                        }
                    }
                }

                // Compute position of max solidification
                int iz = max_rate_idx / (config_.nx * config_.ny);
                int iy = (max_rate_idx - iz * config_.nx * config_.ny) / config_.nx;
                int ix = max_rate_idx % config_.nx;

                // Compute expected df from shrinkage (NEW CORRECTED FORMULA)
                // df = beta * dfl_dt * dt (dimensionless)
                float expected_df_max = beta * max_solidification_rate * config_.dt;

                // Count interface cells that are also solidifying
                std::vector<float> h_fill(num_cells);
                CUDA_CHECK(cudaMemcpy(h_fill.data(), vof_->getFillLevel(), num_cells * sizeof(float), cudaMemcpyDeviceToHost));

                int interface_solidifying = 0;
                float max_interface_rate = 0.0f;
                for (int i = 0; i < num_cells; ++i) {
                    float f = h_fill[i];
                    float rate = h_dfl_dt[i];
                    // NEW CONSTRAINT: Only interface cells (0.01 < f < 0.99) AND solidifying (rate < 0)
                    if (f > 0.01f && f < 0.99f && rate < -1e-10f) {
                        interface_solidifying++;
                        if (rate < max_interface_rate) {
                            max_interface_rate = rate;
                        }
                    }
                }

                printf("[SHRINKAGE DIAGNOSTIC] Step %d:\n", current_step_);
                printf("  beta = %.4f (shrinkage factor)\n", beta);
                printf("  Solidifying cells: %d, Melting cells: %d\n", solidifying_cells, melting_cells);
                printf("  INTERFACE solidifying cells: %d (NEW: only these get shrinkage)\n", interface_solidifying);
                printf("  Max solidification rate (all): %.6e 1/s at (%d, %d, %d)\n",
                       max_solidification_rate, ix, iy, iz);
                printf("  Max solidification rate (interface only): %.6e 1/s\n", max_interface_rate);
                printf("  Expected max df (CORRECTED): %.6e (was %.6e with old /dx formula)\n",
                       expected_df_max, expected_df_max / config_.dx);
                printf("  Sum solidification rate: %.6e 1/s\n", sum_rate);

                if (interface_solidifying > 0 && std::abs(expected_df_max) > 0.01f) {
                    printf("  WARNING: Large shrinkage rate at interface! Max 1%% limiter will engage.\n");
                }
            }
        }
    }

    // DIAGNOSTIC: Compute fill_level change and interface position
    if (do_diagnostic) {
        std::vector<float> h_fill_after(num_cells);
        cudaMemcpy(h_fill_after.data(), vof_->getFillLevel(),
                   num_cells * sizeof(float), cudaMemcpyDeviceToHost);

        // Compute max change in fill_level
        float max_change = 0.0f;
        int max_change_idx = 0;
        for (int i = 0; i < num_cells; ++i) {
            float delta = std::abs(h_fill_after[i] - h_fill_before[i]);
            if (delta > max_change) {
                max_change = delta;
                max_change_idx = i;
            }
        }

        // Compute interface z-centroid (weighted average of z for interface cells)
        float z_sum = 0.0f, f_sum = 0.0f;
        int interface_cell_count = 0;
        for (int k = 0; k < config_.nz; ++k) {
            for (int j = 0; j < config_.ny; ++j) {
                for (int i = 0; i < config_.nx; ++i) {
                    int idx = i + config_.nx * (j + config_.ny * k);
                    float f = h_fill_after[idx];
                    if (f > 0.01f && f < 0.99f) {  // Interface cells only
                        z_sum += k * f;
                        f_sum += f;
                        interface_cell_count++;
                    }
                }
            }
        }
        float z_centroid = (f_sum > 0) ? z_sum / f_sum : -1.0f;

        // Compute current mass
        float mass = vof_->computeTotalMass();
        float mass_change = (prev_mass > 0) ? (mass - prev_mass) : 0.0f;
        float z_change = (prev_z_centroid > 0) ? (z_centroid - prev_z_centroid) : 0.0f;

        // Check velocity magnitude at interface
        std::vector<float> h_vz(num_cells);
        CUDA_CHECK(cudaMemcpy(h_vz.data(), d_velocity_physical_z_, num_cells * sizeof(float), cudaMemcpyDeviceToHost));
        float v_max_interface = 0.0f;
        for (int i = 0; i < num_cells; ++i) {
            if (h_fill_after[i] > 0.01f && h_fill_after[i] < 0.99f) {
                v_max_interface = std::max(v_max_interface, std::abs(h_vz[i]));
            }
        }

        printf("[VOF DIAGNOSTIC] Step %d:\n", vof_diag_count);
        printf("  max fill_level change = %.8f (at idx %d)\n", max_change, max_change_idx);
        printf("  interface z-centroid  = %.3f cells (delta=%.4f)\n", z_centroid, z_change);
        printf("  interface cell count  = %d\n", interface_cell_count);
        printf("  v_z max at interface  = %.4f m/s\n", v_max_interface);
        printf("  total mass            = %.1f (delta=%.4f)\n", mass, mass_change);

        if (max_change < 1e-6f) {
            printf("  WARNING: fill_level NOT changing - surface is STATIC!\n");
            printf("  Possible causes:\n");
            printf("    1. Velocity is zero or near-zero at interface\n");
            printf("    2. enable_vof_advection may be false\n");
            printf("    3. Interface is at domain boundary (z=%d)\n", config_.nz - 1);
        }

        prev_mass = mass;
        prev_z_centroid = z_centroid;
    }
    vof_diag_count++;

    // ========================================================================
    // SMOOTHED CURVATURE (FLOW-3D approach): smooth fill_level before
    // computing normals and curvature for CSF surface tension.
    // Raw fill_level has staircase artifacts at dx=2μm that create
    // spurious high-curvature "hard shells" preventing liquid spreading.
    //
    // Procedure:
    //   1. Save raw fill_level to temp buffer
    //   2. Smooth fill_level in-place (2 passes of 27-point isotropic kernel)
    //   3. Compute normals + curvature from smooth field
    //   4. Restore raw fill_level (advection needs the sharp interface)
    // ========================================================================
    {
        int num_cells_local = config_.nx * config_.ny * config_.nz;

        // Save raw fill_level
        float* d_fill_raw = nullptr;
        CUDA_CHECK(cudaMalloc(&d_fill_raw, num_cells_local * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_fill_raw, vof_->getFillLevel(),
                              num_cells_local * sizeof(float), cudaMemcpyDeviceToDevice));

        // Smooth fill_level in-place (2 passes)
        float* d_fill_tmp = nullptr;
        CUDA_CHECK(cudaMalloc(&d_fill_tmp, num_cells_local * sizeof(float)));

        dim3 blk(8, 8, 8);
        dim3 grd((config_.nx + 7) / 8, (config_.ny + 7) / 8, (config_.nz + 7) / 8);

        // Pass 1: fill_level → tmp
        smoothField27Kernel<<<grd, blk>>>(vof_->getFillLevel(), d_fill_tmp,
                                          config_.nx, config_.ny, config_.nz);
        CUDA_CHECK_KERNEL();
        // Pass 2: tmp → fill_level (in-place overwrite)
        smoothField27Kernel<<<grd, blk>>>(d_fill_tmp, vof_->getFillLevel(),
                                          config_.nx, config_.ny, config_.nz);
        CUDA_CHECK_KERNEL();
        CUDA_CHECK(cudaDeviceSynchronize());

        cudaFree(d_fill_tmp);

        // Compute normals + curvature from smoothed field
        vof_->reconstructInterface();
        vof_->applyBoundaryConditions(1, 10.0f);
        if (config_.enable_surface_tension) {
            vof_->computeCurvature();
        }

        // Restore raw fill_level (sharp interface for advection + VOF transport)
        CUDA_CHECK(cudaMemcpy(vof_->getFillLevel(), d_fill_raw,
                              num_cells_local * sizeof(float), cudaMemcpyDeviceToDevice));
        cudaFree(d_fill_raw);
    }
}

void MultiphysicsSolver::fluidStep(float dt) {
    if (!fluid_) return;

    // Compute total force using ForceAccumulator pipeline
    // NOTE: Forces are already converted to lattice units inside computeTotalForce()
    // via force_accumulator_->convertToLatticeUnits(dx, dt, rho)
    // DO NOT apply conversion again here (was causing 2 million times underestimation!)
    computeTotalForce();

    // Grid parameters needed for CFL limiting and diagnostics
    int num_cells = config_.nx * config_.ny * config_.nz;
    int threads = 256;
    int blocks = (num_cells + threads - 1) / threads;

    // === DIAGNOSTIC: CFL limiter analysis ===
    // Extended to run periodically and sample full domain for recoil-active forces
    static int cfl_diag_count = 0;
    bool enable_cfl_diag = (cfl_diag_count < 5) || (current_step_ % 100 == 0 && cfl_diag_count < 30);
    float max_f_before_cfl = 0.0f;
    float max_v_before = 0.0f;

    if (enable_cfl_diag) {
        // Sample force magnitude from FULL domain (not just first 1000 cells)
        std::vector<float> h_fx_temp(num_cells);
        std::vector<float> h_fy_temp(num_cells);
        std::vector<float> h_fz_temp(num_cells);
        CUDA_CHECK(cudaMemcpy(h_fx_temp.data(), d_force_x_, num_cells * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_fy_temp.data(), d_force_y_, num_cells * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_fz_temp.data(), d_force_z_, num_cells * sizeof(float), cudaMemcpyDeviceToHost));

        float max_f_z = 0.0f;  // Track z-component separately for recoil
        int max_f_idx = 0;
        for (int i = 0; i < num_cells; ++i) {
            float f_mag = std::sqrt(h_fx_temp[i]*h_fx_temp[i] + h_fy_temp[i]*h_fy_temp[i] + h_fz_temp[i]*h_fz_temp[i]);
            if (f_mag > max_f_before_cfl) {
                max_f_before_cfl = f_mag;
                max_f_idx = i;
            }
            max_f_z = std::max(max_f_z, std::abs(h_fz_temp[i]));
        }

        // Sample current velocity from full domain
        std::vector<float> h_vx(num_cells);
        std::vector<float> h_vy(num_cells);
        std::vector<float> h_vz(num_cells);
        CUDA_CHECK(cudaMemcpy(h_vx.data(), fluid_->getVelocityX(), num_cells * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_vy.data(), fluid_->getVelocityY(), num_cells * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_vz.data(), fluid_->getVelocityZ(), num_cells * sizeof(float), cudaMemcpyDeviceToHost));
        for (int i = 0; i < num_cells; ++i) {
            float v_mag = std::sqrt(h_vx[i]*h_vx[i] + h_vy[i]*h_vy[i] + h_vz[i]*h_vz[i]);
            max_v_before = std::max(max_v_before, v_mag);
        }

        std::cout << "\n=== CFL LIMITER DIAGNOSTIC (Step " << current_step_ << ") ===\n";
        std::cout << std::scientific << std::setprecision(3);
        std::cout << "  Force BEFORE CFL: " << max_f_before_cfl << " (lattice units)\n";
        std::cout << "  Max F_z (recoil): " << max_f_z << " (lattice units)\n";
        std::cout << "  F at max idx " << max_f_idx << ": (" << h_fx_temp[max_f_idx]
                  << ", " << h_fy_temp[max_f_idx] << ", " << h_fz_temp[max_f_idx] << ")\n";
        std::cout << std::fixed << std::setprecision(3);
        std::cout << "  Current max velocity (all): " << max_v_before << " (lattice units)\n";
        std::cout << "  Current max velocity (all): " << (max_v_before * config_.dx / config_.dt) << " m/s (physical)\n";
        float v_metal = getMaxMetalVelocity();
        std::cout << "  Current max velocity (metal f>0.01): " << v_metal << " m/s\n";
    }

    // ========================================================================
    // CRITICAL FIX: CFL limiting is ALREADY applied in computeTotalForce()
    // ========================================================================
    // The ForceAccumulator already applies CFL limiting in computeTotalForce()
    // via force_accumulator_->applyCFLLimiting() or applyCFLLimitingAdaptive().
    //
    // Since d_force_x/y/z_ point DIRECTLY to ForceAccumulator's internal arrays
    // (see line 674-676 in constructor), applying CFL limiting here would be
    // DOUBLE LIMITING, which severely over-limits forces and breaks buoyancy.
    //
    // BUG FIX: Remove duplicate CFL limiting here
    // ========================================================================

    if (enable_cfl_diag) {
        // CFL limiting was already applied in computeTotalForce()
        // Report final force magnitudes (after CFL limiting in ForceAccumulator)
        std::vector<float> h_fx_final(num_cells);
        std::vector<float> h_fy_final(num_cells);
        std::vector<float> h_fz_final(num_cells);
        CUDA_CHECK(cudaMemcpy(h_fx_final.data(), d_force_x_, num_cells * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_fy_final.data(), d_force_y_, num_cells * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_fz_final.data(), d_force_z_, num_cells * sizeof(float), cudaMemcpyDeviceToHost));

        float max_f_final = 0.0f;
        float max_f_z_final = 0.0f;
        int max_f_final_idx = 0;
        for (int i = 0; i < num_cells; ++i) {
            float f_mag = std::sqrt(h_fx_final[i]*h_fx_final[i] + h_fy_final[i]*h_fy_final[i] + h_fz_final[i]*h_fz_final[i]);
            if (f_mag > max_f_final) {
                max_f_final = f_mag;
                max_f_final_idx = i;
            }
            max_f_z_final = std::max(max_f_z_final, std::abs(h_fz_final[i]));
        }

        float reduction = (max_f_before_cfl > 0) ?
            ((max_f_before_cfl - max_f_final) / max_f_before_cfl * 100.0f) : 0.0f;

        std::cout << "  Force FINAL (after CFL in ForceAccumulator): " << std::scientific << std::setprecision(3)
                  << max_f_final << " (lattice units)\n";
        std::cout << "  Max F_z final:    " << max_f_z_final << " (lattice units)\n";
        std::cout << std::fixed << std::setprecision(3);
        std::cout << "  CFL reduction: " << reduction << " %\n";

        if (config_.cfl_use_adaptive) {
            std::cout << "  Mode: ADAPTIVE REGION-BASED (applied in ForceAccumulator)\n";
            std::cout << "    v_target_interface = " << config_.cfl_v_target_interface << " (lattice)\n";
            std::cout << "    v_target_interface = " << (config_.cfl_v_target_interface * config_.dx / config_.dt) << " m/s (physical)\n";
            std::cout << "    v_target_bulk      = " << config_.cfl_v_target_bulk << " (lattice)\n";
            std::cout << "    v_target_bulk      = " << (config_.cfl_v_target_bulk * config_.dx / config_.dt) << " m/s (physical)\n";
            std::cout << "    recoil_boost       = " << config_.cfl_recoil_boost_factor << "x\n";
        } else if (config_.cfl_use_gradual_scaling) {
            std::cout << "  Mode: GRADUAL SCALING (applied in ForceAccumulator)\n";
            std::cout << "    v_target = " << config_.cfl_velocity_target << " (lattice)\n";
            std::cout << "    v_target = " << (config_.cfl_velocity_target * config_.dx / config_.dt) << " m/s (physical)\n";
            std::cout << "    ramp_factor = " << config_.cfl_force_ramp_factor << "\n";
        } else {
            std::cout << "  Mode: NO CFL LIMITING (both adaptive and gradual disabled)\n";
        }

        if (reduction > 90.0f) {
            std::cout << "  WARNING: High CFL reduction (>90%)!\n";
            std::cout << "    Consider: increase cfl_velocity_target or decrease force magnitude\n";
        } else if (reduction > 50.0f) {
            std::cout << "  Note: Moderate CFL reduction, deformation should be visible\n";
        } else {
            std::cout << "  Good: Low CFL reduction, forces mostly preserved\n";
        }
        cfl_diag_count++;
    }

    // Get Darcy coefficient field (nullptr if Darcy disabled OR not yet computed)
    const float* darcy_K = (config_.enable_darcy && force_accumulator_) ?
        force_accumulator_->getDarcyCoefficient() : nullptr;

    // Sprint-1 (2026-04-25): warn once if enable_darcy=true but darcy_K is null.
    // This silently falls back to Guo collision and is hard to notice in logs;
    // happens when computeDarcyCoefficientField was never called (e.g. thermal
    // module not yet initialized in early steps).
    if (config_.enable_darcy && !darcy_K) {
        static int darcy_warn_count = 0;
        if (darcy_warn_count++ == 0) {
            std::cerr << "[multiphysics] WARN: enable_darcy=true but darcy_K is null. "
                      << "Falling back to Guo BGK collision (no Darcy drag). "
                      << "Check that thermal_ provides liquid_fraction before fluidStep().\n";
        }
    }

    if (darcy_K) {
        // EDM collision: equilibrium shift replaces Guo source term.
        // Darcy drag absorbed semi-implicitly into u_bare denominator.
        // No distribution anisotropy accumulation — eliminates velocity shocks.
        fluid_->collisionBGKwithEDM(d_force_x_, d_force_y_, d_force_z_, darcy_K);
    } else {
        // Fallback to Guo scheme when no Darcy (e.g., single-phase benchmarks)
        fluid_->collisionBGK(d_force_x_, d_force_y_, d_force_z_);
    }

    // Streaming
    fluid_->streaming();

    // Marangoni stress BC (Inamuro specular) — for flat-wall problems without VOF
    if (config_.enable_marangoni && !vof_ && thermal_) {
        int z_surface = config_.nz - 1;
        fluid_->applyMarangoniStressBC(
            thermal_->getTemperature(),
            thermal_->getLiquidFraction(),
            config_.dsigma_dT,
            config_.dx, config_.dt, config_.density,
            z_surface);
    }

    // Compute macroscopic quantities
    if (darcy_K) {
        // EDM macroscopic: u_bare = m/(ρ+K/2), output u = u_bare + F/(2ρ)
        fluid_->computeMacroscopicEDM(d_force_x_, d_force_y_, d_force_z_, darcy_K);
    } else {
        // Guo macroscopic: u = m/ρ + 0.5*F/ρ
        fluid_->computeMacroscopic(d_force_x_, d_force_y_, d_force_z_);
    }
}

// ForceAccumulator-based implementation
// Note: ForceAccumulator class is now implemented
void MultiphysicsSolver::computeTotalForce() {
    // ========================================================================
    // NEW ROBUST FORCE PIPELINE USING ForceAccumulator
    // ========================================================================
    // This replaces the fragile scattered force computation with a clean,
    // debuggable pipeline that:
    // 1. Accumulates all forces in physical units [N/m³]
    // 2. Converts to lattice units
    // 3. Applies CFL limiting for stability
    //
    // Benefits:
    // - Unit consistency (all forces in [N/m³] before conversion)
    // - Order independence (no in-place modifications)
    // - Debuggable (track each force magnitude separately)
    // - Testable (each force can be tested in isolation)
    // ========================================================================

    // Step 1: Reset force arrays
    force_accumulator_->reset();

    // Step 2: Accumulate all physics forces (in physical units [N/m³])

    // 2a. Buoyancy force (Boussinesq approximation)
    if (config_.enable_buoyancy && fluid_) {
        const float* temperature = config_.enable_thermal ?
            thermal_->getTemperature() : d_temperature_static_;

        if (temperature) {
            // Get liquid fraction for optional masking
            const float* liquid_fraction = nullptr;
            if (thermal_) {
                liquid_fraction = thermal_->getLiquidFraction();
            }
            if (!liquid_fraction) {
                liquid_fraction = d_liquid_fraction_static_;
            }

            force_accumulator_->addBuoyancyForce(
                temperature,
                config_.reference_temperature,
                config_.thermal_expansion_coeff,
                config_.density,
                config_.gravity_x,
                config_.gravity_y,
                config_.gravity_z,
                liquid_fraction);
        }
    }

    // 2b. Surface tension force (CSF model)
    if (config_.enable_surface_tension && surface_tension_ && vof_) {
        const float* fill_level = vof_->getFillLevel();
        const float* curvature = vof_->getCurvature();

        if (fill_level && curvature) {
            force_accumulator_->addSurfaceTensionForce(
                curvature, fill_level,
                config_.surface_tension_coeff,
                config_.nx, config_.ny, config_.nz,
                config_.dx);
        }
    }

    // 2c. Marangoni force (thermocapillary)
    //
    // FIX: Use 27-point smoothed temperature for ∇T computation to remove
    // D3Q7 anisotropic noise that causes parasitic Marangoni currents.
    // The thermodynamic temperature field remains pristine.
    if (config_.enable_marangoni && marangoni_ && vof_) {
        const float* temperature_raw = config_.enable_thermal ?
            thermal_->getTemperature() : d_temperature_static_;
        const float* fill_level = vof_->getFillLevel();
        const float3* normals = vof_->getInterfaceNormals();

        if (temperature_raw && fill_level && normals) {
            // Smooth temperature with 3 passes of 27-point isotropic kernel.
            // Effective smoothing radius ~3 cells (6 μm at dx=2μm).
            // Removes D3Q7 anisotropic ∇T noise that drives parasitic currents.
            dim3 blk(8, 8, 8);
            dim3 grd((config_.nx+7)/8, (config_.ny+7)/8, (config_.nz+7)/8);

            // 3-pass ping-pong: raw→T_sm, T_sm→fl_sm(temp), fl_sm→T_sm
            smoothField27Kernel<<<grd, blk>>>(
                temperature_raw, d_T_smoothed_,
                config_.nx, config_.ny, config_.nz);
            CUDA_CHECK_KERNEL();
            smoothField27Kernel<<<grd, blk>>>(
                d_T_smoothed_, d_fl_smoothed_,
                config_.nx, config_.ny, config_.nz);
            CUDA_CHECK_KERNEL();
            smoothField27Kernel<<<grd, blk>>>(
                d_fl_smoothed_, d_T_smoothed_,
                config_.nx, config_.ny, config_.nz);
            CUDA_CHECK_KERNEL();
            CUDA_CHECK(cudaDeviceSynchronize());

            // CRITICAL FIX: Do NOT pass liquid_fraction to Marangoni.
            //
            // The fl_gate (fl > 0.1) was killing the force at the gas-metal
            // free surface because gas-side interface cells have fl=0. This
            // trapped Marangoni flow sub-surface at the melting front instead
            // of the correct free surface.
            //
            // Marangoni must act at the VOF gas-metal interface (where ∇f ≠ 0),
            // NOT gated by the thermal phase state. Solid surface suppression
            // is already handled by Darcy damping (K > 0 where fl < 1).
            // Apply CSF compensation multiplier (default 4.0×) to counteract
            // the |∇f| integral deficit from discrete 2-cell VOF interface.
            float dsigma_compensated = config_.dsigma_dT * config_.marangoni_csf_multiplier;
            force_accumulator_->addMarangoniForce(
                d_T_smoothed_, fill_level, nullptr, normals,
                dsigma_compensated,
                config_.nx, config_.ny, config_.nz,
                config_.dx,
                1.0f);
        }
    }

    // 2d. Recoil pressure force (evaporation-driven)
    // NOTE: Intentionally moved to step 2h (after fl masking).
    // Reason: maskForceByLiquidFractionKernel (step 2g) fires whenever
    // enable_surface_tension || enable_marangoni, which is true in all
    // production runs.  This mask multiplied the recoil force by fl,
    // suppressing it at partially-melted interface cells (fl < 1) — the
    // exact location where recoil must be largest.  By placing the recoil
    // call after the mask, only Marangoni + surface-tension forces receive
    // the fl-suppression treatment, which is their intended semantic.
    // See: recoil_code_audit.md Rank-2 finding.

    // 2e. Darcy damping — semi-implicit treatment (NOT added to force arrays)
    //
    // Instead of adding explicit Darcy force F = -K·u (which causes NaN when
    // K → ∞ in solid regions), we compute the Darcy coefficient field K_LU
    // and pass it to computeMacroscopic() for semi-implicit velocity update:
    //   u = [Σ(ci·fi) + 0.5·F_other] / (ρ + 0.5·K)
    //
    // The collision step sees only F_other (no Darcy). The Darcy penalty
    // acts purely through the velocity definition. This is unconditionally
    // stable: when K → ∞, u → 0 smoothly.
    //
    // Reference: Voller & Prakash (1987), Brent et al. (1988)
    // 2e. Darcy damping with hydrodynamic smoothing + under-relaxation
    //
    // FIX: The Carman-Kozeny function K = C·(1-fl)²/(fl³+ε) is catastrophically
    // nonlinear — a single-cell fl difference of 0.01→0.1 creates 810× K gradient.
    // This shatters the velocity field in the mushy zone, feeding back into thermal
    // advection → more fl noise → exponential instability.
    //
    // Three-pronged stabilization:
    // 1. Use 27-point smoothed fl (not thermodynamic fl) for Darcy computation
    // 2. Under-relax K: K_new = α·K(fl_smooth) + (1-α)·K_old with α=0.3
    // 3. Thermodynamic fl remains pristine (no energy conservation violation)
    if (config_.enable_darcy && fluid_) {
        const float* liquid_fraction = nullptr;
        if (thermal_) {
            liquid_fraction = thermal_->getLiquidFraction();
        }
        if (!liquid_fraction) {
            liquid_fraction = d_liquid_fraction_static_;
        }

        if (liquid_fraction) {
            // Sprint-1 (2026-04-25): use raw thermodynamic fl for Darcy K instead
            // of the 27-point smoothed field. The smoothing was originally added
            // for stability, but it widened the mushy zone by ~3 cells laterally
            // and shifted the Darcy damping front away from the physical solidus.
            // That artificial widening directly contributes to LBM melt-pool W
            // being 110-150 % of Flow3D. The α=0.3 under-relaxation in Step 3
            // below provides sufficient temporal smoothing to keep K stable.
            //
            // (smoothField27Kernel call removed; d_fl_smoothed_ allocation kept
            // because it is still consumed by the Marangoni / surface-tension
            // gradient stencil at lines 2455-2459 above.)

            // Step 2: Compute Darcy K from RAW fl (with gas-phase exemption)
            force_accumulator_->computeDarcyCoefficientField(
                liquid_fraction,
                vof_ ? vof_->getFillLevel() : nullptr,
                config_.darcy_coefficient,
                config_.density, config_.dx, config_.dt);
            // Note: gas-phase zeroing (f < 0.05) is now inside the kernel itself.

            // Step 3: Under-relax K with previous step (α=0.3)
            int num_cells = config_.nx * config_.ny * config_.nz;
            const float darcy_alpha = 0.3f;
            float* d_darcy_K = const_cast<float*>(force_accumulator_->getDarcyCoefficient());
            if (d_darcy_K && d_darcy_K_prev_) {
                int threads = 256;
                int blocks = (num_cells + threads - 1) / threads;
                darcyUnderRelaxKernel<<<blocks, threads>>>(
                    d_darcy_K, d_darcy_K_prev_, darcy_alpha, num_cells);
                CUDA_CHECK_KERNEL();
                CUDA_CHECK(cudaDeviceSynchronize());

                // Save current K for next step's under-relaxation
                CUDA_CHECK(cudaMemcpy(d_darcy_K_prev_, d_darcy_K,
                                      num_cells * sizeof(float),
                                      cudaMemcpyDeviceToDevice));
            }
        }
    }

    // Step 2f: Smooth Marangoni/surface tension forces to remove sharp spatial
    // gradients that excite high-frequency (checkerboard) modes at low tau.
    // Applied in physical units before conversion to preserve force integral.
    if (config_.enable_marangoni || config_.enable_surface_tension) {
        force_accumulator_->smoothForceField(config_.nx, config_.ny, config_.nz, 2);
    }

    // Step 2g: Force phase masking — suppress surface forces in solid phase
    // Physical rationale: solid metal has yield strength and does not respond
    // to surface tension / Marangoni forces. Only liquid (fl > 0) should feel
    // these forces. Without this, cold powder creeps under spurious CSF forces.
    //
    // BUG-1 FIX (Night Protocol, 2026-04-26): the outer maskForceByLiquidFractionKernel
    // call was REMOVED.  It double-suppressed Marangoni in the mushy zone:
    //   - inner gate (force_accumulator.cu:323-329) already returns early when
    //     fl < 0.1 (cold powder), and ramps a smooth fl-gate factor in [0.1, 0.2];
    //   - the outer F·fl mask additionally multiplied the same force by fl,
    //     giving total suppression of fl·gate(fl) — at fl=0.5 → 50 % loss; at
    //     fl=0.15 → 92.5 % loss; at fl=0.95 → 5 % loss.
    // The mushy zone is exactly where trailing-edge back-flow needs Marangoni to
    // pump liquid into the centre groove (Phase-1 night-run measurement: side
    // ridges +6-8 μm correct, but centre Δh stuck at -16-20 μm).  Cross-confirmed
    // by cfd-cuda-architect + cfd-math-expert reports as a 10-25 % pool-width and
    // surface-flow regression.  Inner gate alone is the physically motivated
    // boundary; outer F·fl is a redundant safety blanket.

    // Step 2h. Recoil pressure force — added AFTER fl masking and force smoothing.
    //
    // BUG FIX (Rank 2 from recoil_code_audit.md):
    // Previously the recoil call was at step 2d, before step 2g which applies
    // maskForceByLiquidFractionKernel(fx, fy, fz, fl).  That mask fires whenever
    // enable_surface_tension || enable_marangoni (always true in production), and
    // it multiplied the ENTIRE force array — including recoil — by fl.  Interface
    // cells at the gas–metal surface have fl < 1 (partially melted), so the mask
    // suppressed recoil by up to 100× at onset.  Moving recoil here bypasses the
    // mask and the 2-iteration Marangoni smoothing (step 2f), both of which were
    // never intended to touch recoil forces.
    if (config_.enable_recoil_pressure && vof_ && thermal_) {
        const float* temperature = thermal_->getTemperature();
        const float* fill_level = vof_->getFillLevel();
        const float3* normals = vof_->getInterfaceNormals();

        if (temperature && fill_level && normals) {
            const float T_boil = config_.material.T_vaporization;
            const float L_v = config_.material.L_vaporization;
            // BUG FIX (Rank 4): source M from material.molar_mass (per-material
            // value, e.g. 0.0558 for 316L) rather than surface.molar_mass which
            // defaults to 0.0476 kg/mol (Ti6Al4V) and is never updated when the
            // material database is loaded.
            const float M = (config_.material.molar_mass > 0.0f)
                            ? config_.material.molar_mass
                            : config_.surface.molar_mass;
            const float P_atm = 101325.0f;

            force_accumulator_->addRecoilPressureForce(
                temperature, fill_level, normals,
                T_boil, L_v, M, P_atm,
                config_.recoil_coefficient,
                config_.recoil_smoothing_width,
                config_.recoil_max_pressure,
                config_.nx, config_.ny, config_.nz,
                config_.dx,
                config_.recoil_force_multiplier);
        }
    }

    // Step 3: Convert from physical units [N/m³] to lattice units
    force_accumulator_->convertToLatticeUnits(config_.dx, config_.dt, config_.density);

    // Step 4: Apply CFL limiting for numerical stability
    if (fluid_) {
        const float* vx = fluid_->getVelocityX();
        const float* vy = fluid_->getVelocityY();
        const float* vz = fluid_->getVelocityZ();

        if (config_.cfl_use_adaptive) {
            // Adaptive region-based CFL limiting (for keyhole simulations)
            const float* fill_level = vof_ ? vof_->getFillLevel() : nullptr;
            const float* liquid_fraction = nullptr;
            if (thermal_) {
                liquid_fraction = thermal_->getLiquidFraction();
            }
            if (!liquid_fraction) {
                liquid_fraction = d_liquid_fraction_static_;
            }

            if (fill_level && liquid_fraction) {
                force_accumulator_->applyCFLLimitingAdaptive(
                    vx, vy, vz,
                    fill_level, liquid_fraction,
                    config_.dx, config_.dt,
                    config_.cfl_v_target_interface,
                    config_.cfl_v_target_bulk,
                    config_.cfl_interface_threshold_lo,
                    config_.cfl_interface_threshold_hi,
                    config_.cfl_recoil_boost_factor,
                    config_.cfl_force_ramp_factor);
            } else {
                // Fallback to basic CFL if VOF not available
                force_accumulator_->applyCFLLimiting(
                    vx, vy, vz,
                    config_.dx, config_.dt,
                    config_.cfl_velocity_target,
                    config_.cfl_force_ramp_factor);
            }
        } else if (config_.cfl_use_gradual_scaling) {
            // Gradual CFL limiting (smooth transitions)
            force_accumulator_->applyCFLLimiting(
                vx, vy, vz,
                config_.dx, config_.dt,
                config_.cfl_velocity_target,
                config_.cfl_force_ramp_factor);
        }
        // Note: If both adaptive and gradual are disabled, no CFL limiting applied
    }

    // Step 5: Catastrophic fail-safe cap (0.38 LU)
    // LBM stability limit: cs = 1/√3 ≈ 0.577. Cap at 0.38 gives 34% margin.
    // Must match CFL adaptive target to avoid double-throttling.
    auto cap_stats = force_accumulator_->capPerCellVelocityIncrement(0.38f);

    // Log cap statistics (should be zero under normal EDM operation)
    static int step_counter = 0;
    step_counter++;
    if (cap_stats.num_capped > 0) {
        std::cout << "[CAP-WARNING] Step " << step_counter
                  << ": " << cap_stats.num_capped << " cells capped at CATASTROPHIC limit"
                  << " (" << (100.0f * cap_stats.num_capped / cap_stats.total_cells) << "%)"
                  << ", max_F_uncapped=" << cap_stats.max_uncapped_force
                  << " LU (cap=" << cap_stats.cap_threshold << ")"
                  << ", deleted_momentum=" << cap_stats.total_deleted_momentum << " LU"
                  << std::endl;
    }

    // Optional: Print force breakdown for debugging (first few calls only)
    static int diagnostic_call_count = 0;
    if (diagnostic_call_count < 5) {
        force_accumulator_->printForceBreakdown();
        diagnostic_call_count++;
    }

    // NOTE: d_force_x/y/z_ now directly point to ForceAccumulator's internal arrays
    // No need to copy - they're already the same memory!
    // Old code: cudaMemcpy(d_force_x_, force_accumulator_->getFx(), ...) is now redundant
}


// ============================================================================
// Diagnostics
// ============================================================================

float MultiphysicsSolver::getMaxVelocity() const {
    if (!fluid_) return 0.0f;

    int num_cells = config_.nx * config_.ny * config_.nz;

    // Compute velocity magnitude
    float* d_vmag;
    CUDA_CHECK(cudaMalloc(&d_vmag, num_cells * sizeof(float)));

    int threads = 256;
    int blocks = (num_cells + threads - 1) / threads;

    computeVelocityMagnitudeKernel<<<blocks, threads>>>(
        fluid_->getVelocityX(),
        fluid_->getVelocityY(),
        fluid_->getVelocityZ(),
        d_vmag, num_cells
    );
    CUDA_CHECK_KERNEL();
    CUDA_CHECK(cudaDeviceSynchronize());

    // Find maximum
    std::vector<float> h_vmag(num_cells);
    cudaMemcpy(h_vmag.data(), d_vmag, num_cells * sizeof(float),
              cudaMemcpyDeviceToHost);

    float max_vel_lattice = *std::max_element(h_vmag.begin(), h_vmag.end());

    CUDA_CHECK(cudaFree(d_vmag));

    // Convert from lattice units to physical units
    // v_phys = v_lattice * (dx / dt)
    return max_vel_lattice * (config_.dx / config_.dt);
}

float MultiphysicsSolver::getMaxMetalVelocity() const {
    if (!fluid_ || !vof_) return getMaxVelocity();

    int num_cells = config_.nx * config_.ny * config_.nz;

    std::vector<float> h_vx(num_cells), h_vy(num_cells), h_vz(num_cells), h_f(num_cells);
    // F-08 (code-audit pass 1): wrap with CUDA_CHECK so silent device-side
    // failures (OOM, invalid pointer) surface immediately instead of
    // returning zero-initialized buffers and producing v_max = 0 m/s.
    CUDA_CHECK(cudaMemcpy(h_vx.data(), fluid_->getVelocityX(), num_cells*sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_vy.data(), fluid_->getVelocityY(), num_cells*sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_vz.data(), fluid_->getVelocityZ(), num_cells*sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_f.data(), vof_->getFillLevel(), num_cells*sizeof(float), cudaMemcpyDeviceToHost));

    float max_v = 0;
    for (int i = 0; i < num_cells; i++) {
        if (h_f[i] > 0.01f) {
            float v = sqrtf(h_vx[i]*h_vx[i] + h_vy[i]*h_vy[i] + h_vz[i]*h_vz[i]);
            if (v > max_v) max_v = v;
        }
    }
    return max_v * (config_.dx / config_.dt);
}

float MultiphysicsSolver::getMaxTemperature() const {
    const float* d_temp = config_.enable_thermal ?
        thermal_->getTemperature() : d_temperature_static_;

    if (!d_temp) return 0.0f;

    int num_cells = config_.nx * config_.ny * config_.nz;
    std::vector<float> h_temp(num_cells);
    // F-08 (code-audit pass 1): same as getMaxMetalVelocity — surface CUDA
    // errors instead of silently returning 0 K.
    CUDA_CHECK(cudaMemcpy(h_temp.data(), d_temp, num_cells * sizeof(float),
                          cudaMemcpyDeviceToHost));

    return *std::max_element(h_temp.begin(), h_temp.end());
}

float MultiphysicsSolver::getMeltPoolDepth() const {
    if (!thermal_) return 0.0f;

    int nx = config_.domain.nx;
    int ny = config_.domain.ny;
    int nz = config_.domain.nz;
    int num_cells = nx * ny * nz;
    std::vector<float> h_temp(num_cells);
    cudaMemcpy(h_temp.data(), thermal_->getTemperature(),
               num_cells * sizeof(float), cudaMemcpyDeviceToHost);

    float T_solidus = config_.material.T_solidus;
    int surface_z = static_cast<int>(interface_z_);

    // Scan downward from surface to find deepest molten cell
    int deepest_molten_z = surface_z;
    for (int z = surface_z; z >= 0; z--) {
        bool any_molten = false;
        for (int y = 0; y < ny && !any_molten; y++) {
            for (int x = 0; x < nx && !any_molten; x++) {
                int idx = x + y * nx + z * nx * ny;
                if (h_temp[idx] > T_solidus) {
                    any_molten = true;
                    deepest_molten_z = z;
                }
            }
        }
        if (!any_molten) break;
    }

    return (interface_z_ - deepest_molten_z) * config_.dx;
}

float MultiphysicsSolver::getSurfaceProtrusion() const {
    if (!vof_) return 0.0f;

    int nx = config_.domain.nx;
    int ny = config_.domain.ny;
    int nz = config_.domain.nz;
    int num_cells = nx * ny * nz;
    std::vector<float> h_fill(num_cells);
    cudaMemcpy(h_fill.data(), vof_->getFillLevel(),
               num_cells * sizeof(float), cudaMemcpyDeviceToHost);

    float max_z = 0.0f;
    for (int z = nz - 1; z >= 0; z--) {
        for (int y = 0; y < ny; y++) {
            for (int x = 0; x < nx; x++) {
                int idx = x + y * nx + z * nx * ny;
                if (h_fill[idx] > 0.5f) {
                    float z_pos = z * config_.dx;
                    max_z = fmaxf(max_z, z_pos);
                }
            }
        }
    }

    float surface_pos = interface_z_ * config_.dx;
    return fmaxf(0.0f, max_z - surface_pos);
}

float MultiphysicsSolver::getTotalMass() const {
    if (!vof_) return 0.0f;
    return vof_->computeTotalMass();
}

bool MultiphysicsSolver::checkNaN() const {
    int num_cells = config_.nx * config_.ny * config_.nz;
    int* d_has_nan;
    CUDA_CHECK(cudaMalloc(&d_has_nan, sizeof(int)));
    CUDA_CHECK(cudaMemset(d_has_nan, 0, sizeof(int)));

    int threads = 256;
    int blocks = (num_cells + threads - 1) / threads;

    // Check velocity
    if (fluid_) {
        checkNaNKernel<<<blocks, threads>>>(fluid_->getVelocityX(), d_has_nan, num_cells);
        CUDA_CHECK_KERNEL();
        checkNaNKernel<<<blocks, threads>>>(fluid_->getVelocityY(), d_has_nan, num_cells);
        CUDA_CHECK_KERNEL();
        checkNaNKernel<<<blocks, threads>>>(fluid_->getVelocityZ(), d_has_nan, num_cells);
        CUDA_CHECK_KERNEL();
    }

    // Check temperature
    const float* d_temp = config_.enable_thermal ?
        thermal_->getTemperature() : d_temperature_static_;
    if (d_temp) {
        checkNaNKernel<<<blocks, threads>>>(d_temp, d_has_nan, num_cells);
        CUDA_CHECK_KERNEL();
    }

    // Check fill level
    if (vof_) {
        checkNaNKernel<<<blocks, threads>>>(vof_->getFillLevel(), d_has_nan, num_cells);
        CUDA_CHECK_KERNEL();
    }

    CUDA_CHECK(cudaDeviceSynchronize());

    int has_nan;
    CUDA_CHECK(cudaMemcpy(&has_nan, d_has_nan, sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_has_nan));

    return has_nan > 0;
}

float MultiphysicsSolver::checkMassConservation() const {
    if (!vof_ || initial_mass_ == 0.0f) return 0.0f;

    float current_mass = vof_->computeTotalMass();
    return std::abs(current_mass - initial_mass_) / initial_mass_;
}

// ============================================================================
// Ray Tracing Diagnostics
// ============================================================================

float MultiphysicsSolver::getRayTracingDepositedPower() const {
    if (!ray_tracing_laser_) return 0.0f;
    return ray_tracing_laser_->getDepositedPower();
}

float MultiphysicsSolver::getRayTracingInputPower() const {
    if (!ray_tracing_laser_) return 0.0f;
    return ray_tracing_laser_->getInputPower();
}

float MultiphysicsSolver::getRayTracingEffectiveAbsorptivity() const {
    if (!ray_tracing_laser_) return 0.0f;
    float input = ray_tracing_laser_->getInputPower();
    if (input < 1e-20f) return 0.0f;
    return ray_tracing_laser_->getDepositedPower() / input;
}

float MultiphysicsSolver::getRayTracingEnergyError() const {
    if (!ray_tracing_laser_) return 0.0f;
    return ray_tracing_laser_->getEnergyError();
}

// ============================================================================
// Energy Conservation Diagnostics
// ============================================================================

float MultiphysicsSolver::getLaserAbsorbedPower() const {
    if (!config_.enable_laser || !laser_) return 0.0f;

    // Check if laser is on
    if (config_.laser_shutoff_time >= 0.0f && current_time_ > config_.laser_shutoff_time) {
        return 0.0f;  // Laser is off
    }

    // ============================================================================
    // ENERGY BALANCE FIX: Compute actual deposited power by integrating Q
    // ============================================================================
    // Problem: Nominal power (P × A) != actual deposited power due to:
    //   1. Discretization error (Gaussian beam on finite grid)
    //   2. Domain truncation (beam extends beyond domain)
    //   3. Surface geometry (VOF interface position)
    //
    // Solution: Integrate actual volumetric heat source Q(x,y,z) over domain
    //   P_actual = ∫∫∫ Q(x,y,z) dV = Σ Q_i * dx³
    //
    // This matches the actual energy entering the thermal solver.
    // ============================================================================

    // Allocate temporary device memory for heat source (RAII guard for exception safety)
    struct CudaFreeGuard { float* p; ~CudaFreeGuard() { if (p) cudaFree(p); } };
    int num_cells = config_.nx * config_.ny * config_.nz;
    float* d_heat_source = nullptr;
    CUDA_CHECK(cudaMalloc(&d_heat_source, num_cells * sizeof(float)));
    CudaFreeGuard guard{d_heat_source};

    // Compute volumetric heat source from laser (same as applyLaserSource)
    dim3 threads(8, 8, 8);
    dim3 blocks(
        (config_.nx + threads.x - 1) / threads.x,
        (config_.ny + threads.y - 1) / threads.y,
        (config_.nz + threads.z - 1) / threads.z
    );

    float z_surface = interface_z_;
    const float* fill_ptr = vof_ ? vof_->getFillLevel() : nullptr;

    const float* T_diag = thermal_ ? thermal_->getTemperature() : nullptr;
    computeLaserHeatSourceKernel<<<blocks, threads>>>(
        d_heat_source,
        fill_ptr,
        T_diag,
        *laser_,
        config_.nx, config_.ny, config_.nz,
        config_.dx,
        z_surface
    );
    CUDA_CHECK_KERNEL();
    CUDA_CHECK(cudaDeviceSynchronize());

    // Integrate heat source over domain using CUB reduction
    float total_power = computeTotalLaserEnergy(
        d_heat_source,
        config_.dx, config_.dx, config_.dx,
        config_.nx, config_.ny, config_.nz
    );
    // d_heat_source freed automatically by CudaFreeGuard destructor

    return total_power;
}

float MultiphysicsSolver::getEvaporationPower() const {
    if (!config_.enable_evaporation_mass_loss) return 0.0f;
    if (!config_.enable_thermal || !thermal_) return 0.0f;
    if (!vof_ || !d_evap_mass_flux_) return 0.0f;

    // R7 OPENFOAM-ALIGNED: d_evap_mass_flux_ now stores J_vol [kg/(m³·s)]
    // weighted by |∇f|. Total evaporation power:
    //   P_evap = Σ (J_vol · V · L_vap)   where V = dx³
    int num_cells = config_.nx * config_.ny * config_.nz;
    std::vector<float> h_J_evap(num_cells);
    cudaMemcpy(h_J_evap.data(), d_evap_mass_flux_,
               num_cells * sizeof(float), cudaMemcpyDeviceToHost);

    float L_vap = config_.material.L_vaporization;  // [J/kg]
    float V = config_.dx * config_.dx * config_.dx; // [m³]

    float total_power = 0.0f;
    for (int i = 0; i < num_cells; ++i) {
        total_power += h_J_evap[i] * V * L_vap;  // [kg/(m³·s)] * [m³] * [J/kg] = [W]
    }

    return total_power;
}

float MultiphysicsSolver::getRadiationPower() const {
    if (!config_.enable_thermal || !thermal_) return 0.0f;
    if (!config_.enable_radiation_bc) return 0.0f;
    // BUG FIX (2026-01-26): Remove VOF requirement
    // Radiation power diagnostic works without VOF (fill_level not used in kernel)
    // Previously returned 0W for thermal-only simulations

    const float* fill_level = vof_ ? vof_->getFillLevel() : nullptr;
    return thermal_->computeRadiationPower(fill_level, config_.dx,
                                            config_.emissivity, config_.ambient_temperature);
}

float MultiphysicsSolver::getSubstratePower() const {
    if (!config_.enable_thermal || !thermal_) return 0.0f;
    if (!config_.enable_substrate_cooling) return 0.0f;

    return thermal_->computeSubstratePower(config_.dx,
                                            config_.substrate_h_conv,
                                            config_.substrate_temperature);
}

float MultiphysicsSolver::getThermalEnergyChangeRate() const {
    if (!config_.enable_thermal || !thermal_) return 0.0f;

    // ============================================================
    // BUG 3 FIX (Solution 2): Time-averaged dE/dt
    // ============================================================
    // Problem: Instantaneous dE/dt has high noise for small dt
    //   - dt=0.05μs: 22.8% error (noise dominates signal)
    //   - dt=0.10μs: 4.8% error (lucky sampling)
    //   - dt=0.20μs: 14.0% error (undersampling)
    //
    // Solution: Moving average with linear regression
    //   - Stores last N=10 energy measurements
    //   - Computes dE/dt as slope of E(t) via least-squares fit
    //   - Reduces noise by factor of ~3×
    //   - Smooths transient spikes
    // ============================================================

    // Compute current energy
    float current_energy = thermal_->computeTotalThermalEnergy(config_.dx);

    // Add to history
    energy_history_.push_back(current_energy);
    time_history_.push_back(current_time_);

    // Maintain fixed history size
    if (energy_history_.size() > ENERGY_HISTORY_SIZE) {
        energy_history_.erase(energy_history_.begin());
        time_history_.erase(time_history_.begin());
    }

    // Need at least 2 points for derivative
    if (energy_history_.size() < 2) {
        return 0.0f;
    }

    // ============================================================
    // Linear regression: dE/dt = slope of E(t)
    // ============================================================
    // Method: Least-squares fit to minimize Σ[E_i - (a + b*t_i)]²
    // Result: slope b = (n*Σ(t*E) - Σt*ΣE) / (n*Σ(t²) - (Σt)²)
    //
    // Benefits over simple difference:
    //   1. Robust to outliers (averages over N points)
    //   2. Accounts for non-uniform sampling
    //   3. Reduces noise by √N ≈ 3.2×
    // ============================================================

    float sum_t = 0.0f;
    float sum_E = 0.0f;
    float sum_tE = 0.0f;
    float sum_t2 = 0.0f;
    int n = energy_history_.size();

    for (int i = 0; i < n; ++i) {
        float t = time_history_[i];
        float E = energy_history_[i];
        sum_t += t;
        sum_E += E;
        sum_tE += t * E;
        sum_t2 += t * t;
    }

    // Compute slope (dE/dt) via least-squares
    float denominator = n * sum_t2 - sum_t * sum_t;
    if (std::abs(denominator) < 1e-12f) {
        // Degenerate case: all times identical (shouldn't happen)
        return 0.0f;
    }

    float dE_dt = (n * sum_tE - sum_t * sum_E) / denominator;

    return dE_dt;
}

float MultiphysicsSolver::computeBoundaryHeatFlux() const {
    // Week 1 Tuesday implementation: Substrate cooling BC is now the only
    // significant boundary heat flux. Lateral boundaries remain adiabatic.
    //
    // Boundary heat fluxes:
    // - Bottom (z=0): Substrate cooling (convective BC) - computed by getSubstratePower()
    // - Lateral (x,y): Adiabatic (∇T·n = 0) → P_cond ≈ 0
    // - Top (z=nz-1): Radiation + evaporation (handled separately)
    //
    // Total boundary conduction = P_substrate (bottom only)

    if (!config_.enable_thermal || !thermal_) {
        return 0.0f;
    }

    // Substrate cooling at bottom boundary
    float P_substrate = config_.enable_substrate_cooling ? getSubstratePower() : 0.0f;

    // Lateral boundaries are adiabatic (periodic or zero-flux)
    // so P_cond_lateral ≈ 0

    return P_substrate;
}

void MultiphysicsSolver::printEnergyBalance() {
    if (!config_.enable_thermal || !thermal_) {
        std::cout << "[WARNING] Energy balance called but thermal solver disabled\n";
        return;
    }

    // ============================================================================
    // BUG FIX (Dec 2, 2025): Use EnergyBalanceTracker's dE/dt computation
    // ============================================================================
    // PROBLEM: getThermalEnergyChangeRate() uses moving average with history,
    //          but returns 0 if called infrequently (< 2 points in history).
    //
    // SOLUTION: Use the energy balance tracker which is updated regularly.
    //           If computeEnergyBalance() hasn't been called this timestep,
    //           call it now. Otherwise, use the cached values.
    //
    // Note: We check if tracker is fresh by comparing time_last_computed_
    //       with current_time_.
    // ============================================================================

    // Compute energy balance if not already done this timestep
    if (time_last_computed_ < current_time_) {
        computeEnergyBalance();
    }

    const auto& balance = energy_tracker_.getCurrent();

    // Get power terms from balance (they're always fresh)
    float P_laser = balance.P_laser;
    float P_evap = balance.P_evaporation;
    float P_rad = balance.P_radiation;
    float P_substrate = balance.P_substrate;
    float P_gas_wipe = balance.P_gas_wipe;
    float P_boiling_cap = balance.P_boiling_cap;
    float dE_dt = balance.dE_dt_computed;

    // Temperature cap power: energy removed by T_boil-100K hard cap
    // Physically represents evaporative cooling self-limiting temperature
    float P_cap = 0.0f;
    if (thermal_ && config_.enable_evaporation_mass_loss) {
        P_cap = thermal_->computeCapPower(config_.dx, config_.dt);
    }

    // Energy balance: P_laser = P_evap + P_rad + P_sub + P_gw + P_bc + dE/dt
    float P_in = P_laser;
    float P_out = P_evap + P_rad + P_substrate + P_gas_wipe + P_boiling_cap;
    float P_storage = dE_dt;

    float balance_error = P_in - P_out - P_storage;
    float error_percent = (P_in > 1e-6f) ? (std::abs(balance_error) / P_in * 100.0f) : 0.0f;

    std::cout << "\n[ENERGY R6] Step=" << current_step_
              << ", t=" << current_time_*1e6 << " μs:" << std::endl;
    printf("  E_total=%.6e J\n", balance.E_total);
    printf("  P_laser=%.2f  P_evap=%.2f  P_rad=%.2f  P_sub=%.2f  P_gw=%.2f  P_bc=%.2f W\n",
           P_laser, P_evap, P_rad, P_substrate, P_gas_wipe, P_boiling_cap);
    printf("  P_out=%.2f  dE/dt=%.2f  balance=%.2f  err=%.1f%%\n",
           P_out, P_storage, balance_error, error_percent);

    if (error_percent < 5.0f) {
        std::cout << "  ✓ PASS (< 5%)" << std::endl;
    } else if (error_percent < 10.0f) {
        std::cout << "  ⚠ WARNING (target < 5%)" << std::endl;
    } else {
        std::cout << "  ✗ FAIL (exceeds 10%)" << std::endl;
    }
    std::cout << std::endl;
}

// ============================================================================
// Data Access
// ============================================================================

const float* MultiphysicsSolver::getTemperature() const {
    return config_.enable_thermal ? thermal_->getTemperature() : d_temperature_static_;
}

const float* MultiphysicsSolver::getLiquidFraction() const {
    // Return liquid fraction if thermal solver has phase change enabled
    return (thermal_ && thermal_->hasPhaseChange()) ? thermal_->getLiquidFraction() : d_liquid_fraction_static_;
}

const float* MultiphysicsSolver::getVelocityX() const {
    return fluid_ ? fluid_->getVelocityX() : nullptr;
}

const float* MultiphysicsSolver::getVelocityY() const {
    return fluid_ ? fluid_->getVelocityY() : nullptr;
}

const float* MultiphysicsSolver::getVelocityZ() const {
    return fluid_ ? fluid_->getVelocityZ() : nullptr;
}

const float* MultiphysicsSolver::getFillLevel() const {
    return vof_ ? vof_->getFillLevel() : nullptr;
}

float* MultiphysicsSolver::getFillLevelMutable() {
    return vof_ ? vof_->getFillLevel() : nullptr;
}

void MultiphysicsSolver::setFillLevel(const float* h_fill_level) {
    if (vof_ && h_fill_level) {
        int num_cells = config_.nx * config_.ny * config_.nz;
        float* d_fill = vof_->getFillLevel();
        CUDA_CHECK(cudaMemcpy(d_fill, h_fill_level, num_cells * sizeof(float), cudaMemcpyHostToDevice));
        // Update cell flags and interface normals after changing fill level
        vof_->convertCells();
        vof_->reconstructInterface();
        printf("[PowderBed] Fill level updated, %d cells modified\n", num_cells);
    }
}

const float* MultiphysicsSolver::getCurvature() const {
    return vof_ ? vof_->getCurvature() : nullptr;
}

const float* MultiphysicsSolver::getPressure() const {
    return fluid_ ? fluid_->getPressure() : nullptr;
}

void MultiphysicsSolver::copyVelocityToHost(float* ux, float* uy, float* uz) const {
    if (fluid_) {
        fluid_->copyVelocityToHost(ux, uy, uz);
    }
}

void MultiphysicsSolver::copyTemperatureToHost(float* temperature) const {
    if (config_.enable_thermal && thermal_) {
        thermal_->copyTemperatureToHost(temperature);
    } else if (d_temperature_static_) {
        int num_cells = config_.nx * config_.ny * config_.nz;
        cudaMemcpy(temperature, d_temperature_static_,
                  num_cells * sizeof(float), cudaMemcpyDeviceToHost);
    }
}

void MultiphysicsSolver::copyFillLevelToHost(float* fill_level) const {
    if (vof_) {
        vof_->copyFillLevelToHost(fill_level);
    }
}

// ============================================================================
// Week 3 P1: Comprehensive Energy Balance Tracking
// ============================================================================

void MultiphysicsSolver::computeEnergyBalance() {
    if (!config_.enable_thermal || !thermal_) {
        return;  // Energy tracking requires thermal solver
    }

    using namespace diagnostics;

    // Create snapshot for current timestep
    EnergyBalance balance;
    balance.time = current_time_;
    balance.step = current_step_;

    // ========================================================================
    // Compute state energies [J]
    // ========================================================================

    const float* T = thermal_->getTemperature();
    const float* f_liquid = getLiquidFraction();
    const float* ux = fluid_ ? fluid_->getVelocityX() : nullptr;
    const float* uy = fluid_ ? fluid_->getVelocityY() : nullptr;
    const float* uz = fluid_ ? fluid_->getVelocityZ() : nullptr;

    // ============================================================================
    // R6 FIX: Use ThermalFDM's own energy computation, avoid double-counting
    // ============================================================================
    // computeTotalThermalEnergy() returns:
    //   Σ(ρ(T) × cp(T) × T × dV) + Σ(fl × ρ × L_f × dV)
    // This ALREADY includes both sensible heat AND latent heat.
    //
    // Previous bug: E_latent was computed separately via computeLatentEnergy()
    // and ADDED to E_total on top of the latent term already inside E_thermal.
    // This double-counted latent energy, inflating dE/dt by ~dE_latent/dt.
    // ============================================================================
    float E_total_f = thermal_->computeTotalThermalEnergy(config_.dx);
    balance.E_thermal = static_cast<double>(E_total_f);

    // Kinetic energy: ∫ 0.5 ρ |u|² dV
    if (config_.enable_fluid && fluid_) {
        computeKineticEnergy(
            ux, uy, uz,
            config_.material.rho_liquid,
            config_.dx,
            config_.nx, config_.ny, config_.nz,
            d_energy_temp_
        );
        CUDA_CHECK(cudaMemcpy(&balance.E_kinetic, d_energy_temp_, sizeof(double), cudaMemcpyDeviceToHost));
    } else {
        balance.E_kinetic = 0.0;
    }

    // Latent energy: already included in E_thermal from computeTotalThermalEnergy().
    // Set to zero to avoid double-counting in E_total.
    balance.E_latent = 0.0;

    // Total energy
    balance.updateTotal();

    // ========================================================================
    // Compute power terms [W]
    // ========================================================================

    // R6 FIX: Use accumulated laser energy (actual deposited) instead of
    // recomputing from current position. This avoids phase error when the
    // laser moves between diagnostic calls.
    double dt_elapsed_for_power = (time_last_computed_ < 0.0f) ? config_.dt : (current_time_ - time_last_computed_);
    dt_elapsed_for_power = std::max(dt_elapsed_for_power, 1e-12);
    balance.P_laser = static_cast<double>(laser_energy_accumulated_) / dt_elapsed_for_power;
    balance.P_evaporation = getEvaporationPower();
    balance.P_radiation = getRadiationPower();
    balance.P_substrate = getSubstratePower();
    balance.P_convection = 0.0;  // TODO: Implement if needed

    // Gas wipe and boiling cap energy tracking (Round 5)
    // These return cumulative energy [J] since last call, and reset the counter.
    // Convert to power: P = E_accumulated / dt_elapsed
    double E_gas_wipe = 0.0, E_boiling_cap = 0.0;
    if (thermal_) {
        E_gas_wipe = thermal_->getGasWipeEnergyRemoved();
        E_boiling_cap = thermal_->getBoilingCapEnergyRemoved();
        double dt_power = dt_elapsed_for_power;
        balance.P_gas_wipe = E_gas_wipe / dt_power;
        balance.P_boiling_cap = E_boiling_cap / dt_power;
    } else {
        balance.P_gas_wipe = 0.0;
        balance.P_boiling_cap = 0.0;
    }

    // ========================================================================
    // Update tracker and compute error
    // ========================================================================
    // R6 FIX: Proper first-call initialization + real elapsed time
    //
    // On first call, E_total_prev_ = 0.0 (tracker constructor), so dE/dt is
    // computed from zero energy — meaningless. Fix: seed the tracker on first
    // call so dE/dt = 0 for the first interval (no history to compare against).
    //
    // Also: use actual elapsed time since last diagnostic, not config_.dt.
    // ========================================================================

    if (time_last_computed_ < 0.0f) {
        // First call: seed tracker, no dE/dt history available yet
        energy_tracker_.seed(balance);
    } else {
        double dt_elapsed = current_time_ - time_last_computed_;
        dt_elapsed = std::max(dt_elapsed, 1e-12);
        energy_tracker_.update(balance, dt_elapsed);
    }

    // Mark that energy balance was computed at this time
    time_last_computed_ = current_time_;

    // Reset laser energy accumulator for next diagnostic interval
    laser_energy_accumulated_ = 0.0;
}

void MultiphysicsSolver::writeEnergyBalanceHistory(const std::string& filename) const {
    energy_tracker_.writeToFile(filename);
}

const diagnostics::EnergyBalance& MultiphysicsSolver::getCurrentEnergyBalance() const {
    return energy_tracker_.getCurrent();
}

void MultiphysicsSolver::validate() const {
    const auto& material = config_.material;

    if (config_.enable_darcy && !config_.enable_phase_change) {
        std::cerr << "WARNING: enable_darcy=true without enable_phase_change=true. "
                  << "Darcy damping uses liquid_fraction which defaults to 1.0, "
                  << "making Darcy ineffective." << std::endl;
    }

    if (config_.enable_phase_change) {
        float alpha_material = material.k_liquid / (material.rho_liquid * material.cp_liquid);
        float alpha_config = config_.thermal_diffusivity;
        if (alpha_config > 0 &&
            std::abs(alpha_config - alpha_material) / alpha_material > 0.5f) {
            std::cerr << "WARNING: thermal_diffusivity=" << alpha_config
                      << " differs from material-derived value=" << alpha_material
                      << " by more than 50%." << std::endl;
        }
    }
}

} // namespace physics
} // namespace lbm
