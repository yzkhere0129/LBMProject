/**
 * @file ray_tracing_laser.cu
 * @brief CUDA kernels and host methods for geometric ray tracing laser
 *
 * Implements:
 *   - Gaussian beam discretization (inverse-CDF + golden angle sampling)
 *   - 3D-DDA (Amanatides-Woo) voxel traversal
 *   - VOF interface hit detection + Fresnel absorption + specular reflection
 *   - Energy deposition via atomicAdd
 *   - Energy conservation diagnostics via shared-memory reduction
 */

#include "physics/ray_tracing_laser.h"
#include "utils/cuda_check.h"
#include <cstdio>
#include <cmath>

namespace lbm {
namespace physics {

// ============================================================================
// Constants
// ============================================================================

static constexpr float GOLDEN_RATIO = 1.6180339887498949f;
static constexpr float TWO_PI       = 6.2831853071795865f;
static constexpr float RT_EPS       = 1e-30f;

// ============================================================================
// Kernel: Initialize rays with Gaussian beam profile
// ============================================================================

__global__ void initializeRaysKernel(
    Ray* rays,
    int num_rays,
    float laser_x,
    float laser_y,
    float spot_radius,
    float total_power,
    float spawn_z,
    float cutoff_radii)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_rays) return;

    // Gaussian beam CDF: F(r) = 1 - exp(-2r²/w₀²)
    // Cutoff at R_cut = cutoff_radii × w₀
    float R_cut = cutoff_radii * spot_radius;
    float F_cut = 1.0f - expf(-2.0f * R_cut * R_cut / (spot_radius * spot_radius));

    // Inverse CDF sampling: stratified u ∈ (0, F_cut)
    float u = (tid + 0.5f) / num_rays * F_cut;
    float r = spot_radius * sqrtf(-0.5f * logf(1.0f - u));

    // Golden-angle azimuthal distribution (low-discrepancy)
    float theta = TWO_PI * fmodf(tid * GOLDEN_RATIO, 1.0f);

    Ray ray;
    ray.pos = make_float3(laser_x + r * cosf(theta),
                          laser_y + r * sinf(theta),
                          spawn_z);
    ray.dir = make_float3(0.0f, 0.0f, -1.0f);
    ray.energy = total_power * F_cut / num_rays;  // Equal power per ray
    ray.bounces = 0;
    ray.active = 1;

    rays[tid] = ray;
}

// ============================================================================
// Device helper: Fresnel absorptivity (angle-dependent, real metal)
// ============================================================================

__device__ inline float fresnelAbsorptivity(float cos_theta, float n_r, float k_r) {
    // Full complex-index Fresnel for metals — Born & Wolf, Optics §13.2.
    // Sprint-1 (2026-04-25): added k_r (extinction coefficient). At normal
    // incidence with (n=2.96, k=4.01) for 316L @1064 nm:
    //   R₀ = ((n-1)² + k²) / ((n+1)² + k²) = 0.685
    //   α₀ = 1 - R₀ = 0.315
    // Multi-bounce inside a keyhole cavity accumulates effective absorption
    // toward Flow3D's calibrated 70 % value. Without k, R₀ degenerated to
    // ((n-1)/(n+1))² = 0.495 with α₀ = 0.505, badly miscounting reflections.
    //
    // Full angle-dependent formula (Born & Wolf 13.2.13):
    //   Let p = n²-k²-sin²θ;  q = 4 n² k².
    //   2a² = sqrt(p² + q) + p
    //   2b² = sqrt(p² + q) - p
    //   R_s = ((a-cosθ)² + b²) / ((a+cosθ)² + b²)
    //   R_p = R_s · ((a-sinθ·tanθ)² + b²) / ((a+sinθ·tanθ)² + b²)
    //   α   = 1 - 0.5·(R_s + R_p)
    float c = fabsf(cos_theta);
    if (c > 1.0f) c = 1.0f;
    float s2 = 1.0f - c*c;          // sin²θ
    float p  = n_r*n_r - k_r*k_r - s2;
    float q  = 4.0f * n_r*n_r * k_r*k_r;
    float disc = sqrtf(fmaxf(p*p + q, 0.0f));
    float a2 = 0.5f * (disc + p);
    float b2 = 0.5f * (disc - p);
    if (a2 < 0.0f) a2 = 0.0f;
    if (b2 < 0.0f) b2 = 0.0f;
    float a  = sqrtf(a2);
    float b2_       = b2;          // already a²,b² are squared; Rs/Rp use them directly
    // R_s
    float Rs_num = (a - c)*(a - c) + b2_;
    float Rs_den = (a + c)*(a + c) + b2_ + RT_EPS;
    float Rs = Rs_num / Rs_den;
    // R_p — use sinθ·tanθ = s²/c (avoids tan blow-up at grazing; falls back to Rs at c→0)
    float st = (c > RT_EPS) ? (s2 / c) : 0.0f;
    float Rp_num = (a - st)*(a - st) + b2_;
    float Rp_den = (a + st)*(a + st) + b2_ + RT_EPS;
    float Rp_factor = Rp_num / Rp_den;
    float Rp = Rs * Rp_factor;
    float alpha = 1.0f - 0.5f * (Rs + Rp);
    if (alpha < 0.0f) alpha = 0.0f;
    if (alpha > 1.0f) alpha = 1.0f;
    return alpha;
}

// ============================================================================
// Kernel: Trace rays through VOF field using 3D-DDA (Amanatides-Woo)
//
// Each thread traces one ray. Outer loop handles bounces (re-init DDA
// per bounce). Inner loop is the DDA traversal for one ray segment.
// ============================================================================

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
    float fresnel_k,
    int max_bounces,
    int max_dda_steps,
    float energy_cutoff,
    int nx, int ny, int nz,
    float dx)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    // Note: caller launches exactly ceil(num_rays/blockSize) blocks,
    // but rays array may be shorter. Guard with ray.active check.

    Ray ray = rays[tid];
    if (!ray.active) {
        d_deposited[tid] = 0.0f;
        d_escaped[tid] = 0.0f;
        return;
    }

    float initial_energy = ray.energy;
    float total_deposited = 0.0f;
    float inv_dx = 1.0f / dx;

    // Domain AABB bounds [m]
    float box_min_x = 0.0f, box_min_y = 0.0f, box_min_z = 0.0f;
    float box_max_x = nx * dx, box_max_y = ny * dx, box_max_z = nz * dx;

    // Outer loop: one iteration per bounce (re-init DDA after reflection)
    while (ray.active) {
        // Save segment origin & direction (needed to compute hit position
        // AFTER reflection overwrites ray.dir)
        float3 seg_origin = ray.pos;
        float3 seg_dir    = ray.dir;

        // ----- Advance ray to domain entry if starting outside -----
        // Ray-AABB intersection (slab method)
        {
            float tmin = -1e30f, tmax = 1e30f;

            // X slab
            if (seg_dir.x != 0.0f) {
                float t1 = (box_min_x - seg_origin.x) / seg_dir.x;
                float t2 = (box_max_x - seg_origin.x) / seg_dir.x;
                if (t1 > t2) { float tmp = t1; t1 = t2; t2 = tmp; }
                tmin = fmaxf(tmin, t1);
                tmax = fminf(tmax, t2);
            } else if (seg_origin.x < box_min_x || seg_origin.x >= box_max_x) {
                ray.active = 0; break;  // Parallel and outside
            }

            // Y slab
            if (seg_dir.y != 0.0f) {
                float t1 = (box_min_y - seg_origin.y) / seg_dir.y;
                float t2 = (box_max_y - seg_origin.y) / seg_dir.y;
                if (t1 > t2) { float tmp = t1; t1 = t2; t2 = tmp; }
                tmin = fmaxf(tmin, t1);
                tmax = fminf(tmax, t2);
            } else if (seg_origin.y < box_min_y || seg_origin.y >= box_max_y) {
                ray.active = 0; break;
            }

            // Z slab
            if (seg_dir.z != 0.0f) {
                float t1 = (box_min_z - seg_origin.z) / seg_dir.z;
                float t2 = (box_max_z - seg_origin.z) / seg_dir.z;
                if (t1 > t2) { float tmp = t1; t1 = t2; t2 = tmp; }
                tmin = fmaxf(tmin, t1);
                tmax = fminf(tmax, t2);
            } else if (seg_origin.z < box_min_z || seg_origin.z >= box_max_z) {
                ray.active = 0; break;
            }

            if (tmin > tmax || tmax < 0.0f) {
                ray.active = 0; break;  // Ray misses domain entirely
            }

            // If ray starts outside domain, advance to entry point
            if (tmin > 0.0f) {
                float nudge_in = 1e-4f * dx;  // Tiny nudge inside
                seg_origin.x += (tmin + nudge_in) * seg_dir.x;
                seg_origin.y += (tmin + nudge_in) * seg_dir.y;
                seg_origin.z += (tmin + nudge_in) * seg_dir.z;
                ray.pos = seg_origin;
            }
        }

        // ----- DDA Initialization -----
        float gx = seg_origin.x * inv_dx;
        float gy = seg_origin.y * inv_dx;
        float gz = seg_origin.z * inv_dx;

        int i = (int)floorf(gx);
        int j = (int)floorf(gy);
        int k = (int)floorf(gz);

        // Clamp to valid range (edge case from float precision)
        i = max(0, min(i, nx - 1));
        j = max(0, min(j, ny - 1));
        k = max(0, min(k, nz - 1));

        int step_i = (seg_dir.x >= 0.0f) ? 1 : -1;
        int step_j = (seg_dir.y >= 0.0f) ? 1 : -1;
        int step_k = (seg_dir.z >= 0.0f) ? 1 : -1;

        // Parametric distance to cross one cell width in each axis
        float tDelta_x = (seg_dir.x != 0.0f) ? fabsf(dx / seg_dir.x) : 1e30f;
        float tDelta_y = (seg_dir.y != 0.0f) ? fabsf(dx / seg_dir.y) : 1e30f;
        float tDelta_z = (seg_dir.z != 0.0f) ? fabsf(dx / seg_dir.z) : 1e30f;

        // Parametric t to reach next cell boundary from seg_origin
        float next_x = ((seg_dir.x >= 0.0f) ? (i + 1) : i) * dx;
        float next_y = ((seg_dir.y >= 0.0f) ? (j + 1) : j) * dx;
        float next_z = ((seg_dir.z >= 0.0f) ? (k + 1) : k) * dx;

        float tMax_x = (seg_dir.x != 0.0f) ? (next_x - seg_origin.x) / seg_dir.x : 1e30f;
        float tMax_y = (seg_dir.y != 0.0f) ? (next_y - seg_origin.y) / seg_dir.y : 1e30f;
        float tMax_z = (seg_dir.z != 0.0f) ? (next_z - seg_origin.z) / seg_dir.z : 1e30f;

        // Edge case: ray starts exactly on cell boundary → tMax can be 0 or negative
        if (tMax_x < 1e-10f) tMax_x += tDelta_x;
        if (tMax_y < 1e-10f) tMax_y += tDelta_y;
        if (tMax_z < 1e-10f) tMax_z += tDelta_z;

        float t_current = 0.0f;
        float prev_fill = 0.0f;  // Gas outside domain
        bool hit = false;

        // ----- DDA Traversal (inner loop) -----
        for (int dda_step = 0; dda_step < max_dda_steps; ++dda_step) {
            // Bounds check → ray escapes domain
            if (i < 0 || i >= nx || j < 0 || j >= ny || k < 0 || k >= nz) {
                ray.active = 0;
                break;
            }

            int idx = i + nx * (j + ny * k);
            float fill = d_fill_level[idx];

            // --- Interface hit: gas (f<0.5) → metal (f≥0.5) transition ---
            if (prev_fill < 0.5f && fill >= 0.5f) {
                // Fetch interface normal
                float3 n;
                if (d_normals != nullptr) {
                    n = d_normals[idx];
                } else {
                    n = make_float3(0.0f, 0.0f, 1.0f);
                }

                float n_mag = sqrtf(n.x * n.x + n.y * n.y + n.z * n.z);
                if (n_mag < 1e-6f) {
                    // Degenerate normal → fallback to upward (typical free surface)
                    n = make_float3(0.0f, 0.0f, 1.0f);
                } else {
                    n.x /= n_mag;
                    n.y /= n_mag;
                    n.z /= n_mag;
                }

                // Orient normal toward incoming ray: need dot(d, n) < 0
                float cos_dn = seg_dir.x * n.x + seg_dir.y * n.y + seg_dir.z * n.z;
                if (cos_dn > 0.0f) {
                    n.x = -n.x; n.y = -n.y; n.z = -n.z;
                    cos_dn = -cos_dn;
                }

                // --- Absorption ---
                float alpha = use_fresnel
                    ? fresnelAbsorptivity(cos_dn, fresnel_n, fresnel_k)
                    : absorptivity;

                float E_absorbed = ray.energy * alpha;
                float q_vol = E_absorbed / (dx * dx * dx);  // [W/m³]
                atomicAdd(&d_heat_source[idx], q_vol);
                total_deposited += E_absorbed;
                ray.energy -= E_absorbed;

                // --- Energy cutoff: deposit remainder and stop ---
                if (ray.energy < energy_cutoff * initial_energy) {
                    float q_remain = ray.energy / (dx * dx * dx);
                    atomicAdd(&d_heat_source[idx], q_remain);
                    total_deposited += ray.energy;
                    ray.energy = 0.0f;
                    ray.active = 0;
                    break;
                }

                // --- Specular reflection: r = d - 2(d·n)n ---
                float3 new_dir;
                new_dir.x = seg_dir.x - 2.0f * cos_dn * n.x;
                new_dir.y = seg_dir.y - 2.0f * cos_dn * n.y;
                new_dir.z = seg_dir.z - 2.0f * cos_dn * n.z;

                // Re-normalize for numerical safety
                float d_mag = sqrtf(new_dir.x * new_dir.x +
                                    new_dir.y * new_dir.y +
                                    new_dir.z * new_dir.z);
                if (d_mag > 1e-10f) {
                    new_dir.x /= d_mag;
                    new_dir.y /= d_mag;
                    new_dir.z /= d_mag;
                }

                // Hit position from segment parametrics, then nudge along reflected dir
                float3 hit_pos;
                hit_pos.x = seg_origin.x + t_current * seg_dir.x;
                hit_pos.y = seg_origin.y + t_current * seg_dir.y;
                hit_pos.z = seg_origin.z + t_current * seg_dir.z;

                float nudge = 0.1f * dx;
                ray.pos.x = hit_pos.x + nudge * new_dir.x;
                ray.pos.y = hit_pos.y + nudge * new_dir.y;
                ray.pos.z = hit_pos.z + nudge * new_dir.z;
                ray.dir = new_dir;

                ray.bounces++;
                if (ray.bounces >= max_bounces) {
                    ray.active = 0;
                }

                hit = true;
                break;  // Exit DDA inner loop → re-init DDA from new pos/dir
            }

            prev_fill = fill;

            // --- DDA step: advance to next voxel along smallest tMax ---
            if (tMax_x <= tMax_y && tMax_x <= tMax_z) {
                t_current = tMax_x;
                i += step_i;
                tMax_x += tDelta_x;
            } else if (tMax_y <= tMax_z) {
                t_current = tMax_y;
                j += step_j;
                tMax_y += tDelta_y;
            } else {
                t_current = tMax_z;
                k += step_k;
                tMax_z += tDelta_z;
            }
        } // end DDA inner loop

        // DDA step limit exhausted without hit or escape → terminate
        if (!hit && ray.active) {
            ray.active = 0;
        }
    } // end bounce loop

    d_deposited[tid] = total_deposited;
    d_escaped[tid] = ray.energy;
}

// ============================================================================
// Host methods
// ============================================================================

RayTracingLaser::RayTracingLaser(const RayTracingConfig& config,
                                 int nx, int ny, int nz, float dx)
    : config_(config), nx_(nx), ny_(ny), nz_(nz), dx_(dx),
      d_rays_(config.num_rays),
      d_deposited_(config.num_rays),
      d_escaped_(config.num_rays)
{
    if (config_.normal_smoothing > 0) {
        d_smoothed_fill_.reset(nx * ny * nz);
    }
    printf("[RayTracing] Initialized: %d rays, max_bounces=%d, alpha=%.3f, "
           "cutoff=%.3f, dx=%.2e m\n",
           config_.num_rays, config_.max_bounces, config_.absorptivity,
           config_.energy_cutoff, dx_);
}

void RayTracingLaser::traceAndDeposit(const float* d_fill_level,
                                       const float3* d_normals,
                                       const LaserSource& laser,
                                       float* d_heat_source)
{
    int N = config_.num_rays;
    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;

    // Spawn Z: above domain top
    float spawn_z = (nz_ + config_.spawn_margin_cells) * dx_;

    // Initialize rays with Gaussian beam profile
    initializeRaysKernel<<<gridSize, blockSize>>>(
        d_rays_.get(), N,
        laser.x0, laser.y0,
        laser.spot_radius, laser.power,
        spawn_z, config_.cutoff_radii
    );
    CUDA_CHECK(cudaGetLastError());

    // Compute input power for diagnostics (power within cutoff radius)
    float R_cut = config_.cutoff_radii * laser.spot_radius;
    float F_cut = 1.0f - expf(-2.0f * R_cut * R_cut /
                               (laser.spot_radius * laser.spot_radius));
    input_power_ = laser.power * F_cut;

    // Zero per-ray diagnostic arrays
    d_deposited_.zero();
    d_escaped_.zero();

    // No VOF field → all rays escape (no surface to hit)
    if (d_fill_level == nullptr) {
        CUDA_CHECK(cudaDeviceSynchronize());
        deposited_power_ = 0.0f;
        escaped_power_ = input_power_;
        return;
    }

    // Trace all rays
    traceRaysKernel<<<gridSize, blockSize>>>(
        d_rays_.get(),
        d_fill_level,
        d_normals,
        d_heat_source,
        d_deposited_.get(),
        d_escaped_.get(),
        config_.absorptivity,
        config_.use_fresnel,
        config_.fresnel_n_refract,
        config_.fresnel_k_extinct,
        config_.max_bounces,
        config_.max_dda_steps,
        config_.energy_cutoff,
        nx_, ny_, nz_, dx_
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Energy conservation diagnostics
    deposited_power_ = reduceSum(d_deposited_.get(), N);
    escaped_power_   = reduceSum(d_escaped_.get(), N);
}

float RayTracingLaser::getEnergyError() const {
    if (input_power_ < 1e-20f) return 0.0f;
    return fabsf(deposited_power_ + escaped_power_ - input_power_) / input_power_;
}

// ============================================================================
// GPU reduction (shared-memory tree, no CUB dependency)
// ============================================================================

__global__ void reduceSumKernel(const float* input, float* output, int n) {
    extern __shared__ float sdata[];

    int tid_local = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid_local;

    sdata[tid_local] = (gid < n) ? input[gid] : 0.0f;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid_local < s) {
            sdata[tid_local] += sdata[tid_local + s];
        }
        __syncthreads();
    }

    if (tid_local == 0) {
        atomicAdd(output, sdata[0]);
    }
}

float RayTracingLaser::reduceSum(const float* d_array, int n) {
    float* d_result;
    CUDA_CHECK(cudaMalloc(&d_result, sizeof(float)));
    CUDA_CHECK(cudaMemset(d_result, 0, sizeof(float)));

    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;
    reduceSumKernel<<<gridSize, blockSize, blockSize * sizeof(float)>>>(
        d_array, d_result, n);
    CUDA_CHECK(cudaGetLastError());

    float result;
    CUDA_CHECK(cudaMemcpy(&result, d_result, sizeof(float), cudaMemcpyDeviceToHost));
    cudaFree(d_result);
    return result;
}

} // namespace physics
} // namespace lbm
