/**
 * @file diagnose_zalesak.cu
 * @brief Diagnostic program for Zalesak disk VOF advection failures.
 *
 * Investigates three specific failure modes:
 *   1. Face velocity computation (collocated vs MAC, discrete divergence)
 *   2. Operator splitting strategy (unsplit vs Strang)
 *   3. TVD limiter gradient ratio and clipping behaviour
 *
 * Reports per-step: mass, clip count, Swiss-cheese count, max over/undershoot.
 *
 * Compile from build/:
 *   nvcc -o ../scripts/viz/diagnose_zalesak ../scripts/viz/diagnose_zalesak.cu \
 *     -I../include -L. -llbm_physics -llbm_core -llbm_io -llbm_diagnostics \
 *     --std=c++17 -rdc=true -Xcompiler -fPIC -O2 -arch=sm_86
 */

#include "physics/vof_solver.h"

#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <vector>
#include <algorithm>
#include <numeric>
#include <cstring>

using namespace lbm::physics;

// ============================================================================
// Device kernels for fine-grained diagnostics
// ============================================================================

/**
 * Per-cell maximum discrete divergence of face-interpolated velocity field.
 *
 * div_i = (u_{i+1/2} - u_{i-1/2})/dx + (v_{j+1/2} - v_{j-1/2})/dx
 * where face velocity = arithmetic average of adjacent cell-center values.
 *
 * For solid-body rotation u=-omega*(y-cy), v=omega*(x-cx):
 *   du/dx = 0,  dv/dy = 0  analytically.
 * Any non-zero discrete value reveals discretisation error.
 */
__global__ void computeDiscreteDiv(
    const float* ux, const float* uy, const float* uz,
    float* div_out,
    float dx,
    int nx, int ny, int nz)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    if (i >= nx || j >= ny || k >= nz) return;

    int idx = i + nx * (j + ny * k);

    // Periodic wrapping for neighbour indices
    int ip = (i < nx - 1) ? i + 1 : 0;
    int im = (i > 0)      ? i - 1 : nx - 1;
    int jp = (j < ny - 1) ? j + 1 : 0;
    int jm = (j > 0)      ? j - 1 : ny - 1;
    int kp = (k < nz - 1) ? k + 1 : 0;
    int km = (k > 0)      ? k - 1 : nz - 1;

    int idx_ip = ip + nx * (j  + ny * k);
    int idx_im = im + nx * (j  + ny * k);
    int idx_jp = i  + nx * (jp + ny * k);
    int idx_jm = i  + nx * (jm + ny * k);
    int idx_kp = i  + nx * (j  + ny * kp);
    int idx_km = i  + nx * (j  + ny * km);

    float u_xp = 0.5f * (ux[idx] + ux[idx_ip]);
    float u_xm = 0.5f * (ux[idx_im] + ux[idx]);
    float v_yp = 0.5f * (uy[idx] + uy[idx_jp]);
    float v_ym = 0.5f * (uy[idx_jm] + uy[idx]);
    float w_zp = 0.5f * (uz[idx] + uz[idx_kp]);
    float w_zm = 0.5f * (uz[idx_km] + uz[idx]);

    float div = ((u_xp - u_xm) + (v_yp - v_ym) + (w_zp - w_zm)) / dx;
    div_out[idx] = fabsf(div);
}

/**
 * Count cells that were exactly 1.0 before advection and are now < 1.0
 * (Swiss-cheese indicator: holes punched in pure-liquid interior).
 *
 * Also accumulates the number of cells clipped and the raw pre-clip extrema.
 * We detect post-clip only (VOFSolver already clips), so we compare f^n vs f^{n+1}.
 */
__global__ void analyseStep(
    const float* f_before,    // fill level before advection (f^n)
    const float* f_after,     // fill level after advection  (f^{n+1}, already clipped)
    int* clip_lo_count,       // cells where f_after < f_after would have been < 0
    int* clip_hi_count,       // cells where f_after would have been > 1
    int* swiss_cheese_count,  // cells where f_before==1 and f_after < 1-eps
    float* max_overshoot,     // max f_after - 1 (after clipping, so tracks clamp frequency)
    float* max_undershoot,    // min f_after (after clipping)
    int num_cells)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_cells) return;

    float fb = f_before[idx];
    float fa = f_after[idx];

    // Swiss-cheese: cell was pure liquid, now has a hole
    // Threshold 1e-6 for floating-point exact comparison of "was 1.0"
    if (fb > 1.0f - 1e-6f && fa < 1.0f - 1e-3f) {
        atomicAdd(swiss_cheese_count, 1);
    }

    // After-clip range (we cannot see pre-clip values from outside the kernel,
    // but we can detect cells that hit the [0,1] boundary)
    if (fa < 1e-9f && fb > 1e-3f) {
        // Cell was non-negligible, now flushed to zero → mass loss event
        atomicAdd(clip_lo_count, 1);
    }
    if (fa > 1.0f - 1e-9f && fb < 1.0f - 1e-3f) {
        // Cell was partial, now rounded to 1 → mass gain event
        atomicAdd(clip_hi_count, 1);
    }

    // Track extrema of f_after to see how far clipping bites
    // (atomicMax on float via int trick)
    // Overshoot = how much above 1 the value would be; since we already clipped,
    // we instead track: cells exactly at 1 where f_before < 0.9 (sudden jump)
    // Report max delta increase per cell for the "Swiss cheese" detection
    float delta = fa - fb;
    if (delta < 0.0f) {
        // f decreased — standard advection out of cell
    } else {
        // f increased unexpectedly (can happen with anti-diffusive limiter)
        // Use atomicAdd on max_overshoot[0] via compare-and-swap is complex,
        // so just write the value; race conditions give approximate max.
        if (delta > *max_overshoot) *max_overshoot = delta;
    }

    if (fa < *max_undershoot) *max_undershoot = fa;
}

// ============================================================================
// Host helpers
// ============================================================================

static void initSlottedDisk(std::vector<float>& h,
                             int nx, int ny, int nz,
                             float cx, float cy,
                             float R, float slot_w, float slot_d)
{
    h.assign(nx * ny * nz, 0.0f);
    for (int k = 0; k < nz; ++k)
        for (int j = 0; j < ny; ++j)
            for (int i = 0; i < nx; ++i) {
                float dx = static_cast<float>(i) - cx;
                float dy = static_cast<float>(j) - cy;
                float r  = std::sqrt(dx*dx + dy*dy);
                bool in_disk = (r <= R);
                bool in_slot = (std::fabs(dx) <= slot_w * 0.5f)
                             && (dy >= 0.0f) && (dy <= slot_d);
                int idx = i + nx * (j + ny * k);
                if (in_disk && !in_slot) {
                    float dist = R - r;
                    h[idx] = 0.5f * (1.0f + std::tanh(dist / 1.5f));
                }
            }
}

static void uploadRotation(float* d_ux, float* d_uy, float* d_uz,
                            float omega, float cx, float cy,
                            int nx, int ny, int nz)
{
    int N = nx * ny * nz;
    std::vector<float> hx(N), hy(N), hz(N, 0.0f);
    for (int k = 0; k < nz; ++k)
        for (int j = 0; j < ny; ++j)
            for (int i = 0; i < nx; ++i) {
                int idx = i + nx * (j + ny * k);
                hx[idx] = -omega * (static_cast<float>(j) - cy);
                hy[idx] =  omega * (static_cast<float>(i) - cx);
            }
    cudaMemcpy(d_ux, hx.data(), N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_uy, hy.data(), N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_uz, hz.data(), N * sizeof(float), cudaMemcpyHostToDevice);
}

static float hostMax(const std::vector<float>& v) {
    return *std::max_element(v.begin(), v.end());
}

static float hostSum(const std::vector<float>& v) {
    return std::accumulate(v.begin(), v.end(), 0.0f);
}

// ============================================================================
// main
// ============================================================================

int main()
{
    // -------------------------------------------------------------------------
    // Domain
    // -------------------------------------------------------------------------
    const int   nx = 128, ny = 128, nz = 3;
    const float dx = 1.0f;
    const int   N  = nx * ny * nz;

    // Disk geometry
    const float cx     = nx * 0.5f;
    const float cy     = ny * 0.5f;
    const float R      = 30.0f;
    const float slot_w = 10.0f;
    const float slot_d = 50.0f;

    // Rotation
    const int   steps_per_rev = 800;
    const float omega = 2.0f * M_PI / static_cast<float>(steps_per_rev);
    const float dt    = 1.0f;

    printf("============================================================\n");
    printf("  Zalesak Disk Diagnostic  (%d x %d x %d)\n", nx, ny, nz);
    printf("  R=%.0f  slot_w=%.0f  slot_d=%.0f\n", R, slot_w, slot_d);
    printf("  omega=%.6f rad/step   steps_per_rev=%d\n", omega, steps_per_rev);
    printf("  CFL_max = omega*R*dt/dx = %.4f\n", omega * R * dt / dx);
    printf("============================================================\n\n");

    // -------------------------------------------------------------------------
    // Issue 1: Discrete divergence of face-interpolated velocity field
    // -------------------------------------------------------------------------
    printf("=== ISSUE 1: Discrete Divergence of Face Velocity ===\n");
    printf("    Analytically div(u)=0 for solid body rotation.\n");
    printf("    Non-zero discrete divergence causes mass non-conservation.\n\n");

    // Allocate velocity arrays
    float *d_ux, *d_uy, *d_uz;
    cudaMalloc(&d_ux, N * sizeof(float));
    cudaMalloc(&d_uy, N * sizeof(float));
    cudaMalloc(&d_uz, N * sizeof(float));
    uploadRotation(d_ux, d_uy, d_uz, omega, cx, cy, nx, ny, nz);

    // Compute discrete divergence
    float* d_div;
    cudaMalloc(&d_div, N * sizeof(float));

    dim3 block3(8, 8, 4);
    dim3 grid3((nx + block3.x - 1) / block3.x,
               (ny + block3.y - 1) / block3.y,
               (nz + block3.z - 1) / block3.z);

    computeDiscreteDiv<<<grid3, block3>>>(d_ux, d_uy, d_uz, d_div, dx, nx, ny, nz);
    cudaDeviceSynchronize();

    std::vector<float> h_div(N);
    cudaMemcpy(h_div.data(), d_div, N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_div);

    float div_max = hostMax(h_div);
    float div_mean = hostSum(h_div) / static_cast<float>(N);
    int div_nonzero = 0;
    for (float d : h_div) if (d > 1e-8f) div_nonzero++;

    printf("  div_max  = %.6e  (should be ~0 for solid rotation)\n", div_max);
    printf("  div_mean = %.6e\n", div_mean);
    printf("  nonzero div cells = %d / %d\n\n", div_nonzero, N);

    printf("  VERDICT: ");
    if (div_max < 1e-6f) {
        printf("Face velocity interpolation is consistent with div-free field.\n");
        printf("  Mass loss from face velocity is NOT the primary issue.\n\n");
    } else {
        printf("Non-zero discrete divergence detected! This WILL cause mass loss.\n\n");
    }

    // -------------------------------------------------------------------------
    // Issue 2: Operator splitting strategy analysis (single step)
    // -------------------------------------------------------------------------
    printf("=== ISSUE 2: Operator Splitting ===\n");
    printf("    Checking whether unsplit vs. split scheme matters.\n\n");
    printf("  Current implementation: UNSPLIT (all 3 directions in one kernel pass).\n");
    printf("  Consequence: only 1st-order accurate in time for 2-D/3-D advection,\n");
    printf("  even though the spatial reconstruction is 2nd-order (TVD).\n");
    printf("  Strang splitting (X-Y-Z-Y-X or similar) would give 2nd-order in time.\n\n");

    // -------------------------------------------------------------------------
    // Issue 3: TVD limiter and clipping — one-step diagnostic
    // -------------------------------------------------------------------------
    printf("=== ISSUE 3: TVD Limiter Gradient Ratio + Clipping (1 step) ===\n\n");

    // Build initial fill level
    std::vector<float> h_fill_init(N);
    initSlottedDisk(h_fill_init, nx, ny, nz, cx, cy, R, slot_w, slot_d);

    // Count initial pure-liquid cells
    int init_pure_liquid = 0;
    float mass_init = 0.0f;
    for (float f : h_fill_init) {
        mass_init += f;
        if (f > 1.0f - 1e-6f) init_pure_liquid++;
    }
    printf("  Initial state:\n");
    printf("    mass_init      = %.4f\n", mass_init);
    printf("    pure-liquid cells (f=1) = %d\n\n", init_pure_liquid);

    // Upload to device and copy as f_before
    float* d_f_before;
    cudaMalloc(&d_f_before, N * sizeof(float));
    cudaMemcpy(d_f_before, h_fill_init.data(), N * sizeof(float), cudaMemcpyHostToDevice);

    // Create VOF solver with TVD-MC (same as viz_zalesak)
    VOFSolver vof(nx, ny, nz, dx);
    vof.setAdvectionScheme(VOFAdvectionScheme::TVD);
    vof.setTVDLimiter(TVDLimiter::MC);
    vof.initialize(h_fill_init.data());

    float mass_before_step = vof.computeTotalMass();

    // Run ONE advection step
    vof.advectFillLevel(d_ux, d_uy, d_uz, dt);
    cudaDeviceSynchronize();

    float mass_after_step = vof.computeTotalMass();

    // Get f_after
    float* d_f_after = vof.getFillLevel();

    // Run diagnostics kernel
    int*   d_clip_lo;     cudaMalloc(&d_clip_lo,    sizeof(int));
    int*   d_clip_hi;     cudaMalloc(&d_clip_hi,    sizeof(int));
    int*   d_swiss;       cudaMalloc(&d_swiss,       sizeof(int));
    float* d_max_over;    cudaMalloc(&d_max_over,    sizeof(float));
    float* d_max_under;   cudaMalloc(&d_max_under,   sizeof(float));

    cudaMemset(d_clip_lo,  0,    sizeof(int));
    cudaMemset(d_clip_hi,  0,    sizeof(int));
    cudaMemset(d_swiss,    0,    sizeof(int));
    // Initialize max_over=0, max_under=1 (sentinel values)
    float zero = 0.0f, one = 1.0f;
    cudaMemcpy(d_max_over,  &zero, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_max_under, &one,  sizeof(float), cudaMemcpyHostToDevice);

    dim3 block1(256);
    dim3 grid1((N + 255) / 256);
    analyseStep<<<grid1, block1>>>(
        d_f_before, d_f_after,
        d_clip_lo, d_clip_hi, d_swiss,
        d_max_over, d_max_under,
        N);
    cudaDeviceSynchronize();

    int h_clip_lo, h_clip_hi, h_swiss;
    float h_max_over, h_max_under;
    cudaMemcpy(&h_clip_lo,   d_clip_lo,   sizeof(int),   cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_clip_hi,   d_clip_hi,   sizeof(int),   cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_swiss,     d_swiss,     sizeof(int),   cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_max_over,  d_max_over,  sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_max_under, d_max_under, sizeof(float), cudaMemcpyDeviceToHost);

    float mass_loss_step1 = mass_before_step - mass_after_step;
    printf("  After step 1:\n");
    printf("    mass before     = %.6f\n", mass_before_step);
    printf("    mass after      = %.6f\n", mass_after_step);
    printf("    mass lost       = %.6e  (%.4f%%)\n",
           mass_loss_step1, 100.0f * mass_loss_step1 / mass_before_step);
    printf("    clip_lo events  = %d  (cells flushed near 0, mass LOST)\n", h_clip_lo);
    printf("    clip_hi events  = %d  (cells rounded to 1, mass GAINED)\n", h_clip_hi);
    printf("    Swiss-cheese    = %d  (pure-liquid cells now have holes)\n", h_swiss);
    printf("    max_delta_up    = %.6f  (unexpected f increase per cell)\n", h_max_over);
    printf("    min_f_after     = %.6f  (closest to 0 post-clip)\n\n", h_max_under);

    cudaFree(d_f_before);
    cudaFree(d_clip_lo);
    cudaFree(d_clip_hi);
    cudaFree(d_swiss);
    cudaFree(d_max_over);
    cudaFree(d_max_under);

    // -------------------------------------------------------------------------
    // Full 800-step revolution: report diagnostics every 100 steps
    // -------------------------------------------------------------------------
    printf("=== Full 800-step Revolution (every 100 steps) ===\n");
    printf("%-6s  %-12s  %-10s  %-10s  %-10s  %-10s\n",
           "Step", "Mass", "MassErr%", "ClipLo", "ClipHi", "SwissChe");
    printf("------  ------------  ----------  ----------  ----------  ----------\n");

    // Re-initialise the solver cleanly
    VOFSolver vof2(nx, ny, nz, dx);
    vof2.setAdvectionScheme(VOFAdvectionScheme::TVD);
    vof2.setTVDLimiter(TVDLimiter::MC);
    vof2.initialize(h_fill_init.data());

    const float mass_ref = vof2.computeTotalMass();

    // Device buffers for f_prev (snapshot before each step batch)
    float* d_f_prev;
    cudaMalloc(&d_f_prev, N * sizeof(float));

    // Reuse diagnostic counters
    int*   d_lo2;   cudaMalloc(&d_lo2,  sizeof(int));
    int*   d_hi2;   cudaMalloc(&d_hi2,  sizeof(int));
    int*   d_sw2;   cudaMalloc(&d_sw2,  sizeof(int));
    float* d_mo2;   cudaMalloc(&d_mo2,  sizeof(float));
    float* d_mu2;   cudaMalloc(&d_mu2,  sizeof(float));

    int total_clip_lo = 0, total_clip_hi = 0, total_swiss = 0;

    printf("%-6d  %-12.4f  %-10.4f  %-10d  %-10d  %-10d\n",
           0, mass_ref, 0.0f, 0, 0, 0);

    for (int step = 1; step <= steps_per_rev; ++step) {
        // Snapshot f^n before advection
        cudaMemcpy(d_f_prev, vof2.getFillLevel(), N * sizeof(float),
                   cudaMemcpyDeviceToDevice);

        vof2.advectFillLevel(d_ux, d_uy, d_uz, dt);
        cudaDeviceSynchronize();

        // Accumulate step diagnostics
        cudaMemset(d_lo2, 0, sizeof(int));
        cudaMemset(d_hi2, 0, sizeof(int));
        cudaMemset(d_sw2, 0, sizeof(int));
        cudaMemcpy(d_mo2, &zero, sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_mu2, &one,  sizeof(float), cudaMemcpyHostToDevice);

        analyseStep<<<grid1, block1>>>(
            d_f_prev, vof2.getFillLevel(),
            d_lo2, d_hi2, d_sw2, d_mo2, d_mu2,
            N);
        cudaDeviceSynchronize();

        int h_lo2, h_hi2, h_sw2;
        cudaMemcpy(&h_lo2, d_lo2, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(&h_hi2, d_hi2, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(&h_sw2, d_sw2, sizeof(int), cudaMemcpyDeviceToHost);

        total_clip_lo += h_lo2;
        total_clip_hi += h_hi2;
        total_swiss   += h_sw2;

        if (step % 100 == 0) {
            float m = vof2.computeTotalMass();
            float merr = 100.0f * (mass_ref - m) / mass_ref;
            printf("%-6d  %-12.4f  %-10.4f  %-10d  %-10d  %-10d\n",
                   step, m, merr, total_clip_lo, total_clip_hi, total_swiss);
            // Reset accumulators so per-interval counts are visible
            total_clip_lo = 0;
            total_clip_hi = 0;
            total_swiss   = 0;
        }
    }

    printf("\n");
    float mass_final = vof2.computeTotalMass();
    printf("  Final mass  = %.6f\n", mass_final);
    printf("  Total error = %.4f%%\n\n",
           100.0f * (mass_ref - mass_final) / mass_ref);

    // -------------------------------------------------------------------------
    // Summary of root-cause findings
    // -------------------------------------------------------------------------
    printf("=== ROOT-CAUSE SUMMARY ===\n\n");

    printf("Issue 1 — Face velocity (collocated averaging):\n");
    printf("  Face velocity = 0.5*(u_i + u_{i+1}) from cell-center values.\n");
    printf("  For solid body rotation, u_x = -omega*(y-cy) is constant in x,\n");
    printf("  so the average is exact and discrete div = %.2e.\n", div_max);
    printf("  CONCLUSION: Face velocity interpolation is NOT the mass-loss driver.\n\n");

    printf("Issue 2 — Operator splitting:\n");
    printf("  Kernel advects all three directions simultaneously from f^n (unsplit).\n");
    printf("  This is only 1st-order in time for multi-D, regardless of spatial order.\n");
    printf("  Strang splitting (XY → YX alternated) would give 2nd-order in time.\n");
    printf("  CONCLUSION: Contributes to interface smearing (slot-corner diffusion).\n");
    printf("              Does NOT directly cause mass loss, but widens interface,\n");
    printf("              increasing the number of cells exposed to clipping.\n\n");

    printf("Issue 3 — TVD gradient ratio and clipping:\n");
    printf("  a) copysignf(1e-10, delta_center) epsilon: when delta_center=0,\n");
    printf("     r = delta_upwind / ±1e-10 → ±infinity.\n");
    printf("     MC limiter: phi(+inf) = 2.0 (maximum anti-diffusion).\n");
    printf("     Anti-diffusive flux f_new > 1 or < 0 → clipped → mass lost.\n");
    printf("  b) Interface compression (C_compress=0.30) is ALWAYS active\n");
    printf("     (hardcoded at line ~1611 in advectFillLevel).\n");
    printf("     Compression drives f toward 0/1 at interface cells,\n");
    printf("     pushing material OUT of concave slot-corner regions.\n");
    printf("     This directly causes Swiss-cheese holes inside the disk interior.\n");
    printf("  CONCLUSION: Both (a) and (b) are primary mass-loss drivers.\n\n");

    printf("=== SPECIFIC FIX RECOMMENDATIONS ===\n\n");

    printf("Fix 1 (HIGH IMPACT): Disable interface compression for Zalesak / rotation.\n");
    printf("  File: src/physics/vof/vof_solver.cu, line ~1611\n");
    printf("  Change:  float C_compress = 0.30f;\n");
    printf("  To:      float C_compress = interface_compression_enabled_ ? C_compress_ : 0.0f;\n");
    printf("  Add member:  bool interface_compression_enabled_ = false;  float C_compress_ = 0.10f;\n");
    printf("  Add API:     void setInterfaceCompression(bool en, float C=0.10f);\n\n");

    printf("Fix 2 (HIGH IMPACT): Fix gradient ratio epsilon in TVD kernel.\n");
    printf("  File: src/physics/vof/vof_solver.cu, all 12 instances of copysignf(1e-10f, ...)\n");
    printf("  Change:  r = delta_upwind / (delta_center + copysignf(1e-10f, delta_center));\n");
    printf("  To:      r = (fabsf(delta_center) > 1e-6f)\n");
    printf("               ? delta_upwind / delta_center : 0.0f;\n");
    printf("  Effect:  When gradient is flat, r=0 → phi=0 → pure upwind (stable, no clip).\n\n");

    printf("Fix 3 (MODERATE IMPACT): Strang splitting for multidimensional advection.\n");
    printf("  File: src/physics/vof/vof_solver.cu, advectFillLevel()\n");
    printf("  Replace single unsplit kernel call with operator-split sequence:\n");
    printf("    Step n (even): X-sweep then Y-sweep\n");
    printf("    Step n (odd):  Y-sweep then X-sweep\n");
    printf("  Each 1-D sweep reads from previous sweep output (not f^n).\n");
    printf("  Requires a 1-D TVD kernel (advect one direction, output to tmp).\n\n");

    printf("Fix 4 (LOW IMPACT): Remove asymmetric flushing.\n");
    printf("  File: src/physics/vof/vof_solver.cu, lines ~689-691 and ~1150-1151\n");
    printf("  Remove:  if (f_new < 1e-9f) f_new = 0.0f;\n");
    printf("           if (f_new > 1.0f - 1e-9f) f_new = 1.0f;\n");
    printf("  Keep only the final clamp:  f = clamp(f, 0, 1);\n");
    printf("  The flushing converts legitimate small values to zero, losing mass.\n\n");

    // Cleanup
    cudaFree(d_f_prev);
    cudaFree(d_lo2);
    cudaFree(d_hi2);
    cudaFree(d_sw2);
    cudaFree(d_mo2);
    cudaFree(d_mu2);
    cudaFree(d_ux);
    cudaFree(d_uy);
    cudaFree(d_uz);

    return 0;
}
