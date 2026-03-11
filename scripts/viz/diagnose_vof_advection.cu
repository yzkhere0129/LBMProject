/**
 * @file diagnose_vof_advection.cu
 * @brief Diagnostic program for Zalesak disk VOF advection.
 *
 * Investigates 4.1% mass loss and Swiss-cheese internal holes by measuring:
 *   1. Velocity field divergence (should be zero for rigid rotation)
 *   2. Per-step mass loss rate (constant vs. accelerating?)
 *   3. Swiss-cheese creation rate (cells that drop from f>=0.99 to f<0.99)
 *   4. Boundedness violations (f < 0 or f > 1)
 *   5. Effect of the always-on interface compression (C=0.30)
 *
 * Compile from LBMProject/build/:
 *   nvcc -o ../scripts/viz/diagnose_vof_advection \
 *        ../scripts/viz/diagnose_vof_advection.cu \
 *        -I../include -L. -llbm_physics -llbm_core \
 *        --std=c++17 -rdc=true -Xcompiler -fPIC -O2 -arch=sm_86
 *
 * Run:
 *   cd /home/yzk/LBMProject/scripts/viz && ./diagnose_vof_advection
 */

#include "physics/vof_solver.h"

#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>
#include <algorithm>
#include <numeric>

using namespace lbm::physics;

// ============================================================================
// Domain / geometry constants (128×128×3, matching viz_zalesak.cu)
// ============================================================================
static const int   NX = 128, NY = 128, NZ = 3;
static const float DX = 1.0f;
static const float CX = NX * 0.5f;   // 64
static const float CY = NY * 0.5f;   // 64
static const float R       = 30.0f;  // disk radius [cells]
static const float SLOT_W  = 6.0f;   // slot width  [cells]
static const float SLOT_D  = 50.0f;  // slot depth  [cells]

// One full revolution in STEPS_PER_REV steps
static const int   STEPS_PER_REV = 800;
static const float OMEGA = 2.0f * static_cast<float>(M_PI) / static_cast<float>(STEPS_PER_REV);
static const float DT    = 1.0f;

// Report interval
static const int REPORT_INTERVAL = 100;

// ============================================================================
// Helper: initialize slotted disk (hard-edged, no tanh smoothing)
// Using a sharp initialisation exposes the true diffusion of the advection.
// ============================================================================
static void initSlottedDisk(VOFSolver& vof)
{
    int N = NX * NY * NZ;
    std::vector<float> h(N, 0.0f);
    for (int k = 0; k < NZ; ++k)
        for (int j = 0; j < NY; ++j)
            for (int i = 0; i < NX; ++i) {
                float dx = i - CX, dy = j - CY;
                float r  = std::sqrt(dx*dx + dy*dy);
                bool in_disk = (r <= R);
                bool in_slot = (std::fabs(dx) <= SLOT_W * 0.5f)
                             && (dy >= 0.0f) && (dy <= SLOT_D);
                int idx = i + NX*(j + NY*k);
                h[idx] = (in_disk && !in_slot) ? 1.0f : 0.0f;
            }
    vof.initialize(h.data());
}

// ============================================================================
// Helper: upload rigid rotation velocity field
// u = -ω(j - cy),  v = ω(i - cx),  w = 0
// ============================================================================
static void uploadRotVel(float* d_ux, float* d_uy, float* d_uz)
{
    int N = NX * NY * NZ;
    std::vector<float> hx(N), hy(N), hz(N, 0.0f);
    for (int k = 0; k < NZ; ++k)
        for (int j = 0; j < NY; ++j)
            for (int i = 0; i < NX; ++i) {
                int idx = i + NX*(j + NY*k);
                hx[idx] = -OMEGA * (j - CY);
                hy[idx] =  OMEGA * (i - CX);
            }
    cudaMemcpy(d_ux, hx.data(), N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_uy, hy.data(), N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_uz, hz.data(), N*sizeof(float), cudaMemcpyHostToDevice);
}

// ============================================================================
// Diagnostic 1: velocity divergence
// div_ij = (u[i+1,j] - u[i-1,j]) / (2*dx) + (v[i,j+1] - v[i,j-1]) / (2*dy)
// Uses central differences on the interior; skips boundaries.
// ============================================================================
struct DivStats {
    float max_abs_div;
    float mean_abs_div;
    int   imax, jmax, kmax;
};

static DivStats computeDivergence(const std::vector<float>& hx,
                                  const std::vector<float>& hy)
{
    DivStats s{0.0f, 0.0f, 0, 0, 0};
    double sum = 0.0;
    int count  = 0;
    for (int k = 0; k < NZ; ++k) {
        for (int j = 1; j < NY-1; ++j) {
            for (int i = 1; i < NX-1; ++i) {
                int xp = (i+1) + NX*(j   + NY*k);
                int xm = (i-1) + NX*(j   + NY*k);
                int yp = i     + NX*((j+1) + NY*k);
                int ym = i     + NX*((j-1) + NY*k);

                float div = (hx[xp] - hx[xm]) / (2.0f * DX)
                          + (hy[yp] - hy[ym]) / (2.0f * DX);
                float ad  = std::fabs(div);
                sum += ad;
                ++count;
                if (ad > s.max_abs_div) {
                    s.max_abs_div = ad;
                    s.imax = i; s.jmax = j; s.kmax = k;
                }
            }
        }
    }
    s.mean_abs_div = (count > 0) ? static_cast<float>(sum / count) : 0.0f;
    return s;
}

// ============================================================================
// Diagnostic 2: per-step fill level statistics
// ============================================================================
struct StepStats {
    float mass;
    float mass_err_pct;   // (mass - mass0) / mass0 * 100
    int   swiss_cheese;   // cells: f_before >= 0.99 && f_after < 0.99
    int   neg_violations; // cells with f < 0
    int   over_violations;// cells with f > 1
    float max_vel;
    float cfl_max;
};

static StepStats computeStepStats(const std::vector<float>& before,
                                   const std::vector<float>& after,
                                   const std::vector<float>& hx,
                                   const std::vector<float>& hy,
                                   float mass0)
{
    StepStats s{};
    int N = NX * NY * NZ;
    double mass = 0.0;
    for (int idx = 0; idx < N; ++idx) {
        mass += after[idx];
        if (before[idx] >= 0.99f && after[idx] < 0.99f) ++s.swiss_cheese;
        if (after[idx] < -1e-7f) ++s.neg_violations;
        if (after[idx] >  1.0f + 1e-7f) ++s.over_violations;
    }
    s.mass = static_cast<float>(mass);
    s.mass_err_pct = (mass0 > 0.0f) ? (s.mass - mass0) / mass0 * 100.0f : 0.0f;

    float vmax2 = 0.0f;
    for (int idx = 0; idx < N; ++idx) {
        float v2 = hx[idx]*hx[idx] + hy[idx]*hy[idx];
        if (v2 > vmax2) vmax2 = v2;
    }
    s.max_vel  = std::sqrt(vmax2);
    s.cfl_max  = s.max_vel * DT / DX;
    return s;
}

// ============================================================================
// Helper: dump middle Z slice to CSV
// ============================================================================
static void dumpCSV(const std::string& path, const std::vector<float>& fill)
{
    int k0 = NZ / 2;
    std::ofstream f(path);
    for (int j = 0; j < NY; ++j) {
        for (int i = 0; i < NX; ++i) {
            int idx = i + NX*(j + NY*k0);
            f << fill[idx];
            if (i < NX-1) f << ',';
        }
        f << '\n';
    }
}

// ============================================================================
// main
// ============================================================================
int main()
{
    const int N = NX * NY * NZ;
    std::printf("=== Zalesak VOF Advection Diagnostic ===\n");
    std::printf("Grid   : %d x %d x %d   (NZ=%d slab)\n", NX, NY, NZ, NZ);
    std::printf("Disk   : center=(%.0f,%.0f)  R=%.0f\n", CX, CY, R);
    std::printf("Slot   : width=%.0f  depth=%.0f\n", SLOT_W, SLOT_D);
    std::printf("Rotation: omega=%.6f rad/step  T=%d steps\n", OMEGA, STEPS_PER_REV);
    std::printf("CFL_max : %.4f  (omega*R*dt/dx)\n\n", OMEGA * R * DT / DX);

    // -------------------------------------------------------------------------
    // 1. Divergence check BEFORE advection
    // -------------------------------------------------------------------------
    {
        std::vector<float> hx(N), hy(N), hz(N, 0.0f);
        for (int k = 0; k < NZ; ++k)
            for (int j = 0; j < NY; ++j)
                for (int i = 0; i < NX; ++i) {
                    int idx = i + NX*(j + NY*k);
                    hx[idx] = -OMEGA * (j - CY);
                    hy[idx] =  OMEGA * (i - CX);
                }

        DivStats ds = computeDivergence(hx, hy);
        std::printf("--- Divergence of velocity field (analytical) ---\n");
        std::printf("  max|div|  = %.6e  at (%d,%d,%d)\n",
                    ds.max_abs_div, ds.imax, ds.jmax, ds.kmax);
        std::printf("  mean|div| = %.6e\n\n", ds.mean_abs_div);

        // Analytical: solid-body rotation is divergence-free.
        // Discrete central-difference should give ~machine epsilon * omega.
        // Large values would indicate cell-center vs face-center mismatch.
    }

    // -------------------------------------------------------------------------
    // 2. Set up VOF solver (TVD-MC, matching viz_zalesak.cu)
    // -------------------------------------------------------------------------
    VOFSolver vof(NX, NY, NZ, DX);   // default: PERIODIC BCs
    vof.setAdvectionScheme(VOFAdvectionScheme::TVD);
    vof.setTVDLimiter(TVDLimiter::MC);
    vof.setMassConservationCorrection(false);  // raw advection, no artificial fix
    initSlottedDisk(vof);

    float mass0 = vof.computeTotalMass();
    std::printf("Initial mass M0 = %.4f\n\n", mass0);

    // -------------------------------------------------------------------------
    // 3. Allocate velocity arrays on device
    // -------------------------------------------------------------------------
    float *d_ux, *d_uy, *d_uz;
    cudaMalloc(&d_ux, N * sizeof(float));
    cudaMalloc(&d_uy, N * sizeof(float));
    cudaMalloc(&d_uz, N * sizeof(float));
    uploadRotVel(d_ux, d_uy, d_uz);

    // Host copies of velocity (static – doesn't change)
    std::vector<float> h_ux(N), h_uy(N), h_uz(N, 0.0f);
    cudaMemcpy(h_ux.data(), d_ux, N*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_uy.data(), d_uy, N*sizeof(float), cudaMemcpyDeviceToHost);

    // -------------------------------------------------------------------------
    // 4. Time loop: 800 steps, report every REPORT_INTERVAL steps
    // -------------------------------------------------------------------------
    std::printf("%-6s  %-12s  %-12s  %-14s  %-10s  %-10s  %-8s  %-8s\n",
                "Step", "Mass", "MassErr(%)", "Swiss-cheese", "Neg viol", "Over viol",
                "MaxVel", "CFL");
    std::printf("%s\n", std::string(90, '-').c_str());

    std::vector<float> h_before(N), h_after(N);
    std::vector<float> cumulative_swiss(STEPS_PER_REV, 0.0f);

    // CSV output for per-step mass tracking
    std::ofstream mass_csv("/home/yzk/LBMProject/scripts/viz/diag_mass_history.csv");
    mass_csv << "step,mass,mass_err_pct,swiss_cheese,neg_viol,over_viol\n";

    // Track total Swiss-cheese cells created across all steps
    int total_swiss_created = 0;
    int total_neg_viol = 0;
    int total_over_viol = 0;

    for (int step = 1; step <= STEPS_PER_REV; ++step) {
        // Capture state BEFORE advection
        vof.copyFillLevelToHost(h_before.data());

        // Advect
        vof.advectFillLevel(d_ux, d_uy, d_uz, DT);
        cudaDeviceSynchronize();

        // Capture state AFTER advection
        vof.copyFillLevelToHost(h_after.data());

        // Per-step stats
        StepStats ss = computeStepStats(h_before, h_after, h_ux, h_uy, mass0);
        total_swiss_created += ss.swiss_cheese;
        total_neg_viol      += ss.neg_violations;
        total_over_viol     += ss.over_violations;

        mass_csv << step << "," << ss.mass << "," << ss.mass_err_pct << ","
                 << ss.swiss_cheese << "," << ss.neg_violations << ","
                 << ss.over_violations << "\n";

        if (step % REPORT_INTERVAL == 0 || step == 1) {
            float angle = step * OMEGA * 180.0f / static_cast<float>(M_PI);
            std::printf("%-6d  %-12.4f  %-12.4f  %-14d  %-10d  %-10d  %-8.4f  %-8.4f"
                        "  (%.0f deg)\n",
                        step, ss.mass, ss.mass_err_pct,
                        ss.swiss_cheese, ss.neg_violations, ss.over_violations,
                        ss.max_vel, ss.cfl_max, angle);
        }
    }

    mass_csv.close();

    // -------------------------------------------------------------------------
    // 5. Final snapshot
    // -------------------------------------------------------------------------
    vof.copyFillLevelToHost(h_after.data());
    dumpCSV("/home/yzk/LBMProject/scripts/viz/diag_final_fill.csv", h_after);

    // -------------------------------------------------------------------------
    // 6. Interior cell analysis (Swiss-cheese deep investigation)
    // Count cells that were solid interior (disk, away from edge) initially
    // but ended up < 0.99 after full rotation.
    // -------------------------------------------------------------------------
    {
        // Re-initialize to get the initial state
        VOFSolver vof2(NX, NY, NZ, DX);
        vof2.setAdvectionScheme(VOFAdvectionScheme::TVD);
        vof2.setTVDLimiter(TVDLimiter::MC);
        initSlottedDisk(vof2);

        std::vector<float> h_init(N);
        vof2.copyFillLevelToHost(h_init.data());

        // Interior cells: in_disk, not_slot, r < R - 3 (away from edge)
        int interior_count = 0;
        int interior_swiss = 0;
        for (int k = 0; k < NZ; ++k)
            for (int j = 0; j < NY; ++j)
                for (int i = 0; i < NX; ++i) {
                    float dx = i - CX, dy = j - CY;
                    float r  = std::sqrt(dx*dx + dy*dy);
                    bool in_disk = (r <= R - 3.0f);  // interior margin
                    bool in_slot = (std::fabs(dx) <= SLOT_W * 0.5f + 1.0f)
                                 && (dy >= -1.0f) && (dy <= SLOT_D + 1.0f);
                    if (in_disk && !in_slot) {
                        ++interior_count;
                        int idx = i + NX*(j + NY*k);
                        if (h_after[idx] < 0.99f) ++interior_swiss;
                    }
                }

        std::printf("\n--- Swiss-Cheese Interior Analysis ---\n");
        std::printf("  Interior cells (r < R-3, not slot): %d\n", interior_count);
        std::printf("  Interior cells with f < 0.99 after full rotation: %d (%.1f%%)\n",
                    interior_swiss,
                    (interior_count > 0) ? 100.0f * interior_swiss / interior_count : 0.0f);
    }

    // -------------------------------------------------------------------------
    // 7. Summary report
    // -------------------------------------------------------------------------
    float mass_final = vof.computeTotalMass();
    float mass_err_final = (mass_final - mass0) / mass0 * 100.0f;

    std::printf("\n=== DIAGNOSTIC SUMMARY ===\n");
    std::printf("  Initial mass M0             : %.6f\n", mass0);
    std::printf("  Final mass                  : %.6f\n", mass_final);
    std::printf("  Mass error at 360 deg       : %.4f%%\n", mass_err_final);
    std::printf("  Total Swiss-cheese events   : %d  (%.3f per step avg)\n",
                total_swiss_created,
                total_swiss_created / static_cast<float>(STEPS_PER_REV));
    std::printf("  Total boundedness neg viol  : %d\n", total_neg_viol);
    std::printf("  Total boundedness over viol : %d\n", total_over_viol);
    std::printf("\n  CSV outputs:\n");
    std::printf("    /home/yzk/LBMProject/scripts/viz/diag_mass_history.csv\n");
    std::printf("    /home/yzk/LBMProject/scripts/viz/diag_final_fill.csv\n");

    // -------------------------------------------------------------------------
    // 8. Compression diagnostic: run WITHOUT compression to isolate its effect
    // -------------------------------------------------------------------------
    // The VOF solver always applies C_compress=0.30 interface compression
    // inside advectFillLevel (hardcoded). To isolate compression's contribution
    // we use a second run with UPWIND (no compression makes TVD the variable).
    // We compare mass at 360 deg: TVD+compression vs UPWIND+compression.
    {
        std::printf("\n--- Scheme Comparison (both runs include compression C=0.30) ---\n");

        VOFSolver vof_up(NX, NY, NZ, DX);
        vof_up.setAdvectionScheme(VOFAdvectionScheme::UPWIND);
        vof_up.setMassConservationCorrection(false);
        initSlottedDisk(vof_up);

        float* d_ux2; float* d_uy2; float* d_uz2;
        cudaMalloc(&d_ux2, N*sizeof(float));
        cudaMalloc(&d_uy2, N*sizeof(float));
        cudaMalloc(&d_uz2, N*sizeof(float));
        uploadRotVel(d_ux2, d_uy2, d_uz2);

        for (int step = 0; step < STEPS_PER_REV; ++step) {
            vof_up.advectFillLevel(d_ux2, d_uy2, d_uz2, DT);
            cudaDeviceSynchronize();
        }
        float mass_upwind = vof_up.computeTotalMass();
        std::printf("  UPWIND+compression (800 steps): mass=%.4f  err=%.4f%%\n",
                    mass_upwind, (mass_upwind - mass0) / mass0 * 100.0f);
        std::printf("  TVD-MC+compression (800 steps): mass=%.4f  err=%.4f%%\n",
                    mass_final, mass_err_final);

        cudaFree(d_ux2);
        cudaFree(d_uy2);
        cudaFree(d_uz2);
    }

    // -------------------------------------------------------------------------
    // 9. Divergence contribution from discrete velocity
    //    For a periodic 128x128 grid: div(u) at cell (i,j) should be exactly 0
    //    for solid-body rotation.  Measure machine-precision residual.
    // -------------------------------------------------------------------------
    {
        DivStats ds = computeDivergence(h_ux, h_uy);
        std::printf("\n--- Discrete Velocity Divergence (after upload, from device copy) ---\n");
        std::printf("  max|div|  = %.3e  at (%d,%d,%d)\n",
                    ds.max_abs_div, ds.imax, ds.jmax, ds.kmax);
        std::printf("  mean|div| = %.3e\n", ds.mean_abs_div);

        // Expected: max|div| = omega (one omega from the linear terms cancelling
        // on a uniform grid). For omega=0.00785, expect ~1e-2.
        // If max|div| >> omega, velocity encoding is incorrect.
        std::printf("  omega=%f, so |u_x+1 - u_x-1|/(2dx) ~ omega at the edge cells\n",
                    OMEGA);
    }

    cudaFree(d_ux);
    cudaFree(d_uy);
    cudaFree(d_uz);

    std::printf("\nDone.\n");
    return 0;
}
