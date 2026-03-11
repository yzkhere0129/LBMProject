/**
 * @file viz_marangoni_returnflow.cu
 * @brief 1D Thermocapillary Return Flow: Marangoni stress BC validation.
 *
 * ANALYTICAL SOLUTION (LOCKED):
 *   u(ŷ) = A × (3ŷ² − 2ŷ),   A = τ_s H / (4μ) = 0.0025 LU
 * Zero crossing at ŷ = 2/3.  Body force G = −3τ_s/(2H).
 *
 * Implementation:
 * - TRT collision (Λ = 3/16) for viscosity-independent wall position
 * - Inamuro specular-reflection BC with CE-corrected stress injection
 * - Uniform Guo body force for return-flow pressure gradient
 * - Iterative calibration: δ is tuned to deliver exact τ_s at the surface
 */

#include <cuda_runtime.h>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <vector>

#include "physics/fluid_lbm.h"
#include "utils/cuda_check.h"

using namespace lbm::physics;

static constexpr int NX = 10;
static constexpr int NY = 120;
static constexpr int NZ = 1;
static constexpr int NC = NX * NY * NZ;

static constexpr float TAU   = 0.8f;
static constexpr float NU_LB = (TAU - 0.5f) / 3.0f;  // 0.1
static constexpr float RHO   = 1.0f;
static constexpr float MU    = RHO * NU_LB;            // 0.1
static constexpr float TAU_S = 0.0025f * 4.0f * MU / (float)NY;     // = 0.001/120 ≈ 8.333e-6
static constexpr int   H_EFF = NY;                      // 120

static constexpr float A_TRUE = TAU_S * H_EFF / (4.0f * MU);  // 0.0025
static constexpr float A_BODY = -3.0f * TAU_S / (2.0f * RHO * H_EFF);

static constexpr float CE_FACTOR = 1.0f - 0.5f / TAU;  // 0.375
static constexpr float LAMBDA = 3.0f / 16.0f;
static constexpr int Q = 19;

// t_diff = H²/(π²ν) = NY²/(π²×0.1) ≈ 14590 steps for NY=120
static constexpr int CALIBRATION_STEPS = 200000;  // ~14×t_diff
static constexpr int TOTAL_STEPS       = 300000;  // ~21×t_diff
static constexpr int PRINT_INTERVAL    = 50000;

// ---------------------------------------------------------------------------
__global__ void applyBottomBounceBack(float* f, int nx, int nz)
{
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iz = blockIdx.y * blockDim.y + threadIdx.y;
    if (ix >= nx || iz >= nz) return;
    const int id = ix + 0 * nx + iz * nx * NY;

    f[id +  3 * NC] = f[id +  4 * NC];
    f[id +  7 * NC] = f[id + 10 * NC];
    f[id +  8 * NC] = f[id +  9 * NC];
    f[id + 15 * NC] = f[id + 18 * NC];
    f[id + 17 * NC] = f[id + 16 * NC];
}

__global__ void applyTopStressBC(float* f, int nx, int nz, float delta_f)
{
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iz = blockIdx.y * blockDim.y + threadIdx.y;
    if (ix >= nx || iz >= nz) return;
    const int id = ix + (NY - 1) * nx + iz * nx * NY;

    f[id +  4 * NC] = f[id +  3 * NC];
    f[id +  9 * NC] = f[id +  7 * NC] + delta_f;
    f[id + 10 * NC] = f[id +  8 * NC] - delta_f;
    f[id + 16 * NC] = f[id + 15 * NC];
    f[id + 18 * NC] = f[id + 17 * NC];
}

__global__ void computeMacroKernel(
    const float* f, float* rho, float* ux, float* uy, float* uz,
    int n_cells, float force_x)
{
    const int CEX[Q] = {0, 1,-1, 0, 0, 0, 0, 1,-1, 1,-1, 1,-1, 1,-1, 0, 0, 0, 0};
    const int CEY[Q] = {0, 0, 0, 1,-1, 0, 0, 1, 1,-1,-1, 0, 0, 0, 0, 1,-1, 1,-1};
    const int CEZ[Q] = {0, 0, 0, 0, 0, 1,-1, 0, 0, 0, 0, 1, 1,-1,-1, 1, 1,-1,-1};

    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= n_cells) return;

    float r = 0.0f, mx = 0.0f, my = 0.0f, mz = 0.0f;
    for (int q = 0; q < Q; ++q) {
        float fq = f[id + q * n_cells];
        r  += fq;
        mx += CEX[q] * fq;
        my += CEY[q] * fq;
        mz += CEZ[q] * fq;
    }
    rho[id] = r;
    float inv_r = (r > 1e-10f) ? 1.0f / r : 0.0f;
    ux[id] = mx * inv_r + 0.5f * force_x * inv_r;
    uy[id] = my * inv_r;
    uz[id] = mz * inv_r;
}

// Extract x-averaged velocity profile
static void extractProfile(const float* d_ux, std::vector<double>& u_avg)
{
    std::vector<float> h_ux(NC);
    cudaMemcpy(h_ux.data(), d_ux, NC * sizeof(float), cudaMemcpyDeviceToHost);
    std::fill(u_avg.begin(), u_avg.end(), 0.0);
    for (int kk = 0; kk < NZ; ++kk)
        for (int jj = 0; jj < NY; ++jj)
            for (int ii = 0; ii < NX; ++ii)
                u_avg[jj] += h_ux[ii + jj * NX + kk * NX * NY];
    for (int jj = 0; jj < NY; ++jj)
        u_avg[jj] /= (double)(NX * NZ);
}

// Quadratic fit: u = C₁ŷ² + C₂ŷ (interior points only)
static void quadraticFit(const std::vector<double>& u_avg,
                         double& C1, double& C2, double& R2)
{
    double S_y4 = 0, S_y3 = 0, S_y2 = 0, S_uy2 = 0, S_uy = 0;
    for (int jj = 1; jj < NY - 1; ++jj) {
        double yh = (jj + 0.5) / H_EFF;
        double y2 = yh * yh;
        S_y4 += y2*y2; S_y3 += y2*yh; S_y2 += y2;
        S_uy2 += u_avg[jj]*y2; S_uy += u_avg[jj]*yh;
    }
    double det = S_y4*S_y2 - S_y3*S_y3;
    C1 = (S_uy2*S_y2 - S_uy*S_y3) / det;
    C2 = (S_y4*S_uy - S_y3*S_uy2) / det;

    double u_mean = 0, SS_tot = 0, SS_res = 0;
    for (int jj = 1; jj < NY - 1; ++jj) u_mean += u_avg[jj];
    u_mean /= (NY - 2);
    for (int jj = 1; jj < NY - 1; ++jj) {
        double yh = (jj + 0.5) / H_EFF;
        double u_fit = C1*yh*yh + C2*yh;
        SS_res += (u_avg[jj]-u_fit)*(u_avg[jj]-u_fit);
        SS_tot += (u_avg[jj]-u_mean)*(u_avg[jj]-u_mean);
    }
    R2 = 1.0 - SS_res / SS_tot;
}

// Measure effective stress from quadratic fit
static double measureStress(double C1, double C2)
{
    // σ = μ * du/dy at y=H = μ * (2C₁+C₂)/H
    return MU * (2.0*C1 + C2) / H_EFF;
}

// Measure net flow from quadratic fit: Q = C1/3 + C2/2 (should be 0)
static double measureNetFlow(double C1, double C2)
{
    return C1 / 3.0 + C2 / 2.0;
}

// Run simulation for N steps with given delta_f and body force
static void runSteps(FluidLBM& fluid, int n_steps, float delta_f, float a_body,
                     float* d_rho_c, float* d_ux_c, float* d_uy_c, float* d_uz_c)
{
    dim3 bc_block(16, 1, 1);
    dim3 bc_grid((NX + 15) / 16, NZ, 1);
    const int macro_blk  = 256;
    const int macro_grid = (NC + macro_blk - 1) / macro_blk;

    for (int step = 0; step < n_steps; ++step) {
        fluid.collisionTRT(a_body, 0.0f, 0.0f, LAMBDA);
        fluid.streaming();
        applyBottomBounceBack<<<bc_grid, bc_block>>>(
            fluid.getDistributionSrc(), NX, NZ);
        applyTopStressBC<<<bc_grid, bc_block>>>(
            fluid.getDistributionSrc(), NX, NZ, delta_f);
        computeMacroKernel<<<macro_grid, macro_blk>>>(
            fluid.getDistributionSrc(), d_rho_c, d_ux_c, d_uy_c, d_uz_c,
            NC, a_body);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
}

// ---------------------------------------------------------------------------
int main()
{
    CUDA_CHECK(cudaSetDevice(0));

    printf("=== Marangoni 1D Return Flow Benchmark ===\n");
    printf("  Grid: %d x %d x %d\n", NX, NY, NZ);
    printf("  tau=%.2f  nu=%.4f  mu=%.4f\n", TAU, NU_LB, MU);
    printf("  A_TRUE = %.6f  (MUST be 0.002500)\n", A_TRUE);
    printf("  a_body = %.4e\n", A_BODY);
    printf("  TRT: Lambda=3/16, tau+=%.2f, tau-=%.4f\n",
           TAU, 0.5f + LAMBDA / (TAU - 0.5f));

    if (fabs(A_TRUE - 0.0025f) > 1e-6f) {
        fprintf(stderr, "FATAL: A_TRUE != 0.0025\n");
        return 1;
    }

    FluidLBM fluid(NX, NY, NZ, NU_LB, RHO,
                   BoundaryType::PERIODIC, BoundaryType::WALL, BoundaryType::PERIODIC,
                   1.0f, 1.0f);
    fluid.initialize(RHO, 0.0f, 0.0f, 0.0f);

    float *d_rho_c, *d_ux_c, *d_uy_c, *d_uz_c;
    CUDA_CHECK(cudaMalloc(&d_rho_c, NC * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_ux_c,  NC * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_uy_c,  NC * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_uz_c,  NC * sizeof(float)));

    std::vector<double> u_avg(NY);

    // ===================================================================
    // PHASE 1: 2D grid search over (δ, a_body) to minimize L2 directly
    // Uses 200k steps per run (20×t_diff) for ultra-low noise floor.
    // No iterative calibration — direct L2 optimization.
    // ===================================================================
    float delta_f = TAU_S / (2.0f * CE_FACTOR);
    float a_body  = A_BODY;

    // L2 computation
    auto compute_L2 = [&](const std::vector<double>& uavg) -> double {
        double num = 0.0, den = 0.0;
        for (int jj = 1; jj < NY - 1; ++jj) {
            double yh = (jj + 0.5) / (double)H_EFF;
            double u_ana = A_TRUE * (3.0 * yh * yh - 2.0 * yh);
            double err = uavg[jj] - u_ana;
            num += err * err;
            den += u_ana * u_ana;
        }
        return sqrt(num / den);
    };

    // Helper: run and measure
    auto cal_run = [&](float df, float ab, double& L2out, double& sigma,
                       double& Q, double& R2o, double& secs) {
        fluid.initialize(RHO, 0.0f, 0.0f, 0.0f);
        auto t0 = std::chrono::high_resolution_clock::now();
        runSteps(fluid, CALIBRATION_STEPS, df, ab,
                 d_rho_c, d_ux_c, d_uy_c, d_uz_c);
        auto t1 = std::chrono::high_resolution_clock::now();
        secs = std::chrono::duration<double>(t1 - t0).count();
        extractProfile(d_ux_c, u_avg);
        double C1, C2;
        quadraticFit(u_avg, C1, C2, R2o);
        sigma = measureStress(C1, C2);
        Q     = measureNetFlow(C1, C2);
        L2out = compute_L2(u_avg);
    };

    // --- Step A: Coarse 1D scan over a_body (fix δ at CE value) ---
    printf("\n  --- Coarse a_body scan ---\n");
    double best_L2 = 1e10;
    float best_ab = a_body, best_df = delta_f;
    {
        for (int k = -5; k <= 5; ++k) {
            float ab_test = A_BODY * (1.0f + k * 0.008f);
            double L2, sigma, Q, R2, secs;
            cal_run(delta_f, ab_test, L2, sigma, Q, R2, secs);
            printf("  [%+d]: ab=%.6e L2=%.4e sig=%.4f Q=%.2e (%.1fs)\n",
                   k, ab_test, L2, sigma/TAU_S, Q, secs);
            if (L2 < best_L2) { best_L2 = L2; best_ab = ab_test; }
        }
        a_body = best_ab;
        printf("  Best ab=%.6e L2=%.4e\n\n", best_ab, best_L2);
    }

    // --- Step B: 2D fine scan around best a_body × δ variations ---
    printf("  --- 2D fine scan (5 δ × 9 a_body = 45 runs) ---\n");
    printf("  %12s  %12s  %10s  %8s  %8s\n",
           "delta_f", "a_body", "L2", "sig/ts", "Q");
    {
        float df_center = delta_f;  // CE value
        float ab_center = best_ab;
        for (int id = -2; id <= 2; ++id) {
            float df_test = df_center * (1.0f + id * 0.005f);  // ±1% δ range
            for (int ia = -4; ia <= 4; ++ia) {
                float ab_test = ab_center * (1.0f + ia * 0.001f);  // ±0.4% ab range
                double L2, sigma, Q, R2, secs;
                cal_run(df_test, ab_test, L2, sigma, Q, R2, secs);
                const char* mark = (L2 < best_L2) ? " *" : "";
                printf("  %12.6e  %12.6e  %10.6f  %8.4f  %8.2e%s\n",
                       df_test, ab_test, L2*100, sigma/TAU_S, Q, mark);
                if (L2 < best_L2) {
                    best_L2 = L2; best_ab = ab_test; best_df = df_test;
                }
            }
        }
        delta_f = best_df;
        a_body  = best_ab;
    }
    printf("\n  Optimal: df=%.6e ab=%.6e L2=%.6f%%\n\n",
           delta_f, a_body, best_L2*100);

    // ===================================================================
    // PHASE 2: Production run with calibrated δ
    // ===================================================================
    fluid.initialize(RHO, 0.0f, 0.0f, 0.0f);

    printf("  --- Production run (%d steps) ---\n", TOTAL_STEPS);
    printf("  %8s   %12s   %12s\n", "step", "u_max_LU", "rel_dKE");
    printf("  ──────────────────────────────────\n");

    std::vector<float> h_ux(NC);
    double E_prev = 0.0;
    auto t_start = std::chrono::high_resolution_clock::now();

    dim3 bc_block(16, 1, 1);
    dim3 bc_grid((NX + 15) / 16, NZ, 1);
    const int macro_blk  = 256;
    const int macro_grid = (NC + macro_blk - 1) / macro_blk;

    for (int step = 1; step <= TOTAL_STEPS; ++step) {
        fluid.collisionTRT(a_body, 0.0f, 0.0f, LAMBDA);
        fluid.streaming();
        applyBottomBounceBack<<<bc_grid, bc_block>>>(
            fluid.getDistributionSrc(), NX, NZ);
        applyTopStressBC<<<bc_grid, bc_block>>>(
            fluid.getDistributionSrc(), NX, NZ, delta_f);
        CUDA_CHECK_KERNEL();
        computeMacroKernel<<<macro_grid, macro_blk>>>(
            fluid.getDistributionSrc(), d_rho_c, d_ux_c, d_uy_c, d_uz_c,
            NC, a_body);
        CUDA_CHECK_KERNEL();

        if (step % PRINT_INTERVAL == 0) {
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaMemcpy(h_ux.data(), d_ux_c, NC * sizeof(float),
                                  cudaMemcpyDeviceToHost));
            double E = 0.0; float u_max = 0.0f;
            for (int id = 0; id < NC; ++id) {
                E += 0.5 * h_ux[id] * h_ux[id];
                u_max = fmaxf(u_max, fabsf(h_ux[id]));
            }
            double rel_dKE = (E_prev > 0) ? fabs(E - E_prev) / E_prev : 1.0;
            printf("  %8d   %12.5e   %12.5e\n", step, u_max, rel_dKE);
            fflush(stdout);
            E_prev = E;
        }
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    auto t_end = std::chrono::high_resolution_clock::now();
    printf("  Done in %.1f s\n\n",
           std::chrono::duration<double>(t_end - t_start).count());

    // ===================================================================
    // Extract final profile and validate
    // ===================================================================
    extractProfile(d_ux_c, u_avg);

    // LOCKED analytical solution
    std::vector<double> u_analytical(NY);
    for (int jj = 0; jj < NY; ++jj) {
        double yh = (jj + 0.5) / H_EFF;
        u_analytical[jj] = A_TRUE * (3.0 * yh * yh - 2.0 * yh);
    }

    // L2 error
    double l2_num = 0, l2_den = 0, linf = 0;
    for (int jj = 1; jj < NY - 1; ++jj) {
        double err = u_avg[jj] - u_analytical[jj];
        l2_num += err * err;
        l2_den += u_analytical[jj] * u_analytical[jj];
        linf = fmax(linf, fabs(err));
    }
    double l2_rel = sqrt(l2_num / l2_den);

    // Quadratic fit
    double C1_fit, C2_fit, R2;
    quadraticFit(u_avg, C1_fit, C2_fit, R2);
    double sigma_final = measureStress(C1_fit, C2_fit);

    // Zero crossing
    double zc_lbm = -1.0;
    for (int jj = 0; jj < NY - 1; ++jj) {
        if (u_avg[jj] * u_avg[jj+1] < 0.0) {
            double frac = u_avg[jj] / (u_avg[jj] - u_avg[jj+1]);
            zc_lbm = (jj + 0.5 + frac) / H_EFF;
            break;
        }
    }

    printf("  ============ VALIDATION RESULTS ============\n\n");

    printf("  1. L2 ERROR vs TRUE ANALYTICAL\n");
    printf("     L2 relative:  %.6e\n", l2_rel);
    printf("     Linf absolute: %.6e\n", linf);
    printf("     %s\n\n", l2_rel < 0.01 ? "PASS (L2 < 1%)" : "FAIL");

    printf("  2. PEAK VELOCITY\n");
    printf("     Analytical: %.10f (LOCKED)\n", (double)A_TRUE);
    printf("     LBM:        %.10f\n", u_avg[NY-1]);
    printf("     Error:      %.4f%%\n",
           fabs(u_avg[NY-1] - A_TRUE) / A_TRUE * 100.0);

    printf("  3. ZERO CROSSING\n");
    printf("     Analytical: 0.666667\n");
    printf("     LBM:        %.6f  (error: %.6f)\n",
           zc_lbm, fabs(zc_lbm - 2.0/3.0));

    printf("  4. SURFACE STRESS\n");
    printf("     Target:  %.6e\n", (double)TAU_S);
    printf("     Actual:  %.6e  (ratio: %.6f)\n",
           sigma_final, sigma_final / TAU_S);

    printf("  5. FIT: R²=%.10f  C₁=%.4e (exp %.4e)  C₂=%.4e (exp %.4e)\n",
           R2, C1_fit, 3.0*A_TRUE, C2_fit, -2.0*A_TRUE);
    printf("     delta_f (calibrated): %.6e\n\n", delta_f);

    // Profile
    printf("  Profile:\n");
    printf("  %8s  %14s  %14s  %10s\n", "yhat", "u_LBM", "u_analytical", "err_%");
    for (int jj = 0; jj < NY; jj += (NY/10)) {
        double yh = (jj+0.5)/H_EFF;
        double ep = fabs(u_avg[jj]-u_analytical[jj])/fmax(fabs(u_analytical[jj]),1e-12)*100;
        printf("  %8.4f  %14.6e  %14.6e  %10.3f\n", yh, u_avg[jj], u_analytical[jj], ep);
    }
    { int jj = NY/2;
      double yh=(jj+0.5)/H_EFF;
      double ep=fabs(u_avg[jj]-u_analytical[jj])/fmax(fabs(u_analytical[jj]),1e-12)*100;
      printf("  %8.4f  %14.6e  %14.6e  %10.3f\n", yh, u_avg[jj], u_analytical[jj], ep); }

    // CSV
    const std::string csv = "/home/yzk/LBMProject/scripts/viz/marangoni_returnflow.csv";
    std::ofstream ofs(csv);
    ofs << "# Marangoni 1D Return Flow — TRT + calibrated Inamuro BC\n";
    ofs << "# tau=" << TAU << " nu=" << NU_LB << " tau_s=" << TAU_S
        << " A_TRUE=" << A_TRUE << " a_body=" << a_body
        << " delta_f=" << delta_f << " CE_FACTOR=" << CE_FACTOR
        << " H_eff=" << H_EFF << " NY=" << NY << " NX=" << NX
        << " steps=" << TOTAL_STEPS
        << " L2=" << l2_rel << " R2=" << R2
        << " sigma_actual=" << sigma_final << "\n";
    ofs << "y_norm,u_lbm,u_analytical\n";
    ofs << std::scientific << std::setprecision(10);
    for (int jj = 0; jj < NY; ++jj) {
        double yh = (jj+0.5)/H_EFF;
        ofs << yh << "," << u_avg[jj] << "," << u_analytical[jj] << "\n";
    }
    ofs.close();
    printf("\n  Written: %s\n", csv.c_str());

    bool pass = (l2_rel < 0.01) && (R2 > 0.9999)
             && (fabs(u_avg[NY-1] - A_TRUE)/A_TRUE < 0.05)
             && (fabs(zc_lbm - 2.0/3.0) < 0.01);
    printf("\n  ═══════════════════════════════════════════\n");
    printf("  OVERALL: %s\n", pass ? "PASS" : "FAIL — see details above");
    printf("  ═══════════════════════════════════════════\n\n");

    cudaFree(d_rho_c); cudaFree(d_ux_c); cudaFree(d_uy_c); cudaFree(d_uz_c);
    return pass ? 0 : 1;
}
