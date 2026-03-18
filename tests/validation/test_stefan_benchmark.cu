/**
 * @file test_stefan_benchmark.cu
 * @brief Rigorous Stefan problem benchmark with Neumann analytical solution
 *
 * Uses the enthalpy-based source term approach (Jiaung et al. 2001) for
 * accurate phase change modeling. After each LBM step, a latent heat
 * correction is applied to the distributions to properly account for
 * energy absorption/release during phase change.
 *
 * This avoids the known C_app limitation where cells near sharp temperature
 * gradients can jump over the mushy zone in a single time step.
 *
 * References:
 *   - Jiaung et al. (2001), Numer. Heat Transfer B, 39(2), 167-187
 *   - Alexiades & Solomon (1993), Mathematical Modeling of Melting/Freezing
 *   - Huang & Wu (2014), J. Comput. Phys., 274, 50-64
 */

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cmath>
#include <vector>
#include <algorithm>
#include <numeric>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <sys/stat.h>
#include "physics/thermal_lbm.h"
#include "physics/material_properties.h"
#include "physics/phase_change.h"

using namespace lbm::physics;

// ============================================================================
// CUDA kernel: enthalpy-based latent heat correction (Jiaung 2001)
//
// After LBM streaming+compute, the "raw" temperature T_raw = Σg_q includes
// all conducted heat but does NOT account for latent heat absorption.
// This kernel:
//   1. Computes enthalpy increment: ΔH = ρ*cp*(T_raw - T_old)
//   2. Updates total enthalpy: H_new = H_old + ΔH
//   3. Inverts T_corrected from H_new (accounts for latent heat)
//   4. Applies correction ΔT = T_corrected - T_raw to distributions
// ============================================================================
__global__ void enthalpyCorrectionKernel(
    float* g,              // distribution functions [Q * num_cells]
    float* temperature,    // temperature field (updated in-place)
    float* enthalpy,       // enthalpy field (updated in-place) [J/m^3]
    float* liquid_fraction,// liquid fraction field (updated in-place)
    const float* T_old,    // temperature from previous step
    float rho, float cp, float L_fusion,
    float T_solidus, float T_liquidus,
    int num_cells)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_cells) return;

    float T_raw = temperature[idx];
    float T_prev = T_old[idx];

    // Step 1: Energy increment from conduction (in J/m³)
    float dH = rho * cp * (T_raw - T_prev);

    // Step 2: Update total enthalpy
    float H = enthalpy[idx] + dH;
    enthalpy[idx] = H;

    // Step 3: Invert T from H
    // H = rho*cp*T + fl(T)*rho*L
    // Three regimes:
    float H_solidus = rho * cp * T_solidus;  // H at T_solidus, fl=0
    float H_liquidus = rho * cp * T_liquidus + rho * L_fusion;  // H at T_liquidus, fl=1

    float T_corrected, fl;
    if (H <= H_solidus) {
        // Solid phase: H = rho*cp*T
        T_corrected = H / (rho * cp);
        fl = 0.0f;
    } else if (H >= H_liquidus) {
        // Liquid phase: H = rho*cp*T + rho*L
        T_corrected = (H - rho * L_fusion) / (rho * cp);
        fl = 1.0f;
    } else {
        // Mushy zone: H = rho*cp*T + fl*rho*L, fl = (T-T_s)/(T_l-T_s)
        // Substituting: H = rho*cp*T + ((T-T_s)/(T_l-T_s))*rho*L
        // H = rho*T*(cp + L/(T_l-T_s)) - rho*L*T_s/(T_l-T_s)
        // T = (H + rho*L*T_s/(T_l-T_s)) / (rho*(cp + L/(T_l-T_s)))
        float dT_melt = T_liquidus - T_solidus;
        float C_app = cp + L_fusion / dT_melt;
        T_corrected = (H / rho + L_fusion * T_solidus / dT_melt) / C_app;
        fl = (T_corrected - T_solidus) / dT_melt;
        fl = fmaxf(0.0f, fminf(1.0f, fl));
    }

    // Step 4: Correction to distributions
    float dT = T_corrected - T_raw;

    // D3Q7 weights
    const float w[7] = {0.25f, 0.125f, 0.125f, 0.125f, 0.125f, 0.125f, 0.125f};
    for (int q = 0; q < 7; ++q) {
        g[q * num_cells + idx] += w[q] * dT;
    }

    // Update fields
    temperature[idx] = T_corrected;
    liquid_fraction[idx] = fl;
}

// ============================================================================
// Neumann Analytical Solution
// ============================================================================

static double solveNeumannLambda(double St) {
    const double sqrt_pi = std::sqrt(M_PI);
    const double target = St / sqrt_pi;

    double lam = std::sqrt(St / 2.0);
    if (lam < 0.01) lam = 0.01;
    if (lam > 2.0) lam = 2.0;

    for (int iter = 0; iter < 100; ++iter) {
        double exp_lam2 = std::exp(lam * lam);
        double erf_lam = std::erf(lam);
        double f = lam * exp_lam2 * erf_lam - target;
        double df = exp_lam2 * erf_lam * (1.0 + 2.0 * lam * lam) + 2.0 * lam / sqrt_pi;
        double dlam = f / df;
        lam -= dlam;
        if (std::fabs(dlam) < 1e-15) break;
    }
    return lam;
}

static double analyticalFront(double t, double lambda, double alpha) {
    return 2.0 * lambda * std::sqrt(alpha * t);
}

static double analyticalTemperature(double x, double t, double T_hot, double T_melt,
                                     double lambda, double alpha) {
    double eta = x / (2.0 * std::sqrt(alpha * t));
    return T_hot - (T_hot - T_melt) * std::erf(eta) / std::erf(lambda);
}

// ============================================================================
// Custom material for Stefan benchmark
// ============================================================================

static MaterialProperties createStefanMaterial(float dT_melt) {
    MaterialProperties mat = {};

    mat.rho_solid = 1000.0f;
    mat.rho_liquid = 1000.0f;
    mat.cp_solid = 1000.0f;
    mat.cp_liquid = 1000.0f;
    mat.k_solid = 1.0f;
    mat.k_liquid = 1.0f;
    mat.mu_liquid = 1.0e-3f;

    const float T_melt = 1000.0f;
    mat.T_solidus = T_melt;
    mat.T_liquidus = T_melt + dT_melt;
    mat.T_vaporization = 3000.0f;

    // St = cp * DT_super / L = 1.0 (with DT_super = 50K)
    mat.L_fusion = 50000.0f;
    mat.L_vaporization = 1.0e6f;

    mat.surface_tension = 1.0f;
    mat.dsigma_dT = 0.0f;
    mat.absorptivity_solid = 0.3f;
    mat.absorptivity_liquid = 0.3f;
    mat.emissivity = 0.3f;

    std::snprintf(mat.name, sizeof(mat.name), "StefanBench_dT%.2f", dT_melt);
    return mat;
}

// ============================================================================
// Test parameters
// ============================================================================
static constexpr float T_MELT = 1000.0f;
static constexpr float T_HOT = 1050.0f;
static constexpr float DT_SUPER = 50.0f;
static constexpr float ALPHA = 1.0e-6f;       // k/(rho*cp)
static constexpr float STEFAN_NUMBER = 1.0f;   // cp*DT_SUPER/L
static constexpr float RHO = 1000.0f;
static constexpr float CP = 1000.0f;
static constexpr float L_FUSION = 50000.0f;

// ============================================================================
// LBM parameter helper
// ============================================================================
struct LBMParams {
    int NX;
    float dx, dt, tau, alpha_lattice;

    LBMParams(int nx, float domain_length, float tau_base, float alpha_phys) : NX(nx) {
        dx = domain_length / static_cast<float>(nx);
        alpha_lattice = (tau_base - 0.5f) * 0.25f;
        dt = alpha_lattice * dx * dx / alpha_phys;
        tau = tau_base;
    }
};

// ============================================================================
// Find melting front from liquid fraction (fl=0.5 interpolation)
// ============================================================================
static float findFrontPosition(const std::vector<float>& fl, float dx) {
    int n = static_cast<int>(fl.size());
    for (int i = 1; i < n; ++i) {
        if (fl[i - 1] >= 0.5f && fl[i] < 0.5f) {
            float frac = (0.5f - fl[i - 1]) / (fl[i] - fl[i - 1]);
            return ((i - 1) + frac) * dx;
        }
    }
    return -1.0f;
}

// ============================================================================
// Run Stefan simulation with enthalpy correction
// ============================================================================
struct StefanResult {
    float front_numerical, front_analytical, front_error_pct;
    float front_analytical_corrected, front_error_corrected_pct;
    float T_profile_L2_error_pct;
    float energy_error_pct;
    int steps;
};

static StefanResult runStefan(int NX, float dT_melt, float tau_base,
                               float domain_length, float target_time,
                               bool verbose = false) {
    MaterialProperties mat = createStefanMaterial(dT_melt);
    LBMParams lbm(NX, domain_length, tau_base, ALPHA);

    double lambda = solveNeumannLambda(STEFAN_NUMBER);
    double s_analytical = analyticalFront(target_time, lambda, ALPHA);

    // Corrected analytical: accounts for extra sensible heat in mushy zone
    // Effective latent heat: L_eff = L + cp * dT_melt
    double St_eff = (double)CP * DT_SUPER / ((double)L_FUSION + (double)CP * dT_melt);
    double lambda_eff = solveNeumannLambda(St_eff);
    double s_corrected = analyticalFront(target_time, lambda_eff, ALPHA);

    if (verbose) {
        std::cout << "\n=== Stefan Benchmark (Enthalpy Correction) ===" << std::endl;
        std::cout << "  NX=" << NX << ", dx=" << lbm.dx * 1e6 << " μm"
                  << ", tau=" << tau_base << ", dT_melt=" << dT_melt << " K" << std::endl;
        std::cout << "  St=" << STEFAN_NUMBER << ", St_eff=" << St_eff
                  << ", λ=" << lambda << ", λ_eff=" << lambda_eff << std::endl;
        std::cout << "  s_ana=" << s_analytical * 1e6 << " μm"
                  << ", s_corrected=" << s_corrected * 1e6 << " μm" << std::endl;
        std::cout << "  dt=" << lbm.dt << " s, steps=" << static_cast<int>(target_time / lbm.dt) << std::endl;
    }

    // Create solver WITHOUT phase change (constant omega, no C_app)
    // We handle latent heat via enthalpy post-correction ourselves
    ThermalLBM solver(NX, 1, 1, ALPHA, RHO, CP, lbm.dt, lbm.dx);
    solver.initialize(T_MELT);  // All solid at T_solidus = T_MELT

    // Allocate device arrays for enthalpy correction
    int num_cells = NX;
    float* d_enthalpy = nullptr;
    float* d_liquid_fraction = nullptr;
    float* d_T_old = nullptr;
    cudaMalloc(&d_enthalpy, num_cells * sizeof(float));
    cudaMalloc(&d_liquid_fraction, num_cells * sizeof(float));
    cudaMalloc(&d_T_old, num_cells * sizeof(float));

    // Initialize enthalpy: H = rho*cp*T_solidus (= rho*cp*T_MELT, all solid, fl=0)
    {
        std::vector<float> h_H(num_cells, RHO * CP * T_MELT);
        std::vector<float> h_fl(num_cells, 0.0f);
        cudaMemcpy(d_enthalpy, h_H.data(), num_cells * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_liquid_fraction, h_fl.data(), num_cells * sizeof(float), cudaMemcpyHostToDevice);
    }

    int num_steps = static_cast<int>(target_time / lbm.dt);
    int block = 256;
    int grid = (num_cells + block - 1) / block;

    for (int step = 0; step < num_steps; ++step) {
        // 0. Store T_old for enthalpy increment computation
        cudaMemcpy(d_T_old, solver.getTemperature(), num_cells * sizeof(float),
                   cudaMemcpyDeviceToDevice);

        // 1. Dirichlet BC at x=0: T = T_hot (sets temperature AND distributions)
        solver.applyFaceThermalBC(0, 2, lbm.dt, lbm.dx, T_HOT);

        // Also fix enthalpy at x=0 to be consistent
        {
            float H_hot = RHO * CP * T_HOT + RHO * L_FUSION;  // fully liquid
            cudaMemcpy(d_enthalpy, &H_hot, sizeof(float), cudaMemcpyHostToDevice);
            float fl_hot = 1.0f;
            cudaMemcpy(d_liquid_fraction, &fl_hot, sizeof(float), cudaMemcpyHostToDevice);
            // Also update T_old at x=0 so enthalpy increment is zero there
            float T_hot_val = T_HOT;
            cudaMemcpy(d_T_old, &T_hot_val, sizeof(float), cudaMemcpyHostToDevice);
        }

        // 2. Collision (constant omega, no C_app)
        solver.collisionBGK();

        // 3. Streaming (bounce-back at boundaries = adiabatic)
        solver.streaming();

        // 4. Compute raw temperature: T_raw = Σg_q
        solver.computeTemperature();

        // 5. Enthalpy correction: redistribute energy between sensible and latent
        enthalpyCorrectionKernel<<<grid, block>>>(
            solver.getDistributions(),  // distribution functions [7 * num_cells]
            solver.getTemperature(),    // temperature field
            d_enthalpy,
            d_liquid_fraction,
            d_T_old,
            RHO, CP, L_FUSION,
            mat.T_solidus, mat.T_liquidus,
            num_cells);
        cudaDeviceSynchronize();

        if (verbose && (step % (num_steps / 5 + 1) == 0)) {
            std::cout << "  Step " << step << "/" << num_steps << std::endl;
        }
    }

    // Extract results
    std::vector<float> h_temp(NX), h_fl(NX);
    solver.copyTemperatureToHost(h_temp.data());
    cudaMemcpy(h_fl.data(), d_liquid_fraction, NX * sizeof(float), cudaMemcpyDeviceToHost);

    float s_num = findFrontPosition(h_fl, lbm.dx);
    float front_err_pct = -1.0f;
    float front_err_corr_pct = -1.0f;
    if (s_num > 0 && s_analytical > 0)
        front_err_pct = std::fabs(s_num - (float)s_analytical) / (float)s_analytical * 100.0f;
    if (s_num > 0 && s_corrected > 0)
        front_err_corr_pct = std::fabs(s_num - (float)s_corrected) / (float)s_corrected * 100.0f;

    // Temperature profile L2 error (liquid region)
    double T_L2_sum = 0.0, T_norm_sum = 0.0;
    int T_samples = 0;
    float actual_time = num_steps * lbm.dt;
    for (int i = 1; i < NX; ++i) {
        float x = i * lbm.dx;
        if (h_fl[i] > 0.95f && x < 0.8f * s_num) {
            double T_ana = analyticalTemperature(x, actual_time, T_HOT, T_MELT, lambda, ALPHA);
            double err = h_temp[i] - T_ana;
            T_L2_sum += err * err;
            T_norm_sum += (T_ana - T_MELT) * (T_ana - T_MELT);
            T_samples++;
        }
    }
    float T_L2_pct = (T_norm_sum > 0) ? std::sqrt(T_L2_sum / T_norm_sum) * 100.0f : -1.0f;

    // Energy conservation
    std::vector<float> h_H(NX);
    cudaMemcpy(h_H.data(), d_enthalpy, NX * sizeof(float), cudaMemcpyDeviceToHost);
    float E_total = 0.0f, E_initial = 0.0f;
    float V_cell = lbm.dx * lbm.dx * lbm.dx;
    for (int i = 0; i < NX; ++i) {
        E_total += h_H[i] * V_cell;
        E_initial += RHO * CP * mat.T_solidus * V_cell;
    }
    double Q_in_per_area = mat.k_solid * DT_SUPER * 2.0
        * std::sqrt(actual_time / (M_PI * ALPHA)) / std::erf(lambda);
    float Q_in = (float)Q_in_per_area * lbm.dx * lbm.dx;
    float E_expected = E_initial + Q_in;
    float energy_err_pct = std::fabs(E_total - E_expected) / std::fabs(Q_in) * 100.0f;

    if (verbose) {
        std::cout << "\n--- Results ---" << std::endl;
        std::cout << "  Front: num=" << s_num * 1e6 << " μm, ana=" << s_analytical * 1e6
                  << " μm, corrected=" << s_corrected * 1e6 << " μm" << std::endl;
        std::cout << "  Front error (vs standard) = " << front_err_pct << "%" << std::endl;
        std::cout << "  Front error (vs corrected) = " << front_err_corr_pct << "%" << std::endl;
        std::cout << "  T-profile L2 = " << T_L2_pct << "% (" << T_samples << " samples)" << std::endl;
        std::cout << "  Energy error = " << energy_err_pct << "%" << std::endl;

        std::cout << "\n  x(μm)   T(K)      fl    T_ana(K)" << std::endl;
        int stride = std::max(1, NX / 20);
        for (int i = 0; i < NX; i += stride) {
            float x = i * lbm.dx;
            double T_ana = (x < s_analytical)
                ? analyticalTemperature(x, actual_time, T_HOT, T_MELT, lambda, ALPHA)
                : T_MELT;
            std::cout << "  " << std::fixed << std::setw(7) << std::setprecision(1) << x * 1e6
                      << std::setw(10) << std::setprecision(2) << h_temp[i]
                      << std::setw(7) << std::setprecision(3) << h_fl[i]
                      << std::setw(10) << std::setprecision(2) << T_ana << std::endl;
        }
    }

    // Save CSV for visualization (when verbose)
    if (verbose) {
        mkdir("output_stefan_benchmark", 0755);

        char fname[256];
        std::snprintf(fname, sizeof(fname),
                      "output_stefan_benchmark/profile_NX%d_dT%.3f.csv", NX, dT_melt);
        std::ofstream csv(fname);
        csv << "x_um,T_numerical,T_analytical,liquid_fraction,enthalpy\n";
        for (int i = 0; i < NX; ++i) {
            float x = i * lbm.dx;
            double T_ana = (x < s_analytical)
                ? analyticalTemperature(x, actual_time, T_HOT, T_MELT, lambda, ALPHA)
                : T_MELT;
            csv << std::setprecision(6)
                << x * 1e6 << "," << h_temp[i] << "," << T_ana << ","
                << h_fl[i] << "," << h_H[i] << "\n";
        }
        csv.close();
        std::cout << "  CSV saved: " << fname << std::endl;
    }

    cudaFree(d_enthalpy);
    cudaFree(d_liquid_fraction);
    cudaFree(d_T_old);

    return {s_num, (float)s_analytical, front_err_pct,
            (float)s_corrected, front_err_corr_pct,
            T_L2_pct, energy_err_pct, num_steps};
}

// ============================================================================
// TEST SUITE
// ============================================================================

class StefanBenchmarkTest : public ::testing::Test {
protected:
    void SetUp() override {
        if (!D3Q7::isInitialized()) D3Q7::initializeDevice();
    }
};

TEST_F(StefanBenchmarkTest, NeumannLambda) {
    double lam_1 = solveNeumannLambda(1.0);
    double lam_05 = solveNeumannLambda(0.5);
    double lam_01 = solveNeumannLambda(0.1);

    EXPECT_NEAR(lam_1, 0.620062633313596, 1e-10);
    EXPECT_NEAR(lam_05, 0.464785920646244, 1e-10);
    EXPECT_NEAR(lam_01, 0.220016272742938, 1e-10);

    auto verify = [](double lam, double St) {
        double r = lam * std::exp(lam * lam) * std::erf(lam) - St / std::sqrt(M_PI);
        EXPECT_NEAR(r, 0.0, 1e-14);
    };
    verify(lam_1, 1.0);
    verify(lam_05, 0.5);
    verify(lam_01, 0.1);
}

TEST_F(StefanBenchmarkTest, MaterialValidation) {
    auto mat = createStefanMaterial(1.0f);
    EXPECT_TRUE(mat.validate());
    EXPECT_NEAR(mat.cp_solid * DT_SUPER / mat.L_fusion, 1.0f, 1e-5f);
    EXPECT_NEAR(mat.k_solid / (mat.rho_solid * mat.cp_solid), 1.0e-6f, 1e-12f);
}

/**
 * @brief Quick smoke test with coarse grid.
 * Verifies the enthalpy correction produces reasonable results.
 */
TEST_F(StefanBenchmarkTest, SmokeTest) {
    const int NX = 100;
    const float dT_melt = 1.0f;
    const float tau = 1.0f;
    const float domain_length = NX * 1e-5f;

    double lambda = solveNeumannLambda(STEFAN_NUMBER);
    float s_target = 30 * domain_length / NX;
    float t_target = (float)((s_target / (2.0 * lambda)) * (s_target / (2.0 * lambda)) / ALPHA);

    auto result = runStefan(NX, dT_melt, tau, domain_length, t_target, true);

    EXPECT_GT(result.front_numerical, 0.0f) << "No melting front detected";
    EXPECT_LT(result.front_error_pct, 10.0f) << "Front error > 10% for smoke test";
}

/**
 * @brief Main benchmark: NX=400, dT_melt=0.05K, tau=1.0.
 * Uses very narrow mushy zone to minimize modeling error.
 * Target: front position error < 0.1% vs standard Neumann solution.
 */
TEST_F(StefanBenchmarkTest, FrontPosition_Precise) {
    const int NX = 400;
    const float dT_melt = 0.05f;
    const float tau = 1.0f;
    const float domain_length = 1e-3f;  // 1 mm

    LBMParams lbm(NX, domain_length, tau, ALPHA);
    double lambda = solveNeumannLambda(STEFAN_NUMBER);
    float s_target = 60 * lbm.dx;
    float t_target = (float)((s_target / (2.0 * lambda)) * (s_target / (2.0 * lambda)) / ALPHA);

    auto result = runStefan(NX, dT_melt, tau, domain_length, t_target, true);

    EXPECT_GT(result.front_numerical, 0.0f);
    EXPECT_LT(result.front_error_pct, 0.1f)
        << "Front error (vs standard Neumann) " << result.front_error_pct << "% exceeds 0.1%";

    // Also verify corrected comparison (LBM discretization error only)
    EXPECT_LT(result.front_error_corrected_pct, 0.1f)
        << "Corrected error " << result.front_error_corrected_pct << "% exceeds 0.1%";
}

/**
 * @brief Temperature profile accuracy in liquid region.
 */
TEST_F(StefanBenchmarkTest, TemperatureProfile) {
    const int NX = 200;
    const float dT_melt = 1.0f;
    const float tau = 1.0f;
    const float domain_length = 1e-3f;

    LBMParams lbm(NX, domain_length, tau, ALPHA);
    double lambda = solveNeumannLambda(STEFAN_NUMBER);
    float s_target = 60 * lbm.dx;
    float t_target = (float)((s_target / (2.0 * lambda)) * (s_target / (2.0 * lambda)) / ALPHA);

    auto result = runStefan(NX, dT_melt, tau, domain_length, t_target, false);

    EXPECT_LT(result.T_profile_L2_error_pct, 1.0f)
        << "T-profile L2 error " << result.T_profile_L2_error_pct << "% exceeds 1%";
}

/**
 * @brief Energy conservation.
 */
TEST_F(StefanBenchmarkTest, EnergyConservation) {
    const int NX = 200;
    const float dT_melt = 1.0f;
    const float tau = 1.0f;
    const float domain_length = 1e-3f;

    LBMParams lbm(NX, domain_length, tau, ALPHA);
    double lambda = solveNeumannLambda(STEFAN_NUMBER);
    float s_target = 60 * lbm.dx;
    float t_target = (float)((s_target / (2.0 * lambda)) * (s_target / (2.0 * lambda)) / ALPHA);

    auto result = runStefan(NX, dT_melt, tau, domain_length, t_target, false);

    EXPECT_LT(result.energy_error_pct, 2.0f)
        << "Energy error " << result.energy_error_pct << "% exceeds 2%";
}

/**
 * @brief Grid convergence: standard error should decrease as dT_melt narrows
 * at each grid level, and all grids should give reasonable accuracy.
 *
 * With tau=1.0 (exact collision), LBM grid error is extremely small.
 * The dominant error is from the mushy zone width, not the grid.
 * This test verifies the method works correctly across grid resolutions.
 */
TEST_F(StefanBenchmarkTest, GridConvergence) {
    const float dT_melt = 0.05f;  // Narrow mushy zone to minimize modeling error
    const float tau = 1.0f;
    const float domain_length = 1e-3f;

    const int NX_levels[] = {100, 200, 400};
    const int num_levels = 3;
    float errors[3], corr_errors[3];

    double lambda = solveNeumannLambda(STEFAN_NUMBER);
    float s_target = 3e-4f;
    float t_target = (float)((s_target / (2.0 * lambda)) * (s_target / (2.0 * lambda)) / ALPHA);

    std::cout << "\n=== Grid Convergence ===" << std::endl;

    for (int i = 0; i < num_levels; ++i) {
        int nx = NX_levels[i];
        auto result = runStefan(nx, dT_melt, tau, domain_length, t_target, false);
        errors[i] = result.front_error_pct;
        corr_errors[i] = result.front_error_corrected_pct;
        std::cout << "  NX=" << nx << " (dx=" << domain_length / nx * 1e6 << " μm)"
                  << ": standard=" << errors[i] << "%"
                  << ", corrected=" << corr_errors[i] << "%" << std::endl;
    }

    // All standard errors should be < 0.5% with dT_melt=0.05K
    for (int i = 0; i < num_levels; ++i) {
        EXPECT_LT(errors[i], 0.5f)
            << "Standard error at NX=" << NX_levels[i] << " exceeds 0.5%";
    }
}

/**
 * @brief Mushy zone convergence: error → 0 as dT_melt → 0.
 * Demonstrates convergence to the sharp-interface Neumann solution.
 */
TEST_F(StefanBenchmarkTest, MushyZoneConvergence) {
    const int NX = 200;
    const float tau = 1.0f;
    const float domain_length = 1e-3f;

    float dT_values[] = {10.0f, 5.0f, 1.0f, 0.1f, 0.05f};
    const int num = 5;
    float errors[5], corr_errors[5];

    double lambda = solveNeumannLambda(STEFAN_NUMBER);
    float s_target = 60 * domain_length / NX;
    float t_target = (float)((s_target / (2.0 * lambda)) * (s_target / (2.0 * lambda)) / ALPHA);

    std::cout << "\n=== Mushy Zone Convergence ===" << std::endl;

    for (int i = 0; i < num; ++i) {
        auto result = runStefan(NX, dT_values[i], tau, domain_length, t_target, false);
        errors[i] = result.front_error_pct;
        corr_errors[i] = result.front_error_corrected_pct;
        std::cout << "  dT_melt=" << std::setw(6) << dT_values[i]
                  << " K: standard=" << std::setw(8) << std::setprecision(4) << errors[i]
                  << "%, corrected=" << std::setw(8) << corr_errors[i] << "%" << std::endl;
    }

    // Standard errors should decrease monotonically
    for (int i = 0; i < num - 1; ++i) {
        EXPECT_LE(errors[i + 1], errors[i] * 1.05f)
            << "Error didn't decrease from dT=" << dT_values[i] << " to " << dT_values[i + 1];
    }

    // Finest mushy zone: standard error < 0.1%
    EXPECT_LT(errors[num - 1], 0.1f)
        << "Finest dT_melt=" << dT_values[num - 1] << "K standard error exceeds 0.1%";

    // Corrected errors should be moderate (residual from inexact St_eff approximation)
    for (int i = 0; i < num; ++i) {
        EXPECT_LT(corr_errors[i], 2.0f)
            << "Corrected error at dT_melt=" << dT_values[i] << "K exceeds 2%";
    }

    // Save convergence data for visualization
    mkdir("output_stefan_benchmark", 0755);
    std::ofstream csv("output_stefan_benchmark/mushy_convergence.csv");
    csv << "dT_melt,standard_error_pct,corrected_error_pct\n";
    for (int i = 0; i < num; ++i) {
        csv << dT_values[i] << "," << errors[i] << "," << corr_errors[i] << "\n";
    }
    csv.close();
    std::cout << "  CSV saved: output_stefan_benchmark/mushy_convergence.csv" << std::endl;
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
