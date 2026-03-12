/**
 * @file test_stefan_problem.cu
 * @brief Stefan problem benchmark: ESM phase change validation
 *
 * Validates the Enthalpy Source Term (ESM, Jiaung 2001) implementation
 * against the Neumann analytical solution for 1D semi-infinite melting.
 *
 * Setup:
 *   - Pure metal: T_solidus ≈ T_liquidus (ΔT_melt = 0.1 K)
 *   - Constant properties: cp=500, k=50, ρ=7000 (solid = liquid)
 *   - Stefan number Ste = cp·ΔT_wall / L = 0.5
 *   - Left wall Dirichlet T = T_hot = 1100 K
 *   - Domain initialized at T_melt = 1000 K
 *
 * Analytical:
 *   s(t) = 2λ√(αt),  λ·exp(λ²)·erf(λ) = Ste/√π
 *   T(x,t) = T_hot - (T_hot - T_melt)·erf(x/(2√(αt))) / erf(λ)
 *
 * Pass criteria:
 *   - Front position error < 5% at all snapshots
 *   - Temperature L2 error < 5% in liquid region
 *   - Convergence with grid refinement
 */

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cmath>
#include <vector>
#include <algorithm>
#include <iostream>
#include <iomanip>
#include <cstring>
#include "physics/thermal_lbm.h"
#include "physics/material_properties.h"

using namespace lbm::physics;

// ============================================================================
// Pure metal material (eliminates mushy-zone ambiguity)
// ============================================================================
static MaterialProperties createPureMetal() {
    MaterialProperties mat = {};
    std::strncpy(mat.name, "PureMetal_Ste0.5", sizeof(mat.name) - 1);

    mat.rho_solid  = 7000.0f;   mat.rho_liquid  = 7000.0f;
    mat.cp_solid   = 500.0f;    mat.cp_liquid   = 500.0f;
    mat.k_solid    = 50.0f;     mat.k_liquid    = 50.0f;
    mat.mu_liquid  = 0.005f;

    // Near-isothermal melting (ΔT = 0.1 K avoids /0, approximates pure metal)
    mat.T_solidus      = 1000.0f;
    mat.T_liquidus     = 1000.1f;
    mat.T_vaporization = 3000.0f;

    // L = cp·ΔT_wall / Ste = 500·100 / 0.5 = 100,000 J/kg
    mat.L_fusion       = 100000.0f;
    mat.L_vaporization = 6.0e6f;

    mat.surface_tension    = 1.0f;
    mat.dsigma_dT          = -1.0e-4f;
    mat.absorptivity_solid = 0.3f;
    mat.absorptivity_liquid= 0.3f;
    mat.emissivity         = 0.3f;

    return mat;
}

// ============================================================================
// Solve Neumann transcendental equation: λ·exp(λ²)·erf(λ) = Ste/√π
// ============================================================================
static float solveLambda(float Ste) {
    const float target = Ste / sqrtf(static_cast<float>(M_PI));
    float lam = 0.3f;

    for (int iter = 0; iter < 200; ++iter) {
        float exp_l2 = expf(lam * lam);
        float erf_l  = erff(lam);
        float f  = lam * exp_l2 * erf_l - target;
        float df = exp_l2 * erf_l
                 + lam * exp_l2 * (2.0f * lam * erf_l + 2.0f / sqrtf(static_cast<float>(M_PI)));
        float dlam = f / df;
        lam -= dlam;
        if (fabsf(dlam) < 1.0e-10f) break;
    }
    return lam;
}

// ============================================================================
// Test fixture
// ============================================================================
class StefanProblemTest : public ::testing::Test {
protected:
    // Grid and material
    static constexpr int NX = 400;
    static constexpr int NY = 1;
    static constexpr int NZ = 1;
    static constexpr float DOMAIN_LENGTH = 2.0e-3f;  // 2 mm

    MaterialProperties mat;
    float T_melt, T_hot, T_cold;
    float alpha, Ste, lam;
    float dx, dt, tau;
    ThermalLBM* solver = nullptr;

    void SetUp() override {
        if (!D3Q7::isInitialized()) {
            D3Q7::initializeDevice();
        }

        mat    = createPureMetal();
        T_melt = mat.T_solidus;           // 1000 K
        T_hot  = T_melt + 100.0f;         // 1100 K
        T_cold = T_melt;                  // 1000 K

        alpha = mat.k_solid / (mat.rho_solid * mat.cp_solid);  // 1.4286e-5
        Ste   = mat.cp_solid * (T_hot - T_melt) / mat.L_fusion; // 0.5
        lam   = solveLambda(Ste);

        dx  = DOMAIN_LENGTH / static_cast<float>(NX);
        tau = 0.8f;
        float alpha_LU = (tau - 0.5f) * D3Q7::CS2;  // 0.075
        dt  = alpha_LU * dx * dx / alpha;

        solver = new ThermalLBM(NX, NY, NZ, mat, alpha, true, dt, dx);
        solver->initialize(T_cold);

        std::cout << "\n=== Pure-Metal Stefan (ESM) ===" << std::endl;
        std::cout << "  Ste=" << Ste << "  λ=" << lam
                  << "  α=" << alpha << " m²/s" << std::endl;
        std::cout << "  NX=" << NX << "  dx=" << dx*1e6 << " µm"
                  << "  dt=" << dt << " s  τ=" << tau << std::endl;
    }

    void TearDown() override {
        delete solver;
    }

    // Run LBM to target_time, applying Dirichlet BC AFTER computeTemperature
    void runToTime(float target_time, float& current_time) {
        int steps = static_cast<int>((target_time - current_time) / dt + 0.5f);
        for (int s = 0; s < steps; ++s) {
            solver->collisionBGK();
            solver->streaming();
            solver->computeTemperature();
            // Dirichlet BC at x=0: bc_type=2 (DIRICHLET), MUST be after computeTemperature
            solver->applyFaceThermalBC(0, 2, dt, dx, T_hot);
        }
        current_time = target_time;
    }

    // Find numerical front position (fl=0.5 crossing with linear interpolation)
    float findFrontPosition(const std::vector<float>& h_fl) {
        for (int i = 1; i < NX; ++i) {
            if (h_fl[i - 1] >= 0.5f && h_fl[i] < 0.5f) {
                float x0 = (i - 1) * dx;
                return x0 + (0.5f - h_fl[i - 1]) / (h_fl[i] - h_fl[i - 1]) * dx;
            }
        }
        return 0.0f;  // no front found
    }
};

// ============================================================================
// Test 1: Front position at multiple snapshots (HARD: < 5% error)
// ============================================================================
TEST_F(StefanProblemTest, FrontPositionAccuracy) {
    const float t_snaps[] = {0.2e-3f, 0.5e-3f, 1.0e-3f, 1.5e-3f, 2.0e-3f};
    const int n_snaps = 5;
    float current_time = 0.0f;

    std::vector<float> h_fl(NX);

    std::cout << "\n  snap  t[ms]   s_anal[µm]  s_num[µm]  err[%]" << std::endl;
    std::cout << "  ----  ------  ----------  ---------  ------" << std::endl;

    for (int s = 0; s < n_snaps; ++s) {
        runToTime(t_snaps[s], current_time);
        solver->copyLiquidFractionToHost(h_fl.data());

        float s_anal = 2.0f * lam * sqrtf(alpha * current_time);
        float s_num  = findFrontPosition(h_fl);
        float err_pct = (s_anal > 0.0f) ? fabsf(s_num - s_anal) / s_anal * 100.0f : -1.0f;

        std::cout << "  " << s << "     "
                  << std::fixed << std::setprecision(3) << current_time * 1e3f << "   "
                  << std::setprecision(1) << s_anal * 1e6f << "       "
                  << s_num * 1e6f << "      "
                  << std::setprecision(2) << err_pct << std::endl;

        EXPECT_GT(s_num, 0.0f) << "No melting front detected at t=" << t_snaps[s]*1e3 << " ms";
        EXPECT_LT(err_pct, 5.0f) << "Front position error > 5% at t=" << t_snaps[s]*1e3 << " ms";
    }
}

// ============================================================================
// Test 2: Temperature profile in liquid region (L2 error < 5%)
// ============================================================================
TEST_F(StefanProblemTest, TemperatureProfileAccuracy) {
    float target_time = 1.0e-3f;
    float current_time = 0.0f;
    runToTime(target_time, current_time);

    std::vector<float> h_temp(NX), h_fl(NX);
    solver->copyTemperatureToHost(h_temp.data());
    solver->copyLiquidFractionToHost(h_fl.data());

    float s_anal = 2.0f * lam * sqrtf(alpha * current_time);
    float sqrt_at = sqrtf(alpha * current_time);
    float erf_lam = erff(lam);

    float sum_sq = 0.0f;
    int count = 0;

    for (int i = 0; i < NX; ++i) {
        float x = i * dx;
        if (x < 0.8f * s_anal && h_fl[i] > 0.95f) {
            float eta = x / (2.0f * sqrt_at);
            float T_anal = T_hot - (T_hot - T_melt) * erff(eta) / erf_lam;
            float err = (h_temp[i] - T_anal) / (T_hot - T_melt);
            sum_sq += err * err;
            count++;
        }
    }

    float L2 = (count > 0) ? sqrtf(sum_sq / count) * 100.0f : -1.0f;
    std::cout << "\n  Temperature L2 error in liquid: " << L2 << "% (" << count << " samples)" << std::endl;

    EXPECT_GT(count, 5) << "Too few fully-liquid cells for profile validation";
    EXPECT_LT(L2, 5.0f) << "Temperature L2 error > 5% in liquid region";
}

// ============================================================================
// Test 3: Latent heat storage (positive, reasonable magnitude)
// ============================================================================
TEST_F(StefanProblemTest, LatentHeatStorage) {
    float target_time = 1.0e-3f;
    float current_time = 0.0f;
    runToTime(target_time, current_time);

    std::vector<float> h_fl(NX);
    solver->copyLiquidFractionToHost(h_fl.data());

    float V_cell = dx * dx * dx;
    float Q_latent = 0.0f;
    for (int i = 0; i < NX; ++i) {
        Q_latent += h_fl[i] * mat.rho_solid * mat.L_fusion * V_cell;
    }

    std::cout << "\n  Latent heat stored: " << Q_latent << " J" << std::endl;
    EXPECT_GT(Q_latent, 0.0f) << "No latent heat stored";

    // Verify monotonic fl profile (should decrease from wall toward solid)
    bool monotonic = true;
    for (int i = 1; i < NX; ++i) {
        if (h_fl[i] > h_fl[i-1] + 0.01f) {
            monotonic = false;
            break;
        }
    }
    EXPECT_TRUE(monotonic) << "Liquid fraction not monotonically decreasing from wall";
}

// ============================================================================
// Test 4: Grid convergence (error decreases with refinement)
// ============================================================================
TEST_F(StefanProblemTest, SpatialConvergence) {
    // Use the fixture's solver for NX=400, create coarser ones manually
    const int NX_levels[] = {100, 200, 400};
    const int n_levels = 3;
    float errors[3];

    float test_time = 1.0e-3f;

    std::cout << "\n  Grid convergence:" << std::endl;

    for (int level = 0; level < n_levels; ++level) {
        int nx = NX_levels[level];
        float dx_l = DOMAIN_LENGTH / static_cast<float>(nx);
        float alpha_LU = (tau - 0.5f) * D3Q7::CS2;
        float dt_l = alpha_LU * dx_l * dx_l / alpha;

        ThermalLBM* s = new ThermalLBM(nx, 1, 1, mat, alpha, true, dt_l, dx_l);
        s->initialize(T_cold);

        int steps = static_cast<int>(test_time / dt_l + 0.5f);
        for (int st = 0; st < steps; ++st) {
            s->collisionBGK();
            s->streaming();
            s->computeTemperature();
            s->applyFaceThermalBC(0, 2, dt_l, dx_l, T_hot);
        }

        std::vector<float> fl(nx);
        s->copyLiquidFractionToHost(fl.data());

        // Find front
        float s_num = 0.0f;
        for (int i = 1; i < nx; ++i) {
            if (fl[i-1] >= 0.5f && fl[i] < 0.5f) {
                float x0 = (i-1) * dx_l;
                s_num = x0 + (0.5f - fl[i-1]) / (fl[i] - fl[i-1]) * dx_l;
                break;
            }
        }

        float s_anal = 2.0f * lam * sqrtf(alpha * test_time);
        float err_pct = fabsf(s_num - s_anal) / s_anal * 100.0f;
        errors[level] = err_pct;

        std::cout << "    NX=" << nx << "  dx=" << dx_l*1e6 << " µm  err=" << err_pct << "%" << std::endl;

        delete s;
    }

    // Verify convergence: finer grid should have lower error
    for (int level = 0; level < n_levels - 1; ++level) {
        EXPECT_LT(errors[level + 1], errors[level] * 1.05f)
            << "Error did not decrease from NX=" << NX_levels[level]
            << " to NX=" << NX_levels[level + 1];
    }

    // Finest grid must be < 5%
    EXPECT_LT(errors[n_levels - 1], 5.0f) << "Finest grid error > 5%";
}

// ============================================================================
// Test 5: sqrt(t) scaling of front position
// ============================================================================
TEST_F(StefanProblemTest, SqrtTScaling) {
    const float t_snaps[] = {0.5e-3f, 1.0e-3f, 2.0e-3f};
    const int n = 3;
    float current_time = 0.0f;
    float s_num[3];

    std::vector<float> h_fl(NX);

    for (int i = 0; i < n; ++i) {
        runToTime(t_snaps[i], current_time);
        solver->copyLiquidFractionToHost(h_fl.data());
        s_num[i] = findFrontPosition(h_fl);
    }

    // s(t) = C·√t  →  s(t2)/s(t1) ≈ √(t2/t1)
    // t2/t1 = 2.0 → expected ratio = √2 ≈ 1.414
    float ratio_1_2 = s_num[1] / s_num[0];  // s(1ms)/s(0.5ms) ≈ √2
    float ratio_2_3 = s_num[2] / s_num[1];  // s(2ms)/s(1ms) ≈ √2

    float expected = sqrtf(2.0f);

    std::cout << "\n  sqrt(t) scaling:" << std::endl;
    std::cout << "    s(1ms)/s(0.5ms) = " << ratio_1_2 << " (expected " << expected << ")" << std::endl;
    std::cout << "    s(2ms)/s(1ms)   = " << ratio_2_3 << " (expected " << expected << ")" << std::endl;

    EXPECT_NEAR(ratio_1_2, expected, 0.15f) << "s(t) does not scale as sqrt(t)";
    EXPECT_NEAR(ratio_2_3, expected, 0.15f) << "s(t) does not scale as sqrt(t)";
}

// ============================================================================
// Test 6: Analytical parameter verification
// ============================================================================
TEST_F(StefanProblemTest, AnalyticalParameters) {
    // Verify Ste = 0.5
    EXPECT_NEAR(Ste, 0.5f, 0.001f);

    // Verify λ ≈ 0.4654 (for Ste=0.5)
    EXPECT_NEAR(lam, 0.4654f, 0.005f);

    // Verify α = k/(ρ·cp) = 50/(7000·500)
    float alpha_expected = 50.0f / (7000.0f * 500.0f);
    EXPECT_NEAR(alpha, alpha_expected, 1e-8f);

    // Verify transcendental equation: λ·exp(λ²)·erf(λ) = Ste/√π
    float lhs = lam * expf(lam * lam) * erff(lam);
    float rhs = Ste / sqrtf(static_cast<float>(M_PI));
    EXPECT_NEAR(lhs, rhs, 1e-5f);

    std::cout << "\n  Ste=" << Ste << "  λ=" << lam << "  α=" << alpha << std::endl;
    std::cout << "  λ·exp(λ²)·erf(λ) = " << lhs << "  Ste/√π = " << rhs << std::endl;
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
