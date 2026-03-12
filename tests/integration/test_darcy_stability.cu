/**
 * @file test_darcy_stability.cu
 * @brief Integration test: Semi-implicit Darcy damping in mushy/solid zones
 *
 * Validates that the semi-implicit Darcy treatment:
 *   u = [Σ(ci·fi) + 0.5·F_other] / (ρ + 0.5·K)
 * correctly and stably drives velocity to zero in solid regions, even with
 * extreme Darcy coefficients (C = 1e15, fl = 1e-5).
 *
 * Test scenarios:
 * 1. Channel flow entering a near-solid wall (fl=1e-5, C=1e15) — no NaN
 * 2. Mushy zone deceleration (fl from 1.0 to 0.0 gradient) — smooth profile
 * 3. Comparison with explicit method (should NaN) vs semi-implicit (stable)
 */

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cmath>
#include <vector>
#include <iostream>
#include <iomanip>
#include <numeric>
#include "physics/fluid_lbm.h"
#include "physics/force_accumulator.h"

using namespace lbm::physics;

class DarcyStabilityTest : public ::testing::Test {
protected:
    void SetUp() override {
        cudaSetDevice(0);
    }

    void TearDown() override {
        cudaDeviceSynchronize();
    }

    /**
     * Check if any element is NaN or Inf
     */
    bool hasNaN(const std::vector<float>& v) {
        for (float x : v) {
            if (std::isnan(x) || std::isinf(x)) return true;
        }
        return false;
    }

    /**
     * Compute max absolute value
     */
    float maxAbs(const std::vector<float>& v) {
        float m = 0.0f;
        for (float x : v) m = std::max(m, std::abs(x));
        return m;
    }
};

/**
 * Test 1: Extreme Darcy braking — fl=1e-5, C=1e15
 *
 * Setup: Small domain with uniform initial velocity. A solid zone (fl~0)
 * covers the right half. Run 200 steps. Verify:
 * - No NaN anywhere
 * - Velocity in solid zone < 1e-6
 * - Velocity in liquid zone remains physical (not exploded)
 */
TEST_F(DarcyStabilityTest, ExtremeDarcyBraking_NoNaN) {
    // Small domain: 32 x 8 x 1
    const int NX = 32, NY = 8, NZ = 1;
    const int NC = NX * NY * NZ;

    // LBM parameters (lattice units: dx=1, dt=1)
    const float nu = 0.1f;      // kinematic viscosity
    const float rho = 1.0f;     // density
    const float dx = 1.0f;
    const float dt = 1.0f;

    // Extreme Darcy parameters
    const float C_darcy = 1e15f;
    const float fl_solid = 1e-5f;

    // Create fluid solver (walls in y, periodic in x/z)
    FluidLBM fluid(NX, NY, NZ, nu, rho,
                   BoundaryType::PERIODIC,
                   BoundaryType::WALL,
                   BoundaryType::PERIODIC,
                   dt, dx);

    // Initialize with uniform rightward velocity
    const float u0 = 0.05f;
    fluid.initialize(rho, u0, 0.0f, 0.0f);

    // Create liquid fraction field: left half liquid, right half near-solid
    std::vector<float> h_lf(NC);
    for (int j = 0; j < NY; ++j) {
        for (int i = 0; i < NX; ++i) {
            int idx = i + NX * j;  // NZ=1
            if (i < NX / 2) {
                h_lf[idx] = 1.0f;       // Liquid
            } else {
                h_lf[idx] = fl_solid;   // Near-solid
            }
        }
    }

    float* d_lf;
    cudaMalloc(&d_lf, NC * sizeof(float));
    cudaMemcpy(d_lf, h_lf.data(), NC * sizeof(float), cudaMemcpyHostToDevice);

    // Create ForceAccumulator and compute Darcy coefficient field
    ForceAccumulator forces(NX, NY, NZ);
    forces.computeDarcyCoefficientField(d_lf, C_darcy, rho, dx, dt);

    // Zero force arrays (no body forces, only Darcy)
    forces.reset();
    const float* darcy_K = forces.getDarcyCoefficient();

    // Time-stepping
    const int STEPS = 200;
    for (int step = 0; step < STEPS; ++step) {
        // Collision with zero force (Darcy is semi-implicit, not in force)
        fluid.collisionBGK(forces.getFx(), forces.getFy(), forces.getFz());
        fluid.streaming();
        // Semi-implicit macroscopic with Darcy coefficient
        fluid.computeMacroscopic(forces.getFx(), forces.getFy(),
                                  forces.getFz(), darcy_K);
    }

    // Extract velocity
    std::vector<float> h_ux(NC), h_uy(NC), h_uz(NC);
    fluid.copyVelocityToHost(h_ux.data(), h_uy.data(), h_uz.data());

    // Check: no NaN
    ASSERT_FALSE(hasNaN(h_ux)) << "NaN in ux after " << STEPS << " steps with C=" << C_darcy;
    ASSERT_FALSE(hasNaN(h_uy)) << "NaN in uy after " << STEPS << " steps with C=" << C_darcy;

    // Check: velocity in solid zone is effectively zero
    float max_v_solid = 0.0f;
    float max_v_liquid = 0.0f;
    for (int j = 1; j < NY - 1; ++j) {  // Skip walls
        for (int i = 0; i < NX; ++i) {
            int idx = i + NX * j;
            float v = std::sqrt(h_ux[idx] * h_ux[idx] + h_uy[idx] * h_uy[idx]);
            if (i >= NX / 2 + 2) {  // Deep in solid (skip transition cells)
                max_v_solid = std::max(max_v_solid, v);
            }
            if (i < NX / 2 - 2) {  // Deep in liquid
                max_v_liquid = std::max(max_v_liquid, v);
            }
        }
    }

    std::cout << "  max |u| in solid zone: " << std::scientific << max_v_solid << std::endl;
    std::cout << "  max |u| in liquid zone: " << std::scientific << max_v_liquid << std::endl;

    EXPECT_LT(max_v_solid, 1e-6f)
        << "Velocity in solid zone should be ~0 with extreme Darcy";
    EXPECT_GT(max_v_liquid, 1e-8f)
        << "Liquid zone should still have some velocity";

    // Check: density is physical (near 1.0)
    std::vector<float> h_rho(NC);
    fluid.copyDensityToHost(h_rho.data());
    ASSERT_FALSE(hasNaN(h_rho)) << "NaN in density!";

    float max_rho_dev = 0.0f;
    for (int i = 0; i < NC; ++i) {
        max_rho_dev = std::max(max_rho_dev, std::abs(h_rho[i] - rho));
    }
    std::cout << "  max |ρ - 1|: " << std::scientific << max_rho_dev << std::endl;
    EXPECT_LT(max_rho_dev, 0.1f) << "Density deviation too large — pressure wave";

    cudaFree(d_lf);
}

/**
 * Test 2: Smooth mushy zone deceleration gradient
 *
 * Setup: Linear liquid fraction gradient from 1.0 (left) to 0.0 (right).
 * Moderate Darcy coefficient C=1e6. Verify velocity decreases monotonically
 * through the mushy zone.
 */
TEST_F(DarcyStabilityTest, MushyZoneSmooth_MonotonicDeceleration) {
    const int NX = 64, NY = 8, NZ = 1;
    const int NC = NX * NY * NZ;

    const float nu = 0.1f;
    const float rho = 1.0f;
    const float dx = 1.0f;
    const float dt = 1.0f;
    const float C_darcy = 1e6f;

    FluidLBM fluid(NX, NY, NZ, nu, rho,
                   BoundaryType::PERIODIC,
                   BoundaryType::WALL,
                   BoundaryType::PERIODIC,
                   dt, dx);

    fluid.initialize(rho, 0.05f, 0.0f, 0.0f);

    // Linear fl gradient: 1.0 at i=0, 0.001 at i=NX-1
    std::vector<float> h_lf(NC);
    for (int j = 0; j < NY; ++j) {
        for (int i = 0; i < NX; ++i) {
            float frac = 1.0f - (float)i / (float)(NX - 1);
            h_lf[i + NX * j] = std::max(0.001f, frac);  // Minimum 0.001
        }
    }

    float* d_lf;
    cudaMalloc(&d_lf, NC * sizeof(float));
    cudaMemcpy(d_lf, h_lf.data(), NC * sizeof(float), cudaMemcpyHostToDevice);

    ForceAccumulator forces(NX, NY, NZ);
    forces.computeDarcyCoefficientField(d_lf, C_darcy, rho, dx, dt);
    forces.reset();
    const float* darcy_K = forces.getDarcyCoefficient();

    // Run to near-steady state
    for (int step = 0; step < 500; ++step) {
        fluid.collisionBGK(forces.getFx(), forces.getFy(), forces.getFz());
        fluid.streaming();
        fluid.computeMacroscopic(forces.getFx(), forces.getFy(),
                                  forces.getFz(), darcy_K);
    }

    std::vector<float> h_ux(NC);
    float* dummy_y = new float[NC];
    float* dummy_z = new float[NC];
    fluid.copyVelocityToHost(h_ux.data(), dummy_y, dummy_z);
    delete[] dummy_y;
    delete[] dummy_z;

    // Check no NaN
    ASSERT_FALSE(hasNaN(h_ux)) << "NaN in mushy zone gradient test";

    // Check velocity profile at mid-height (j = NY/2)
    int j_mid = NY / 2;
    std::cout << "  Velocity profile at j=" << j_mid << ":" << std::endl;
    int non_monotonic_count = 0;
    float prev_v = 1.0f;
    for (int i = 0; i < NX; ++i) {
        float v = std::abs(h_ux[i + NX * j_mid]);
        if (i % 8 == 0) {
            std::cout << "    i=" << std::setw(3) << i
                      << "  fl=" << std::fixed << std::setprecision(3) << h_lf[i + NX * j_mid]
                      << "  |ux|=" << std::scientific << std::setprecision(4) << v
                      << std::endl;
        }
        // Allow small non-monotonicity at boundaries (first 2, last 2 cells)
        if (i > 2 && i < NX - 2 && v > prev_v * 1.1f) {
            non_monotonic_count++;
        }
        prev_v = v;
    }

    // Velocity should be roughly decreasing (allow some tolerance for LBM noise)
    EXPECT_LE(non_monotonic_count, 3)
        << "Velocity should decrease roughly monotonically through mushy zone";

    // Velocity at solid end should be much smaller than liquid end
    float v_liquid = std::abs(h_ux[2 + NX * j_mid]);
    float v_solid = std::abs(h_ux[(NX - 3) + NX * j_mid]);
    std::cout << "  Liquid end |ux|: " << std::scientific << v_liquid << std::endl;
    std::cout << "  Solid end |ux|:  " << std::scientific << v_solid << std::endl;
    EXPECT_LT(v_solid, v_liquid * 0.1f)
        << "Solid end should be much slower than liquid end";

    cudaFree(d_lf);
}

/**
 * Test 3: Verify Darcy coefficient computation is correct
 *
 * For fl=0.5 (mushy), C=1e6, ρ=1, dt=1:
 *   K_CK = C·(1-fl)²/(fl³+ε) = 1e6·0.25/(0.125+0.001) = 1.984e6
 *   K_LU = K_CK · ρ · dt = 1.984e6
 *
 * For fl=1.0 (liquid): K = 0
 * For fl=0.001 (near-solid): K very large
 */
TEST_F(DarcyStabilityTest, DarcyCoefficientValues_Correct) {
    const int N = 4;
    const int NC = N;  // 4 x 1 x 1

    const float C = 1e6f;
    const float rho = 1.0f;
    const float dx = 1.0f;
    const float dt = 1.0f;

    // Test values: fl = {1.0, 0.5, 0.1, 0.001}
    std::vector<float> h_lf = {1.0f, 0.5f, 0.1f, 0.001f};
    float* d_lf;
    cudaMalloc(&d_lf, NC * sizeof(float));
    cudaMemcpy(d_lf, h_lf.data(), NC * sizeof(float), cudaMemcpyHostToDevice);

    ForceAccumulator forces(N, 1, 1);
    forces.computeDarcyCoefficientField(d_lf, C, rho, dx, dt);

    // Read back K values
    std::vector<float> h_K(NC);
    cudaMemcpy(h_K.data(), forces.getDarcyCoefficient(), NC * sizeof(float),
               cudaMemcpyDeviceToHost);

    const float eps = 1e-3f;

    // fl=1.0: K = C · 0² / (1+ε) = 0
    std::cout << "  K(fl=1.0) = " << std::scientific << h_K[0] << std::endl;
    EXPECT_NEAR(h_K[0], 0.0f, 1.0f) << "Liquid should have ~0 Darcy coefficient";

    // fl=0.5: K = C·0.25/(0.125+ε)·ρ·dt
    float K_expected_05 = C * 0.25f / (0.125f + eps) * rho * dt;
    std::cout << "  K(fl=0.5) = " << std::scientific << h_K[1]
              << "  expected=" << K_expected_05 << std::endl;
    EXPECT_NEAR(h_K[1], K_expected_05, K_expected_05 * 0.01f);

    // fl=0.1: K = C·0.81/(0.001+ε)·ρ·dt
    float K_expected_01 = C * 0.81f / (0.001f + eps) * rho * dt;
    std::cout << "  K(fl=0.1) = " << std::scientific << h_K[2]
              << "  expected=" << K_expected_01 << std::endl;
    EXPECT_NEAR(h_K[2], K_expected_01, K_expected_01 * 0.01f);

    // fl=0.001: K should be very large (near-solid)
    float K_expected_001 = C * (0.999f * 0.999f) / (0.001f * 0.001f * 0.001f + eps) * rho * dt;
    std::cout << "  K(fl=0.001) = " << std::scientific << h_K[3]
              << "  expected=" << K_expected_001 << std::endl;
    EXPECT_NEAR(h_K[3], K_expected_001, K_expected_001 * 0.01f);

    cudaFree(d_lf);
}

/**
 * Test 4: No pressure oscillation in steady state
 *
 * After many steps, the flow in the solid zone should settle to zero
 * without oscillatory pressure waves.
 */
TEST_F(DarcyStabilityTest, NoPressureOscillation) {
    const int NX = 32, NY = 8, NZ = 1;
    const int NC = NX * NY * NZ;

    const float nu = 0.1f;
    const float rho = 1.0f;
    const float dx = 1.0f;
    const float dt = 1.0f;
    const float C_darcy = 1e10f;  // Very high but not extreme

    FluidLBM fluid(NX, NY, NZ, nu, rho,
                   BoundaryType::PERIODIC,
                   BoundaryType::WALL,
                   BoundaryType::PERIODIC,
                   dt, dx);

    fluid.initialize(rho, 0.05f, 0.0f, 0.0f);

    // Uniform solid: fl = 0.01 everywhere
    std::vector<float> h_lf(NC, 0.01f);
    float* d_lf;
    cudaMalloc(&d_lf, NC * sizeof(float));
    cudaMemcpy(d_lf, h_lf.data(), NC * sizeof(float), cudaMemcpyHostToDevice);

    ForceAccumulator forces(NX, NY, NZ);
    forces.computeDarcyCoefficientField(d_lf, C_darcy, rho, dx, dt);
    forces.reset();
    const float* darcy_K = forces.getDarcyCoefficient();

    // Record velocity at 3 time points to check for oscillation
    std::vector<float> v_snapshots;

    for (int step = 0; step < 300; ++step) {
        fluid.collisionBGK(forces.getFx(), forces.getFy(), forces.getFz());
        fluid.streaming();
        fluid.computeMacroscopic(forces.getFx(), forces.getFy(),
                                  forces.getFz(), darcy_K);

        if (step == 50 || step == 150 || step == 299) {
            std::vector<float> h_ux(NC);
            float* dy = new float[NC];
            float* dz = new float[NC];
            fluid.copyVelocityToHost(h_ux.data(), dy, dz);
            delete[] dy;
            delete[] dz;

            ASSERT_FALSE(hasNaN(h_ux)) << "NaN at step " << step;
            float max_v = maxAbs(h_ux);
            v_snapshots.push_back(max_v);
            std::cout << "  step " << std::setw(3) << step
                      << "  max|ux| = " << std::scientific << max_v << std::endl;
        }
    }

    // Velocity should be monotonically decreasing (no oscillation)
    ASSERT_EQ(v_snapshots.size(), 3u);
    EXPECT_LE(v_snapshots[1], v_snapshots[0] * 1.01f)
        << "Velocity should not increase between step 50 and 150";
    EXPECT_LE(v_snapshots[2], v_snapshots[1] * 1.01f)
        << "Velocity should not increase between step 150 and 300";

    // Final velocity should be very small
    EXPECT_LT(v_snapshots[2], 1e-4f)
        << "Velocity should be ~0 after 300 steps in uniform solid";

    // Check pressure: should be uniform (no oscillations)
    std::vector<float> h_p(NC);
    fluid.copyPressureToHost(h_p.data());
    ASSERT_FALSE(hasNaN(h_p)) << "NaN in pressure";

    float p_min = *std::min_element(h_p.begin(), h_p.end());
    float p_max = *std::max_element(h_p.begin(), h_p.end());
    float p_range = p_max - p_min;
    std::cout << "  Pressure range: " << std::scientific << p_range
              << "  (min=" << p_min << " max=" << p_max << ")" << std::endl;

    // Pressure variation should be small (no checkerboard oscillation)
    EXPECT_LT(p_range, 0.01f) << "Pressure oscillation detected in solid zone";

    cudaFree(d_lf);
}
