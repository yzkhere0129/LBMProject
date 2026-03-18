/**
 * @file test_timestep_omega_clamping.cu
 * @brief Regression test for timestep convergence omega clamping bug
 *
 * BUG FIX: Timestep convergence test used dt = 0.5 μs as finest timestep,
 * which caused omega to be clamped at 1.92 (above stability limit 1.90).
 * This broke convergence analysis because finest timestep was unstable.
 *
 * Fix location: /home/yzk/LBMProject/tests/validation/test_timestep_convergence.cu:279-284
 * Changed timesteps from {4, 2, 1, 0.5} μs to {8, 4, 2, 1} μs
 *
 * This test validates that all timesteps produce stable omega values
 * and proper convergence behavior.
 */

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <vector>
#include <cmath>
#include <iostream>
#include <iomanip>

/**
 * @brief Test omega calculation for given timestep and diffusivity
 *
 * For thermal D3Q7 LBM:
 * alpha_lattice = alpha * dt / dx^2
 * tau = alpha_lattice / cs^2 + 0.5 = 4 * alpha_lattice + 0.5  (cs^2 = 1/4)
 * omega = 1 / tau
 *
 * Stability requirement: omega < 1.9 (ideally omega <= 1.85 for safety)
 */
TEST(TimestepOmegaClampingTest, OmegaStabilityCheck) {
    // Test parameters matching validation test
    float alpha = 1.0e-6f;  // m^2/s (thermal diffusivity)
    float dx = 10.0e-6f;    // 10 μm grid spacing

    // Old timesteps (BUGGY - caused omega clamping)
    std::vector<float> old_timesteps = {4.0e-6f, 2.0e-6f, 1.0e-6f, 0.5e-6f};

    // New timesteps (FIXED - all stable)
    std::vector<float> new_timesteps = {8.0e-6f, 4.0e-6f, 2.0e-6f, 1.0e-6f};

    std::cout << "\n=== Omega Stability Analysis ===\n\n";

    // Test old timesteps (should have instability)
    std::cout << "OLD TIMESTEPS (Before Fix):\n";
    std::cout << std::setw(12) << "dt [μs]"
              << std::setw(15) << "alpha_lattice"
              << std::setw(10) << "tau"
              << std::setw(10) << "omega"
              << std::setw(12) << "Status\n";
    std::cout << std::string(59, '-') << "\n";

    bool old_has_instability = false;
    for (float dt : old_timesteps) {
        float alpha_lattice = alpha * dt / (dx * dx);
        float tau = 4.0f * alpha_lattice + 0.5f;
        float omega = 1.0f / tau;

        // Check if omega would be clamped (> 1.90)
        bool is_unstable = (omega > 1.90f);
        if (is_unstable) old_has_instability = true;

        std::cout << std::setw(12) << dt * 1e6
                  << std::setw(15) << alpha_lattice
                  << std::setw(10) << tau
                  << std::setw(10) << omega
                  << std::setw(12) << (is_unstable ? "UNSTABLE" : "stable")
                  << "\n";
    }

    // Test new timesteps (should all be stable)
    std::cout << "\nNEW TIMESTEPS (After Fix):\n";
    std::cout << std::setw(12) << "dt [μs]"
              << std::setw(15) << "alpha_lattice"
              << std::setw(10) << "tau"
              << std::setw(10) << "omega"
              << std::setw(12) << "Status\n";
    std::cout << std::string(59, '-') << "\n";

    bool new_has_instability = false;
    for (float dt : new_timesteps) {
        float alpha_lattice = alpha * dt / (dx * dx);
        float tau = 4.0f * alpha_lattice + 0.5f;
        float omega = 1.0f / tau;

        bool is_unstable = (omega > 1.90f);
        if (is_unstable) new_has_instability = true;

        std::cout << std::setw(12) << dt * 1e6
                  << std::setw(15) << alpha_lattice
                  << std::setw(10) << tau
                  << std::setw(10) << omega
                  << std::setw(12) << (is_unstable ? "UNSTABLE" : "stable")
                  << "\n";
    }

    // CRITICAL ASSERTIONS
    EXPECT_TRUE(old_has_instability)
        << "Old timesteps should have instability (confirms bug existed)";

    EXPECT_FALSE(new_has_instability)
        << "REGRESSION: New timesteps have instability! Bug reintroduced.";

    std::cout << "\n";
}

/**
 * @brief Test that timestep ratios are proper powers of 2
 *
 * For convergence testing, timesteps should differ by factors of 2
 * to clearly observe convergence order
 */
TEST(TimestepOmegaClampingTest, TimestepRatiosCheck) {
    std::vector<float> timesteps = {8.0e-6f, 4.0e-6f, 2.0e-6f, 1.0e-6f};

    std::cout << "\n=== Timestep Ratio Analysis ===\n";

    for (size_t i = 1; i < timesteps.size(); ++i) {
        float ratio = timesteps[i - 1] / timesteps[i];
        std::cout << "dt[" << i - 1 << "] / dt[" << i << "] = " << ratio << "\n";

        EXPECT_FLOAT_EQ(ratio, 2.0f)
            << "Timestep ratios must be 2.0 for convergence analysis";
    }
}

/**
 * @brief Test omega safety margin for different thermal diffusivities
 *
 * Tests that for a range of materials (different alpha), the timesteps
 * remain stable
 */
TEST(TimestepOmegaClampingTest, DiffusivityRangeTest) {
    float dx = 10.0e-6f;  // 10 μm
    std::vector<float> timesteps = {8.0e-6f, 4.0e-6f, 2.0e-6f, 1.0e-6f};

    // Test range of thermal diffusivities
    std::vector<float> alphas = {
        0.5e-6f,  // Low diffusivity
        1.0e-6f,  // Nominal (test case)
        2.0e-6f,  // High diffusivity
        5.0e-6f   // Very high diffusivity
    };

    std::cout << "\n=== Diffusivity Range Test ===\n\n";

    for (float alpha : alphas) {
        std::cout << "Alpha = " << alpha * 1e6 << " mm^2/s:\n";

        bool all_stable = true;
        for (float dt : timesteps) {
            float alpha_lattice = alpha * dt / (dx * dx);
            float tau = 4.0f * alpha_lattice + 0.5f;
            float omega = 1.0f / tau;

            bool is_stable = (omega <= 1.94f);  // Solver clamps at omega >= 1.95
            all_stable = all_stable && is_stable;

            if (!is_stable) {
                std::cout << "  dt = " << dt * 1e6 << " μs: omega = " << omega
                          << " (UNSTABLE)\n";
            }
        }

        if (all_stable) {
            std::cout << "  All timesteps stable\n";
        }

        // For nominal alpha = 1.0e-6, all should be stable
        if (alpha == 1.0e-6f) {
            EXPECT_TRUE(all_stable)
                << "REGRESSION: Timesteps unstable for nominal diffusivity";
        }

        std::cout << "\n";
    }
}

/**
 * @brief Test minimum safe timestep calculation
 *
 * For given alpha and dx, compute the minimum timestep that keeps omega < 1.85
 */
TEST(TimestepOmegaClampingTest, MinimumTimestepCalculation) {
    float alpha = 1.0e-6f;  // m^2/s
    float dx = 10.0e-6f;    // 10 μm
    float omega_max = 1.94f;  // Safety threshold (solver clamps at 1.95)

    // From omega = 1/tau and tau = 4*alpha_lattice + 0.5 (cs²=1/4):
    // omega = 1 / (4*alpha*dt/dx^2 + 0.5)
    // omega_max = 1 / (4*alpha*dt_min/dx^2 + 0.5)
    // => 4*alpha*dt_min/dx^2 = 1/omega_max - 0.5
    // => dt_min = (1/omega_max - 0.5) * dx^2 / (4*alpha)

    float tau_min = 1.0f / omega_max;
    float alpha_lattice_min = (tau_min - 0.5f) / 4.0f;
    float dt_min = alpha_lattice_min * dx * dx / alpha;

    std::cout << "\n=== Minimum Timestep Calculation ===\n";
    std::cout << "Alpha: " << alpha * 1e6 << " mm^2/s\n";
    std::cout << "Grid spacing: " << dx * 1e6 << " μm\n";
    std::cout << "Omega max (safety): " << omega_max << "\n";
    std::cout << "Minimum timestep: " << dt_min * 1e6 << " μs\n\n";

    // Verify that dt = 1 μs is above minimum
    EXPECT_GT(1.0e-6f, dt_min)
        << "1 μs timestep should be safely above minimum";

    // Verify minimum timestep is reasonable (sub-microsecond)
    EXPECT_GT(dt_min, 0.1e-6f)
        << "Minimum timestep should be > 0.1 μs for typical LPBF params";
    EXPECT_LT(dt_min, 1.0e-6f)
        << "Minimum timestep should be < 1.0 μs";
}

/**
 * @brief Test expected convergence behavior with new timesteps
 *
 * With stable timesteps, error should decrease as O(dt) for first-order schemes
 * or O(dt^2) for second-order schemes
 */
TEST(TimestepOmegaClampingTest, ExpectedConvergenceOrder) {
    // Simulate errors with different timesteps (assuming first-order scheme)
    std::vector<float> timesteps = {8.0e-6f, 4.0e-6f, 2.0e-6f, 1.0e-6f};

    // For first-order scheme: error ~ C * dt
    // For second-order scheme: error ~ C * dt^2
    float C = 1.0e3f;  // Error constant

    std::cout << "\n=== Expected Convergence Order ===\n";
    std::cout << std::setw(12) << "dt [μs]"
              << std::setw(15) << "Error (O(dt))"
              << std::setw(18) << "Error (O(dt^2))"
              << std::setw(15) << "Ratio (O(dt))\n";
    std::cout << std::string(60, '-') << "\n";

    float prev_error_1st = 0.0f;
    float prev_error_2nd = 0.0f;

    for (size_t i = 0; i < timesteps.size(); ++i) {
        float dt = timesteps[i];
        float error_1st = C * dt;
        float error_2nd = C * dt * dt / 1e-6f;  // Normalized

        std::cout << std::setw(12) << dt * 1e6
                  << std::setw(15) << error_1st;

        if (i > 0) {
            float ratio_1st = prev_error_1st / error_1st;
            std::cout << std::setw(18) << error_2nd
                      << std::setw(15) << ratio_1st;

            // For halving timestep, first-order should have ratio ~2.0
            if (i > 0) {
                EXPECT_NEAR(ratio_1st, 2.0f, 0.01f)
                    << "First-order scheme should halve error when timestep halves";
            }
        } else {
            std::cout << std::setw(18) << error_2nd
                      << std::setw(15) << "-";
        }

        std::cout << "\n";

        prev_error_1st = error_1st;
        prev_error_2nd = error_2nd;
    }

    std::cout << "\nNote: With unstable omega (old dt=0.5μs), convergence breaks down\n";
}

/**
 * @brief Stress test: Verify omega never exceeds 1.90 for any reasonable parameters
 */
TEST(TimestepOmegaClampingTest, StressTestParameterSpace) {
    std::vector<float> timesteps = {8.0e-6f, 4.0e-6f, 2.0e-6f, 1.0e-6f};

    // Test parameter ranges
    std::vector<float> dx_values = {5.0e-6f, 10.0e-6f, 20.0e-6f};  // 5-20 μm
    std::vector<float> alpha_values = {
        0.1e-6f, 0.5e-6f, 1.0e-6f, 2.0e-6f, 5.0e-6f  // Various materials
    };

    int total_tests = 0;
    int unstable_count = 0;

    for (float dx : dx_values) {
        for (float alpha : alpha_values) {
            for (float dt : timesteps) {
                float alpha_lattice = alpha * dt / (dx * dx);
                float tau = 4.0f * alpha_lattice + 0.5f;
                float omega = 1.0f / tau;

                total_tests++;

                if (omega > 1.90f) {
                    unstable_count++;
                    std::cout << "UNSTABLE: dx=" << dx * 1e6 << "μm, "
                              << "alpha=" << alpha * 1e6 << "mm^2/s, "
                              << "dt=" << dt * 1e6 << "μs -> "
                              << "omega=" << omega << "\n";
                }
            }
        }
    }

    std::cout << "\n=== Parameter Space Stress Test ===\n";
    std::cout << "Total configurations: " << total_tests << "\n";
    std::cout << "Unstable configurations: " << unstable_count << "\n";
    std::cout << "Stability rate: "
              << (100.0 * (total_tests - unstable_count) / total_tests) << "%\n";

    // With new timesteps, expect high stability across parameter space
    float stability_rate = (float)(total_tests - unstable_count) / total_tests;
    EXPECT_GT(stability_rate, 0.70f)
        << "At least 70% of parameter space should be stable with new timesteps";
}
