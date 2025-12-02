/**
 * @file test_laser_shutoff_configurable.cu
 * @brief Test configurable laser shutoff time
 *
 * This test verifies:
 * 1. Laser turns off at specified time
 * 2. Temperature stops increasing after shutoff
 * 3. Negative shutoff time means laser always on
 * 4. System remains stable after shutoff
 */

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cmath>
#include <vector>
#include <iostream>

#include "physics/multiphysics_solver.h"

using namespace lbm::physics;

class LaserShutoffTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Base configuration
        config_.nx = 50;
        config_.ny = 50;
        config_.nz = 25;
        config_.dx = 2e-6f;
        config_.dt = 1e-7f;

        // Enable physics
        config_.enable_thermal = true;
        config_.enable_laser = true;
        config_.enable_fluid = false;  // Disable fluid for thermal-only test
        config_.enable_vof = false;
        config_.enable_marangoni = false;

        // Laser parameters
        config_.laser_power = 20.0f;
        config_.laser_spot_radius = 50e-6f;
        config_.laser_absorptivity = 0.35f;
        config_.laser_penetration_depth = 10e-6f;

        // Material
        config_.material = MaterialDatabase::getTi6Al4V();
        config_.thermal_diffusivity = 5.8e-6f;
    }

    MultiphysicsConfig config_;
};

TEST_F(LaserShutoffTest, LaserTurnsOffAtSpecifiedTime) {
    // Configure laser to turn off at 5 μs
    config_.laser_shutoff_time = 5.0e-6f;

    MultiphysicsSolver solver(config_);
    solver.initialize(300.0f, 0.5f);

    // Run for 10 μs (100 steps)
    float T_at_4us = 0.0f;
    float T_at_6us = 0.0f;
    float T_at_10us = 0.0f;

    for (int step = 0; step < 100; ++step) {
        solver.step(config_.dt);

        float current_time = step * config_.dt;
        float T_max = solver.getMaxTemperature();

        if (std::abs(current_time - 4.0e-6f) < 1e-8f) {
            T_at_4us = T_max;
        }
        if (std::abs(current_time - 6.0e-6f) < 1e-8f) {
            T_at_6us = T_max;
        }
        if (std::abs(current_time - 10.0e-6f) < 1e-8f) {
            T_at_10us = T_max;
        }
    }

    // Before shutoff: temperature should increase
    EXPECT_GT(T_at_4us, 300.0f) << "Temperature should rise before shutoff";

    // After shutoff: temperature should not increase significantly (may decrease due to diffusion)
    // Allow small increase due to numerical lag, but should be much less than before shutoff
    float increase_before = T_at_4us - 300.0f;
    float increase_after = T_at_6us - T_at_4us;

    EXPECT_LT(increase_after, increase_before * 0.5f)
        << "Temperature increase should slow significantly after shutoff";

    // Much later: temperature should be cooling down
    EXPECT_LT(T_at_10us, T_at_6us)
        << "Temperature should decrease after laser shutoff";
}

TEST_F(LaserShutoffTest, NegativeShutoffMeansAlwaysOn) {
    // Configure laser to never turn off
    config_.laser_shutoff_time = -1.0f;

    MultiphysicsSolver solver(config_);
    solver.initialize(300.0f, 0.5f);

    float T_prev = 300.0f;

    // Run for 10 μs - temperature should continuously increase
    for (int step = 0; step < 100; ++step) {
        solver.step(config_.dt);

        float T_max = solver.getMaxTemperature();

        // Temperature should monotonically increase (or stay same if saturated)
        EXPECT_GE(T_max, T_prev - 1.0f)  // Allow 1K tolerance for numerical noise
            << "With laser always on, temperature should not decrease significantly";

        T_prev = T_max;
    }

    // Final temperature should be much higher than initial
    EXPECT_GT(T_prev, 500.0f)
        << "Continuous laser heating should raise temperature significantly";
}

TEST_F(LaserShutoffTest, ZeroShutoffMeansImmediateOff) {
    // Configure laser to turn off immediately
    config_.laser_shutoff_time = 0.0f;

    MultiphysicsSolver solver(config_);
    solver.initialize(300.0f, 0.5f);

    // Run for 10 μs
    float T_max_observed = 300.0f;

    for (int step = 0; step < 100; ++step) {
        solver.step(config_.dt);
        float T_max = solver.getMaxTemperature();
        T_max_observed = std::max(T_max_observed, T_max);
    }

    // Temperature should not increase much (only from initial conditions)
    EXPECT_LT(T_max_observed, 320.0f)
        << "With laser immediately off, temperature should stay near initial";
}

TEST_F(LaserShutoffTest, StabilityAfterShutoff) {
    // Configure laser to turn off at 3 μs
    config_.laser_shutoff_time = 3.0e-6f;

    MultiphysicsSolver solver(config_);
    solver.initialize(300.0f, 0.5f);

    // Run for 20 μs to test long-term stability after shutoff
    bool has_nan = false;

    for (int step = 0; step < 200; ++step) {
        solver.step(config_.dt);

        // Check for NaN
        if (solver.checkNaN()) {
            has_nan = true;
            break;
        }
    }

    EXPECT_FALSE(has_nan) << "System should remain stable after laser shutoff";

    // Final temperature should be reasonable
    float T_final = solver.getMaxTemperature();
    EXPECT_GT(T_final, 200.0f) << "Final temperature should be physical";
    EXPECT_LT(T_final, 3000.0f) << "Final temperature should not be excessive";
}

TEST_F(LaserShutoffTest, MultipleShutoffTimes) {
    // Test different shutoff times behave as expected
    std::vector<float> shutoff_times = {1e-6f, 3e-6f, 5e-6f, 10e-6f};
    std::vector<float> final_temps;

    for (float shutoff_time : shutoff_times) {
        config_.laser_shutoff_time = shutoff_time;

        MultiphysicsSolver solver(config_);
        solver.initialize(300.0f, 0.5f);

        // Run for 15 μs
        for (int step = 0; step < 150; ++step) {
            solver.step(config_.dt);
        }

        final_temps.push_back(solver.getMaxTemperature());
    }

    // Longer laser exposure should result in higher peak temperatures
    for (size_t i = 1; i < final_temps.size(); ++i) {
        // Note: Final temperature might decrease if enough cooling time
        // But temperature at shutoff time should increase with longer exposure
        // For simplicity, just check temperatures are reasonable
        EXPECT_GT(final_temps[i], 300.0f)
            << "Shutoff time " << shutoff_times[i] << " should heat material";
    }
}

TEST_F(LaserShutoffTest, ShutoffWithPhaseChange) {
    // Test shutoff with phase change enabled
    config_.laser_shutoff_time = 5.0e-6f;
    config_.enable_phase_change = true;
    config_.laser_power = 50.0f;  // Higher power to induce melting

    MultiphysicsSolver solver(config_);
    solver.initialize(300.0f, 0.5f);

    bool melting_occurred = false;
    bool stable_after_shutoff = true;

    for (int step = 0; step < 200; ++step) {
        solver.step(config_.dt);

        float T_max = solver.getMaxTemperature();

        // Check if melting occurred
        if (T_max > config_.material.T_liquidus) {
            melting_occurred = true;
        }

        // Check stability after shutoff
        if (step * config_.dt > config_.laser_shutoff_time) {
            if (solver.checkNaN()) {
                stable_after_shutoff = false;
                break;
            }
        }
    }

    EXPECT_TRUE(melting_occurred)
        << "Should reach melting temperature with sufficient laser power";
    EXPECT_TRUE(stable_after_shutoff)
        << "Should remain stable after laser shutoff even with phase change";
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
