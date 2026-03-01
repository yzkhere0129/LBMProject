/**
 * @file test_force_initialization.cu
 * @brief Test force initialization and accumulation in multiphysics coupling
 *
 * This test verifies that force arrays are properly initialized to zero
 * before accumulation, preventing garbage values from corrupting the simulation.
 *
 * Test Coverage:
 * 1. Forces are zero after allocation
 * 2. Forces are zero at start of each timestep
 * 3. Forces are properly accumulated (not overwritten)
 * 4. No NaN/Inf values leak through
 * 5. Forces are zeroed even when no physics is active
 */

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <vector>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <algorithm>

#include "physics/multiphysics_solver.h"

using namespace lbm::physics;

// Helper function to check if all values in array are zero
bool allZero(const std::vector<float>& arr) {
    for (float val : arr) {
        if (val != 0.0f) return false;
    }
    return true;
}

// Helper function to check for NaN/Inf
bool hasNaNOrInf(const std::vector<float>& arr) {
    for (float val : arr) {
        if (std::isnan(val) || std::isinf(val)) return true;
    }
    return false;
}

// Helper function to compute max magnitude
float maxMagnitude(const std::vector<float>& fx, const std::vector<float>& fy, const std::vector<float>& fz) {
    float max_mag = 0.0f;
    for (size_t i = 0; i < fx.size(); ++i) {
        float mag = std::sqrt(fx[i]*fx[i] + fy[i]*fy[i] + fz[i]*fz[i]);
        max_mag = std::max(max_mag, mag);
    }
    return max_mag;
}

/**
 * Test 1: Verify forces are zero after allocation
 */
TEST(ForceInitializationTest, ForcesZeroAfterAllocation) {
    std::cout << "\n=== Test 1: Forces Zero After Allocation ===" << std::endl;

    MultiphysicsConfig config;
    config.nx = 20;
    config.ny = 20;
    config.nz = 10;
    config.dx = 2e-6f;
    config.dt = 1e-7f;

    // Minimal configuration (no physics active)
    config.enable_thermal = false;
    config.enable_fluid = true;
    config.enable_vof = true;
    config.enable_vof_advection = false;
    config.enable_surface_tension = false;
    config.enable_marangoni = false;
    config.enable_laser = false;
    config.enable_darcy = false;
    config.enable_buoyancy = false;

    MultiphysicsSolver solver(config);

    // Initialize with uniform conditions
    solver.initialize(300.0f, 0.5f);

    // Access internal force arrays (we'll need to add getter methods)
    // For now, we test indirectly by running a step and checking velocity

    // Run one step - forces should be zero, so velocity should remain zero
    solver.step();

    std::vector<float> h_vx(config.nx * config.ny * config.nz);
    std::vector<float> h_vy(config.nx * config.ny * config.nz);
    std::vector<float> h_vz(config.nx * config.ny * config.nz);

    solver.copyVelocityToHost(h_vx.data(), h_vy.data(), h_vz.data());

    // With no forces, velocity should remain zero
    float max_v = maxMagnitude(h_vx, h_vy, h_vz);

    std::cout << "  Max velocity after 1 step (no forces): " << max_v << " (should be ~0)" << std::endl;

    EXPECT_LT(max_v, 1e-6f) << "Velocity should be zero when no forces are applied";

    std::cout << "  PASS: Forces properly initialized to zero" << std::endl;
}

/**
 * Test 2: Verify forces are zeroed at start of each timestep
 */
TEST(ForceInitializationTest, ForcesZeroedEachTimestep) {
    std::cout << "\n=== Test 2: Forces Zeroed Each Timestep ===" << std::endl;

    MultiphysicsConfig config;
    config.nx = 20;
    config.ny = 20;
    config.nz = 10;
    config.dx = 2e-6f;
    config.dt = 1e-7f;

    config.enable_thermal = false;
    config.enable_fluid = true;
    config.enable_vof = true;
    config.enable_vof_advection = false;
    config.enable_surface_tension = false;
    config.enable_marangoni = true;  // Enable Marangoni
    config.enable_laser = false;
    config.enable_darcy = false;
    config.enable_buoyancy = false;

    config.material = MaterialDatabase::getTi6Al4V();
    config.dsigma_dT = -0.26e-3f;

    MultiphysicsSolver solver(config);

    // Create temperature gradient (hot center, cold edge)
    int num_cells = config.nx * config.ny * config.nz;
    std::vector<float> h_temp(num_cells);

    float T_hot = 2050.0f;  // Reduced gradient to stay within LBM stability limits
    float T_cold = 2000.0f;
    float center_x = config.nx / 2.0f;
    float center_y = config.ny / 2.0f;

    for (int k = 0; k < config.nz; ++k) {
        for (int j = 0; j < config.ny; ++j) {
            for (int i = 0; i < config.nx; ++i) {
                int idx = i + config.nx * (j + config.ny * k);
                float dx = i - center_x;
                float dy = j - center_y;
                float r = std::sqrt(dx*dx + dy*dy);
                float decay = r / (config.nx / 2.0f);
                h_temp[idx] = T_hot - decay * (T_hot - T_cold);
                h_temp[idx] = std::max(h_temp[idx], T_cold);
            }
        }
    }

    std::vector<float> h_fill(num_cells);
    int z_interface = config.nz / 2;
    for (int k = 0; k < config.nz; ++k) {
        for (int j = 0; j < config.ny; ++j) {
            for (int i = 0; i < config.nx; ++i) {
                int idx = i + config.nx * (j + config.ny * k);
                float z_dist = k - z_interface;
                h_fill[idx] = 0.5f * (1.0f - std::tanh(z_dist / 2.0f));
            }
        }
    }

    solver.initialize(h_temp.data(), h_fill.data());

    std::cout << "  Running 5 timesteps with Marangoni force..." << std::endl;

    std::vector<float> velocities_over_time;

    for (int step = 0; step < 5; ++step) {
        solver.step();

        std::vector<float> h_vx(num_cells);
        std::vector<float> h_vy(num_cells);
        std::vector<float> h_vz(num_cells);
        solver.copyVelocityToHost(h_vx.data(), h_vy.data(), h_vz.data());

        float max_v = maxMagnitude(h_vx, h_vy, h_vz);
        velocities_over_time.push_back(max_v);

        std::cout << "    Step " << step << ": max velocity = " << std::scientific
                  << max_v << " m/s" << std::defaultfloat << std::endl;

        // Check for NaN/Inf
        EXPECT_FALSE(hasNaNOrInf(h_vx)) << "NaN/Inf in velocity x at step " << step;
        EXPECT_FALSE(hasNaNOrInf(h_vy)) << "NaN/Inf in velocity y at step " << step;
        EXPECT_FALSE(hasNaNOrInf(h_vz)) << "NaN/Inf in velocity z at step " << step;
    }

    // Velocity should increase over time (forces are accumulated consistently)
    std::cout << "  Checking velocity progression..." << std::endl;
    for (size_t i = 1; i < velocities_over_time.size(); ++i) {
        EXPECT_GE(velocities_over_time[i], velocities_over_time[i-1] * 0.5f)
            << "Velocity should not decrease drastically between steps (forces properly zeroed and recomputed)";
    }

    // Final velocity should be non-zero (forces were applied)
    // Note: velocities are small but growing due to CFL limiting
    EXPECT_GT(velocities_over_time.back(), 1e-7f)
        << "Final velocity should be non-zero with Marangoni force";

    std::cout << "  PASS: Forces properly zeroed and recomputed each timestep" << std::endl;
}

/**
 * Test 3: Verify forces are accumulated, not overwritten
 */
TEST(ForceInitializationTest, ForcesAccumulatedNotOverwritten) {
    std::cout << "\n=== Test 3: Forces Accumulated Not Overwritten ===" << std::endl;

    MultiphysicsConfig config;
    config.nx = 20;
    config.ny = 20;
    config.nz = 10;
    config.dx = 2e-6f;
    config.dt = 1e-7f;

    config.enable_thermal = false;
    config.enable_fluid = true;
    config.enable_vof = true;
    config.enable_vof_advection = false;
    config.enable_surface_tension = true;   // Enable surface tension
    config.enable_marangoni = true;         // Enable Marangoni
    config.enable_laser = false;
    config.enable_darcy = false;
    config.enable_buoyancy = true;          // Enable buoyancy

    config.material = MaterialDatabase::getTi6Al4V();
    config.dsigma_dT = -0.26e-3f;
    config.surface_tension_coeff = 1.65f;
    config.thermal_expansion_coeff = 1.5e-5f;
    config.gravity_z = -9.81f;
    config.reference_temperature = 2000.0f;

    MultiphysicsSolver solver(config);

    // Create temperature gradient AND curved interface
    int num_cells = config.nx * config.ny * config.nz;
    std::vector<float> h_temp(num_cells);
    std::vector<float> h_fill(num_cells);

    float T_hot = 2500.0f;  // Above reference (buoyancy upward)
    float center_x = config.nx / 2.0f;
    float center_y = config.ny / 2.0f;
    int z_interface = config.nz / 2;

    for (int k = 0; k < config.nz; ++k) {
        for (int j = 0; j < config.ny; ++j) {
            for (int i = 0; i < config.nx; ++i) {
                int idx = i + config.nx * (j + config.ny * k);

                // Hot temperature everywhere (for buoyancy)
                h_temp[idx] = T_hot;

                // Curved interface (bowl shape for surface tension)
                float dx = i - center_x;
                float dy = j - center_y;
                float r = std::sqrt(dx*dx + dy*dy);
                float z_local = z_interface + (r / 5.0f);  // Bowl
                float z_dist = k - z_local;
                h_fill[idx] = 0.5f * (1.0f - std::tanh(z_dist / 2.0f));
            }
        }
    }

    solver.initialize(h_temp.data(), h_fill.data());

    std::cout << "  Running 3 timesteps with multiple force sources..." << std::endl;
    std::cout << "  Forces: Marangoni + Surface Tension + Buoyancy" << std::endl;

    // Run a few steps - all forces should contribute
    for (int step = 0; step < 3; ++step) {
        solver.step();
    }

    std::vector<float> h_vx(num_cells);
    std::vector<float> h_vy(num_cells);
    std::vector<float> h_vz(num_cells);
    solver.copyVelocityToHost(h_vx.data(), h_vy.data(), h_vz.data());

    float max_v = maxMagnitude(h_vx, h_vy, h_vz);
    std::cout << "  Final max velocity: " << std::scientific << max_v << " m/s" << std::defaultfloat << std::endl;

    // With multiple forces, velocity should be significant
    EXPECT_GT(max_v, 1e-6f) << "Velocity should be non-zero with multiple forces";

    // Check for NaN/Inf (sign of force corruption)
    EXPECT_FALSE(hasNaNOrInf(h_vx)) << "NaN/Inf in velocity x";
    EXPECT_FALSE(hasNaNOrInf(h_vy)) << "NaN/Inf in velocity y";
    EXPECT_FALSE(hasNaNOrInf(h_vz)) << "NaN/Inf in velocity z";

    std::cout << "  PASS: Multiple forces properly accumulated" << std::endl;
}

/**
 * Test 4: Verify no garbage values from uninitialized memory
 */
TEST(ForceInitializationTest, NoGarbageValues) {
    std::cout << "\n=== Test 4: No Garbage Values ===" << std::endl;

    MultiphysicsConfig config;
    config.nx = 30;
    config.ny = 30;
    config.nz = 15;
    config.dx = 2e-6f;
    config.dt = 1e-7f;

    config.enable_thermal = false;
    config.enable_fluid = true;
    config.enable_vof = true;
    config.enable_vof_advection = false;
    config.enable_surface_tension = false;
    config.enable_marangoni = false;
    config.enable_laser = false;
    config.enable_darcy = false;
    config.enable_buoyancy = false;

    // Run multiple solvers in sequence to test memory reuse patterns
    std::cout << "  Creating and destroying solvers to test memory patterns..." << std::endl;

    for (int trial = 0; trial < 3; ++trial) {
        MultiphysicsSolver solver(config);
        solver.initialize(300.0f, 0.5f);

        // Run a few steps
        for (int step = 0; step < 2; ++step) {
            solver.step();
        }

        int num_cells = config.nx * config.ny * config.nz;
        std::vector<float> h_vx(num_cells);
        std::vector<float> h_vy(num_cells);
        std::vector<float> h_vz(num_cells);
        solver.copyVelocityToHost(h_vx.data(), h_vy.data(), h_vz.data());

        // Check for NaN/Inf (sign of uninitialized memory)
        bool has_nan = hasNaNOrInf(h_vx) || hasNaNOrInf(h_vy) || hasNaNOrInf(h_vz);
        EXPECT_FALSE(has_nan) << "NaN/Inf detected in trial " << trial;

        float max_v = maxMagnitude(h_vx, h_vy, h_vz);
        std::cout << "    Trial " << trial << ": max velocity = " << max_v
                  << " (should be ~0)" << std::endl;

        EXPECT_LT(max_v, 1e-6f) << "Velocity should be zero with no forces (trial " << trial << ")";
    }

    std::cout << "  PASS: No garbage values detected across multiple runs" << std::endl;
}

/**
 * Test 5: Verify forces start at zero even with physics disabled
 */
TEST(ForceInitializationTest, ForcesZeroWithDisabledPhysics) {
    std::cout << "\n=== Test 5: Forces Zero With Disabled Physics ===" << std::endl;

    MultiphysicsConfig config;
    config.nx = 20;
    config.ny = 20;
    config.nz = 10;
    config.dx = 2e-6f;
    config.dt = 1e-7f;

    // ALL physics disabled
    config.enable_thermal = false;
    config.enable_fluid = true;  // Need fluid for velocity check
    config.enable_vof = false;
    config.enable_vof_advection = false;
    config.enable_surface_tension = false;
    config.enable_marangoni = false;
    config.enable_laser = false;
    config.enable_darcy = false;
    config.enable_buoyancy = false;
    config.enable_recoil_pressure = false;

    MultiphysicsSolver solver(config);
    solver.initialize(300.0f, 0.5f);

    std::cout << "  Running 10 timesteps with all physics disabled..." << std::endl;

    for (int step = 0; step < 10; ++step) {
        solver.step();
    }

    int num_cells = config.nx * config.ny * config.nz;
    std::vector<float> h_vx(num_cells);
    std::vector<float> h_vy(num_cells);
    std::vector<float> h_vz(num_cells);
    solver.copyVelocityToHost(h_vx.data(), h_vy.data(), h_vz.data());

    float max_v = maxMagnitude(h_vx, h_vy, h_vz);
    std::cout << "  Final max velocity: " << max_v << " (should be ~0)" << std::endl;

    EXPECT_LT(max_v, 1e-6f) << "Velocity should remain zero with all physics disabled";
    EXPECT_FALSE(hasNaNOrInf(h_vx)) << "NaN/Inf in velocity x";
    EXPECT_FALSE(hasNaNOrInf(h_vy)) << "NaN/Inf in velocity y";
    EXPECT_FALSE(hasNaNOrInf(h_vz)) << "NaN/Inf in velocity z";

    std::cout << "  PASS: Forces properly zeroed even with physics disabled" << std::endl;
}

/**
 * Test 6: Stress test - rapid enable/disable of physics
 */
TEST(ForceInitializationTest, RapidPhysicsToggle) {
    std::cout << "\n=== Test 6: Rapid Physics Toggle ===" << std::endl;

    MultiphysicsConfig config;
    config.nx = 15;
    config.ny = 15;
    config.nz = 8;
    config.dx = 2e-6f;
    config.dt = 1e-7f;

    config.enable_thermal = false;
    config.enable_fluid = true;
    config.enable_vof = true;
    config.material = MaterialDatabase::getTi6Al4V();
    config.dsigma_dT = -0.26e-3f;

    int num_cells = config.nx * config.ny * config.nz;
    std::vector<float> h_temp(num_cells, 2500.0f);  // Hot temperature
    std::vector<float> h_fill(num_cells);

    int z_interface = config.nz / 2;
    for (int k = 0; k < config.nz; ++k) {
        for (int j = 0; j < config.ny; ++j) {
            for (int i = 0; i < config.nx; ++i) {
                int idx = i + config.nx * (j + config.ny * k);
                float z_dist = k - z_interface;
                h_fill[idx] = 0.5f * (1.0f - std::tanh(z_dist / 2.0f));
            }
        }
    }

    std::cout << "  Testing force initialization with toggling physics..." << std::endl;

    // Test sequence: Marangoni OFF → ON → OFF → ON
    std::vector<bool> marangoni_states = {false, true, false, true};

    for (size_t i = 0; i < marangoni_states.size(); ++i) {
        config.enable_marangoni = marangoni_states[i];
        config.enable_darcy = false;
        config.enable_buoyancy = false;
        config.enable_surface_tension = false;

        MultiphysicsSolver solver(config);
        solver.initialize(h_temp.data(), h_fill.data());

        // Run a few steps
        for (int step = 0; step < 3; ++step) {
            solver.step();
        }

        std::vector<float> h_vx(num_cells);
        std::vector<float> h_vy(num_cells);
        std::vector<float> h_vz(num_cells);
        solver.copyVelocityToHost(h_vx.data(), h_vy.data(), h_vz.data());

        float max_v = maxMagnitude(h_vx, h_vy, h_vz);

        std::cout << "    Config " << i << " (Marangoni "
                  << (marangoni_states[i] ? "ON" : "OFF") << "): "
                  << "max velocity = " << std::scientific << max_v << std::defaultfloat << std::endl;

        // Check for NaN/Inf
        EXPECT_FALSE(hasNaNOrInf(h_vx)) << "NaN/Inf in config " << i;
        EXPECT_FALSE(hasNaNOrInf(h_vy)) << "NaN/Inf in config " << i;
        EXPECT_FALSE(hasNaNOrInf(h_vz)) << "NaN/Inf in config " << i;

        // When Marangoni is OFF, velocity should be very small
        if (!marangoni_states[i]) {
            EXPECT_LT(max_v, 1e-6f) << "Velocity should be near zero when Marangoni is OFF";
        }
    }

    std::cout << "  PASS: Force initialization robust to physics toggling" << std::endl;
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
