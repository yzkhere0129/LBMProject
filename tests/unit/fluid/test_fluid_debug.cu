/**
 * @file test_fluid_debug.cu
 * @brief Minimal debug test for FluidLBM
 */

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <iostream>
#include "physics/fluid_lbm.h"
#include "core/lattice_d3q19.h"

using namespace lbm::physics;
using namespace lbm::core;

class FluidDebugTest : public ::testing::Test {
protected:
    void SetUp() override {
        cudaSetDevice(0);
        D3Q19::initializeDevice();
    }

    void TearDown() override {
        cudaDeviceReset();
    }
};

// Minimal test: just initialization and one step
TEST_F(FluidDebugTest, SingleStepDebug) {
    const int nx = 8, ny = 8, nz = 8;  // Same as failing test
    float nu = 0.1f;
    float rho0 = 1.0f;
    float u0 = 0.05f;  // Non-zero velocity

    FluidLBM solver(nx, ny, nz, nu, rho0);

    std::cout << "Initializing with u0=" << u0 << "..." << std::endl;
    solver.initialize(rho0, u0, 0.0f, 0.0f);

    // Check initialization
    int n_cells = nx * ny * nz;
    std::vector<float> rho(n_cells), ux(n_cells);

    solver.copyDensityToHost(rho.data());
    solver.copyVelocityToHost(ux.data(), nullptr, nullptr);

    std::cout << "After init: rho[0]=" << rho[0] << ", ux[0]=" << ux[0] << std::endl;

    // First computeMacroscopic
    std::cout << "Computing macroscopic..." << std::endl;
    solver.computeMacroscopic();

    solver.copyDensityToHost(rho.data());
    solver.copyVelocityToHost(ux.data(), nullptr, nullptr);

    std::cout << "After computeMacro: rho[0]=" << rho[0] << ", ux[0]=" << ux[0] << std::endl;

    // Check for NaN
    for (int i = 0; i < n_cells; ++i) {
        ASSERT_FALSE(std::isnan(rho[i])) << "NaN in rho after computeMacroscopic at cell " << i;
        ASSERT_FALSE(std::isnan(ux[i])) << "NaN in ux after computeMacroscopic at cell " << i;
    }

    // Run 100 steps like the failing test
    const int n_steps = 100;
    for (int step = 0; step < n_steps; ++step) {
        solver.computeMacroscopic();
        solver.collisionBGK(0.0f, 0.0f, 0.0f);
        solver.streaming();

        if (step % 20 == 0) {
            solver.copyDensityToHost(rho.data());
            solver.copyVelocityToHost(ux.data(), nullptr, nullptr);
            std::cout << "Step " << step << ": rho[0]=" << rho[0] << ", ux[0]=" << ux[0] << std::endl;

            // Check for NaN during simulation
            for (int i = 0; i < n_cells; ++i) {
                if (std::isnan(rho[i]) || std::isnan(ux[i])) {
                    std::cout << "NaN detected at step " << step << ", cell " << i
                             << ": rho=" << rho[i] << ", ux=" << ux[i] << std::endl;
                    FAIL() << "NaN detected during simulation";
                }
            }
        }
    }

    solver.computeMacroscopic();
    solver.copyDensityToHost(rho.data());
    solver.copyVelocityToHost(ux.data(), nullptr, nullptr);

    std::cout << "After " << n_steps << " steps: rho[0]=" << rho[0] << ", ux[0]=" << ux[0] << std::endl;

    // Check for NaN
    for (int i = 0; i < n_cells; ++i) {
        if (std::isnan(rho[i]) || std::isnan(ux[i])) {
            std::cout << "NaN detected at cell " << i << ": rho=" << rho[i] << ", ux=" << ux[i] << std::endl;
        }
        ASSERT_FALSE(std::isnan(rho[i])) << "NaN in rho after simulation at cell " << i;
        ASSERT_FALSE(std::isnan(ux[i])) << "NaN in ux after simulation at cell " << i;
    }

    // Check that velocity is preserved
    for (int i = 0; i < n_cells; ++i) {
        EXPECT_NEAR(ux[i], u0, 5e-3f) << "Velocity not preserved at cell " << i;
    }
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
