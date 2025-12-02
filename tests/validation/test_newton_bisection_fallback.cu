/**
 * @file test_newton_bisection_fallback.cu
 * @brief Test Newton solver with bisection fallback for phase change
 *
 * This test verifies that:
 * 1. Newton-Raphson converges for normal cases
 * 2. Bisection fallback works when Newton-Raphson fails
 * 3. The solver doesn't silently fail on difficult cases
 */

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cmath>
#include <vector>
#include <iostream>

#include "physics/phase_change.h"
#include "physics/material_properties.h"

using namespace lbm::physics;

class NewtonBisectionTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Use Ti6Al4V material
        material_ = MaterialDatabase::getTi6Al4V();

        // Create solver (1x1x1 grid for unit testing)
        solver_ = new PhaseChangeSolver(1, 1, 1, material_);

        // Allocate device memory for test
        cudaMalloc(&d_temperature_, sizeof(float));
        cudaMalloc(&d_enthalpy_, sizeof(float));
    }

    void TearDown() override {
        delete solver_;
        cudaFree(d_temperature_);
        cudaFree(d_enthalpy_);
    }

    MaterialProperties material_;
    PhaseChangeSolver* solver_;
    float* d_temperature_;
    float* d_enthalpy_;
};

TEST_F(NewtonBisectionTest, SolidPhaseConvergence) {
    // Test: Solid phase (T < T_solidus) should converge quickly
    float T_test = material_.T_solidus - 100.0f;  // 100K below solidus

    // Set temperature on device
    cudaMemcpy(d_temperature_, &T_test, sizeof(float), cudaMemcpyHostToDevice);

    // Compute enthalpy from temperature
    solver_->initializeFromTemperature(d_temperature_);

    // Copy enthalpy to our buffer
    solver_->copyEnthalpyToHost(&T_test);  // Reusing variable
    cudaMemcpy(d_enthalpy_, &T_test, sizeof(float), cudaMemcpyHostToDevice);

    // Reset temperature to a bad initial guess
    float T_bad_guess = material_.T_liquidus + 100.0f;
    cudaMemcpy(d_temperature_, &T_bad_guess, sizeof(float), cudaMemcpyHostToDevice);

    // Solve for temperature from enthalpy
    int iterations = solver_->updateTemperatureFromEnthalpy(d_temperature_, 1e-6f, 50);

    // Check result
    float T_result;
    cudaMemcpy(&T_result, d_temperature_, sizeof(float), cudaMemcpyDeviceToHost);

    EXPECT_NEAR(T_result, T_test, 1.0f) << "Solid phase should converge accurately";
    EXPECT_GT(iterations, 0) << "Should require at least one iteration";
    EXPECT_LT(iterations, 50) << "Should converge before max iterations";
}

TEST_F(NewtonBisectionTest, LiquidPhaseConvergence) {
    // Test: Liquid phase (T > T_liquidus) should converge
    float T_test = material_.T_liquidus + 100.0f;  // 100K above liquidus

    cudaMemcpy(d_temperature_, &T_test, sizeof(float), cudaMemcpyHostToDevice);
    solver_->initializeFromTemperature(d_temperature_);

    float H_test;
    solver_->copyEnthalpyToHost(&H_test);
    cudaMemcpy(d_enthalpy_, &H_test, sizeof(float), cudaMemcpyHostToDevice);

    // Bad initial guess
    float T_bad_guess = material_.T_solidus - 100.0f;
    cudaMemcpy(d_temperature_, &T_bad_guess, sizeof(float), cudaMemcpyHostToDevice);

    int iterations = solver_->updateTemperatureFromEnthalpy(d_temperature_, 1e-6f, 50);

    float T_result;
    cudaMemcpy(&T_result, d_temperature_, sizeof(float), cudaMemcpyDeviceToHost);

    EXPECT_NEAR(T_result, T_test, 1.0f) << "Liquid phase should converge accurately";
    EXPECT_LT(iterations, 50) << "Should converge before max iterations";
}

TEST_F(NewtonBisectionTest, MushyZoneConvergence) {
    // Test: Mushy zone is the hardest case (non-smooth derivative)
    float T_test = (material_.T_solidus + material_.T_liquidus) / 2.0f;  // Middle of mushy zone

    cudaMemcpy(d_temperature_, &T_test, sizeof(float), cudaMemcpyHostToDevice);
    solver_->initializeFromTemperature(d_temperature_);

    float H_test;
    solver_->copyEnthalpyToHost(&H_test);
    cudaMemcpy(d_enthalpy_, &H_test, sizeof(float), cudaMemcpyHostToDevice);

    // Very bad initial guess (opposite phase)
    float T_bad_guess = material_.T_solidus - 200.0f;
    cudaMemcpy(d_temperature_, &T_bad_guess, sizeof(float), cudaMemcpyHostToDevice);

    int iterations = solver_->updateTemperatureFromEnthalpy(d_temperature_, 1e-6f, 50);

    float T_result;
    cudaMemcpy(&T_result, d_temperature_, sizeof(float), cudaMemcpyDeviceToHost);

    // CRITICAL: This should converge even with bad initial guess
    // If Newton-Raphson fails, bisection fallback should work
    EXPECT_NEAR(T_result, T_test, 2.0f) << "Mushy zone should converge with bisection fallback";
    EXPECT_FALSE(std::isnan(T_result)) << "Result should not be NaN";
    EXPECT_FALSE(std::isinf(T_result)) << "Result should not be Inf";
}

TEST_F(NewtonBisectionTest, ExtremeCaseRobustness) {
    // Test: Extreme enthalpy values should not cause failure
    float T_test = material_.T_liquidus + 500.0f;  // Very hot

    cudaMemcpy(d_temperature_, &T_test, sizeof(float), cudaMemcpyHostToDevice);
    solver_->initializeFromTemperature(d_temperature_);

    float H_test;
    solver_->copyEnthalpyToHost(&H_test);
    cudaMemcpy(d_enthalpy_, &H_test, sizeof(float), cudaMemcpyHostToDevice);

    // Extremely bad initial guess
    float T_bad_guess = 100.0f;  // Very cold
    cudaMemcpy(d_temperature_, &T_bad_guess, sizeof(float), cudaMemcpyHostToDevice);

    int iterations = solver_->updateTemperatureFromEnthalpy(d_temperature_, 1e-6f, 50);

    float T_result;
    cudaMemcpy(&T_result, d_temperature_, sizeof(float), cudaMemcpyDeviceToHost);

    // Should converge to reasonable value (within 10K is acceptable for extreme case)
    EXPECT_NEAR(T_result, T_test, 10.0f) << "Extreme case should converge";
    EXPECT_FALSE(std::isnan(T_result)) << "Should not return NaN";
}

TEST_F(NewtonBisectionTest, MultiplePhaseTransitions) {
    // Test: Cycling through solid -> mushy -> liquid -> mushy -> solid
    std::vector<float> test_temps = {
        material_.T_solidus - 100.0f,  // Solid
        material_.T_solidus + 10.0f,   // Mushy (near solidus)
        material_.T_liquidus - 10.0f,  // Mushy (near liquidus)
        material_.T_liquidus + 100.0f, // Liquid
        material_.T_liquidus - 10.0f,  // Back to mushy
        material_.T_solidus - 50.0f    // Back to solid
    };

    for (size_t i = 0; i < test_temps.size(); ++i) {
        float T_test = test_temps[i];

        // Compute enthalpy
        cudaMemcpy(d_temperature_, &T_test, sizeof(float), cudaMemcpyHostToDevice);
        solver_->updateEnthalpyFromTemperature(d_temperature_);

        // Get enthalpy
        float H_test;
        solver_->copyEnthalpyToHost(&H_test);

        // Reset temperature to previous value (simulates time evolution)
        float T_prev = (i > 0) ? test_temps[i-1] : material_.T_solidus;
        cudaMemcpy(d_temperature_, &T_prev, sizeof(float), cudaMemcpyHostToDevice);

        // Solve
        int iterations = solver_->updateTemperatureFromEnthalpy(d_temperature_, 1e-6f, 50);

        float T_result;
        cudaMemcpy(&T_result, d_temperature_, sizeof(float), cudaMemcpyDeviceToHost);

        EXPECT_NEAR(T_result, T_test, 2.0f)
            << "Phase transition " << i << " should converge";
        EXPECT_LT(iterations, 50)
            << "Phase transition " << i << " should converge before max iterations";
    }
}

TEST_F(NewtonBisectionTest, NumericalStability) {
    // Test: Small perturbations should not cause instability
    float T_base = material_.T_liquidus;

    for (int i = -10; i <= 10; ++i) {
        float T_test = T_base + i * 0.1f;  // 0.1K perturbations

        cudaMemcpy(d_temperature_, &T_test, sizeof(float), cudaMemcpyHostToDevice);
        solver_->updateEnthalpyFromTemperature(d_temperature_);

        // Use nearby temperature as initial guess
        float T_guess = T_test + 5.0f;
        cudaMemcpy(d_temperature_, &T_guess, sizeof(float), cudaMemcpyHostToDevice);

        solver_->updateTemperatureFromEnthalpy(d_temperature_, 1e-6f, 50);

        float T_result;
        cudaMemcpy(&T_result, d_temperature_, sizeof(float), cudaMemcpyDeviceToHost);

        EXPECT_NEAR(T_result, T_test, 0.5f)
            << "Small perturbation i=" << i << " should be stable";
        EXPECT_FALSE(std::isnan(T_result)) << "Should not produce NaN";
    }
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
