/**
 * @file test_3d_heat_diffusion_fixed.cu
 * @brief Case 1: 3D Heat Diffusion Validation - FIXED VERSION
 *
 * This is the corrected version that aligns output times with LBM timesteps
 * to eliminate the staircase pattern artifact.
 *
 * KEY FIX: Output times are aligned with actual timesteps
 * - Original: outputs every 5ms, but dt = 15.11ms → staircase
 * - Fixed: outputs every dt = 15.11ms → smooth decay
 *
 * Configuration:
 * - Domain size: 9.5×9.5×9.5 mm (51×51×51 grid)
 * - Simulation time: 0.1 s
 * - Wall temperature: T_w = 273.13 K
 * - Initial heat pulse: Q = 1000 J (Gaussian at center)
 * - Material properties:
 *   - ρ = 1000 kg/m³
 *   - c_p = 4186 J/(kg·K)
 *   - k = 1 W/(m·K)
 *   - α = k/(ρ·c_p) = 2.39×10⁻⁷ m²/s
 *
 * Analytical Solution (3D Gaussian Diffusion):
 *   T(r,t) = T_w + A(t) · exp(-r²/(2σ²(t)))
 *   where σ²(t) = σ₀² + 2αt
 *
 * Success Criteria:
 *   - L2 error < 1% at all output times (relaxed from 5% due to better sampling)
 *   - Smooth temperature decay (no staircase pattern)
 */

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cmath>
#include <vector>
#include <iostream>
#include <iomanip>
#include <string>
#include "physics/thermal_lbm.h"
#include "physics/material_properties.h"
#include "io/vtk_writer.h"

using namespace lbm::physics;
using namespace lbm::io;

// ============================================================================
// CONFIGURATION (DO NOT MODIFY - MATCHES ORIGINAL TEST)
// ============================================================================

// Domain parameters
constexpr int NX = 51;
constexpr int NY = 51;
constexpr int NZ = 51;
constexpr float DOMAIN_SIZE = 9.5e-3f;      // 9.5 mm in meters

// Material properties (water-like for validation)
constexpr float RHO = 1000.0f;              // kg/m³
constexpr float CP = 4186.0f;               // J/(kg·K)
constexpr float K_THERMAL = 1.0f;           // W/(m·K)
constexpr float ALPHA = K_THERMAL / (RHO * CP);  // m²/s = 2.39e-7

// Thermal parameters
constexpr float T_WALL = 273.13f;           // K
constexpr float Q_TOTAL = 1000.0f;          // J (total heat energy)
constexpr float SIGMA_0 = 1.0e-3f;          // Initial Gaussian width: 1 mm

// Time parameters
constexpr float T_FINAL = 0.1f;             // 0.1 s

// Derived parameters
constexpr float DX = DOMAIN_SIZE / (NX - 1);  // Grid spacing
constexpr float DT = 0.1f * DX * DX / ALPHA;  // Time step (CFL-limited)

// ============================================================================
// FIX: ALIGNED OUTPUT TIMES
// ============================================================================
// Output times aligned with LBM timesteps to eliminate staircase artifacts
//
// Original problem: OUTPUT_TIMES = {0, 0.005, 0.01, 0.015, ...} every 5ms
//                   but dt = 15.11ms → multiple outputs per timestep
//
// Solution: Output every timestep or every N timesteps
// ============================================================================

constexpr int STEPS_BETWEEN_OUTPUTS = 1;  // Output every timestep
constexpr int TOTAL_STEPS = static_cast<int>(T_FINAL / DT + 0.5f);
constexpr int NUM_OUTPUTS = TOTAL_STEPS / STEPS_BETWEEN_OUTPUTS + 1;

// Generate output times aligned with timesteps
void generateOutputTimes(std::vector<float>& output_times) {
    output_times.clear();
    for (int i = 0; i < NUM_OUTPUTS; ++i) {
        float t = i * STEPS_BETWEEN_OUTPUTS * DT;
        if (t <= T_FINAL) {
            output_times.push_back(t);
        }
    }
}

// ============================================================================
// ANALYTICAL SOLUTION (SAME AS ORIGINAL)
// ============================================================================

float analyticalSolution3D(float r, float t) {
    float sigma_sq = SIGMA_0 * SIGMA_0 + 2.0f * ALPHA * t;
    float sigma = sqrtf(sigma_sq);
    float A_0 = Q_TOTAL / (RHO * CP * powf(2.0f * M_PI, 1.5f) * powf(SIGMA_0, 3.0f));
    float A_t = A_0 * powf(SIGMA_0 / sigma, 3.0f);
    float spatial_term = expf(-r * r / (2.0f * sigma_sq));
    return T_WALL + A_t * spatial_term;
}

void initializeGaussianPulse(std::vector<float>& h_temp) {
    float x_center = DOMAIN_SIZE / 2.0f;
    float y_center = DOMAIN_SIZE / 2.0f;
    float z_center = DOMAIN_SIZE / 2.0f;

    for (int k = 0; k < NZ; ++k) {
        for (int j = 0; j < NY; ++j) {
            for (int i = 0; i < NX; ++i) {
                int idx = i + NX * (j + NY * k);
                float x = i * DX;
                float y = j * DX;
                float z = k * DX;
                float dx = x - x_center;
                float dy = y - y_center;
                float dz = z - z_center;
                float r = sqrtf(dx*dx + dy*dy + dz*dz);
                h_temp[idx] = analyticalSolution3D(r, 0.0f);
            }
        }
    }
}

float computeL2Error3D(const std::vector<float>& numerical, float time) {
    float x_center = DOMAIN_SIZE / 2.0f;
    float y_center = DOMAIN_SIZE / 2.0f;
    float z_center = DOMAIN_SIZE / 2.0f;

    float sum_squared_error = 0.0f;
    float sum_squared_analytical = 0.0f;

    for (int k = 0; k < NZ; ++k) {
        for (int j = 0; j < NY; ++j) {
            for (int i = 0; i < NX; ++i) {
                int idx = i + NX * (j + NY * k);
                float x = i * DX;
                float y = j * DX;
                float z = k * DX;
                float dx = x - x_center;
                float dy = y - y_center;
                float dz = z - z_center;
                float r = sqrtf(dx*dx + dy*dy + dz*dz);

                float T_analytical = analyticalSolution3D(r, time);
                float T_numerical = numerical[idx];
                float error = T_numerical - T_analytical;

                sum_squared_error += error * error;
                sum_squared_analytical += T_analytical * T_analytical;
            }
        }
    }

    return sqrtf(sum_squared_error / sum_squared_analytical);
}

// ============================================================================
// TEST FIXTURE
// ============================================================================

class HeatDiffusion3DFixedTest : public ::testing::Test {
protected:
    void SetUp() override {
        if (!D3Q7::isInitialized()) {
            D3Q7::initializeDevice();
        }

        std::cout << "\n========================================\n";
        std::cout << "Case 1: 3D Heat Diffusion (FIXED - Aligned Outputs)\n";
        std::cout << "========================================\n\n";

        std::cout << "Domain Configuration:\n";
        std::cout << "  Grid: " << NX << "×" << NY << "×" << NZ << "\n";
        std::cout << "  Size: " << DOMAIN_SIZE * 1e3 << " mm\n";
        std::cout << "  dx = " << DX * 1e6 << " µm\n";
        std::cout << "  dt = " << DT * 1e6 << " µs (" << DT * 1e3 << " ms)\n";
        std::cout << "\nMaterial Properties:\n";
        std::cout << "  ρ = " << RHO << " kg/m³\n";
        std::cout << "  c_p = " << CP << " J/(kg·K)\n";
        std::cout << "  k = " << K_THERMAL << " W/(m·K)\n";
        std::cout << "  α = " << ALPHA << " m²/s\n";
        std::cout << "\nThermal Parameters:\n";
        std::cout << "  T_wall = " << T_WALL << " K\n";
        std::cout << "  Q_total = " << Q_TOTAL << " J\n";
        std::cout << "  t_final = " << T_FINAL << " s\n\n";

        // Generate aligned output times
        generateOutputTimes(output_times_);

        std::cout << "Output Configuration (FIXED):\n";
        std::cout << "  Total timesteps: " << TOTAL_STEPS << "\n";
        std::cout << "  Output interval: every " << STEPS_BETWEEN_OUTPUTS << " timestep(s)\n";
        std::cout << "  Number of outputs: " << output_times_.size() << "\n";
        std::cout << "  Output times: ";
        for (size_t i = 0; i < std::min(size_t(5), output_times_.size()); ++i) {
            std::cout << output_times_[i] * 1e3 << "ms";
            if (i < std::min(size_t(5), output_times_.size()) - 1) std::cout << ", ";
        }
        if (output_times_.size() > 5) std::cout << ", ...";
        std::cout << "\n\n";

        // Create material properties
        MaterialProperties material;
        material.k_solid = K_THERMAL;
        material.rho_solid = RHO;
        material.cp_solid = CP;
        material.k_liquid = K_THERMAL;
        material.rho_liquid = RHO;
        material.cp_liquid = CP;
        material.T_solidus = 1000000.0f;
        material.T_liquidus = 1000000.0f;
        material.T_vaporization = 2000000.0f;

        solver_ = new ThermalLBM(NX, NY, NZ, material, ALPHA, false, DT, DX);

        std::cout << "LBM Parameters:\n";
        std::cout << "  tau_T = " << solver_->getThermalTau() << "\n\n";

        // Initialize
        std::vector<float> h_temp(NX * NY * NZ);
        initializeGaussianPulse(h_temp);
        solver_->initialize(h_temp.data());

        system("mkdir -p /home/yzk/LBMProject/benchmark/validation_output/case1_thermal_fixed");
    }

    void TearDown() override {
        delete solver_;
    }

    void runSteps(int num_steps) {
        for (int step = 0; step < num_steps; ++step) {
            solver_->collisionBGK();
            solver_->streaming();
            solver_->computeTemperature();
            solver_->applyBoundaryConditions(1, T_WALL);
        }
    }

    ThermalLBM* solver_ = nullptr;
    std::vector<float> output_times_;
};

// ============================================================================
// TESTS
// ============================================================================

TEST_F(HeatDiffusion3DFixedTest, SmoothTemperatureDecay) {
    std::cout << "Running smooth temperature decay validation...\n\n";

    int current_step = 0;
    std::vector<float> errors;
    std::vector<float> center_temps_num;
    std::vector<float> center_temps_ana;

    int i_center = NX / 2;
    int j_center = NY / 2;
    int k_center = NZ / 2;
    int idx_center = i_center + NX * (j_center + NY * k_center);

    for (size_t i = 0; i < output_times_.size(); ++i) {
        float target_time = output_times_[i];
        int target_step = static_cast<int>(target_time / DT + 0.5f);
        int steps_to_run = target_step - current_step;

        std::cout << "Time point " << (i+1) << "/" << output_times_.size()
                  << " (t = " << target_time * 1e3 << " ms, step " << target_step << "):\n";

        if (steps_to_run > 0) {
            runSteps(steps_to_run);
            current_step = target_step;
        }

        // Get numerical solution
        std::vector<float> h_temp(NX * NY * NZ);
        solver_->copyTemperatureToHost(h_temp.data());

        // Write VTK output
        std::string filename = "/home/yzk/LBMProject/benchmark/validation_output/case1_thermal_fixed/temperature_t"
                             + std::to_string(static_cast<int>(i)).insert(0, 3 - std::min(3, static_cast<int>(std::to_string(i).length())), '0')
                             + ".vtk";
        VTKWriter::writeStructuredPoints(
            filename, h_temp.data(),
            NX, NY, NZ,
            DX, DX, DX,
            "Temperature"
        );

        // Compute error
        float l2_error = computeL2Error3D(h_temp, target_time);
        errors.push_back(l2_error);

        // Center temperature
        float T_num = h_temp[idx_center];
        float T_ana = analyticalSolution3D(0.0f, target_time);
        center_temps_num.push_back(T_num);
        center_temps_ana.push_back(T_ana);

        std::cout << "  L2 error = " << l2_error * 100.0f << "%\n";
        std::cout << "  T_center: num = " << T_num << " K, ana = " << T_ana << " K\n";
        std::cout << "  Ran " << steps_to_run << " step(s)\n\n";
    }

    // Summary
    std::cout << "========================================\n";
    std::cout << "SUMMARY\n";
    std::cout << "========================================\n\n";
    std::cout << "Time[ms]   L2 Error[%]   T_center[K]   Status\n";
    std::cout << "------------------------------------------------------\n";

    bool all_passed = true;
    for (size_t i = 0; i < output_times_.size(); ++i) {
        bool passed = errors[i] < 0.01f;  // 1% threshold
        all_passed = all_passed && passed;

        std::cout << std::fixed << std::setprecision(2) << std::setw(7) << output_times_[i] * 1e3
                  << std::setw(13) << std::setprecision(2) << errors[i] * 100.0f
                  << std::setw(15) << std::setprecision(1) << center_temps_num[i]
                  << std::setw(10) << (passed ? "PASS" : "FAIL")
                  << "\n";
    }

    std::cout << "------------------------------------------------------\n";
    std::cout << "Overall: " << (all_passed ? "PASS" : "FAIL") << "\n\n";

    // Check for smooth decay (no staircase)
    std::cout << "Checking for smooth decay (no staircase pattern)...\n";
    bool is_smooth = true;
    for (size_t i = 1; i < center_temps_num.size(); ++i) {
        float delta = center_temps_num[i] - center_temps_num[i-1];
        if (delta == 0.0f && i > 1) {
            std::cout << "  WARNING: Temperature frozen between t=" << output_times_[i-1]*1e3
                      << "ms and t=" << output_times_[i]*1e3 << "ms\n";
            is_smooth = false;
        }
    }

    if (is_smooth) {
        std::cout << "  Result: Temperature evolves smoothly at every output!\n";
    }
    std::cout << "\n";

    EXPECT_TRUE(is_smooth) << "Temperature should evolve at every output (no frozen steps)";

    for (size_t i = 0; i < errors.size(); ++i) {
        EXPECT_LT(errors[i], 0.01f)
            << "L2 error exceeds 1% at t = " << output_times_[i] << " s";
    }
}

// ============================================================================
// MAIN
// ============================================================================

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
