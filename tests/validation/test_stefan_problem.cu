/**
 * @file test_stefan_problem.cu
 * @brief Benchmark 2: Stefan problem (1D melting) with analytical solution
 *
 * Validates phase change implementation against the classical Stefan problem
 * for one-dimensional melting with a moving boundary.
 *
 * Problem Setup:
 *   - Semi-infinite solid initially at T = T_solidus
 *   - Boundary condition at x=0: T = T_liquidus (constant)
 *   - Track melting front position s(t)
 *
 * Analytical Solution (Stefan problem):
 *   Melting front position: s(t) = 2λ√(αt)
 *
 *   where λ is the solution to the transcendental equation:
 *   λ·exp(λ²)·erf(λ) = St / √π
 *
 *   Stefan number: St = cp·ΔT / L_f
 *   ΔT = T_liquidus - T_solidus
 *
 * For Ti6Al4V:
 *   - ΔT = T_liquidus - T_solidus ≈ 100K (mushy zone width)
 *   - cp ≈ 546 J/(kg·K)
 *   - L_f ≈ 286000 J/kg
 *   - St ≈ 0.191
 *   - λ ≈ 0.139 (numerical solution)
 *
 * Success Criteria:
 *   - Front position error < 5% compared to analytical solution
 */

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cmath>
#include <vector>
#include <iostream>
#include <iomanip>
#include "physics/thermal_lbm.h"
#include "physics/material_properties.h"

using namespace lbm::physics;

// Physical parameters
constexpr int NX = 500;          // 1D grid points
constexpr int NY = 1;
constexpr int NZ = 1;

// Ti6Al4V properties
MaterialProperties material = MaterialDatabase::getTi6Al4V();

// Domain parameters
constexpr float DOMAIN_LENGTH = 2000.0e-6f;  // 2000 microns (2 mm)
constexpr float DX = DOMAIN_LENGTH / (NX - 1);

// Stefan number calculation
float computeStefanNumber() {
    float dT = material.T_liquidus - material.T_solidus;
    float St = material.cp_solid * dT / material.L_fusion;
    return St;
}

// Compute λ from Stefan number using Newton-Raphson
// λ·exp(λ²)·erf(λ) = St / √π
float computeLambda(float St) {
    const float sqrt_pi = sqrtf(M_PI);
    float target = St / sqrt_pi;

    // Newton-Raphson iteration
    float lambda = 0.15f;  // Initial guess
    const int max_iter = 50;
    const float tolerance = 1e-6f;

    for (int iter = 0; iter < max_iter; ++iter) {
        float exp_term = expf(lambda * lambda);
        float erf_term = erf(lambda);

        // f(λ) = λ·exp(λ²)·erf(λ) - St/√π
        float f = lambda * exp_term * erf_term - target;

        // f'(λ) = exp(λ²)·erf(λ) + λ·exp(λ²)·(2λ·erf(λ) + 2/√π)
        float df = exp_term * erf_term +
                   lambda * exp_term * (2.0f * lambda * erf_term + 2.0f / sqrt_pi);

        float lambda_new = lambda - f / df;

        if (fabsf(lambda_new - lambda) < tolerance) {
            return lambda_new;
        }

        lambda = lambda_new;
    }

    return lambda;
}

// Analytical front position
float analyticalFrontPosition(float t, float lambda, float alpha) {
    return 2.0f * lambda * sqrtf(alpha * t);
}

/**
 * @brief Find melting front position in numerical solution
 * @param h_temp Temperature field
 * @param threshold Liquid fraction threshold (0.5 = interface center)
 */
float findMeltingFront(const std::vector<float>& h_temp,
                       const std::vector<float>& h_fl) {
    // Find position where liquid fraction = 0.5
    for (int i = 1; i < NX; ++i) {
        if (h_fl[i-1] >= 0.5f && h_fl[i] < 0.5f) {
            // Linear interpolation
            float x0 = (i - 1) * DX;
            float x1 = i * DX;
            float fl0 = h_fl[i-1];
            float fl1 = h_fl[i];

            float x_interface = x0 + (0.5f - fl0) / (fl1 - fl0) * (x1 - x0);
            return x_interface;
        }
    }

    // If not found, return domain length
    return DOMAIN_LENGTH;
}

class StefanProblemTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Initialize D3Q7 lattice
        if (!D3Q7::isInitialized()) {
            D3Q7::initializeDevice();
        }

        // Get Ti6Al4V material properties
        material = MaterialDatabase::getTi6Al4V();

        // Compute Stefan number and lambda
        St = computeStefanNumber();
        lambda = computeLambda(St);
        alpha = material.getThermalDiffusivity(material.T_solidus);

        std::cout << "\nStefan Problem Parameters:" << std::endl;
        std::cout << "  T_solidus = " << material.T_solidus << " K" << std::endl;
        std::cout << "  T_liquidus = " << material.T_liquidus << " K" << std::endl;
        std::cout << "  ΔT = " << material.T_liquidus - material.T_solidus << " K" << std::endl;
        std::cout << "  cp = " << material.cp_solid << " J/(kg·K)" << std::endl;
        std::cout << "  L_f = " << material.L_fusion << " J/kg" << std::endl;
        std::cout << "  Stefan number = " << St << std::endl;
        std::cout << "  λ (analytical) = " << lambda << std::endl;
        std::cout << "  α = " << alpha * 1e6 << " mm²/s" << std::endl;

        // Compute LBM parameters
        dt = 0.05f * DX * DX / alpha;  // Conservative time step
        alpha_lattice = alpha * dt / (DX * DX);

        std::cout << "\nLBM Parameters:" << std::endl;
        std::cout << "  dx = " << DX * 1e6 << " µm" << std::endl;
        std::cout << "  dt = " << dt * 1e6 << " µs" << std::endl;
        std::cout << "  alpha (physical) = " << alpha * 1e6 << " mm²/s" << std::endl;
        std::cout << "  alpha (lattice, expected) = " << alpha_lattice << std::endl;

        // Create thermal solver with phase change - CRITICAL FIX: Pass physical alpha, dt, dx
        // Constructor now handles conversion to lattice units internally
        solver = new ThermalLBM(NX, NY, NZ, material, alpha, true, dt, DX);

        // Initialize: solid at T_solidus everywhere
        solver->initialize(material.T_solidus);
    }

    void TearDown() override {
        delete solver;
    }

    void runSimulation(float target_time, bool apply_bc = true) {
        int num_steps = static_cast<int>(target_time / dt);

        for (int step = 0; step < num_steps; ++step) {
            // Apply fixed temperature BC at x=0 (melting boundary)
            if (apply_bc) {
                applyMeltingBoundary();
            }

            // LBM collision and streaming
            solver->collisionBGK();
            solver->streaming();
            solver->computeTemperature();

            // Optional: print progress every 100 steps
            if (step % 100 == 0) {
                std::cout << "  Step " << step << "/" << num_steps
                          << " (t = " << step * dt * 1e3 << " ms)" << std::endl;
            }
        }
    }

    void applyMeltingBoundary() {
        // Set boundary temperature at x=0 to T_liquidus
        // This is a simple implementation - ideally use proper BC kernel
        std::vector<float> h_temp(NX * NY * NZ);
        solver->copyTemperatureToHost(h_temp.data());

        // Set first cell to melting temperature
        h_temp[0] = material.T_liquidus;

        // Copy back (Note: this is inefficient, but simple for testing)
        float* d_temp = solver->getTemperature();
        cudaMemcpy(d_temp, h_temp.data(), sizeof(float), cudaMemcpyHostToDevice);
    }

    float testFrontPosition(float time) {
        // Get numerical solution
        std::vector<float> h_temp(NX * NY * NZ);
        std::vector<float> h_fl(NX * NY * NZ);

        solver->copyTemperatureToHost(h_temp.data());
        solver->copyLiquidFractionToHost(h_fl.data());

        // Find numerical front position
        float s_numerical = findMeltingFront(h_temp, h_fl);

        // Compute analytical front position
        float s_analytical = analyticalFrontPosition(time, lambda, alpha);

        // Compute error
        float error = fabsf(s_numerical - s_analytical) / s_analytical;

        std::cout << "\n  Front position at t = " << time * 1e3 << " ms:" << std::endl;
        std::cout << "    Numerical:   " << s_numerical * 1e6 << " µm" << std::endl;
        std::cout << "    Analytical:  " << s_analytical * 1e6 << " µm" << std::endl;
        std::cout << "    Error:       " << error * 100.0f << "%" << std::endl;

        // Print temperature profile
        std::cout << "\n  Temperature profile:" << std::endl;
        std::cout << "    x[µm]    T[K]    fl" << std::endl;
        for (int i = 0; i < NX; i += NX/10) {
            std::cout << "    " << std::setw(6) << std::fixed << std::setprecision(1)
                      << i * DX * 1e6
                      << std::setw(8) << std::setprecision(1) << h_temp[i]
                      << std::setw(7) << std::setprecision(2) << h_fl[i]
                      << std::endl;
        }

        return error;
    }

    ThermalLBM* solver = nullptr;
    float dt = 0.0f;
    float alpha_lattice = 0.0f;
    float St = 0.0f;
    float lambda = 0.0f;
    float alpha = 0.0f;
};

TEST_F(StefanProblemTest, ShortTime) {
    std::cout << "\nBenchmark 2.1: Stefan problem at t = 0.5 ms" << std::endl;

    float test_time = 0.5e-3f;  // 0.5 ms
    runSimulation(test_time);

    float error = testFrontPosition(test_time);

    EXPECT_LT(error, 0.05f) << "Front position error exceeds 5% threshold";
}

TEST_F(StefanProblemTest, MediumTime) {
    std::cout << "\nBenchmark 2.2: Stefan problem at t = 1.0 ms" << std::endl;

    float test_time = 1.0e-3f;  // 1.0 ms
    runSimulation(test_time);

    float error = testFrontPosition(test_time);

    EXPECT_LT(error, 0.05f) << "Front position error exceeds 5% threshold";
}

TEST_F(StefanProblemTest, LongTime) {
    std::cout << "\nBenchmark 2.3: Stefan problem at t = 2.0 ms" << std::endl;

    float test_time = 2.0e-3f;  // 2.0 ms
    runSimulation(test_time);

    float error = testFrontPosition(test_time);

    EXPECT_LT(error, 0.05f) << "Front position error exceeds 5% threshold";
}

TEST_F(StefanProblemTest, LatentHeatStorage) {
    std::cout << "\nBenchmark 2.4: Latent heat storage check" << std::endl;

    float test_time = 1.0e-3f;
    runSimulation(test_time);

    // Get liquid fraction field
    std::vector<float> h_fl(NX * NY * NZ);
    solver->copyLiquidFractionToHost(h_fl.data());

    // Compute total latent heat stored
    float V_cell = DX * DX * DX;  // Cell volume (approximation for 1D)
    float latent_heat_stored = 0.0f;

    for (int i = 0; i < NX; ++i) {
        latent_heat_stored += h_fl[i] * material.rho_solid * material.L_fusion * V_cell;
    }

    // Expected: latent heat should increase with melted volume
    std::cout << "  Total latent heat stored: " << latent_heat_stored << " J" << std::endl;
    std::cout << "  Melted volume (approximate): "
              << latent_heat_stored / (material.rho_solid * material.L_fusion) * 1e9
              << " µm³" << std::endl;

    // Latent heat should be positive and reasonable
    EXPECT_GT(latent_heat_stored, 0.0f) << "No latent heat stored (phase change not working)";
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
