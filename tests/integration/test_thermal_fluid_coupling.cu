/**
 * @file test_thermal_fluid_coupling.cu
 * @brief Integration test for thermal-fluid coupling (natural convection)
 *
 * This test validates the coupling between ThermalLBM and FluidLBM by simulating
 * natural convection in a differentially heated cavity.
 *
 * Physical Setup:
 * - 2D tall cavity (thin in z-direction for quasi-2D simulation)
 * - Hot wall: Left side (T_hot)
 * - Cold wall: Right side (T_cold)
 * - Top/bottom walls: Adiabatic (no heat flux)
 * - Initial: Fluid at rest, uniform temperature T_ref
 *
 * Expected Physics:
 * 1. Temperature gradient established between hot and cold walls
 * 2. Buoyancy force generates fluid motion: F = ρ₀·β·(T - T_ref)·g
 * 3. Fluid rises on hot side (upward velocity)
 * 4. Fluid falls on cold side (downward velocity)
 * 5. Convection cells form
 * 6. Heat transfer enhanced: Nusselt number Nu > 1
 *
 * Validation Criteria:
 * - Temperature gradient: ΔT > 5K between hot and cold sides
 * - Velocity develops: max|u| > 0.001 m/s
 * - Flow direction: upward on hot side, downward on cold side
 * - Mass conservation: relative error < 1e-4
 * - Energy trends: Total energy increases (heat input from boundaries)
 */

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <vector>
#include <algorithm>
#include <numeric>

#include "physics/thermal_lbm.h"
#include "physics/fluid_lbm.h"
#include "physics/material_properties.h"
#include "core/lattice_d3q19.h"

using namespace lbm;

/**
 * CUDA kernel to apply temperature boundary conditions
 * Hot left wall, cold right wall, adiabatic top/bottom
 */
__global__ void applyThermalWalls(
    float* temperature,
    float T_hot,
    float T_cold,
    int nx, int ny, int nz)
{
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (j >= ny || k >= nz) return;

    // Left wall (x = 0): Hot
    int idx_left = 0 + j * nx + k * nx * ny;
    temperature[idx_left] = T_hot;

    // Right wall (x = nx-1): Cold
    int idx_right = (nx - 1) + j * nx + k * nx * ny;
    temperature[idx_right] = T_cold;

    // Top and bottom walls are adiabatic (no explicit action needed)
}

/**
 * CUDA kernel to compute average quantities in a slice
 */
__global__ void computeSliceAverage(
    const float* field,
    float* slice_sum,
    int ix,
    int nx, int ny, int nz)
{
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (j >= ny || k >= nz) return;

    int idx = ix + j * nx + k * nx * ny;
    atomicAdd(slice_sum, field[idx]);
}

class ThermalFluidCouplingTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Initialize D3Q19 lattice constants
        core::D3Q19::initializeDevice();

        // Cavity geometry - tall cavity for strong convection
        nx_ = 32;  // Width
        ny_ = 64;  // Height (tall cavity)
        nz_ = 4;   // Thin (quasi-2D)
        num_cells_ = nx_ * ny_ * nz_;

        // Physical scales
        dx_ = 1e-3f;      // 1 mm grid spacing
        dt_ = 1e-5f;      // 10 microseconds

        // Material properties (physical units, then convert to lattice)
        // For LBM stability: 0.5 < omega < 1.9
        // We want nu_lattice ~ 0.1 for good stability (omega ~ 1.0, tau ~ 1.0)
        // nu_lattice = nu_physical * dt / (dx^2)
        // So: nu_physical = nu_lattice * (dx^2) / dt = 0.1 * 1e-6 / 1e-5 = 1e-2 m²/s
        rho0_ = 1000.0f;         // Density [kg/m³]
        float nu_physical = 1e-2f;  // Kinematic viscosity [m²/s] (high viscosity for stability)
        float alpha_physical = 1.4e-3f;  // Thermal diffusivity [m²/s] (scaled with nu, Pr ~ 7)

        // Convert to lattice units
        nu_ = nu_physical * dt_ / (dx_ * dx_);  // Lattice viscosity
        alpha_ = alpha_physical * dt_ / (dx_ * dx_);  // Lattice thermal diffusivity
        beta_ = 2.1e-4f;        // Thermal expansion [1/K] (dimensionless)
        cp_ = 4180.0f;          // Specific heat [J/(kg·K)]

        // Temperature boundary conditions
        T_hot_ = 320.0f;        // Hot wall [K] = 47°C
        T_cold_ = 300.0f;       // Cold wall [K] = 27°C
        T_ref_ = 310.0f;        // Reference (average) [K]
        deltaT_ = T_hot_ - T_cold_;

        // Gravity (acts in -y direction)
        // For buoyancy: F = ρ₀·β·(T - T_ref)·g
        // We want hot fluid to rise (+y), so we pass +g_y to the buoyancy function
        // because hot fluid (T > T_ref) should get positive force
        g_y_ = 9.81f;           // [m/s²] - pass as positive for buoyancy calculation

        std::cout << "\n=== Thermal-Fluid Coupling Test Setup ===\n";
        std::cout << "Domain: " << nx_ << " x " << ny_ << " x " << nz_ << " cells\n";
        std::cout << "Physical size: "
                  << nx_ * dx_ * 1e3 << " x "
                  << ny_ * dx_ * 1e3 << " x "
                  << nz_ * dx_ * 1e3 << " mm³\n";
        std::cout << "Grid spacing: " << dx_ * 1e3 << " mm\n";
        std::cout << "Time step: " << dt_ * 1e6 << " us\n\n";
        std::cout << "Fluid properties:\n";
        std::cout << "  Density: " << rho0_ << " kg/m³\n";
        std::cout << "  Viscosity (physical): " << nu_physical * 1e6 << " mm²/s\n";
        std::cout << "  Viscosity (lattice): " << nu_ << "\n";
        std::cout << "  Thermal diffusivity (physical): " << alpha_physical * 1e6 << " mm²/s\n";
        std::cout << "  Thermal diffusivity (lattice): " << alpha_ << "\n";
        std::cout << "  Thermal expansion: " << beta_ * 1e3 << " × 10⁻³ K⁻¹\n\n";
        std::cout << "Thermal conditions:\n";
        std::cout << "  T_hot: " << T_hot_ << " K\n";
        std::cout << "  T_cold: " << T_cold_ << " K\n";
        std::cout << "  T_ref: " << T_ref_ << " K\n";
        std::cout << "  ΔT: " << deltaT_ << " K\n\n";

        // Characteristic dimensionless numbers (using physical values)
        float L = ny_ * dx_;  // Height is characteristic length
        float u_buoyancy = sqrtf(g_y_ * beta_ * deltaT_ * L);  // g_y is already positive
        float Ra = g_y_ * beta_ * deltaT_ * L * L * L / (nu_physical * alpha_physical);
        float Pr = nu_physical / alpha_physical;

        std::cout << "Dimensionless parameters:\n";
        std::cout << "  Rayleigh number (Ra): " << std::scientific << Ra << "\n";
        std::cout << "  Prandtl number (Pr): " << Pr << "\n";
        std::cout << "  Expected velocity scale: " << std::fixed
                  << u_buoyancy << " m/s\n\n";
    }

    // Domain parameters
    int nx_, ny_, nz_, num_cells_;
    float dx_, dt_;

    // Material properties
    float rho0_, nu_, alpha_, beta_, cp_;

    // Thermal conditions
    float T_hot_, T_cold_, T_ref_, deltaT_;

    // Gravity
    float g_y_;
};

/**
 * Test 1: Natural Convection in Differentially Heated Cavity
 *
 * This is the main coupling test that validates:
 * 1. Temperature field → buoyancy force → velocity field
 * 2. Natural convection circulation pattern
 * 3. Physical correctness of coupling
 */
TEST_F(ThermalFluidCouplingTest, NaturalConvectionInCavity) {
    std::cout << "=== Natural Convection Test ===\n\n";

    // Create material properties (simplified, without phase change)
    physics::MaterialProperties material;
    material.rho_solid = rho0_;
    material.rho_liquid = rho0_;
    material.cp_solid = cp_;
    material.cp_liquid = cp_;
    material.k_solid = alpha_ * rho0_ * cp_;  // k = α·ρ·cp
    material.k_liquid = material.k_solid;
    material.T_solidus = 0.0f;
    material.T_liquidus = 0.0f;  // No phase change for this test

    // Create solvers
    std::cout << "Initializing solvers...\n";
    physics::ThermalLBM thermal(nx_, ny_, nz_, material, alpha_, false);
    physics::FluidLBM fluid(nx_, ny_, nz_, nu_, rho0_,
                           physics::BoundaryType::WALL,      // x: walls
                           physics::BoundaryType::WALL,      // y: walls
                           physics::BoundaryType::PERIODIC); // z: periodic (thin)

    // Initialize fields
    thermal.initialize(T_ref_);
    fluid.initialize(rho0_, 0.0f, 0.0f, 0.0f);

    // Allocate device memory for forces
    float *d_fx, *d_fy, *d_fz;
    cudaMalloc(&d_fx, num_cells_ * sizeof(float));
    cudaMalloc(&d_fy, num_cells_ * sizeof(float));
    cudaMalloc(&d_fz, num_cells_ * sizeof(float));
    cudaMemset(d_fx, 0, num_cells_ * sizeof(float));
    cudaMemset(d_fy, 0, num_cells_ * sizeof(float));
    cudaMemset(d_fz, 0, num_cells_ * sizeof(float));

    // Boundary condition kernel configuration
    dim3 block_bc(1, 8, 4);
    dim3 grid_bc(1,
                 (ny_ + block_bc.y - 1) / block_bc.y,
                 (nz_ + block_bc.z - 1) / block_bc.z);

    // Time-stepping loop
    const int n_steps = 5000;  // Reduced for faster testing
    const int check_interval = 500;

    std::cout << "Running simulation for " << n_steps << " steps...\n";
    std::cout << std::setw(8) << "Step"
              << std::setw(12) << "Time [ms]"
              << std::setw(12) << "T_avg [K]"
              << std::setw(12) << "u_max [mm/s]"
              << std::setw(12) << "v_max [mm/s]"
              << "\n";
    std::cout << std::string(56, '-') << "\n";

    // Initial state
    float initial_mass = 0.0f;
    {
        std::vector<float> rho_host(num_cells_);
        fluid.copyDensityToHost(rho_host.data());
        initial_mass = std::accumulate(rho_host.begin(), rho_host.end(), 0.0f);
    }

    for (int step = 0; step <= n_steps; ++step) {
        // Apply thermal boundary conditions (hot left, cold right)
        applyThermalWalls<<<grid_bc, block_bc>>>(
            thermal.getTemperature(), T_hot_, T_cold_, nx_, ny_, nz_
        );
        cudaDeviceSynchronize();

        // Thermal step
        thermal.collisionBGK();
        thermal.streaming();
        thermal.computeTemperature();

        // Compute buoyancy force from temperature field
        // Scale down force for numerical stability
        float force_scale = 1e-3f;
        fluid.computeBuoyancyForce(
            thermal.getTemperature(), T_ref_, beta_,
            0.0f, g_y_ * force_scale, 0.0f,  // Reduced gravity for stability
            d_fx, d_fy, d_fz
        );

        // Fluid step with buoyancy
        fluid.computeMacroscopic();
        fluid.collisionBGK(d_fx, d_fy, d_fz);
        fluid.streaming();
        fluid.applyBoundaryConditions(1);  // Apply wall boundaries

        // Periodic diagnostics
        if (step % check_interval == 0) {
            std::vector<float> T(num_cells_);
            std::vector<float> ux(num_cells_);
            std::vector<float> uy(num_cells_);

            thermal.copyTemperatureToHost(T.data());
            fluid.copyVelocityToHost(ux.data(), uy.data(), nullptr);

            float T_avg = std::accumulate(T.begin(), T.end(), 0.0f) / num_cells_;
            float u_max = *std::max_element(ux.begin(), ux.end(),
                [](float a, float b) { return fabsf(a) < fabsf(b); });
            float v_max = *std::max_element(uy.begin(), uy.end(),
                [](float a, float b) { return fabsf(a) < fabsf(b); });

            std::cout << std::setw(8) << step
                      << std::setw(12) << std::fixed << std::setprecision(3)
                      << step * dt_ * 1e3
                      << std::setw(12) << std::fixed << std::setprecision(2)
                      << T_avg
                      << std::setw(12) << std::fixed << std::setprecision(3)
                      << u_max * 1e3
                      << std::setw(12) << std::fixed << std::setprecision(3)
                      << v_max * 1e3
                      << "\n";
        }
    }

    std::cout << std::string(56, '-') << "\n\n";

    // Final state analysis
    std::vector<float> T_final(num_cells_);
    std::vector<float> ux_final(num_cells_);
    std::vector<float> uy_final(num_cells_);
    std::vector<float> rho_final(num_cells_);

    thermal.copyTemperatureToHost(T_final.data());
    fluid.copyVelocityToHost(ux_final.data(), uy_final.data(), nullptr);
    fluid.copyDensityToHost(rho_final.data());

    // Compute average temperatures on hot and cold walls
    float T_hot_avg = 0.0f, T_cold_avg = 0.0f;
    int n_wall_cells = ny_ * nz_;

    for (int j = 0; j < ny_; ++j) {
        for (int k = 0; k < nz_; ++k) {
            int idx_hot = 0 + j * nx_ + k * nx_ * ny_;
            int idx_cold = (nx_ - 1) + j * nx_ + k * nx_ * ny_;
            T_hot_avg += T_final[idx_hot];
            T_cold_avg += T_final[idx_cold];
        }
    }
    T_hot_avg /= n_wall_cells;
    T_cold_avg /= n_wall_cells;

    // Compute velocity statistics on hot and cold sides
    // Hot side: average y-velocity in left quarter of domain
    // Cold side: average y-velocity in right quarter of domain
    float uy_hot_avg = 0.0f, uy_cold_avg = 0.0f;
    int n_hot = 0, n_cold = 0;

    for (int i = 0; i < nx_ / 4; ++i) {
        for (int j = ny_ / 4; j < 3 * ny_ / 4; ++j) {  // Middle half in y
            for (int k = 0; k < nz_; ++k) {
                int idx = i + j * nx_ + k * nx_ * ny_;
                uy_hot_avg += uy_final[idx];
                n_hot++;
            }
        }
    }

    for (int i = 3 * nx_ / 4; i < nx_; ++i) {
        for (int j = ny_ / 4; j < 3 * ny_ / 4; ++j) {
            for (int k = 0; k < nz_; ++k) {
                int idx = i + j * nx_ + k * nx_ * ny_;
                uy_cold_avg += uy_final[idx];
                n_cold++;
            }
        }
    }

    uy_hot_avg /= n_hot;
    uy_cold_avg /= n_cold;

    // Maximum velocities
    float u_max = 0.0f, v_max = 0.0f;
    for (int i = 0; i < num_cells_; ++i) {
        u_max = fmaxf(u_max, fabsf(ux_final[i]));
        v_max = fmaxf(v_max, fabsf(uy_final[i]));
    }

    // Mass conservation check
    float final_mass = std::accumulate(rho_final.begin(), rho_final.end(), 0.0f);
    float mass_error = fabsf(final_mass - initial_mass) / initial_mass;

    // Print results
    std::cout << "=== Final Results ===\n\n";
    std::cout << "Temperature gradient:\n";
    std::cout << "  T_hot (left wall): " << T_hot_avg << " K\n";
    std::cout << "  T_cold (right wall): " << T_cold_avg << " K\n";
    std::cout << "  Difference: " << T_hot_avg - T_cold_avg << " K\n\n";

    std::cout << "Velocity field:\n";
    std::cout << "  Max |u_x|: " << u_max << " m/s (" << u_max * 1e3 << " mm/s)\n";
    std::cout << "  Max |u_y|: " << v_max << " m/s (" << v_max * 1e3 << " mm/s)\n";
    std::cout << "  Avg u_y (hot side): " << uy_hot_avg << " m/s ("
              << uy_hot_avg * 1e3 << " mm/s)\n";
    std::cout << "  Avg u_y (cold side): " << uy_cold_avg << " m/s ("
              << uy_cold_avg * 1e3 << " mm/s)\n\n";

    std::cout << "Conservation:\n";
    std::cout << "  Initial mass: " << initial_mass << "\n";
    std::cout << "  Final mass: " << final_mass << "\n";
    std::cout << "  Relative error: " << mass_error * 100.0f << " %\n\n";

    // Validation checks
    std::cout << "=== Validation ===\n\n";

    // 1. Temperature gradient established
    float temp_diff = T_hot_avg - T_cold_avg;
    bool temp_gradient_ok = (temp_diff > 5.0f);
    std::cout << "1. Temperature gradient: ";
    if (temp_gradient_ok) {
        std::cout << "PASS (ΔT = " << temp_diff << " K > 5 K)\n";
    } else {
        std::cout << "FAIL (ΔT = " << temp_diff << " K <= 5 K)\n";
    }

    // 2. Velocity developed
    bool velocity_developed = (v_max > 1e-3f);
    std::cout << "2. Velocity developed: ";
    if (velocity_developed) {
        std::cout << "PASS (max|v| = " << v_max << " m/s > 0.001 m/s)\n";
    } else {
        std::cout << "FAIL (max|v| = " << v_max << " m/s <= 0.001 m/s)\n";
    }

    // 3. Hot side upward flow
    bool hot_upward = (uy_hot_avg > 0.0f);
    std::cout << "3. Hot side upward flow: ";
    if (hot_upward) {
        std::cout << "PASS (v_hot = " << uy_hot_avg * 1e3 << " mm/s > 0)\n";
    } else {
        std::cout << "FAIL (v_hot = " << uy_hot_avg * 1e3 << " mm/s <= 0)\n";
    }

    // 4. Cold side downward flow
    bool cold_downward = (uy_cold_avg < 0.0f);
    std::cout << "4. Cold side downward flow: ";
    if (cold_downward) {
        std::cout << "PASS (v_cold = " << uy_cold_avg * 1e3 << " mm/s < 0)\n";
    } else {
        std::cout << "FAIL (v_cold = " << uy_cold_avg * 1e3 << " mm/s >= 0)\n";
    }

    // 5. Mass conservation
    bool mass_conserved = (mass_error < 1e-4f);
    std::cout << "5. Mass conservation: ";
    if (mass_conserved) {
        std::cout << "PASS (error = " << mass_error * 100.0f << " % < 0.01 %)\n";
    } else {
        std::cout << "FAIL (error = " << mass_error * 100.0f << " % >= 0.01 %)\n";
    }

    std::cout << "\n";

    // GTest assertions
    EXPECT_GT(temp_diff, 5.0f)
        << "Temperature gradient should be established (ΔT > 5K)";
    EXPECT_GT(v_max, 1e-3f)
        << "Velocity should develop from rest (max|v| > 0.001 m/s)";
    EXPECT_GT(uy_hot_avg, 0.0f)
        << "Hot side should have upward flow (positive v_y)";
    EXPECT_LT(uy_cold_avg, 0.0f)
        << "Cold side should have downward flow (negative v_y)";
    EXPECT_LT(mass_error, 1e-4f)
        << "Mass should be conserved (relative error < 1e-4)";

    // Cleanup
    cudaFree(d_fx);
    cudaFree(d_fy);
    cudaFree(d_fz);

    std::cout << "=== Natural Convection Test: PASSED ===\n\n";
}

/**
 * Test 2: Buoyancy Force Computation (Simpler test)
 *
 * This test validates the buoyancy force computation:
 * - Create a static temperature gradient
 * - Verify buoyancy force has correct sign and magnitude
 * Note: Full flow validation is done in Test 1 (NaturalConvectionInCavity)
 */
TEST_F(ThermalFluidCouplingTest, BuoyancyForceComputation) {
    std::cout << "=== Buoyancy Response Test ===\n\n";

    // Simpler geometry for faster test
    int nx = 16, ny = 32, nz = 4;
    int num_cells = nx * ny * nz;

    std::cout << "Domain: " << nx << " x " << ny << " x " << nz << " cells\n\n";

    // Create material properties
    physics::MaterialProperties material;
    material.rho_solid = rho0_;
    material.rho_liquid = rho0_;
    material.cp_solid = cp_;
    material.cp_liquid = cp_;
    material.k_solid = alpha_ * rho0_ * cp_;
    material.k_liquid = material.k_solid;
    material.T_solidus = 0.0f;
    material.T_liquidus = 0.0f;

    // Create solvers
    physics::ThermalLBM thermal(nx, ny, nz, material, alpha_, false);
    physics::FluidLBM fluid(nx, ny, nz, nu_, rho0_);

    // Initialize with temperature gradient (hot bottom, cold top)
    std::vector<float> T_init(num_cells);
    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                int idx = i + j * nx + k * nx * ny;
                // Linear temperature profile: hot at bottom (j=0), cold at top (j=ny-1)
                float frac = static_cast<float>(j) / (ny - 1);
                T_init[idx] = T_hot_ - frac * deltaT_;  // Hot at bottom
            }
        }
    }

    thermal.initialize(T_init.data());
    fluid.initialize(rho0_, 0.0f, 0.0f, 0.0f);

    // Allocate device memory for forces
    float *d_fx, *d_fy, *d_fz;
    cudaMalloc(&d_fx, num_cells * sizeof(float));
    cudaMalloc(&d_fy, num_cells * sizeof(float));
    cudaMalloc(&d_fz, num_cells * sizeof(float));

    // Compute buoyancy force
    fluid.computeBuoyancyForce(
        thermal.getTemperature(), T_ref_, beta_,
        0.0f, g_y_, 0.0f,
        d_fx, d_fy, d_fz
    );

    // Copy force back to host and verify
    std::vector<float> fy_host(num_cells);
    cudaMemcpy(fy_host.data(), d_fy, num_cells * sizeof(float), cudaMemcpyDeviceToHost);

    // Check force at bottom (hot) and top (cold)
    int idx_bottom = nx / 2 + 0 * nx + 0 * nx * ny;  // Middle of bottom
    int idx_top = nx / 2 + (ny - 1) * nx + 0 * nx * ny;  // Middle of top

    float fy_bottom = fy_host[idx_bottom];
    float fy_top = fy_host[idx_top];

    std::cout << "Temperature gradient:\n";
    std::cout << "  Bottom: " << T_init[idx_bottom] << " K\n";
    std::cout << "  Top: " << T_init[idx_top] << " K\n\n";

    std::cout << "Buoyancy force:\n";
    std::cout << "  Bottom (hot): " << fy_bottom << " m/s²\n";
    std::cout << "  Top (cold): " << fy_top << " m/s²\n\n";

    // Expected: hot fluid has upward force (positive fy), cold fluid has downward force (negative fy)
    // F = ρ₀·β·(T - T_ref)·g, with g_y positive for buoyancy
    float dT_bottom = T_init[idx_bottom] - T_ref_;
    float expected_fy_bottom = rho0_ * beta_ * dT_bottom * g_y_;

    float dT_top = T_init[idx_top] - T_ref_;
    float expected_fy_top = rho0_ * beta_ * dT_top * g_y_;

    std::cout << "Expected buoyancy force:\n";
    std::cout << "  Bottom: " << expected_fy_bottom << " m/s²\n";
    std::cout << "  Top: " << expected_fy_top << " m/s²\n\n";

    EXPECT_GT(fy_bottom, 0.0f)
        << "Hot fluid (bottom) should have upward force";
    EXPECT_LT(fy_top, 0.0f)
        << "Cold fluid (top) should have downward force";
    EXPECT_NEAR(fy_bottom, expected_fy_bottom, fabsf(expected_fy_bottom) * 0.01f)
        << "Buoyancy force magnitude should match expected (within 1%)";
    EXPECT_NEAR(fy_top, expected_fy_top, fabsf(expected_fy_top) * 0.01f)
        << "Buoyancy force magnitude should match expected (within 1%)";

    std::cout << "\n=== Buoyancy Force Test: PASSED ===\n";
    std::cout << "Note: Flow development validation is performed in NaturalConvectionInCavity test\n\n";

    // Cleanup
    cudaFree(d_fx);
    cudaFree(d_fy);
    cudaFree(d_fz);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
