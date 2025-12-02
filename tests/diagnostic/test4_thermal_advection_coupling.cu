/**
 * @file test4_thermal_advection_coupling.cu
 * @brief Test 4: Multiphysics Velocity Coupling
 *
 * PURPOSE: Are velocity fields correctly passed from Fluid to Thermal solver?
 *
 * TEST CASE:
 * - Set FluidLBM velocity to known values: ux = 1.0 m/s
 * - Pass to ThermalLBM via collisionBGK(ux, uy, uz)
 * - Check inside thermal kernel: Is velocity non-null?
 * - Check advection term contribution: v·∇T
 *
 * EXPECTED: Thermal solver should receive and use velocity
 * IF velocity is null in thermal kernel: Pointer passing bug
 */

#include <iostream>
#include <iomanip>
#include <cmath>
#include <cuda_runtime.h>
#include "physics/fluid_lbm.h"
#include "physics/thermal_lbm.h"
#include "core/lattice_d3q19.h"
#include "physics/lattice_d3q7.h"

using namespace lbm;

int main() {
    std::cout << "================================================================\n";
    std::cout << "  TEST 4: Multiphysics Velocity Coupling\n";
    std::cout << "  Isolates: Velocity field passing from Fluid to Thermal\n";
    std::cout << "================================================================\n\n";

    cudaSetDevice(0);
    core::D3Q19::initializeDevice();
    physics::D3Q7::initializeDevice();

    // Test parameters
    const int nx = 10, ny = 10, nz = 10;
    const int n_cells = nx * ny * nz;

    float nu = 0.15f;
    float rho0 = 4000.0f;
    float alpha = 5.8e-6f;  // Thermal diffusivity [m²/s]

    std::cout << "Setup:\n";
    std::cout << "  Domain: " << nx << " x " << ny << " x " << nz << " cells\n";
    std::cout << "  Fluid viscosity: " << nu << " (lattice units)\n";
    std::cout << "  Thermal diffusivity: " << alpha << " m²/s\n\n";

    // Create solvers
    physics::FluidLBM fluid_solver(nx, ny, nz, nu, rho0,
                    physics::BoundaryType::PERIODIC,
                    physics::BoundaryType::PERIODIC,
                    physics::BoundaryType::PERIODIC);

    physics::ThermalLBM thermal_solver(nx, ny, nz, alpha);

    // Initialize fluid with known velocity
    const float u_x = 1.0f;  // 1 m/s in x-direction
    const float u_y = 0.0f;
    const float u_z = 0.0f;

    fluid_solver.initialize(rho0, u_x, u_y, u_z);

    std::cout << "Fluid initial conditions:\n";
    std::cout << "  Velocity: (" << u_x << ", " << u_y << ", " << u_z << ") m/s\n\n";

    // Initialize thermal with temperature gradient in x-direction
    float* h_temperature = new float[n_cells];
    float T_left = 300.0f;
    float T_right = 600.0f;

    for (int iz = 0; iz < nz; iz++) {
        for (int iy = 0; iy < ny; iy++) {
            for (int ix = 0; ix < nx; ix++) {
                int id = ix + iy * nx + iz * nx * ny;
                // Linear gradient in x-direction
                float frac = static_cast<float>(ix) / (nx - 1);
                h_temperature[id] = T_left + frac * (T_right - T_left);
            }
        }
    }

    float* d_temperature;
    cudaMalloc(&d_temperature, n_cells * sizeof(float));
    cudaMemcpy(d_temperature, h_temperature, n_cells * sizeof(float), cudaMemcpyHostToDevice);

    thermal_solver.initialize(d_temperature);

    std::cout << "Thermal initial conditions:\n";
    std::cout << "  Temperature gradient: " << T_left << " K (left) to " << T_right << " K (right)\n";
    std::cout << "  dT/dx ≈ " << (T_right - T_left) / (nx - 1) << " K per cell\n\n";

    // Expected advection effect:
    // Without advection: T diffuses symmetrically
    // With advection (u_x > 0): T advects in +x direction, asymmetric profile

    std::cout << "Test 1: Thermal solver WITHOUT advection\n";
    std::cout << "----------------------------------------------------\n";

    // Copy current temperature
    float* h_temp_no_advection = new float[n_cells];
    cudaMemcpy(h_temp_no_advection, thermal_solver.getTemperature(), n_cells * sizeof(float), cudaMemcpyDeviceToHost);

    // Run thermal solver without velocity (pure diffusion)
    const int n_steps = 100;
    for (int step = 0; step < n_steps; step++) {
        thermal_solver.collisionBGK();  // No velocity
        thermal_solver.streaming();
        thermal_solver.computeTemperature();
    }

    thermal_solver.copyTemperatureToHost(h_temp_no_advection);

    float T_center_no_adv = h_temp_no_advection[nx/2 + (ny/2)*nx + (nz/2)*nx*ny];
    std::cout << "  After " << n_steps << " steps (diffusion only):\n";
    std::cout << "  T_center = " << T_center_no_adv << " K\n\n";

    // Reset thermal solver
    cudaMemcpy(d_temperature, h_temperature, n_cells * sizeof(float), cudaMemcpyHostToDevice);
    thermal_solver.initialize(d_temperature);

    std::cout << "Test 2: Thermal solver WITH advection\n";
    std::cout << "----------------------------------------------------\n";

    // Get velocity from fluid solver
    const float* d_ux = fluid_solver.getVelocityX();
    const float* d_uy = fluid_solver.getVelocityY();
    const float* d_uz = fluid_solver.getVelocityZ();

    // Verify velocity is non-zero
    float* h_ux_check = new float[n_cells];
    cudaMemcpy(h_ux_check, d_ux, n_cells * sizeof(float), cudaMemcpyDeviceToHost);

    float u_x_avg = 0.0f;
    for (int i = 0; i < n_cells; i++) {
        u_x_avg += h_ux_check[i];
    }
    u_x_avg /= n_cells;

    std::cout << "  Velocity from FluidLBM:\n";
    std::cout << "    Average u_x = " << u_x_avg << " m/s\n";
    std::cout << "    Expected: " << u_x << " m/s\n\n";

    if (std::abs(u_x_avg - u_x) > 1e-6f) {
        std::cout << "  [WARNING] Velocity mismatch! FluidLBM not initialized correctly.\n\n";
    }

    // Run thermal solver WITH velocity (advection + diffusion)
    float* h_temp_with_advection = new float[n_cells];

    for (int step = 0; step < n_steps; step++) {
        thermal_solver.collisionBGK(d_ux, d_uy, d_uz);  // WITH velocity
        thermal_solver.streaming();
        thermal_solver.computeTemperature();
    }

    thermal_solver.copyTemperatureToHost(h_temp_with_advection);

    float T_center_with_adv = h_temp_with_advection[nx/2 + (ny/2)*nx + (nz/2)*nx*ny];
    std::cout << "  After " << n_steps << " steps (diffusion + advection):\n";
    std::cout << "  T_center = " << T_center_with_adv << " K\n\n";

    // Compare results
    float T_diff = std::abs(T_center_with_adv - T_center_no_adv);

    std::cout << "================================================================\n";
    std::cout << "RESULTS:\n";
    std::cout << "  T_center (no advection): " << T_center_no_adv << " K\n";
    std::cout << "  T_center (with advection): " << T_center_with_adv << " K\n";
    std::cout << "  Difference: " << T_diff << " K\n\n";

    // Diagnosis
    std::cout << "DIAGNOSIS:\n";
    bool passed = false;

    if (std::abs(u_x_avg - u_x) > 1e-3f) {
        std::cout << "  [FAIL] Velocity field not initialized correctly in FluidLBM!\n";
        std::cout << "  Expected: " << u_x << " m/s, Got: " << u_x_avg << " m/s\n";
    } else if (T_diff < 1e-3f) {
        std::cout << "  [FAIL] Advection has NO EFFECT on thermal field!\n";
        std::cout << "  ROOT CAUSE: Velocity NOT coupled to ThermalLBM.\n";
        std::cout << "\n";
        std::cout << "  Possible issues:\n";
        std::cout << "  1. Velocity pointers not passed correctly to thermal.collisionBGK()\n";
        std::cout << "  2. Advection term (v·∇T) not implemented in thermal kernel\n";
        std::cout << "  3. Velocity is null inside thermal kernel (pointer bug)\n";
        std::cout << "  4. enable_thermal_advection flag not set\n";
    } else {
        std::cout << "  [PASS] Velocity field correctly coupled to thermal solver!\n";
        std::cout << "  Advection changes temperature profile by " << T_diff << " K\n";
        std::cout << "  This confirms velocity is passed and used in thermal kernel.\n";
        std::cout << "\n";
        std::cout << "  CONCLUSION: Thermal-fluid velocity coupling works.\n";
        std::cout << "  If v5 test has zero velocity, the problem is NOT in coupling.\n";
        std::cout << "  Check buoyancy force or Darcy damping.\n";
        passed = true;
    }

    // Additional check: Look at spatial temperature profile
    std::cout << "\nSpatial profiles (center line, y=ny/2, z=nz/2):\n";
    std::cout << std::setw(8) << "x"
              << std::setw(15) << "T_no_adv [K]"
              << std::setw(15) << "T_with_adv [K]"
              << std::setw(15) << "Difference"
              << "\n";
    std::cout << std::string(53, '-') << "\n";

    for (int ix = 0; ix < nx; ix += 2) {  // Every other cell
        int id = ix + (ny/2)*nx + (nz/2)*nx*ny;
        float T_no = h_temp_no_advection[id];
        float T_yes = h_temp_with_advection[id];

        std::cout << std::setw(8) << ix
                  << std::setw(15) << std::fixed << std::setprecision(2)
                  << T_no
                  << std::setw(15) << T_yes
                  << std::setw(15) << std::scientific << std::setprecision(2)
                  << (T_yes - T_no)
                  << "\n";
    }

    // Cleanup
    delete[] h_temperature;
    delete[] h_temp_no_advection;
    delete[] h_temp_with_advection;
    delete[] h_ux_check;
    cudaFree(d_temperature);

    std::cout << "\n================================================================\n";
    std::cout << "  Test 4 Complete: " << (passed ? "PASS" : "FAIL") << "\n";
    std::cout << "================================================================\n";

    return passed ? 0 : 1;
}
