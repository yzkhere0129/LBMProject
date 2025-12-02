/**
 * @file test2_buoyancy_force.cu
 * @brief Test 2: Buoyancy Force Magnitude
 *
 * PURPOSE: Is buoyancy force being computed and applied?
 *
 * TEST CASE:
 * - Create temperature field: T_bottom = 300 K, T_top = 3000 K
 * - Compute buoyancy: F = rho*beta*g*(T - T_ref)
 * - Print max force magnitude
 * - Apply to FluidLBM
 * - Run 10 steps
 * - Check velocity response
 *
 * EXPECTED:
 * - Force: rho*beta*g*dT = 4000 * 1e-4 * 10 * 2700 = 10.8 N/m³
 * - Velocity after 10 steps: F*dt/rho ~ 10 * 1us / 4000 ~ 0.025 mm/s
 *
 * IF force is zero: Buoyancy not implemented
 * IF force OK but v=0: Force not coupled to FluidLBM
 */

#include <iostream>
#include <iomanip>
#include <cmath>
#include <cuda_runtime.h>
#include "physics/fluid_lbm.h"
#include "core/lattice_d3q19.h"

using namespace lbm;

int main() {
    std::cout << "================================================================\n";
    std::cout << "  TEST 2: Buoyancy Force Magnitude\n";
    std::cout << "  Isolates: Buoyancy force computation and application\n";
    std::cout << "================================================================\n\n";

    cudaSetDevice(0);
    core::D3Q19::initializeDevice();

    // Test parameters
    const int nx = 10, ny = 10, nz = 10;
    const int n_cells = nx * ny * nz;

    float nu = 0.15f;      // Lattice viscosity
    float rho0 = 4000.0f;  // Liquid density [kg/m³]

    std::cout << "Setup:\n";
    std::cout << "  Domain: " << nx << " x " << ny << " x " << nz << " cells\n";
    std::cout << "  Density: " << rho0 << " kg/m³\n";
    std::cout << "  Boundaries: All periodic\n\n";

    // Create solver
    physics::FluidLBM solver(nx, ny, nz, nu, rho0,
                    physics::BoundaryType::PERIODIC,
                    physics::BoundaryType::PERIODIC,
                    physics::BoundaryType::PERIODIC);

    solver.initialize(rho0, 0.0f, 0.0f, 0.0f);

    // Create temperature field: linear gradient from bottom to top
    float* h_temperature = new float[n_cells];
    float T_bottom = 300.0f;   // Cold bottom
    float T_top = 3000.0f;     // Hot top

    for (int iz = 0; iz < nz; iz++) {
        for (int iy = 0; iy < ny; iy++) {
            for (int ix = 0; ix < nx; ix++) {
                int id = ix + iy * nx + iz * nx * ny;
                // Temperature increases with z (vertical)
                float frac = static_cast<float>(iz) / (nz - 1);
                h_temperature[id] = T_bottom + frac * (T_top - T_bottom);
            }
        }
    }

    float* d_temperature;
    cudaMalloc(&d_temperature, n_cells * sizeof(float));
    cudaMemcpy(d_temperature, h_temperature, n_cells * sizeof(float), cudaMemcpyHostToDevice);

    // Allocate force arrays
    float *d_fx, *d_fy, *d_fz;
    cudaMalloc(&d_fx, n_cells * sizeof(float));
    cudaMalloc(&d_fy, n_cells * sizeof(float));
    cudaMalloc(&d_fz, n_cells * sizeof(float));

    // Buoyancy parameters
    float T_ref = 300.0f;       // Reference temperature [K]
    float beta = 1e-4f;         // Thermal expansion coefficient [1/K]
    float g = 10.0f;            // Gravity magnitude [m/s²]

    std::cout << "Temperature field:\n";
    std::cout << "  T_bottom (z=0): " << T_bottom << " K\n";
    std::cout << "  T_top (z=" << nz-1 << "): " << T_top << " K\n";
    std::cout << "  dT = " << (T_top - T_bottom) << " K\n\n";

    std::cout << "Buoyancy parameters:\n";
    std::cout << "  T_ref = " << T_ref << " K\n";
    std::cout << "  beta = " << beta << " 1/K\n";
    std::cout << "  g = " << g << " m/s² (in +z direction)\n";
    std::cout << "  rho0 = " << rho0 << " kg/m³\n\n";

    // Expected force at top (hottest region)
    float dT_max = T_top - T_ref;
    float F_expected = rho0 * beta * g * dT_max;

    std::cout << "Expected buoyancy force at top:\n";
    std::cout << "  F = rho*beta*g*dT = " << rho0 << " * " << beta << " * " << g
              << " * " << dT_max << "\n";
    std::cout << "  F = " << F_expected << " N/m³ (or m/s² per unit mass)\n\n";

    // Compute buoyancy force
    std::cout << "Computing buoyancy force...\n";
    solver.computeBuoyancyForce(
        d_temperature, T_ref, beta,
        0.0f, 0.0f, g,  // Gravity in +z direction
        d_fx, d_fy, d_fz
    );

    // Check force magnitude
    float* h_fx = new float[n_cells];
    float* h_fy = new float[n_cells];
    float* h_fz = new float[n_cells];

    cudaMemcpy(h_fx, d_fx, n_cells * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_fy, d_fy, n_cells * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_fz, d_fz, n_cells * sizeof(float), cudaMemcpyDeviceToHost);

    float f_max = 0.0f;
    float f_z_max = 0.0f;
    float f_z_min = 0.0f;
    int nonzero_force = 0;

    for (int i = 0; i < n_cells; i++) {
        float f_mag = std::sqrt(h_fx[i]*h_fx[i] + h_fy[i]*h_fy[i] + h_fz[i]*h_fz[i]);
        f_max = std::max(f_max, f_mag);
        f_z_max = std::max(f_z_max, h_fz[i]);
        f_z_min = std::min(f_z_min, h_fz[i]);

        if (f_mag > 1e-10f) nonzero_force++;
    }

    std::cout << "\nForce statistics:\n";
    std::cout << "  Max |F|: " << f_max << " N/m³\n";
    std::cout << "  F_z range: [" << f_z_min << ", " << f_z_max << "] N/m³\n";
    std::cout << "  Expected F_max: " << F_expected << " N/m³\n";
    std::cout << "  Cells with nonzero force: " << nonzero_force << " / " << n_cells << "\n\n";

    // Now apply force and check velocity response
    std::cout << "Running 10 time steps with buoyancy force...\n";
    std::cout << std::setw(8) << "Step"
              << std::setw(15) << "v_max [m/s]"
              << std::setw(15) << "v_z_max [m/s]"
              << "\n";
    std::cout << std::string(38, '-') << "\n";

    float* h_ux = new float[n_cells];
    float* h_uy = new float[n_cells];
    float* h_uz = new float[n_cells];

    const int n_steps = 10;

    for (int step = 0; step <= n_steps; step++) {
        // Recompute buoyancy (in real sim, temperature would change)
        solver.computeBuoyancyForce(
            d_temperature, T_ref, beta,
            0.0f, 0.0f, g,
            d_fx, d_fy, d_fz
        );

        // FluidLBM step
        solver.computeMacroscopic();
        solver.collisionBGK(d_fx, d_fy, d_fz);
        solver.streaming();

        // Check velocity
        solver.copyVelocityToHost(h_ux, h_uy, h_uz);

        float v_max = 0.0f;
        float v_z_max = 0.0f;

        for (int i = 0; i < n_cells; i++) {
            float v_mag = std::sqrt(h_ux[i]*h_ux[i] + h_uy[i]*h_uy[i] + h_uz[i]*h_uz[i]);
            v_max = std::max(v_max, v_mag);
            v_z_max = std::max(v_z_max, std::abs(h_uz[i]));
        }

        std::cout << std::setw(8) << step
                  << std::setw(15) << std::scientific << std::setprecision(4)
                  << v_max
                  << std::setw(15) << v_z_max
                  << "\n";
    }

    std::cout << "\n";

    // Final velocity check
    solver.copyVelocityToHost(h_ux, h_uy, h_uz);

    float v_final_max = 0.0f;
    float v_z_final_max = 0.0f;

    for (int i = 0; i < n_cells; i++) {
        float v_mag = std::sqrt(h_ux[i]*h_ux[i] + h_uy[i]*h_uy[i] + h_uz[i]*h_uz[i]);
        v_final_max = std::max(v_final_max, v_mag);
        v_z_final_max = std::max(v_z_final_max, std::abs(h_uz[i]));
    }

    std::cout << "================================================================\n";
    std::cout << "RESULTS:\n";
    std::cout << "  Force max: " << f_max << " N/m³\n";
    std::cout << "  Force expected: " << F_expected << " N/m³\n";
    std::cout << "  Velocity max (final): " << v_final_max << " m/s\n";
    std::cout << "  Velocity_z max (final): " << v_z_final_max << " m/s\n\n";

    // Diagnosis
    std::cout << "DIAGNOSIS:\n";
    bool passed = false;

    if (f_max < 1e-6f) {
        std::cout << "  [FAIL] Buoyancy force is ZERO!\n";
        std::cout << "  ROOT CAUSE: computeBuoyancyForce() is not working.\n";
        std::cout << "\n";
        std::cout << "  Possible issues:\n";
        std::cout << "  1. Kernel not launched\n";
        std::cout << "  2. Formula error: F = rho*beta*(T-T_ref)*g\n";
        std::cout << "  3. Device memory not allocated/initialized\n";
    } else if (std::abs(f_max - F_expected) / F_expected > 0.1f) {
        std::cout << "  [WARNING] Force magnitude differs from expected by >10%\n";
        std::cout << "  May indicate scaling or formula error.\n";
        std::cout << "  Relative error: " << std::abs(f_max - F_expected) / F_expected * 100.0f << "%\n";
    } else if (v_final_max < 1e-10f) {
        std::cout << "  [FAIL] Force is computed correctly but velocity is ZERO!\n";
        std::cout << "  ROOT CAUSE: Buoyancy force NOT coupled to FluidLBM.\n";
        std::cout << "\n";
        std::cout << "  Possible issues:\n";
        std::cout << "  1. Force arrays not passed to collisionBGK\n";
        std::cout << "  2. Force not added to velocity in collision kernel\n";
        std::cout << "  3. Pointer passing bug between modules\n";
    } else {
        std::cout << "  [PASS] Buoyancy force computed AND applied correctly!\n";
        std::cout << "  Force magnitude: " << f_max << " N/m³ (matches expected: "
                  << F_expected << " N/m³)\n";
        std::cout << "  Velocity response: " << v_final_max << " m/s (reasonable)\n";
        std::cout << "\n";
        std::cout << "  CONCLUSION: Buoyancy force pipeline works.\n";
        std::cout << "  If v5 test has zero velocity, the problem is NOT buoyancy.\n";
        std::cout << "  Check Darcy damping or multiphysics coupling.\n";
        passed = true;
    }

    // Cleanup
    delete[] h_temperature;
    delete[] h_fx;
    delete[] h_fy;
    delete[] h_fz;
    delete[] h_ux;
    delete[] h_uy;
    delete[] h_uz;

    cudaFree(d_temperature);
    cudaFree(d_fx);
    cudaFree(d_fy);
    cudaFree(d_fz);

    std::cout << "\n================================================================\n";
    std::cout << "  Test 2 Complete: " << (passed ? "PASS" : "FAIL") << "\n";
    std::cout << "================================================================\n";

    return passed ? 0 : 1;
}
