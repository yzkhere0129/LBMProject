/**
 * @file test3_darcy_damping.cu
 * @brief Test 3: Darcy Damping Isolation
 *
 * PURPOSE: Is Darcy damping killing ALL flow instead of just solid regions?
 *
 * TEST CASE:
 * - Set up FluidLBM with initial velocity ux = 1.0 m/s everywhere
 * - Set liquid_fraction = 1.0 (all liquid)
 * - Apply Darcy damping with C = 20000
 * - Check velocity after damping
 *
 * CORRECT BEHAVIOR:
 * - In liquid (f=1): u_new = u_old * (1 - C*(1-1)) = u_old (no damping)
 * - In solid (f=0): u_new = u_old * (1 - C*(1-0)) ≈ 0 (full damping)
 *
 * BUG BEHAVIOR:
 * - Everywhere: u_new = u_old * (1 - C) = negative (kills flow)
 *
 * EXPECTED: Velocity should remain 1.0 m/s in liquid regions
 * IF ZERO everywhere: Darcy implementation bug confirmed
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
    std::cout << "  TEST 3: Darcy Damping Isolation\n";
    std::cout << "  Isolates: Darcy damping implementation\n";
    std::cout << "================================================================\n\n";

    cudaSetDevice(0);
    core::D3Q19::initializeDevice();

    // Test parameters
    const int nx = 10, ny = 10, nz = 10;
    const int n_cells = nx * ny * nz;

    float nu = 0.15f;
    float rho0 = 4000.0f;

    std::cout << "Setup:\n";
    std::cout << "  Domain: " << nx << " x " << ny << " x " << nz << " cells\n";
    std::cout << "  Density: " << rho0 << " kg/m³\n\n";

    // Create solver
    physics::FluidLBM solver(nx, ny, nz, nu, rho0,
                    physics::BoundaryType::PERIODIC,
                    physics::BoundaryType::PERIODIC,
                    physics::BoundaryType::PERIODIC);

    // Initialize with NON-ZERO velocity
    const float u_initial = 1.0f;  // 1 m/s
    solver.initialize(rho0, u_initial, 0.0f, 0.0f);

    std::cout << "Initial conditions:\n";
    std::cout << "  Velocity: u_x = " << u_initial << " m/s (uniform everywhere)\n\n";

    // Create liquid fraction field
    // Test 3 scenarios: fully liquid, fully solid, and mixed
    float* h_liquid_fraction = new float[n_cells];

    std::cout << "Liquid fraction field:\n";
    std::cout << "  Bottom third (z=0 to z=3): f_l = 0.0 (fully solid)\n";
    std::cout << "  Middle third (z=3 to z=6): f_l = 0.5 (mushy)\n";
    std::cout << "  Top third (z=6 to z=9): f_l = 1.0 (fully liquid)\n\n";

    for (int iz = 0; iz < nz; iz++) {
        for (int iy = 0; iy < ny; iy++) {
            for (int ix = 0; ix < nx; ix++) {
                int id = ix + iy * nx + iz * nx * ny;

                if (iz < nz/3) {
                    h_liquid_fraction[id] = 0.0f;  // Solid
                } else if (iz < 2*nz/3) {
                    h_liquid_fraction[id] = 0.5f;  // Mushy
                } else {
                    h_liquid_fraction[id] = 1.0f;  // Liquid
                }
            }
        }
    }

    float* d_liquid_fraction;
    cudaMalloc(&d_liquid_fraction, n_cells * sizeof(float));
    cudaMemcpy(d_liquid_fraction, h_liquid_fraction, n_cells * sizeof(float), cudaMemcpyHostToDevice);

    // Allocate force arrays (initialize to zero)
    float *d_fx, *d_fy, *d_fz;
    cudaMalloc(&d_fx, n_cells * sizeof(float));
    cudaMalloc(&d_fy, n_cells * sizeof(float));
    cudaMalloc(&d_fz, n_cells * sizeof(float));
    cudaMemset(d_fx, 0, n_cells * sizeof(float));
    cudaMemset(d_fy, 0, n_cells * sizeof(float));
    cudaMemset(d_fz, 0, n_cells * sizeof(float));

    // Darcy damping constant (typical value from main sim)
    const float C_darcy = 20000.0f;

    std::cout << "Darcy parameters:\n";
    std::cout << "  C_darcy = " << C_darcy << "\n";
    std::cout << "  Formula: F_darcy = -C * (1-f_l)^2 / (f_l^3 + epsilon) * u\n";
    std::cout << "  Epsilon = 1e-3 (small number to avoid division by zero)\n\n";

    // Expected behavior:
    // In fully liquid (f=1): F_darcy = -C * 0^2 / (1 + eps) * u = 0 (no damping)
    // In fully solid (f=0): F_darcy = -C * 1^2 / (0 + eps) * u = -C/eps * u (strong damping)
    // In mushy (f=0.5): F_darcy = -C * 0.25 / (0.125 + 0.001) * u ≈ -C*2*u (moderate damping)

    std::cout << "Expected Darcy force:\n";
    std::cout << "  Solid (f=0): F ≈ -" << C_darcy << " * u (kills flow)\n";
    std::cout << "  Mushy (f=0.5): F ≈ -" << C_darcy * 0.25 / (0.125 + 0.001) << " * u\n";
    std::cout << "  Liquid (f=1): F = 0 (no damping)\n\n";

    // Apply Darcy damping
    std::cout << "Applying Darcy damping...\n";
    solver.applyDarcyDamping(d_liquid_fraction, C_darcy, d_fx, d_fy, d_fz);

    // Check force magnitude
    float* h_fx = new float[n_cells];
    float* h_fy = new float[n_cells];
    float* h_fz = new float[n_cells];

    cudaMemcpy(h_fx, d_fx, n_cells * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_fy, d_fy, n_cells * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_fz, d_fz, n_cells * sizeof(float), cudaMemcpyDeviceToHost);

    // Analyze by region
    float f_solid_avg = 0.0f, f_mushy_avg = 0.0f, f_liquid_avg = 0.0f;
    int count_solid = 0, count_mushy = 0, count_liquid = 0;

    for (int iz = 0; iz < nz; iz++) {
        for (int iy = 0; iy < ny; iy++) {
            for (int ix = 0; ix < nx; ix++) {
                int id = ix + iy * nx + iz * nx * ny;
                float f_mag = std::abs(h_fx[id]);  // x-component (velocity is in x)

                if (iz < nz/3) {
                    f_solid_avg += f_mag;
                    count_solid++;
                } else if (iz < 2*nz/3) {
                    f_mushy_avg += f_mag;
                    count_mushy++;
                } else {
                    f_liquid_avg += f_mag;
                    count_liquid++;
                }
            }
        }
    }

    f_solid_avg /= count_solid;
    f_mushy_avg /= count_mushy;
    f_liquid_avg /= count_liquid;

    std::cout << "\nDarcy force by region:\n";
    std::cout << "  Solid region: avg |F_x| = " << f_solid_avg << " N/m³\n";
    std::cout << "  Mushy region: avg |F_x| = " << f_mushy_avg << " N/m³\n";
    std::cout << "  Liquid region: avg |F_x| = " << f_liquid_avg << " N/m³\n\n";

    // Now run simulation with Darcy damping
    std::cout << "Running 10 time steps with Darcy damping...\n";
    std::cout << std::setw(8) << "Step"
              << std::setw(18) << "v_solid [m/s]"
              << std::setw(18) << "v_mushy [m/s]"
              << std::setw(18) << "v_liquid [m/s]"
              << "\n";
    std::cout << std::string(62, '-') << "\n";

    float* h_ux = new float[n_cells];
    float* h_uy = new float[n_cells];
    float* h_uz = new float[n_cells];

    const int n_steps = 10;

    for (int step = 0; step <= n_steps; step++) {
        // Recompute Darcy damping based on current velocity
        cudaMemset(d_fx, 0, n_cells * sizeof(float));
        cudaMemset(d_fy, 0, n_cells * sizeof(float));
        cudaMemset(d_fz, 0, n_cells * sizeof(float));

        solver.applyDarcyDamping(d_liquid_fraction, C_darcy, d_fx, d_fy, d_fz);

        // FluidLBM step
        solver.computeMacroscopic();
        solver.collisionBGK(d_fx, d_fy, d_fz);
        solver.streaming();

        // Check velocity by region
        solver.copyVelocityToHost(h_ux, h_uy, h_uz);

        float v_solid = 0.0f, v_mushy = 0.0f, v_liquid = 0.0f;

        for (int iz = 0; iz < nz; iz++) {
            for (int iy = 0; iy < ny; iy++) {
                for (int ix = 0; ix < nx; ix++) {
                    int id = ix + iy * nx + iz * nx * ny;
                    float v = std::abs(h_ux[id]);

                    if (iz < nz/3) {
                        v_solid += v;
                    } else if (iz < 2*nz/3) {
                        v_mushy += v;
                    } else {
                        v_liquid += v;
                    }
                }
            }
        }

        v_solid /= count_solid;
        v_mushy /= count_mushy;
        v_liquid /= count_liquid;

        std::cout << std::setw(8) << step
                  << std::setw(18) << std::scientific << std::setprecision(4)
                  << v_solid
                  << std::setw(18) << v_mushy
                  << std::setw(18) << v_liquid
                  << "\n";
    }

    std::cout << "\n";

    // Final analysis
    solver.copyVelocityToHost(h_ux, h_uy, h_uz);

    float v_solid_final = 0.0f, v_mushy_final = 0.0f, v_liquid_final = 0.0f;

    for (int iz = 0; iz < nz; iz++) {
        for (int iy = 0; iy < ny; iy++) {
            for (int ix = 0; ix < nx; ix++) {
                int id = ix + iy * nx + iz * nx * ny;
                float v = std::abs(h_ux[id]);

                if (iz < nz/3) {
                    v_solid_final += v;
                } else if (iz < 2*nz/3) {
                    v_mushy_final += v;
                } else {
                    v_liquid_final += v;
                }
            }
        }
    }

    v_solid_final /= count_solid;
    v_mushy_final /= count_mushy;
    v_liquid_final /= count_liquid;

    std::cout << "================================================================\n";
    std::cout << "RESULTS:\n";
    std::cout << "  Initial velocity: " << u_initial << " m/s\n";
    std::cout << "  Final velocity (solid region): " << v_solid_final << " m/s\n";
    std::cout << "  Final velocity (mushy region): " << v_mushy_final << " m/s\n";
    std::cout << "  Final velocity (liquid region): " << v_liquid_final << " m/s\n\n";

    // Diagnosis
    std::cout << "DIAGNOSIS:\n";
    bool passed = false;

    if (v_liquid_final < 1e-3f) {
        std::cout << "  [FAIL] Velocity in LIQUID region is nearly zero!\n";
        std::cout << "  ROOT CAUSE: Darcy damping is killing flow EVERYWHERE.\n";
        std::cout << "\n";
        std::cout << "  Expected behavior:\n";
        std::cout << "    - Liquid (f=1): No damping, velocity should remain ~" << u_initial << " m/s\n";
        std::cout << "    - Solid (f=0): Strong damping, velocity should drop to ~0\n";
        std::cout << "\n";
        std::cout << "  Possible bugs:\n";
        std::cout << "  1. Formula error: Should be F = -C*(1-f)^2/(f^3+eps)*u\n";
        std::cout << "  2. Incorrect: F = -C*u (damping everywhere)\n";
        std::cout << "  3. Liquid fraction not read correctly\n";
        std::cout << "  4. Force magnitude too large (wrong units/scaling)\n";
    } else if (v_solid_final > 0.1f * u_initial) {
        std::cout << "  [FAIL] Velocity in SOLID region is still large!\n";
        std::cout << "  ROOT CAUSE: Darcy damping not strong enough in solid.\n";
        std::cout << "\n";
        std::cout << "  Solid region should have velocity near zero due to damping.\n";
    } else if (std::abs(v_liquid_final - u_initial) / u_initial > 0.2f) {
        std::cout << "  [WARNING] Velocity in liquid changed by >20%\n";
        std::cout << "  Expected: ~" << u_initial << " m/s\n";
        std::cout << "  Actual: " << v_liquid_final << " m/s\n";
        std::cout << "  May indicate weak damping in liquid (should be zero).\n";
    } else {
        std::cout << "  [PASS] Darcy damping works correctly!\n";
        std::cout << "  - Liquid region: velocity preserved (" << v_liquid_final << " m/s)\n";
        std::cout << "  - Solid region: velocity killed (" << v_solid_final << " m/s)\n";
        std::cout << "  - Mushy region: intermediate damping (" << v_mushy_final << " m/s)\n";
        std::cout << "\n";
        std::cout << "  CONCLUSION: Darcy damping implementation is correct.\n";
        std::cout << "  If v5 test has zero velocity, the problem is NOT Darcy.\n";
        std::cout << "  Check multiphysics coupling or buoyancy force.\n";
        passed = true;
    }

    // Cleanup
    delete[] h_liquid_fraction;
    delete[] h_fx;
    delete[] h_fy;
    delete[] h_fz;
    delete[] h_ux;
    delete[] h_uy;
    delete[] h_uz;

    cudaFree(d_liquid_fraction);
    cudaFree(d_fx);
    cudaFree(d_fy);
    cudaFree(d_fz);

    std::cout << "\n================================================================\n";
    std::cout << "  Test 3 Complete: " << (passed ? "PASS" : "FAIL") << "\n";
    std::cout << "================================================================\n";

    return passed ? 0 : 1;
}
