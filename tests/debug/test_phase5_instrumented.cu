/**
 * @file test_phase5_instrumented.cu
 * @brief TEST LEVEL 4: Instrumented reproduction of Phase 5 problem
 *
 * Purpose: Reproduce Phase 5 exactly but with detailed instrumentation
 *
 * Test Strategy:
 *   - Run exact Phase 5 setup but smaller/shorter
 *   - Log forces and velocities at each step
 *   - Track where values go to zero
 *
 * Expected Output:
 *   - Identify exact step where velocity becomes zero
 *   - Or identify that forces are zero from the start
 *
 * This will pinpoint: Buoyancy → Darcy → Collision → Streaming → Macroscopic
 */

#include <iostream>
#include <iomanip>
#include <cmath>
#include <algorithm>
#include <vector>
#include <cuda_runtime.h>

#include "physics/thermal_lbm.h"
#include "physics/fluid_lbm.h"
#include "physics/laser_source.h"
#include "physics/material_properties.h"
#include "core/lattice_d3q19.h"

using namespace lbm;

// Scaling kernel from Phase 5
__global__ void scaleForceArrayKernel(
    float* fx, float* fy, float* fz,
    float scale, int num_cells)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= num_cells) return;
    fx[id] *= scale;
    fy[id] *= scale;
    fz[id] *= scale;
}

int main() {
    std::cout << "\n========================================\n";
    std::cout << "TEST LEVEL 4: Phase 5 Instrumented\n";
    std::cout << "========================================\n\n";

    // Smaller domain for faster testing
    const int nx = 40, ny = 40, nz = 20;
    const int num_cells = nx * ny * nz;

    const float dx = 2e-6f;
    const float dy = 2e-6f;
    const float dz = 2e-6f;
    const float dt = 5e-10f;

    std::cout << "Test setup (same as Phase 5, but smaller):\n";
    std::cout << "  Grid: " << nx << "x" << ny << "x" << nz << "\n";
    std::cout << "  dx: " << dx * 1e6 << " um\n";
    std::cout << "  dt: " << dt * 1e9 << " ns\n\n";

    // Material
    physics::MaterialProperties ti64 = physics::MaterialDatabase::getTi6Al4V();
    float beta_thermal = 9.0e-6f;
    float alpha_thermal = ti64.getThermalDiffusivity(300.0f);
    float nu_lattice = 0.15f;
    float T_ref = 0.5f * (ti64.T_solidus + ti64.T_liquidus);
    float g = 9.81f;
    float darcy_constant = 1e5f;

    std::cout << "Physical parameters:\n";
    std::cout << "  T_ref: " << T_ref << " K\n";
    std::cout << "  Beta: " << beta_thermal * 1e6 << " × 10⁻⁶ K⁻¹\n";
    std::cout << "  Gravity: " << g << " m/s²\n";
    std::cout << "  Darcy: " << darcy_constant << " kg/(m³·s)\n";
    std::cout << "  Density: " << ti64.rho_liquid << " kg/m³\n\n";

    // Initialize
    core::D3Q19::initializeDevice();

    physics::ThermalLBM thermal(nx, ny, nz, ti64, alpha_thermal, true);
    thermal.initialize(300.0f);

    physics::FluidLBM fluid(nx, ny, nz, nu_lattice, ti64.rho_liquid,
                           physics::BoundaryType::WALL,
                           physics::BoundaryType::WALL,
                           physics::BoundaryType::PERIODIC);
    fluid.initialize(ti64.rho_liquid, 0.0f, 0.0f, 0.0f);

    // Laser
    float laser_power = 1200.0f;
    float spot_radius = 50e-6f;
    float penetration_depth = 15e-6f;
    LaserSource laser(laser_power, spot_radius, ti64.absorptivity_solid, penetration_depth);

    float Lx = nx * dx;
    float Ly = ny * dy;
    float Lz = nz * dz;
    laser.setPosition(Lx / 2.0f, Ly / 2.0f, 0.0f);

    // Allocate
    float *d_heat_source, *d_fx, *d_fy, *d_fz;
    cudaMalloc(&d_heat_source, num_cells * sizeof(float));
    cudaMalloc(&d_fx, num_cells * sizeof(float));
    cudaMalloc(&d_fy, num_cells * sizeof(float));
    cudaMalloc(&d_fz, num_cells * sizeof(float));

    cudaMemset(d_heat_source, 0, num_cells * sizeof(float));
    cudaMemset(d_fx, 0, num_cells * sizeof(float));
    cudaMemset(d_fy, 0, num_cells * sizeof(float));
    cudaMemset(d_fz, 0, num_cells * sizeof(float));

    float force_conversion = (dt * dt) / dx;
    float target_force_magnitude = 1e-4f;
    float estimated_phys_force = 200.0f;
    float force_scale = target_force_magnitude / (estimated_phys_force * force_conversion);

    std::cout << "Unit conversion:\n";
    std::cout << "  Force conversion factor: " << force_conversion << "\n";
    std::cout << "  Force scale: " << force_scale << "\n\n";

    // Host arrays
    std::vector<float> h_fx(num_cells);
    std::vector<float> h_fy(num_cells);
    std::vector<float> h_fz(num_cells);
    std::vector<float> h_ux(num_cells);
    std::vector<float> h_uy(num_cells);
    std::vector<float> h_uz(num_cells);
    std::vector<float> h_temperature(num_cells);
    std::vector<float> h_liquid_fraction(num_cells);

    // Run simulation with instrumentation
    const int n_steps = 1200;
    const int start_fluid = 600;

    std::cout << "Running instrumented simulation...\n";
    std::cout << "Fluid solver starts at step " << start_fluid << "\n\n";

    std::cout << "Step  | T_max | Melt% | max|f_buoy| | max|f_darcy| | max|f_final| | max|u|\n";
    std::cout << "------|-------|-------|--------------|---------------|---------------|-------\n";

    dim3 block(8, 8, 8);
    dim3 grid((nx + block.x - 1) / block.x,
             (ny + block.y - 1) / block.y,
             (nz + block.z - 1) / block.z);

    int block_size = 256;
    int grid_size = (num_cells + block_size - 1) / block_size;

    for (int step = 0; step <= n_steps; ++step) {
        // Thermal
        computeLaserHeatSourceKernel<<<grid, block>>>(
            d_heat_source, laser, dx, dy, dz, nx, ny, nz
        );
        cudaDeviceSynchronize();

        thermal.addHeatSource(d_heat_source, dt);
        thermal.collisionBGK();
        thermal.streaming();
        thermal.computeTemperature();

        // Fluid
        bool has_liquid = (step >= start_fluid);

        if (has_liquid) {
            // INSTRUMENTATION POINT 1: After buoyancy
            fluid.computeBuoyancyForce(
                thermal.getTemperature(), T_ref, beta_thermal,
                0.0f, g, 0.0f,
                d_fx, d_fy, d_fz
            );

            cudaMemcpy(h_fx.data(), d_fx, num_cells * sizeof(float), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_fy.data(), d_fy, num_cells * sizeof(float), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_fz.data(), d_fz, num_cells * sizeof(float), cudaMemcpyDeviceToHost);

            float max_f_buoy = *std::max_element(h_fy.begin(), h_fy.end(),
                                                 [](float a, float b) { return std::abs(a) < std::abs(b); });

            // INSTRUMENTATION POINT 2: After Darcy
            fluid.applyDarcyDamping(
                thermal.getLiquidFraction(), darcy_constant,
                d_fx, d_fy, d_fz
            );

            cudaMemcpy(h_fx.data(), d_fx, num_cells * sizeof(float), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_fy.data(), d_fy, num_cells * sizeof(float), cudaMemcpyDeviceToHost);

            float max_f_darcy = *std::max_element(h_fy.begin(), h_fy.end(),
                                                  [](float a, float b) { return std::abs(a) < std::abs(b); });

            // INSTRUMENTATION POINT 3: After scaling
            scaleForceArrayKernel<<<grid_size, block_size>>>(
                d_fx, d_fy, d_fz, force_scale, num_cells
            );
            cudaDeviceSynchronize();

            cudaMemcpy(h_fy.data(), d_fy, num_cells * sizeof(float), cudaMemcpyDeviceToHost);

            float max_f_final = *std::max_element(h_fy.begin(), h_fy.end(),
                                                  [](float a, float b) { return std::abs(a) < std::abs(b); });

            // Solve Navier-Stokes
            fluid.computeMacroscopic();
            fluid.collisionBGK(d_fx, d_fy, d_fz);
            fluid.streaming();

            // INSTRUMENTATION POINT 4: After collision/streaming
            fluid.computeMacroscopic();
            fluid.copyVelocityToHost(h_ux.data(), h_uy.data(), h_uz.data());

            float max_u = 0.0f;
            for (int i = 0; i < num_cells; ++i) {
                float u_mag = std::sqrt(h_ux[i]*h_ux[i] + h_uy[i]*h_uy[i] + h_uz[i]*h_uz[i]);
                max_u = std::max(max_u, u_mag);
            }

            // Get temperature stats
            thermal.copyTemperatureToHost(h_temperature.data());
            thermal.copyLiquidFractionToHost(h_liquid_fraction.data());

            float T_max = *std::max_element(h_temperature.begin(), h_temperature.end());
            int n_melting = std::count_if(h_liquid_fraction.begin(), h_liquid_fraction.end(),
                                         [](float fl) { return fl > 0.01f; });
            float melt_pct = 100.0f * n_melting / num_cells;

            // Print instrumentation
            if (step % 100 == 0) {
                std::cout << std::setw(5) << step << " | "
                          << std::setw(5) << std::fixed << std::setprecision(0) << T_max << " | "
                          << std::setw(5) << std::setprecision(1) << melt_pct << " | "
                          << std::setw(12) << std::scientific << std::setprecision(2) << max_f_buoy << " | "
                          << std::setw(13) << max_f_darcy << " | "
                          << std::setw(13) << max_f_final << " | "
                          << std::setw(6) << max_u << "\n";
            }

            // CRITICAL CHECK: If forces go to zero
            if (std::abs(max_f_final) < 1e-10f && step > start_fluid + 10) {
                std::cout << "\n⚠️  CRITICAL: Forces went to ZERO at step " << step << "!\n";
                std::cout << "  Buoyancy force before scaling: " << max_f_buoy << "\n";
                std::cout << "  Force after Darcy: " << max_f_darcy << "\n";
                std::cout << "  Force after scaling: " << max_f_final << "\n";
                std::cout << "  → BUG LOCATION IDENTIFIED\n\n";
                break;
            }

            // CRITICAL CHECK: If velocity is zero despite forces
            if (max_u < 1e-10f && std::abs(max_f_final) > 1e-8f && step > start_fluid + 100) {
                std::cout << "\n⚠️  CRITICAL: Velocity is ZERO despite non-zero forces at step " << step << "!\n";
                std::cout << "  Max force: " << max_f_final << "\n";
                std::cout << "  Max velocity: " << max_u << "\n";
                std::cout << "  → BUG: Force not translating to velocity\n";
                std::cout << "  → Likely issue in collision kernel or macroscopic computation\n\n";
                break;
            }

        } else {
            fluid.computeMacroscopic();
        }
    }

    std::cout << "\n=== VERDICT ===\n";
    std::cout << "Check the instrumentation output above to identify:\n";
    std::cout << "1. If max|f_buoy| is zero → buoyancy kernel failed\n";
    std::cout << "2. If max|f_darcy| drops to near-zero → Darcy damping too strong\n";
    std::cout << "3. If max|f_final| is zero → scaling issue\n";
    std::cout << "4. If max|u| is zero despite forces → collision/macroscopic issue\n";

    // Cleanup
    cudaFree(d_heat_source);
    cudaFree(d_fx);
    cudaFree(d_fy);
    cudaFree(d_fz);

    return 0;
}
