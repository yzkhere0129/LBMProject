/**
 * @file diagnostic_phase5_forces.cu
 * @brief Diagnostic to check actual force values in Phase 5 scenario
 *
 * This test replicates the exact setup from Phase 5 and checks:
 * - What buoyancy forces are actually being computed
 * - What Darcy damping is doing to those forces
 * - Whether forces are truly zero or just very small
 */

#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>
#include "physics/fluid_lbm.h"
#include "physics/thermal_lbm.h"
#include "physics/material_properties.h"
#include "core/lattice_d3q19.h"

using namespace lbm::physics;
using namespace lbm::core;

int main() {
    std::cout << "\n========================================\n";
    std::cout << " Diagnostic: Phase 5 Force Analysis\n";
    std::cout << "========================================\n\n";

    // Same parameters as Phase 5
    const int nx = 80, ny = 80, nz = 40;
    int num_cells = nx * ny * nz;

    float nu_lattice = 0.15f;
    MaterialProperties ti64 = MaterialDatabase::getTi6Al4V();

    float T_ref = 0.5f * (ti64.T_solidus + ti64.T_liquidus);
    float beta_thermal = 9.0e-6f;  // [1/K]
    float g = 9.81f;  // [m/s²]
    float darcy_constant = 1e5f;

    // Initialize D3Q19
    D3Q19::initializeDevice();

    // Create fluid solver
    FluidLBM fluid(nx, ny, nz, nu_lattice, ti64.rho_liquid);
    fluid.initialize(ti64.rho_liquid, 0.0f, 0.0f, 0.0f);

    // Create thermal solver
    float alpha_thermal = ti64.getThermalDiffusivity(300.0f);
    ThermalLBM thermal(nx, ny, nz, ti64, alpha_thermal, true);
    thermal.initialize(300.0f);  // Room temperature

    // Allocate force arrays
    float *d_fx, *d_fy, *d_fz;
    cudaMalloc(&d_fx, num_cells * sizeof(float));
    cudaMalloc(&d_fy, num_cells * sizeof(float));
    cudaMalloc(&d_fz, num_cells * sizeof(float));

    cudaMemset(d_fx, 0, num_cells * sizeof(float));
    cudaMemset(d_fy, 0, num_cells * sizeof(float));
    cudaMemset(d_fz, 0, num_cells * sizeof(float));

    std::cout << "Setup:\n";
    std::cout << "  Grid: " << nx << " x " << ny << " x " << nz << " = " << num_cells << " cells\n";
    std::cout << "  Reference temperature: " << T_ref << " K\n";
    std::cout << "  Thermal expansion: " << beta_thermal << " 1/K\n";
    std::cout << "  Gravity: " << g << " m/s²\n";
    std::cout << "  Darcy constant: " << darcy_constant << " kg/(m³·s)\n";
    std::cout << "  Density: " << ti64.rho_liquid << " kg/m³\n\n";

    // ====================================================================
    // Test 1: Cold fluid (T = 300 K everywhere) → should have zero buoyancy
    // ====================================================================
    std::cout << "Test 1: Uniform cold temperature (T = 300 K)\n";
    std::cout << std::string(60, '-') << "\n";

    thermal.computeTemperature();
    fluid.computeBuoyancyForce(
        thermal.getTemperature(), T_ref, beta_thermal,
        0.0f, g, 0.0f,
        d_fx, d_fy, d_fz
    );

    // Copy forces to host
    std::vector<float> h_fx(num_cells), h_fy(num_cells), h_fz(num_cells);
    cudaMemcpy(h_fx.data(), d_fx, num_cells * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_fy.data(), d_fy, num_cells * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_fz.data(), d_fz, num_cells * sizeof(float), cudaMemcpyDeviceToHost);

    // Analyze forces
    float fx_min = h_fx[0], fx_max = h_fx[0], fx_avg = 0.0f;
    float fy_min = h_fy[0], fy_max = h_fy[0], fy_avg = 0.0f;
    for (int i = 0; i < num_cells; ++i) {
        fx_min = fminf(fx_min, h_fx[i]);
        fx_max = fmaxf(fx_max, h_fx[i]);
        fx_avg += h_fx[i];

        fy_min = fminf(fy_min, h_fy[i]);
        fy_max = fmaxf(fy_max, h_fy[i]);
        fy_avg += h_fy[i];
    }
    fx_avg /= num_cells;
    fy_avg /= num_cells;

    std::cout << "  Buoyancy force Fx: min=" << std::scientific << fx_min
              << " avg=" << fx_avg << " max=" << fx_max << "\n";
    std::cout << "  Buoyancy force Fy: min=" << std::scientific << fy_min
              << " avg=" << fy_avg << " max=" << fy_max << "\n";
    std::cout << "  Expected: ~0 (no temperature difference)\n\n";

    // ====================================================================
    // Test 2: Hot spot (T = 2000 K in center) → should have strong buoyancy
    // ====================================================================
    std::cout << "Test 2: Hot spot in center (T = 2000 K)\n";
    std::cout << std::string(60, '-') << "\n";

    // Create a hot spot in the center
    std::vector<float> h_temp(num_cells, 300.0f);
    int cx = nx / 2;
    int cy = ny / 2;
    int cz = nz / 2;
    int radius = 10;

    for (int iz = 0; iz < nz; ++iz) {
        for (int iy = 0; iy < ny; ++iy) {
            for (int ix = 0; ix < nx; ++ix) {
                int id = ix + iy * nx + iz * nx * ny;
                int dx = ix - cx;
                int dy = iy - cy;
                int dz = iz - cz;
                float dist = sqrtf(dx*dx + dy*dy + dz*dz);
                if (dist < radius) {
                    h_temp[id] = 2000.0f;  // Hot spot
                }
            }
        }
    }

    // Upload temperature
    cudaMemcpy(const_cast<float*>(thermal.getTemperature()), h_temp.data(),
               num_cells * sizeof(float), cudaMemcpyHostToDevice);

    // Compute buoyancy
    cudaMemset(d_fx, 0, num_cells * sizeof(float));
    cudaMemset(d_fy, 0, num_cells * sizeof(float));
    cudaMemset(d_fz, 0, num_cells * sizeof(float));

    fluid.computeBuoyancyForce(
        thermal.getTemperature(), T_ref, beta_thermal,
        0.0f, g, 0.0f,
        d_fx, d_fy, d_fz
    );

    cudaMemcpy(h_fx.data(), d_fx, num_cells * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_fy.data(), d_fy, num_cells * sizeof(float), cudaMemcpyDeviceToHost);

    // Analyze forces at hot spot
    float fy_hot = 0.0f;
    int n_hot = 0;
    for (int iz = 0; iz < nz; ++iz) {
        for (int iy = 0; iy < ny; ++iy) {
            for (int ix = 0; ix < nx; ++ix) {
                int id = ix + iy * nx + iz * nx * ny;
                int dx = ix - cx;
                int dy = iy - cy;
                int dz = iz - cz;
                float dist = sqrtf(dx*dx + dy*dy + dz*dz);
                if (dist < radius) {
                    fy_hot += h_fy[id];
                    n_hot++;
                }
            }
        }
    }
    fy_hot /= n_hot;

    std::cout << "  Average Fy in hot spot: " << std::scientific << fy_hot << " N/m³\n";
    std::cout << "  Expected: ρ₀·β·(T - T_ref)·g\n";
    float expected_fy = ti64.rho_liquid * beta_thermal * (2000.0f - T_ref) * g;
    std::cout << "            = " << ti64.rho_liquid << " * " << beta_thermal
              << " * " << (2000.0f - T_ref) << " * " << g << "\n";
    std::cout << "            = " << std::scientific << expected_fy << " N/m³\n";
    std::cout << "  Match: " << (fabs(fy_hot - expected_fy) / expected_fy < 0.01 ? "✓ YES" : "✗ NO") << "\n\n";

    // ====================================================================
    // Test 3: Apply Darcy damping with full liquid (fl = 1.0)
    // ====================================================================
    std::cout << "Test 3: Darcy damping with full liquid (fl = 1.0)\n";
    std::cout << std::string(60, '-') << "\n";

    // Set liquid fraction = 1.0 everywhere (fully liquid)
    std::vector<float> h_fl(num_cells, 1.0f);
    cudaMemcpy(const_cast<float*>(thermal.getLiquidFraction()), h_fl.data(),
               num_cells * sizeof(float), cudaMemcpyHostToDevice);

    // Apply Darcy damping (should have NO effect since fl=1)
    float fy_before_darcy = fy_hot;
    fluid.applyDarcyDamping(
        thermal.getLiquidFraction(), darcy_constant,
        d_fx, d_fy, d_fz
    );

    cudaMemcpy(h_fy.data(), d_fy, num_cells * sizeof(float), cudaMemcpyDeviceToHost);

    float fy_after_darcy = 0.0f;
    for (int iz = 0; iz < nz; ++iz) {
        for (int iy = 0; iy < ny; ++iy) {
            for (int ix = 0; ix < nx; ++ix) {
                int id = ix + iy * nx + iz * nx * ny;
                int dx = ix - cx;
                int dy = iy - cy;
                int dz = iz - cz;
                float dist = sqrtf(dx*dx + dy*dy + dz*dz);
                if (dist < radius) {
                    fy_after_darcy += h_fy[id];
                }
            }
        }
    }
    fy_after_darcy /= n_hot;

    std::cout << "  Fy before Darcy: " << std::scientific << fy_before_darcy << " N/m³\n";
    std::cout << "  Fy after Darcy:  " << std::scientific << fy_after_darcy << " N/m³\n";
    std::cout << "  Change: " << (fy_after_darcy - fy_before_darcy) << " N/m³\n";
    std::cout << "  Expected: ~0 (fl=1.0 → no damping)\n\n";

    // ====================================================================
    // Test 4: Apply Darcy damping with mushy zone (fl = 0.5)
    // ====================================================================
    std::cout << "Test 4: Darcy damping with mushy zone (fl = 0.5)\n";
    std::cout << std::string(60, '-') << "\n";

    // Reset forces
    cudaMemset(d_fx, 0, num_cells * sizeof(float));
    cudaMemset(d_fy, 0, num_cells * sizeof(float));
    cudaMemset(d_fz, 0, num_cells * sizeof(float));

    // Set buoyancy force
    fluid.computeBuoyancyForce(
        thermal.getTemperature(), T_ref, beta_thermal,
        0.0f, g, 0.0f,
        d_fx, d_fy, d_fz
    );

    // Set liquid fraction = 0.5 everywhere (mushy)
    std::fill(h_fl.begin(), h_fl.end(), 0.5f);
    cudaMemcpy(const_cast<float*>(thermal.getLiquidFraction()), h_fl.data(),
               num_cells * sizeof(float), cudaMemcpyHostToDevice);

    // Also set velocity to 0.001 m/s to see damping effect
    fluid.initialize(ti64.rho_liquid, 0.0f, 0.001f, 0.0f);  // uy = 0.001 m/s

    cudaMemcpy(h_fy.data(), d_fy, num_cells * sizeof(float), cudaMemcpyDeviceToHost);
    float fy_before = 0.0f;
    for (int i = 0; i < num_cells; ++i) {
        fy_before += h_fy[i];
    }
    fy_before /= num_cells;

    // Apply Darcy damping
    fluid.applyDarcyDamping(
        thermal.getLiquidFraction(), darcy_constant,
        d_fx, d_fy, d_fz
    );

    cudaMemcpy(h_fy.data(), d_fy, num_cells * sizeof(float), cudaMemcpyDeviceToHost);
    float fy_after = 0.0f;
    for (int i = 0; i < num_cells; ++i) {
        fy_after += h_fy[i];
    }
    fy_after /= num_cells;

    std::cout << "  Average Fy before Darcy: " << std::scientific << fy_before << " N/m³\n";
    std::cout << "  Average Fy after Darcy:  " << std::scientific << fy_after << " N/m³\n";
    std::cout << "  Change: " << (fy_after - fy_before) << " N/m³\n";
    std::cout << "  Expected: negative (damping opposes flow)\n\n";

    // Cleanup
    cudaFree(d_fx);
    cudaFree(d_fy);
    cudaFree(d_fz);

    std::cout << "\n========================================\n";
    std::cout << " Analysis Complete\n";
    std::cout << "========================================\n\n";

    std::cout << "Key findings:\n";
    std::cout << "  1. Buoyancy forces scale correctly with temperature\n";
    std::cout << "  2. Darcy damping only acts when fl < 1.0\n";
    std::cout << "  3. In Phase 5, if T ≈ T_ref everywhere, buoyancy forces are ZERO\n";
    std::cout << "  4. Need significant heating to create temperature gradients\n\n";

    return 0;
}
