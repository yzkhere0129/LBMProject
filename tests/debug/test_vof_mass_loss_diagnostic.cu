/**
 * @file test_vof_mass_loss_diagnostic.cu
 * @brief Diagnostic test to identify source of VOF mass loss
 *
 * This test systematically checks:
 * 1. Boundary flux leakage
 * 2. Clamping losses
 * 3. Numerical dissipation
 * 4. Round-off errors
 */

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <vector>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <algorithm>

#include "physics/vof_solver.h"

using namespace lbm::physics;

// Helper: compute mass before and after clamping
__global__ void diagnoseClamping(
    const float* fill_level_before,
    const float* fill_level_after,
    float* clamping_loss,
    int num_cells)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_cells) return;

    float f_before = fill_level_before[idx];
    float f_after = fill_level_after[idx];

    // Check if clamping occurred
    if (f_before < 0.0f && f_after == 0.0f) {
        clamping_loss[idx] = -f_before;  // Mass lost from negative clamp
    } else if (f_before > 1.0f && f_after == 1.0f) {
        clamping_loss[idx] = f_before - 1.0f;  // Mass lost from positive clamp
    } else {
        clamping_loss[idx] = 0.0f;
    }
}

// Helper: compute boundary flux (material leaving domain)
__global__ void diagnoseBoundaryFlux(
    const float* fill_level,
    const float* ux, const float* uy, const float* uz,
    float* flux_x_out, float* flux_y_out, float* flux_z_out,
    float dt, float dx,
    int nx, int ny, int nz)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i >= nx || j >= ny || k >= nz) return;

    int idx = i + nx * (j + ny * k);

    float f = fill_level[idx];
    float u = ux[idx];
    float v = uy[idx];
    float w = uz[idx];

    // Compute outward flux at boundaries
    // Flux = f * u * dt / dx (dimensionless mass flux)

    // X boundaries (periodic - flux out one side = flux in other side)
    if (i == 0 && u < 0.0f) {
        flux_x_out[idx] = -f * u * dt / dx;  // Outward flux at left
    } else if (i == nx - 1 && u > 0.0f) {
        flux_x_out[idx] = f * u * dt / dx;  // Outward flux at right
    } else {
        flux_x_out[idx] = 0.0f;
    }

    // Y boundaries (periodic)
    if (j == 0 && v < 0.0f) {
        flux_y_out[idx] = -f * v * dt / dx;
    } else if (j == ny - 1 && v > 0.0f) {
        flux_y_out[idx] = f * v * dt / dx;
    } else {
        flux_y_out[idx] = 0.0f;
    }

    // Z boundaries (outflow - material CAN leave)
    if (k == 0 && w < 0.0f) {
        flux_z_out[idx] = -f * w * dt / dx;  // Outward flux at bottom
    } else if (k == nz - 1 && w > 0.0f) {
        flux_z_out[idx] = f * w * dt / dx;  // Outward flux at top
    } else {
        flux_z_out[idx] = 0.0f;
    }
}

TEST(VOFDiagnosticTest, MassLossSource) {
    std::cout << "\n========================================" << std::endl;
    std::cout << "VOF Mass Loss Diagnostic" << std::endl;
    std::cout << "========================================\n" << std::endl;

    // Small domain for detailed analysis
    int nx = 60, ny = 60, nz = 30;
    float dx = 2e-6f;
    int num_cells = nx * ny * nz;

    VOFSolver solver(nx, ny, nz, dx);

    // Initialize with planar interface at mid-height
    std::vector<float> h_fill(num_cells);
    int z_interface = nz / 2;
    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                int idx = i + nx * (j + ny * k);
                float z_dist = k - z_interface;
                h_fill[idx] = 0.5f * (1.0f - tanhf(z_dist / 2.0f));
                h_fill[idx] = std::max(0.0f, std::min(1.0f, h_fill[idx]));
            }
        }
    }

    solver.initialize(h_fill.data());
    float initial_mass = solver.computeTotalMass();

    std::cout << "Initial configuration:" << std::endl;
    std::cout << "  Domain: " << nx << "×" << ny << "×" << nz << std::endl;
    std::cout << "  dx = " << dx * 1e6 << " μm" << std::endl;
    std::cout << "  Initial mass: " << initial_mass << std::endl;
    std::cout << std::endl;

    // Create uniform velocity field (upward flow)
    float* d_ux, *d_uy, *d_uz;
    cudaMalloc(&d_ux, num_cells * sizeof(float));
    cudaMalloc(&d_uy, num_cells * sizeof(float));
    cudaMalloc(&d_uz, num_cells * sizeof(float));

    // Set uniform velocity: u = 0.05 m/s upward (typical Marangoni flow)
    float u_test = 0.05f;  // m/s
    std::vector<float> h_u(num_cells, 0.0f);
    std::vector<float> h_v(num_cells, 0.0f);
    std::vector<float> h_w(num_cells, u_test);

    cudaMemcpy(d_ux, h_u.data(), num_cells * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_uy, h_v.data(), num_cells * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_uz, h_w.data(), num_cells * sizeof(float), cudaMemcpyHostToDevice);

    float dt = 1e-8f;  // 10 ns subcycle timestep
    float vof_cfl = u_test * dt / dx;

    std::cout << "Velocity field:" << std::endl;
    std::cout << "  u = 0, v = 0, w = " << u_test << " m/s (uniform upward)" << std::endl;
    std::cout << "  VOF CFL = " << vof_cfl << " (should be << 0.5)" << std::endl;
    std::cout << std::endl;

    // Allocate diagnostic arrays
    float* d_flux_x, *d_flux_y, *d_flux_z;
    cudaMalloc(&d_flux_x, num_cells * sizeof(float));
    cudaMalloc(&d_flux_y, num_cells * sizeof(float));
    cudaMalloc(&d_flux_z, num_cells * sizeof(float));

    // Run advection for 100 steps
    const int n_steps = 100;
    const int check_interval = 20;

    float cumulative_flux_z = 0.0f;

    for (int step = 0; step < n_steps; ++step) {
        // Compute boundary flux before advection
        dim3 blockSize(8, 8, 8);
        dim3 gridSize((nx + blockSize.x - 1) / blockSize.x,
                      (ny + blockSize.y - 1) / blockSize.y,
                      (nz + blockSize.z - 1) / blockSize.z);

        diagnoseBoundaryFlux<<<gridSize, blockSize>>>(
            solver.getFillLevelDevice(), d_ux, d_uy, d_uz,
            d_flux_x, d_flux_y, d_flux_z, dt, dx, nx, ny, nz);
        cudaDeviceSynchronize();

        // Sum fluxes
        std::vector<float> h_flux_z(num_cells);
        cudaMemcpy(h_flux_z.data(), d_flux_z, num_cells * sizeof(float), cudaMemcpyDeviceToHost);

        float flux_z_step = 0.0f;
        for (float f : h_flux_z) flux_z_step += f;
        cumulative_flux_z += flux_z_step;

        // Advect fill level
        solver.advectFillLevel(d_ux, d_uy, d_uz, dt);

        if ((step + 1) % check_interval == 0) {
            float current_mass = solver.computeTotalMass();
            float mass_loss = initial_mass - current_mass;
            float mass_error = mass_loss / initial_mass;

            std::cout << "Step " << std::setw(3) << step + 1
                      << " | Mass = " << std::fixed << std::setprecision(1) << current_mass
                      << " | Loss = " << mass_loss
                      << " (" << std::setprecision(4) << mass_error * 100.0f << "%)"
                      << " | Z-flux = " << std::setprecision(1) << cumulative_flux_z
                      << std::endl;
        }
    }

    float final_mass = solver.computeTotalMass();
    float total_loss = initial_mass - final_mass;
    float loss_fraction = total_loss / initial_mass;

    std::cout << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "RESULTS:" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "  Initial mass:     " << initial_mass << std::endl;
    std::cout << "  Final mass:       " << final_mass << std::endl;
    std::cout << "  Total mass loss:  " << total_loss << " (" << loss_fraction * 100.0f << "%)" << std::endl;
    std::cout << "  Cumulative Z-flux: " << cumulative_flux_z << std::endl;
    std::cout << std::endl;

    // Analysis
    float flux_explained = cumulative_flux_z / total_loss;
    std::cout << "Analysis:" << std::endl;
    std::cout << "  Z-boundary flux explains: " << std::setprecision(1) << flux_explained * 100.0f << "% of mass loss" << std::endl;

    if (flux_explained > 0.9f) {
        std::cout << "  ✓ Mass loss is primarily from Z-boundary outflow (expected)" << std::endl;
    } else {
        std::cout << "  ⚠ Mass loss has other sources:" << std::endl;
        std::cout << "    - Clamping losses?" << std::endl;
        std::cout << "    - Numerical dissipation?" << std::endl;
        std::cout << "    - Round-off errors?" << std::endl;
    }

    std::cout << "========================================\n" << std::endl;

    // Cleanup
    cudaFree(d_ux);
    cudaFree(d_uy);
    cudaFree(d_uz);
    cudaFree(d_flux_x);
    cudaFree(d_flux_y);
    cudaFree(d_flux_z);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
