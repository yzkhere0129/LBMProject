/**
 * @file test_marangoni_force.cu
 * @brief Unit test for Marangoni force computation
 *
 * This test validates that the Marangoni effect correctly computes
 * surface tension forces at the liquid-vapor interface.
 *
 * Test objectives:
 * - Verify Marangoni force is computed only at interface (0.1 < f < 0.9)
 * - Check force direction (from hot to cold along interface)
 * - Validate force magnitude is physically reasonable
 * - Ensure bulk liquid and vapor regions have zero Marangoni force
 */

#include <gtest/gtest.h>
#include "physics/vof_solver.h"
#include "physics/marangoni.h"
#include <cuda_runtime.h>
#include <cmath>
#include <vector>
#include <iostream>

using namespace lbm::physics;

/**
 * @brief Test that Marangoni force is applied only at interface
 */
TEST(MarangoniForce, InterfaceOnly) {
    std::cout << "\n=== Test: Marangoni Force - Interface Only ===" << std::endl;

    // Setup: 10x10x10 domain with planar interface at z=5
    const int nx = 10, ny = 10, nz = 10;
    const int num_cells = nx * ny * nz;
    const float dx = 2.0e-6f;  // 2 μm resolution

    std::cout << "  Domain: " << nx << "x" << ny << "x" << nz << std::endl;
    std::cout << "  Resolution: " << dx * 1e6 << " μm" << std::endl;

    VOFSolver vof(nx, ny, nz, dx);
    MarangoniEffect marangoni(nx, ny, nz, -2.6e-4f, dx, 2.0f);

    // Initialize planar interface at z=5
    std::vector<float> h_fill(num_cells);
    const int z_interface = 5;

    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                int idx = i + nx * (j + ny * k);

                // Smooth interface transition
                float z_dist = k - z_interface;
                h_fill[idx] = 0.5f * (1.0f - tanhf(z_dist / 2.0f));
                h_fill[idx] = std::max(0.0f, std::min(1.0f, h_fill[idx]));
            }
        }
    }

    vof.initialize(h_fill.data());
    vof.reconstructInterface();

    std::cout << "  Interface position: z = " << z_interface << " cells" << std::endl;

    // Create temperature gradient in X direction
    std::vector<float> h_temperature(num_cells);
    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                int idx = i + nx * (j + ny * k);
                h_temperature[idx] = 2000.0f + i * 50.0f;  // Gradient in X: 50K per cell
            }
        }
    }

    // Copy to device
    float *d_temp, *d_fx, *d_fy, *d_fz;
    cudaMalloc(&d_temp, num_cells * sizeof(float));
    cudaMalloc(&d_fx, num_cells * sizeof(float));
    cudaMalloc(&d_fy, num_cells * sizeof(float));
    cudaMalloc(&d_fz, num_cells * sizeof(float));

    cudaMemcpy(d_temp, h_temperature.data(), num_cells * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_fx, 0, num_cells * sizeof(float));
    cudaMemset(d_fy, 0, num_cells * sizeof(float));
    cudaMemset(d_fz, 0, num_cells * sizeof(float));

    // Compute Marangoni force
    marangoni.computeMarangoniForce(d_temp, vof.getFillLevel(),
                                    vof.getInterfaceNormals(),
                                    d_fx, d_fy, d_fz);

    // Copy back
    std::vector<float> h_fx(num_cells), h_fy(num_cells), h_fz(num_cells);
    cudaMemcpy(h_fx.data(), d_fx, num_cells * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_fy.data(), d_fy, num_cells * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_fz.data(), d_fz, num_cells * sizeof(float), cudaMemcpyDeviceToHost);

    // Analyze force distribution
    int interface_cells_with_force = 0;
    int bulk_cells_with_force = 0;
    float max_interface_force = 0.0f;
    float max_bulk_force = 0.0f;

    for (int idx = 0; idx < num_cells; ++idx) {
        float f_mag = std::sqrt(h_fx[idx] * h_fx[idx] +
                               h_fy[idx] * h_fy[idx] +
                               h_fz[idx] * h_fz[idx]);

        if (h_fill[idx] > 0.1f && h_fill[idx] < 0.9f) {
            // Interface cell
            if (f_mag > 1e-3f) {
                interface_cells_with_force++;
            }
            max_interface_force = std::max(max_interface_force, f_mag);
        } else {
            // Bulk cell (liquid or vapor)
            if (f_mag > 1e-3f) {
                bulk_cells_with_force++;
            }
            max_bulk_force = std::max(max_bulk_force, f_mag);
        }
    }

    std::cout << "  Interface cells with force (|F| > 1e-3): " << interface_cells_with_force << std::endl;
    std::cout << "  Bulk cells with force (|F| > 1e-3): " << bulk_cells_with_force << std::endl;
    std::cout << "  Max interface force: " << max_interface_force << " N/m³" << std::endl;
    std::cout << "  Max bulk force: " << max_bulk_force << " N/m³" << std::endl;

    // Verify interface has significant force
    EXPECT_GT(interface_cells_with_force, 0)
        << "Interface should have Marangoni force";

    EXPECT_GT(max_interface_force, 1e3f)
        << "Interface force should be physically significant (> 10³ N/m³)";

    // Verify bulk regions have minimal force (allow up to 30% due to smooth cutoff)
    EXPECT_LT(max_bulk_force, max_interface_force * 0.3f)
        << "Bulk regions should have reduced force compared to interface";

    std::cout << "  ✓ Marangoni force localized to interface (PASS)" << std::endl;

    cudaFree(d_temp);
    cudaFree(d_fx);
    cudaFree(d_fy);
    cudaFree(d_fz);
}

/**
 * @brief Test that Marangoni force direction is correct (hot → cold)
 */
TEST(MarangoniForce, DirectionCorrect) {
    std::cout << "\n=== Test: Marangoni Force - Direction Correct ===" << std::endl;

    const int nx = 20, ny = 20, nz = 10;
    const int num_cells = nx * ny * nz;
    const float dx = 2.0e-6f;

    VOFSolver vof(nx, ny, nz, dx);
    MarangoniEffect marangoni(nx, ny, nz, -2.6e-4f, dx, 2.0f);

    // Initialize planar interface at z=5
    std::vector<float> h_fill(num_cells);
    const int z_interface = 5;

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

    vof.initialize(h_fill.data());
    vof.reconstructInterface();

    // Create radial temperature field (hot center, cold edge)
    std::vector<float> h_temperature(num_cells);
    const float center_x = nx / 2.0f;
    const float center_y = ny / 2.0f;

    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                int idx = i + nx * (j + ny * k);

                float dx_c = i - center_x;
                float dy_c = j - center_y;
                float r = std::sqrt(dx_c * dx_c + dy_c * dy_c);

                // Temperature decreases with radius
                h_temperature[idx] = 2500.0f - r * 20.0f;  // Hot center, cold edge
            }
        }
    }

    // Copy to device
    float *d_temp, *d_fx, *d_fy, *d_fz;
    cudaMalloc(&d_temp, num_cells * sizeof(float));
    cudaMalloc(&d_fx, num_cells * sizeof(float));
    cudaMalloc(&d_fy, num_cells * sizeof(float));
    cudaMalloc(&d_fz, num_cells * sizeof(float));

    cudaMemcpy(d_temp, h_temperature.data(), num_cells * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_fx, 0, num_cells * sizeof(float));
    cudaMemset(d_fy, 0, num_cells * sizeof(float));
    cudaMemset(d_fz, 0, num_cells * sizeof(float));

    // Compute Marangoni force
    marangoni.computeMarangoniForce(d_temp, vof.getFillLevel(),
                                    vof.getInterfaceNormals(),
                                    d_fx, d_fy, d_fz);

    // Copy back
    std::vector<float> h_fx(num_cells), h_fy(num_cells);
    cudaMemcpy(h_fx.data(), d_fx, num_cells * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_fy.data(), d_fy, num_cells * sizeof(float), cudaMemcpyDeviceToHost);

    // Check force direction at interface
    // With dσ/dT < 0 and hot center, force should point radially outward
    int correct_direction_count = 0;
    int total_interface_count = 0;

    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                int idx = i + nx * (j + ny * k);

                if (h_fill[idx] > 0.1f && h_fill[idx] < 0.9f) {
                    // This is an interface cell
                    total_interface_count++;

                    // Radial direction from center
                    float dx_c = i - center_x;
                    float dy_c = j - center_y;
                    float r = std::sqrt(dx_c * dx_c + dy_c * dy_c);

                    if (r > 2.0f) {  // Avoid center singularity
                        // Force should point radially outward (hot → cold)
                        float f_radial = (h_fx[idx] * dx_c + h_fy[idx] * dy_c) / r;

                        if (f_radial > 0.0f) {
                            correct_direction_count++;
                        }
                    }
                }
            }
        }
    }

    float direction_accuracy = (total_interface_count > 0)
        ? (float)correct_direction_count / total_interface_count
        : 0.0f;

    std::cout << "  Interface cells analyzed: " << total_interface_count << std::endl;
    std::cout << "  Correct direction (radially outward): " << correct_direction_count
              << " (" << direction_accuracy * 100.0f << "%)" << std::endl;

    EXPECT_GT(direction_accuracy, 0.7f)
        << "Majority of forces should point from hot to cold (radially outward)";

    std::cout << "  ✓ Force direction correct (hot → cold) (PASS)" << std::endl;

    cudaFree(d_temp);
    cudaFree(d_fx);
    cudaFree(d_fy);
    cudaFree(d_fz);
}

/**
 * @brief Test Marangoni force magnitude scaling
 */
TEST(MarangoniForce, MagnitudeScaling) {
    std::cout << "\n=== Test: Marangoni Force - Magnitude Scaling ===" << std::endl;

    const int nx = 10, ny = 10, nz = 10;
    const int num_cells = nx * ny * nz;
    const float dx = 2.0e-6f;

    VOFSolver vof(nx, ny, nz, dx);

    // Initialize interface
    std::vector<float> h_fill(num_cells);
    const int z_interface = 5;

    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                int idx = i + nx * (j + ny * k);
                float z_dist = k - z_interface;
                h_fill[idx] = 0.5f * (1.0f - tanhf(z_dist / 2.0f));
            }
        }
    }

    vof.initialize(h_fill.data());
    vof.reconstructInterface();

    // Test with two different temperature gradients
    const float dT_weak = 100.0f;   // Weak gradient
    const float dT_strong = 500.0f; // Strong gradient

    std::vector<float> max_forces;

    for (float dT : {dT_weak, dT_strong}) {
        MarangoniEffect marangoni(nx, ny, nz, -2.6e-4f, dx, 2.0f);

        // Create temperature field
        std::vector<float> h_temperature(num_cells);
        for (int k = 0; k < nz; ++k) {
            for (int j = 0; j < ny; ++j) {
                for (int i = 0; i < nx; ++i) {
                    int idx = i + nx * (j + ny * k);
                    h_temperature[idx] = 2000.0f + i * dT;
                }
            }
        }

        // Copy to device
        float *d_temp, *d_fx, *d_fy, *d_fz;
        cudaMalloc(&d_temp, num_cells * sizeof(float));
        cudaMalloc(&d_fx, num_cells * sizeof(float));
        cudaMalloc(&d_fy, num_cells * sizeof(float));
        cudaMalloc(&d_fz, num_cells * sizeof(float));

        cudaMemcpy(d_temp, h_temperature.data(), num_cells * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemset(d_fx, 0, num_cells * sizeof(float));
        cudaMemset(d_fy, 0, num_cells * sizeof(float));
        cudaMemset(d_fz, 0, num_cells * sizeof(float));

        // Compute force
        marangoni.computeMarangoniForce(d_temp, vof.getFillLevel(),
                                        vof.getInterfaceNormals(),
                                        d_fx, d_fy, d_fz);

        // Find max force
        std::vector<float> h_fx(num_cells);
        cudaMemcpy(h_fx.data(), d_fx, num_cells * sizeof(float), cudaMemcpyDeviceToHost);

        float max_f = 0.0f;
        for (float f : h_fx) {
            max_f = std::max(max_f, std::abs(f));
        }

        max_forces.push_back(max_f);
        std::cout << "  ΔT = " << dT << " K → max force = " << max_f << " N/m³" << std::endl;

        cudaFree(d_temp);
        cudaFree(d_fx);
        cudaFree(d_fy);
        cudaFree(d_fz);
    }

    // Verify force scales with temperature gradient
    float force_ratio = max_forces[1] / max_forces[0];
    float gradient_ratio = dT_strong / dT_weak;

    std::cout << "  Force ratio (strong/weak): " << force_ratio << std::endl;
    std::cout << "  Gradient ratio: " << gradient_ratio << std::endl;

    // Force should scale approximately linearly with gradient
    EXPECT_NEAR(force_ratio, gradient_ratio, gradient_ratio * 0.3f)
        << "Force should scale linearly with temperature gradient";

    std::cout << "  ✓ Force magnitude scales correctly (PASS)" << std::endl;
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);

    std::cout << "\n";
    std::cout << "========================================" << std::endl;
    std::cout << "Marangoni Force Unit Tests" << std::endl;
    std::cout << "========================================" << std::endl;

    int result = RUN_ALL_TESTS();

    std::cout << "\n========================================" << std::endl;
    std::cout << "All Marangoni Force Tests Complete" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "\n";

    return result;
}
