/**
 * @file test_vof_cell_conversion.cu
 * @brief Unit test for cell type conversion (GAS/LIQUID/INTERFACE)
 */

#include <gtest/gtest.h>
#include "physics/vof_solver.h"
#include <cuda_runtime.h>
#include <vector>

using namespace lbm::physics;

/**
 * @brief Test cell flag assignment based on fill level
 */
TEST(VOFCellConversionTest, FlagAssignment) {
    int nx = 32, ny = 32, nz = 32;
    float dx = 1.0f;

    VOFSolver vof(nx, ny, nz, dx);

    // Initialize with known fill levels
    std::vector<float> h_fill(nx * ny * nz);

    // Create regions: pure gas, interface, pure liquid
    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                int idx = i + nx * (j + ny * k);

                if (i < nx / 3) {
                    h_fill[idx] = 0.0f;  // Pure gas
                } else if (i < 2 * nx / 3) {
                    h_fill[idx] = 0.5f;  // Interface
                } else {
                    h_fill[idx] = 1.0f;  // Pure liquid
                }
            }
        }
    }

    vof.initialize(h_fill.data());

    // Get cell flags
    int num_cells = nx * ny * nz;
    std::vector<uint8_t> h_flags(num_cells);
    vof.copyCellFlagsToHost(h_flags.data());

    // Check flags in each region
    int test_y = ny / 2;
    int test_z = nz / 2;

    // Gas region
    int idx_gas = (nx / 6) + nx * (test_y + ny * test_z);
    EXPECT_EQ(h_flags[idx_gas], static_cast<uint8_t>(CellFlag::GAS))
        << "Cell with f=0 should be GAS";

    // Interface region
    int idx_interface = (nx / 2) + nx * (test_y + ny * test_z);
    EXPECT_EQ(h_flags[idx_interface], static_cast<uint8_t>(CellFlag::INTERFACE))
        << "Cell with f=0.5 should be INTERFACE";

    // Liquid region
    int idx_liquid = (5 * nx / 6) + nx * (test_y + ny * test_z);
    EXPECT_EQ(h_flags[idx_liquid], static_cast<uint8_t>(CellFlag::LIQUID))
        << "Cell with f=1 should be LIQUID";
}

/**
 * @brief Test conversion after advection
 */
TEST(VOFCellConversionTest, ConversionAfterAdvection) {
    int nx = 32, ny = 32, nz = 32;
    float dx = 1.0f;

    VOFSolver vof(nx, ny, nz, dx);

    // Initialize interface
    vof.initializeDroplet(nx / 2.0f, ny / 2.0f, nz / 2.0f, 8.0f);

    // Count initial interface cells
    int num_cells = nx * ny * nz;
    std::vector<uint8_t> h_flags(num_cells);
    vof.copyCellFlagsToHost(h_flags.data());

    int initial_interface_count = 0;
    for (int i = 0; i < num_cells; ++i) {
        if (h_flags[i] == static_cast<uint8_t>(CellFlag::INTERFACE)) {
            initial_interface_count++;
        }
    }

    EXPECT_GT(initial_interface_count, 0) << "Should have interface cells initially";

    // Apply advection
    float* d_ux, *d_uy, *d_uz;
    cudaMalloc(&d_ux, num_cells * sizeof(float));
    cudaMalloc(&d_uy, num_cells * sizeof(float));
    cudaMalloc(&d_uz, num_cells * sizeof(float));

    std::vector<float> h_velocity(num_cells, 1.0f);
    cudaMemcpy(d_ux, h_velocity.data(), num_cells * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_uy, 0, num_cells * sizeof(float));
    cudaMemset(d_uz, 0, num_cells * sizeof(float));

    float dt = 0.1f;
    for (int step = 0; step < 3; ++step) {
        vof.advectFillLevel(d_ux, d_uy, d_uz, dt);
        vof.convertCells();  // Update cell flags
    }

    // Check that interface cells still exist
    vof.copyCellFlagsToHost(h_flags.data());

    int final_interface_count = 0;
    for (int i = 0; i < num_cells; ++i) {
        if (h_flags[i] == static_cast<uint8_t>(CellFlag::INTERFACE)) {
            final_interface_count++;
        }
    }

    EXPECT_GT(final_interface_count, 0) << "Should still have interface cells after advection";

    cudaFree(d_ux);
    cudaFree(d_uy);
    cudaFree(d_uz);
}

/**
 * @brief Test flag consistency with fill level
 */
TEST(VOFCellConversionTest, FlagFillLevelConsistency) {
    int nx = 32, ny = 32, nz = 32;
    float dx = 1.0f;

    VOFSolver vof(nx, ny, nz, dx);

    // Initialize random droplets
    vof.initializeDroplet(nx / 2.0f, ny / 2.0f, nz / 2.0f, 10.0f);

    // Get flags and fill levels
    int num_cells = nx * ny * nz;
    std::vector<uint8_t> h_flags(num_cells);
    std::vector<float> h_fill(num_cells);

    vof.copyCellFlagsToHost(h_flags.data());
    vof.copyFillLevelToHost(h_fill.data());

    // Check consistency
    float eps = 0.01f;  // Threshold for interface detection

    for (int i = 0; i < num_cells; ++i) {
        CellFlag flag = static_cast<CellFlag>(h_flags[i]);
        float f = h_fill[i];

        if (f < eps) {
            EXPECT_EQ(flag, CellFlag::GAS)
                << "f=" << f << " should be GAS at " << i;
        } else if (f > 1.0f - eps) {
            EXPECT_EQ(flag, CellFlag::LIQUID)
                << "f=" << f << " should be LIQUID at " << i;
        } else {
            EXPECT_EQ(flag, CellFlag::INTERFACE)
                << "f=" << f << " should be INTERFACE at " << i;
        }
    }
}
