#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include "physics/vof_solver.h"

using namespace lbm::physics;

int main() {
    int nx = 32, ny = 32, nz = 32;
    float dx = 1.0f;

    VOFSolver vof(nx, ny, nz, dx);

    // Initialize droplet touching bottom wall
    vof.initializeDroplet(nx / 2.0f, ny / 2.0f, 5.0f, 8.0f);
    vof.reconstructInterface();
    vof.convertCells();

    // Check what we have at bottom boundary before applying contact angle
    int num_cells = nx * ny * nz;
    std::vector<float3> h_normals_before(num_cells);
    std::vector<uint8_t> h_flags(num_cells);
    std::vector<float> h_fill(num_cells);

    cudaMemcpy(h_normals_before.data(), vof.getInterfaceNormals(),
               num_cells * sizeof(float3), cudaMemcpyDeviceToHost);
    vof.copyCellFlagsToHost(h_flags.data());
    vof.copyFillLevelToHost(h_fill.data());

    std::cout << "Checking bottom boundary (k=0):\n";
    int n_interface = 0, n_liquid = 0, n_gas = 0;
    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            int k = 0;
            int idx = i + nx * (j + ny * k);

            if (h_flags[idx] == static_cast<uint8_t>(CellFlag::INTERFACE)) {
                n_interface++;
                float norm = std::sqrt(h_normals_before[idx].x * h_normals_before[idx].x +
                                      h_normals_before[idx].y * h_normals_before[idx].y +
                                      h_normals_before[idx].z * h_normals_before[idx].z);
                std::cout << "  Interface at (" << i << "," << j << ",0): "
                          << "fill=" << h_fill[idx] << ", norm=" << norm << "\n";
            } else if (h_flags[idx] == static_cast<uint8_t>(CellFlag::LIQUID)) {
                n_liquid++;
            } else {
                n_gas++;
            }
        }
    }

    std::cout << "\nBottom boundary summary:\n";
    std::cout << "  Interface cells: " << n_interface << "\n";
    std::cout << "  Liquid cells: " << n_liquid << "\n";
    std::cout << "  Gas cells: " << n_gas << "\n";

    // Now apply contact angle
    vof.applyBoundaryConditions(1, 90.0f);

    // Check after
    std::vector<float3> h_normals_after(num_cells);
    cudaMemcpy(h_normals_after.data(), vof.getInterfaceNormals(),
               num_cells * sizeof(float3), cudaMemcpyDeviceToHost);

    std::cout << "\nAfter applying contact angle:\n";
    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            int k = 0;
            int idx = i + nx * (j + ny * k);

            if (h_flags[idx] == static_cast<uint8_t>(CellFlag::INTERFACE)) {
                float norm_after = std::sqrt(h_normals_after[idx].x * h_normals_after[idx].x +
                                            h_normals_after[idx].y * h_normals_after[idx].y +
                                            h_normals_after[idx].z * h_normals_after[idx].z);
                std::cout << "  Interface at (" << i << "," << j << ",0): norm=" << norm_after << "\n";
            }
        }
    }

    return 0;
}
