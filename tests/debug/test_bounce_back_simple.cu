/**
 * @file test_bounce_back_simple.cu
 * @brief Simple test to verify bounce-back is working
 */

#include <iostream>
#include <vector>
#include <cmath>
#include "physics/fluid_lbm.h"

using namespace lbm::physics;

int main() {
    std::cout << "\n=== Simple Bounce-Back Test ===" << std::endl;

    // Small domain: 3x3x3 with wall at z=0 and z=2
    const int nx = 3, ny = 3, nz = 3;
    const float nu = 0.033f;
    const float rho = 1000.0f;

    FluidLBM fluid(nx, ny, nz, nu, rho,
                   BoundaryType::PERIODIC,
                   BoundaryType::PERIODIC,
                   BoundaryType::WALL);

    // Initialize with uniform velocity in +x direction
    fluid.initialize(rho, 0.1f, 0.0f, 0.0f);

    std::cout << "\nInitial state:" << std::endl;
    std::vector<float> h_ux(nx*ny*nz), h_uy(nx*ny*nz), h_uz(nx*ny*nz);
    fluid.copyVelocityToHost(h_ux.data(), h_uy.data(), h_uz.data());

    // Print wall node velocities (z=0)
    std::cout << "Wall at z=0:" << std::endl;
    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            int idx = i + nx * (j + ny * 0);
            std::cout << "  (" << i << "," << j << ",0): ux=" << h_ux[idx]
                      << ", uy=" << h_uy[idx] << ", uz=" << h_uz[idx] << std::endl;
        }
    }

    // Run one iteration
    std::cout << "\nApplying one LBM step..." << std::endl;
    fluid.collisionBGK();
    fluid.streaming();
    fluid.applyBoundaryConditions(1);
    fluid.computeMacroscopic();

    // Check wall velocities
    fluid.copyVelocityToHost(h_ux.data(), h_uy.data(), h_uz.data());

    std::cout << "\nAfter 1 step:" << std::endl;
    std::cout << "Wall at z=0:" << std::endl;
    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            int idx = i + nx * (j + ny * 0);
            float v_mag = std::sqrt(h_ux[idx]*h_ux[idx] +
                                   h_uy[idx]*h_uy[idx] +
                                   h_uz[idx]*h_uz[idx]);
            std::cout << "  (" << i << "," << j << ",0): ux=" << h_ux[idx]
                      << ", |v|=" << v_mag << std::endl;
        }
    }

    // Check if wall velocity is near zero
    float max_wall_v = 0.0f;
    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            int idx = i + nx * (j + ny * 0);
            float v = std::sqrt(h_ux[idx]*h_ux[idx] + h_uy[idx]*h_uy[idx] + h_uz[idx]*h_uz[idx]);
            max_wall_v = std::max(max_wall_v, v);
        }
    }

    std::cout << "\nMax wall velocity: " << max_wall_v << std::endl;
    if (max_wall_v < 0.01f) {
        std::cout << "PASS: Wall velocity is near zero" << std::endl;
        return 0;
    } else {
        std::cout << "FAIL: Wall velocity is " << max_wall_v << " (should be < 0.01)" << std::endl;
        return 1;
    }
}
