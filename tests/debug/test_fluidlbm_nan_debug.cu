/**
 * @file test_fluidlbm_nan_debug.cu
 * @brief Minimal test to debug NaN issue in FluidLBM velocity field
 */

#include <iostream>
#include <cmath>
#include "physics/fluid_lbm.h"
#include "core/lattice_d3q19.h"

using namespace lbm;
using namespace lbm::physics;
using namespace lbm::core;

int main() {
    std::cout << "=== FluidLBM NaN Debug Test ===\n\n";

    // Initialize D3Q19 constants
    D3Q19::initializeDevice();
    std::cout << "Step 1: D3Q19 constants initialized\n";

    // Create minimal FluidLBM
    int nx = 10, ny = 10, nz = 10;
    int num_cells = nx * ny * nz;
    float nu = 0.15f;
    float rho0 = 1.0f;

    std::cout << "Step 2: Creating FluidLBM solver (domain: " << nx << "x" << ny << "x" << nz << ")...\n";
    FluidLBM fluid(nx, ny, nz, nu, rho0,
                   physics::BoundaryType::PERIODIC,
                   physics::BoundaryType::PERIODIC,
                   physics::BoundaryType::PERIODIC);

    std::cout << "Step 3: Initializing with zero velocity...\n";
    fluid.initialize(rho0, 0.0f, 0.0f, 0.0f);

    std::cout << "Step 4: Computing macroscopic quantities...\n";
    fluid.computeMacroscopic();

    // Copy to host and check
    float* h_ux = new float[num_cells];
    float* h_uy = new float[num_cells];
    float* h_uz = new float[num_cells];
    float* h_rho = new float[num_cells];

    std::cout << "Step 5: Copying results to host...\n";
    fluid.copyVelocityToHost(h_ux, h_uy, h_uz);
    fluid.copyDensityToHost(h_rho);

    // Check for NaN
    std::cout << "Step 6: Checking for NaN values...\n";
    int nan_count = 0;
    for (int i = 0; i < num_cells; ++i) {
        if (std::isnan(h_ux[i]) || std::isnan(h_uy[i]) || std::isnan(h_uz[i]) || std::isnan(h_rho[i])) {
            if (nan_count < 5) {
                std::cerr << "  ERROR: NaN at cell " << i
                          << ": rho=" << h_rho[i]
                          << " ux=" << h_ux[i]
                          << " uy=" << h_uy[i]
                          << " uz=" << h_uz[i] << std::endl;
            }
            nan_count++;
        }
    }

    std::cout << "\n=== Test Results ===\n";
    if (nan_count > 0) {
        std::cerr << "FAIL: " << nan_count << " cells have NaN values\n";
        delete[] h_ux;
        delete[] h_uy;
        delete[] h_uz;
        delete[] h_rho;
        return 1;
    } else {
        std::cout << "SUCCESS: All values are valid\n";
        std::cout << "\nSample values:\n";
        std::cout << "  Cell 0: rho=" << h_rho[0] << " ux=" << h_ux[0]
                  << " uy=" << h_uy[0] << " uz=" << h_uz[0] << std::endl;
        std::cout << "  Cell 100: rho=" << h_rho[std::min(100, num_cells-1)]
                  << " ux=" << h_ux[std::min(100, num_cells-1)]
                  << " uy=" << h_uy[std::min(100, num_cells-1)]
                  << " uz=" << h_uz[std::min(100, num_cells-1)] << std::endl;
        std::cout << "  Cell " << num_cells-1 << ": rho=" << h_rho[num_cells-1]
                  << " ux=" << h_ux[num_cells-1]
                  << " uy=" << h_uy[num_cells-1]
                  << " uz=" << h_uz[num_cells-1] << std::endl;

        delete[] h_ux;
        delete[] h_uy;
        delete[] h_uz;
        delete[] h_rho;
        return 0;
    }
}
