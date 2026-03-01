/**
 * @file test_visualization.cu
 * @brief Quick test of visualization capabilities
 */

#include <iostream>
#include <cuda_runtime.h>
#include "physics/thermal_lbm.h"
#include "physics/laser_source.h"
#include "physics/material_properties.h"
#include "io/vtk_writer.h"

using namespace lbm;
using namespace lbm::physics;

int main() {
    std::cout << "\n=== Quick Visualization Test ===\n\n";

    // Small domain for quick test
    const int nx = 32, ny = 32, nz = 16;
    const float dx = 3e-6f;  // 3 micrometers
    const float dy = 3e-6f;
    const float dz = 3e-6f;
    const float dt = 1e-9f;   // 1 nanosecond

    std::cout << "Domain: " << nx << " x " << ny << " x " << nz << "\n";
    std::cout << "Physical size: " << nx*dx*1e6 << " x " << ny*dy*1e6
              << " x " << nz*dz*1e6 << " micrometers\n\n";

    // Material
    MaterialProperties ti64 = MaterialDatabase::getTi6Al4V();
    float thermal_diffusivity = ti64.getThermalDiffusivity(300.0f);

    // Initialize thermal solver
    // CRITICAL FIX: Pass dt and dx for proper tau scaling
    ThermalLBM thermal(nx, ny, nz, thermal_diffusivity,
                       ti64.rho_solid, ti64.cp_solid, dt, dx);
    thermal.initialize(300.0f);

    // Laser
    LaserSource laser(100.0f, 30e-6f, ti64.absorptivity_solid, 10e-6f);
    laser.setPosition(nx*dx/2, ny*dy/2, 0.0f);

    // GPU memory
    float* d_heat_source;
    cudaMalloc(&d_heat_source, nx * ny * nz * sizeof(float));

    // Host memory
    float* h_temperature = new float[nx * ny * nz];

    // Create output directory
    system("mkdir -p test_output");

    // Run 10 steps
    std::cout << "Running 10 simulation steps...\n";
    for (int step = 0; step <= 10; ++step) {
        // Compute heat source
        dim3 block(8, 8, 4);
        dim3 grid((nx + block.x - 1) / block.x,
                 (ny + block.y - 1) / block.y,
                 (nz + block.z - 1) / block.z);

        lbm::physics::computeLaserHeatSourceKernel<<<grid, block>>>(
            d_heat_source, laser, dx, dy, dz, nx, ny, nz
        );

        // Add heat and evolve
        thermal.addHeatSource(d_heat_source, dt);
        thermal.collisionBGK();
        thermal.streaming();
        thermal.computeTemperature();

        // Output at step 0, 5, and 10
        if (step % 5 == 0) {
            thermal.copyTemperatureToHost(h_temperature);

            // Find max temperature
            float T_max = 300.0f;
            for (int i = 0; i < nx * ny * nz; ++i) {
                if (h_temperature[i] > T_max) T_max = h_temperature[i];
            }

            std::cout << "  Step " << step << ": T_max = " << T_max << " K\n";

            // Write VTK file
            std::string filename = "test_output/test_" + std::to_string(step);
            io::VTKWriter::writeStructuredPoints(
                filename, h_temperature,
                nx, ny, nz, dx, dy, dz,
                "Temperature"
            );
        }
    }

    // Cleanup
    delete[] h_temperature;
    cudaFree(d_heat_source);

    std::cout << "\nTest complete! Check test_output/ directory for VTK files.\n";
    std::cout << "You can open them in ParaView to verify visualization works.\n\n";

    return 0;
}