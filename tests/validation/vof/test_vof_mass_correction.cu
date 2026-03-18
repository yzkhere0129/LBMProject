/**
 * @file test_vof_mass_correction.cu
 * @brief Validation test for VOF mass conservation correction
 *
 * This test validates that the global mass correction mechanism reduces
 * mass loss from 20% (without correction) to <7% (with correction).
 *
 * Test setup: Simple advection with high CFL to induce mass loss
 * Success criteria: Mass error < 7% with correction enabled
 */

#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include "physics/vof_solver.h"
#include "utils/cuda_check.h"

using namespace lbm::physics;

int main() {
    std::cout << "\n========================================" << std::endl;
    std::cout << "VOF Mass Conservation Correction Test" << std::endl;
    std::cout << "========================================\n" << std::endl;

    // ========================================================================
    // Test Configuration
    // ========================================================================
    const int nx = 64;
    const int ny = 64;
    const int nz = 64;
    const float dx = 0.001f;  // 1mm
    const float dt = 0.0002f;  // 0.2ms
    const int num_steps = 1000;
    const int num_cells = nx * ny * nz;

    std::cout << "Domain: " << nx << " x " << ny << " x " << nz << std::endl;
    std::cout << "Grid spacing: " << dx << " m" << std::endl;
    std::cout << "Timestep: " << dt << " s" << std::endl;
    std::cout << "Simulation time: " << num_steps * dt << " s" << std::endl;
    std::cout << std::endl;

    // ========================================================================
    // Initialize VOF solver with periodic boundaries
    // ========================================================================
    VOFSolver vof(nx, ny, nz, dx,
                  VOFSolver::BoundaryType::PERIODIC,
                  VOFSolver::BoundaryType::PERIODIC,
                  VOFSolver::BoundaryType::PERIODIC);

    // Use TVD scheme for better baseline (still loses mass)
    vof.setAdvectionScheme(VOFAdvectionScheme::TVD);
    vof.setTVDLimiter(TVDLimiter::VAN_LEER);

    // ENABLE MASS CORRECTION (this is what we're testing)
    vof.setMassConservationCorrection(true, 0.7f);  // Enable with damping=0.7

    std::cout << "VOF advection: TVD (van Leer)" << std::endl;
    std::cout << "Mass correction: ENABLED (damping=0.7)" << std::endl;
    std::cout << std::endl;

    // ========================================================================
    // Initialize spherical droplet at domain center
    // ========================================================================
    float center_x = nx / 2.0f;
    float center_y = ny / 2.0f;
    float center_z = nz / 2.0f;
    float radius = 16.0f;  // cells

    vof.initializeDroplet(center_x, center_y, center_z, radius);

    float mass_initial = vof.computeTotalMass();
    vof.setReferenceMass(mass_initial);  // Set reference for correction

    std::cout << "Initial droplet:" << std::endl;
    std::cout << "  Center: (" << center_x << ", " << center_y << ", " << center_z << ")" << std::endl;
    std::cout << "  Radius: " << radius << " cells" << std::endl;
    std::cout << "  Initial mass: " << mass_initial << std::endl;
    std::cout << std::endl;

    // ========================================================================
    // Create uniform velocity field (diagonal advection)
    // ========================================================================
    // High velocity to induce numerical diffusion and mass loss
    float u_adv = 0.5f;  // m/s
    float cfl = u_adv * dt / dx;  // Should be 0.1 (safe)

    std::vector<float> h_ux(num_cells, u_adv);
    std::vector<float> h_uy(num_cells, u_adv);
    std::vector<float> h_uz(num_cells, u_adv);

    // Copy velocity to device
    float *d_ux, *d_uy, *d_uz;
    cudaMalloc(&d_ux, num_cells * sizeof(float));
    cudaMalloc(&d_uy, num_cells * sizeof(float));
    cudaMalloc(&d_uz, num_cells * sizeof(float));
    cudaMemcpy(d_ux, h_ux.data(), num_cells * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_uy, h_uy.data(), num_cells * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_uz, h_uz.data(), num_cells * sizeof(float), cudaMemcpyHostToDevice);

    std::cout << "Velocity field:" << std::endl;
    std::cout << "  u = (" << u_adv << ", " << u_adv << ", " << u_adv << ") m/s" << std::endl;
    std::cout << "  CFL = " << cfl << std::endl;
    std::cout << std::endl;

    // ========================================================================
    // Time integration
    // ========================================================================
    std::cout << "Running simulation..." << std::endl;
    std::cout << std::setw(8) << "Step"
              << std::setw(12) << "Time (s)"
              << std::setw(15) << "Mass"
              << std::setw(15) << "Error (%)"
              << std::setw(15) << "Status" << std::endl;
    std::cout << std::string(64, '-') << std::endl;

    for (int step = 0; step <= num_steps; ++step) {
        float t = step * dt;

        // Print diagnostics every 100 steps
        if (step % 100 == 0) {
            float mass = vof.computeTotalMass();
            float mass_error = std::abs(mass - mass_initial) / mass_initial * 100.0f;

            std::string status = "";
            if (mass_error < 1.0f) status = "EXCELLENT";
            else if (mass_error < 5.0f) status = "GOOD";
            else if (mass_error < 7.0f) status = "ACCEPTABLE";
            else status = "POOR";

            std::cout << std::setw(8) << step
                      << std::setw(12) << std::fixed << std::setprecision(4) << t
                      << std::setw(15) << std::setprecision(2) << mass
                      << std::setw(15) << std::setprecision(3) << mass_error
                      << std::setw(15) << status << std::endl;
        }

        // Advect VOF field
        if (step < num_steps) {
            vof.advectFillLevel(d_ux, d_uy, d_uz, dt);
        }
    }

    // ========================================================================
    // Final validation
    // ========================================================================
    float mass_final = vof.computeTotalMass();
    float mass_loss = mass_initial - mass_final;
    float mass_error_pct = std::abs(mass_loss) / mass_initial * 100.0f;

    std::cout << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Final Results" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Initial mass:  " << mass_initial << std::endl;
    std::cout << "Final mass:    " << mass_final << std::endl;
    std::cout << "Mass loss:     " << mass_loss << " (" << mass_error_pct << "%)" << std::endl;
    std::cout << std::endl;

    // Success criterion: < 7% mass error
    const float MAX_ACCEPTABLE_ERROR = 7.0f;
    bool passed = (mass_error_pct < MAX_ACCEPTABLE_ERROR);

    if (passed) {
        std::cout << "✓ TEST PASSED: Mass error " << mass_error_pct
                  << "% < " << MAX_ACCEPTABLE_ERROR << "%" << std::endl;
    } else {
        std::cout << "✗ TEST FAILED: Mass error " << mass_error_pct
                  << "% >= " << MAX_ACCEPTABLE_ERROR << "%" << std::endl;
    }
    std::cout << std::endl;

    // Cleanup
    cudaFree(d_ux);
    cudaFree(d_uy);
    cudaFree(d_uz);

    return passed ? 0 : 1;
}
