/**
 * @file test_vof_buoyancy.cu
 * @brief Verification test for VOF-based buoyancy force calculation
 *
 * This test validates the implementation of addVOFBuoyancyForce() against
 * analytical expectations for a two-phase stratified system.
 */

#include "physics/force_accumulator.h"
#include "utils/cuda_check.h"
#include <cuda_runtime.h>
#include <iostream>
#include <cmath>

using namespace lbm::physics;

// Test parameters
constexpr int NX = 10;
constexpr int NY = 10;
constexpr int NZ = 10;
constexpr int NUM_CELLS = NX * NY * NZ;

// Physical parameters
constexpr float RHO_LIQUID = 1000.0f;  // kg/m³ (water)
constexpr float RHO_GAS = 1.2f;        // kg/m³ (air)
constexpr float G_MAGNITUDE = 9.81f;   // m/s²
constexpr float GX = 0.0f;
constexpr float GY = 0.0f;
constexpr float GZ = -G_MAGNITUDE;     // Downward gravity

// Expected forces using average-density reference model:
// F = (f - 0.5) * (rho_liquid - rho_gas) * g  [N/m³]
// Reference: ρ_avg = (ρ_liquid + ρ_gas) / 2
constexpr float DENSITY_DIFF = RHO_LIQUID - RHO_GAS;  // 998.8 kg/m³
constexpr float F_LIQUID = (1.0f - 0.5f) * DENSITY_DIFF * GZ;     // f=1: -4899.14 N/m³ (downward)
constexpr float F_INTERFACE = (0.5f - 0.5f) * DENSITY_DIFF * GZ;  // f=0.5: 0.0 N/m³ (balanced)
constexpr float F_GAS = (0.0f - 0.5f) * DENSITY_DIFF * GZ;       // f=0: +4899.14 N/m³ (upward)

/**
 * @brief Setup stratified fill level field: gas at top, interface in middle, liquid at bottom
 */
void setupStratifiedFillLevel(float* h_fill_level) {
    for (int k = 0; k < NZ; ++k) {
        for (int j = 0; j < NY; ++j) {
            for (int i = 0; i < NX; ++i) {
                int idx = i + NX * (j + NY * k);

                if (k < NZ / 3) {
                    // Bottom third: pure liquid (f=1)
                    h_fill_level[idx] = 1.0f;
                } else if (k < 2 * NZ / 3) {
                    // Middle third: interface (f=0.5)
                    h_fill_level[idx] = 0.5f;
                } else {
                    // Top third: pure gas (f=0)
                    h_fill_level[idx] = 0.0f;
                }
            }
        }
    }
}

/**
 * @brief Verify buoyancy forces match analytical expectations
 */
bool verifyBuoyancyForces(const float* h_fx, const float* h_fy, const float* h_fz) {
    bool all_passed = true;
    constexpr float tolerance = 1e-4f;  // Relative tolerance for float precision

    int num_gas = 0, num_interface = 0, num_liquid = 0;

    for (int k = 0; k < NZ; ++k) {
        for (int j = 0; j < NY; ++j) {
            for (int i = 0; i < NX; ++i) {
                int idx = i + NX * (j + NY * k);

                // X and Y components should always be zero (gravity is purely vertical)
                if (std::abs(h_fx[idx] - GX) > tolerance || std::abs(h_fy[idx] - GY) > tolerance) {
                    std::cerr << "ERROR: Non-zero horizontal force at (" << i << "," << j << "," << k << "): "
                              << "fx=" << h_fx[idx] << ", fy=" << h_fy[idx] << std::endl;
                    all_passed = false;
                }

                // Z component depends on layer
                float expected_fz;
                if (k < NZ / 3) {
                    expected_fz = F_LIQUID;
                    num_liquid++;
                } else if (k < 2 * NZ / 3) {
                    expected_fz = F_INTERFACE;
                    num_interface++;
                } else {
                    expected_fz = F_GAS;
                    num_gas++;
                }

                float abs_error = std::abs(h_fz[idx] - expected_fz);
                // For zero expected (interface), use absolute tolerance scaled by max force
                float scale = std::max(std::abs(expected_fz), std::abs(F_GAS));
                float rel_error = abs_error / scale;

                if (rel_error > tolerance) {
                    std::cerr << "ERROR: Incorrect vertical force at (" << i << "," << j << "," << k << "): "
                              << "expected=" << expected_fz << ", got=" << h_fz[idx]
                              << ", rel_error=" << rel_error << std::endl;
                    all_passed = false;
                }
            }
        }
    }

    std::cout << "\n=== Verification Statistics ===\n";
    std::cout << "  Gas cells:       " << num_gas << " (expected fz=" << F_GAS << " N/m³)\n";
    std::cout << "  Interface cells: " << num_interface << " (expected fz=" << F_INTERFACE << " N/m³)\n";
    std::cout << "  Liquid cells:    " << num_liquid << " (expected fz=" << F_LIQUID << " N/m³)\n";
    std::cout << "================================\n";

    return all_passed;
}

int main() {
    std::cout << "=== VOF Buoyancy Force Test ===\n";
    std::cout << "Grid: " << NX << "x" << NY << "x" << NZ << "\n";
    std::cout << "Physical parameters:\n";
    std::cout << "  ρ_liquid = " << RHO_LIQUID << " kg/m³\n";
    std::cout << "  ρ_gas    = " << RHO_GAS << " kg/m³\n";
    std::cout << "  g        = " << GZ << " m/s²\n";
    std::cout << "  Δρ       = " << DENSITY_DIFF << " kg/m³\n";
    std::cout << "================================\n\n";

    // Allocate host memory
    float* h_fill_level = new float[NUM_CELLS];
    float* h_fx = new float[NUM_CELLS];
    float* h_fy = new float[NUM_CELLS];
    float* h_fz = new float[NUM_CELLS];

    // Setup stratified fill level
    setupStratifiedFillLevel(h_fill_level);

    // Allocate device memory
    float* d_fill_level;
    CUDA_CHECK(cudaMalloc(&d_fill_level, NUM_CELLS * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_fill_level, h_fill_level, NUM_CELLS * sizeof(float), cudaMemcpyHostToDevice));

    // Create ForceAccumulator
    ForceAccumulator force_acc(NX, NY, NZ);

    // Reset and add VOF buoyancy force
    force_acc.reset();
    force_acc.addVOFBuoyancyForce(d_fill_level, RHO_LIQUID, RHO_GAS, GX, GY, GZ);

    // Copy forces back to host
    CUDA_CHECK(cudaMemcpy(h_fx, force_acc.getFx(), NUM_CELLS * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_fy, force_acc.getFy(), NUM_CELLS * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_fz, force_acc.getFz(), NUM_CELLS * sizeof(float), cudaMemcpyDeviceToHost));

    // Verify results
    bool passed = verifyBuoyancyForces(h_fx, h_fy, h_fz);

    // Get force breakdown
    float buoyancy_mag, darcy_mag, surface_tension_mag, marangoni_mag, recoil_mag;
    force_acc.getForceBreakdown(buoyancy_mag, darcy_mag, surface_tension_mag, marangoni_mag, recoil_mag);

    std::cout << "\n=== Force Breakdown ===\n";
    std::cout << "  Buoyancy magnitude: " << buoyancy_mag << " N/m³\n";
    std::cout << "  Expected maximum:   " << std::abs(F_GAS) << " N/m³\n";
    std::cout << "========================\n\n";

    // Cleanup
    CUDA_CHECK(cudaFree(d_fill_level));
    delete[] h_fill_level;
    delete[] h_fx;
    delete[] h_fy;
    delete[] h_fz;

    if (passed) {
        std::cout << "\n✓ VOF Buoyancy Force Test PASSED\n";
        return 0;
    } else {
        std::cerr << "\n✗ VOF Buoyancy Force Test FAILED\n";
        return 1;
    }
}
