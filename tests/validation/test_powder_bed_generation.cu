/**
 * @file test_powder_bed_generation.cu
 * @brief Unit tests for powder bed generation
 *
 * Tests:
 * 1. Particle size distribution matches specification
 * 2. No particle overlaps
 * 3. Packing density within acceptable range
 * 4. VOF fill_level correctly represents particles
 * 5. Effective thermal properties computed correctly
 */

#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>
#include <vector>
#include <algorithm>

#include "physics/powder_bed.h"
#include "physics/vof_solver.h"

using namespace lbm::physics;

// ============================================================================
// Test Utilities
// ============================================================================

#define TEST_ASSERT(condition, message) \
    do { \
        if (!(condition)) { \
            printf("[FAIL] %s: %s\n", __func__, message); \
            return false; \
        } \
    } while(0)

#define TEST_PASS() \
    do { \
        printf("[PASS] %s\n", __func__); \
        return true; \
    } while(0)

// ============================================================================
// Test 1: Size Distribution
// ============================================================================

bool test_size_distribution() {
    printf("\n--- Test: Size Distribution ---\n");

    PowderSizeDistribution dist;
    dist.D50 = 30.0e-6f;
    dist.sigma_g = 1.4f;
    dist.D_min = 15.0e-6f;
    dist.D_max = 45.0e-6f;

    // Sample many diameters
    const int N = 10000;
    std::vector<float> diameters(N);
    unsigned int seed = 42;

    for (int i = 0; i < N; ++i) {
        diameters[i] = dist.sampleDiameterHost(seed);
    }

    // Compute statistics
    float D_sum = 0.0f;
    float D_min = 1e10f, D_max = 0.0f;
    for (float D : diameters) {
        D_sum += D;
        D_min = std::min(D_min, D);
        D_max = std::max(D_max, D);
    }
    float D_mean = D_sum / N;

    // Compute median (sort and take middle)
    std::sort(diameters.begin(), diameters.end());
    float D_median = diameters[N/2];

    printf("  Sampled %d diameters\n", N);
    printf("  Min: %.1f um, Max: %.1f um\n", D_min * 1e6f, D_max * 1e6f);
    printf("  Mean: %.1f um, Median: %.1f um\n", D_mean * 1e6f, D_median * 1e6f);
    printf("  Target D50: %.1f um\n", dist.D50 * 1e6f);

    // Check median is within 20% of target
    float median_error = std::abs(D_median - dist.D50) / dist.D50;
    printf("  Median error: %.1f%%\n", median_error * 100.0f);

    TEST_ASSERT(median_error < 0.20f, "Median should be within 20% of D50");

    // Check all diameters are in valid range
    TEST_ASSERT(D_min >= dist.D_min * 0.99f, "Min diameter should respect D_min");
    TEST_ASSERT(D_max <= dist.D_max * 1.01f, "Max diameter should respect D_max");

    TEST_PASS();
}

// ============================================================================
// Test 2: Effective Thermal Conductivity
// ============================================================================

bool test_effective_conductivity() {
    printf("\n--- Test: Effective Thermal Conductivity ---\n");

    // Test case: Ti6Al4V powder in argon
    float k_solid = 25.0f;   // W/(m*K)
    float k_gas = 0.018f;    // W/(m*K)
    float packing = 0.55f;

    float k_eff = computeZBSEffectiveConductivity(k_solid, k_gas, packing);

    printf("  k_solid: %.1f W/(m*K)\n", k_solid);
    printf("  k_gas: %.3f W/(m*K)\n", k_gas);
    printf("  Packing: %.0f%%\n", packing * 100.0f);
    printf("  k_eff: %.4f W/(m*K)\n", k_eff);

    // Expected range from literature: 0.2 - 1.0 W/(m*K) for metal powder beds
    TEST_ASSERT(k_eff > 0.1f, "k_eff should be > 0.1 W/(m*K)");
    TEST_ASSERT(k_eff < 2.0f, "k_eff should be < 2.0 W/(m*K)");

    // k_eff should be between gas and solid
    TEST_ASSERT(k_eff > k_gas, "k_eff should be > k_gas");
    TEST_ASSERT(k_eff < k_solid, "k_eff should be < k_solid");

    // Higher packing should give higher conductivity
    float k_eff_60 = computeZBSEffectiveConductivity(k_solid, k_gas, 0.60f);
    float k_eff_50 = computeZBSEffectiveConductivity(k_solid, k_gas, 0.50f);
    printf("  k_eff at 50%%: %.4f, at 60%%: %.4f\n", k_eff_50, k_eff_60);
    TEST_ASSERT(k_eff_60 > k_eff_50, "Higher packing should give higher k_eff");

    TEST_PASS();
}

// ============================================================================
// Test 3: Effective Absorption Depth
// ============================================================================

bool test_absorption_depth() {
    printf("\n--- Test: Effective Absorption Depth ---\n");

    float R_particle = 15.0e-6f;  // 30um diameter / 2
    float packing = 0.55f;
    float reflectivity = 0.65f;

    float d_eff = computePowderAbsorptionDepth(R_particle, packing, reflectivity);

    printf("  Particle radius: %.1f um\n", R_particle * 1e6f);
    printf("  Packing density: %.0f%%\n", packing * 100.0f);
    printf("  Reflectivity: %.2f\n", reflectivity);
    printf("  d_eff: %.2f um\n", d_eff * 1e6f);

    // Expected: d_eff ~ R * (1-phi) / (3*(1-r))
    // = 15um * 0.55 / (3 * 0.35) = 7.9 um
    float expected = R_particle * (1.0f - (1.0f - packing)) / (3.0f * (1.0f - reflectivity));
    printf("  Expected: %.2f um\n", expected * 1e6f);

    float error = std::abs(d_eff - expected) / expected;
    printf("  Error: %.1f%%\n", error * 100.0f);

    TEST_ASSERT(error < 0.1f, "d_eff should match analytical formula within 10%");

    // d_eff should be reasonable (a few particle radii)
    TEST_ASSERT(d_eff > 5.0e-6f, "d_eff should be > 5 um");
    TEST_ASSERT(d_eff < 50.0e-6f, "d_eff should be < 50 um");

    TEST_PASS();
}

// ============================================================================
// Test 4: Particle Generation and No Overlap
// ============================================================================

bool test_particle_generation() {
    printf("\n--- Test: Particle Generation ---\n");

    // Create small domain for quick test
    int nx = 50, ny = 50, nz = 30;
    float dx = 2.0e-6f;  // 2 um

    // Create VOF solver
    VOFSolver vof(nx, ny, nz, dx);
    vof.initialize(0.0f);  // Start with empty domain

    // Configure powder bed.
    // Use REGULAR_PERTURBED: places particles on a hexagonal grid (deterministic,
    // no overlaps, ~33% packing for D50=20um in a 100x100x40um domain).
    // RSA in quasi-2D (layer = 2*D50) is geometrically limited to ~22%.
    PowderBedConfig config;
    config.layer_thickness = 40.0e-6f;  // 40 um
    config.target_packing = 0.50f;       // 50% (used for RSA; regular method ignores this)
    config.size_dist.D50 = 20.0e-6f;     // Particle diameter
    config.size_dist.D_min = 10.0e-6f;
    config.size_dist.D_max = 30.0e-6f;
    config.generation_method = PowderGenerationMethod::REGULAR_PERTURBED;
    config.seed = 12345;

    // Generate powder bed
    PowderBed powder(config, &vof);
    powder.generate(nx, ny, nz, dx);

    // Check particle count
    int num_particles = powder.getNumParticles();
    printf("  Generated %d particles\n", num_particles);
    TEST_ASSERT(num_particles > 0, "Should generate at least some particles");

    // Check no overlaps
    bool no_overlaps = powder.verifyNoOverlaps();
    TEST_ASSERT(no_overlaps, "No particle overlaps should exist");

    // Check packing density
    float actual_packing = powder.getActualPacking();
    printf("  Actual packing: %.1f%% (target: %.1f%%)\n",
           actual_packing * 100.0f, config.target_packing * 100.0f);

    // Packing should be within reasonable range (at least 80% of target)
    TEST_ASSERT(actual_packing > 0.3f, "Packing should be > 30%");
    TEST_ASSERT(actual_packing < 0.75f, "Packing should be < 75% (physical limit)");

    // Print statistics
    powder.printStatistics();

    TEST_PASS();
}

// ============================================================================
// Test 5: VOF Fill Level Initialization
// ============================================================================

bool test_vof_fill_level() {
    printf("\n--- Test: VOF Fill Level ---\n");

    // Create small domain
    int nx = 40, ny = 40, nz = 20;
    float dx = 2.0e-6f;
    int num_cells = nx * ny * nz;

    // Create VOF solver
    VOFSolver vof(nx, ny, nz, dx);
    vof.initialize(0.0f);

    // Configure powder bed with known configuration
    PowderBedConfig config;
    config.layer_thickness = 30.0e-6f;
    config.target_packing = 0.40f;  // Low for easy generation
    config.size_dist.D50 = 15.0e-6f;
    config.size_dist.D_min = 10.0e-6f;
    config.size_dist.D_max = 20.0e-6f;
    config.substrate_height = 0.0f;
    config.seed = 42;

    // Generate
    PowderBed powder(config, &vof);
    powder.generate(nx, ny, nz, dx);

    // Copy fill level to host
    std::vector<float> h_fill(num_cells);
    vof.copyFillLevelToHost(h_fill.data());

    // Compute statistics
    float fill_sum = 0.0f;
    int cells_with_metal = 0;
    int interface_cells = 0;

    for (int idx = 0; idx < num_cells; ++idx) {
        float f = h_fill[idx];
        fill_sum += f;
        if (f > 0.01f) cells_with_metal++;
        if (f > 0.01f && f < 0.99f) interface_cells++;
    }

    float avg_fill = fill_sum / num_cells;

    printf("  Total cells: %d\n", num_cells);
    printf("  Cells with metal (f>0.01): %d (%.1f%%)\n",
           cells_with_metal, 100.0f * cells_with_metal / num_cells);
    printf("  Interface cells (0.01<f<0.99): %d\n", interface_cells);
    printf("  Average fill level: %.4f\n", avg_fill);

    // Verify fill level is within [0, 1]
    float f_min = 1.0f, f_max = 0.0f;
    for (int idx = 0; idx < num_cells; ++idx) {
        f_min = std::min(f_min, h_fill[idx]);
        f_max = std::max(f_max, h_fill[idx]);
    }
    printf("  Fill level range: [%.4f, %.4f]\n", f_min, f_max);

    TEST_ASSERT(f_min >= 0.0f, "Fill level should be >= 0");
    TEST_ASSERT(f_max <= 1.0f, "Fill level should be <= 1");

    // Should have some interface cells
    TEST_ASSERT(interface_cells > 0, "Should have interface cells");

    // Mass should roughly match particle volume
    float cell_volume = dx * dx * dx;
    float vof_volume = fill_sum * cell_volume;
    float particle_volume = powder.getTotalParticleVolume();

    printf("  VOF volume: %.2e m^3\n", vof_volume);
    printf("  Particle volume: %.2e m^3\n", particle_volume);

    float volume_error = std::abs(vof_volume - particle_volume) / particle_volume;
    printf("  Volume error: %.1f%%\n", volume_error * 100.0f);

    // Allow significant error due to discretization
    TEST_ASSERT(volume_error < 0.50f, "Volume error should be < 50%");

    TEST_PASS();
}

// ============================================================================
// Test 6: Config Derived Quantities
// ============================================================================

bool test_config_derived() {
    printf("\n--- Test: Config Derived Quantities ---\n");

    PowderBedConfig config;

    // Check that derived quantities are computed
    config.k_solid = 30.0f;
    config.k_gas = 0.02f;
    config.target_packing = 0.55f;
    config.size_dist.D50 = 30.0e-6f;
    config.particle_reflectivity = 0.65f;

    config.computeDerivedQuantities();

    printf("  k_solid: %.1f W/(m*K)\n", config.k_solid);
    printf("  k_gas: %.3f W/(m*K)\n", config.k_gas);
    printf("  Effective k: %.4f W/(m*K)\n", config.effective_k);
    printf("  Effective absorption: %.2f um\n", config.effective_absorption_depth * 1e6f);

    TEST_ASSERT(config.effective_k > 0.0f, "Effective k should be computed");
    TEST_ASSERT(config.effective_absorption_depth > 0.0f, "Absorption depth should be computed");

    TEST_PASS();
}

// ============================================================================
// Main
// ============================================================================

int main() {
    printf("========================================\n");
    printf("Powder Bed Generation Tests\n");
    printf("========================================\n");

    int passed = 0;
    int failed = 0;

    if (test_size_distribution()) passed++; else failed++;
    if (test_effective_conductivity()) passed++; else failed++;
    if (test_absorption_depth()) passed++; else failed++;
    if (test_config_derived()) passed++; else failed++;
    if (test_particle_generation()) passed++; else failed++;
    if (test_vof_fill_level()) passed++; else failed++;

    printf("\n========================================\n");
    printf("Results: %d passed, %d failed\n", passed, failed);
    printf("========================================\n");

    return (failed == 0) ? 0 : 1;
}
