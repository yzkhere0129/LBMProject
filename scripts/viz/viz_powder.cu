/**
 * @file viz_powder.cu
 * @brief Visualization data generator for powder bed particle distribution
 *
 * Generates a random powder bed using PowderBed class, then dumps:
 *   1. Particle centers and radii to CSV (for 3D scatter plot)
 *   2. VOF fill_level z-slice to CSV (for 2D contour plot)
 */

#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>
#include <vector>
#include <algorithm>

#include "physics/powder_bed.h"
#include "physics/vof_solver.h"

using namespace lbm::physics;

int main() {
    // ========================================================================
    // Domain setup
    // ========================================================================
    const int nx = 100, ny = 100, nz = 50;
    const float dx = 2.0e-6f;  // 2 um spacing
    const int num_cells = nx * ny * nz;

    printf("=== Powder Bed Visualization ===\n");
    printf("Grid: %d x %d x %d, dx = %.1f um\n", nx, ny, nz, dx * 1e6f);
    printf("Domain: %.0f x %.0f x %.0f um\n",
           nx * dx * 1e6f, ny * dx * 1e6f, nz * dx * 1e6f);

    // ========================================================================
    // Create VOF solver (needed for PowderBed constructor)
    // ========================================================================
    VOFSolver vof(nx, ny, nz, dx);
    vof.initialize(0.0f);

    // ========================================================================
    // Configure powder bed
    // ========================================================================
    PowderBedConfig config;
    config.layer_thickness = 80.0e-6f;   // 80 um layer
    config.target_packing = 0.55f;
    config.size_dist.D50 = 20.0e-6f;     // 20 um median diameter
    config.size_dist.D_min = 10.0e-6f;
    config.size_dist.D_max = 32.0e-6f;
    config.size_dist.sigma_g = 1.5f;
    config.generation_method = PowderGenerationMethod::RANDOM_SEQUENTIAL;
    config.seed = 42;
    config.max_placement_attempts = 5000;
    config.min_gap = 0.0f;

    // ========================================================================
    // Generate powder bed
    // ========================================================================
    PowderBed powder(config, &vof);
    powder.generate(nx, ny, nz, dx);
    powder.printStatistics();

    // ========================================================================
    // Dump particle data to CSV
    // ========================================================================
    const auto& particles = powder.getParticles();
    int np = powder.getNumParticles();
    printf("\nWriting %d particles to particles.csv ...\n", np);

    FILE* fp = fopen("particles.csv", "w");
    if (!fp) { fprintf(stderr, "Cannot open particles.csv\n"); return 1; }
    fprintf(fp, "id,x_um,y_um,z_um,radius_um\n");
    for (int i = 0; i < np; ++i) {
        const auto& p = particles[i];
        fprintf(fp, "%d,%.4f,%.4f,%.4f,%.4f\n",
                p.id,
                p.x * 1e6f, p.y * 1e6f, p.z * 1e6f,
                p.radius * 1e6f);
    }
    fclose(fp);
    printf("  -> particles.csv written\n");

    // ========================================================================
    // Dump fill_level z-slices to CSV
    // ========================================================================
    std::vector<float> h_fill(num_cells);
    vof.copyFillLevelToHost(h_fill.data());

    // Write mid-z slice
    int z_mid = nz / 2;
    printf("Writing fill_level z-slice (z=%d) to fill_level_zmid.csv ...\n", z_mid);

    FILE* fz = fopen("fill_level_zmid.csv", "w");
    if (!fz) { fprintf(stderr, "Cannot open fill_level_zmid.csv\n"); return 1; }
    // Header: x_um values
    for (int i = 0; i < nx; ++i) {
        fprintf(fz, "%.2f", i * dx * 1e6f);
        if (i < nx - 1) fprintf(fz, ",");
    }
    fprintf(fz, "\n");
    // Data rows: one per y
    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            int idx = i + nx * (j + ny * z_mid);
            fprintf(fz, "%.6f", h_fill[idx]);
            if (i < nx - 1) fprintf(fz, ",");
        }
        fprintf(fz, "\n");
    }
    fclose(fz);
    printf("  -> fill_level_zmid.csv written\n");

    // Also write a lower slice near the substrate
    int z_low = nz / 4;
    printf("Writing fill_level z-slice (z=%d) to fill_level_zlow.csv ...\n", z_low);

    FILE* fl = fopen("fill_level_zlow.csv", "w");
    if (!fl) { fprintf(stderr, "Cannot open fill_level_zlow.csv\n"); return 1; }
    for (int i = 0; i < nx; ++i) {
        fprintf(fl, "%.2f", i * dx * 1e6f);
        if (i < nx - 1) fprintf(fl, ",");
    }
    fprintf(fl, "\n");
    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            int idx = i + nx * (j + ny * z_low);
            fprintf(fl, "%.6f", h_fill[idx]);
            if (i < nx - 1) fprintf(fl, ",");
        }
        fprintf(fl, "\n");
    }
    fclose(fl);
    printf("  -> fill_level_zlow.csv written\n");

    // ========================================================================
    // Summary
    // ========================================================================
    printf("\n=== Done ===\n");
    printf("Files generated:\n");
    printf("  particles.csv         - Particle centers and radii\n");
    printf("  fill_level_zmid.csv   - VOF fill level at z=%d\n", z_mid);
    printf("  fill_level_zlow.csv   - VOF fill level at z=%d\n", z_low);

    return 0;
}
