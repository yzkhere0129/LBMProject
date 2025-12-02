/**
 * @file demo_powder_bed_lpbf.cu
 * @brief Demonstration: LPBF simulation with powder bed layer
 *
 * This example shows how to:
 * 1. Configure a powder bed layer on top of substrate
 * 2. Initialize VOF fill_level with discrete particles
 * 3. Run LPBF simulation with powder melting
 * 4. Visualize particle coalescence and melt pool formation
 *
 * Usage:
 *   ./demo_powder_bed_lpbf [output_dir]
 *
 * Output:
 *   - VTK files for ParaView visualization
 *   - Console output with simulation statistics
 */

#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>
#include <string>
#include <vector>

#include "physics/powder_bed.h"
#include "physics/multiphysics_solver.h"
#include "io/vtk_writer.h"

using namespace lbm::physics;
using namespace lbm::io;

// ============================================================================
// Helper Functions
// ============================================================================

void printUsage(const char* program) {
    printf("Usage: %s [output_dir]\n", program);
    printf("  output_dir: Directory for VTK output files (default: ./powder_output)\n");
}

void printGPUInfo() {
    int device;
    cudaGetDevice(&device);

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);

    printf("\n========================================\n");
    printf("GPU Information\n");
    printf("========================================\n");
    printf("Device: %s\n", prop.name);
    printf("Compute capability: %d.%d\n", prop.major, prop.minor);
    printf("Total memory: %.1f GB\n", prop.totalGlobalMem / 1e9);
    printf("SM count: %d\n", prop.multiProcessorCount);
    printf("========================================\n\n");
}

// ============================================================================
// Main Simulation
// ============================================================================

int main(int argc, char* argv[]) {
    // Parse arguments
    std::string output_dir = "./powder_output";
    if (argc > 1) {
        output_dir = argv[1];
    }

    printGPUInfo();

    printf("========================================\n");
    printf("LPBF Powder Bed Demonstration\n");
    printf("========================================\n");
    printf("Output directory: %s\n", output_dir.c_str());

    // ========================================================================
    // Domain Configuration
    // ========================================================================

    // Grid parameters
    // REALISTIC CONFIGURATION based on architect recommendations:
    // - Domain: 300×300×160 um physical for proper melt pool containment
    // - dx = 2 um gives ~10 cells per D20 particle (adequate for VOF)
    // - Expected ~300-400 particles for 55% packing
    // - Memory: ~280 MB (fits RTX 3050 4GB easily)
    const int nx = 150;   // 300 um
    const int ny = 150;   // 300 um
    const int nz = 80;    // 160 um (40um substrate + 40um layer + 80um headspace)
    const float dx = 2.0e-6f;  // 2 um resolution (10 cells per D20 particle)

    printf("\nDomain Configuration:\n");
    printf("  Grid: %d x %d x %d\n", nx, ny, nz);
    printf("  Resolution: %.1f um\n", dx * 1e6f);
    printf("  Physical size: %.0f x %.0f x %.0f um\n",
           nx * dx * 1e6f, ny * dx * 1e6f, nz * dx * 1e6f);

    // ========================================================================
    // Multiphysics Configuration
    // ========================================================================

    MultiphysicsConfig config;

    // Domain
    config.nx = nx;
    config.ny = ny;
    config.nz = nz;
    config.dx = dx;

    // Time stepping
    config.dt = 1.0e-8f;  // 10 ns (reasonable for thermal diffusion)

    // Enable physics modules
    config.enable_thermal = true;
    config.enable_thermal_advection = true;
    config.enable_phase_change = true;
    config.enable_fluid = true;
    config.enable_vof = true;
    config.enable_vof_advection = true;
    config.enable_surface_tension = true;
    config.enable_marangoni = true;
    config.enable_laser = true;
    config.enable_darcy = true;
    config.enable_buoyancy = true;
    config.enable_evaporation_mass_loss = true;
    config.enable_substrate_cooling = true;
    config.enable_radiation_bc = true;

    // Laser parameters (typical LPBF)
    config.laser_power = 200.0f;           // 200 W (increased for proper melting)
    config.laser_spot_radius = 40.0e-6f;   // 40 um spot
    config.laser_absorptivity = 0.35f;
    config.laser_penetration_depth = 10.0e-6f;  // Penetration depth

    // Adaptive CFL for keyhole/high-velocity simulation
    config.cfl_use_adaptive = true;
    config.cfl_v_target_interface = 0.55f;    // 11 m/s for interface cells
    config.cfl_v_target_bulk = 0.35f;         // 7 m/s for bulk liquid
    config.cfl_recoil_boost_factor = 2.0f;    // 2x boost for recoil forces
    config.cfl_force_ramp_factor = 0.95f;     // Smoother ramp-up

    // Laser scanning - start at domain center
    config.laser_start_x = 150.0e-6f;      // Start at center (150 um for 300 um domain)
    config.laser_start_y = 150.0e-6f;
    config.laser_scan_vx = 0.3f;           // 300 mm/s scan speed
    config.laser_scan_vy = 0.0f;

    // Material properties (Ti6Al4V)
    config.material.rho_solid = 4420.0f;      // kg/m³
    config.material.rho_liquid = 4110.0f;     // kg/m³
    config.material.cp_solid = 670.0f;        // J/(kg·K)
    config.material.cp_liquid = 831.0f;       // J/(kg·K)
    config.material.k_solid = 7.0f;           // W/(m·K)
    config.material.k_liquid = 33.0f;         // W/(m·K)
    config.material.mu_liquid = 5e-3f;        // Pa·s
    config.material.T_solidus = 1878.0f;      // K
    config.material.T_liquidus = 1923.0f;     // K (melting point)
    config.material.T_vaporization = 3533.0f; // K (boiling point)
    config.material.L_fusion = 365000.0f;     // J/kg (latent heat)
    config.material.L_vaporization = 8.9e6f;  // J/kg
    config.material.surface_tension = 1.65f;  // N/m
    config.material.dsigma_dT = -0.26e-3f;    // N/(m·K)
    config.material.absorptivity_solid = 0.35f;
    config.material.absorptivity_liquid = 0.35f;
    config.material.emissivity = 0.3f;

    config.surface_tension_coeff = 1.65f;
    config.dsigma_dT = -0.26e-3f;
    config.density = 4110.0f;
    config.thermal_diffusivity = 5.8e-6f;
    config.kinematic_viscosity = 0.0333f;  // Lattice units

    printf("\nLaser Parameters:\n");
    printf("  Power: %.0f W\n", config.laser_power);
    printf("  Spot radius: %.0f um\n", config.laser_spot_radius * 1e6f);
    printf("  Scan speed: %.0f mm/s\n", config.laser_scan_vx * 1000.0f);

    // ========================================================================
    // Powder Bed Configuration
    // ========================================================================

    PowderBedConfig powder_config;

    // Layer geometry - realistic LPBF powder layer
    // - Layer thickness = 2-3 x D50 = 40 um (typical LPBF layer)
    // - Substrate provides thermal sink below powder
    // - Domain height 160 um leaves headspace for vapor
    powder_config.layer_thickness = 40.0e-6f;      // 40 um layer (2.7 x D50)
    powder_config.target_packing = 0.60f;          // 60% packing
    powder_config.substrate_height = 60.0e-6f;     // 60 um substrate thickness

    // Size distribution - realistic Ti6Al4V powder (15-45 um range)
    // At dx=2um, D50=15um = 7.5 cells (clearly visible particles)
    powder_config.size_dist.D50 = 15.0e-6f;        // 15 um median (7.5 cells @ 2um resolution)
    powder_config.size_dist.sigma_g = 1.3f;        // Moderate distribution spread
    powder_config.size_dist.D_min = 10.0e-6f;      // 10 um min (5 cells)
    powder_config.size_dist.D_max = 25.0e-6f;      // 25 um max (12.5 cells)

    // Thermal properties
    powder_config.k_solid = 25.0f;                 // W/(m*K)
    powder_config.k_gas = 0.018f;                  // Argon

    // Compute derived quantities
    powder_config.computeDerivedQuantities();

    printf("\nPowder Bed Configuration:\n");
    printf("  Layer thickness: %.0f um\n", powder_config.layer_thickness * 1e6f);
    printf("  Target packing: %.0f%%\n", powder_config.target_packing * 100.0f);
    printf("  Median particle size: %.0f um\n", powder_config.size_dist.D50 * 1e6f);
    printf("  Effective k: %.4f W/(m*K)\n", powder_config.effective_k);
    printf("  Effective absorption depth: %.1f um\n",
           powder_config.effective_absorption_depth * 1e6f);

    // ========================================================================
    // Create Solver and Generate Powder Bed
    // ========================================================================

    printf("\nInitializing simulation...\n");

    MultiphysicsSolver solver(config);

    // Initialize with interface at TOP of powder layer (where laser enters)
    // CRITICAL FIX: Previous code used substrate height, but laser should heat from powder surface
    float substrate_temp = 300.0f;
    float powder_top_z = powder_config.substrate_height + powder_config.layer_thickness;
    float interface_height = powder_top_z / (nz * dx);  // = (30+40)um / 80um = 0.875

    printf("  Interface height: %.3f (z = %.0f um)\n", interface_height, powder_top_z * 1e6f);
    solver.initialize(substrate_temp, interface_height);

    // ========================================================================
    // Generate Powder Bed with Spherical Particles
    // ========================================================================

    printf("Generating powder bed with spherical particles...\n");

    // Create host fill_level array
    std::vector<float> h_fill(nx * ny * nz, 0.0f);

    // Parameters for powder layer
    int z_substrate_top = static_cast<int>(powder_config.substrate_height / dx);  // ~15 cells
    int z_powder_top = static_cast<int>((powder_config.substrate_height + powder_config.layer_thickness) / dx);  // ~35 cells

    // Fill substrate region (solid metal below powder)
    // Use <= to include z_substrate_top layer (particles will overlap with this)
    for (int k = 0; k <= z_substrate_top; ++k) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                int idx = i + j * nx + k * nx * ny;
                h_fill[idx] = 1.0f;  // Solid substrate
            }
        }
    }

    // ========================================================================
    // IMPROVED PARTICLE GENERATION WITH LOG-NORMAL SIZE DISTRIBUTION
    // ========================================================================
    // Uses proper log-normal sampling and collision detection for realistic packing

    printf("  Substrate: z = 0 to %d cells (%.0f um)\n", z_substrate_top, powder_config.substrate_height * 1e6f);
    printf("  Powder layer: z = %d to %d cells\n", z_substrate_top, z_powder_top);

    // Calculate expected number of particles based on layer volume and packing
    float layer_volume_um3 = (nx * dx * 1e6f) * (ny * dx * 1e6f) * (powder_config.layer_thickness * 1e6f);
    float mean_particle_radius_um = powder_config.size_dist.D50 * 0.5e6f;
    float mean_particle_volume_um3 = (4.0f/3.0f) * 3.14159f * mean_particle_radius_um * mean_particle_radius_um * mean_particle_radius_um;
    int target_particles = static_cast<int>(powder_config.target_packing * layer_volume_um3 / mean_particle_volume_um3);

    printf("  Target particles: %d (for %.0f%% packing)\n", target_particles, powder_config.target_packing * 100.0f);
    printf("  Mean particle diameter: %.1f um (%.1f cells)\n",
           powder_config.size_dist.D50 * 1e6f, powder_config.size_dist.D50 / dx);

    // Particle storage for collision detection
    struct ParticlePos { float x, y, z, r; };
    std::vector<ParticlePos> particles;
    particles.reserve(target_particles);

    // Log-normal distribution parameters
    float ln_D50 = logf(powder_config.size_dist.D50);
    float ln_sigma = logf(powder_config.size_dist.sigma_g);

    srand(42);  // Fixed seed for reproducibility
    int particles_placed = 0;
    int max_attempts_per_particle = 500;
    float total_particle_volume = 0.0f;

    for (int p = 0; p < target_particles; ++p) {
        bool placed = false;

        for (int attempt = 0; attempt < max_attempts_per_particle && !placed; ++attempt) {
            // Sample diameter from log-normal distribution (Box-Muller)
            float u1 = float(rand()) / RAND_MAX + 1e-10f;
            float u2 = float(rand()) / RAND_MAX;
            float z_normal = sqrtf(-2.0f * logf(u1)) * cosf(2.0f * 3.14159f * u2);
            float D = expf(ln_D50 + ln_sigma * z_normal);
            D = std::max(powder_config.size_dist.D_min, std::min(powder_config.size_dist.D_max, D));
            float r_physical = D / 2.0f;
            float r_cells = r_physical / dx;

            // Random position within powder layer (in cells)
            // Particles sit directly on substrate: z_min = z_substrate_top (particle bottom touches substrate)
            float px = r_cells + (float(rand()) / RAND_MAX) * (nx - 2.0f * r_cells);
            float py = r_cells + (float(rand()) / RAND_MAX) * (ny - 2.0f * r_cells);
            // pz range: from substrate top (touching) to powder layer top
            // particle center at z means particle bottom at z - r_cells
            // for particle to touch substrate: z_center >= z_substrate_top (so bottom is at z_substrate_top - r_cells, overlapping substrate)
            float pz = z_substrate_top + (float(rand()) / RAND_MAX) *
                       std::max(0.0f, (z_powder_top - z_substrate_top - r_cells));

            // Check collision with existing particles
            bool collision = false;
            for (const auto& existing : particles) {
                float ddx = px - existing.x;
                float ddy = py - existing.y;
                float ddz = pz - existing.z;
                float dist = sqrtf(ddx*ddx + ddy*ddy + ddz*ddz);
                if (dist < (r_cells + existing.r) * 1.02f) {  // 2% safety margin
                    collision = true;
                    break;
                }
            }

            if (!collision) {
                // Place particle
                particles.push_back({px, py, pz, r_cells});

                // Fill voxels inside sphere with smooth interface
                int k_min = std::max(0, int(pz - r_cells - 1));
                int k_max = std::min(nz, int(pz + r_cells + 2));
                int j_min = std::max(0, int(py - r_cells - 1));
                int j_max = std::min(ny, int(py + r_cells + 2));
                int i_min = std::max(0, int(px - r_cells - 1));
                int i_max = std::min(nx, int(px + r_cells + 2));

                for (int k = k_min; k < k_max; ++k) {
                    for (int j = j_min; j < j_max; ++j) {
                        for (int i = i_min; i < i_max; ++i) {
                            float dist = sqrtf((i - px) * (i - px) +
                                              (j - py) * (j - py) +
                                              (k - pz) * (k - pz));
                            if (dist <= r_cells + 1.0f) {
                                int idx = i + j * nx + k * nx * ny;
                                // Smooth tanh interface (interface width = 1 cell)
                                float fill = 0.5f * (1.0f - tanhf(2.0f * (dist - r_cells)));
                                h_fill[idx] = std::max(h_fill[idx], fill);
                            }
                        }
                    }
                }

                total_particle_volume += (4.0f/3.0f) * 3.14159f * r_cells * r_cells * r_cells;
                particles_placed++;
                placed = true;
            }
        }

        if (!placed && (p % 50 == 0)) {
            printf("  Warning: Difficulty placing particle %d (packing limit reached)\n", p);
        }
    }

    // Calculate achieved packing
    float layer_volume_cells = float(nx) * float(ny) * float(z_powder_top - z_substrate_top);
    float achieved_packing = total_particle_volume / layer_volume_cells;

    printf("  Particles placed: %d / %d target\n", particles_placed, target_particles);
    printf("  Achieved packing: %.1f%% (target: %.1f%%)\n",
           achieved_packing * 100.0f, powder_config.target_packing * 100.0f);

    // Set fill level in solver
    solver.setFillLevel(h_fill.data());

    // ========================================================================
    // Simulation Loop
    // ========================================================================

    // Simulation parameters
    const float total_time = 50.0e-6f;      // 50 us simulation (enough for melting)
    const int total_steps = static_cast<int>(total_time / config.dt);  // 5000 steps
    const int output_interval = total_steps / 50;  // 50 VTK files
    const int print_interval = total_steps / 20;    // 20 status updates

    printf("\nSimulation Parameters:\n");
    printf("  Total time: %.0f us\n", total_time * 1e6f);
    printf("  Time step: %.2f ns\n", config.dt * 1e9f);
    printf("  Total steps: %d\n", total_steps);
    printf("  Output interval: %d steps\n", output_interval);

    // Host arrays for output
    std::vector<float> h_temperature(nx * ny * nz);
    std::vector<float> h_fill_level(nx * ny * nz);
    std::vector<float> h_liquid_frac(nx * ny * nz);
    std::vector<float> h_phase(nx * ny * nz);
    std::vector<float> h_ux(nx * ny * nz);
    std::vector<float> h_uy(nx * ny * nz);
    std::vector<float> h_uz(nx * ny * nz);

    printf("\n========================================\n");
    printf("Starting simulation...\n");
    printf("========================================\n\n");

    float max_T = 0.0f;
    float max_v = 0.0f;

    for (int step = 0; step <= total_steps; ++step) {
        float time = step * config.dt;

        // ====================================================================
        // Output VTK
        // ====================================================================
        if (step % output_interval == 0) {
            // Get device pointers
            const float* d_T = solver.getTemperature();
            const float* d_vx = solver.getVelocityX();
            const float* d_vy = solver.getVelocityY();
            const float* d_vz = solver.getVelocityZ();
            const float* d_f = solver.getFillLevel();
            const float* d_lf = solver.getLiquidFraction();

            int num_cells = nx * ny * nz;

            // Copy data to host
            cudaMemcpy(h_temperature.data(), d_T, num_cells * sizeof(float), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_ux.data(), d_vx, num_cells * sizeof(float), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_uy.data(), d_vy, num_cells * sizeof(float), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_uz.data(), d_vz, num_cells * sizeof(float), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_fill_level.data(), d_f, num_cells * sizeof(float), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_liquid_frac.data(), d_lf, num_cells * sizeof(float), cudaMemcpyDeviceToHost);

            // Compute phase state from temperature
            const float T_solidus = config.material.T_solidus;
            const float T_liquidus = config.material.T_liquidus;
            for (int i = 0; i < num_cells; ++i) {
                if (h_temperature[i] >= T_liquidus) {
                    h_phase[i] = 2.0f;  // Liquid
                } else if (h_temperature[i] >= T_solidus) {
                    h_phase[i] = 1.0f;  // Mushy
                } else {
                    h_phase[i] = 0.0f;  // Solid
                }
            }

            // Write VTK
            std::string filename = VTKWriter::getTimeSeriesFilename(
                output_dir + "/powder", step);

            VTKWriter::writeStructuredGridWithVectors(
                filename,
                h_temperature.data(),
                h_liquid_frac.data(),
                h_phase.data(),
                h_fill_level.data(),
                h_ux.data(), h_uy.data(), h_uz.data(),
                nx, ny, nz,
                dx, dx, dx
            );
        }

        // ====================================================================
        // Progress Output
        // ====================================================================
        if (step % print_interval == 0) {
            max_T = solver.getMaxTemperature();
            max_v = solver.getMaxVelocity();
            float mass = solver.getTotalMass();

            printf("[Step %6d / %d] t = %.2f us | T_max = %.0f K | v_max = %.2f m/s | mass = %.1f\n",
                   step, total_steps, time * 1e6f, max_T, max_v, mass);

            // Check for NaN
            if (solver.checkNaN()) {
                printf("\n[ERROR] NaN detected! Stopping simulation.\n");
                break;
            }
        }

        // ====================================================================
        // Time Step
        // ====================================================================
        if (step < total_steps) {
            solver.step();
        }
    }

    printf("\n========================================\n");
    printf("Simulation complete!\n");
    printf("========================================\n");
    printf("Final T_max: %.0f K\n", max_T);
    printf("Final v_max: %.2f m/s\n", max_v);
    printf("Output written to: %s\n", output_dir.c_str());

    return 0;
}
