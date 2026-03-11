/**
 * @file powder_bed.cu
 * @brief CUDA implementation of powder bed generator
 */

#include "physics/powder_bed.h"
#include "physics/vof_solver.h"
#include "utils/cuda_check.h"

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <stdexcept>
#include <algorithm>
#include <random>
#include <cmath>

namespace lbm {
namespace physics {

// ============================================================================
// PowderSizeDistribution Implementation
// ============================================================================

__device__ float PowderSizeDistribution::sampleDiameter(curandState* rng) const {
    // Sample from log-normal distribution
    float z = curand_normal(rng);
    float ln_sigma = logf(sigma_g);
    float ln_D = logf(D50) + ln_sigma * z;
    float D = expf(ln_D);

    // Clamp to [D_min, D_max]
    D = fmaxf(D_min, fminf(D_max, D));
    return D;
}

float PowderSizeDistribution::sampleDiameterHost(unsigned int& seed) const {
    // Simple LCG random number generator
    seed = seed * 1103515245 + 12345;
    float u1 = static_cast<float>((seed >> 16) & 0x7FFF) / 32767.0f;
    seed = seed * 1103515245 + 12345;
    float u2 = static_cast<float>((seed >> 16) & 0x7FFF) / 32767.0f;

    // Box-Muller transform for normal distribution
    float z = sqrtf(-2.0f * logf(u1 + 1e-10f)) * cosf(2.0f * 3.14159265f * u2);

    // Log-normal sample
    float ln_sigma = logf(sigma_g);
    float ln_D = logf(D50) + ln_sigma * z;
    float D = expf(ln_D);

    // Clamp to [D_min, D_max]
    D = std::max(D_min, std::min(D_max, D));
    return D;
}

// ============================================================================
// PowderBedConfig Implementation
// ============================================================================

PowderBedConfig::PowderBedConfig() {
    // Default Ti6Al4V parameters already set in declaration
    computeDerivedQuantities();
}

void PowderBedConfig::computeDerivedQuantities() {
    // Effective thermal conductivity (Zehner-Bauer-Schlunder)
    effective_k = computeZBSEffectiveConductivity(k_solid, k_gas, target_packing);

    // Effective absorption depth (Gusarov 2005)
    float mean_radius = size_dist.getMeanDiameter() / 2.0f;
    effective_absorption_depth = computePowderAbsorptionDepth(
        mean_radius, target_packing, particle_reflectivity);
}

// ============================================================================
// Utility Functions
// ============================================================================

float computeZBSEffectiveConductivity(float k_solid, float k_gas, float packing_density) {
    // Maxwell-Clausius-Mossotti effective medium model for spherical inclusions
    // This is valid for packed beds across all conductivity ratios, including
    // the high-ratio case (metal/gas ~1000) where the Zehner-Schlunder model
    // breaks down. Reference: Maxwell (1873), Garnett (1904).
    //
    // (k_eff - k_g)/(k_eff + 2*k_g) = phi_s * (k_s - k_g)/(k_s + 2*k_g)
    // Solving for k_eff: k_eff = k_g * (1 + 3*beta)/(1 - beta)
    // where beta = phi_s * (k_s - k_g)/(k_s + 2*k_g)

    float phi_s = packing_density;  // solid volume fraction
    float beta = phi_s * (k_solid - k_gas) / (k_solid + 2.0f * k_gas);

    // Guard against degenerate case (phi -> 1 with high k_ratio)
    if (beta >= 1.0f - 1e-6f) {
        return k_solid;
    }

    float k_eff = k_gas * (1.0f + 3.0f * beta) / (1.0f - beta);

    // Sanity check: must lie between k_gas and k_solid
    k_eff = std::max(k_gas, std::min(k_solid, k_eff));

    return k_eff;
}

float computePowderAbsorptionDepth(float particle_radius,
                                    float packing_density,
                                    float reflectivity) {
    // Gusarov & Kruth (2005) model
    // d_eff = R_particle * (1 - porosity) / (3 * (1 - reflectivity))

    float porosity = 1.0f - packing_density;

    // Avoid division by zero
    if (reflectivity >= 0.99f) {
        reflectivity = 0.99f;
    }

    float d_eff = particle_radius * (1.0f - porosity) / (3.0f * (1.0f - reflectivity));

    // Sanity check: must be positive (reflectivity < 1 is enforced above)
    d_eff = std::max(d_eff, 1.0e-9f);

    return d_eff;
}

// ============================================================================
// PowderBed Implementation
// ============================================================================

PowderBed::PowderBed(const PowderBedConfig& config, VOFSolver* vof)
    : config_(config), vof_(vof),
      actual_packing_(0.0f),
      nx_(0), ny_(0), nz_(0), dx_(0.0f),
      d_particle_x_(nullptr), d_particle_y_(nullptr),
      d_particle_z_(nullptr), d_particle_radius_(nullptr),
      num_particles_device_(0)
{
    // Recompute derived quantities in case config was modified
    config_.computeDerivedQuantities();
}

PowderBed::~PowderBed() {
    freeDeviceMemory();
}

void PowderBed::allocateDeviceMemory(int num_particles) {
    freeDeviceMemory();

    if (num_particles > 0) {
        CUDA_CHECK(cudaMalloc(&d_particle_x_, num_particles * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_particle_y_, num_particles * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_particle_z_, num_particles * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_particle_radius_, num_particles * sizeof(float)));
        num_particles_device_ = num_particles;

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            throw std::runtime_error("PowderBed: Failed to allocate device memory: " +
                                     std::string(cudaGetErrorString(err)));
        }
    }
}

void PowderBed::freeDeviceMemory() {
    if (d_particle_x_) { CUDA_CHECK(cudaFree(d_particle_x_)); d_particle_x_ = nullptr; }
    if (d_particle_y_) { CUDA_CHECK(cudaFree(d_particle_y_)); d_particle_y_ = nullptr; }
    if (d_particle_z_) { CUDA_CHECK(cudaFree(d_particle_z_)); d_particle_z_ = nullptr; }
    if (d_particle_radius_) { CUDA_CHECK(cudaFree(d_particle_radius_)); d_particle_radius_ = nullptr; }
    num_particles_device_ = 0;
}

void PowderBed::generate(int domain_nx, int domain_ny, int domain_nz, float dx) {
    nx_ = domain_nx;
    ny_ = domain_ny;
    nz_ = domain_nz;
    dx_ = dx;

    // Clear existing particles
    particles_.clear();

    // Generate based on method
    switch (config_.generation_method) {
        case PowderGenerationMethod::RANDOM_SEQUENTIAL:
            generateRandomSequential();
            break;
        case PowderGenerationMethod::RAIN_DEPOSITION:
            generateRainDeposition();
            break;
        case PowderGenerationMethod::REGULAR_PERTURBED:
            generateRegularPerturbed();
            break;
    }

    // Compute actual packing density
    float domain_area = (nx_ * dx_) * (ny_ * dx_);
    float layer_volume = domain_area * config_.layer_thickness;
    float particle_volume = getTotalParticleVolume();
    actual_packing_ = particle_volume / layer_volume;

    // Copy to device and update VOF
    if (!particles_.empty()) {
        copyParticlesToDevice();
        updateVOFFillLevel();
    }

    // Print summary
    printf("[PowderBed] Generated %d particles\n", getNumParticles());
    printf("[PowderBed] Actual packing density: %.2f%% (target: %.2f%%)\n",
           actual_packing_ * 100.0f, config_.target_packing * 100.0f);
    printf("[PowderBed] Effective k: %.4f W/(m*K)\n", config_.effective_k);
    printf("[PowderBed] Effective absorption depth: %.2f um\n",
           config_.effective_absorption_depth * 1e6f);
}

void PowderBed::regenerate(unsigned int new_seed) {
    config_.seed = new_seed;
    generate(nx_, ny_, nz_, dx_);
}

// ============================================================================
// Random Sequential Addition
// ============================================================================

void PowderBed::generateRandomSequential() {
    // Domain dimensions in physical units
    float Lx = nx_ * dx_;
    float Ly = ny_ * dx_;

    // Powder layer bounds
    float z_min = config_.substrate_height;

    // Target volume to achieve packing
    float layer_volume = Lx * Ly * config_.layer_thickness;
    float target_particle_volume = layer_volume * config_.target_packing;

    // Current state
    float current_volume = 0.0f;
    unsigned int seed = config_.seed;
    int next_id = 0;

    // Generate particles using RSA with periodic BCs in x,y.
    // Periodic BCs avoid the wasted margin from placing centers at [R, Lx-R],
    // enabling higher packing density in small domains.
    while (current_volume < target_particle_volume) {
        // Sample diameter
        float D = config_.size_dist.sampleDiameterHost(seed);
        float R = D / 2.0f;

        // Skip particles that don't fit in the z direction
        float z_range = config_.layer_thickness - 2.0f * R;
        if (z_range <= 0.0f) continue;

        // Try to place particle
        bool placed = false;
        for (int attempt = 0; attempt < config_.max_placement_attempts; ++attempt) {
            // Random (x, y) over full domain — periodic wrap handles collisions
            seed = seed * 1103515245 + 12345;
            float rx = static_cast<float>((seed >> 16) & 0x7FFF) / 32767.0f;
            seed = seed * 1103515245 + 12345;
            float ry = static_cast<float>((seed >> 16) & 0x7FFF) / 32767.0f;

            float x = rx * Lx;
            float y = ry * Ly;

            seed = seed * 1103515245 + 12345;
            float rz = static_cast<float>((seed >> 16) & 0x7FFF) / 32767.0f;
            float z = z_min + R + rz * z_range;

            // Create candidate particle
            Particle candidate;
            candidate.x = x;
            candidate.y = y;
            candidate.z = z;
            candidate.radius = R;
            candidate.id = next_id;
            candidate.is_melted = false;

            // Check collision with periodic x,y wrap
            if (!checkCollisionPeriodic(candidate, config_.min_gap, Lx, Ly)) {
                particles_.push_back(candidate);
                current_volume += candidate.volume();
                next_id++;
                placed = true;
                break;
            }
        }

        // If we can't place after many attempts, we're likely at max packing
        if (!placed) {
            printf("[PowderBed] Warning: Could not place particle after %d attempts\n",
                   config_.max_placement_attempts);
            printf("[PowderBed] Stopping at %.2f%% of target volume\n",
                   100.0f * current_volume / target_particle_volume);
            break;
        }
    }
}

// ============================================================================
// Rain Deposition (Placeholder)
// ============================================================================

void PowderBed::generateRainDeposition() {
    // TODO: Implement rain deposition algorithm
    // For now, fall back to random sequential
    printf("[PowderBed] Rain deposition not yet implemented, using random sequential\n");
    generateRandomSequential();
}

// ============================================================================
// Regular Perturbed (Placeholder)
// ============================================================================

void PowderBed::generateRegularPerturbed() {
    // Place particles on a hexagonally-stacked grid with exact periodic tiling.
    //
    // Grid design:
    //   - In-plane: square grid, nx = floor(Lx / (1.2*D50)), ny = floor(Ly / (1.2*D50))
    //     The actual spacing (Lx/nx, Ly/ny) is always >= nominal, so no overlaps.
    //   - In z: HCP layer stacking, dz = D50*sqrt(2/3) (< D50, so more layers fit).
    //   - Alternate layers are offset by (a_x/2, a_y/2) to break symmetry.
    //   - All positions are fully in-bounds; no fmod wrapping that could cause
    //     collisions at the domain boundary.
    //
    // Achieves ~33% packing for D50=20um in a 100x100x40um domain (2 layers of 16).

    const float Lx = nx_ * dx_;
    const float Ly = ny_ * dx_;
    const float z_min = config_.substrate_height;

    const float D50 = config_.size_dist.D50;
    const float R50 = D50 / 2.0f;

    // Nominal gap factor: 20% between touching spheres
    const float gap_factor = 1.2f;

    // Number of grid points that tile the domain without wrapping
    // Use floor so actual_a >= D50*gap_factor (no accidental touching)
    const int nx_grid = std::max(1, static_cast<int>(Lx / (D50 * gap_factor)));
    const int ny_grid = std::max(1, static_cast<int>(Ly / (D50 * gap_factor)));

    // Actual spacings (uniform within domain)
    const float a_x = Lx / nx_grid;
    const float a_y = Ly / ny_grid;

    // HCP z-spacing: sqrt(2/3)*D50 ~ 0.816*D50, smaller than D50 so more layers
    const float dz = D50 * std::sqrt(2.0f / 3.0f);

    int next_id = 0;
    int layer_idx = 0;
    float z = z_min + R50;

    while (z + R50 <= z_min + config_.layer_thickness) {
        // Alternate layers offset by half a cell in both directions
        const float x_offset = (layer_idx % 2) * a_x * 0.5f;
        const float y_offset = (layer_idx % 2) * a_y * 0.5f;

        for (int ix = 0; ix < nx_grid; ++ix) {
            for (int iy = 0; iy < ny_grid; ++iy) {
                float x = x_offset + ix * a_x;
                float y = y_offset + iy * a_y;

                // Keep in bounds (the offset can push the last column/row out slightly)
                if (x >= Lx - R50 * 0.01f) x = Lx - R50 * 0.01f;
                if (y >= Ly - R50 * 0.01f) y = Ly - R50 * 0.01f;

                Particle p;
                p.x = x;
                p.y = y;
                p.z = z;
                p.radius = R50;
                p.id = next_id++;
                p.is_melted = false;

                particles_.push_back(p);
            }
        }

        z += dz;
        layer_idx++;
    }
}

// ============================================================================
// Collision Detection
// ============================================================================

bool PowderBed::checkCollision(const Particle& p, float min_gap) const {
    for (const Particle& existing : particles_) {
        float dx = p.x - existing.x;
        float dy = p.y - existing.y;
        float dz = p.z - existing.z;
        float dist = sqrtf(dx*dx + dy*dy + dz*dz);
        float min_dist = p.radius + existing.radius + min_gap;

        if (dist < min_dist) {
            return true;  // Collision detected
        }
    }
    return false;
}

bool PowderBed::checkCollisionPeriodic(const Particle& p, float min_gap,
                                        float Lx, float Ly) const {
    for (const Particle& existing : particles_) {
        float ddx = p.x - existing.x;
        float ddy = p.y - existing.y;
        float ddz = p.z - existing.z;

        // Minimum image convention for x and y
        ddx -= Lx * roundf(ddx / Lx);
        ddy -= Ly * roundf(ddy / Ly);

        float dist = sqrtf(ddx*ddx + ddy*ddy + ddz*ddz);
        float min_dist = p.radius + existing.radius + min_gap;

        if (dist < min_dist) {
            return true;
        }
    }
    return false;
}

// ============================================================================
// VOF Integration
// ============================================================================

void PowderBed::copyParticlesToDevice() {
    int n = static_cast<int>(particles_.size());
    if (n == 0) return;

    // Allocate device memory
    allocateDeviceMemory(n);

    // Create host arrays
    std::vector<float> h_x(n), h_y(n), h_z(n), h_r(n);
    for (int i = 0; i < n; ++i) {
        h_x[i] = particles_[i].x;
        h_y[i] = particles_[i].y;
        h_z[i] = particles_[i].z;
        h_r[i] = particles_[i].radius;
    }

    // Copy to device
    CUDA_CHECK(cudaMemcpy(d_particle_x_, h_x.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_particle_y_, h_y.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_particle_z_, h_z.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_particle_radius_, h_r.data(), n * sizeof(float), cudaMemcpyHostToDevice));
}

void PowderBed::updateVOFFillLevel() {
    if (!vof_ || particles_.empty()) return;

    // Launch kernel to initialize fill_level
    dim3 blockSize(8, 8, 8);
    dim3 gridSize((nx_ + blockSize.x - 1) / blockSize.x,
                  (ny_ + blockSize.y - 1) / blockSize.y,
                  (nz_ + blockSize.z - 1) / blockSize.z);

    float interface_width = 2.0f;  // Interface smoothing in cells

    initializeParticleFillLevelKernel<<<gridSize, blockSize>>>(
        vof_->getFillLevel(),
        d_particle_x_, d_particle_y_, d_particle_z_, d_particle_radius_,
        num_particles_device_,
        dx_, interface_width,
        nx_, ny_, nz_);
    CUDA_CHECK_KERNEL();

    CUDA_CHECK(cudaDeviceSynchronize());

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("PowderBed: Fill level kernel failed: " +
                                 std::string(cudaGetErrorString(err)));
    }

    // Update VOF cell flags
    vof_->convertCells();
}

// ============================================================================
// Thermal Properties
// ============================================================================

void PowderBed::initializeThermalConductivity(float* d_thermal_conductivity,
                                               float k_bulk,
                                               int nx, int ny, int nz, float dx) const {
    float z_min, z_max;
    getPowderLayerBounds(z_min, z_max);

    dim3 blockSize(8, 8, 8);
    dim3 gridSize((nx + blockSize.x - 1) / blockSize.x,
                  (ny + blockSize.y - 1) / blockSize.y,
                  (nz + blockSize.z - 1) / blockSize.z);

    initializePowderThermalKernel<<<gridSize, blockSize>>>(
        d_thermal_conductivity,
        config_.effective_k, k_bulk,
        z_min, z_max, dx,
        nx, ny, nz);
    CUDA_CHECK_KERNEL();

    CUDA_CHECK(cudaDeviceSynchronize());
}

// ============================================================================
// Diagnostics
// ============================================================================

float PowderBed::getTotalParticleVolume() const {
    float total = 0.0f;
    for (const Particle& p : particles_) {
        total += p.volume();
    }
    return total;
}

void PowderBed::printStatistics() const {
    if (particles_.empty()) {
        printf("[PowderBed] No particles generated\n");
        return;
    }

    // Compute statistics
    float D_min = 1e10f, D_max = 0.0f, D_sum = 0.0f;
    for (const Particle& p : particles_) {
        float D = 2.0f * p.radius;
        D_min = std::min(D_min, D);
        D_max = std::max(D_max, D);
        D_sum += D;
    }
    float D_mean = D_sum / particles_.size();

    // Compute standard deviation
    float D_var = 0.0f;
    for (const Particle& p : particles_) {
        float D = 2.0f * p.radius;
        D_var += (D - D_mean) * (D - D_mean);
    }
    D_var /= particles_.size();
    float D_std = sqrtf(D_var);

    printf("\n========================================\n");
    printf("Powder Bed Statistics\n");
    printf("========================================\n");
    printf("Number of particles: %d\n", getNumParticles());
    printf("Diameter range: %.1f - %.1f um\n", D_min * 1e6f, D_max * 1e6f);
    printf("Mean diameter: %.1f um\n", D_mean * 1e6f);
    printf("Std deviation: %.1f um\n", D_std * 1e6f);
    printf("Target D50: %.1f um\n", config_.size_dist.D50 * 1e6f);
    printf("Actual packing: %.2f%%\n", actual_packing_ * 100.0f);
    printf("Target packing: %.2f%%\n", config_.target_packing * 100.0f);
    printf("Effective k: %.4f W/(m*K)\n", config_.effective_k);
    printf("Effective absorption: %.2f um\n", config_.effective_absorption_depth * 1e6f);
    printf("========================================\n\n");
}

bool PowderBed::verifyNoOverlaps() const {
    for (size_t i = 0; i < particles_.size(); ++i) {
        for (size_t j = i + 1; j < particles_.size(); ++j) {
            float dx = particles_[i].x - particles_[j].x;
            float dy = particles_[i].y - particles_[j].y;
            float dz = particles_[i].z - particles_[j].z;
            float dist = sqrtf(dx*dx + dy*dy + dz*dz);
            float min_dist = particles_[i].radius + particles_[j].radius;

            if (dist < min_dist * 0.99f) {  // Allow 1% tolerance
                printf("[PowderBed] Overlap detected: particle %d and %d\n",
                       particles_[i].id, particles_[j].id);
                return false;
            }
        }
    }
    return true;
}

void PowderBed::getPowderLayerBounds(float& z_min, float& z_max) const {
    z_min = config_.substrate_height;
    z_max = z_min + config_.layer_thickness;
}

// ============================================================================
// CUDA Kernels
// ============================================================================

__global__ void initializeParticleFillLevelKernel(
    float* fill_level,
    const float* particle_x,
    const float* particle_y,
    const float* particle_z,
    const float* particle_radius,
    int num_particles,
    float dx,
    float interface_width,
    int nx, int ny, int nz)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i >= nx || j >= ny || k >= nz) return;

    int idx = i + nx * (j + ny * k);

    // Physical position of grid cell center
    float x = (i + 0.5f) * dx;
    float y = (j + 0.5f) * dx;
    float z = (k + 0.5f) * dx;

    // Find maximum fill level from all particles
    float max_fill = 0.0f;

    for (int p = 0; p < num_particles; ++p) {
        // Distance from particle center
        float px = particle_x[p];
        float py = particle_y[p];
        float pz = particle_z[p];
        float pr = particle_radius[p];

        float ddx = x - px;
        float ddy = y - py;
        float ddz = z - pz;
        float dist = sqrtf(ddx*ddx + ddy*ddy + ddz*ddz);

        // Smooth fill level using tanh profile
        // fill = 0.5 * (1 - tanh((dist - R) / width))
        float scaled_dist = (dist - pr) / (interface_width * dx);
        float fill = 0.5f * (1.0f - tanhf(scaled_dist));

        max_fill = fmaxf(max_fill, fill);
    }

    fill_level[idx] = fminf(1.0f, max_fill);
}

__global__ void computeLocalThermalConductivityKernel(
    float* k_effective,
    const float* fill_level,
    float k_metal,
    float k_gas,
    int num_cells)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_cells) return;

    float f = fill_level[idx];

    // Simple linear mixing
    k_effective[idx] = f * k_metal + (1.0f - f) * k_gas;
}

__global__ void initializePowderThermalKernel(
    float* k_field,
    float k_powder,
    float k_bulk,
    float powder_z_min,
    float powder_z_max,
    float dx,
    int nx, int ny, int nz)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i >= nx || j >= ny || k >= nz) return;

    int idx = i + nx * (j + ny * k);

    // Physical z position
    float z = (k + 0.5f) * dx;

    // Check if in powder layer
    if (z >= powder_z_min && z <= powder_z_max) {
        k_field[idx] = k_powder;
    } else {
        k_field[idx] = k_bulk;
    }
}

} // namespace physics
} // namespace lbm
