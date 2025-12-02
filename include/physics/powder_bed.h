/**
 * @file powder_bed.h
 * @brief Powder bed generator and manager for LPBF simulations
 *
 * This module provides:
 * - Powder particle generation with configurable size distribution
 * - VOF fill_level initialization from particle arrangement
 * - Effective thermal property computation for powder layer
 * - Integration with LaserSource for modified absorption
 *
 * Physical model:
 * - Particles represented as spheres with log-normal size distribution
 * - Random sequential addition for packing (~55% density achievable)
 * - Inter-particle gaps treated as gas-filled (argon)
 * - Effective thermal conductivity via Zehner-Bauer-Schlunder model
 * - Modified Beer-Lambert absorption for powder (Gusarov 2005)
 *
 * References:
 * - Khairallah et al. (2016) Acta Materialia - Mesoscale LPBF simulation
 * - Gusarov & Kruth (2005) Int. J. Heat Mass Transfer - Powder optics
 * - Zehner & Schlunder (1970) - Effective thermal conductivity
 */

#pragma once

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <vector>
#include <memory>

namespace lbm {
namespace physics {

// Forward declarations
class VOFSolver;

// ============================================================================
// Data Structures
// ============================================================================

/**
 * @brief Single powder particle representation
 */
struct Particle {
    float x, y, z;      ///< Center position [m]
    float radius;       ///< Radius [m]
    int id;             ///< Unique identifier
    bool is_melted;     ///< Melting state (for tracking)

    /// Compute particle volume
    __host__ __device__ float volume() const {
        return (4.0f / 3.0f) * 3.14159265f * radius * radius * radius;
    }

    /// Check if point is inside particle
    __host__ __device__ bool contains(float px, float py, float pz) const {
        float dx = px - x;
        float dy = py - y;
        float dz = pz - z;
        return (dx*dx + dy*dy + dz*dz) <= (radius * radius);
    }

    /// Compute distance from particle surface (negative inside)
    __host__ __device__ float signedDistance(float px, float py, float pz) const {
        float dx = px - x;
        float dy = py - y;
        float dz = pz - z;
        return sqrtf(dx*dx + dy*dy + dz*dz) - radius;
    }
};

/**
 * @brief Powder size distribution parameters (log-normal)
 *
 * The log-normal distribution is standard for gas-atomized metal powders.
 * PDF: f(D) = 1/(D*sigma*sqrt(2*pi)) * exp(-(ln(D)-ln(D50))^2 / (2*sigma^2))
 */
struct PowderSizeDistribution {
    float D50 = 30.0e-6f;       ///< Median diameter [m]
    float sigma_g = 1.4f;       ///< Geometric standard deviation [-]
    float D_min = 15.0e-6f;     ///< Minimum diameter [m]
    float D_max = 45.0e-6f;     ///< Maximum diameter [m]

    /// Sample diameter from distribution (device function)
    __device__ float sampleDiameter(curandState* rng) const;

    /// Sample diameter from distribution (host function using C++ random)
    float sampleDiameterHost(unsigned int& seed) const;

    /// Get mean diameter (for log-normal: D_mean = D50 * exp(sigma^2/2))
    float getMeanDiameter() const {
        float ln_sigma = logf(sigma_g);
        return D50 * expf(ln_sigma * ln_sigma / 2.0f);
    }
};

/**
 * @brief Powder bed generation method
 */
enum class PowderGenerationMethod {
    RANDOM_SEQUENTIAL,    ///< Simple random placement with collision check
    RAIN_DEPOSITION,      ///< Drop particles from above with settling
    REGULAR_PERTURBED     ///< Regular lattice with random perturbation
};

/**
 * @brief Powder bed configuration parameters
 */
struct PowderBedConfig {
    // ========================================================================
    // Geometric Parameters
    // ========================================================================

    float layer_thickness = 40.0e-6f;     ///< Powder layer thickness [m]
    float target_packing = 0.55f;         ///< Target packing density [-]
    float substrate_height = 0.0f;        ///< Z offset from domain bottom [m]

    // ========================================================================
    // Size Distribution
    // ========================================================================

    PowderSizeDistribution size_dist;     ///< Particle size distribution

    // ========================================================================
    // Generation Parameters
    // ========================================================================

    PowderGenerationMethod generation_method = PowderGenerationMethod::RANDOM_SEQUENTIAL;
    unsigned int seed = 42;               ///< Random seed for reproducibility
    int max_placement_attempts = 1000;    ///< Max attempts per particle
    float min_gap = 0.0f;                 ///< Minimum inter-particle gap [m]

    // ========================================================================
    // Thermal Properties
    // ========================================================================

    float k_solid = 25.0f;                ///< Solid particle conductivity [W/(m*K)]
    float k_gas = 0.018f;                 ///< Inter-particle gas (argon) [W/(m*K)]

    // ========================================================================
    // Laser Absorption
    // ========================================================================

    float particle_reflectivity = 0.65f;  ///< Single particle reflectivity [-]

    // ========================================================================
    // Computed/Derived Quantities
    // ========================================================================

    float effective_k;                    ///< Effective thermal conductivity [W/(m*K)]
    float effective_absorption_depth;     ///< Effective laser penetration [m]

    /**
     * @brief Compute derived quantities from input parameters
     * @note Must be called after setting other parameters
     */
    void computeDerivedQuantities();

    /**
     * @brief Default constructor with Ti6Al4V parameters
     */
    PowderBedConfig();
};

// ============================================================================
// PowderBed Class
// ============================================================================

/**
 * @brief Powder bed generator and manager
 *
 * This class handles:
 * 1. Generating particle arrangements using specified algorithm
 * 2. Converting particle geometry to VOF fill_level field
 * 3. Computing effective thermal properties for powder region
 * 4. Providing particle statistics and diagnostics
 *
 * Usage:
 * @code
 * PowderBedConfig config;
 * config.layer_thickness = 40e-6f;
 * config.target_packing = 0.55f;
 *
 * PowderBed powder(config, vof_solver);
 * powder.generate(nx, ny, nz, dx);
 * @endcode
 */
class PowderBed {
public:
    /**
     * @brief Constructor
     * @param config Powder bed configuration
     * @param vof Pointer to VOF solver (for fill_level initialization)
     */
    PowderBed(const PowderBedConfig& config, VOFSolver* vof);

    /**
     * @brief Destructor
     */
    ~PowderBed();

    // ========================================================================
    // Generation
    // ========================================================================

    /**
     * @brief Generate powder bed and initialize VOF fill_level
     * @param domain_nx Grid dimension in x
     * @param domain_ny Grid dimension in y
     * @param domain_nz Grid dimension in z
     * @param dx Grid spacing [m]
     * @note This is the main entry point for powder bed creation
     */
    void generate(int domain_nx, int domain_ny, int domain_nz, float dx);

    /**
     * @brief Regenerate powder bed with new seed
     * @param new_seed New random seed
     * @note Useful for stochastic studies
     */
    void regenerate(unsigned int new_seed);

    // ========================================================================
    // Thermal Property Access
    // ========================================================================

    /**
     * @brief Initialize thermal conductivity field for powder region
     * @param d_thermal_conductivity Device array to modify [size: nx*ny*nz]
     * @param k_bulk Bulk metal conductivity [W/(m*K)]
     * @note Sets k_eff in powder layer, k_bulk elsewhere
     */
    void initializeThermalConductivity(float* d_thermal_conductivity,
                                        float k_bulk,
                                        int nx, int ny, int nz, float dx) const;

    /**
     * @brief Get effective thermal conductivity for powder layer
     * @return k_eff [W/(m*K)]
     */
    float getEffectiveThermalConductivity() const { return config_.effective_k; }

    /**
     * @brief Get effective laser absorption depth for powder
     * @return d_eff [m]
     */
    float getEffectiveAbsorptionDepth() const { return config_.effective_absorption_depth; }

    // ========================================================================
    // Particle Access
    // ========================================================================

    /**
     * @brief Get particle list (const)
     */
    const std::vector<Particle>& getParticles() const { return particles_; }

    /**
     * @brief Get number of particles
     */
    int getNumParticles() const { return static_cast<int>(particles_.size()); }

    /**
     * @brief Get total particle volume
     */
    float getTotalParticleVolume() const;

    /**
     * @brief Get actual achieved packing density
     */
    float getActualPacking() const { return actual_packing_; }

    // ========================================================================
    // Diagnostics
    // ========================================================================

    /**
     * @brief Compute and print particle-level statistics
     * @note Prints: particle count, size distribution, packing density, etc.
     */
    void printStatistics() const;

    /**
     * @brief Verify no particle overlaps exist
     * @return True if all particles are non-overlapping
     */
    bool verifyNoOverlaps() const;

    /**
     * @brief Get bounding box of powder layer
     * @param z_min Output: minimum z coordinate [m]
     * @param z_max Output: maximum z coordinate [m]
     */
    void getPowderLayerBounds(float& z_min, float& z_max) const;

    // ========================================================================
    // Configuration Access
    // ========================================================================

    const PowderBedConfig& getConfig() const { return config_; }

private:
    PowderBedConfig config_;
    VOFSolver* vof_;            ///< Non-owning pointer to VOF solver

    std::vector<Particle> particles_;
    float actual_packing_;

    // Domain info (stored after generate())
    int nx_, ny_, nz_;
    float dx_;

    // Device memory for particle data (for GPU kernels)
    float* d_particle_x_;
    float* d_particle_y_;
    float* d_particle_z_;
    float* d_particle_radius_;
    int num_particles_device_;

    // ========================================================================
    // Generation Methods
    // ========================================================================

    /**
     * @brief Random sequential addition algorithm
     * @note Achieves ~55% packing density
     */
    void generateRandomSequential();

    /**
     * @brief Rain deposition algorithm
     * @note Achieves ~60% packing density
     */
    void generateRainDeposition();

    /**
     * @brief Regular lattice with perturbation
     * @note Good for testing, controllable packing
     */
    void generateRegularPerturbed();

    // ========================================================================
    // Helper Functions
    // ========================================================================

    /**
     * @brief Check if particle collides with existing particles
     * @param p Candidate particle
     * @param min_gap Minimum gap required between particles [m]
     * @return True if collision detected
     */
    bool checkCollision(const Particle& p, float min_gap) const;

    /**
     * @brief Update VOF fill_level field from particle list
     */
    void updateVOFFillLevel();

    /**
     * @brief Copy particle data to device memory
     */
    void copyParticlesToDevice();

    /**
     * @brief Free device memory
     */
    void freeDeviceMemory();

    /**
     * @brief Allocate device memory
     */
    void allocateDeviceMemory(int num_particles);
};

// ============================================================================
// CUDA Kernels
// ============================================================================

/**
 * @brief Initialize fill_level field from particle spheres
 *
 * For each grid cell, computes fill_level as smooth union of all particles:
 * fill_level = max over all particles of: 0.5*(1 - tanh((dist - R) / width))
 *
 * @param fill_level Output fill_level field [nx*ny*nz]
 * @param particle_x Particle center x coordinates [num_particles]
 * @param particle_y Particle center y coordinates [num_particles]
 * @param particle_z Particle center z coordinates [num_particles]
 * @param particle_radius Particle radii [num_particles]
 * @param num_particles Number of particles
 * @param dx Grid spacing [m]
 * @param interface_width Interface smoothing width [cells]
 * @param nx, ny, nz Grid dimensions
 */
__global__ void initializeParticleFillLevelKernel(
    float* fill_level,
    const float* particle_x,
    const float* particle_y,
    const float* particle_z,
    const float* particle_radius,
    int num_particles,
    float dx,
    float interface_width,
    int nx, int ny, int nz);

/**
 * @brief Compute local effective thermal conductivity based on fill_level
 *
 * Uses simple linear mixing: k_local = f * k_metal + (1-f) * k_gas
 *
 * @param k_effective Output conductivity field [num_cells]
 * @param fill_level Fill level field [num_cells]
 * @param k_metal Solid metal conductivity [W/(m*K)]
 * @param k_gas Gas conductivity [W/(m*K)]
 * @param num_cells Total number of grid cells
 */
__global__ void computeLocalThermalConductivityKernel(
    float* k_effective,
    const float* fill_level,
    float k_metal,
    float k_gas,
    int num_cells);

/**
 * @brief Initialize thermal conductivity field for powder region
 *
 * Sets:
 * - k = k_powder (effective) where z is in powder layer
 * - k = k_bulk elsewhere
 *
 * @param k_field Output conductivity field [nx*ny*nz]
 * @param k_powder Effective powder conductivity [W/(m*K)]
 * @param k_bulk Bulk metal conductivity [W/(m*K)]
 * @param powder_z_min Powder layer bottom [m]
 * @param powder_z_max Powder layer top [m]
 * @param dx Grid spacing [m]
 * @param nx, ny, nz Grid dimensions
 */
__global__ void initializePowderThermalKernel(
    float* k_field,
    float k_powder,
    float k_bulk,
    float powder_z_min,
    float powder_z_max,
    float dx,
    int nx, int ny, int nz);

// ============================================================================
// Utility Functions
// ============================================================================

/**
 * @brief Compute effective thermal conductivity using Zehner-Bauer-Schlunder model
 *
 * This is a well-validated model for packed beds of spheres.
 *
 * @param k_solid Solid particle conductivity [W/(m*K)]
 * @param k_gas Gas conductivity [W/(m*K)]
 * @param packing_density Solid volume fraction [-]
 * @return Effective conductivity [W/(m*K)]
 */
float computeZBSEffectiveConductivity(float k_solid, float k_gas, float packing_density);

/**
 * @brief Compute effective laser absorption depth for powder (Gusarov model)
 *
 * d_eff = R_particle * (1 - porosity) / (3 * (1 - reflectivity))
 *
 * @param particle_radius Mean particle radius [m]
 * @param packing_density Solid volume fraction [-]
 * @param reflectivity Single particle reflectivity [-]
 * @return Effective absorption depth [m]
 */
float computePowderAbsorptionDepth(float particle_radius,
                                    float packing_density,
                                    float reflectivity);

} // namespace physics
} // namespace lbm
