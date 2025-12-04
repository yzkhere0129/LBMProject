# Powder Bed Implementation Architecture Design

## LBM-CFD Platform: LPBF Powder Layer Module

**Version**: 1.0
**Date**: 2025-11-22
**Author**: LBM-CFD Architecture Team
**Status**: Design Document

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Physical Model Design](#2-physical-model-design)
3. [VOF Implementation Strategy](#3-vof-implementation-strategy)
4. [Thermal Physics Considerations](#4-thermal-physics-considerations)
5. [Melting Dynamics](#5-melting-dynamics)
6. [Code Architecture](#6-code-architecture)
7. [Computational Cost Analysis](#7-computational-cost-analysis)
8. [Implementation Roadmap](#8-implementation-roadmap)
9. [Validation Strategy](#9-validation-strategy)
10. [References](#10-references)

---

## 1. Executive Summary

### 1.1 Background

Current simulations use flat metal surfaces, but real LPBF processes involve laser scanning over powder beds. The powder layer introduces:

- **Geometric complexity**: Discrete spherical particles with gaps
- **Thermal heterogeneity**: Reduced effective thermal conductivity
- **Melting dynamics**: Particle-by-particle melting and coalescence
- **Porosity effects**: Trapped gas, keyhole formation

### 1.2 Design Goals

| Goal | Priority | Description |
|------|----------|-------------|
| Physical fidelity | High | Capture particle-scale phenomena |
| Computational efficiency | High | Maintain practical simulation times |
| Modular integration | Medium | Clean interface with existing VOF solver |
| Extensibility | Medium | Support different powder configurations |

### 1.3 Key Design Decisions

1. **VOF-based representation**: Use fill_level field to represent discrete particles
2. **Hybrid initialization**: Random particle placement with collision detection
3. **Effective property models**: Account for powder thermal properties
4. **Phased implementation**: Start simple, add complexity incrementally

---

## 2. Physical Model Design

### 2.1 Powder Particle Size Distribution

**Ti6Al4V Typical Parameters**:

| Parameter | Value | Source |
|-----------|-------|--------|
| Mean diameter (D50) | 30 um | Typical gas-atomized |
| Range | 15-45 um | Standard LPBF grade |
| Distribution | Log-normal | ISO 13320 |
| Standard deviation | sigma_g = 1.3-1.5 | Typical value |

**Mathematical Model**:

```
f(D) = 1/(D * sigma * sqrt(2*pi)) * exp(-(ln(D) - ln(D50))^2 / (2*sigma^2))

Where:
  D: Particle diameter [m]
  D50: Median diameter [m]
  sigma: Log-standard deviation [-]
```

**Implementation**:

```cpp
struct PowderSizeDistribution {
    float D50;           // Median diameter [m]
    float sigma_g;       // Geometric standard deviation [-]
    float D_min;         // Minimum diameter [m]
    float D_max;         // Maximum diameter [m]

    // Sample from distribution
    float sampleDiameter(curandState* rng) const {
        float z = curand_normal(rng);
        return D50 * exp(sigma_g * z);
    }
};
```

### 2.2 Powder Layer Geometry

**Layer Parameters**:

| Parameter | Typical Value | Design Choice |
|-----------|---------------|---------------|
| Layer thickness | 30-50 um | 40 um default |
| Packing density | 50-60% | 55% (random packing) |
| Substrate gap | 0 | Particles sit on substrate |

**Packing Configurations**:

1. **Random Loose Packing (RLP)**: phi ~ 0.60
2. **Random Close Packing (RCP)**: phi ~ 0.64
3. **Ordered (FCC/HCP)**: phi ~ 0.74 (unrealistic for powder)

**Selected Approach**: Random packing with collision avoidance

### 2.3 Interparticle Gap Treatment

**Options**:

| Option | Thermal | Fluid | Complexity |
|--------|---------|-------|------------|
| Gas-filled | Conduction+radiation | Flow possible | Medium |
| Vacuum | Radiation only | No flow | Low |
| Effective medium | Homogenized | Averaged | Low |

**Recommendation**: Gas-filled gaps with argon properties (realistic LPBF)

```cpp
struct InterparticleGas {
    float thermal_conductivity = 0.018f;  // Argon at 300K [W/(m*K)]
    float density = 1.784f;               // [kg/m^3]
    float specific_heat = 520.0f;         // [J/(kg*K)]
    float viscosity = 2.27e-5f;           // [Pa*s]
};
```

---

## 3. VOF Implementation Strategy

### 3.1 Representing Discrete Particles with VOF

**Challenge**: VOF is designed for continuous interfaces, not discrete objects.

**Solution**: Initialize fill_level field with superimposed spheres:

```
fill_level(x,y,z) = max_i{ sphere_fill(x,y,z; center_i, radius_i) }

where sphere_fill = 0.5 * (1 - tanh((|r - center| - radius) / interface_width))
```

**Key Insight**: Each particle is a "droplet" in VOF. Merging is handled naturally.

### 3.2 Initialization Strategies

#### Option A: Random Sequential Addition (Recommended for Phase 1)

```cpp
Algorithm: RandomSequentialAddition
Input: domain, layer_thickness, packing_target, size_distribution
Output: particle_list, fill_level_field

1. Initialize fill_level = 0 everywhere
2. target_volume = layer_thickness * domain_area * packing_target
3. current_volume = 0
4. While current_volume < target_volume:
   a. Sample diameter from size_distribution
   b. Generate random (x, y) position
   c. Compute z from substrate + radius
   d. Check collision with existing particles
   e. If no collision:
      - Add particle to list
      - Update fill_level field
      - current_volume += particle_volume
5. Return particle_list, fill_level
```

**Pros**: Simple, robust
**Cons**: Limited packing density (max ~55%)

#### Option B: Rain Algorithm (Higher Packing)

```cpp
Algorithm: RainDeposition
Input: domain, layer_thickness, size_distribution
Output: particle_list, fill_level_field

1. Generate particles at random (x,y) above domain
2. For each particle:
   a. Drop vertically until collision
   b. Apply gravity settling (optional: roll to local minimum)
   c. Record final position
3. Convert particle list to fill_level field
```

**Pros**: Higher packing density (~60%)
**Cons**: More complex, requires collision physics

#### Option C: Regular Array with Perturbation (Testing)

```cpp
Algorithm: RegularWithPerturbation
Input: domain, particle_diameter, perturbation_amplitude
Output: particle_list, fill_level_field

1. Create regular lattice of particles (BCC or FCC)
2. Perturb each position by random vector (magnitude < particle_radius)
3. Remove overlapping particles
4. Convert to fill_level field
```

**Pros**: Controllable, reproducible
**Cons**: May not represent real powder

### 3.3 Resolution Requirements

**Minimum cells per particle diameter**:

| Resolution | Cells/D | Quality | Use Case |
|------------|---------|---------|----------|
| Coarse | 4 | Poor | Quick tests |
| Medium | 8 | Acceptable | Standard runs |
| Fine | 16 | Good | Validation |
| High | 32+ | Excellent | Publication |

**Calculation for Ti6Al4V powder (D=30um)**:

```
For 8 cells/D: dx = 30um / 8 = 3.75um
For 16 cells/D: dx = 30um / 16 = 1.875um

Layer thickness 40um:
  At dx=3.75um: ~10 cells in z
  At dx=1.875um: ~21 cells in z
```

**Recommendation**: Start with dx = 2um (15 cells per D30 particle)

### 3.4 Interface Reconstruction for Particles

**Current Implementation**: Central difference gradient

**Enhancement for Particles**: Height function method works better for spheres

```cpp
// Curvature for sphere should be 2/R
// Test: Initialize single particle, check curvature
float expected_curvature = 2.0f / particle_radius;
float measured_curvature = vof_.getCurvature()[center_idx];
float error = fabs(measured_curvature - expected_curvature) / expected_curvature;
// Should be < 30% for 8 cells/D
```

---

## 4. Thermal Physics Considerations

### 4.1 Effective Thermal Conductivity

**Challenge**: Powder bed has much lower thermal conductivity than bulk metal.

**Model**: Effective medium theory (Zehner-Bauer-Schlunder)

```
k_eff = k_gas * [(1-sqrt(1-phi)) + sqrt(1-phi) * f(B, k_s/k_g)]

Where:
  phi: Porosity (1 - packing_density)
  k_gas: Gas thermal conductivity
  k_s: Solid thermal conductivity
  B: Shape factor (sqrt(1.25*(phi/(1-phi))^(10/9)) for spheres)
```

**Typical Values for Ti6Al4V Powder in Argon**:

| Condition | k_eff [W/(m*K)] | Ratio to Bulk |
|-----------|-----------------|---------------|
| Room temperature | 0.2 - 0.5 | 1-2% |
| At melting | 0.5 - 1.0 | 2-5% |
| Bulk Ti6Al4V | 20-35 | 100% |

**Implementation**:

```cpp
struct PowderThermalProperties {
    float k_solid;            // Solid particle conductivity [W/(m*K)]
    float k_gas;              // Gas conductivity [W/(m*K)]
    float packing_density;    // Solid fraction [-]

    float getEffectiveConductivity() const {
        float phi = 1.0f - packing_density;  // Porosity
        float B = sqrt(1.25f * pow(phi / (1.0f - phi), 10.0f/9.0f));
        float ratio = k_solid / k_gas;

        // Zehner-Bauer-Schlunder correlation
        float term1 = 1.0f - sqrt(1.0f - phi);
        float term2 = sqrt(1.0f - phi) * 2.0f / (1.0f - B/ratio) *
                      (((1.0f - B/ratio)/(1.0f - B)) * log(ratio/B) -
                       (B - 1.0f)/(1.0f - B));

        return k_gas * (term1 + term2);
    }
};
```

### 4.2 Interparticle Thermal Resistance

**Contact Conductance Model**:

Particles touch at contact points with finite conductance:

```
h_contact = 2 * k_s * a_contact / (pi * R^2)

Where:
  a_contact: Contact radius (depends on pressure, material)
  R: Particle radius
```

**Simplified Approach**: Use VOF-weighted conductivity

```cpp
// In thermal solver, compute local conductivity based on fill_level
float k_local = fill_level * k_metal + (1.0f - fill_level) * k_gas;
```

### 4.3 Laser Penetration in Powder Layer

**Challenge**: Laser light undergoes multiple scattering in powder.

**Model Options**:

1. **Simple Beer-Lambert** (current): Single exponential decay
2. **Effective Absorption**: Modified absorption depth for powder
3. **Ray Tracing**: Full multiple scattering (computationally expensive)

**Recommended Approach**: Modified Beer-Lambert with effective depth

```cpp
// Effective penetration depth for powder (Gusarov & Kruth 2005)
// d_eff = R_p * (1 - phi) / (3 * (1 - reflectivity))
// Where R_p is particle radius, phi is porosity

float getEffectivePenetrationDepth(float particle_radius,
                                    float packing_density,
                                    float reflectivity) {
    float phi = 1.0f - packing_density;
    return particle_radius * (1.0f - phi) / (3.0f * (1.0f - reflectivity));
}

// For Ti6Al4V: R_p=15um, phi=0.45, r=0.65
// d_eff = 15e-6 * 0.55 / (3 * 0.35) = 7.9 um
```

**Integration with LaserSource**:

```cpp
struct LaserSource {
    // ... existing parameters ...

    // Powder-specific parameters
    bool is_powder_bed;
    float powder_packing_density;
    float effective_absorption_depth;  // Computed from powder properties

    float computeVolumetricHeatSource(float x, float y, float z) const {
        if (is_powder_bed) {
            // Use effective absorption for powder
            return computePowderAbsorption(x, y, z);
        }
        return computeBulkAbsorption(x, y, z);
    }
};
```

---

## 5. Melting Dynamics

### 5.1 Particle Melting Process

**Physical Sequence**:

1. **Heating**: Particle surface heats first
2. **Melting initiation**: Surface reaches T_liquidus
3. **Inward melting**: Solid-liquid interface moves inward
4. **Full melting**: Particle becomes liquid droplet
5. **Coalescence**: Adjacent liquid particles merge
6. **Melt pool formation**: Connected liquid region

**VOF Representation**:

- Before melting: Each particle is distinct VOF region (fill_level from sphere)
- During melting: Liquid fraction from thermal solver determines mobility
- After melting: VOF interfaces naturally merge (handled by advection)

### 5.2 Coalescence Mechanism

**Natural Handling by VOF**:

When two adjacent particles melt:
1. Their fill_level fields overlap at contact point
2. VOF advection smooths the interface
3. Surface tension minimizes total surface area
4. Result: Single merged droplet

**No Special Treatment Needed**: This is a strength of the VOF approach.

### 5.3 Marangoni Forces on Particles

**Effect**: Temperature gradient creates surface tension gradient

**Physical Behavior**:
- Hot side (laser center): Lower surface tension
- Cold side (periphery): Higher surface tension
- Flow direction: Hot -> Cold (for metals with dsigma/dT < 0)

**Result**: Outward flow from melt pool center

**Particle-Scale Considerations**:

```cpp
// Marangoni force already implemented in marangoni.cu
// Key consideration: Interface detection must work for particle surfaces

// Current hybrid detection (from marangoni.cu):
bool is_vof_interface = (f >= 0.01f && f <= 0.99f);
bool is_melt_surface = (T > T_MELT && k >= nz - 10);

// For powder: Both conditions work
// - VOF interface captures particle surfaces
// - Melt surface captures active melt pool
```

### 5.4 Wetting and Spreading

**Contact Angle Effects**:

When liquid metal contacts:
- Solid particles: Wetting promotes coalescence
- Substrate: Determines melt pool shape

**Implementation via VOF**:

```cpp
// Already supported in vof_solver.cu
void applyBoundaryConditions(int boundary_type, float contact_angle = 90.0f);

// Typical values for Ti6Al4V:
// - On itself (liquid on solid): theta ~ 10-30 degrees (good wetting)
// - On oxide: theta ~ 90-120 degrees (poor wetting)
```

**Enhancement for Powder**:

```cpp
struct ContactAngleModel {
    float theta_metal_metal;      // Ti6Al4V liquid on solid Ti6Al4V
    float theta_metal_substrate;  // Ti6Al4V on build plate
    float theta_metal_oxide;      // If oxide film present

    float getContactAngle(CellType neighbor_type) const {
        switch (neighbor_type) {
            case SOLID_METAL: return theta_metal_metal;
            case SUBSTRATE: return theta_metal_substrate;
            case OXIDE: return theta_metal_oxide;
            default: return 90.0f;
        }
    }
};
```

---

## 6. Code Architecture

### 6.1 New Module: PowderBed

**File Structure**:

```
include/
  physics/
    powder_bed.h           // Main header
    powder_generator.h     // Particle generation algorithms
    powder_thermal.h       // Effective property models

src/
  physics/
    powder/
      powder_bed.cu        // PowderBed class implementation
      powder_generator.cu  // Generation CUDA kernels
      powder_thermal.cu    // Thermal property kernels
```

### 6.2 Class Design

```cpp
// include/physics/powder_bed.h

#pragma once

#include <cuda_runtime.h>
#include <vector>
#include "physics/vof_solver.h"

namespace lbm {
namespace physics {

/**
 * @brief Single powder particle representation
 */
struct Particle {
    float x, y, z;      // Center position [m]
    float radius;       // Radius [m]
    int id;             // Unique identifier
    bool is_melted;     // Melting state
};

/**
 * @brief Powder size distribution parameters
 */
struct PowderSizeDistribution {
    float D50 = 30.0e-6f;       // Median diameter [m]
    float sigma_g = 1.4f;       // Geometric std dev
    float D_min = 15.0e-6f;     // Minimum diameter [m]
    float D_max = 45.0e-6f;     // Maximum diameter [m]

    float sampleDiameter(curandState* rng) const;
};

/**
 * @brief Powder bed generation parameters
 */
struct PowderBedConfig {
    // Geometry
    float layer_thickness = 40.0e-6f;     // [m]
    float target_packing = 0.55f;         // [-]
    float substrate_height = 0.0f;        // [m] (z offset)

    // Size distribution
    PowderSizeDistribution size_dist;

    // Generation method
    enum class Method { RANDOM_SEQUENTIAL, RAIN, REGULAR_PERTURBED };
    Method generation_method = Method::RANDOM_SEQUENTIAL;

    // Random seed
    unsigned int seed = 42;

    // Thermal properties
    float k_gas = 0.018f;                 // Argon [W/(m*K)]
    float effective_absorption_depth;     // Computed from other params

    // Compute derived quantities
    void computeDerivedQuantities();
};

/**
 * @brief Powder bed generator and manager
 */
class PowderBed {
public:
    /**
     * @brief Constructor
     * @param config Powder bed configuration
     * @param vof Pointer to VOF solver (for fill_level initialization)
     */
    PowderBed(const PowderBedConfig& config, VOFSolver* vof);

    ~PowderBed();

    /**
     * @brief Generate powder bed and initialize VOF fill_level
     * @param domain_nx, domain_ny, domain_nz Grid dimensions
     * @param dx Grid spacing [m]
     */
    void generate(int domain_nx, int domain_ny, int domain_nz, float dx);

    /**
     * @brief Initialize thermal properties for powder region
     * @param d_thermal_conductivity Device array to modify
     */
    void initializeThermalProperties(float* d_thermal_conductivity) const;

    /**
     * @brief Get effective thermal conductivity for powder layer
     * @return k_eff [W/(m*K)]
     */
    float getEffectiveThermalConductivity() const;

    /**
     * @brief Get effective laser absorption depth for powder
     * @return d_eff [m]
     */
    float getEffectiveAbsorptionDepth() const;

    /**
     * @brief Get particle list
     */
    const std::vector<Particle>& getParticles() const { return particles_; }

    /**
     * @brief Get actual achieved packing density
     */
    float getActualPacking() const { return actual_packing_; }

    /**
     * @brief Get number of particles
     */
    int getNumParticles() const { return static_cast<int>(particles_.size()); }

    /**
     * @brief Diagnostic: Compute particle-level statistics
     */
    void computeStatistics() const;

private:
    PowderBedConfig config_;
    VOFSolver* vof_;            // Non-owning pointer

    std::vector<Particle> particles_;
    float actual_packing_;

    // Generation methods
    void generateRandomSequential(int nx, int ny, int nz, float dx);
    void generateRainDeposition(int nx, int ny, int nz, float dx);
    void generateRegularPerturbed(int nx, int ny, int nz, float dx);

    // Helper functions
    bool checkCollision(const Particle& p, float min_gap) const;
    void updateVOFFillLevel(int nx, int ny, int nz, float dx);
};

// CUDA Kernels
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

__global__ void computeEffectiveThermalConductivityKernel(
    float* k_effective,
    const float* fill_level,
    float k_metal,
    float k_gas,
    int num_cells);

} // namespace physics
} // namespace lbm
```

### 6.3 Integration with Existing Modules

**Modified Files**:

| File | Changes |
|------|---------|
| `multiphysics_solver.h` | Add PowderBed member, config flags |
| `multiphysics_solver.cu` | Initialize powder bed, use effective properties |
| `laser_source.h` | Add powder absorption mode |
| `thermal_lbm.cu` | Support spatially varying conductivity |
| `surface_initialization.h` | Extend POWDER_BED type |

**MultiphysicsConfig Extension**:

```cpp
struct MultiphysicsConfig {
    // ... existing parameters ...

    // Powder bed parameters (new)
    bool enable_powder_bed;
    PowderBedConfig powder_config;
};
```

**MultiphysicsSolver Integration**:

```cpp
class MultiphysicsSolver {
    // ... existing members ...

    std::unique_ptr<PowderBed> powder_bed_;

    void initialize(float initial_temperature, float interface_height) {
        // ... existing initialization ...

        if (config_.enable_powder_bed) {
            powder_bed_ = std::make_unique<PowderBed>(
                config_.powder_config, vof_.get());
            powder_bed_->generate(config_.nx, config_.ny, config_.nz, config_.dx);

            // Update laser absorption depth
            if (laser_) {
                laser_->setEffectivePenetrationDepth(
                    powder_bed_->getEffectiveAbsorptionDepth());
            }
        }
    }
};
```

### 6.4 Configuration File Support

**YAML Configuration Example**:

```yaml
powder_bed:
  enabled: true
  layer_thickness: 40.0e-6     # [m]
  target_packing: 0.55
  generation_method: "random_sequential"  # or "rain", "regular_perturbed"
  seed: 42

  size_distribution:
    D50: 30.0e-6               # [m]
    sigma_g: 1.4
    D_min: 15.0e-6             # [m]
    D_max: 45.0e-6             # [m]

  thermal:
    gas_conductivity: 0.018    # Argon [W/(m*K)]
```

---

## 7. Computational Cost Analysis

### 7.1 Resolution Requirements

**Baseline Flat Surface Simulation**:

| Parameter | Value |
|-----------|-------|
| Domain | 100 x 100 x 50 um |
| dx | 1 um |
| Cells | 500,000 |
| Time step | 1 ns |
| Simulation time | 100 us |
| Wall time | ~1 hour (RTX 3080) |

### 7.2 Powder Bed Resolution Impact

**Required Resolution for Particle-Resolved Simulation**:

| dx [um] | Cells/D30 | Total Cells | Memory [GB] | Relative Cost |
|---------|-----------|-------------|-------------|---------------|
| 2.0 | 15 | 3.1M | ~1.5 | 6x |
| 1.5 | 20 | 7.4M | ~3.5 | 15x |
| 1.0 | 30 | 25M | ~12 | 50x |

**Recommendation**: Start with dx = 2 um (15 cells per D30 particle)

### 7.3 Time Step Considerations

**CFL Constraints**:

| Physics | CFL Condition | Limiting Factor |
|---------|---------------|-----------------|
| Fluid LBM | dt < dx / (3 * c_s) | Sound speed |
| Thermal LBM | dt < dx^2 / (6 * alpha) | Diffusivity |
| VOF advection | dt < dx / u_max | Marangoni velocity |

**For dx = 2 um**:

```
Fluid: dt < 2e-6 / (3 * 300) ~ 2.2e-9 s
Thermal: dt < (2e-6)^2 / (6 * 5.8e-6) ~ 1.1e-10 s  // Most restrictive!
VOF: dt < 2e-6 / 1.0 ~ 2e-6 s  // If u_max ~ 1 m/s
```

**Challenge**: Thermal diffusion requires very small time steps at fine resolution.

**Mitigation Options**:

1. **Implicit thermal solver**: Remove stability restriction
2. **Operator splitting**: Larger thermal substeps
3. **Coarser thermal grid**: Multi-resolution approach

### 7.4 Memory Requirements

**Per-Cell Storage**:

| Field | Size (float) | Bytes |
|-------|--------------|-------|
| Temperature | 1 | 4 |
| Thermal LBM (D3Q7) | 7 | 28 |
| Velocity (3) | 3 | 12 |
| Fluid LBM (D3Q19) | 19 | 76 |
| Fill level | 1 | 4 |
| Interface normal | 3 | 12 |
| Curvature | 1 | 4 |
| Forces (3) | 3 | 12 |
| Liquid fraction | 1 | 4 |
| **Total** | | **156 bytes/cell** |

**For 3.1M cells**: ~484 MB
**For 7.4M cells**: ~1.15 GB
**For 25M cells**: ~3.9 GB

### 7.5 Simplification Strategies

**Strategy 1: Reduced Powder Layer**

Instead of full powder bed, use a thin layer (~3-5 particles deep):

```
Layer thickness: 30-50 um (1-2 particle diameters)
Benefit: Captures surface effects with minimal cells
```

**Strategy 2: Multi-Resolution Approach**

```
Fine grid (dx = 2um): Near laser (50x50x50 um cube)
Coarse grid (dx = 5um): Far field
```

**Strategy 3: Effective Medium Near Substrate**

```
Top layer: Particle-resolved (individual particles)
Bulk: Effective thermal conductivity (no particles)
```

**Strategy 4: 2D Cross-Section**

```
For initial development: 2D slice through powder bed
Significant speedup for algorithm testing
```

---

## 8. Implementation Roadmap

### Phase 1: Basic Powder Generation (Week 1)

**Deliverables**:

1. `PowderBed` class with random sequential generation
2. VOF fill_level initialization from particle list
3. Static visualization (no dynamics)
4. Unit tests for generation algorithms

**Validation**:
- Visual inspection of particle arrangement
- Packing density calculation
- Size distribution verification

### Phase 2: Thermal Property Integration (Week 2)

**Deliverables**:

1. Effective thermal conductivity model
2. Modified laser absorption for powder
3. Static heating test (no melting)

**Validation**:
- Compare temperature field to flat surface case
- Verify reduced heat penetration
- Energy conservation check

### Phase 3: Melting Dynamics (Week 3)

**Deliverables**:

1. Full coupling with thermal/phase-change solvers
2. Particle melting and coalescence
3. Marangoni flow on particle surfaces

**Validation**:
- Single particle melting test
- Two-particle coalescence test
- Melt pool formation comparison with literature

### Phase 4: Production Simulations (Week 4+)

**Deliverables**:

1. Full LPBF simulation with powder bed
2. Parameter study (power, speed, layer thickness)
3. Porosity prediction capability

**Validation**:
- Compare melt pool dimensions to experiments
- Track balling defects
- Validate against published results

---

## 9. Validation Strategy

### 9.1 Unit Tests

```cpp
// Test 1: Single particle curvature
TEST(PowderBed, SingleParticleCurvature) {
    // Create single particle, verify VOF curvature = 2/R
}

// Test 2: Packing density
TEST(PowderBed, PackingDensity) {
    // Generate powder bed, verify achieved packing within 5% of target
}

// Test 3: Size distribution
TEST(PowderBed, SizeDistribution) {
    // Generate many particles, verify D50 and sigma match input
}

// Test 4: No particle overlap
TEST(PowderBed, NoOverlap) {
    // Check all particle pairs for collision
}
```

### 9.2 Integration Tests

```cpp
// Test 1: Static heating
TEST(PowderBed, StaticHeating) {
    // Heat powder layer, verify reduced penetration vs flat surface
}

// Test 2: Single particle melting
TEST(PowderBed, SingleParticleMelting) {
    // Melt one particle, verify coalescence with substrate
}

// Test 3: Energy conservation
TEST(PowderBed, EnergyConservation) {
    // Track energy in/out, verify balance within 5%
}
```

### 9.3 Validation Against Literature

**Key References**:

1. **Khairallah et al. (2016)**: Mesoscale powder bed simulation
   - Compare melt pool shape, balling threshold

2. **Matthews et al. (2016)**: Denudation in LPBF
   - Compare particle ejection patterns

3. **Korner et al. (2011)**: LBM for selective laser sintering
   - Compare coalescence dynamics

### 9.4 Experimental Comparison

**Available Data** (from literature):

| Metric | Expected Range | Source |
|--------|---------------|--------|
| Melt pool width | 100-200 um | Panwisawas 2017 |
| Melt pool depth | 50-150 um | King 2014 |
| Balling threshold | P/v < 50 J/m | Gu 2013 |

---

## 10. References

### Core LBM/VOF References

1. Koerner, C., et al. (2005). "Lattice Boltzmann model for free surface flow for modeling foaming." J. Stat. Phys.

2. Korner, C., et al. (2011). "Mesoscopic simulation of selective beam melting processes." J. Mater. Process. Technol.

3. Thuerey, N. (2007). "A single-phase free-surface lattice Boltzmann method." Ph.D. thesis.

### LPBF Physics References

4. Khairallah, S.A., et al. (2016). "Laser powder-bed fusion additive manufacturing: Physics of complex melt flow and formation mechanisms of pores, spatter, and denudation zones." Acta Mater.

5. King, W.E., et al. (2014). "Observation of keyhole-mode laser melting in laser powder-bed fusion additive manufacturing." J. Mater. Process. Technol.

6. Panwisawas, C., et al. (2017). "Mesoscale modelling of selective laser melting: Thermal fluid dynamics and microstructural evolution." Comput. Mater. Sci.

### Powder Property References

7. Gusarov, A.V., Kruth, J.P. (2005). "Modelling of radiation transfer in metallic powders at laser treatment." Int. J. Heat Mass Transf.

8. Zehner, P., Schlunder, E.U. (1970). "Thermal conductivity of granular materials at moderate temperatures." Chem. Ing. Tech.

### Marangoni/Surface Tension References

9. Mills, K.C. (2002). "Recommended values of thermophysical properties for selected commercial alloys." Woodhead Publishing.

---

## Appendix A: Quick Start Example

```cpp
// Example: Initialize powder bed in existing simulation

#include "physics/powder_bed.h"
#include "physics/multiphysics_solver.h"

int main() {
    MultiphysicsConfig config;

    // Domain setup (100x100x60 um with powder layer)
    config.nx = 50;
    config.ny = 50;
    config.nz = 30;
    config.dx = 2.0e-6f;  // 2 um resolution

    // Enable powder bed
    config.enable_powder_bed = true;
    config.powder_config.layer_thickness = 40.0e-6f;
    config.powder_config.target_packing = 0.55f;
    config.powder_config.size_dist.D50 = 30.0e-6f;

    // Create solver
    MultiphysicsSolver solver(config);

    // Initialize (generates powder bed automatically)
    solver.initialize(300.0f, 0.8f);  // T=300K, interface at 80% height

    // Run simulation
    for (int step = 0; step < 100000; ++step) {
        solver.step();
    }

    return 0;
}
```

---

**Document Status**: Draft
**Next Review**: After Phase 1 implementation
**Maintainer**: LBM-CFD Architecture Team
