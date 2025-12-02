/**
 * @file material_properties.h
 * @brief Material properties database for metal additive manufacturing simulations
 *
 * This module provides temperature-dependent material properties for various metals
 * used in additive manufacturing processes. Properties include thermal, mechanical,
 * optical, and surface characteristics.
 */

#ifndef MATERIAL_PROPERTIES_H
#define MATERIAL_PROPERTIES_H

#include <cuda_runtime.h>
#include <cmath>
#include <string>
#include <stdexcept>

namespace lbm {
namespace physics {

/**
 * @brief Temperature ranges for validation
 */
constexpr float T_MIN = 273.0f;     // 0°C - Minimum reasonable temperature
constexpr float T_MAX = 10000.0f;   // Maximum temperature for simulations (raised for high-power LPBF)

/**
 * @brief Material properties structure with temperature-dependent functions
 *
 * This structure holds all necessary material properties for simulating
 * metal additive manufacturing processes, including phase transitions,
 * surface effects, and optical properties.
 */
struct MaterialProperties {
    // Basic thermophysical properties - Solid state (at room temperature)
    float rho_solid;          ///< Solid density [kg/m³]
    float cp_solid;           ///< Solid specific heat capacity [J/(kg·K)]
    float k_solid;            ///< Solid thermal conductivity [W/(m·K)]

    // Liquid state properties
    float rho_liquid;         ///< Liquid density [kg/m³]
    float cp_liquid;          ///< Liquid specific heat capacity [J/(kg·K)]
    float k_liquid;           ///< Liquid thermal conductivity [W/(m·K)]
    float mu_liquid;          ///< Dynamic viscosity of liquid [Pa·s]

    // Phase change parameters
    float T_solidus;          ///< Solidus temperature [K]
    float T_liquidus;         ///< Liquidus temperature (melting point) [K]
    float T_vaporization;     ///< Vaporization/boiling temperature [K]
    float L_fusion;           ///< Latent heat of fusion [J/kg]
    float L_vaporization;     ///< Latent heat of vaporization [J/kg]

    // Surface properties
    float surface_tension;    ///< Surface tension at melting point [N/m]
    float dsigma_dT;          ///< Temperature coefficient of surface tension [N/(m·K)]

    // Optical properties
    float absorptivity_solid; ///< Laser absorptivity in solid state (at 1064nm)
    float absorptivity_liquid;///< Laser absorptivity in liquid state (at 1064nm)
    float emissivity;         ///< Thermal emissivity

    // Material name for identification
    char name[64];            ///< Material name/identifier

    /**
     * @brief Get density at given temperature
     * @param T Temperature [K]
     * @return Density [kg/m³]
     */
    __host__ __device__ float getDensity(float T) const {
        if (T < T_solidus) {
            return rho_solid;
        } else if (T > T_liquidus) {
            return rho_liquid;
        } else {
            // Linear interpolation in mushy zone
            float fl = liquidFraction(T);
            return rho_solid * (1.0f - fl) + rho_liquid * fl;
        }
    }

    /**
     * @brief Get specific heat capacity at given temperature
     * @param T Temperature [K]
     * @return Specific heat capacity [J/(kg·K)]
     */
    __host__ __device__ float getSpecificHeat(float T) const {
        if (T < T_solidus) {
            return cp_solid;
        } else if (T > T_liquidus) {
            return cp_liquid;
        } else {
            // Linear interpolation in mushy zone
            float fl = liquidFraction(T);
            return cp_solid * (1.0f - fl) + cp_liquid * fl;
        }
    }

    /**
     * @brief Get thermal conductivity at given temperature
     * @param T Temperature [K]
     * @return Thermal conductivity [W/(m·K)]
     */
    __host__ __device__ float getThermalConductivity(float T) const {
        if (T < T_solidus) {
            return k_solid;
        } else if (T > T_liquidus) {
            return k_liquid;
        } else {
            // Linear interpolation in mushy zone
            float fl = liquidFraction(T);
            return k_solid * (1.0f - fl) + k_liquid * fl;
        }
    }

    /**
     * @brief Calculate thermal diffusivity at given temperature
     * @param T Temperature [K]
     * @return Thermal diffusivity [m²/s]
     */
    __host__ __device__ float getThermalDiffusivity(float T) const {
        float k = getThermalConductivity(T);
        float rho = getDensity(T);
        float cp = getSpecificHeat(T);
        return k / (rho * cp);
    }

    /**
     * @brief Get dynamic viscosity at given temperature
     * @param T Temperature [K]
     * @return Dynamic viscosity [Pa·s] (returns very high value for solid)
     */
    __host__ __device__ float getDynamicViscosity(float T) const {
        if (T < T_solidus) {
            return 1e10f;  // Very high viscosity for solid (effectively rigid)
        } else if (T > T_liquidus) {
            return mu_liquid;
        } else {
            // Exponential increase in mushy zone (more realistic than linear)
            float fl = liquidFraction(T);
            return mu_liquid * expf(5.0f * (1.0f - fl));
        }
    }

    /**
     * @brief Get surface tension at given temperature
     * @param T Temperature [K]
     * @return Surface tension [N/m]
     */
    __host__ __device__ float getSurfaceTension(float T) const {
        // Linear temperature dependence: σ(T) = σ₀ + (dσ/dT) * (T - T_melt)
        return surface_tension + dsigma_dT * (T - T_liquidus);
    }

    /**
     * @brief Get laser absorptivity at given temperature
     * @param T Temperature [K]
     * @return Absorptivity (dimensionless, 0-1)
     */
    __host__ __device__ float getAbsorptivity(float T) const {
        if (T < T_solidus) {
            return absorptivity_solid;
        } else if (T > T_liquidus) {
            return absorptivity_liquid;
        } else {
            // Linear interpolation in mushy zone
            float fl = liquidFraction(T);
            return absorptivity_solid * (1.0f - fl) + absorptivity_liquid * fl;
        }
    }

    /**
     * @brief Check if material is solid at given temperature
     * @param T Temperature [K]
     * @return True if completely solid
     */
    __host__ __device__ bool isSolid(float T) const {
        return T < T_solidus;
    }

    /**
     * @brief Check if material is liquid at given temperature
     * @param T Temperature [K]
     * @return True if completely liquid
     */
    __host__ __device__ bool isLiquid(float T) const {
        return T > T_liquidus;
    }

    /**
     * @brief Check if material is in mushy zone at given temperature
     * @param T Temperature [K]
     * @return True if in mushy zone (partially melted)
     */
    __host__ __device__ bool isMushy(float T) const {
        return T >= T_solidus && T <= T_liquidus;
    }

    /**
     * @brief Calculate liquid fraction at given temperature
     * @param T Temperature [K]
     * @return Liquid fraction (0 = fully solid, 1 = fully liquid)
     */
    __host__ __device__ float liquidFraction(float T) const {
        if (T < T_solidus) {
            return 0.0f;
        } else if (T > T_liquidus) {
            return 1.0f;
        } else {
            // Linear variation in mushy zone
            return (T - T_solidus) / (T_liquidus - T_solidus);
        }
    }

    /**
     * @brief Calculate solidification shrinkage factor
     * @return Shrinkage factor beta = 1 - rho_liquid/rho_solid (dimensionless)
     *
     * For Ti6Al4V: beta ~ 0.07 (7% volume shrinkage during solidification)
     */
    __host__ __device__ float getShrinkageFactor() const {
        return 1.0f - rho_liquid / rho_solid;
    }

    /**
     * @brief Get effective heat capacity including latent heat effects
     * @param T Temperature [K]
     * @param dT Temperature increment for enthalpy method [K]
     * @return Effective heat capacity [J/(kg·K)]
     */
    __host__ __device__ float getEffectiveHeatCapacity(float T, float dT = 1.0f) const {
        float cp = getSpecificHeat(T);

        // Add latent heat contribution if in phase change region
        if (isMushy(T) && dT > 0.0f) {
            // Approximate latent heat contribution
            float dfl_dT = 1.0f / (T_liquidus - T_solidus);
            cp += L_fusion * dfl_dT;
        }

        return cp;
    }

    /**
     * @brief Validate material properties for physical consistency
     * @return True if all properties are physically reasonable
     */
    __host__ bool validate() const {
        // Check that all properties are positive
        if (rho_solid <= 0.0f || rho_liquid <= 0.0f) return false;
        if (cp_solid <= 0.0f || cp_liquid <= 0.0f) return false;
        if (k_solid <= 0.0f || k_liquid <= 0.0f) return false;
        if (mu_liquid <= 0.0f) return false;

        // Check temperature ordering
        if (T_solidus <= 0.0f || T_liquidus <= T_solidus) return false;
        if (T_vaporization <= T_liquidus) return false;

        // Check latent heats
        if (L_fusion <= 0.0f || L_vaporization <= 0.0f) return false;

        // Check optical properties
        if (absorptivity_solid < 0.0f || absorptivity_solid > 1.0f) return false;
        if (absorptivity_liquid < 0.0f || absorptivity_liquid > 1.0f) return false;
        if (emissivity < 0.0f || emissivity > 1.0f) return false;

        // Check surface tension
        if (surface_tension <= 0.0f) return false;

        return true;
    }
};

/**
 * @brief Material property cache for performance optimization
 *
 * Caches frequently accessed properties at a specific temperature
 * to avoid redundant calculations
 */
struct MaterialCache {
    float T;              ///< Cached temperature [K]
    float k;              ///< Cached thermal conductivity [W/(m·K)]
    float cp;             ///< Cached specific heat [J/(kg·K)]
    float rho;            ///< Cached density [kg/m³]
    float alpha;          ///< Cached thermal diffusivity [m²/s]
    float liquid_fraction;///< Cached liquid fraction
    bool valid;           ///< Cache validity flag

    __host__ __device__ MaterialCache() : valid(false) {}

    /**
     * @brief Update cache for new temperature
     * @param mat Material properties
     * @param temperature New temperature [K]
     */
    __host__ __device__ void update(const MaterialProperties& mat, float temperature) {
        if (fabsf(temperature - T) > 0.1f || !valid) {
            T = temperature;
            k = mat.getThermalConductivity(T);
            cp = mat.getSpecificHeat(T);
            rho = mat.getDensity(T);
            alpha = mat.getThermalDiffusivity(T);
            liquid_fraction = mat.liquidFraction(T);
            valid = true;
        }
    }
};

/**
 * @brief Material database class providing predefined materials
 */
class MaterialDatabase {
public:
    /**
     * @brief Get Ti6Al4V (Titanium alloy) properties
     * @return MaterialProperties structure for Ti6Al4V
     */
    static MaterialProperties getTi6Al4V();

    /**
     * @brief Get 316L stainless steel properties
     * @return MaterialProperties structure for SS316L
     */
    static MaterialProperties get316L();

    /**
     * @brief Get Inconel 718 properties
     * @return MaterialProperties structure for IN718
     */
    static MaterialProperties getInconel718();

    /**
     * @brief Get AlSi10Mg aluminum alloy properties
     * @return MaterialProperties structure for AlSi10Mg
     */
    static MaterialProperties getAlSi10Mg();

    /**
     * @brief Copy material properties to device constant memory
     * @param mat Material properties to copy
     */
    static void copyToDevice(const MaterialProperties& mat);

    /**
     * @brief Get material by name
     * @param name Material name (Ti6Al4V, 316L, IN718, AlSi10Mg)
     * @return MaterialProperties structure
     * @throws std::runtime_error if material not found
     */
    static MaterialProperties getMaterialByName(const std::string& name);
};

/**
 * @brief Unit conversion utilities for material properties
 */
namespace MaterialUnits {
    /**
     * @brief Convert lattice temperature to physical temperature
     * @param T_lattice Lattice temperature (dimensionless)
     * @param T_ref Reference temperature [K]
     * @param deltaT Temperature scale [K]
     * @return Physical temperature [K]
     */
    inline float latticeToPhysicalTemperature(float T_lattice, float T_ref, float deltaT) {
        return T_ref + T_lattice * deltaT;
    }

    /**
     * @brief Convert physical temperature to lattice temperature
     * @param T_phys Physical temperature [K]
     * @param T_ref Reference temperature [K]
     * @param deltaT Temperature scale [K]
     * @return Lattice temperature (dimensionless)
     */
    inline float physicalToLatticeTemperature(float T_phys, float T_ref, float deltaT) {
        return (T_phys - T_ref) / deltaT;
    }

    /**
     * @brief Convert physical thermal diffusivity to lattice units
     * @param alpha_phys Physical thermal diffusivity [m²/s]
     * @param dx Lattice spacing [m]
     * @param dt Time step [s]
     * @return Lattice thermal diffusivity (dimensionless)
     */
    inline float physicalToLatticeDiffusivity(float alpha_phys, float dx, float dt) {
        return alpha_phys * dt / (dx * dx);
    }

    /**
     * @brief Convert lattice thermal diffusivity to physical units
     * @param alpha_lattice Lattice thermal diffusivity (dimensionless)
     * @param dx Lattice spacing [m]
     * @param dt Time step [s]
     * @return Physical thermal diffusivity [m²/s]
     */
    inline float latticeToPhysicalDiffusivity(float alpha_lattice, float dx, float dt) {
        return alpha_lattice * dx * dx / dt;
    }
}

} // namespace physics
} // namespace lbm

// Declare device constant memory for material properties
// This will be defined in the .cu file
#ifdef __CUDACC__
extern __constant__ lbm::physics::MaterialProperties d_material;
#endif

#endif // MATERIAL_PROPERTIES_H