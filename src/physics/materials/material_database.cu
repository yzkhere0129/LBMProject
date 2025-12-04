/**
 * @file material_database.cu
 * @brief Implementation of material properties database
 *
 * Material data sourced from MATERIAL_DATABASE.yaml
 */

#include "physics/material_properties.h"
#include <cstring>

// Define device constant memory for material properties globally
__constant__ lbm::physics::MaterialProperties d_material;

namespace lbm {
namespace physics {

MaterialProperties MaterialDatabase::getTi6Al4V() {
    MaterialProperties mat;

    // Material name
    strcpy(mat.name, "Ti6Al4V");

    // Solid state properties (from MATERIAL_DATABASE.yaml)
    mat.rho_solid = 4420.0f;           // kg/m³
    mat.cp_solid = 610.0f;             // J/(kg·K)
    mat.k_solid = 7.0f;                // W/(m·K)

    // Liquid state properties
    mat.rho_liquid = 4110.0f;          // kg/m³ (Mills 2002)
    mat.cp_liquid = 831.0f;            // J/(kg·K)
    mat.k_liquid = 33.0f;              // W/(m·K)
    mat.mu_liquid = 5.0e-3f;           // Pa·s (Valencia & Quested 2008)

    // Phase change parameters
    mat.T_solidus = 1878.0f;           // K
    mat.T_liquidus = 1923.0f;          // K (melting point)
    mat.T_vaporization = 3560.0f;      // K (boiling point, ASM Handbook)
    mat.L_fusion = 286000.0f;          // J/kg
    mat.L_vaporization = 9830000.0f;   // J/kg

    // Surface properties
    mat.surface_tension = 1.65f;       // N/m at melting point
    mat.dsigma_dT = -2.6e-4f;          // N/(m·K) (Mills 2002)

    // Optical properties (at 1064nm laser wavelength)
    mat.absorptivity_solid = 0.35f;
    mat.absorptivity_liquid = 0.40f;
    mat.emissivity = 0.25f;

    return mat;
}

MaterialProperties MaterialDatabase::get316L() {
    MaterialProperties mat;

    // Material name
    strcpy(mat.name, "SS316L");

    // Solid state properties (from MATERIAL_DATABASE.yaml)
    mat.rho_solid = 7990.0f;           // kg/m³
    mat.cp_solid = 500.0f;             // J/(kg·K)
    mat.k_solid = 16.2f;               // W/(m·K)

    // Liquid state properties
    mat.rho_liquid = 6900.0f;          // kg/m³
    mat.cp_liquid = 775.0f;            // J/(kg·K)
    mat.k_liquid = 30.0f;              // W/(m·K)
    mat.mu_liquid = 6.0e-3f;           // Pa·s

    // Phase change parameters
    mat.T_solidus = 1658.0f;           // K
    mat.T_liquidus = 1700.0f;          // K (melting point)
    mat.T_vaporization = 3090.0f;      // K (boiling point)
    mat.L_fusion = 260000.0f;          // J/kg
    mat.L_vaporization = 7450000.0f;   // J/kg

    // Surface properties
    mat.surface_tension = 1.75f;       // N/m at melting point
    mat.dsigma_dT = -4.3e-4f;          // N/(m·K)

    // Optical properties (at 1064nm laser wavelength)
    mat.absorptivity_solid = 0.38f;
    mat.absorptivity_liquid = 0.42f;
    mat.emissivity = 0.28f;

    return mat;
}

MaterialProperties MaterialDatabase::getInconel718() {
    MaterialProperties mat;

    // Material name
    strcpy(mat.name, "IN718");

    // Solid state properties (from MATERIAL_DATABASE.yaml)
    mat.rho_solid = 8190.0f;           // kg/m³
    mat.cp_solid = 435.0f;             // J/(kg·K)
    mat.k_solid = 11.4f;               // W/(m·K)

    // Liquid state properties
    mat.rho_liquid = 7450.0f;          // kg/m³
    mat.cp_liquid = 620.0f;            // J/(kg·K)
    mat.k_liquid = 29.0f;              // W/(m·K)
    mat.mu_liquid = 5.0e-3f;           // Pa·s

    // Phase change parameters
    mat.T_solidus = 1533.0f;           // K
    mat.T_liquidus = 1609.0f;          // K (melting point)
    mat.T_vaporization = 3100.0f;      // K (boiling point)
    mat.L_fusion = 210000.0f;          // J/kg
    mat.L_vaporization = 6430000.0f;   // J/kg

    // Surface properties
    mat.surface_tension = 1.89f;       // N/m at melting point
    mat.dsigma_dT = -3.8e-4f;          // N/(m·K)

    // Optical properties (at 1064nm laser wavelength)
    mat.absorptivity_solid = 0.37f;
    mat.absorptivity_liquid = 0.41f;
    mat.emissivity = 0.27f;

    return mat;
}

MaterialProperties MaterialDatabase::getAlSi10Mg() {
    MaterialProperties mat;

    // Material name
    strcpy(mat.name, "AlSi10Mg");

    // Solid state properties (from MATERIAL_DATABASE.yaml)
    mat.rho_solid = 2680.0f;           // kg/m³
    mat.cp_solid = 963.0f;             // J/(kg·K)
    mat.k_solid = 157.0f;              // W/(m·K)

    // Liquid state properties
    mat.rho_liquid = 2375.0f;          // kg/m³
    mat.cp_liquid = 1180.0f;           // J/(kg·K)
    mat.k_liquid = 90.0f;              // W/(m·K)
    mat.mu_liquid = 1.3e-3f;           // Pa·s

    // Phase change parameters
    mat.T_solidus = 833.0f;            // K
    mat.T_liquidus = 873.0f;           // K (melting point)
    mat.T_vaporization = 2743.0f;      // K (boiling point)
    mat.L_fusion = 395000.0f;          // J/kg
    mat.L_vaporization = 10900000.0f;  // J/kg

    // Surface properties
    mat.surface_tension = 0.914f;      // N/m at melting point
    mat.dsigma_dT = -1.5e-4f;          // N/(m·K)

    // Optical properties (at 1064nm laser wavelength)
    // Note: Aluminum is highly reflective!
    mat.absorptivity_solid = 0.09f;
    mat.absorptivity_liquid = 0.13f;
    mat.emissivity = 0.09f;

    return mat;
}

void MaterialDatabase::copyToDevice(const MaterialProperties& mat) {
    cudaError_t error = cudaMemcpyToSymbol(d_material, &mat, sizeof(MaterialProperties));
    if (error != cudaSuccess) {
        throw std::runtime_error("Failed to copy material properties to device: " +
                                 std::string(cudaGetErrorString(error)));
    }
}

MaterialProperties MaterialDatabase::getMaterialByName(const std::string& name) {
    if (name == "Ti6Al4V" || name == "ti6al4v" || name == "TI6AL4V") {
        return getTi6Al4V();
    } else if (name == "316L" || name == "SS316L" || name == "ss316l") {
        return get316L();
    } else if (name == "IN718" || name == "Inconel718" || name == "inconel718") {
        return getInconel718();
    } else if (name == "AlSi10Mg" || name == "alsi10mg" || name == "ALSI10MG") {
        return getAlSi10Mg();
    } else {
        throw std::runtime_error("Unknown material: " + name +
                                 ". Available materials: Ti6Al4V, 316L, IN718, AlSi10Mg");
    }
}

} // namespace physics
} // namespace lbm