/**
 * @file test_steel_material.cpp
 * @brief Unit test for Steel material properties
 */

#include "physics/material_properties.h"
#include <iostream>
#include <iomanip>
#include <cmath>

using namespace lbm::physics;

bool testSteelProperties() {
    MaterialProperties steel = MaterialDatabase::getSteel();

    // Test validation case parameters
    const float expected_rho_solid = 7900.0f;
    const float expected_rho_liquid = 7433.0f;
    const float expected_T_m = 1723.0f;
    const float expected_T_v = 3090.0f;

    bool passed = true;

    // Check densities match validation case
    if (std::abs(steel.rho_solid - expected_rho_solid) > 1.0f) {
        std::cout << "FAIL: rho_solid = " << steel.rho_solid
                  << ", expected " << expected_rho_solid << "\n";
        passed = false;
    }

    if (std::abs(steel.rho_liquid - expected_rho_liquid) > 1.0f) {
        std::cout << "FAIL: rho_liquid = " << steel.rho_liquid
                  << ", expected " << expected_rho_liquid << "\n";
        passed = false;
    }

    if (std::abs(steel.T_liquidus - expected_T_m) > 1.0f) {
        std::cout << "FAIL: T_liquidus = " << steel.T_liquidus
                  << ", expected " << expected_T_m << "\n";
        passed = false;
    }

    if (std::abs(steel.T_vaporization - expected_T_v) > 1.0f) {
        std::cout << "FAIL: T_vaporization = " << steel.T_vaporization
                  << ", expected " << expected_T_v << "\n";
        passed = false;
    }

    // Validate all properties are physically reasonable
    if (!steel.validate()) {
        std::cout << "FAIL: Material validation failed\n";
        passed = false;
    }

    return passed;
}

bool testTemperatureDependence() {
    MaterialProperties steel = MaterialDatabase::getSteel();

    // Test at solid temperature (300K)
    float T_solid = 300.0f;
    if (steel.getDensity(T_solid) != steel.rho_solid) {
        std::cout << "FAIL: Density at 300K should be rho_solid\n";
        return false;
    }

    // Test at liquid temperature (2000K)
    float T_liquid = 2000.0f;
    if (steel.getDensity(T_liquid) != steel.rho_liquid) {
        std::cout << "FAIL: Density at 2000K should be rho_liquid\n";
        return false;
    }

    // Test viscosity behavior
    if (steel.getDynamicViscosity(T_solid) < 1e9f) {
        std::cout << "FAIL: Solid viscosity should be very high\n";
        return false;
    }

    if (steel.getDynamicViscosity(T_liquid) != steel.mu_liquid) {
        std::cout << "FAIL: Liquid viscosity incorrect\n";
        return false;
    }

    return true;
}

bool testMaterialByName() {
    std::string test_names[] = {"Steel", "steel", "Fe", "fe", "Iron", "iron"};

    for (const auto& name : test_names) {
        try {
            MaterialProperties mat = MaterialDatabase::getMaterialByName(name);
            if (std::string(mat.name) != "Steel") {
                std::cout << "FAIL: '" << name << "' returned wrong material: "
                          << mat.name << "\n";
                return false;
            }
        } catch (const std::exception& e) {
            std::cout << "FAIL: '" << name << "' threw exception: "
                      << e.what() << "\n";
            return false;
        }
    }

    return true;
}

int main() {
    std::cout << "\n=== Steel Material Properties Unit Test ===\n\n";

    bool all_passed = true;

    // Test 1: Basic properties
    std::cout << "Test 1: Steel properties validation... ";
    if (testSteelProperties()) {
        std::cout << "PASSED\n";
    } else {
        std::cout << "FAILED\n";
        all_passed = false;
    }

    // Test 2: Temperature dependence
    std::cout << "Test 2: Temperature-dependent functions... ";
    if (testTemperatureDependence()) {
        std::cout << "PASSED\n";
    } else {
        std::cout << "FAILED\n";
        all_passed = false;
    }

    // Test 3: Name lookup
    std::cout << "Test 3: Material name lookup... ";
    if (testMaterialByName()) {
        std::cout << "PASSED\n";
    } else {
        std::cout << "FAILED\n";
        all_passed = false;
    }

    // Print summary
    std::cout << "\n";
    if (all_passed) {
        std::cout << "=== ALL TESTS PASSED ===\n";

        // Print material summary
        MaterialProperties steel = MaterialDatabase::getSteel();
        std::cout << "\nSteel Material Summary:\n";
        std::cout << "  Solid density:  " << steel.rho_solid << " kg/m³\n";
        std::cout << "  Liquid density: " << steel.rho_liquid << " kg/m³\n";
        std::cout << "  Melting point:  " << steel.T_liquidus << " K\n";
        std::cout << "  Boiling point:  " << steel.T_vaporization << " K\n";
        std::cout << "  Surface tension: " << steel.surface_tension << " N/m\n";
        std::cout << "  dσ/dT:          " << steel.dsigma_dT << " N/(m·K)\n";

        return 0;
    } else {
        std::cout << "=== SOME TESTS FAILED ===\n";
        return 1;
    }
}
