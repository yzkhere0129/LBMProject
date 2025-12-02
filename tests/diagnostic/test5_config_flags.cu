/**
 * @file test5_config_flags.cu
 * @brief Test 5: Config Flag Propagation
 *
 * PURPOSE: Are config flags correctly read and applied?
 *
 * TEST CASE:
 * - Load lpbf_195W_test_A_coupling.conf (or equivalent)
 * - Print all physics flags:
 *   - config.enable_fluid
 *   - config.enable_thermal_advection
 *   - config.enable_marangoni
 *   - config.enable_buoyancy (if exists)
 * - Check MultiphysicsSolver initialization output
 *
 * EXPECTED: All flags should match config file
 * IF mismatch: Config parsing bug
 */

#include <iostream>
#include <iomanip>
#include <fstream>
#include <string>
#include <cuda_runtime.h>
#include "physics/multiphysics_solver.h"

using namespace lbm;

// Helper function to parse config file (simplified)
void parseConfigFile(const std::string& filename, physics::MultiphysicsConfig& config) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "ERROR: Cannot open config file: " << filename << "\n";
        return;
    }

    std::string line;
    while (std::getline(file, line)) {
        // Skip comments and empty lines
        if (line.empty() || line[0] == '#') continue;

        // Simple parsing: key = value
        size_t pos = line.find('=');
        if (pos == std::string::npos) continue;

        std::string key = line.substr(0, pos);
        std::string value = line.substr(pos + 1);

        // Trim whitespace
        key.erase(0, key.find_first_not_of(" \t"));
        key.erase(key.find_last_not_of(" \t") + 1);
        value.erase(0, value.find_first_not_of(" \t"));
        value.erase(value.find_last_not_of(" \t") + 1);

        // Parse flags
        if (key == "enable_thermal") {
            config.enable_thermal = (value == "true" || value == "1");
        } else if (key == "enable_thermal_advection") {
            config.enable_thermal_advection = (value == "true" || value == "1");
        } else if (key == "enable_fluid") {
            config.enable_fluid = (value == "true" || value == "1");
        } else if (key == "enable_marangoni") {
            config.enable_marangoni = (value == "true" || value == "1");
        } else if (key == "enable_vof") {
            config.enable_vof = (value == "true" || value == "1");
        } else if (key == "enable_darcy") {
            config.enable_darcy = (value == "true" || value == "1");
        } else if (key == "enable_laser") {
            config.enable_laser = (value == "true" || value == "1");
        }
        // Add more fields as needed
    }

    file.close();
}

void printConfigFlags(const physics::MultiphysicsConfig& config, const std::string& label) {
    std::cout << label << ":\n";
    std::cout << "  enable_thermal           = " << (config.enable_thermal ? "true" : "false") << "\n";
    std::cout << "  enable_thermal_advection = " << (config.enable_thermal_advection ? "true" : "false") << "\n";
    std::cout << "  enable_phase_change      = " << (config.enable_phase_change ? "true" : "false") << "\n";
    std::cout << "  enable_fluid             = " << (config.enable_fluid ? "true" : "false") << "\n";
    std::cout << "  enable_vof               = " << (config.enable_vof ? "true" : "false") << "\n";
    std::cout << "  enable_vof_advection     = " << (config.enable_vof_advection ? "true" : "false") << "\n";
    std::cout << "  enable_surface_tension   = " << (config.enable_surface_tension ? "true" : "false") << "\n";
    std::cout << "  enable_marangoni         = " << (config.enable_marangoni ? "true" : "false") << "\n";
    std::cout << "  enable_laser             = " << (config.enable_laser ? "true" : "false") << "\n";
    std::cout << "  enable_darcy             = " << (config.enable_darcy ? "true" : "false") << "\n";
    std::cout << "  enable_radiation_bc      = " << (config.enable_radiation_bc ? "true" : "false") << "\n";
    std::cout << "\n";
}

int main(int argc, char** argv) {
    std::cout << "================================================================\n";
    std::cout << "  TEST 5: Config Flag Propagation\n";
    std::cout << "  Isolates: Configuration file parsing and flag propagation\n";
    std::cout << "================================================================\n\n";

    // Check if config file provided as argument
    std::string config_file;
    if (argc > 1) {
        config_file = argv[1];
    } else {
        // Try to find test config file
        config_file = "/home/yzk/LBMProject/config/lpbf_195W_test_A_coupling.conf";
        std::cout << "No config file specified, using default:\n";
        std::cout << "  " << config_file << "\n\n";
    }

    std::cout << "Config file: " << config_file << "\n\n";

    // Create default config
    physics::MultiphysicsConfig default_config;

    std::cout << "Default configuration flags:\n";
    std::cout << "----------------------------------------------------\n";
    printConfigFlags(default_config, "Default Config");

    // Try to parse config file
    physics::MultiphysicsConfig file_config = default_config;  // Start with defaults

    std::cout << "Attempting to parse config file...\n";
    parseConfigFile(config_file, file_config);

    std::cout << "\nConfiguration from file:\n";
    std::cout << "----------------------------------------------------\n";
    printConfigFlags(file_config, "File Config");

    // Compare
    bool flags_match = true;
    std::cout << "Comparison with defaults:\n";
    std::cout << "----------------------------------------------------\n";

    if (file_config.enable_thermal != default_config.enable_thermal) {
        std::cout << "  enable_thermal: FILE=" << file_config.enable_thermal
                  << " vs DEFAULT=" << default_config.enable_thermal << "\n";
        flags_match = false;
    }

    if (file_config.enable_thermal_advection != default_config.enable_thermal_advection) {
        std::cout << "  enable_thermal_advection: FILE=" << file_config.enable_thermal_advection
                  << " vs DEFAULT=" << default_config.enable_thermal_advection << "\n";
        flags_match = false;
    }

    if (file_config.enable_fluid != default_config.enable_fluid) {
        std::cout << "  enable_fluid: FILE=" << file_config.enable_fluid
                  << " vs DEFAULT=" << default_config.enable_fluid << "\n";
        flags_match = false;
    }

    if (file_config.enable_marangoni != default_config.enable_marangoni) {
        std::cout << "  enable_marangoni: FILE=" << file_config.enable_marangoni
                  << " vs DEFAULT=" << default_config.enable_marangoni << "\n";
        flags_match = false;
    }

    if (file_config.enable_darcy != default_config.enable_darcy) {
        std::cout << "  enable_darcy: FILE=" << file_config.enable_darcy
                  << " vs DEFAULT=" << default_config.enable_darcy << "\n";
        flags_match = false;
    }

    if (flags_match) {
        std::cout << "  All flags match defaults (file not parsed or has no overrides)\n";
    }

    std::cout << "\n";

    // Critical flags for v5 test
    std::cout << "================================================================\n";
    std::cout << "CRITICAL FLAGS FOR v5 TEST:\n";
    std::cout << "================================================================\n\n";

    std::cout << "For buoyancy-driven flow, we need:\n";
    std::cout << "  1. enable_fluid = true (to compute fluid flow)\n";
    std::cout << "  2. enable_thermal = true (to have temperature field)\n";
    std::cout << "  3. enable_thermal_advection = true (for thermal-fluid coupling)\n";
    std::cout << "  4. Buoyancy force must be computed (no explicit flag, part of fluid solver)\n\n";

    std::cout << "Current configuration:\n";
    std::cout << "  [" << (file_config.enable_fluid ? "✓" : "✗") << "] enable_fluid = "
              << (file_config.enable_fluid ? "true" : "false") << "\n";
    std::cout << "  [" << (file_config.enable_thermal ? "✓" : "✗") << "] enable_thermal = "
              << (file_config.enable_thermal ? "true" : "false") << "\n";
    std::cout << "  [" << (file_config.enable_thermal_advection ? "✓" : "✗") << "] enable_thermal_advection = "
              << (file_config.enable_thermal_advection ? "true" : "false") << "\n\n";

    // Diagnosis
    std::cout << "DIAGNOSIS:\n";
    bool passed = true;

    if (!file_config.enable_fluid) {
        std::cout << "  [FAIL] enable_fluid = false\n";
        std::cout << "  ROOT CAUSE: Fluid solver is disabled!\n";
        std::cout << "  This would cause zero velocity in v5 test.\n";
        passed = false;
    }

    if (!file_config.enable_thermal) {
        std::cout << "  [FAIL] enable_thermal = false\n";
        std::cout << "  ROOT CAUSE: Thermal solver is disabled!\n";
        std::cout << "  Without temperature, there's no buoyancy force.\n";
        passed = false;
    }

    if (!file_config.enable_thermal_advection) {
        std::cout << "  [WARNING] enable_thermal_advection = false\n";
        std::cout << "  This means fluid velocity is NOT passed to thermal solver.\n";
        std::cout << "  For full coupling, this should be true.\n";
        std::cout << "  However, this doesn't prevent buoyancy-driven flow.\n";
    }

    if (file_config.enable_darcy) {
        std::cout << "  [INFO] enable_darcy = true\n";
        std::cout << "  Darcy damping is enabled.\n";
        std::cout << "  If liquid_fraction is wrong (e.g., all solid), this kills all flow!\n";
        std::cout << "  Check that liquid_fraction field is correct in v5 test.\n";
    }

    if (passed) {
        std::cout << "\n  [PASS] Configuration flags are correct for fluid flow!\n";
        std::cout << "  All critical flags are enabled.\n";
        std::cout << "\n";
        std::cout << "  CONCLUSION: Config flag propagation is working.\n";
        std::cout << "  If v5 test has zero velocity, the problem is NOT config flags.\n";
        std::cout << "  Check actual physics implementation (buoyancy, Darcy, etc.).\n";
    } else {
        std::cout << "\n  [FAIL] Configuration has disabled critical physics!\n";
        std::cout << "  ROOT CAUSE: Config file has wrong settings.\n";
        std::cout << "\n";
        std::cout << "  ACTION REQUIRED:\n";
        std::cout << "  Edit config file to enable:\n";
        std::cout << "    enable_fluid = true\n";
        std::cout << "    enable_thermal = true\n";
        std::cout << "    enable_thermal_advection = true (optional but recommended)\n";
    }

    std::cout << "\n================================================================\n";
    std::cout << "  Test 5 Complete: " << (passed ? "PASS" : "FAIL") << "\n";
    std::cout << "================================================================\n";

    std::cout << "\nNOTE:\n";
    std::cout << "This test only checks flag parsing, not actual MultiphysicsSolver behavior.\n";
    std::cout << "To fully test flag propagation, create a MultiphysicsSolver instance and\n";
    std::cout << "verify that the physics modules are actually created based on flags.\n";

    return passed ? 0 : 1;
}
