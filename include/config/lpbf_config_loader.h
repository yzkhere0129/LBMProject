#pragma once
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <map>
#include <set>

namespace lbm {
namespace config {

class ConfigLoader {
public:
    std::map<std::string, std::string> params;

    bool load(const std::string& filename) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Warning: Could not open config file: " << filename << "\n";
            std::cerr << "Using default parameters.\n";
            return false;
        }

        std::string line;
        while (std::getline(file, line)) {
            // Skip comments and empty lines
            if (line.empty() || line[0] == '#') continue;

            // Parse key = value
            size_t pos = line.find('=');
            if (pos != std::string::npos) {
                std::string key = trim(line.substr(0, pos));
                std::string value = trim(line.substr(pos + 1));
                params[key] = value;
            }
        }

        used_keys_.clear();
        std::cout << "Loaded configuration from: " << filename << "\n";
        std::cout << "Parameters loaded: " << params.size() << "\n";
        return true;
    }

    template<typename T>
    T get(const std::string& key, T default_value) {
        used_keys_.insert(key);
        auto it = params.find(key);
        if (it == params.end()) return default_value;

        std::istringstream iss(it->second);
        T value;
        iss >> value;
        return value;
    }

    /// Check if a key exists in the loaded parameters
    bool has(const std::string& key) const {
        return params.find(key) != params.end();
    }

    /// Get raw string value for a key (marks it as used)
    std::string getString(const std::string& key) {
        used_keys_.insert(key);
        auto it = params.find(key);
        if (it == params.end()) return "";
        return it->second;
    }

    /// Warn about any keys that were loaded but never accessed
    void warnUnusedKeys() const {
        for (const auto& kv : params) {
            if (used_keys_.find(kv.first) == used_keys_.end()) {
                std::cerr << "WARNING: Unknown config key '" << kv.first << "' ignored" << std::endl;
            }
        }
    }

private:
    std::set<std::string> used_keys_;

    std::string trim(const std::string& str) {
        size_t first = str.find_first_not_of(" \t");
        if (first == std::string::npos) return "";
        size_t last = str.find_last_not_of(" \t");
        return str.substr(first, (last - first + 1));
    }
};

// ============================================================================
// Configuration Metadata
// ============================================================================

struct ConfigMetadata {
    std::string name;
    std::string status;
    std::string reference;
    std::string purpose;
};

// ============================================================================
// Helper Functions for LPBF Config File Loading
// ============================================================================

}} // Close lbm::config namespace temporarily

// Forward declaration
namespace lbm {
namespace physics {
    struct MultiphysicsConfig;
}
}

namespace lbm {
namespace config {

// Parse command line arguments
inline bool parseCommandLineArgs(int argc, char** argv, std::string& config_file,
                                  lbm::physics::MultiphysicsConfig& config,
                                  int& num_steps, std::string& output_dir) {
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--config" && i + 1 < argc) {
            config_file = argv[i + 1];
            i++;
        } else if (arg == "--steps" && i + 1 < argc) {
            num_steps = std::atoi(argv[i + 1]);
            i++;
        } else if (arg == "--output" && i + 1 < argc) {
            output_dir = argv[i + 1];
            i++;
        } else if (arg == "--help") {
            std::cout << "Usage: " << argv[0] << " [config_file] [options]\n";
            std::cout << "Arguments:\n";
            std::cout << "  config_file        Configuration file (positional, optional)\n";
            std::cout << "Options:\n";
            std::cout << "  --config <file>    Load configuration from file (alternative)\n";
            std::cout << "  --steps <N>        Override number of simulation steps\n";
            std::cout << "  --output <dir>     Override output directory\n";
            std::cout << "  --help             Show this help message\n";
            return false;
        } else if (arg[0] != '-' && config_file.empty()) {
            // Positional argument: treat as config file
            config_file = arg;
        }
    }
    return true;
}

// Load LPBF configuration from file
inline bool loadLPBFConfig(const std::string& filename,
                           lbm::physics::MultiphysicsConfig& config,
                           int& num_steps, int& output_interval,
                           std::string& output_dir,
                           ConfigMetadata* metadata = nullptr);

// Print configuration summary
inline void printConfigSummary(const lbm::physics::MultiphysicsConfig& config,
                               int num_steps, int output_interval,
                               const std::string& output_dir,
                               const ConfigMetadata* metadata = nullptr);

}} // namespace lbm::config

// ============================================================================
// Implementation (must be after physics::MultiphysicsConfig is defined)
// ============================================================================

#include "physics/multiphysics_solver.h"

namespace lbm {
namespace config {

inline bool loadLPBFConfig(const std::string& filename,
                           lbm::physics::MultiphysicsConfig& config,
                           int& num_steps, int& output_interval,
                           std::string& output_dir,
                           ConfigMetadata* metadata) {
    ConfigLoader loader;
    if (!loader.load(filename)) {
        return false;
    }

    // Load material selection (before individual property overrides)
    if (loader.has("material_type") || loader.has("material")) {
        // Mark both keys as used to avoid spurious warnings when both are present
        std::string mat_name = loader.getString("material_type");
        if (mat_name.empty()) mat_name = loader.getString("material");
        else loader.getString("material");  // Mark as used even if not selected
        // Strip surrounding quotes if present
        if (mat_name.size() >= 2 &&
            ((mat_name.front() == '"' && mat_name.back() == '"') ||
             (mat_name.front() == '\'' && mat_name.back() == '\''))) {
            mat_name = mat_name.substr(1, mat_name.size() - 2);
        }
        try {
            config.material = lbm::physics::MaterialDatabase::getMaterialByName(mat_name);
            std::cout << "[CONFIG LOADING] Material: " << config.material.name << std::endl;
        } catch (const std::runtime_error& e) {
            std::cerr << "WARNING: " << e.what() << ", using default Ti6Al4V" << std::endl;
        }
    }

    // Load individual phase change property overrides
    if (loader.has("T_solidus"))  config.material.T_solidus = loader.get("T_solidus", config.material.T_solidus);
    if (loader.has("T_liquidus")) config.material.T_liquidus = loader.get("T_liquidus", config.material.T_liquidus);
    if (loader.has("L_fusion"))   config.material.L_fusion = loader.get("L_fusion", config.material.L_fusion);
    if (loader.has("T_vaporization")) config.material.T_vaporization = loader.get("T_vaporization", config.material.T_vaporization);
    if (loader.has("L_vaporization")) config.material.L_vaporization = loader.get("L_vaporization", config.material.L_vaporization);

    // Load domain configuration
    config.nx = loader.get("nx", config.nx);
    config.ny = loader.get("ny", config.ny);
    config.nz = loader.get("nz", config.nz);
    config.dx = loader.get("dx", config.dx);

    // Load time stepping
    config.dt = loader.get("dt", config.dt);
    // Support both "num_steps" and "total_steps" for backward compatibility
    num_steps = loader.get("num_steps", num_steps);
    num_steps = loader.get("total_steps", num_steps);  // Legacy alias

    // Load laser parameters (with backward-compatible aliases)
    config.laser_power = loader.get("laser_power", config.laser_power);
    config.laser_spot_radius = loader.get("laser_spot_radius", config.laser_spot_radius);
    config.laser_spot_radius = loader.get("laser_radius", config.laser_spot_radius);  // Legacy alias
    config.laser_absorptivity = loader.get("laser_absorptivity", config.laser_absorptivity);
    config.laser_penetration_depth = loader.get("laser_penetration_depth", config.laser_penetration_depth);
    config.laser_start_x = loader.get("laser_start_x", config.laser_start_x);
    config.laser_start_y = loader.get("laser_start_y", config.laser_start_y);
    config.laser_scan_vx = loader.get("laser_scan_vx", config.laser_scan_vx);
    config.laser_scan_vy = loader.get("laser_scan_vy", config.laser_scan_vy);
    config.laser_shutoff_time = loader.get("laser_shutoff_time", config.laser_shutoff_time);
    config.laser_shutoff_time = loader.get("laser_duration", config.laser_shutoff_time);  // Legacy alias

    // Load material properties
    config.thermal_diffusivity = loader.get("thermal_diffusivity", config.thermal_diffusivity);
    config.kinematic_viscosity = loader.get("kinematic_viscosity", config.kinematic_viscosity);
    config.density = loader.get("density", config.density);

    // Load force model parameters
    config.darcy_coefficient = loader.get("darcy_coefficient", config.darcy_coefficient);
    config.surface_tension_coeff = loader.get("surface_tension_coeff", config.surface_tension_coeff);
    config.dsigma_dT = loader.get("dsigma_dT", config.dsigma_dT);

    // Load surface config
    config.surface.molar_mass = loader.get("molar_mass", config.surface.molar_mass);
    config.surface.recoil_coefficient = loader.get("recoil_coefficient", config.surface.recoil_coefficient);
    config.surface.recoil_max_pressure = loader.get("recoil_max_pressure", config.surface.recoil_max_pressure);

    // Load physics enable flags (CRITICAL FIX: These were missing!)
    config.enable_thermal = (loader.get<std::string>("enable_thermal", "true") == "true");
    config.enable_thermal_advection = (loader.get<std::string>("enable_thermal_advection", "true") == "true");
    config.enable_phase_change = (loader.get<std::string>("enable_phase_change", "false") == "true");
    config.enable_fluid = (loader.get<std::string>("enable_fluid", "true") == "true");
    config.enable_vof = (loader.get<std::string>("enable_vof", "true") == "true");
    config.enable_vof_advection = (loader.get<std::string>("enable_vof_advection", "false") == "true");
    config.enable_surface_tension = (loader.get<std::string>("enable_surface_tension", "false") == "true");
    config.enable_marangoni = (loader.get<std::string>("enable_marangoni", "false") == "true");
    config.enable_laser = (loader.get<std::string>("enable_laser", "true") == "true");
    config.enable_darcy = (loader.get<std::string>("enable_darcy", "false") == "true");
    config.enable_buoyancy = (loader.get<std::string>("enable_buoyancy", "true") == "true");

    // Load evaporation and solidification shrinkage flags (diagnostic test support)
    config.enable_evaporation_mass_loss = (loader.get<std::string>("enable_evaporation_mass_loss", "true") == "true");
    config.enable_solidification_shrinkage = (loader.get<std::string>("enable_solidification_shrinkage", "true") == "true");

    // Load recoil pressure flag
    config.enable_recoil_pressure = (loader.get<std::string>("enable_recoil_pressure", "false") == "true");

    // Load radiation boundary condition parameters (v4 fix)
    config.enable_radiation_bc = (loader.get<std::string>("enable_radiation_bc", "false") == "true");
    config.emissivity = loader.get("emissivity", config.emissivity);
    config.ambient_temperature = loader.get("ambient_temperature", config.ambient_temperature);

    // Load substrate cooling boundary condition parameters (Week 1 Tuesday fix)
    config.enable_substrate_cooling = (loader.get<std::string>("enable_substrate_cooling", "true") == "true");
    config.substrate_h_conv = loader.get("substrate_h_conv", config.substrate_h_conv);
    config.substrate_temperature = loader.get("substrate_temperature", config.substrate_temperature);

    // PLAN C DIAGNOSTIC: Verify substrate BC parameters are loaded correctly
    std::cout << "[CONFIG LOADING] Substrate BC parameters:" << std::endl;
    std::cout << "  enable_substrate_cooling = " << (config.enable_substrate_cooling ? "true" : "false") << std::endl;
    std::cout << "  substrate_h_conv = " << config.substrate_h_conv << " W/(m^2*K)" << std::endl;
    std::cout << "  substrate_temperature = " << config.substrate_temperature << " K" << std::endl;

    // DIAGNOSTIC: Verify laser position parameters are loaded correctly
    std::cout << "[CONFIG LOADING] Laser position parameters:" << std::endl;
    std::cout << "  laser_start_x = " << config.laser_start_x << " m (" << config.laser_start_x * 1e6 << " um)" << std::endl;
    std::cout << "  laser_start_y = " << config.laser_start_y << " m (" << config.laser_start_y * 1e6 << " um)" << std::endl;
    std::cout << "  (Negative values mean auto-center at domain center)" << std::endl;

    // Load output configuration
    output_dir = loader.get("output_directory", output_dir);
    output_interval = loader.get("output_interval", output_interval);

    // Parse metadata from comments (if needed in future)
    if (metadata) {
        metadata->name = "Loaded from " + filename;
        metadata->status = "Unknown";
        metadata->reference = "";
        metadata->purpose = "";
    }

    // Warn about any unrecognized config keys (catches typos)
    loader.warnUnusedKeys();

    return true;
}

inline void printConfigSummary(const lbm::physics::MultiphysicsConfig& config,
                               int num_steps, int output_interval,
                               const std::string& output_dir,
                               const ConfigMetadata* metadata) {
    std::cout << "==============================================\n";
    std::cout << "Configuration Summary\n";
    std::cout << "==============================================\n";

    if (metadata && !metadata->name.empty()) {
        std::cout << "Configuration: " << metadata->name << "\n";
        if (!metadata->status.empty()) {
            std::cout << "Status: " << metadata->status << "\n";
        }
        if (!metadata->purpose.empty()) {
            std::cout << "Purpose: " << metadata->purpose << "\n";
        }
        std::cout << "----------------------------------------------\n";
    }

    std::cout << "Domain: " << config.nx << " x " << config.ny << " x " << config.nz << " cells\n";
    std::cout << "Physical size: " << config.nx * config.dx * 1e6 << " x "
              << config.ny * config.dx * 1e6 << " x " << config.nz * config.dx * 1e6 << " μm\n";
    std::cout << "Cell size: " << config.dx * 1e6 << " μm\n";
    std::cout << "Time step: " << config.dt * 1e6 << " μs\n";
    std::cout << "Total steps: " << num_steps << "\n";
    std::cout << "Simulation time: " << num_steps * config.dt * 1e6 << " μs\n";
    std::cout << "----------------------------------------------\n";
    std::cout << "Laser power: " << config.laser_power << " W\n";
    std::cout << "Laser spot radius: " << config.laser_spot_radius * 1e6 << " μm\n";
    std::cout << "Absorptivity: " << config.laser_absorptivity << "\n";
    std::cout << "----------------------------------------------\n";
    std::cout << "Darcy coefficient: " << config.darcy_coefficient << "\n";
    std::cout << "Surface tension: " << config.surface_tension_coeff << " N/m\n";
    std::cout << "dσ/dT: " << config.dsigma_dT * 1e3 << " mN/(m·K)\n";
    std::cout << "----------------------------------------------\n";
    std::cout << "Radiation BC: " << (config.enable_radiation_bc ? "ENABLED" : "DISABLED");
    if (config.enable_radiation_bc) {
        std::cout << " (ε=" << config.emissivity
                  << ", T_amb=" << config.ambient_temperature << " K)";
    }
    std::cout << "\n";
    std::cout << "Substrate cooling BC: " << (config.enable_substrate_cooling ? "ENABLED" : "DISABLED");
    if (config.enable_substrate_cooling) {
        std::cout << " (h=" << config.substrate_h_conv << " W/(m²·K)"
                  << ", T_sub=" << config.substrate_temperature << " K)";
    }
    std::cout << "\n";
    std::cout << "----------------------------------------------\n";
    std::cout << "VOF DIAGNOSTIC FLAGS:\n";
    std::cout << "  Evaporation mass loss:      " << (config.enable_evaporation_mass_loss ? "ENABLED" : "DISABLED") << "\n";
    std::cout << "  Solidification shrinkage:   " << (config.enable_solidification_shrinkage ? "ENABLED" : "DISABLED") << "\n";
    std::cout << "  Recoil pressure:            " << (config.enable_recoil_pressure ? "ENABLED" : "DISABLED") << "\n";
    std::cout << "----------------------------------------------\n";
    std::cout << "Output directory: " << output_dir << "\n";
    std::cout << "Output interval: " << output_interval << " steps\n";
    std::cout << "==============================================\n\n";
}

}} // namespace lbm::config
