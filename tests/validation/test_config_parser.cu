/**
 * @file test_config_parser.cu
 * @brief WEEK 2: Configuration Parser Validation Test
 *
 * Purpose:
 *   Validates that the configuration parser correctly reads ALL parameters
 *   from config files, especially num_steps which was found to be hardcoded.
 *
 * Test Method:
 *   1. Create test config with unusual num_steps value (12345)
 *   2. Run simulation
 *   3. Verify simulation executes EXACTLY 12345 steps (not hardcoded 6000)
 *
 * Success Criteria:
 *   PASS if simulation respects config file parameter
 *
 * This test prevents regression of the Week 2 discovered bug:
 *   "Configuration parser reads hardcoded 6000 steps instead of config value"
 */

#include <iostream>
#include <fstream>
#include <string>
#include <cstdlib>

const int TEST_NUM_STEPS = 12345;  // Unusual value to detect hardcoding

bool createTestConfig(const std::string& config_file) {
    std::ofstream file(config_file);
    if (!file.is_open()) {
        std::cerr << "ERROR: Could not create test config file" << std::endl;
        return false;
    }

    // Minimal configuration with unusual num_steps
    file << "# Test configuration for parser validation\n";
    file << "# This config has an UNUSUAL num_steps value to detect hardcoding\n";
    file << "\n";
    file << "# Domain (small for fast test)\n";
    file << "nx = 50\n";
    file << "ny = 50\n";
    file << "nz = 25\n";
    file << "dx = 4.0e-6\n";
    file << "\n";
    file << "# Time stepping - UNUSUAL VALUE to test parser\n";
    file << "dt = 1.0e-7\n";
    file << "total_steps = " << TEST_NUM_STEPS << "  # Unusual value!\n";
    file << "\n";
    file << "# Laser (disabled for quick test)\n";
    file << "laser_power = 0.0\n";
    file << "laser_spot_radius = 50.0e-6\n";
    file << "laser_absorptivity = 0.35\n";
    file << "laser_penetration_depth = 10.0e-6\n";
    file << "\n";
    file << "# Material\n";
    file << "darcy_coefficient = 8.0e4\n";
    file << "kinematic_viscosity = 0.0333\n";
    file << "thermal_diffusivity = 5.8e-6\n";
    file << "density = 4110.0\n";
    file << "\n";
    file << "# Output (minimal)\n";
    file << "output_directory = config_parser_test\n";
    file << "output_interval = 10000  # No output during test\n";

    file.close();
    return true;
}

int parseExecutedSteps(const std::string& log_file) {
    std::ifstream file(log_file);
    if (!file.is_open()) {
        std::cerr << "WARNING: Could not open log file: " << log_file << std::endl;
        return -1;
    }

    int max_step = 0;
    std::string line;

    while (std::getline(file, line)) {
        // Look for step number in output
        size_t pos = line.find("Step");
        if (pos != std::string::npos) {
            size_t num_start = pos + 4;
            while (num_start < line.length() && !isdigit(line[num_start])) num_start++;

            std::string num_str;
            while (num_start < line.length() && isdigit(line[num_start])) {
                num_str += line[num_start++];
            }

            if (!num_str.empty()) {
                int step = std::stoi(num_str);
                max_step = std::max(max_step, step);
            }
        }

        // Also check for explicit completion message
        if (line.find("Step " + std::to_string(TEST_NUM_STEPS)) != std::string::npos) {
            return TEST_NUM_STEPS;
        }
    }

    file.close();
    return max_step;
}

bool runSimulation(const std::string& config_file, const std::string& log_file) {
    std::string cmd = "./build/run_simulation " + config_file + " > " + log_file + " 2>&1";
    std::cout << "Running: " << cmd << std::endl;

    int ret = system(cmd.c_str());
    if (ret != 0) {
        std::cerr << "WARNING: Simulation returned non-zero code " << ret << std::endl;
        std::cerr << "This may be expected for quick test termination" << std::endl;
    }

    return true;
}

int main(int argc, char** argv) {
    std::cout << "=========================================" << std::endl;
    std::cout << "WEEK 2: Configuration Parser Test" << std::endl;
    std::cout << "=========================================" << std::endl;
    std::cout << std::endl;

    const std::string config_file = "/tmp/test_config_parser.conf";
    const std::string log_file = "/tmp/test_config_parser.log";

    // Step 1: Create test configuration
    std::cout << "Creating test configuration..." << std::endl;
    std::cout << "  File: " << config_file << std::endl;
    std::cout << "  num_steps: " << TEST_NUM_STEPS << " (unusual value to detect hardcoding)" << std::endl;
    std::cout << std::endl;

    if (!createTestConfig(config_file)) {
        std::cerr << "FAIL: Could not create test config" << std::endl;
        return 1;
    }

    std::cout << "Test configuration created successfully" << std::endl;
    std::cout << std::endl;

    // Step 2: Run simulation
    std::cout << "Running simulation with test config..." << std::endl;

    if (!runSimulation(config_file, log_file)) {
        std::cerr << "FAIL: Simulation failed to run" << std::endl;
        return 1;
    }

    std::cout << "Simulation completed" << std::endl;
    std::cout << std::endl;

    // Step 3: Parse log to verify executed steps
    std::cout << "Parsing simulation output..." << std::endl;

    int executed_steps = parseExecutedSteps(log_file);

    std::cout << "  Expected steps: " << TEST_NUM_STEPS << std::endl;
    std::cout << "  Executed steps: " << executed_steps << std::endl;
    std::cout << std::endl;

    // Step 4: Analyze result
    std::cout << "=========================================" << std::endl;
    std::cout << "Configuration Parser Analysis" << std::endl;
    std::cout << "=========================================" << std::endl;
    std::cout << std::endl;

    if (executed_steps < 0) {
        std::cout << "WARN: Could not determine executed steps from log" << std::endl;
        std::cout << "Manual inspection required:" << std::endl;
        std::cout << "  cat " << log_file << std::endl;
        std::cout << std::endl;
        return 1;
    }

    bool passed = (executed_steps == TEST_NUM_STEPS);
    double error_pct = 100.0 * std::abs(executed_steps - TEST_NUM_STEPS) / double(TEST_NUM_STEPS);

    std::cout << "Difference: " << (executed_steps - TEST_NUM_STEPS) << " steps" << std::endl;
    std::cout << "Error:      " << error_pct << "%" << std::endl;
    std::cout << std::endl;

    // Common hardcoded values to check
    const int COMMON_HARDCODED[] = {2000, 3000, 6000, 8000, 10000};

    for (int hardcoded : COMMON_HARDCODED) {
        if (executed_steps == hardcoded) {
            std::cout << "DETECTED: Hardcoded value " << hardcoded << " steps!" << std::endl;
            std::cout << "This is the Week 2 discovered bug." << std::endl;
            std::cout << std::endl;
            break;
        }
    }

    // Diagnosis
    if (!passed) {
        std::cout << "Diagnosis:" << std::endl;

        if (executed_steps == 6000) {
            std::cout << "  BUG: Configuration parser uses hardcoded 6000 steps" << std::endl;
            std::cout << "  This is the EXACT bug found in Week 2 timestep study" << std::endl;
            std::cout << std::endl;
            std::cout << "Root Cause (likely):" << std::endl;
            std::cout << "  1. Default value (6000) not overridden by config file" << std::endl;
            std::cout << "  2. Parameter name mismatch ('num_steps' vs 'total_steps'?)" << std::endl;
            std::cout << "  3. Silent parsing failure with fallback to default" << std::endl;
            std::cout << std::endl;
            std::cout << "Fix Location:" << std::endl;
            std::cout << "  - Check src/config/simulation_config.cpp" << std::endl;
            std::cout << "  - Check include/config/simulation_config.h" << std::endl;
            std::cout << "  - Verify TimeConfig::n_steps parameter reading" << std::endl;

        } else if (error_pct < 10.0) {
            std::cout << "  WARN: Steps close but not exact (within 10%)" << std::endl;
            std::cout << "  Possible rounding or off-by-one error" << std::endl;
            std::cout << "  This may be acceptable depending on termination logic" << std::endl;

        } else {
            std::cout << "  ERROR: Steps significantly different (>10% error)" << std::endl;
            std::cout << "  Configuration parser is not reading total_steps correctly" << std::endl;
            std::cout << "  Manual debugging required" << std::endl;
        }

        std::cout << std::endl;
    }

    // Final verdict
    std::cout << "=========================================" << std::endl;
    std::cout << "Test Result" << std::endl;
    std::cout << "=========================================" << std::endl;

    if (passed) {
        std::cout << "PASS: Configuration parser correctly reads num_steps" << std::endl;
        std::cout << std::endl;
        std::cout << "  Expected: " << TEST_NUM_STEPS << " steps" << std::endl;
        std::cout << "  Executed: " << executed_steps << " steps" << std::endl;
        std::cout << "  Match:    EXACT" << std::endl;
        std::cout << std::endl;
        std::cout << "The Week 2 configuration parser bug has been FIXED!" << std::endl;
        return 0;

    } else {
        std::cout << "FAIL: Configuration parser bug detected" << std::endl;
        std::cout << std::endl;
        std::cout << "  Expected: " << TEST_NUM_STEPS << " steps" << std::endl;
        std::cout << "  Executed: " << executed_steps << " steps" << std::endl;
        std::cout << "  Error:    " << error_pct << "%" << std::endl;
        std::cout << std::endl;
        std::cout << "CRITICAL: Do NOT trust any convergence studies until this is fixed!" << std::endl;
        std::cout << std::endl;
        std::cout << "Impact:" << std::endl;
        std::cout << "  - Grid convergence tests run for wrong physical time" << std::endl;
        std::cout << "  - Timestep convergence tests are invalid" << std::endl;
        std::cout << "  - All Week 2 results are unreliable" << std::endl;
        std::cout << std::endl;
        std::cout << "Action Required:" << std::endl;
        std::cout << "  1. Fix configuration parser IMMEDIATELY" << std::endl;
        std::cout << "  2. Re-run ALL Week 2 convergence studies" << std::endl;
        std::cout << "  3. Verify this test PASSES before proceeding" << std::endl;
        return 1;
    }
}
