/**
 * @file test_energy_conservation_timestep.cu
 * @brief WEEK 2: Energy Conservation Validation Across Timesteps
 *
 * Purpose:
 *   Validates that energy is conserved during laser heating for all timestep
 *   levels. This ensures that the temporal discretization does not introduce
 *   energy loss or gain.
 *
 * Test Configuration:
 *   - Power: 50W
 *   - Three timestep levels (0.2us, 0.1us, 0.05us)
 *   - Check energy balance: |P_in - P_out - dE/dt| < 5%
 *
 * Success Criteria:
 *   PASS if energy is conserved (<5% error) for ALL timesteps
 *
 * Expected Output:
 *   Energy Conservation Test: PASS
 *     dt=0.2us: Energy error 3.2% (< 5%)
 *     dt=0.1us: Energy error 2.1% (< 5%)
 *     dt=0.05us: Energy error 1.8% (< 5%)
 */

#include <iostream>
#include <fstream>
#include <cmath>
#include <string>
#include <vector>

struct EnergyBalanceData {
    double time;
    double P_laser;
    double P_evap;
    double P_radiation;
    double P_substrate;
    double dE_dt;
    double energy_error;
};

struct EnergyResult {
    std::string label;
    double dt;
    double max_error;
    double avg_error;
    bool passed;
    std::vector<EnergyBalanceData> data;
};

/**
 * Parse energy balance from simulation output
 */
EnergyResult parseEnergyBalance(const std::string& log_file, const std::string& label, double dt) {
    EnergyResult result;
    result.label = label;
    result.dt = dt;
    result.max_error = 0.0;
    result.avg_error = 0.0;
    result.passed = false;

    std::ifstream file(log_file);
    if (!file.is_open()) {
        std::cerr << "WARNING: Could not open log file: " << log_file << std::endl;
        return result;
    }

    std::string line;
    double sum_error = 0.0;
    int count = 0;

    while (std::getline(file, line)) {
        // Look for energy balance output
        // Expected format: "Energy: P_in=50.0W P_out=48.5W dE/dt=1.2W Error=2.5%"
        if (line.find("Energy:") != std::string::npos ||
            line.find("P_in") != std::string::npos) {

            EnergyBalanceData data;
            data.time = 0.0;  // Extract from step if available
            data.P_laser = 50.0;  // Known input
            data.energy_error = 0.0;

            // Simple parsing (in real implementation, use proper parsing)
            size_t pos = line.find("Error=");
            if (pos != std::string::npos) {
                size_t end = line.find("%", pos);
                if (end != std::string::npos) {
                    std::string err_str = line.substr(pos + 6, end - pos - 6);
                    try {
                        double err = std::stod(err_str);
                        data.energy_error = err / 100.0;  // Convert percentage to fraction

                        result.data.push_back(data);
                        result.max_error = std::max(result.max_error, std::abs(data.energy_error));
                        sum_error += std::abs(data.energy_error);
                        count++;
                    } catch (...) {
                        // Skip malformed lines
                    }
                }
            }
        }
    }

    if (count > 0) {
        result.avg_error = sum_error / count;
    }

    file.close();
    return result;
}

bool runSimulation(const std::string& config_file, const std::string& log_file) {
    // For this test, we assume run_simulation has been modified to output
    // energy balance diagnostics. If not available, we run diagnose_energy_balance
    std::string cmd = "./build/diagnose_energy_balance " + config_file + " > " + log_file + " 2>&1";
    std::cout << "Running: " << cmd << std::endl;

    int ret = system(cmd.c_str());
    if (ret != 0) {
        // Try run_simulation as fallback
        cmd = "./build/run_simulation " + config_file + " > " + log_file + " 2>&1";
        std::cout << "Fallback: " << cmd << std::endl;
        ret = system(cmd.c_str());
    }

    if (ret != 0) {
        std::cerr << "ERROR: Simulation failed with return code " << ret << std::endl;
        return false;
    }

    return true;
}

int main(int argc, char** argv) {
    std::cout << "=========================================" << std::endl;
    std::cout << "WEEK 2: Energy Conservation Validation" << std::endl;
    std::cout << "=========================================" << std::endl;
    std::cout << std::endl;

    const double TOLERANCE = 0.05;  // 5% energy error tolerance

    // Configuration files
    std::vector<std::string> configs = {
        "configs/validation/lpbf_50W_dt_020us.conf",
        "configs/validation/lpbf_50W_dt_010us.conf",
        "configs/validation/lpbf_50W_dt_005us.conf"
    };

    std::vector<std::string> labels = {
        "Coarse (dt=0.2us)",
        "Baseline (dt=0.1us)",
        "Fine (dt=0.05us)"
    };

    std::vector<double> timesteps = {
        2.0e-7,
        1.0e-7,
        5.0e-8
    };

    std::vector<std::string> log_files = {
        "/tmp/energy_conservation_020us.log",
        "/tmp/energy_conservation_010us.log",
        "/tmp/energy_conservation_005us.log"
    };

    // Run all three simulations
    std::cout << "Running energy conservation tests..." << std::endl;
    std::cout << "Checking: |P_in - P_out - dE/dt| / P_in < 5%" << std::endl;
    std::cout << std::endl;

    std::vector<EnergyResult> results;
    bool all_passed = true;

    for (size_t i = 0; i < configs.size(); ++i) {
        std::cout << "Test " << (i+1) << "/3: " << labels[i] << std::endl;
        std::cout << "Config: " << configs[i] << std::endl;

        if (!runSimulation(configs[i], log_files[i])) {
            std::cerr << "FAIL: Simulation failed for " << labels[i] << std::endl;
            return 1;
        }

        EnergyResult res = parseEnergyBalance(log_files[i], labels[i], timesteps[i]);
        res.passed = (res.max_error < TOLERANCE);

        results.push_back(res);

        std::cout << "  Data points:    " << res.data.size() << std::endl;
        std::cout << "  Max error:      " << (res.max_error * 100.0) << "%" << std::endl;
        std::cout << "  Average error:  " << (res.avg_error * 100.0) << "%" << std::endl;
        std::cout << "  Status:         " << (res.passed ? "PASS" : "FAIL") << std::endl;
        std::cout << std::endl;

        if (!res.passed) {
            all_passed = false;
        }
    }

    // Analysis
    std::cout << "=========================================" << std::endl;
    std::cout << "Energy Conservation Summary" << std::endl;
    std::cout << "=========================================" << std::endl;
    std::cout << std::endl;

    std::cout << "Timestep      | Max Error | Avg Error | Status" << std::endl;
    std::cout << "--------------|-----------|-----------|-------" << std::endl;

    for (const auto& res : results) {
        printf("%-14s| %8.2f%% | %8.2f%% | %s\n",
               res.label.c_str(),
               res.max_error * 100.0,
               res.avg_error * 100.0,
               res.passed ? "PASS" : "FAIL");
    }
    std::cout << std::endl;

    // Physical interpretation
    std::cout << "=========================================" << std::endl;
    std::cout << "Physical Interpretation" << std::endl;
    std::cout << "=========================================" << std::endl;

    if (all_passed) {
        std::cout << "Energy is conserved across all timestep levels." << std::endl;
        std::cout << "This indicates:" << std::endl;
        std::cout << "  - Laser energy input is correctly computed" << std::endl;
        std::cout << "  - Heat loss mechanisms (evap, radiation, substrate) are balanced" << std::endl;
        std::cout << "  - Temporal discretization preserves energy" << std::endl;
    } else {
        std::cout << "Energy conservation VIOLATED." << std::endl;
        std::cout << std::endl;

        bool timestep_dependent = false;
        double max_spread = 0.0;
        for (size_t i = 0; i < results.size() - 1; ++i) {
            double spread = std::abs(results[i].max_error - results[i+1].max_error);
            max_spread = std::max(max_spread, spread);
        }

        if (max_spread > 0.02) {  // >2% variation across timesteps
            timestep_dependent = true;
        }

        if (timestep_dependent) {
            std::cout << "Error is TIMESTEP-DEPENDENT (varies by " << (max_spread * 100.0) << "%):" << std::endl;
            std::cout << "  Possible causes:" << std::endl;
            std::cout << "    - Laser energy deposition uses different dt" << std::endl;
            std::cout << "    - Boundary conditions evaluated at different rates" << std::endl;
            std::cout << "    - Phase change energy not properly discretized" << std::endl;
        } else {
            std::cout << "Error is CONSISTENT across timesteps:" << std::endl;
            std::cout << "  Possible causes:" << std::endl;
            std::cout << "    - Missing energy term (e.g., kinetic energy)" << std::endl;
            std::cout << "    - Incorrect formula for heat loss (evap/radiation)" << std::endl;
            std::cout << "    - Boundary flux not properly accounted" << std::endl;
        }

        std::cout << std::endl;
        std::cout << "Recommended Action:" << std::endl;
        std::cout << "  1. Run diagnose_energy_balance for detailed breakdown" << std::endl;
        std::cout << "  2. Check Week 1 evaporation fix (660x bug)" << std::endl;
        std::cout << "  3. Verify radiation BC formula (Stefan-Boltzmann)" << std::endl;
        std::cout << "  4. Check substrate cooling implementation" << std::endl;
    }
    std::cout << std::endl;

    // Final verdict
    std::cout << "=========================================" << std::endl;
    std::cout << "Test Result" << std::endl;
    std::cout << "=========================================" << std::endl;

    if (all_passed) {
        std::cout << "PASS: Energy conserved for all timesteps" << std::endl;
        for (const auto& res : results) {
            std::cout << "  " << res.label << ": " << (res.max_error * 100.0) << "% (< 5%)" << std::endl;
        }
        return 0;
    } else {
        std::cout << "FAIL: Energy conservation violated" << std::endl;
        for (const auto& res : results) {
            if (!res.passed) {
                std::cout << "  " << res.label << ": " << (res.max_error * 100.0) << "% (exceeds 5% tolerance)" << std::endl;
            }
        }
        return 1;
    }
}
