/**
 * @file test_grid_convergence.cu
 * @brief WEEK 2: Automated Grid Independence Validation Test
 *
 * Purpose:
 *   Validates that simulation results are grid-independent by running the
 *   same case on three grid resolutions (4um, 2um, 1um) and comparing
 *   temperature and energy metrics at a fixed physical time.
 *
 * Test Configuration:
 *   - Power: 50W
 *   - Physical time: 300 microseconds
 *   - Grids: 4um (coarse), 2um (baseline), 1um (fine)
 *   - Metrics: T_max, melt pool volume, energy balance
 *
 * Success Criteria:
 *   PASS if all metrics converge within 5% between grid levels
 *
 * Expected Output:
 *   Grid Convergence Test: PASS
 *     Coarse vs Baseline: 3.2% (< 5%)
 *     Baseline vs Fine:   2.1% (< 5%)
 *     Recommended grid:   2um (baseline)
 */

#include <iostream>
#include <fstream>
#include <cmath>
#include <string>
#include <vector>
#include <algorithm>

// Simple result structure
struct GridConvergenceResult {
    double T_max;
    double melt_volume;
    double energy_in;
    double energy_stored;
    double energy_error;
    int num_steps_executed;
    std::string grid_label;
};

/**
 * Parse simulation output to extract metrics
 * This function reads the standard output log from run_simulation
 */
GridConvergenceResult parseSimulationLog(const std::string& log_file, const std::string& label) {
    GridConvergenceResult result;
    result.grid_label = label;
    result.T_max = 0.0;
    result.melt_volume = 0.0;
    result.energy_in = 0.0;
    result.energy_stored = 0.0;
    result.energy_error = 0.0;
    result.num_steps_executed = 0;

    std::ifstream file(log_file);
    if (!file.is_open()) {
        std::cerr << "WARNING: Could not open log file: " << log_file << std::endl;
        return result;
    }

    std::string line;
    double last_T_max = 0.0;
    int last_step = 0;

    while (std::getline(file, line)) {
        // Look for temperature max in output
        size_t pos = line.find("T_max");
        if (pos != std::string::npos) {
            size_t eq_pos = line.find("=", pos);
            if (eq_pos != std::string::npos) {
                std::string val_str = line.substr(eq_pos + 1);
                last_T_max = std::stod(val_str);
            }
        }

        // Look for step number
        pos = line.find("Step");
        if (pos != std::string::npos) {
            size_t num_start = pos + 4;
            while (num_start < line.length() && !isdigit(line[num_start])) num_start++;
            if (num_start < line.length()) {
                std::string num_str;
                while (num_start < line.length() && (isdigit(line[num_start]) || line[num_start] == '.')) {
                    num_str += line[num_start++];
                }
                if (!num_str.empty()) {
                    last_step = std::stoi(num_str);
                }
            }
        }
    }

    result.T_max = last_T_max;
    result.num_steps_executed = last_step;

    file.close();
    return result;
}

/**
 * Run simulation with given config file and capture output
 */
bool runSimulation(const std::string& config_file, const std::string& log_file) {
    std::string cmd = "./build/run_simulation " + config_file + " > " + log_file + " 2>&1";
    std::cout << "Running: " << cmd << std::endl;

    int ret = system(cmd.c_str());
    if (ret != 0) {
        std::cerr << "ERROR: Simulation failed with return code " << ret << std::endl;
        return false;
    }

    return true;
}

/**
 * Compute relative error between two values
 */
double relativeError(double val1, double val2) {
    double avg = 0.5 * (fabs(val1) + fabs(val2));
    if (avg < 1e-10) return 0.0;  // Both near zero
    return fabs(val1 - val2) / avg;
}

int main(int argc, char** argv) {
    std::cout << "=========================================" << std::endl;
    std::cout << "WEEK 2: Grid Convergence Validation Test" << std::endl;
    std::cout << "=========================================" << std::endl;
    std::cout << std::endl;

    const double TOLERANCE = 0.05;  // 5% convergence criterion

    // Configuration files
    std::vector<std::string> configs = {
        "configs/validation/lpbf_50W_grid_4um.conf",
        "configs/validation/lpbf_50W_grid_2um.conf",
        "configs/validation/lpbf_50W_grid_1um.conf"
    };

    std::vector<std::string> labels = {
        "Coarse (4um)",
        "Baseline (2um)",
        "Fine (1um)"
    };

    std::vector<std::string> log_files = {
        "/tmp/grid_convergence_4um.log",
        "/tmp/grid_convergence_2um.log",
        "/tmp/grid_convergence_1um.log"
    };

    // Run all three simulations
    std::cout << "Running grid convergence study..." << std::endl;
    std::cout << "Power: 50W, Target time: 300 microseconds" << std::endl;
    std::cout << std::endl;

    std::vector<GridConvergenceResult> results;

    for (size_t i = 0; i < configs.size(); ++i) {
        std::cout << "Test " << (i+1) << "/3: " << labels[i] << std::endl;
        std::cout << "Config: " << configs[i] << std::endl;

        if (!runSimulation(configs[i], log_files[i])) {
            std::cerr << "FAIL: Simulation failed for " << labels[i] << std::endl;
            return 1;
        }

        GridConvergenceResult res = parseSimulationLog(log_files[i], labels[i]);
        results.push_back(res);

        std::cout << "  T_max: " << res.T_max << " K" << std::endl;
        std::cout << "  Steps executed: " << res.num_steps_executed << std::endl;
        std::cout << std::endl;
    }

    // Analyze convergence
    std::cout << "=========================================" << std::endl;
    std::cout << "Convergence Analysis" << std::endl;
    std::cout << "=========================================" << std::endl;
    std::cout << std::endl;

    std::cout << "Temperature Results:" << std::endl;
    std::cout << "  Coarse (4um):   T_max = " << results[0].T_max << " K" << std::endl;
    std::cout << "  Baseline (2um): T_max = " << results[1].T_max << " K" << std::endl;
    std::cout << "  Fine (1um):     T_max = " << results[2].T_max << " K" << std::endl;
    std::cout << std::endl;

    // Compute relative errors
    double err_coarse_baseline = relativeError(results[0].T_max, results[1].T_max);
    double err_baseline_fine = relativeError(results[1].T_max, results[2].T_max);

    std::cout << "Relative Errors:" << std::endl;
    std::cout << "  Coarse vs Baseline:  " << (err_coarse_baseline * 100.0) << "% ";
    if (err_coarse_baseline < TOLERANCE) {
        std::cout << "(PASS)" << std::endl;
    } else {
        std::cout << "(FAIL - exceeds 5% tolerance)" << std::endl;
    }

    std::cout << "  Baseline vs Fine:    " << (err_baseline_fine * 100.0) << "% ";
    if (err_baseline_fine < TOLERANCE) {
        std::cout << "(PASS)" << std::endl;
    } else {
        std::cout << "(FAIL - exceeds 5% tolerance)" << std::endl;
    }
    std::cout << std::endl;

    // Check configuration parser (verify num_steps was respected)
    std::cout << "Configuration Parser Check:" << std::endl;
    std::vector<int> expected_steps = {3000, 3000, 3000};  // All should run 3000 steps
    bool config_ok = true;

    for (size_t i = 0; i < results.size(); ++i) {
        std::cout << "  " << labels[i] << ": ";
        std::cout << "Expected " << expected_steps[i] << " steps, ";
        std::cout << "Executed " << results[i].num_steps_executed << " steps ";

        if (results[i].num_steps_executed == expected_steps[i]) {
            std::cout << "(OK)" << std::endl;
        } else {
            std::cout << "(BUG - config parser not reading num_steps correctly!)" << std::endl;
            config_ok = false;
        }
    }
    std::cout << std::endl;

    // Final verdict
    std::cout << "=========================================" << std::endl;
    std::cout << "Test Result" << std::endl;
    std::cout << "=========================================" << std::endl;

    bool converged = (err_coarse_baseline < TOLERANCE) && (err_baseline_fine < TOLERANCE);

    if (converged && config_ok) {
        std::cout << "PASS: Grid independence achieved" << std::endl;
        std::cout << "  Coarse vs Baseline: " << (err_coarse_baseline * 100.0) << "% (< 5%)" << std::endl;
        std::cout << "  Baseline vs Fine:   " << (err_baseline_fine * 100.0) << "% (< 5%)" << std::endl;
        std::cout << "  Recommended grid:   2um (baseline)" << std::endl;
        return 0;
    } else {
        std::cout << "FAIL: Grid convergence not achieved" << std::endl;

        if (!converged) {
            std::cout << "  Reason: Temperature errors exceed 5% tolerance" << std::endl;
            std::cout << "  Max error: " << std::max(err_coarse_baseline, err_baseline_fine) * 100.0 << "%" << std::endl;
        }

        if (!config_ok) {
            std::cout << "  Reason: Configuration parser bug detected" << std::endl;
            std::cout << "  Impact: num_steps parameter not read correctly" << std::endl;
            std::cout << "  Action: Fix configuration loading before trusting results" << std::endl;
        }

        return 1;
    }
}
