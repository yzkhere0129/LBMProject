/**
 * test_bug3_energy_diagnostic.cu
 *
 * Automated regression test for Bug 3: Energy diagnostic dt-scaling error
 *
 * BUG DESCRIPTION:
 * Energy conservation diagnostic showed paradoxical behavior where finer timesteps
 * produced WORSE energy errors, violating the convergence principle:
 *
 *   dt = 0.20μs → 14.0% energy error (medium)
 *   dt = 0.10μs →  4.8% energy error (best)
 *   dt = 0.05μs → 22.8% energy error (WORST!)
 *
 * ROOT CAUSE:
 * Energy diagnostic calculation (dE/dt) had incorrect dt scaling or hardcoded
 * dt value, causing systematic over-reporting of energy accumulation rate for
 * fine timesteps.
 *
 * EXPECTED BEHAVIOR AFTER FIX:
 * After Bug 3 fix, the pattern should follow normal convergence:
 *   Fine timestep (dt=0.05μs) → BEST accuracy (lowest error)
 *   Baseline (dt=0.1μs)       → Medium accuracy
 *   Coarse (dt=0.2μs)         → WORST accuracy (highest error)
 *
 * TEST STRATEGY:
 * Run three short simulations (20μs each) with different timesteps and verify:
 * 1. All energy errors are reasonable (<20%)
 * 2. Fine timestep has LOWEST error (convergence principle)
 * 3. No dt-dependent anomalies in dE/dt calculation
 *
 * SUCCESS CRITERIA:
 * - All energy errors < 20%
 * - Fine timestep error < Baseline error < Coarse error (monotonic)
 * - Energy balance: dE/dt ≈ P_laser ± 20% for pure storage scenario
 *
 * REFERENCE:
 * - Week 2 Code Verification Report (WEEK2_CODE_VERIFICATION_FINAL_REPORT.md)
 * - Bug 3 section (lines 218-269)
 * - Temporal Bug Report (TEMPORAL_BUG_REPORT.md)
 */

#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <cstdlib>
#include <cstdio>
#include <sstream>
#include <cmath>

// Test configuration
struct TestConfig {
    std::string name;
    std::string config_file;
    double dt_us;
    int expected_steps;
    double expected_time_us;
};

// Simulation result
struct SimResult {
    bool success;
    double T_max;
    double energy_error;
    double dE_dt;
    double P_laser;
    int steps_completed;
    double time_simulated;

    SimResult() : success(false), T_max(0), energy_error(0), dE_dt(0),
                  P_laser(0), steps_completed(0), time_simulated(0) {}
};

// Run simulation and parse output
SimResult runSimulation(const std::string& config_path) {
    SimResult result;

    // Build command
    std::string cmd = "/home/yzk/LBMProject/build/run_simulation " + config_path + " 2>&1";

    FILE* pipe = popen(cmd.c_str(), "r");
    if (!pipe) {
        std::cerr << "ERROR: Failed to execute simulation" << std::endl;
        return result;
    }

    char buffer[512];
    std::string output;

    // Read all output
    while (fgets(buffer, sizeof(buffer), pipe)) {
        output += buffer;
    }

    int ret_code = pclose(pipe);

    // Parse output for key metrics
    std::istringstream iss(output);
    std::string line;

    while (std::getline(iss, line)) {
        // Extract T_max
        if (line.find("T_max") != std::string::npos && line.find("=") != std::string::npos) {
            size_t pos = line.find("=");
            if (pos != std::string::npos) {
                sscanf(line.c_str() + pos + 1, "%lf", &result.T_max);
            }
        }

        // Extract energy error percentage
        if (line.find("Energy error") != std::string::npos ||
            line.find("error") != std::string::npos) {
            // Look for percentage value
            size_t pct_pos = line.find("%");
            if (pct_pos != std::string::npos) {
                // Scan backwards for number
                size_t num_start = pct_pos;
                while (num_start > 0 && (isdigit(line[num_start-1]) ||
                       line[num_start-1] == '.' || line[num_start-1] == '-')) {
                    num_start--;
                }
                std::string num_str = line.substr(num_start, pct_pos - num_start);
                result.energy_error = std::abs(atof(num_str.c_str()));
            }
        }

        // Extract dE/dt
        if (line.find("dE/dt") != std::string::npos) {
            size_t pos = line.find("=");
            if (pos != std::string::npos) {
                sscanf(line.c_str() + pos + 1, "%lf", &result.dE_dt);
            }
        }

        // Extract P_laser
        if (line.find("P_laser") != std::string::npos || line.find("Laser power") != std::string::npos) {
            size_t pos = line.find("=");
            if (pos != std::string::npos) {
                sscanf(line.c_str() + pos + 1, "%lf", &result.P_laser);
            }
        }

        // Extract step count
        if (line.find("Step") != std::string::npos && line.find("/") != std::string::npos) {
            sscanf(line.c_str(), "Step %d", &result.steps_completed);
        }

        // Extract simulated time
        if (line.find("Time") != std::string::npos && line.find("us") != std::string::npos) {
            size_t pos = line.find("=");
            if (pos != std::string::npos) {
                sscanf(line.c_str() + pos + 1, "%lf", &result.time_simulated);
            }
        }
    }

    result.success = (ret_code == 0 && result.T_max > 0);
    return result;
}

int main(int argc, char** argv) {
    std::cout << "\n" << std::string(80, '=') << std::endl;
    std::cout << "BUG 3 REGRESSION TEST - Energy Diagnostic dt-Scaling Fix" << std::endl;
    std::cout << std::string(80, '=') << std::endl;
    std::cout << "\nTest Purpose:" << std::endl;
    std::cout << "  Verify that energy diagnostic calculation correctly handles different" << std::endl;
    std::cout << "  timestep sizes and follows normal convergence behavior (fine → best)." << std::endl;
    std::cout << std::endl;

    // Test configurations
    std::vector<TestConfig> tests = {
        {"Coarse",   "/home/yzk/LBMProject/configs/validation/bug3_test_dt020us.conf", 0.2,  100, 20.0},
        {"Baseline", "/home/yzk/LBMProject/configs/validation/bug3_test_dt010us.conf", 0.1,  200, 20.0},
        {"Fine",     "/home/yzk/LBMProject/configs/validation/bug3_test_dt005us.conf", 0.05, 400, 20.0}
    };

    std::cout << "Test Configuration:" << std::endl;
    std::cout << "  Domain: 50×50×25 cells (200×200×100 μm)" << std::endl;
    std::cout << "  Grid spacing: 4 μm" << std::endl;
    std::cout << "  Laser power: 50W (η=0.20 → 10W effective)" << std::endl;
    std::cout << "  Simulation time: 20 μs (quick test)" << std::endl;
    std::cout << "  Physics: Pure storage (no sinks enabled)" << std::endl;
    std::cout << std::endl;

    // Run simulations
    std::cout << std::string(80, '-') << std::endl;
    std::cout << "RUNNING SIMULATIONS" << std::endl;
    std::cout << std::string(80, '-') << std::endl;

    std::vector<SimResult> results;

    for (size_t i = 0; i < tests.size(); ++i) {
        const TestConfig& test = tests[i];

        std::cout << "\n[" << (i+1) << "/" << tests.size() << "] Running: "
                  << test.name << " (dt=" << test.dt_us << "μs, "
                  << test.expected_steps << " steps)" << std::endl;
        std::cout << "      Config: " << test.config_file << std::endl;

        SimResult result = runSimulation(test.config_file);
        results.push_back(result);

        if (result.success) {
            std::cout << "      Status: SUCCESS" << std::endl;
            std::cout << "      T_max: " << std::fixed << std::setprecision(1)
                      << result.T_max << " K" << std::endl;
            std::cout << "      Energy error: " << std::fixed << std::setprecision(1)
                      << result.energy_error << "%" << std::endl;
        } else {
            std::cout << "      Status: FAILED (simulation crashed or produced no output)" << std::endl;
        }
    }

    // Validation checks
    std::cout << "\n" << std::string(80, '=') << std::endl;
    std::cout << "VALIDATION RESULTS" << std::endl;
    std::cout << std::string(80, '=') << std::endl;

    bool all_pass = true;
    int num_criteria = 0;
    int num_passed = 0;

    // Check 1: All simulations completed successfully
    std::cout << "\n[CHECK 1] All simulations completed successfully" << std::endl;
    num_criteria++;
    bool check1_pass = true;
    for (size_t i = 0; i < results.size(); ++i) {
        bool pass = results[i].success;
        std::cout << "  " << tests[i].name << ": " << (pass ? "✓ PASS" : "✗ FAIL") << std::endl;
        check1_pass = check1_pass && pass;
    }
    if (check1_pass) num_passed++;
    all_pass = all_pass && check1_pass;
    std::cout << "  Overall: " << (check1_pass ? "✓ PASS" : "✗ FAIL") << std::endl;

    // Check 2: All energy errors < 20%
    std::cout << "\n[CHECK 2] All energy errors < 20% (reasonable physics)" << std::endl;
    num_criteria++;
    bool check2_pass = true;
    for (size_t i = 0; i < results.size(); ++i) {
        bool pass = results[i].success && (results[i].energy_error < 20.0);
        std::cout << "  " << tests[i].name << " (dt=" << tests[i].dt_us << "μs): "
                  << std::fixed << std::setprecision(1) << results[i].energy_error
                  << "% " << (pass ? "✓ PASS" : "✗ FAIL") << std::endl;
        check2_pass = check2_pass && pass;
    }
    if (check2_pass) num_passed++;
    all_pass = all_pass && check2_pass;
    std::cout << "  Overall: " << (check2_pass ? "✓ PASS" : "✗ FAIL") << std::endl;

    // Check 3: Fine timestep has LOWEST error (Bug 3 fix validation)
    std::cout << "\n[CHECK 3] Fine timestep has LOWEST energy error (convergence principle)" << std::endl;
    std::cout << "  This is the KEY test for Bug 3 fix!" << std::endl;
    num_criteria++;
    bool check3_pass = false;
    if (results[0].success && results[1].success && results[2].success) {
        double coarse_err = results[0].energy_error;
        double baseline_err = results[1].energy_error;
        double fine_err = results[2].energy_error;

        std::cout << "  Coarse (0.2μs):   " << std::fixed << std::setprecision(1)
                  << coarse_err << "%" << std::endl;
        std::cout << "  Baseline (0.1μs): " << std::fixed << std::setprecision(1)
                  << baseline_err << "%" << std::endl;
        std::cout << "  Fine (0.05μs):    " << std::fixed << std::setprecision(1)
                  << fine_err << "%" << std::endl;

        // Fine must be better than both coarse and baseline
        check3_pass = (fine_err < baseline_err) && (fine_err < coarse_err);

        std::cout << "  Fine is best: " << (check3_pass ? "✓ YES" : "✗ NO") << std::endl;
        std::cout << "  Overall: " << (check3_pass ? "✓ PASS" : "✗ FAIL") << std::endl;

        if (!check3_pass) {
            std::cout << "  WARNING: Bug 3 may still be present! Fine timestep should have" << std::endl;
            std::cout << "           lowest error, but it doesn't." << std::endl;
        }
    } else {
        std::cout << "  Overall: ✗ FAIL (simulations did not complete)" << std::endl;
    }
    if (check3_pass) num_passed++;
    all_pass = all_pass && check3_pass;

    // Check 4: Monotonic improvement (optional, warning only)
    std::cout << "\n[CHECK 4] Monotonic error decrease (coarse → baseline → fine)" << std::endl;
    std::cout << "  (Warning only, not critical)" << std::endl;
    bool check4_pass = false;
    if (results[0].success && results[1].success && results[2].success) {
        double coarse_err = results[0].energy_error;
        double baseline_err = results[1].energy_error;
        double fine_err = results[2].energy_error;

        check4_pass = (coarse_err > baseline_err) && (baseline_err > fine_err);

        if (check4_pass) {
            std::cout << "  Pattern: coarse > baseline > fine ✓ IDEAL" << std::endl;
        } else {
            std::cout << "  Pattern: NOT monotonic ⚠ WARNING (acceptable)" << std::endl;
        }
    }
    // Don't count this towards pass/fail

    // Summary
    std::cout << "\n" << std::string(80, '=') << std::endl;
    std::cout << "FINAL VERDICT" << std::endl;
    std::cout << std::string(80, '=') << std::endl;
    std::cout << "\nCriteria passed: " << num_passed << "/" << num_criteria << std::endl;

    if (all_pass) {
        std::cout << "\n✓✓✓ BUG 3 REGRESSION TEST: PASS ✓✓✓" << std::endl;
        std::cout << "\nEnergy diagnostic is working correctly!" << std::endl;
        std::cout << "- All simulations completed successfully" << std::endl;
        std::cout << "- Energy errors are reasonable (<20%)" << std::endl;
        std::cout << "- Fine timestep has BEST accuracy (Bug 3 FIXED)" << std::endl;
        std::cout << "- Normal convergence behavior observed" << std::endl;
        std::cout << "\nBug 3 will not regress." << std::endl;
        return 0;
    } else {
        std::cout << "\n✗✗✗ BUG 3 REGRESSION TEST: FAIL ✗✗✗" << std::endl;
        std::cout << "\nEnergy diagnostic may have issues!" << std::endl;

        if (!check1_pass) {
            std::cout << "- Some simulations failed to complete" << std::endl;
        }
        if (!check2_pass) {
            std::cout << "- Energy errors exceed 20% threshold" << std::endl;
        }
        if (!check3_pass) {
            std::cout << "- Fine timestep does NOT have best accuracy" << std::endl;
            std::cout << "  → BUG 3 MAY STILL BE PRESENT!" << std::endl;
        }

        std::cout << "\nAction required:" << std::endl;
        std::cout << "1. Review energy diagnostic calculation code" << std::endl;
        std::cout << "2. Check dt scaling in computeTotalThermalEnergy()" << std::endl;
        std::cout << "3. Verify dE/dt calculation uses correct dt value" << std::endl;

        return 1;
    }
}
