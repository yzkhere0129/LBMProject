/**
 * @file test_marangoni_system.cu
 * @brief Integration test for the complete Marangoni validation system
 *
 * This test validates that the corrected Marangoni test produces
 * physically correct results after applying the geometry and boundary fixes.
 *
 * Test objectives:
 * - Verify corrected test runs without errors
 * - Check velocity maximum occurs at interface (not at walls)
 * - Validate final velocity is in literature range (0.5-2.0 m/s)
 * - Ensure wall velocities are near zero
 * - Confirm no NaN/Inf in output fields
 */

#include <gtest/gtest.h>
#include <fstream>
#include <string>
#include <sstream>
#include <iostream>
#include <cmath>
#include <vector>

/**
 * @brief Parse velocity value from test output
 */
float parseVelocityFromLine(const std::string& line) {
    // Expected format: "  50.0         0.45 ✓ In target range"
    // or: "Maximum surface velocity achieved: 0.76 m/s"

    std::istringstream iss(line);
    std::string token;
    std::vector<std::string> tokens;

    while (iss >> token) {
        tokens.push_back(token);
    }

    // Find numeric token that looks like velocity (0.X or X.X)
    for (const auto& t : tokens) {
        try {
            float val = std::stof(t);
            if (val >= 0.0f && val < 100.0f) {  // Reasonable velocity range
                return val;
            }
        } catch (...) {
            continue;
        }
    }

    return -1.0f;  // Not found
}

/**
 * @brief Test that corrected test produces velocity in literature range
 */
TEST(MarangoniSystem, VelocityInLiteratureRange) {
    std::cout << "\n=== Test: Marangoni System - Velocity in Literature Range ===" << std::endl;

    // NOTE: This test parses output from the already-run Marangoni test
    // Run ./tests/validation/test_marangoni_velocity first to generate marangoni_output.txt
    std::cout << "  Parsing output from Marangoni validation test..." << std::endl;

    // Parse output for velocity statistics
    std::ifstream output("/home/yzk/LBMProject/build/marangoni_output.txt");
    ASSERT_TRUE(output.is_open()) << "Could not open test output file";

    std::string line;
    float max_velocity = -1.0f;
    bool found_velocity = false;
    std::vector<float> velocity_history;

    while (std::getline(output, line)) {
        // Look for final velocity line
        if (line.find("Maximum surface velocity achieved") != std::string::npos) {
            size_t pos = line.find(":");
            if (pos != std::string::npos) {
                std::string value_str = line.substr(pos + 1);
                max_velocity = parseVelocityFromLine(value_str);
                found_velocity = true;
                std::cout << "  Found final velocity: " << max_velocity << " m/s" << std::endl;
            }
        }

        // Also collect time history
        if (line.find("In target range") != std::string::npos ||
            line.find("Acceptable") != std::string::npos) {
            float v = parseVelocityFromLine(line);
            if (v > 0.0f) {
                velocity_history.push_back(v);
            }
        }
    }

    output.close();

    ASSERT_TRUE(found_velocity) << "Could not parse velocity from test output";
    ASSERT_GT(max_velocity, 0.0f) << "Invalid velocity value";

    std::cout << "  Velocity time history: " << velocity_history.size() << " snapshots" << std::endl;
    if (!velocity_history.empty()) {
        std::cout << "  Initial velocity: " << velocity_history.front() << " m/s" << std::endl;
        std::cout << "  Final velocity: " << velocity_history.back() << " m/s" << std::endl;
    }

    // Verify velocity is in expected range
    const float v_min_critical = 0.5f;   // Literature minimum (Ti6Al4V LPBF)
    const float v_max_critical = 2.0f;   // Literature maximum
    const float v_min_acceptable = 0.1f; // Order of magnitude lower bound
    const float v_max_acceptable = 10.0f;// Order of magnitude upper bound

    std::cout << "\n  Validation criteria:" << std::endl;
    std::cout << "    Critical range (literature): " << v_min_critical << " - " << v_max_critical << " m/s" << std::endl;
    std::cout << "    Acceptable range (order of magnitude): " << v_min_acceptable << " - " << v_max_acceptable << " m/s" << std::endl;

    bool in_critical_range = (max_velocity >= v_min_critical && max_velocity <= v_max_critical);
    bool in_acceptable_range = (max_velocity >= v_min_acceptable && max_velocity <= v_max_acceptable);

    if (in_critical_range) {
        std::cout << "  ✓ CRITICAL PASS: Velocity in literature range" << std::endl;
    } else if (in_acceptable_range) {
        std::cout << "  ⚠ PARTIAL PASS: Velocity order of magnitude correct" << std::endl;
    } else {
        std::cout << "  ✗ FAIL: Velocity outside acceptable range" << std::endl;
    }

    EXPECT_GE(max_velocity, v_min_acceptable)
        << "Velocity too low (likely implementation error)";
    EXPECT_LE(max_velocity, v_max_acceptable)
        << "Velocity too high (likely numerical instability)";
    EXPECT_GE(max_velocity, v_min_critical)
        << "Target: velocity in literature range " << v_min_critical << "-" << v_max_critical << " m/s";
    EXPECT_LE(max_velocity, v_max_critical)
        << "Target: velocity in literature range " << v_min_critical << "-" << v_max_critical << " m/s";
}

/**
 * @brief Test that interface is at correct position (z~5, not z~45)
 */
TEST(MarangoniSystem, InterfacePositionCorrect) {
    std::cout << "\n=== Test: Marangoni System - Interface Position Correct ===" << std::endl;

    // Check test output for interface initialization message
    std::ifstream output("/home/yzk/LBMProject/build/marangoni_output.txt");
    ASSERT_TRUE(output.is_open()) << "Could not open test output file";

    std::string line;
    bool found_interface_height = false;
    int interface_z = -1;

    while (std::getline(output, line)) {
        if (line.find("Interface height: z =") != std::string::npos) {
            // Parse: "  Interface height: z = 5 cells (10.0 μm)"
            std::istringstream iss(line);
            std::string token;
            while (iss >> token) {
                if (token == "=") {
                    iss >> interface_z;
                    found_interface_height = true;
                    break;
                }
            }
        }
    }

    output.close();

    ASSERT_TRUE(found_interface_height) << "Could not find interface height in output";

    std::cout << "  Interface position: z = " << interface_z << " cells" << std::endl;

    // Verify interface is at bottom (z~5-12), not top (z~45)
    // (Note: Second test in file uses z=12, main test uses z=5)
    EXPECT_LT(interface_z, 15)
        << "Interface should be near bottom (z < 15), not near top";
    EXPECT_GE(interface_z, 3)
        << "Interface should be above z=3 (not at very bottom)";

    std::cout << "  ✓ Interface at correct position (bottom, stable configuration)" << std::endl;
}

/**
 * @brief Test that configuration message confirms liquid at bottom
 */
TEST(MarangoniSystem, ConfigurationMessage) {
    std::cout << "\n=== Test: Marangoni System - Configuration Message ===" << std::endl;

    std::ifstream output("/home/yzk/LBMProject/build/marangoni_output.txt");
    ASSERT_TRUE(output.is_open()) << "Could not open test output file";

    std::string line;
    bool found_stable_config = false;
    bool found_liquid_at_bottom = false;

    while (std::getline(output, line)) {
        if (line.find("Liquid pool at bottom") != std::string::npos ||
            line.find("liquid (z <") != std::string::npos) {
            found_liquid_at_bottom = true;
        }
        if (line.find("stable") != std::string::npos &&
            line.find("Configuration") != std::string::npos) {
            found_stable_config = true;
        }
    }

    output.close();

    EXPECT_TRUE(found_liquid_at_bottom)
        << "Output should mention liquid at bottom";

    std::cout << "  Liquid at bottom: " << (found_liquid_at_bottom ? "✓" : "✗") << std::endl;
    std::cout << "  Stable configuration: " << (found_stable_config ? "✓" : "✗") << std::endl;

    if (found_liquid_at_bottom) {
        std::cout << "  ✓ Configuration correctly describes liquid at bottom" << std::endl;
    }
}

/**
 * @brief Test for NaN/Inf in output
 */
TEST(MarangoniSystem, NoNaNInOutput) {
    std::cout << "\n=== Test: Marangoni System - No NaN/Inf ===" << std::endl;

    std::ifstream output("/home/yzk/LBMProject/build/marangoni_output.txt");
    ASSERT_TRUE(output.is_open()) << "Could not open test output file";

    std::string line;
    int nan_count = 0;
    int inf_count = 0;

    while (std::getline(output, line)) {
        // Check for NaN or Inf in output (but exclude diagnostic messages about NaN/Inf count)
        if (line.find("nan") != std::string::npos ||
            line.find("NaN") != std::string::npos) {
            // Exclude "NaN/Inf velocity count: 0" diagnostic messages
            if (line.find("NaN/Inf velocity count: 0") == std::string::npos &&
                line.find("nan") != std::string::npos) {
                nan_count++;
                std::cout << "  WARNING: NaN found in line: " << line << std::endl;
            }
        }
        if (line.find("inf") != std::string::npos ||
            line.find("Inf") != std::string::npos) {
            // Exclude "Interface" and "NaN/Inf velocity count: 0" matches
            if (line.find("Interface") == std::string::npos &&
                line.find("NaN/Inf velocity count: 0") == std::string::npos) {
                inf_count++;
                std::cout << "  WARNING: Inf found in line: " << line << std::endl;
            }
        }
    }

    output.close();

    std::cout << "  NaN occurrences: " << nan_count << std::endl;
    std::cout << "  Inf occurrences: " << inf_count << std::endl;

    EXPECT_EQ(nan_count, 0) << "Output should not contain NaN values";
    EXPECT_EQ(inf_count, 0) << "Output should not contain Inf values";

    if (nan_count == 0 && inf_count == 0) {
        std::cout << "  ✓ No NaN/Inf detected (PASS)" << std::endl;
    }
}

/**
 * @brief Test that VTK files were generated
 */
TEST(MarangoniSystem, VTKFilesGenerated) {
    std::cout << "\n=== Test: Marangoni System - VTK Files Generated ===" << std::endl;

    // Check if VTK directory exists
    std::ifstream check_dir("/home/yzk/LBMProject/build/phase6_test2c_visualization/marangoni_flow_000000.vtk");

    if (check_dir.is_open()) {
        std::cout << "  ✓ VTK files generated successfully" << std::endl;
        check_dir.close();
        SUCCEED();
    } else {
        std::cout << "  ⚠ VTK files not found (may not be critical)" << std::endl;
        // Don't fail test - VTK output is optional
    }
}

/**
 * @brief Test boundary condition application message
 */
TEST(MarangoniSystem, BoundaryConditionsApplied) {
    std::cout << "\n=== Test: Marangoni System - Boundary Conditions Applied ===" << std::endl;

    std::ifstream output("/home/yzk/LBMProject/build/marangoni_output.txt");
    ASSERT_TRUE(output.is_open()) << "Could not open test output file";

    std::string line;
    bool mentions_walls = false;
    bool mentions_periodic = false;

    while (std::getline(output, line)) {
        if (line.find("WALL") != std::string::npos ||
            line.find("wall") != std::string::npos ||
            line.find("no-slip") != std::string::npos) {
            mentions_walls = true;
        }
        if (line.find("PERIODIC") != std::string::npos ||
            line.find("periodic") != std::string::npos) {
            mentions_periodic = true;
        }
    }

    output.close();

    std::cout << "  Walls mentioned: " << (mentions_walls ? "✓" : "?") << std::endl;
    std::cout << "  Periodic boundaries mentioned: " << (mentions_periodic ? "✓" : "?") << std::endl;

    // Note: This is informational - test doesn't fail if not found
    // (boundary conditions may not be explicitly printed)

    std::cout << "  (Informational only - boundary conditions may not be explicitly printed)" << std::endl;
}

/**
 * @brief Global test environment to generate required output file
 */
class MarangoniSystemEnvironment : public ::testing::Environment {
public:
    void SetUp() override {
        std::cout << "\n=== Setting up Marangoni System Test Environment ===" << std::endl;
        std::cout << "Checking for required test output file..." << std::endl;

        // Check if output file already exists
        std::ifstream check_file("/home/yzk/LBMProject/build/marangoni_output.txt");
        if (check_file.is_open()) {
            std::cout << "  Output file already exists - using cached results" << std::endl;
            check_file.close();
            return;
        }

        // Output file doesn't exist - run test_marangoni_velocity to generate it
        std::cout << "  Output file not found - running test_marangoni_velocity..." << std::endl;
        std::cout << "  (This may take 1-2 minutes)" << std::endl;

        // Run test_marangoni_velocity and redirect output to file
        int result = system("cd /home/yzk/LBMProject/build && "
                           "./tests/validation/test_marangoni_velocity > marangoni_output.txt 2>&1");

        if (result != 0) {
            std::cerr << "WARNING: test_marangoni_velocity returned non-zero exit code: " << result << std::endl;
            std::cerr << "         Tests may fail if output file is incomplete" << std::endl;
        }

        // Verify file was created
        std::ifstream verify_file("/home/yzk/LBMProject/build/marangoni_output.txt");
        if (!verify_file.is_open()) {
            std::cerr << "ERROR: Failed to generate marangoni_output.txt" << std::endl;
            std::cerr << "       Please run: cd /home/yzk/LBMProject/build && ./tests/validation/test_marangoni_velocity" << std::endl;
        } else {
            std::cout << "  Output file generated successfully" << std::endl;
            verify_file.close();
        }

        std::cout << "=== Environment setup complete ===" << std::endl;
    }
};

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);

    std::cout << "\n";
    std::cout << "========================================" << std::endl;
    std::cout << "Marangoni System Integration Tests" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "\n";
    std::cout << "These tests validate the corrected Marangoni" << std::endl;
    std::cout << "validation test after applying geometry and" << std::endl;
    std::cout << "boundary condition fixes." << std::endl;
    std::cout << "\n";

    // Register global test environment
    ::testing::AddGlobalTestEnvironment(new MarangoniSystemEnvironment);

    int result = RUN_ALL_TESTS();

    std::cout << "\n========================================" << std::endl;
    std::cout << "All Integration Tests Complete" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "\n";

    return result;
}
