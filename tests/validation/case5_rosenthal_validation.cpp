/**
 * @file case5_rosenthal_validation.cpp
 * @brief Rosenthal analytical solution validation for Case 5 laser melting
 *
 * This program computes theoretical predictions using Rosenthal's steady-state
 * point source solution and can compare with numerical LBM results.
 *
 * Compile:
 *   g++ -std=c++17 -O3 -I../../include -I../validation \
 *       case5_rosenthal_validation.cpp -o case5_rosenthal -lm
 *
 * Run:
 *   ./case5_rosenthal
 */

#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>
#include <fstream>
#include <string>
#include "analytical/rosenthal_solution.h"

// Ti6Al4V material properties
namespace Ti6Al4V {
    constexpr float k_solid = 21.9f;       // W/(m·K)
    constexpr float k_liquid = 25.0f;      // W/(m·K)
    constexpr float rho = 4430.0f;         // kg/m³
    constexpr float cp = 546.0f;           // J/(kg·K)
    constexpr float T_solidus = 1853.0f;   // K
    constexpr float T_liquidus = 1923.0f;  // K
    constexpr float L_fusion = 286e3f;     // J/kg
    constexpr float T_vaporization = 3560.0f; // K
}

// Case 5 laser parameters
namespace Case5 {
    constexpr float laser_power = 200.0f;      // W
    constexpr float absorptivity = 0.35f;      // dimensionless
    constexpr float spot_radius = 30e-6f;      // m
    constexpr float penetration_depth = 10e-6f; // m
    constexpr float T_initial = 300.0f;        // K
}

/**
 * @brief Compute theoretical predictions for Case 5
 */
void computeTheoreticalPredictions() {
    using namespace Ti6Al4V;
    using namespace Case5;

    std::cout << std::string(70, '=') << "\n";
    std::cout << "CASE 5: ROSENTHAL ANALYTICAL SOLUTION VALIDATION\n";
    std::cout << std::string(70, '=') << "\n\n";

    // Absorbed power
    float Q = laser_power * absorptivity;

    std::cout << "Laser Parameters:\n";
    std::cout << "  Power:           " << laser_power << " W\n";
    std::cout << "  Absorptivity:    " << absorptivity << "\n";
    std::cout << "  Absorbed Power:  " << Q << " W\n";
    std::cout << "  Spot Radius:     " << spot_radius * 1e6f << " μm\n\n";

    std::cout << "Material Properties (Ti6Al4V):\n";
    std::cout << "  k (liquid):      " << k_liquid << " W/(m·K)\n";
    std::cout << "  ρ:               " << rho << " kg/m³\n";
    std::cout << "  c_p:             " << cp << " J/(kg·K)\n";
    std::cout << "  T_solidus:       " << T_solidus << " K\n";
    std::cout << "  T_liquidus:      " << T_liquidus << " K\n";
    std::cout << "  L_fusion:        " << L_fusion * 1e-3f << " kJ/kg\n\n";

    // Compute melt pool depth using Rosenthal
    float depth_liquidus = analytical::rosenthal_melt_depth_estimate(
        laser_power, absorptivity, k_liquid, T_liquidus, T_initial);

    float depth_solidus = analytical::rosenthal_melt_depth_estimate(
        laser_power, absorptivity, k_liquid, T_solidus, T_initial);

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Theoretical Predictions (Stationary Point Source):\n";
    std::cout << "  Melt pool depth (T_liquidus):  " << depth_liquidus * 1e6f << " μm\n";
    std::cout << "  Mushy zone depth (T_solidus):  " << depth_solidus * 1e6f << " μm\n";
    std::cout << "  Mushy zone thickness:          " << (depth_solidus - depth_liquidus) * 1e6f << " μm\n\n";

    // Temperature profile at various distances
    std::cout << "Temperature Profile:\n";
    std::cout << std::setw(15) << "Distance [μm]"
              << std::setw(15) << "T [K]"
              << std::setw(15) << "State\n";
    std::cout << std::string(45, '-') << "\n";

    std::vector<float> distances_um = {5, 10, 20, 30, 50, 75, 100, 150, 200};

    for (float d_um : distances_um) {
        float r = d_um * 1e-6f;
        float T = analytical::rosenthal_stationary(
            r, laser_power, absorptivity, k_liquid, T_initial, true);

        std::string state;
        if (T >= T_liquidus) {
            state = "Liquid";
        } else if (T >= T_solidus) {
            state = "Mushy";
        } else {
            state = "Solid";
        }

        std::cout << std::setw(15) << d_um
                  << std::setw(15) << std::setprecision(1) << T
                  << std::setw(15) << state << "\n";
    }

    std::cout << std::string(70, '=') << "\n\n";
}

/**
 * @brief Generate temperature profile data for plotting
 */
void generateProfileData(const std::string& filename) {
    using namespace Ti6Al4V;
    using namespace Case5;

    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "ERROR: Cannot open file " << filename << "\n";
        return;
    }

    // Write header
    file << "# Rosenthal Temperature Profile (Case 5)\n";
    file << "# Columns: r[μm], T[K], State\n";
    file << "# State: 0=Solid, 1=Mushy, 2=Liquid\n";

    // Generate profile from 5 μm to 200 μm
    for (float r_um = 5.0f; r_um <= 200.0f; r_um += 1.0f) {
        float r = r_um * 1e-6f;
        float T = analytical::rosenthal_stationary(
            r, laser_power, absorptivity, k_liquid, T_initial, true);

        int state = 0;
        if (T >= T_liquidus) {
            state = 2;  // Liquid
        } else if (T >= T_solidus) {
            state = 1;  // Mushy
        }

        file << std::fixed << std::setprecision(4)
             << r_um << " " << T << " " << state << "\n";
    }

    file.close();
    std::cout << "Profile data written to: " << filename << "\n";
}

/**
 * @brief Validation acceptance criteria
 */
void printValidationCriteria() {
    std::cout << "Validation Acceptance Criteria:\n";
    std::cout << "  1. Melt pool depth:  Within ±20% of Rosenthal prediction\n";
    std::cout << "  2. Melt pool width:  Within ±20% of Rosenthal prediction\n";
    std::cout << "  3. L2 error:         < 0.15 (15% relative error)\n";
    std::cout << "  4. Peak temperature: < T_vaporization = "
              << Ti6Al4V::T_vaporization << " K\n\n";

    std::cout << "Limitations of Rosenthal Solution:\n";
    std::cout << "  - Assumes stationary heat source (Case 5: stationary for first 50 μs)\n";
    std::cout << "  - Point source approximation (actual: Gaussian with Beer-Lambert)\n";
    std::cout << "  - Constant thermal properties (actual: T-dependent)\n";
    std::cout << "  - No phase change latent heat effect\n";
    std::cout << "  - No convection or Marangoni flow\n";
    std::cout << "  - Semi-infinite solid (actual: finite domain with substrate)\n\n";

    std::cout << "Expected Deviations:\n";
    std::cout << "  - Numerical melt pool may be ~10-30% shallower due to:\n";
    std::cout << "    * Latent heat absorption during melting\n";
    std::cout << "    * Substrate cooling boundary condition\n";
    std::cout << "    * Distributed Gaussian source vs point source\n";
    std::cout << "  - Near-source temperatures may differ due to:\n";
    std::cout << "    * Beer-Lambert volumetric absorption\n";
    std::cout << "    * Grid resolution limiting peak temperature\n\n";
}

/**
 * @brief Compare numerical results with analytical prediction
 *
 * This function template demonstrates how to validate LBM results.
 * In practice, you would load actual data from VTK files.
 */
void demonstrateComparison() {
    using namespace Ti6Al4V;
    using namespace Case5;

    std::cout << "Example Comparison Workflow:\n";
    std::cout << "  1. Run LBM simulation: ./test_laser_melting_senior\n";
    std::cout << "  2. Extract temperature profile from VTK output\n";
    std::cout << "  3. Compute melt pool depth/width from liquid fraction field\n";
    std::cout << "  4. Compare with Rosenthal predictions\n\n";

    // Example pseudo-code
    std::cout << "C++ Validation Code Template:\n";
    std::cout << R"(
    // Load numerical results
    std::vector<float> r_numerical = {...};  // distances [m]
    std::vector<float> T_numerical = {...};  // temperatures [K]
    float numerical_melt_depth = ...;         // from simulation [m]

    // Compute analytical values
    std::vector<float> T_analytical(r_numerical.size());
    for (size_t i = 0; i < r_numerical.size(); ++i) {
        T_analytical[i] = analytical::rosenthal_stationary(
            r_numerical[i], P, absorptivity, k, T0, true);
    }

    // Compute L2 error
    float l2_error = analytical::compute_l2_error(
        T_numerical.data(), T_analytical.data(), T0, r_numerical.size());

    // Compare melt pool depth
    float analytical_depth = analytical::rosenthal_melt_depth_estimate(
        P, absorptivity, k, T_liquidus, T0);
    float depth_error = std::abs(numerical_melt_depth - analytical_depth)
                       / analytical_depth * 100.0f;

    // Validation
    bool passed = (depth_error < 20.0f) && (l2_error < 0.15f);

)" << "\n";
}

/**
 * @brief Main program
 */
int main(int argc, char** argv) {
    std::cout << "\n";

    // 1. Compute theoretical predictions
    computeTheoreticalPredictions();

    // 2. Generate profile data for plotting
    std::string output_file = "rosenthal_profile_case5.dat";
    generateProfileData(output_file);
    std::cout << "\n";

    // 3. Print validation criteria
    printValidationCriteria();

    // 4. Demonstrate comparison workflow
    demonstrateComparison();

    std::cout << "Next Steps:\n";
    std::cout << "  1. Run: ./test_laser_melting_senior\n";
    std::cout << "  2. Analyze VTK output with Python script:\n";
    std::cout << "     python3 case5_rosenthal_analysis.py\n";
    std::cout << "  3. Compare numerical vs analytical melt pool metrics\n\n";

    std::cout << std::string(70, '=') << "\n";

    return 0;
}
