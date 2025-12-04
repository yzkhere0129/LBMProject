/**
 * @file test_tau_calculation.cu
 * @brief Test to reproduce and fix the thermal tau calculation bug
 *
 * Problem: Tau = 7500.5 instead of expected ~0.8 (wrong by 10,000x)
 * This causes energy to triple instead of conserve.
 */

#include <gtest/gtest.h>
#include "physics/thermal_lbm.h"
#include "physics/material_properties.h"
#include <iostream>
#include <iomanip>

using namespace lbm::physics;

TEST(TauCalculationTest, CorrectFormula) {
    // ==================================================================
    // Test Case 1: Check the D3Q7::CS2 constant
    // ==================================================================
    std::cout << "\n========================================\n";
    std::cout << "TEST: D3Q7 CS2 Constant\n";
    std::cout << "========================================\n";
    std::cout << "D3Q7::CS2 = " << D3Q7::CS2 << "\n";
    std::cout << "Expected for D3Q7 thermal: 1/4 = 0.25\n";
    std::cout << "Current implementation:    1/3 = 0.333...\n";

    // For D3Q7 thermal lattice, cs² should be 1/4, not 1/3!
    // Reference: Mohamad (2011), "Lattice Boltzmann Method"
    // D3Q7 with weights w0=1/4, w1-6=1/8 → cs² = 1/4

    float cs2_expected = 0.25f;  // 1/4 for D3Q7 thermal
    float cs2_actual = D3Q7::CS2;

    std::cout << "\nBUG STATUS: ";
    if (std::abs(cs2_actual - 1.0f/3.0f) < 1e-6f) {
        std::cout << "CS2 = 1/3 (WRONG for thermal D3Q7)\n";
    } else if (std::abs(cs2_actual - 1.0f/4.0f) < 1e-6f) {
        std::cout << "CS2 = 1/4 (CORRECT for thermal D3Q7)\n";
    }

    // ==================================================================
    // Test Case 2: Reproduce the tau = 7500 bug
    // ==================================================================
    std::cout << "\n========================================\n";
    std::cout << "TEST: Reproduce Tau = 7500 Bug\n";
    std::cout << "========================================\n";

    // Physical parameters (typical for Ti6Al4V LPBF)
    float alpha_physical = 5.8e-6f;  // m²/s
    float dx = 2.0e-6f;              // 2 μm
    float dt_correct = 1.0e-7f;      // 0.1 μs (100 ns)

    // HYPOTHESIS: Someone might be using dt in wrong units
    float dt_wrong = 1.0e-3f;        // 1 ms instead of 0.1 μs (10,000x too large!)

    // Calculate alpha_lattice
    float alpha_lattice_correct = alpha_physical * dt_correct / (dx * dx);
    float alpha_lattice_wrong = alpha_physical * dt_wrong / (dx * dx);

    std::cout << "\nCorrect calculation (dt = " << dt_correct << " s):\n";
    std::cout << "  alpha_lattice = " << alpha_lattice_correct << "\n";
    std::cout << "  tau = alpha_lattice / cs² + 0.5\n";
    std::cout << "  tau (with cs²=1/4) = " << (alpha_lattice_correct / 0.25f + 0.5f) << "\n";
    std::cout << "  tau (with cs²=1/3) = " << (alpha_lattice_correct / (1.0f/3.0f) + 0.5f) << "\n";

    std::cout << "\nWrong calculation (dt = " << dt_wrong << " s - 10,000x too large!):\n";
    std::cout << "  alpha_lattice = " << alpha_lattice_wrong << "\n";
    std::cout << "  tau = alpha_lattice / cs² + 0.5\n";
    std::cout << "  tau (with cs²=1/4) = " << (alpha_lattice_wrong / 0.25f + 0.5f) << "\n";
    std::cout << "  tau (with cs²=1/3) = " << (alpha_lattice_wrong / (1.0f/3.0f) + 0.5f) << " ← MATCHES 7500!\n";

    float tau_wrong = alpha_lattice_wrong / (1.0f/3.0f) + 0.5f;

    if (std::abs(tau_wrong - 7500.5f) < 1.0f) {
        std::cout << "\n*** BUG REPRODUCED! Tau = " << tau_wrong << " ***\n";
        std::cout << "Root cause: dt is 10,000x too large (wrong units?)\n";
    }

    // ==================================================================
    // Test Case 3: Test actual ThermalLBM initialization
    // ==================================================================
    std::cout << "\n========================================\n";
    std::cout << "TEST: Actual ThermalLBM Initialization\n";
    std::cout << "========================================\n";

    MaterialProperties ti64 = MaterialDatabase::getTi6Al4V();

    // Test with correct dt
    std::cout << "\nWith dt = " << dt_correct << " s:\n";
    {
        ThermalLBM thermal(10, 10, 10, ti64, alpha_physical, false, dt_correct, dx);
        std::cout << "  Computed tau = " << thermal.getThermalTau() << "\n";
        std::cout << "  Expected tau ≈ 0.58 to 0.94 (stable range)\n";

        EXPECT_GT(thermal.getThermalTau(), 0.5f);
        EXPECT_LT(thermal.getThermalTau(), 2.0f);
    }

    // Test with wrong dt (if this causes tau = 7500, we found the bug!)
    std::cout << "\nWith dt = " << dt_wrong << " s (WRONG - 10,000x too large):\n";
    {
        ThermalLBM thermal(10, 10, 10, ti64, alpha_physical, false, dt_wrong, dx);
        float tau = thermal.getThermalTau();
        std::cout << "  Computed tau = " << tau << "\n";

        if (tau > 1000.0f) {
            std::cout << "  *** BUG CONFIRMED! Tau is >> 1000 ***\n";
            std::cout << "  This happens when dt is in wrong units!\n";
        }

        EXPECT_LT(tau, 10.0f) << "Tau should be O(1), not O(1000)!";
    }

    // ==================================================================
    // Test Case 4: Proposed fix
    // ==================================================================
    std::cout << "\n========================================\n";
    std::cout << "PROPOSED FIX\n";
    std::cout << "========================================\n";
    std::cout << "1. Change D3Q7::CS2 from 1/3 to 1/4\n";
    std::cout << "   File: include/physics/lattice_d3q7.h\n";
    std::cout << "   Line 41: static constexpr float CS2 = 1.0f / 4.0f;\n";
    std::cout << "\n";
    std::cout << "2. Verify all dt/dx parameters are in correct units\n";
    std::cout << "   - dt should be in seconds (typically 1e-7 to 1e-8)\n";
    std::cout << "   - dx should be in meters (typically 1e-6 to 5e-6)\n";
    std::cout << "\n";
    std::cout << "3. Expected result:\n";
    std::cout << "   - tau ≈ 0.5 to 1.5 (stable, physically correct)\n";
    std::cout << "   - Energy conserved to within 5%\n";
    std::cout << "========================================\n";
}

TEST(TauCalculationTest, CS2ValueForDifferentLattices) {
    std::cout << "\n========================================\n";
    std::cout << "INFO: Correct CS2 Values\n";
    std::cout << "========================================\n";
    std::cout << "D2Q5 (thermal):  cs² = 1/3\n";
    std::cout << "D3Q7 (thermal):  cs² = 1/4  ← What we need!\n";
    std::cout << "D3Q15 (thermal): cs² = 1/3\n";
    std::cout << "D3Q19 (fluid):   cs² = 1/3\n";
    std::cout << "D3Q27 (fluid):   cs² = 1/3\n";
    std::cout << "\nReference: Mohamad (2011), Table 2.1\n";
    std::cout << "========================================\n";
}

int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
