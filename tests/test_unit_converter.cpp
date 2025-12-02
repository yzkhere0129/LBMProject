/**
 * @file test_unit_converter.cpp
 * @brief Unit tests for UnitConverter class
 *
 * This test verifies that all unit conversions are mathematically correct
 * and consistent across the codebase.
 */

#include "core/unit_converter.h"
#include <iostream>
#include <cmath>
#include <cassert>

using namespace lbm::core;

// Helper function to check floating point equality with tolerance
bool approxEqual(float a, float b, float tol = 1e-6f) {
    return std::abs(a - b) < tol;
}

void testVelocityConversion() {
    std::cout << "Testing velocity conversion..." << std::endl;

    // Typical LBM parameters for metal AM
    float dx = 2e-6f;    // 2 microns
    float dt = 1e-7f;    // 0.1 microseconds
    float rho = 4420.0f; // Ti6Al4V density

    UnitConverter uc(dx, dt, rho);

    // Test: lattice velocity 0.1 should be 20 m/s physical
    // v_phys = 0.1 * (2e-6 / 1e-7) = 0.1 * 20 = 2.0 m/s
    float v_lattice = 0.1f;
    float v_phys_expected = 2.0f;
    float v_phys = uc.velocityToPhysical(v_lattice);

    std::cout << "  v_lattice = " << v_lattice << std::endl;
    std::cout << "  v_phys (expected) = " << v_phys_expected << " m/s" << std::endl;
    std::cout << "  v_phys (computed) = " << v_phys << " m/s" << std::endl;

    assert(approxEqual(v_phys, v_phys_expected));

    // Test round-trip conversion
    float v_lattice_roundtrip = uc.velocityToLattice(v_phys);
    std::cout << "  v_lattice (round-trip) = " << v_lattice_roundtrip << std::endl;
    assert(approxEqual(v_lattice_roundtrip, v_lattice));

    std::cout << "  PASSED" << std::endl;
}

void testForceConversion() {
    std::cout << "\nTesting force conversion..." << std::endl;

    float dx = 2e-6f;
    float dt = 1e-7f;
    float rho = 4420.0f;

    UnitConverter uc(dx, dt, rho);

    // Test: Physical force 1e8 N/m³ (typical Marangoni force)
    float F_phys = 1e8f;  // N/m³

    // Expected: F_lattice = F_phys * (dt² / (dx * rho))
    //         = 1e8 * ((1e-7)² / (2e-6 * 4420))
    //         = 1e8 * (1e-14 / 8.84e-3)
    //         = 1e8 * 1.131e-12
    //         = 1.131e-4
    float F_lattice_expected = F_phys * (dt * dt) / (dx * rho);
    float F_lattice = uc.forceToLattice(F_phys);

    std::cout << "  F_phys = " << F_phys << " N/m³" << std::endl;
    std::cout << "  F_lattice (expected) = " << F_lattice_expected << std::endl;
    std::cout << "  F_lattice (computed) = " << F_lattice << std::endl;

    assert(approxEqual(F_lattice, F_lattice_expected));

    // Test round-trip conversion
    float F_phys_roundtrip = uc.forceToPhysical(F_lattice);
    std::cout << "  F_phys (round-trip) = " << F_phys_roundtrip << " N/m³" << std::endl;
    assert(approxEqual(F_phys_roundtrip, F_phys, 1.0f));  // Larger tolerance for large numbers

    std::cout << "  PASSED" << std::endl;
}

void testPressureConversion() {
    std::cout << "\nTesting pressure conversion..." << std::endl;

    float dx = 2e-6f;
    float dt = 1e-7f;
    float rho = 4420.0f;

    UnitConverter uc(dx, dt, rho);

    // Test: Lattice pressure 0.1 (typical LBM value)
    float p_lattice = 0.1f;

    // Expected: p_phys = p_lattice * (rho * dx² / dt²)
    //         = 0.1 * (4420 * (2e-6)² / (1e-7)²)
    //         = 0.1 * (4420 * 4e-12 / 1e-14)
    //         = 0.1 * (17.68e-9 / 1e-14)
    //         = 0.1 * 1.768e6
    //         = 1.768e5 Pa
    float p_phys_expected = p_lattice * (rho * dx * dx) / (dt * dt);
    float p_phys = uc.pressureToPhysical(p_lattice);

    std::cout << "  p_lattice = " << p_lattice << std::endl;
    std::cout << "  p_phys (expected) = " << p_phys_expected << " Pa" << std::endl;
    std::cout << "  p_phys (computed) = " << p_phys << " Pa" << std::endl;

    assert(approxEqual(p_phys, p_phys_expected));

    // Test round-trip conversion
    float p_lattice_roundtrip = uc.pressureToLattice(p_phys);
    std::cout << "  p_lattice (round-trip) = " << p_lattice_roundtrip << std::endl;
    assert(approxEqual(p_lattice_roundtrip, p_lattice));

    std::cout << "  PASSED" << std::endl;
}

void testDiffusivityConversion() {
    std::cout << "\nTesting diffusivity conversion..." << std::endl;

    float dx = 2e-6f;
    float dt = 1e-7f;
    float rho = 4420.0f;

    UnitConverter uc(dx, dt, rho);

    // Test: Thermal diffusivity of Ti6Al4V liquid
    float alpha_phys = 5.8e-6f;  // m²/s

    // Expected: alpha_lattice = alpha_phys * (dt / dx²)
    //         = 5.8e-6 * (1e-7 / (2e-6)²)
    //         = 5.8e-6 * (1e-7 / 4e-12)
    //         = 5.8e-6 * 2.5e4
    //         = 0.145
    float alpha_lattice_expected = alpha_phys * dt / (dx * dx);
    float alpha_lattice = uc.diffusivityToLattice(alpha_phys);

    std::cout << "  alpha_phys = " << alpha_phys << " m²/s" << std::endl;
    std::cout << "  alpha_lattice (expected) = " << alpha_lattice_expected << std::endl;
    std::cout << "  alpha_lattice (computed) = " << alpha_lattice << std::endl;

    assert(approxEqual(alpha_lattice, alpha_lattice_expected));

    // Test round-trip conversion
    float alpha_phys_roundtrip = uc.diffusivityToPhysical(alpha_lattice);
    std::cout << "  alpha_phys (round-trip) = " << alpha_phys_roundtrip << " m²/s" << std::endl;
    assert(approxEqual(alpha_phys_roundtrip, alpha_phys, 1e-8f));

    std::cout << "  PASSED" << std::endl;
}

void testViscosityConversion() {
    std::cout << "\nTesting viscosity conversion..." << std::endl;

    float dx = 2e-6f;
    float dt = 1e-7f;
    float rho = 4420.0f;

    UnitConverter uc(dx, dt, rho);

    // Test: Kinematic viscosity of Ti6Al4V liquid
    float nu_phys = 1.217e-6f;  // m²/s

    // Expected: nu_lattice = nu_phys * (dt / dx²)
    // Same formula as diffusivity
    float nu_lattice_expected = nu_phys * dt / (dx * dx);
    float nu_lattice = uc.viscosityToLattice(nu_phys);

    std::cout << "  nu_phys = " << nu_phys << " m²/s" << std::endl;
    std::cout << "  nu_lattice (expected) = " << nu_lattice_expected << std::endl;
    std::cout << "  nu_lattice (computed) = " << nu_lattice << std::endl;

    assert(approxEqual(nu_lattice, nu_lattice_expected));

    // Check that tau is reasonable for LBM stability
    // tau = 3 * nu_lattice + 0.5
    float tau = 3.0f * nu_lattice + 0.5f;
    std::cout << "  Implied tau = " << tau << " (should be > 0.5 for stability)" << std::endl;
    assert(tau > 0.5f);

    std::cout << "  PASSED" << std::endl;
}

void testTimeConversion() {
    std::cout << "\nTesting time conversion..." << std::endl;

    float dx = 2e-6f;
    float dt = 1e-7f;  // 0.1 microseconds
    float rho = 4420.0f;

    UnitConverter uc(dx, dt, rho);

    // Test: 1000 timesteps should be 100 microseconds
    int timesteps = 1000;
    float t_phys_expected = 1000 * 1e-7f;  // 1e-4 seconds = 100 microseconds
    float t_phys = uc.timeToPhysical(timesteps);

    std::cout << "  timesteps = " << timesteps << std::endl;
    std::cout << "  t_phys (expected) = " << t_phys_expected * 1e6 << " us" << std::endl;
    std::cout << "  t_phys (computed) = " << t_phys * 1e6 << " us" << std::endl;

    assert(approxEqual(t_phys, t_phys_expected));

    // Test round-trip conversion
    int timesteps_roundtrip = uc.timeToLattice(t_phys);
    std::cout << "  timesteps (round-trip) = " << timesteps_roundtrip << std::endl;
    assert(timesteps_roundtrip == timesteps);

    std::cout << "  PASSED" << std::endl;
}

void testConsistencyWithMultiphysics() {
    std::cout << "\nTesting consistency with MultiphysicsSolver parameters..." << std::endl;

    // Use typical MultiphysicsSolver parameters
    float dx = 2e-6f;
    float dt = 1e-9f;    // 1 nanosecond (from default config)
    float rho = 4110.0f; // Ti6Al4V liquid density

    UnitConverter uc(dx, dt, rho);

    // Test 1: CFL limiter velocity target
    // If v_target_lattice = 0.15, what is the physical velocity?
    float v_target_lattice = 0.15f;
    float v_target_phys = uc.velocityToPhysical(v_target_lattice);
    std::cout << "  CFL target velocity: " << v_target_lattice << " (lattice) = "
              << v_target_phys << " m/s" << std::endl;

    // Test 2: Force conversion factor
    // Should match the formula in MultiphysicsSolver::fluidStep()
    float force_conversion_manual = (dt * dt) / (dx * rho);
    std::cout << "  Force conversion factor (manual): " << force_conversion_manual << std::endl;

    // Apply to a test force
    float F_test = 1e6f;  // N/m³
    float F_lattice_manual = F_test * force_conversion_manual;
    float F_lattice_uc = uc.forceToLattice(F_test);
    std::cout << "  Test force " << F_test << " N/m³:" << std::endl;
    std::cout << "    Manual conversion: " << F_lattice_manual << std::endl;
    std::cout << "    UnitConverter:     " << F_lattice_uc << std::endl;

    assert(approxEqual(F_lattice_manual, F_lattice_uc));

    std::cout << "  PASSED" << std::endl;
}

int main() {
    std::cout << "================================================" << std::endl;
    std::cout << "Unit Converter Test Suite" << std::endl;
    std::cout << "================================================" << std::endl;

    try {
        testVelocityConversion();
        testForceConversion();
        testPressureConversion();
        testDiffusivityConversion();
        testViscosityConversion();
        testTimeConversion();
        testConsistencyWithMultiphysics();

        std::cout << "\n================================================" << std::endl;
        std::cout << "ALL TESTS PASSED" << std::endl;
        std::cout << "================================================" << std::endl;

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "\nTEST FAILED: " << e.what() << std::endl;
        return 1;
    }
}
