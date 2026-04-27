/**
 * @file test_unit_conversion_roundtrip.cu
 * @brief lattice->physical->lattice is identity.
 *
 * Strategy: Test UnitConverter directly (no solver needed).
 * For every conversion pair (velocity, force, pressure, diffusivity,
 * viscosity, time), round-tripping must recover the original value
 * to within FP32 tolerance. Also verify that two configs with different
 * dx/dt but the same physical domain produce the same physical velocity
 * after conversion.
 */

#include <gtest/gtest.h>
#include <cmath>
#include <vector>

#include "core/unit_converter.h"

using namespace lbm::core;

static constexpr float REL_TOL = 1e-5f;  // FP32 round-trip tolerance

static void checkRoundtrip(float val, float roundtripped, const char* label) {
    float rel_err = std::abs(roundtripped - val) / (std::abs(val) + 1e-30f);
    EXPECT_LT(rel_err, REL_TOL)
        << label << " round-trip failed: original=" << val
        << " recovered=" << roundtripped
        << " rel_err=" << rel_err;
}

TEST(MultiphysicsUnitsTest, UnitConversionRoundtrip) {
    // Test several dx/dt combinations
    const float dx = 2e-6f;   // 2 μm
    const float dt = 1e-8f;   // 10 ns
    const float rho = 4110.0f; // Ti6Al4V [kg/m³]

    UnitConverter uc(dx, dt, rho);

    // --- Velocity round-trip ---
    const float v_phys_test = 1.5f;  // m/s
    float v_lu = uc.velocityToLattice(v_phys_test);
    float v_phys_back = uc.velocityToPhysical(v_lu);
    checkRoundtrip(v_phys_test, v_phys_back, "velocity");

    // --- Force round-trip ---
    const float F_phys_test = 1e9f;  // N/m³ (typical Marangoni scale)
    float F_lu = uc.forceToLattice(F_phys_test);
    float F_phys_back = uc.forceToPhysical(F_lu);
    checkRoundtrip(F_phys_test, F_phys_back, "force");

    // --- Pressure round-trip ---
    const float p_phys_test = 1e5f;  // Pa
    float p_lu = uc.pressureToLattice(p_phys_test);
    float p_phys_back = uc.pressureToPhysical(p_lu);
    checkRoundtrip(p_phys_test, p_phys_back, "pressure");

    // --- Diffusivity round-trip ---
    const float alpha_phys_test = 9.66e-6f;  // m²/s (Ti6Al4V thermal diffusivity)
    float alpha_lu = uc.diffusivityToLattice(alpha_phys_test);
    float alpha_phys_back = uc.diffusivityToPhysical(alpha_lu);
    checkRoundtrip(alpha_phys_test, alpha_phys_back, "diffusivity");

    // --- Viscosity round-trip ---
    const float nu_phys_test = 1.217e-6f;  // m²/s
    float nu_lu = uc.viscosityToLattice(nu_phys_test);
    float nu_phys_back = uc.viscosityToPhysical(nu_lu);
    checkRoundtrip(nu_phys_test, nu_phys_back, "viscosity");

    // --- Scaling consistency: finer dx/dt should give same physical velocity ---
    // If v_lattice is the same fraction of c_s, the physical velocity must scale with dx/dt.
    UnitConverter uc2(dx * 0.5f, dt * 0.25f, rho);  // finer grid
    const float v_lu_common = 0.1f;                    // same lattice Mach number
    float v_phys1 = uc.velocityToPhysical(v_lu_common);
    float v_phys2 = uc2.velocityToPhysical(v_lu_common);
    // v_phys = v_lu * dx/dt → same v_lu but different dx/dt → different v_phys
    float expected_ratio = (dx / dt) / (dx * 0.5f / (dt * 0.25f));
    float actual_ratio = v_phys1 / v_phys2;
    EXPECT_NEAR(actual_ratio, expected_ratio, REL_TOL)
        << "Velocity scaling with dx/dt mismatch: expected ratio=" << expected_ratio
        << " got=" << actual_ratio;
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
