/**
 * @file test_marangoni_velocity_scale.cu
 * @brief Test that resulting Marangoni velocity is 0.5-2.0 m/s for LPBF conditions
 */

#include "physics/marangoni.h"
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>

using namespace lbm::physics;

int main() {
    printf("=== Test: Marangoni Velocity Scale ===\n");

    // LPBF conditions for Ti6Al4V
    const float dsigma_dT = -0.00026f;  // N/(m·K)
    const float delta_T = 1000.0f;       // K (typical melt pool ΔT)
    const float mu_liquid = 0.005f;      // Pa·s (dynamic viscosity)

    MarangoniEffect marangoni(10, 10, 10, dsigma_dT, 1.0e-6f);

    // Compute characteristic Marangoni velocity
    float v_marangoni = marangoni.computeMarangoniVelocity(delta_T, mu_liquid);

    printf("Parameters:\n");
    printf("  dσ/dT = %.2e N/(m·K)\n", dsigma_dT);
    printf("  ΔT = %.0f K\n", delta_T);
    printf("  μ = %.3f Pa·s\n", mu_liquid);
    printf("\nCharacteristic Marangoni velocity:\n");
    printf("  v = |dσ/dT| × ΔT / μ\n");
    printf("    = %.2e × %.0f / %.3f\n", std::abs(dsigma_dT), delta_T, mu_liquid);
    printf("    = %.3f m/s\n", v_marangoni);
    printf("\nExpected range for LPBF: 0.5 - 2.0 m/s\n");
    printf("Reference: Khairallah et al. (2016)\n");

    bool passed = (v_marangoni >= 0.5f && v_marangoni <= 2.0f);

    if (passed) {
        printf("\nPASSED: Velocity scale matches literature\n");
        return 0;
    } else {
        printf("\nFAILED: Velocity scale outside expected range\n");
        return 1;
    }
}
