/**
 * @file ghia_1982_data.h
 * @brief Reference data from Ghia et al. (1982) for lid-driven cavity validation
 *
 * Reference:
 * Ghia, U., Ghia, K. N., & Shin, C. T. (1982).
 * "High-Re solutions for incompressible flow using the Navier-Stokes
 * equations and a multigrid method."
 * Journal of Computational Physics, 48(3), 387-411.
 *
 * This file contains benchmark data for lid-driven cavity flow at:
 * - Re = 100
 * - Re = 400
 *
 * Domain: 129×129 uniform grid
 * Boundary conditions:
 * - Top wall (y=1): u=1, v=0 (moving lid)
 * - Other walls: u=0, v=0 (no-slip)
 */

#pragma once

namespace lbm {
namespace reference {

// ============================================================================
// Re = 100
// ============================================================================

// Number of data points for Re=100
constexpr int GHIA_RE100_N_POINTS_VERTICAL = 17;
constexpr int GHIA_RE100_N_POINTS_HORIZONTAL = 17;

// U-velocity along vertical centerline (x = 0.5)
// y/H coordinates (normalized domain 0-1)
constexpr float GHIA_RE100_Y[] = {
    1.0000f, 0.9766f, 0.9688f, 0.9609f, 0.9531f, 0.8516f, 0.7344f,
    0.6172f, 0.5000f, 0.4531f, 0.2813f, 0.1719f, 0.1016f, 0.0703f,
    0.0625f, 0.0547f, 0.0000f
};

// U-velocity values at centerline (x = 0.5)
constexpr float GHIA_RE100_U[] = {
    1.0000f, 0.84123f, 0.78871f, 0.73722f, 0.68717f, 0.23151f, 0.00332f,
    -0.13641f, -0.20581f, -0.21090f, -0.15662f, -0.10150f, -0.06434f, -0.04775f,
    -0.04192f, -0.03717f, 0.00000f
};

// V-velocity along horizontal centerline (y = 0.5)
// x/L coordinates (normalized domain 0-1)
constexpr float GHIA_RE100_X[] = {
    1.0000f, 0.9688f, 0.9609f, 0.9531f, 0.9453f, 0.9063f, 0.8594f,
    0.8047f, 0.5000f, 0.2344f, 0.2266f, 0.1563f, 0.0938f, 0.0781f,
    0.0703f, 0.0625f, 0.0000f
};

// V-velocity values at centerline (y = 0.5)
constexpr float GHIA_RE100_V[] = {
    0.00000f, -0.05906f, -0.07391f, -0.08864f, -0.10313f, -0.16914f, -0.22445f,
    -0.24533f, 0.05454f, 0.17527f, 0.17507f, 0.16077f, 0.12317f, 0.10890f,
    0.10091f, 0.09233f, 0.00000f
};

// Primary vortex center location (Re=100)
constexpr float GHIA_RE100_VORTEX_X = 0.6172f;  // x/L
constexpr float GHIA_RE100_VORTEX_Y = 0.7344f;  // y/H

// ============================================================================
// Re = 400
// ============================================================================

// Number of data points for Re=400
constexpr int GHIA_RE400_N_POINTS_VERTICAL = 17;
constexpr int GHIA_RE400_N_POINTS_HORIZONTAL = 17;

// U-velocity along vertical centerline (x = 0.5)
// y/H coordinates (normalized domain 0-1)
constexpr float GHIA_RE400_Y[] = {
    1.0000f, 0.9766f, 0.9688f, 0.9609f, 0.9531f, 0.8516f, 0.7344f,
    0.6172f, 0.5000f, 0.4531f, 0.2813f, 0.1719f, 0.1016f, 0.0703f,
    0.0625f, 0.0547f, 0.0000f
};

// U-velocity values at centerline (x = 0.5)
constexpr float GHIA_RE400_U[] = {
    1.0000f, 0.75837f, 0.68439f, 0.61756f, 0.55892f, 0.29093f, 0.16256f,
    0.02135f, -0.11477f, -0.17119f, -0.32726f, -0.24299f, -0.14612f, -0.10338f,
    -0.09266f, -0.08186f, 0.00000f
};

// V-velocity along horizontal centerline (y = 0.5)
// x/L coordinates (normalized domain 0-1)
constexpr float GHIA_RE400_X[] = {
    1.0000f, 0.9688f, 0.9609f, 0.9531f, 0.9453f, 0.9063f, 0.8594f,
    0.8047f, 0.5000f, 0.2344f, 0.2266f, 0.1563f, 0.0938f, 0.0781f,
    0.0703f, 0.0625f, 0.0000f
};

// V-velocity values at centerline (y = 0.5)
constexpr float GHIA_RE400_V[] = {
    0.00000f, -0.12146f, -0.15663f, -0.19254f, -0.22847f, -0.23827f, -0.44993f,
    -0.38598f, 0.05186f, 0.30174f, 0.30203f, 0.28124f, 0.22965f, 0.20920f,
    0.19713f, 0.18360f, 0.00000f
};

// Primary vortex center location (Re=400)
constexpr float GHIA_RE400_VORTEX_X = 0.5547f;  // x/L
constexpr float GHIA_RE400_VORTEX_Y = 0.6055f;  // y/H

} // namespace reference
} // namespace lbm
