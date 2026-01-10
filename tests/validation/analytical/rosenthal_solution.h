/**
 * @file rosenthal_solution.h
 * @brief Analytical solutions for laser heating validation
 *
 * This file provides Rosenthal's analytical solution for moving heat sources,
 * used to validate the thermal solver in conduction-dominated regimes.
 *
 * References:
 * - Rosenthal, D. (1946). "The theory of moving sources of heat and its
 *   application to metal treatments." Transactions of the ASME, 68, 849-866.
 * - Carslaw, H. S., & Jaeger, J. C. (1959). "Conduction of heat in solids."
 *   Oxford University Press.
 *
 * Usage:
 *   #include "analytical/rosenthal_solution.h"
 *   float T = analytical::rosenthal_stationary(r, P, alpha, k, T0);
 */

#ifndef ROSENTHAL_SOLUTION_H
#define ROSENTHAL_SOLUTION_H

#include <cmath>

namespace analytical {

/**
 * @brief Rosenthal solution for moving point source in 2D (thin plate)
 *
 * Assumptions:
 * - Quasi-steady state (moving reference frame)
 * - Semi-infinite medium or thin plate
 * - Constant thermal properties
 * - Point heat source
 * - No phase change
 *
 * Equation:
 *   T(x,y) - T₀ = (Q / (2πkt)) × exp(-v·(r+x) / (2α))
 *
 * where:
 *   Q = P·α_abs (absorbed power)
 *   r = sqrt(x² + y²)
 *   α = k/(ρ·cp) (thermal diffusivity)
 *
 * @param x Coordinate along scan direction [m] (x > 0 ahead of source)
 * @param y Coordinate perpendicular to scan [m]
 * @param P Laser power [W]
 * @param alpha_abs Absorptivity (dimensionless, 0 < alpha < 1)
 * @param k Thermal conductivity [W/(m·K)]
 * @param thickness Plate thickness [m]
 * @param v Scan velocity [m/s]
 * @param thermal_diff Thermal diffusivity α = k/(ρ·cp) [m²/s]
 * @param T0 Ambient temperature [K]
 * @return Temperature [K]
 */
inline float rosenthal_2d_moving(
    float x, float y,
    float P, float alpha_abs,
    float k, float thickness,
    float v, float thermal_diff,
    float T0)
{
    // Absorbed power
    float Q = P * alpha_abs;

    // Distance from source
    float r = sqrtf(x*x + y*y);
    if (r < 1e-10f) r = 1e-10f; // Avoid singularity at origin

    // Peclet number term
    float peclet_term = -v * (r + x) / (2.0f * thermal_diff);

    // Temperature rise
    float dT = (Q / (2.0f * M_PI * k * thickness)) * expf(peclet_term);

    return T0 + dT;
}

/**
 * @brief Rosenthal solution for stationary point source (v = 0)
 *
 * This is the limiting case of the moving source when v → 0.
 * Results in axisymmetric temperature distribution.
 *
 * Equation:
 *   T(r) - T₀ = (Q / (2πkr)) for 2D
 *   T(r) - T₀ = (Q / (4πkr)) for 3D
 *
 * @param r Distance from source [m]
 * @param P Laser power [W]
 * @param alpha_abs Absorptivity
 * @param k Thermal conductivity [W/(m·K)]
 * @param T0 Ambient temperature [K]
 * @param is_3d Use 3D formula (true) or 2D formula (false)
 * @return Temperature [K]
 */
inline float rosenthal_stationary(
    float r,
    float P, float alpha_abs,
    float k,
    float T0,
    bool is_3d = false)
{
    if (r < 1e-10f) r = 1e-10f; // Avoid singularity

    float Q = P * alpha_abs;
    float dT;

    if (is_3d) {
        // 3D point source in semi-infinite solid
        dT = Q / (4.0f * M_PI * k * r);
    } else {
        // 2D line source (infinite cylinder)
        dT = Q / (2.0f * M_PI * k * r);
    }

    return T0 + dT;
}

/**
 * @brief Estimate melt pool width from Rosenthal (2D moving source)
 *
 * Derived from setting T(x,y) = T_melt and solving for y at x = 0
 * (maximum width perpendicular to scan direction).
 *
 * Approximate formula:
 *   W ≈ 2 × sqrt[(2·Q) / (π·k·t·v·ρ·cp·(Tm - T₀))]
 *
 * @param P Laser power [W]
 * @param alpha_abs Absorptivity
 * @param k Thermal conductivity [W/(m·K)]
 * @param thickness Plate thickness [m]
 * @param v Scan velocity [m/s]
 * @param rho Density [kg/m³]
 * @param cp Specific heat [J/(kg·K)]
 * @param Tm Melting temperature [K]
 * @param T0 Ambient temperature [K]
 * @return Melt pool width [m]
 */
inline float rosenthal_melt_width(
    float P, float alpha_abs,
    float k, float thickness,
    float v, float rho, float cp,
    float Tm, float T0)
{
    float Q = P * alpha_abs;
    float dT_melt = Tm - T0;

    // Avoid division by zero
    if (v < 1e-10f) v = 1e-10f;
    if (dT_melt < 1.0f) dT_melt = 1.0f;

    float numerator = 2.0f * Q;
    float denominator = M_PI * k * thickness * v * rho * cp * dT_melt;

    return 2.0f * sqrtf(numerator / denominator);
}

/**
 * @brief Estimate melt pool depth from Rosenthal (3D semi-infinite)
 *
 * Approximate formula for depth (z-direction) of isotherm T = Tm
 * at stationary point source:
 *
 *   D ≈ Q / (4π·k·(Tm - T₀))
 *
 * Note: This is a rough estimate. Actual depth depends on:
 * - Heat source geometry (Gaussian vs point)
 * - Substrate cooling
 * - Convection (not included in Rosenthal)
 *
 * @param P Laser power [W]
 * @param alpha_abs Absorptivity
 * @param k Thermal conductivity [W/(m·K)]
 * @param Tm Melting temperature [K]
 * @param T0 Ambient temperature [K]
 * @return Approximate melt pool depth [m]
 */
inline float rosenthal_melt_depth_estimate(
    float P, float alpha_abs,
    float k,
    float Tm, float T0)
{
    float Q = P * alpha_abs;
    float dT_melt = Tm - T0;

    if (dT_melt < 1.0f) dT_melt = 1.0f;

    return Q / (4.0f * M_PI * k * dT_melt);
}

/**
 * @brief Compute L2 relative error between numerical and analytical solutions
 *
 * L2 error = sqrt(Σ(T_num - T_ana)²) / sqrt(Σ(T_ana - T₀)²)
 *
 * @param T_numerical Array of numerical temperatures [K]
 * @param T_analytical Array of analytical temperatures [K]
 * @param T0 Reference temperature [K]
 * @param n Number of points
 * @return L2 relative error (dimensionless)
 */
inline float compute_l2_error(
    const float* T_numerical,
    const float* T_analytical,
    float T0,
    int n)
{
    double sum_squared_error = 0.0;
    double sum_squared_analytical = 0.0;

    for (int i = 0; i < n; ++i) {
        double error = T_numerical[i] - T_analytical[i];
        double ana_deviation = T_analytical[i] - T0;

        sum_squared_error += error * error;
        sum_squared_analytical += ana_deviation * ana_deviation;
    }

    if (sum_squared_analytical < 1e-20) {
        return 0.0f; // Avoid division by zero
    }

    return sqrtf(sum_squared_error / sum_squared_analytical);
}

/**
 * @brief Example usage function (for testing this header)
 */
inline void rosenthal_example() {
    // Ti6Al4V properties
    float k = 21.9f;           // W/(m·K)
    float rho = 4430.0f;       // kg/m³
    float cp = 546.0f;         // J/(kg·K)
    float alpha = k / (rho * cp); // m²/s

    // Laser parameters
    float P = 200.0f;          // W
    float absorptivity = 0.35f;
    float T0 = 300.0f;         // K
    float Tm = 1923.0f;        // K

    // Stationary laser
    float r = 50e-6f;          // 50 μm from center
    float T_stat = rosenthal_stationary(r, P, absorptivity, k, T0, true);

    printf("Rosenthal Example:\n");
    printf("  T(r=50 μm, stationary) = %.1f K\n", T_stat);

    // Moving laser
    float v = 0.5f;            // m/s
    float thickness = 150e-6f; // 150 μm
    float x = 0.0f, y = 50e-6f; // At side of beam
    float T_moving = rosenthal_2d_moving(x, y, P, absorptivity, k,
                                         thickness, v, alpha, T0);

    printf("  T(x=0, y=50 μm, v=0.5 m/s) = %.1f K\n", T_moving);

    // Melt pool width estimate
    float width = rosenthal_melt_width(P, absorptivity, k, thickness,
                                       v, rho, cp, Tm, T0);

    printf("  Estimated melt pool width = %.1f μm\n", width * 1e6f);
}

} // namespace analytical

#endif // ROSENTHAL_SOLUTION_H
