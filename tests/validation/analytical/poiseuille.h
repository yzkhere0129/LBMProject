/**
 * @file poiseuille.h
 * @brief Analytical solution for Poiseuille flow (2D plane channel flow)
 *
 * This file provides the analytical solution for steady-state Poiseuille flow
 * (pressure-driven or force-driven laminar flow between parallel plates).
 * Used to validate the fluid LBM solver for viscous incompressible flow.
 *
 * Physical setup:
 *   - 2D channel with parallel walls at y = 0 and y = H
 *   - No-slip boundary conditions at walls (u = 0)
 *   - Driven by constant body force F or pressure gradient dp/dx
 *   - Steady-state parabolic velocity profile
 *
 * Governing equation:
 *   0 = -dp/dx + μ·∂²u/∂y² + ρ·F
 *
 * Analytical solution:
 *   u(y) = u_max · [1 - (2y/H - 1)²]
 *   u_max = F·H²/(8·μ)  for body force F
 *   u_max = (dp/dx)·H²/(8·μ)  for pressure gradient
 *
 * where:
 *   y: Distance from bottom wall [m]
 *   H: Channel height [m]
 *   F: Body force per unit mass [m/s²]
 *   μ: Dynamic viscosity [Pa·s]
 *   ρ: Density [kg/m³]
 *   u_max: Maximum velocity at centerline [m/s]
 *
 * References:
 * - White, F. M. (2006). "Viscous Fluid Flow" (3rd ed.). McGraw-Hill.
 * - Succi, S. (2001). "The Lattice Boltzmann Equation for Fluid Dynamics
 *   and Beyond." Oxford University Press.
 * - Krüger et al. (2017). "The Lattice Boltzmann Method: Principles and
 *   Practice." Springer.
 *
 * Usage:
 *   #include "analytical/poiseuille.h"
 *   float u = analytical::poiseuille_velocity(y, H, u_max);
 */

#ifndef POISEUILLE_H
#define POISEUILLE_H

#include <cmath>

namespace analytical {

/**
 * @brief Compute maximum velocity for force-driven Poiseuille flow
 *
 * Formula: u_max = F·H²/(8·ν)
 *
 * Derivation:
 *   Force balance: F = ν·∂²u/∂y²
 *   Boundary conditions: u(0) = u(H) = 0
 *   Solution: u(y) = (F/(2ν))·y·(H - y)
 *   Maximum at y = H/2: u_max = F·H²/(8·ν)
 *
 * @param F Body force per unit mass [m/s²]
 * @param H Channel height [m]
 * @param nu Kinematic viscosity ν = μ/ρ [m²/s]
 * @return Maximum velocity u_max [m/s]
 */
inline float poiseuille_u_max(float F, float H, float nu) {
    return (F * H * H) / (8.0f * nu);
}

/**
 * @brief Compute velocity at position y for Poiseuille flow
 *
 * Formula: u(y) = u_max · [1 - (2y/H - 1)²]
 *
 * This is the normalized parabolic profile:
 *   - u(0) = 0 (bottom wall, no-slip)
 *   - u(H/2) = u_max (centerline, maximum)
 *   - u(H) = 0 (top wall, no-slip)
 *
 * Coordinate convention:
 *   - y = 0: bottom wall
 *   - y = H/2: centerline
 *   - y = H: top wall
 *
 * @param y Distance from bottom wall [m]
 * @param H Channel height [m]
 * @param u_max Maximum velocity at centerline [m/s]
 * @return Velocity u(y) [m/s]
 */
inline float poiseuille_velocity(float y, float H, float u_max) {
    // Normalized position: η = y/H ∈ [0, 1]
    float eta = y / H;

    // Parabolic profile: u(η) = u_max · 4·η·(1 - η)
    // Equivalent form: u(η) = u_max · [1 - (2η - 1)²]
    float xi = 2.0f * eta - 1.0f;  // ξ = 2η - 1 ∈ [-1, 1]

    return u_max * (1.0f - xi * xi);
}

/**
 * @brief Compute L2 error between numerical and analytical Poiseuille profile
 *
 * L2 error = sqrt(Σ(u_num - u_ana)²) / sqrt(Σ u_ana²)
 *
 * This computes the relative L2 norm error for a 1D velocity profile:
 *   ||e||_L2 / ||u_analytical||_L2
 *
 * @param u_numerical Array of numerical velocities [m/s]
 * @param u_analytical Array of analytical velocities [m/s]
 * @param n Number of points
 * @return L2 relative error (dimensionless)
 */
inline float poiseuille_l2_error(
    const float* u_numerical,
    const float* u_analytical,
    int n)
{
    double sum_squared_error = 0.0;
    double sum_squared_analytical = 0.0;

    for (int i = 0; i < n; ++i) {
        double error = u_numerical[i] - u_analytical[i];
        double ana = u_analytical[i];

        sum_squared_error += error * error;
        sum_squared_analytical += ana * ana;
    }

    if (sum_squared_analytical < 1e-20) {
        return 0.0f; // Avoid division by zero
    }

    return sqrtf(sum_squared_error / sum_squared_analytical);
}

/**
 * @brief Compute Reynolds number for Poiseuille flow
 *
 * Re = (u_mean · H) / ν
 *
 * where u_mean = (2/3)·u_max for parabolic profile
 *
 * @param u_max Maximum velocity [m/s]
 * @param H Channel height [m]
 * @param nu Kinematic viscosity [m²/s]
 * @return Reynolds number (dimensionless)
 */
inline float poiseuille_reynolds(float u_max, float H, float nu) {
    float u_mean = (2.0f / 3.0f) * u_max;
    return (u_mean * H) / nu;
}

/**
 * @brief Compute volumetric flow rate for Poiseuille flow
 *
 * Q = (2/3) · u_max · H · W
 *
 * where W is channel width (unit width assumed if not specified)
 *
 * This comes from integrating the parabolic profile:
 *   Q = ∫₀ᴴ u(y) dy = (2/3)·u_max·H
 *
 * @param u_max Maximum velocity [m/s]
 * @param H Channel height [m]
 * @param W Channel width [m] (default: 1.0)
 * @return Flow rate [m³/s]
 */
inline float poiseuille_flow_rate(float u_max, float H, float W = 1.0f) {
    return (2.0f / 3.0f) * u_max * H * W;
}

/**
 * @brief Compute wall shear stress for Poiseuille flow
 *
 * τ_wall = μ · (∂u/∂y)|_wall = (4·μ·u_max) / H
 *
 * At walls (y=0 or y=H), the velocity gradient is:
 *   ∂u/∂y = ±4·u_max/H
 *
 * @param u_max Maximum velocity [m/s]
 * @param H Channel height [m]
 * @param mu Dynamic viscosity μ [Pa·s]
 * @return Wall shear stress [Pa]
 */
inline float poiseuille_wall_shear(float u_max, float H, float mu) {
    return (4.0f * mu * u_max) / H;
}

/**
 * @brief Example usage function (for testing this header)
 */
inline void poiseuille_example() {
    // Problem setup
    float H = 100.0e-6f;       // 100 μm channel height
    float nu = 1e-5f;          // Kinematic viscosity (m²/s)
    float F = 1000.0f;         // Body force (m/s²)
    float rho = 1000.0f;       // Density (kg/m³)

    // Compute maximum velocity
    float u_max = poiseuille_u_max(F, H, nu);

    printf("Poiseuille Flow Example:\n");
    printf("  Channel height: %.1f μm\n", H * 1e6f);
    printf("  Kinematic viscosity: %.2e m²/s\n", nu);
    printf("  Body force: %.1f m/s²\n", F);
    printf("  Max velocity: %.4f m/s\n", u_max);

    // Compute Reynolds number
    float Re = poiseuille_reynolds(u_max, H, nu);
    printf("  Reynolds number: %.2f\n", Re);

    // Sample velocity at centerline and quarter-height
    float u_center = poiseuille_velocity(H/2, H, u_max);
    float u_quarter = poiseuille_velocity(H/4, H, u_max);

    printf("  u(H/2) = %.4f m/s (should equal u_max)\n", u_center);
    printf("  u(H/4) = %.4f m/s (75%% of u_max)\n", u_quarter);

    // Wall shear stress
    float mu = nu * rho;
    float tau_wall = poiseuille_wall_shear(u_max, H, mu);
    printf("  Wall shear stress: %.2f Pa\n", tau_wall);
}

} // namespace analytical

#endif // POISEUILLE_H
