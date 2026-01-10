/**
 * @file taylor_green.h
 * @brief Analytical solution for 2D Taylor-Green vortex
 *
 * This file provides the exact analytical solution for the 2D Taylor-Green vortex,
 * a fundamental benchmark for validating incompressible Navier-Stokes solvers.
 *
 * ANALYTICAL SOLUTION:
 * The 2D Taylor-Green vortex is a periodic array of counter-rotating vortices
 * that decay through viscous diffusion. The velocity field is:
 *
 *   u(x,y,t) = U₀ × sin(kx) × cos(ky) × exp(-2νk²t)
 *   v(x,y,t) = -U₀ × cos(kx) × sin(ky) × exp(-2νk²t)
 *
 * where:
 *   U₀ = initial velocity amplitude [m/s]
 *   k = 2π/L = wavenumber [1/m]
 *   ν = kinematic viscosity [m²/s]
 *   t = time [s]
 *
 * KEY PROPERTIES:
 * 1. Divergence-free: ∇·u = 0 (incompressible)
 * 2. Kinetic energy decays as E(t) = E₀ × exp(-4νk²t)
 * 3. Enstrophy (vorticity squared) decays as Ω(t) = Ω₀ × exp(-4νk²t)
 * 4. Exact solution to incompressible Navier-Stokes
 *
 * PHYSICAL INTERPRETATION:
 * - Energy dissipation rate: dE/dt = -2νk²E
 * - Decay time scale: τ = 1/(2νk²) = L²/(2πν)
 * - Reynolds number: Re = U₀L/ν
 *
 * VALIDATION METRICS:
 * - Velocity field L2 error vs analytical solution
 * - Kinetic energy decay rate
 * - Enstrophy conservation/decay
 * - Vorticity field accuracy
 *
 * REFERENCES:
 * - Taylor & Green (1937): "Mechanism of the production of small eddies from large ones"
 * - Brachet et al. (1983): "Small-scale structure of the Taylor-Green vortex"
 * - Tutty (1988): "Accurate solutions of the 3D Navier-Stokes equations"
 */

#pragma once

#include <cmath>

namespace analytical {

/**
 * @brief 2D Taylor-Green vortex analytical solution
 *
 * This class provides exact solutions for velocity, vorticity, and energy
 * for the 2D Taylor-Green vortex benchmark.
 */
class TaylorGreen2D {
public:
    float U0;      ///< Initial velocity amplitude [m/s]
    float L;       ///< Domain size (periodic) [m]
    float nu;      ///< Kinematic viscosity [m²/s]
    float k;       ///< Wavenumber = 2π/L [1/m]
    float E0;      ///< Initial kinetic energy density [J/m³]
    float rho;     ///< Density [kg/m³]

    /**
     * @brief Constructor
     * @param U0_ Initial velocity amplitude [m/s]
     * @param L_ Domain size (periodic in x and y) [m]
     * @param nu_ Kinematic viscosity [m²/s]
     * @param rho_ Density [kg/m³] (default: 1.0)
     */
    TaylorGreen2D(float U0_, float L_, float nu_, float rho_ = 1.0f)
        : U0(U0_), L(L_), nu(nu_), rho(rho_)
    {
        k = 2.0f * M_PI / L;
        // Initial kinetic energy density: E₀ = 0.5 × ρ × U₀²
        // (averaged over domain, factor of 1/2 from sin²/cos² average)
        E0 = 0.25f * rho * U0 * U0;
    }

    /**
     * @brief Compute decay factor at time t
     * @param t Time [s]
     * @return Exponential decay factor exp(-2νk²t)
     */
    float decayFactor(float t) const {
        return std::exp(-2.0f * nu * k * k * t);
    }

    /**
     * @brief Compute x-velocity component
     * @param x Position x-coordinate [m]
     * @param y Position y-coordinate [m]
     * @param t Time [s]
     * @return u(x,y,t) [m/s]
     */
    float velocityU(float x, float y, float t) const {
        float decay = decayFactor(t);
        return U0 * std::sin(k * x) * std::cos(k * y) * decay;
    }

    /**
     * @brief Compute y-velocity component
     * @param x Position x-coordinate [m]
     * @param y Position y-coordinate [m]
     * @param t Time [s]
     * @return v(x,y,t) [m/s]
     */
    float velocityV(float x, float y, float t) const {
        float decay = decayFactor(t);
        return -U0 * std::cos(k * x) * std::sin(k * y) * decay;
    }

    /**
     * @brief Compute vorticity (z-component, ω = ∂v/∂x - ∂u/∂y)
     * @param x Position x-coordinate [m]
     * @param y Position y-coordinate [m]
     * @param t Time [s]
     * @return ω(x,y,t) [1/s]
     */
    float vorticity(float x, float y, float t) const {
        float decay = decayFactor(t);
        // ω = ∂v/∂x - ∂u/∂y
        //   = U₀k sin(kx)sin(ky) - (-U₀k sin(kx)sin(ky))
        //   = 2U₀k sin(kx)sin(ky) × exp(-2νk²t)
        return 2.0f * U0 * k * std::sin(k * x) * std::sin(k * y) * decay;
    }

    /**
     * @brief Compute kinetic energy density at time t
     * @param t Time [s]
     * @return E(t) = E₀ × exp(-4νk²t) [J/m³]
     */
    float kineticEnergy(float t) const {
        return E0 * std::exp(-4.0f * nu * k * k * t);
    }

    /**
     * @brief Compute enstrophy (integrated vorticity squared) at time t
     * @param t Time [s]
     * @return Ω(t) = Ω₀ × exp(-4νk²t) [1/s²]
     */
    float enstrophy(float t) const {
        // Enstrophy: Ω = 0.5 × ∫∫ ω² dA
        // For Taylor-Green: Ω₀ = 2k²E₀/ρ
        float omega0 = 2.0f * k * k * E0 / rho;
        return omega0 * std::exp(-4.0f * nu * k * k * t);
    }

    /**
     * @brief Compute viscous time scale (decay time)
     * @return τ = L²/(2π²ν) [s]
     */
    float viscousTimeScale() const {
        return L * L / (2.0f * M_PI * M_PI * nu);
    }

    /**
     * @brief Compute Reynolds number
     * @return Re = U₀L/ν
     */
    float reynoldsNumber() const {
        return U0 * L / nu;
    }

    /**
     * @brief Compute energy decay rate
     * @param t Time [s]
     * @return dE/dt = -4νk²E(t) [W/m³]
     */
    float energyDecayRate(float t) const {
        return -4.0f * nu * k * k * kineticEnergy(t);
    }
};

} // namespace analytical
