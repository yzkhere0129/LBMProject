/**
 * @file momentum_exchange_force.h
 * @brief Momentum-Exchange-Method (MEM) force on immersed obstacles
 *
 * For halfway bounce-back, the momentum transferred per LBM time step from
 * the fluid to the solid across one fluid-solid link is
 *
 *   ΔP_link  =  (f_q^pre + f_{opp(q)}^post) · c_q   [LU]
 *
 * where f_q^pre is the population the fluid cell would have pushed toward
 * the solid in direction q, and f_{opp}^post is the population it receives
 * back after the bounce. With pure halfway BB on a stationary wall, these
 * are equal, so
 *
 *   F_LU  =  Σ_(fluid, q : neighbour solid)  2 · f_q^pre · c_q
 *
 * Conversion to physical units: F_phys = F_LU · ρ_phys · dx⁴ / dt²
 *
 * Reference:
 *   Ladd, A. J. C. (1994). "Numerical simulations of particulate suspensions
 *   via a discretized Boltzmann equation. Part 1." J. Fluid Mech. 271:285.
 *   Mei, Yu, Shyy & Luo (2002). Phys. Rev. E 65:041203 (correction).
 */

#pragma once

#include <cuda_runtime.h>

namespace lbm {
namespace physics {
namespace aero {

/**
 * @brief Compute MEM force on solid in lattice units.
 *
 * Generalised momentum-exchange formula valid for both halfway BB and BFL:
 *
 *   F_link = c_q · (f_in + f_out)
 *
 * where f_in = f^pre(X, q) is the post-collision population about to push
 * into the solid, and f_out is the population that arrives back at X in
 * direction opposite[q] after the bounce. With halfway BB f_out = f_in
 * and the formula reduces to 2·c_q·f_in (Ladd 1994). With BFL using a
 * per-link q-fraction, f_out is computed via the BFL interpolation rule
 * (using f_in, f^pre(X, opp[q]), and possibly the upstream fluid neighbour).
 *
 * @param d_f         Post-collision pre-streaming distribution field
 *                    (FluidLBM::getDistributionSrc() AFTER collisionTRT/BGK,
 *                    BEFORE streaming).
 * @param d_solid_mask Per-cell solid mask.
 * @param d_qfrac     Per-link Bouzidi q-fraction (nullptr → halfway BB
 *                    formula on every link).
 * @param nx,ny,nz    Domain extents.
 * @param periodic_x,y,z 1 if face is periodic.
 * @param Fx,Fy,Fz    Output in lattice units.
 *
 * Reference: Mei R., Yu D., Shyy W. & Luo L.S. (2002),
 *   "Force evaluation in the lattice Boltzmann method involving curved
 *    geometry", Phys. Rev. E 65:041203.
 */
void computeObstacleForceLU(
    const float* d_f,
    const unsigned char* d_solid_mask,
    const float* d_qfrac,
    int nx, int ny, int nz,
    int periodic_x, int periodic_y, int periodic_z,
    float& Fx, float& Fy, float& Fz);

/**
 * @brief MEM force variant for single-node second-order QBB.
 *
 * Same general formula F_link = c_q · (f_in + f_out_new), but f_out_new is
 * computed via the lbmpy single-node QBB rule (using local equilibrium).
 * Required when streaming uses fluidStreamingKernelWithSingleNodeQBB so the
 * force evaluation stays consistent with the BC.
 */
void computeObstacleForceLU_SingleNodeQBB(
    const float* d_f,
    const unsigned char* d_solid_mask,
    const float* d_qfrac,
    float omega,
    int nx, int ny, int nz,
    int periodic_x, int periodic_y, int periodic_z,
    float& Fx, float& Fy, float& Fz);

/**
 * @brief MEM force variant for 3-cell quadratic Bouzidi.
 *
 * Mirrors fluidStreamingKernelWithQuadBouzidi. F_link = c_q · (f_in + f_out_new)
 * where f_out_new is computed via the BFL 2001 Eq. 12 quadratic formula
 * (with defensive fallback to BFL linear when upstream stencil missing).
 */
void computeObstacleForceLU_QuadBouzidi(
    const float* d_f,
    const unsigned char* d_solid_mask,
    const float* d_qfrac,
    int nx, int ny, int nz,
    int periodic_x, int periodic_y, int periodic_z,
    float& Fx, float& Fy, float& Fz);

} // namespace aero
} // namespace physics
} // namespace lbm
