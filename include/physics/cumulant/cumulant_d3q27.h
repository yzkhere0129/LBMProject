/**
 * @file cumulant_d3q27.h
 * @brief D3Q27 Cumulant LBM collision (Geier-Schoenherr-Pasquali 2017)
 *
 * Pipeline:
 *   f → m_pqr (raw moments)
 *   m → κ_pqr (central moments around u)
 *   κ → K_pqr (cumulants; subtractive correction for order ≥ 4)
 *   K → K_post (relax with per-order ω, equilibrium = MB cumulants)
 *   K_post → κ_post → m_post → f_post (inverse)
 *
 * Index convention: m[(p,q,r)] = m[p + 3*q + 9*r]  for p,q,r ∈ {0,1,2}.
 *
 * Per-axis 1D inverse moment matrix (M^{-1}_1D):
 *     for c ∈ {-1, 0, +1}, p ∈ {0,1,2}:
 *
 *     c\p   0    1    2
 *     -1   0  -1/2  1/2
 *      0   1   0   -1
 *     +1   0  +1/2  1/2
 *
 * Then f_q = Σ_(p,qq,r) M^{-1}_1D[c_x; p] · M^{-1}_1D[c_y; qq] · M^{-1}_1D[c_z; r] · m_(p,qq,r).
 *
 * Reference:
 *   Geier M., Schoenherr M., Pasquali A. (2017) "The cumulant lattice
 *   Boltzmann equation in three dimensions." Comput. Math. Appl. 70:507-547.
 */

#pragma once

#include <cuda_runtime.h>
#include "core/lattice_d3q27.h"

namespace lbm {
namespace physics {
namespace cumulant {

#ifdef __CUDA_ARCH__
#define LBM_CUM_EX(q) ::lbm::core::ex27[q]
#define LBM_CUM_EY(q) ::lbm::core::ey27[q]
#define LBM_CUM_EZ(q) ::lbm::core::ez27[q]
#else
#define LBM_CUM_EX(q) ::lbm::core::D3Q27::h_ex[q]
#define LBM_CUM_EY(q) ::lbm::core::D3Q27::h_ey[q]
#define LBM_CUM_EZ(q) ::lbm::core::D3Q27::h_ez[q]
#endif

#define MIDX(p, q, r) ((p) + 3 * (q) + 9 * (r))

// ===========================================================================
// 1D inverse moment factor: returns M^{-1}_1D[c; p].
// c ∈ {-1, 0, +1}, p ∈ {0, 1, 2}.
// Used to factorize the 3D inverse f_q = Σ Π M^{-1}_1D[c_x][p] · m_pqr.
// ===========================================================================
__host__ __device__ inline float invMoment1D(int c, int p) {
    // Row c=-1: { 0,  -0.5,   0.5}
    // Row c= 0: { 1,   0  ,  -1.0}
    // Row c=+1: { 0,   0.5,   0.5}
    if (c == -1) {
        if (p == 0) return 0.0f;
        if (p == 1) return -0.5f;
        return 0.5f;  // p == 2
    } else if (c == 0) {
        if (p == 0) return 1.0f;
        if (p == 1) return 0.0f;
        return -1.0f;  // p == 2
    } else {  // c == +1
        if (p == 0) return 0.0f;
        if (p == 1) return 0.5f;
        return 0.5f;  // p == 2
    }
}

// ===========================================================================
// Stage 2a: Raw moments  m_pqr = Σ_q c_x^p c_y^q c_z^r f_q
// ===========================================================================
__host__ __device__ inline void computeRawMoments27(
    const float* f, float* m)
{
    #pragma unroll
    for (int i = 0; i < 27; ++i) m[i] = 0.0f;

    #pragma unroll
    for (int q = 0; q < 27; ++q) {
        const float fq = f[q];
        const int cx = LBM_CUM_EX(q);
        const int cy = LBM_CUM_EY(q);
        const int cz = LBM_CUM_EZ(q);
        const int cx_pow[3] = {1, cx, cx * cx};
        const int cy_pow[3] = {1, cy, cy * cy};
        const int cz_pow[3] = {1, cz, cz * cz};
        #pragma unroll
        for (int r = 0; r < 3; ++r)
        #pragma unroll
        for (int qq = 0; qq < 3; ++qq)
        #pragma unroll
        for (int p = 0; p < 3; ++p) {
            m[MIDX(p, qq, r)] += cx_pow[p] * cy_pow[qq] * cz_pow[r] * fq;
        }
    }
}

// ===========================================================================
// Stage 2b: Central moment shift via binomial:
//   κ_pqr = Σ_(i≤p,j≤q,k≤r) C(p,i) C(q,j) C(r,k) (-u_x)^(p-i) (-u_y)^(q-j) (-u_z)^(r-k) m_(i,j,k)
// ===========================================================================
__host__ __device__ inline void shiftToCentralMoments27(
    const float* m_in, float ux, float uy, float uz, float* kappa)
{
    static constexpr int C_TBL[3][3] = {{1, 0, 0}, {1, 1, 0}, {1, 2, 1}};
    const float negux = -ux, neguy = -uy, neguz = -uz;
    const float px[3] = {1.0f, negux, negux * negux};
    const float py[3] = {1.0f, neguy, neguy * neguy};
    const float pz[3] = {1.0f, neguz, neguz * neguz};

    #pragma unroll
    for (int r = 0; r < 3; ++r)
    #pragma unroll
    for (int qq = 0; qq < 3; ++qq)
    #pragma unroll
    for (int p = 0; p < 3; ++p) {
        float val = 0.0f;
        for (int k = 0; k <= r; ++k)
        for (int j = 0; j <= qq; ++j)
        for (int i = 0; i <= p; ++i) {
            val += C_TBL[p][i] * C_TBL[qq][j] * C_TBL[r][k]
                 * px[p - i] * py[qq - j] * pz[r - k]
                 * m_in[MIDX(i, j, k)];
        }
        kappa[MIDX(p, qq, r)] = val;
    }
}

// ===========================================================================
// Stage 2e Part 2: Inverse central → raw moment shift
//   m_pqr = Σ C(p,i) C(q,j) C(r,k) (u_x)^(p-i) (u_y)^(q-j) (u_z)^(r-k) κ_(i,j,k)
// ===========================================================================
__host__ __device__ inline void centralToRawMoments27(
    const float* kappa, float ux, float uy, float uz, float* m_out)
{
    static constexpr int C_TBL[3][3] = {{1, 0, 0}, {1, 1, 0}, {1, 2, 1}};
    const float px[3] = {1.0f, ux, ux * ux};
    const float py[3] = {1.0f, uy, uy * uy};
    const float pz[3] = {1.0f, uz, uz * uz};

    #pragma unroll
    for (int r = 0; r < 3; ++r)
    #pragma unroll
    for (int qq = 0; qq < 3; ++qq)
    #pragma unroll
    for (int p = 0; p < 3; ++p) {
        float val = 0.0f;
        for (int k = 0; k <= r; ++k)
        for (int j = 0; j <= qq; ++j)
        for (int i = 0; i <= p; ++i) {
            val += C_TBL[p][i] * C_TBL[qq][j] * C_TBL[r][k]
                 * px[p - i] * py[qq - j] * pz[r - k]
                 * kappa[MIDX(i, j, k)];
        }
        m_out[MIDX(p, qq, r)] = val;
    }
}

// ===========================================================================
// Stage 2e Part 3: Inverse raw moments → PDFs via tensor product M^{-1}_1D
// ===========================================================================
__host__ __device__ inline void rawMomentsToPDFs27(
    const float* m, float* f)
{
    #pragma unroll
    for (int q = 0; q < 27; ++q) {
        const int cx = LBM_CUM_EX(q);
        const int cy = LBM_CUM_EY(q);
        const int cz = LBM_CUM_EZ(q);
        // Per-axis precompute the 3 inverse-moment factors
        const float ax[3] = {invMoment1D(cx, 0), invMoment1D(cx, 1), invMoment1D(cx, 2)};
        const float ay[3] = {invMoment1D(cy, 0), invMoment1D(cy, 1), invMoment1D(cy, 2)};
        const float az[3] = {invMoment1D(cz, 0), invMoment1D(cz, 1), invMoment1D(cz, 2)};
        float val = 0.0f;
        #pragma unroll
        for (int r = 0; r < 3; ++r)
        #pragma unroll
        for (int qq = 0; qq < 3; ++qq)
        #pragma unroll
        for (int p = 0; p < 3; ++p) {
            val += ax[p] * ay[qq] * az[r] * m[MIDX(p, qq, r)];
        }
        f[q] = val;
    }
}

// ===========================================================================
// Stage 2c: Central moments → cumulants
// For order ≤ 3 in CENTRAL frame: K_pqr = κ_pqr (since κ_1xx = 0)
// For order ≥ 4: subtractive corrections following Geier-Schoenherr-Pasquali 2017
// ===========================================================================
__host__ __device__ inline void centralToCumulants27(
    const float* k, float rho, float* K)
{
    const float inv_rho = 1.0f / rho;

    // Order 0 / 1
    K[MIDX(0, 0, 0)] = k[MIDX(0, 0, 0)];
    K[MIDX(1, 0, 0)] = 0.0f;  // κ_100 in central frame = 0
    K[MIDX(0, 1, 0)] = 0.0f;
    K[MIDX(0, 0, 1)] = 0.0f;

    // Order 2: K = κ
    K[MIDX(2, 0, 0)] = k[MIDX(2, 0, 0)];
    K[MIDX(0, 2, 0)] = k[MIDX(0, 2, 0)];
    K[MIDX(0, 0, 2)] = k[MIDX(0, 0, 2)];
    K[MIDX(1, 1, 0)] = k[MIDX(1, 1, 0)];
    K[MIDX(1, 0, 1)] = k[MIDX(1, 0, 1)];
    K[MIDX(0, 1, 1)] = k[MIDX(0, 1, 1)];

    // Order 3: K = κ
    K[MIDX(2, 1, 0)] = k[MIDX(2, 1, 0)];
    K[MIDX(2, 0, 1)] = k[MIDX(2, 0, 1)];
    K[MIDX(1, 2, 0)] = k[MIDX(1, 2, 0)];
    K[MIDX(0, 2, 1)] = k[MIDX(0, 2, 1)];
    K[MIDX(1, 0, 2)] = k[MIDX(1, 0, 2)];
    K[MIDX(0, 1, 2)] = k[MIDX(0, 1, 2)];
    K[MIDX(1, 1, 1)] = k[MIDX(1, 1, 1)];

    // Order 4: subtract sum of products over partitions of 2+2.
    // For K_(2,2,0): partition (xx)(yy) and (xy)(xy) → 1·κ_200·κ_020 + 2·κ_110²
    K[MIDX(2, 2, 0)] = k[MIDX(2, 2, 0)] -
                       (k[MIDX(2, 0, 0)] * k[MIDX(0, 2, 0)] +
                        2.0f * k[MIDX(1, 1, 0)] * k[MIDX(1, 1, 0)]) * inv_rho;
    K[MIDX(2, 0, 2)] = k[MIDX(2, 0, 2)] -
                       (k[MIDX(2, 0, 0)] * k[MIDX(0, 0, 2)] +
                        2.0f * k[MIDX(1, 0, 1)] * k[MIDX(1, 0, 1)]) * inv_rho;
    K[MIDX(0, 2, 2)] = k[MIDX(0, 2, 2)] -
                       (k[MIDX(0, 2, 0)] * k[MIDX(0, 0, 2)] +
                        2.0f * k[MIDX(0, 1, 1)] * k[MIDX(0, 1, 1)]) * inv_rho;
    // For K_(2,1,1): partition (xx)(yz) + 2·(xy)(xz)
    K[MIDX(2, 1, 1)] = k[MIDX(2, 1, 1)] -
                       (k[MIDX(2, 0, 0)] * k[MIDX(0, 1, 1)] +
                        2.0f * k[MIDX(1, 1, 0)] * k[MIDX(1, 0, 1)]) * inv_rho;
    K[MIDX(1, 2, 1)] = k[MIDX(1, 2, 1)] -
                       (k[MIDX(0, 2, 0)] * k[MIDX(1, 0, 1)] +
                        2.0f * k[MIDX(1, 1, 0)] * k[MIDX(0, 1, 1)]) * inv_rho;
    K[MIDX(1, 1, 2)] = k[MIDX(1, 1, 2)] -
                       (k[MIDX(0, 0, 2)] * k[MIDX(1, 1, 0)] +
                        2.0f * k[MIDX(1, 0, 1)] * k[MIDX(0, 1, 1)]) * inv_rho;

    // Order 5: K_(2,2,1) etc. Use K=κ for order ≤ 3 in subtractive terms.
    K[MIDX(2, 2, 1)] = k[MIDX(2, 2, 1)] - (
        K[MIDX(2, 0, 0)] * K[MIDX(0, 2, 1)] +
        K[MIDX(0, 2, 0)] * K[MIDX(2, 0, 1)] +
        4.0f * K[MIDX(1, 1, 0)] * K[MIDX(1, 1, 1)] +
        2.0f * K[MIDX(1, 0, 1)] * K[MIDX(1, 2, 0)] +
        2.0f * K[MIDX(0, 1, 1)] * K[MIDX(2, 1, 0)]
    ) * inv_rho;
    K[MIDX(2, 1, 2)] = k[MIDX(2, 1, 2)] - (
        K[MIDX(2, 0, 0)] * K[MIDX(0, 1, 2)] +
        K[MIDX(0, 0, 2)] * K[MIDX(2, 1, 0)] +
        4.0f * K[MIDX(1, 0, 1)] * K[MIDX(1, 1, 1)] +
        2.0f * K[MIDX(1, 1, 0)] * K[MIDX(1, 0, 2)] +
        2.0f * K[MIDX(0, 1, 1)] * K[MIDX(2, 0, 1)]
    ) * inv_rho;
    K[MIDX(1, 2, 2)] = k[MIDX(1, 2, 2)] - (
        K[MIDX(0, 2, 0)] * K[MIDX(1, 0, 2)] +
        K[MIDX(0, 0, 2)] * K[MIDX(1, 2, 0)] +
        4.0f * K[MIDX(0, 1, 1)] * K[MIDX(1, 1, 1)] +
        2.0f * K[MIDX(1, 1, 0)] * K[MIDX(0, 1, 2)] +
        2.0f * K[MIDX(1, 0, 1)] * K[MIDX(0, 2, 1)]
    ) * inv_rho;

    // Order 6: K_(2,2,2)
    {
        const float t1 = K[MIDX(2, 0, 0)] * k[MIDX(0, 2, 2)] +
                         K[MIDX(0, 2, 0)] * k[MIDX(2, 0, 2)] +
                         K[MIDX(0, 0, 2)] * k[MIDX(2, 2, 0)];
        const float t2 = 8.0f * K[MIDX(1, 1, 1)] * K[MIDX(1, 1, 1)];
        const float t3 = 4.0f * (K[MIDX(1, 1, 0)] * k[MIDX(1, 1, 2)] +
                                 K[MIDX(1, 0, 1)] * k[MIDX(1, 2, 1)] +
                                 K[MIDX(0, 1, 1)] * k[MIDX(2, 1, 1)]);
        const float t4 = 2.0f * (K[MIDX(2, 0, 0)] * K[MIDX(0, 2, 0)] * K[MIDX(0, 0, 2)] +
                                 2.0f * K[MIDX(1, 1, 0)] * K[MIDX(1, 0, 1)] * K[MIDX(0, 1, 1)]);
        K[MIDX(2, 2, 2)] = k[MIDX(2, 2, 2)] - (t1 + t2 + t3) * inv_rho + t4 * inv_rho * inv_rho;
    }
}

// ===========================================================================
// Stage 2d: Relax cumulants toward Maxwell-Boltzmann equilibrium
//   K_eq:  K_000 = ρ; K_diag_2nd = ρ/3; everything else = 0
// ===========================================================================
__host__ __device__ inline void relaxCumulants27(
    float* K, float rho,
    float omega_nu, float omega_b,
    float omega_3, float omega_4, float omega_5, float omega_6)
{
    const float K_diag_eq = rho / 3.0f;

    // 2nd order: split trace (bulk) and traceless (shear)
    {
        const float K_xx = K[MIDX(2, 0, 0)];
        const float K_yy = K[MIDX(0, 2, 0)];
        const float K_zz = K[MIDX(0, 0, 2)];
        const float trace = K_xx + K_yy + K_zz;
        const float trace_eq = 3.0f * K_diag_eq;
        const float trace_post = trace - omega_b * (trace - trace_eq);
        const float dxx = K_xx - trace / 3.0f;
        const float dyy = K_yy - trace / 3.0f;
        const float dzz = K_zz - trace / 3.0f;
        const float dxx_post = dxx - omega_nu * dxx;
        const float dyy_post = dyy - omega_nu * dyy;
        const float dzz_post = dzz - omega_nu * dzz;
        K[MIDX(2, 0, 0)] = dxx_post + trace_post / 3.0f;
        K[MIDX(0, 2, 0)] = dyy_post + trace_post / 3.0f;
        K[MIDX(0, 0, 2)] = dzz_post + trace_post / 3.0f;
    }
    K[MIDX(1, 1, 0)] -= omega_nu * K[MIDX(1, 1, 0)];
    K[MIDX(1, 0, 1)] -= omega_nu * K[MIDX(1, 0, 1)];
    K[MIDX(0, 1, 1)] -= omega_nu * K[MIDX(0, 1, 1)];

    // 3rd order
    K[MIDX(2, 1, 0)] -= omega_3 * K[MIDX(2, 1, 0)];
    K[MIDX(2, 0, 1)] -= omega_3 * K[MIDX(2, 0, 1)];
    K[MIDX(1, 2, 0)] -= omega_3 * K[MIDX(1, 2, 0)];
    K[MIDX(0, 2, 1)] -= omega_3 * K[MIDX(0, 2, 1)];
    K[MIDX(1, 0, 2)] -= omega_3 * K[MIDX(1, 0, 2)];
    K[MIDX(0, 1, 2)] -= omega_3 * K[MIDX(0, 1, 2)];
    K[MIDX(1, 1, 1)] -= omega_3 * K[MIDX(1, 1, 1)];

    // 4th order
    K[MIDX(2, 2, 0)] -= omega_4 * K[MIDX(2, 2, 0)];
    K[MIDX(2, 0, 2)] -= omega_4 * K[MIDX(2, 0, 2)];
    K[MIDX(0, 2, 2)] -= omega_4 * K[MIDX(0, 2, 2)];
    K[MIDX(2, 1, 1)] -= omega_4 * K[MIDX(2, 1, 1)];
    K[MIDX(1, 2, 1)] -= omega_4 * K[MIDX(1, 2, 1)];
    K[MIDX(1, 1, 2)] -= omega_4 * K[MIDX(1, 1, 2)];

    // 5th order
    K[MIDX(2, 2, 1)] -= omega_5 * K[MIDX(2, 2, 1)];
    K[MIDX(2, 1, 2)] -= omega_5 * K[MIDX(2, 1, 2)];
    K[MIDX(1, 2, 2)] -= omega_5 * K[MIDX(1, 2, 2)];

    // 6th order
    K[MIDX(2, 2, 2)] -= omega_6 * K[MIDX(2, 2, 2)];
}

// ===========================================================================
// Stage 2e Part 1: Cumulants → central moments (inverse of Stage 2c)
// ===========================================================================
__host__ __device__ inline void cumulantsToCentralMoments27(
    const float* K, float rho, float* k)
{
    const float inv_rho = 1.0f / rho;

    // Order 0..3: identity
    for (int i = 0; i < 27; ++i) k[i] = 0.0f;
    k[MIDX(0, 0, 0)] = K[MIDX(0, 0, 0)];
    // 1st-order central moments are ALWAYS 0 (definition of central frame).
    k[MIDX(1, 0, 0)] = 0.0f;
    k[MIDX(0, 1, 0)] = 0.0f;
    k[MIDX(0, 0, 1)] = 0.0f;
    k[MIDX(2, 0, 0)] = K[MIDX(2, 0, 0)];
    k[MIDX(0, 2, 0)] = K[MIDX(0, 2, 0)];
    k[MIDX(0, 0, 2)] = K[MIDX(0, 0, 2)];
    k[MIDX(1, 1, 0)] = K[MIDX(1, 1, 0)];
    k[MIDX(1, 0, 1)] = K[MIDX(1, 0, 1)];
    k[MIDX(0, 1, 1)] = K[MIDX(0, 1, 1)];

    k[MIDX(2, 1, 0)] = K[MIDX(2, 1, 0)];
    k[MIDX(2, 0, 1)] = K[MIDX(2, 0, 1)];
    k[MIDX(1, 2, 0)] = K[MIDX(1, 2, 0)];
    k[MIDX(0, 2, 1)] = K[MIDX(0, 2, 1)];
    k[MIDX(1, 0, 2)] = K[MIDX(1, 0, 2)];
    k[MIDX(0, 1, 2)] = K[MIDX(0, 1, 2)];
    k[MIDX(1, 1, 1)] = K[MIDX(1, 1, 1)];

    // 4th order: invert
    k[MIDX(2, 2, 0)] = K[MIDX(2, 2, 0)] +
                       (K[MIDX(2, 0, 0)] * K[MIDX(0, 2, 0)] +
                        2.0f * K[MIDX(1, 1, 0)] * K[MIDX(1, 1, 0)]) * inv_rho;
    k[MIDX(2, 0, 2)] = K[MIDX(2, 0, 2)] +
                       (K[MIDX(2, 0, 0)] * K[MIDX(0, 0, 2)] +
                        2.0f * K[MIDX(1, 0, 1)] * K[MIDX(1, 0, 1)]) * inv_rho;
    k[MIDX(0, 2, 2)] = K[MIDX(0, 2, 2)] +
                       (K[MIDX(0, 2, 0)] * K[MIDX(0, 0, 2)] +
                        2.0f * K[MIDX(0, 1, 1)] * K[MIDX(0, 1, 1)]) * inv_rho;
    k[MIDX(2, 1, 1)] = K[MIDX(2, 1, 1)] +
                       (K[MIDX(2, 0, 0)] * K[MIDX(0, 1, 1)] +
                        2.0f * K[MIDX(1, 1, 0)] * K[MIDX(1, 0, 1)]) * inv_rho;
    k[MIDX(1, 2, 1)] = K[MIDX(1, 2, 1)] +
                       (K[MIDX(0, 2, 0)] * K[MIDX(1, 0, 1)] +
                        2.0f * K[MIDX(1, 1, 0)] * K[MIDX(0, 1, 1)]) * inv_rho;
    k[MIDX(1, 1, 2)] = K[MIDX(1, 1, 2)] +
                       (K[MIDX(0, 0, 2)] * K[MIDX(1, 1, 0)] +
                        2.0f * K[MIDX(1, 0, 1)] * K[MIDX(0, 1, 1)]) * inv_rho;

    // 5th order
    k[MIDX(2, 2, 1)] = K[MIDX(2, 2, 1)] + (
        K[MIDX(2, 0, 0)] * k[MIDX(0, 2, 1)] +
        K[MIDX(0, 2, 0)] * k[MIDX(2, 0, 1)] +
        4.0f * K[MIDX(1, 1, 0)] * K[MIDX(1, 1, 1)] +
        2.0f * K[MIDX(1, 0, 1)] * K[MIDX(1, 2, 0)] +
        2.0f * K[MIDX(0, 1, 1)] * K[MIDX(2, 1, 0)]
    ) * inv_rho;
    k[MIDX(2, 1, 2)] = K[MIDX(2, 1, 2)] + (
        K[MIDX(2, 0, 0)] * k[MIDX(0, 1, 2)] +
        K[MIDX(0, 0, 2)] * k[MIDX(2, 1, 0)] +
        4.0f * K[MIDX(1, 0, 1)] * K[MIDX(1, 1, 1)] +
        2.0f * K[MIDX(1, 1, 0)] * K[MIDX(1, 0, 2)] +
        2.0f * K[MIDX(0, 1, 1)] * K[MIDX(2, 0, 1)]
    ) * inv_rho;
    k[MIDX(1, 2, 2)] = K[MIDX(1, 2, 2)] + (
        K[MIDX(0, 2, 0)] * k[MIDX(1, 0, 2)] +
        K[MIDX(0, 0, 2)] * k[MIDX(1, 2, 0)] +
        4.0f * K[MIDX(0, 1, 1)] * K[MIDX(1, 1, 1)] +
        2.0f * K[MIDX(1, 1, 0)] * K[MIDX(0, 1, 2)] +
        2.0f * K[MIDX(1, 0, 1)] * K[MIDX(0, 2, 1)]
    ) * inv_rho;

    // 6th order
    {
        const float t1 = K[MIDX(2, 0, 0)] * k[MIDX(0, 2, 2)] +
                         K[MIDX(0, 2, 0)] * k[MIDX(2, 0, 2)] +
                         K[MIDX(0, 0, 2)] * k[MIDX(2, 2, 0)];
        const float t2 = 8.0f * K[MIDX(1, 1, 1)] * K[MIDX(1, 1, 1)];
        const float t3 = 4.0f * (K[MIDX(1, 1, 0)] * k[MIDX(1, 1, 2)] +
                                 K[MIDX(1, 0, 1)] * k[MIDX(1, 2, 1)] +
                                 K[MIDX(0, 1, 1)] * k[MIDX(2, 1, 1)]);
        const float t4 = 2.0f * (K[MIDX(2, 0, 0)] * K[MIDX(0, 2, 0)] * K[MIDX(0, 0, 2)] +
                                 2.0f * K[MIDX(1, 1, 0)] * K[MIDX(1, 0, 1)] * K[MIDX(0, 1, 1)]);
        k[MIDX(2, 2, 2)] = K[MIDX(2, 2, 2)] + (t1 + t2 + t3) * inv_rho - t4 * inv_rho * inv_rho;
    }
}

// ===========================================================================
// Combined collision (Stage 2f). Reads f from f_src, writes f_dst.
// ===========================================================================
__global__ void fluidCumulantCollisionKernel(
    const float* f_src, float* f_dst,
    float* rho_out, float* ux_out, float* uy_out, float* uz_out,
    int nx, int ny, int nz,
    float omega_nu, float omega_b,
    float omega_3, float omega_4, float omega_5, float omega_6);

#undef MIDX
#undef LBM_CUM_EX
#undef LBM_CUM_EY
#undef LBM_CUM_EZ

} // namespace cumulant
} // namespace physics
} // namespace lbm
