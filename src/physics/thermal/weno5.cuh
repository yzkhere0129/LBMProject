/**
 * @file weno5.cuh
 * @brief WENO5 (5th-order Weighted Essentially Non-Oscillatory) reconstruction
 *
 * Computes the face value T_{i+1/2} from a 5-point stencil using three
 * candidate parabolic reconstructions with nonlinear smoothness-based weights.
 *
 * For u > 0 (left-biased), the stencil is {i-2, i-1, i, i+1, i+2}
 * and the three candidates are:
 *   q0 = (2·v[0] - 7·v[1] + 11·v[2]) / 6     (most upwind)
 *   q1 = (-v[1] + 5·v[2] + 2·v[3]) / 6        (central)
 *   q2 = (2·v[2] + 5·v[3] - v[4]) / 6          (most downwind)
 *
 * Ideal (linear) weights: d0=0.1, d1=0.6, d2=0.3
 * Smoothness indicators: β0, β1, β2 (measures oscillation in each stencil)
 * Nonlinear weights: ω_k = α_k / Σα_k, where α_k = d_k / (ε + β_k)²
 *
 * Reference: Jiang & Shu (1996), J. Comput. Phys. 126:202-228
 */

#pragma once
#include <cuda_runtime.h>

/**
 * @brief WENO5 left-biased reconstruction at face i+1/2
 *
 * @param v0 T[i-2]
 * @param v1 T[i-1]
 * @param v2 T[i]
 * @param v3 T[i+1]
 * @param v4 T[i+2]
 * @return Reconstructed T at the i+1/2 face
 */
__device__ __forceinline__ float weno5_left(
    float v0, float v1, float v2, float v3, float v4)
{
    constexpr float eps = 1e-6f;

    // Three candidate fluxes (3rd-order each)
    float q0 = ( 2.0f*v0 - 7.0f*v1 + 11.0f*v2) / 6.0f;
    float q1 = (    -v1  + 5.0f*v2 +  2.0f*v3) / 6.0f;
    float q2 = ( 2.0f*v2 + 5.0f*v3 -       v4) / 6.0f;

    // Smoothness indicators (Jiang-Shu)
    float b0 = (13.0f/12.0f)*(v0 - 2.0f*v1 + v2)*(v0 - 2.0f*v1 + v2)
             + (1.0f/4.0f)*(v0 - 4.0f*v1 + 3.0f*v2)*(v0 - 4.0f*v1 + 3.0f*v2);
    float b1 = (13.0f/12.0f)*(v1 - 2.0f*v2 + v3)*(v1 - 2.0f*v2 + v3)
             + (1.0f/4.0f)*(v1 - v3)*(v1 - v3);
    float b2 = (13.0f/12.0f)*(v2 - 2.0f*v3 + v4)*(v2 - 2.0f*v3 + v4)
             + (1.0f/4.0f)*(3.0f*v2 - 4.0f*v3 + v4)*(3.0f*v2 - 4.0f*v3 + v4);

    // Nonlinear weights (ideal: d0=0.1, d1=0.6, d2=0.3)
    float a0 = 0.1f / ((eps + b0) * (eps + b0));
    float a1 = 0.6f / ((eps + b1) * (eps + b1));
    float a2 = 0.3f / ((eps + b2) * (eps + b2));
    float a_sum = a0 + a1 + a2;

    float w0 = a0 / a_sum;
    float w1 = a1 / a_sum;
    float w2 = a2 / a_sum;

    return w0*q0 + w1*q1 + w2*q2;
}

/**
 * @brief WENO5 right-biased reconstruction at face i+1/2
 *
 * Mirror of weno5_left: stencil {i-1, i, i+1, i+2, i+3}
 * Used for u < 0 (information travels right-to-left).
 */
__device__ __forceinline__ float weno5_right(
    float v0, float v1, float v2, float v3, float v4)
{
    constexpr float eps = 1e-6f;

    // Candidates (mirror of left)
    float q0 = (-v0 + 5.0f*v1 + 2.0f*v2) / 6.0f;
    float q1 = (2.0f*v1 + 5.0f*v2 - v3) / 6.0f;
    float q2 = (11.0f*v2 - 7.0f*v3 + 2.0f*v4) / 6.0f;

    // Smoothness indicators (mirror)
    float b0 = (13.0f/12.0f)*(v0 - 2.0f*v1 + v2)*(v0 - 2.0f*v1 + v2)
             + (1.0f/4.0f)*(v0 - 4.0f*v1 + 3.0f*v2)*(v0 - 4.0f*v1 + 3.0f*v2);
    float b1 = (13.0f/12.0f)*(v1 - 2.0f*v2 + v3)*(v1 - 2.0f*v2 + v3)
             + (1.0f/4.0f)*(v1 - v3)*(v1 - v3);
    float b2 = (13.0f/12.0f)*(v2 - 2.0f*v3 + v4)*(v2 - 2.0f*v3 + v4)
             + (1.0f/4.0f)*(3.0f*v2 - 4.0f*v3 + v4)*(3.0f*v2 - 4.0f*v3 + v4);

    // Nonlinear weights (ideal: d0=0.3, d1=0.6, d2=0.1 — mirror)
    float a0 = 0.3f / ((eps + b0) * (eps + b0));
    float a1 = 0.6f / ((eps + b1) * (eps + b1));
    float a2 = 0.1f / ((eps + b2) * (eps + b2));
    float a_sum = a0 + a1 + a2;

    float w0 = a0 / a_sum;
    float w1 = a1 / a_sum;
    float w2 = a2 / a_sum;

    return w0*q0 + w1*q1 + w2*q2;
}
