/**
 * @file interface_geometry.h
 * @brief PLIC interface geometry view for cross-module consumption
 *
 * This header defines the data contract between VOFSolver (which reconstructs
 * the PLIC plane) and downstream physics modules (laser, CSF, Marangoni,
 * recoil, evaporation) that need sharp-interface geometry instead of the
 * smeared `|∇f|` kernel-based delta.
 *
 * Architectural decision (Phase 1 of PLIC upgrade roadmap):
 *
 *   - VOFSolver owns the per-cell PLIC arrays (plic_nx, plic_ny, plic_nz,
 *     plic_alpha) and a `plic_dirty_` flag.
 *   - VOFSolver exposes them as `InterfaceGeometryView`, a plain device-pointer
 *     bundle. Downstream modules consume the view as a single kernel argument.
 *   - Physics modules (force_accumulator, multiphysics) DO NOT include
 *     `vof_solver.h` — they only need this header. This preserves the L2/L3
 *     module boundary (modules are "leaf" in the layering; VOFSolver sits at L3).
 *   - When `plic_ready == false`, downstream kernels MUST fall back to the
 *     legacy `|∇f|`-based path. This preserves "独立插拔": physics modules
 *     remain testable in isolation with `plic_ready=false`.
 *
 * Storage convention (CRITICAL — different from textbook PLIC):
 *
 *   The plane equation in the cell-corner unit-cube frame [0,1]^3 is:
 *
 *       n_x · X + n_y · Y + n_z · Z = alpha_signed
 *
 *   where (X, Y, Z) ∈ [0, 1]^3 are local coordinates with the cell CORNER at
 *   the origin (NOT the cell center), and (n_x, n_y, n_z) is a unit vector
 *   stored with its true sign (can be negative). `alpha_signed` is the signed
 *   plane offset such that:
 *
 *       Vol({(X,Y,Z) ∈ [0,1]^3 : n_x·X + n_y·Y + n_z·Z < alpha_signed}) = f
 *
 *   This is the "below-the-plane" volume in the original signed coordinate
 *   frame and is consumed by `plicVolumeInBox(alpha_signed, n_x, n_y, n_z,
 *   Lx, Ly, Lz)` directly — see vof_solver.cu.
 *
 *   To convert to a more physically intuitive "signed distance from cell
 *   center" convention, use `plicSignedDistanceFromCenter()` below.
 *
 * @section UnitConventions Unit conventions
 *
 *   - Normals (n_x, n_y, n_z) are dimensionless unit vectors.
 *   - alpha_signed is dimensionless (in units of cell side dx).
 *   - To get physical signed distance, multiply by dx.
 *
 * @section PhaseAvailability Phase availability
 *
 *   - Phase 1 (this header): defines the contract. Only VOFSolver provides it.
 *     Consumers can read but few do — Phase 2/3/4 of the roadmap migrate
 *     laser, CSF, Marangoni, recoil, evap to the new contract.
 */

#pragma once

#include <cuda_runtime.h>

namespace lbm {
namespace physics {

/**
 * @brief Read-only view of PLIC interface geometry.
 *
 * All pointers refer to device memory of size `n` (= nx*ny*nz). Pointers are
 * valid for the lifetime of the producing VOFSolver instance and become stale
 * the moment any kernel writes to its fill_level field. Re-fetch the view
 * after any operation that modifies fill_level.
 *
 * `plic_ready` is a host-side hint set by VOFSolver. When false, callers must
 * not dereference `d_alpha` (it may be null or stale) and must fall back to
 * the legacy `|∇f|` path. `d_normal_x/y/z` and `d_fill` are always valid
 * after VOFSolver construction.
 */
struct InterfaceGeometryView {
    // Per-cell unit normal components (separate float arrays for coalesced
    // GPU access — packing into float3 is more expensive on Hopper/Ampere).
    // Points from liquid (f=1) toward gas (f=0).
    const float* d_normal_x = nullptr;
    const float* d_normal_y = nullptr;
    const float* d_normal_z = nullptr;

    // Signed alpha in cell-corner unit-cube frame. See header doc for the
    // exact convention. Null if `plic_ready == false`.
    const float* d_alpha = nullptr;

    // Fill level f ∈ [0,1] (= liquid volume fraction). Always valid.
    const float* d_fill = nullptr;

    // True iff the PLIC reconstruction in (d_normal_*, d_alpha) is consistent
    // with the current d_fill. False until VOFSolver::recomputePLICReconstruction()
    // (or advectFillLevelPLIC) is called after a fill modification.
    bool plic_ready = false;

    // Domain dimensions (for kernel index decoding).
    int nx = 0;
    int ny = 0;
    int nz = 0;
    int n  = 0;        // = nx * ny * nz

    // Lattice spacing [m]. Multiply lattice-unit distances by this for
    // physical-unit results.
    float dx = 1.0f;
};

// ============================================================================
// Device-side helper functions
// ============================================================================
//
// These are header-inline __device__ functions so any consumer kernel can call
// them without requiring a separate translation unit. They encode the
// API-boundary conversion from the internal "cell-corner signed-alpha" storage
// to the physically intuitive "signed distance from cell center" convention.
// ============================================================================

/**
 * @brief Signed distance from cell center to PLIC plane, in lattice units.
 *
 * Given the internal storage convention (cell-corner unit cube, signed alpha),
 * the cell center sits at X = Y = Z = 0.5 in local coordinates. The plane
 * equation evaluated at the center gives:
 *
 *     n · center = 0.5 * (n_x + n_y + n_z)
 *
 * The signed plane offset relative to the center is then:
 *
 *     d_center = alpha_signed - 0.5 * (n_x + n_y + n_z)
 *
 * Sign convention: d_center > 0 means the cell center is on the LIQUID side
 * (below the plane in the n̂ direction). d_center < 0 means the center is on
 * the GAS side (above the plane). This matches the convention that n̂ points
 * from liquid into gas.
 *
 * Result is in lattice units (cell side = 1). Multiply by `dx` for meters.
 *
 * @param idx  Linear cell index (i + nx*(j + ny*k))
 * @param view InterfaceGeometryView with valid d_alpha, d_normal_*
 * @return Signed distance d ∈ approx [-sqrt(3)/2, +sqrt(3)/2] lattice units.
 *         Caller must guard against `view.plic_ready == false`.
 */
__device__ __forceinline__ float plicSignedDistanceFromCenter(
    int idx, const InterfaceGeometryView& view)
{
    const float nx = view.d_normal_x[idx];
    const float ny = view.d_normal_y[idx];
    const float nz = view.d_normal_z[idx];
    const float a  = view.d_alpha[idx];
    return a - 0.5f * (nx + ny + nz);
}

/**
 * @brief Brackbill-Kothe-Zemach sharp surface delta function (cosine kernel).
 *
 * Implements the partition-of-unity cosine kernel used in CSF surface tension:
 *
 *     δ_h(d) = (1 / h) * 0.5 * (1 + cos(π d / h))    for |d| < h
 *     δ_h(d) = 0                                      otherwise
 *
 * Properties:
 *   - ∫_{-h}^{+h} δ_h(d) dd = 1 (normalized)
 *   - δ_h(0) = 1/h (peak at the interface)
 *   - Smooth (C^1) at the support boundaries d = ±h
 *
 * Units: `d` and `h` must be in the same units (typically lattice units).
 * Returned value has units 1/[d] (e.g. 1/lattice-unit). Multiply by surface
 * force [N/m^2] / dx [m] (after unit conversion) for volumetric force [N/m^3].
 *
 * Reference:
 *   Brackbill, Kothe & Zemach (1992). A continuum method for modeling surface
 *   tension. JCP 100, 335-354.
 *
 * @param d         Signed distance to the interface
 * @param h_smooth  Smoothing half-width (typical: 1.0–1.5 in lattice units)
 */
__device__ __forceinline__ float plicCosineDelta(float d, float h_smooth)
{
    if (h_smooth <= 0.0f) return 0.0f;
    if (fabsf(d) >= h_smooth) return 0.0f;
    constexpr float kPi = 3.14159265358979323846f;
    return (1.0f / h_smooth) * 0.5f * (1.0f + cosf(kPi * d / h_smooth));
}

/**
 * @brief Identify whether a cell is a PLIC interface cell.
 *
 * Returns true iff f ∈ (eps, 1-eps) — i.e. the cell carries a meaningful
 * sub-cell interface. Bulk cells (f≈0 or f≈1) and out-of-range cells return
 * false. Use the same `eps` as the internal PLIC kernels (1e-6) for
 * consistency.
 */
__device__ __forceinline__ bool plicIsInterfaceCell(
    int idx, const InterfaceGeometryView& view, float eps = 1e-6f)
{
    const float f = view.d_fill[idx];
    return (f > eps) && (f < (1.0f - eps));
}

// ============================================================================
// PLIC cell surface area (Phase 4b)
// ============================================================================

/**
 * @brief Area of the PLIC plane polygon inside the unit cube [0,1]³.
 *
 * Computes A_PLIC = dV/dα, the cross-section area of the plane
 *
 *     n_x·X + n_y·Y + n_z·Z = alpha_signed
 *
 * inside the unit cell, using a numerical central-difference on the
 * Scardovelli-Zaleski volume formula (same inclusion-exclusion used by
 * plicVolume3D).  The relation dV/dα = A is exact because |n̂| = 1 maps
 * the unit-cube alpha derivative directly to 3-D polygon area.
 *
 * Inputs use the same signed-frame convention as plicVolumeInBox:
 *   - (n_x, n_y, n_z) is the unit outward normal (can be negative).
 *   - alpha_signed is the plane offset such that the liquid volume below
 *     the plane equals f.
 *
 * Return value is dimensionless (in units of dx²/dx² = 1). Multiply by
 * dx² to get physical area [m²].
 *
 * Edge cases: |n|≈0, f≈0, f≈1, or alpha out of range → return 0.
 *
 * Accuracy: the central-difference step ε = 1e-4 gives relative error
 * < 0.001% for smooth V(α) — well within the 1–5% test tolerances.
 *
 * @param n_x, n_y, n_z  Unit normal components (signed, |n|≈1).
 * @param alpha_signed    Plane offset in cell-corner unit-cube frame.
 * @return               Area in lattice units (cell-side = 1).
 */
__device__ __forceinline__ float plicCellSurfaceArea(
    float n_x, float n_y, float n_z, float alpha_signed)
{
    // Guard: degenerate normal
    float n2 = n_x*n_x + n_y*n_y + n_z*n_z;
    if (n2 < 0.25f) return 0.0f;

    // Sign-flip to all-positive: same logic as plicVolumeInBox.
    float alpha = alpha_signed;
    float mx = fabsf(n_x);
    float my = fabsf(n_y);
    float mz = fabsf(n_z);
    if (n_x < 0.0f) alpha += mx;
    if (n_y < 0.0f) alpha += my;
    if (n_z < 0.0f) alpha += mz;

    // Sort m1 ≥ m2 ≥ m3 (required by the SZ volume formula).
    float m1 = mx, m2 = my, m3 = mz;
    if (m1 < m2) { float t = m1; m1 = m2; m2 = t; }
    if (m1 < m3) { float t = m1; m1 = m3; m3 = t; }
    if (m2 < m3) { float t = m2; m2 = m3; m3 = t; }

    float S = m1 + m2 + m3;
    // Guard: out-of-range alpha → zero area (plane misses the cell)
    if (alpha <= 0.0f || alpha >= S) return 0.0f;

    // Volume function V(a) = plicVolume3D(a, m1, m2, m3) via the
    // inclusion-exclusion formula, inlined to avoid a forward-declaration
    // dependency on the static __device__ in vof_solver.cu.
    //
    // The 3D formula (Scardovelli & Zaleski 2000, eq. 28), valid for
    // 0 ≤ a ≤ S/2; apply V(a) = 1 - V(S-a) for a > S/2.
    auto szVol = [&](float a) -> float {
        if (a <= 0.0f) return 0.0f;
        if (a >= S)    return 1.0f;
        bool flip = (a > 0.5f * S);
        if (flip) a = S - a;

        // 2D degenerate (m3 ≈ 0)
        if (m3 < 1e-8f) {
            if (m2 < 1e-8f) {
                float v = (m1 > 1e-30f) ? a / m1 : 0.0f;
                if (flip) v = 1.0f - v;
                return fmaxf(0.0f, fminf(1.0f, v));
            }
            float S2 = m1 + m2;
            if (a >= S2) { float v = flip ? 0.0f : 1.0f; return v; }
            float v;
            if (a <= m2)       v = (a*a) / (2.0f*m1*m2);
            else if (a <= m1)  v = (a - 0.5f*m2) / m1;
            else               { float tt = S2 - a; v = 1.0f - (tt*tt)/(2.0f*m1*m2); }
            if (flip) v = 1.0f - v;
            return fmaxf(0.0f, fminf(1.0f, v));
        }

        // 3D full formula
        float denom = 6.0f * m1 * m2 * m3;
        float vol = a*a*a;
        float t1 = a - m1; if (t1 > 0.0f) vol -= t1*t1*t1;
        float t2 = a - m2; if (t2 > 0.0f) vol -= t2*t2*t2;
        float t3 = a - m3; if (t3 > 0.0f) vol -= t3*t3*t3;
        float t12 = a - m1 - m2; if (t12 > 0.0f) vol += t12*t12*t12;
        float t13 = a - m1 - m3; if (t13 > 0.0f) vol += t13*t13*t13;
        float t23 = a - m2 - m3; if (t23 > 0.0f) vol += t23*t23*t23;
        float v = vol / denom;
        if (flip) v = 1.0f - v;
        return fmaxf(0.0f, fminf(1.0f, v));
    };

    // Central-difference: A = dV/dα  (exact relation for unit-normal plane)
    constexpr float kEps = 1e-4f;
    float v_hi = szVol(alpha + kEps);
    float v_lo = szVol(alpha - kEps);
    float area = (v_hi - v_lo) / (2.0f * kEps);

    return fmaxf(0.0f, area);
}

}  // namespace physics
}  // namespace lbm
