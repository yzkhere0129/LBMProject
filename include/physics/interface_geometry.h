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

}  // namespace physics
}  // namespace lbm
