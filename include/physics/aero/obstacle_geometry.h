/**
 * @file obstacle_geometry.h
 * @brief Host-side helpers for stamping immersed obstacles into a solid mask
 *
 * These helpers fill a host-side per-cell uint8 mask (size nx*ny*nz, layout
 * id = x + y*nx + z*nx*ny) with Streaming::CELL_SOLID at every cell whose
 * centre falls inside the obstacle. The result is uploaded to FluidLBM via
 * setSolidMask().
 *
 * Stair-step (cell-centre-in-solid) test only — first-order at the surface.
 * Sufficient for Schäfer–Turek validation; QBB curved BC is a Phase 1b
 * upgrade for tighter agreement.
 */

#pragma once

#include <vector>
#include <cmath>
#include <algorithm>
#include "core/streaming.h"
#include "core/lattice_d3q19.h"
#include "core/lattice_d3q27.h"

namespace lbm {
namespace physics {
namespace aero {

/**
 * @brief Allocate a host-side fluid mask filled with CELL_FLUID.
 */
inline std::vector<unsigned char> makeFluidMask(int nx, int ny, int nz) {
    return std::vector<unsigned char>(static_cast<size_t>(nx) * ny * nz,
                                      core::Streaming::CELL_FLUID);
}

/**
 * @brief Stamp a circular cylinder with axis along z (3D thin-slab benchmarks).
 *
 * @param mask   Host mask of size nx*ny*nz (modified in place).
 * @param nx,ny,nz Domain extents in cells.
 * @param dx     Lattice spacing [m].
 * @param cx_phys,cy_phys Cylinder centre in physical units [m] (x, y).
 * @param radius_phys Cylinder radius [m].
 *
 * Every cell whose centre (i+0.5)·dx, (j+0.5)·dx satisfies
 *   (x-cx)^2 + (y-cy)^2 <= R^2
 * is marked CELL_SOLID for all z.
 */
inline void stampCylinderZ(std::vector<unsigned char>& mask,
                           int nx, int ny, int nz, float dx,
                           float cx_phys, float cy_phys, float radius_phys) {
    const float r2 = radius_phys * radius_phys;
    for (int j = 0; j < ny; ++j) {
        const float y = (j + 0.5f) * dx;
        const float dy_ = y - cy_phys;
        for (int i = 0; i < nx; ++i) {
            const float x = (i + 0.5f) * dx;
            const float dx_ = x - cx_phys;
            if (dx_ * dx_ + dy_ * dy_ <= r2) {
                for (int k = 0; k < nz; ++k) {
                    const size_t id =
                        static_cast<size_t>(i) +
                        static_cast<size_t>(j) * nx +
                        static_cast<size_t>(k) * nx * ny;
                    mask[id] = core::Streaming::CELL_SOLID;
                }
            }
        }
    }
}

/**
 * @brief Stamp a NACA 4-digit symmetric airfoil (NACA 00XX, e.g. NACA0012).
 *
 * Uses the standard NACA thickness distribution (Selig formulation):
 *   y_t(s) = 5·t·(0.2969·√s − 0.126·s − 0.3516·s² + 0.2843·s³ − 0.1015·s⁴)
 * where s = x/c ∈ [0, 1]. The airfoil profile is upper = +y_t, lower = −y_t,
 * relative to the chord line. With angle of attack α, the airfoil is rotated
 * so the chord makes angle α with +x axis.
 *
 * @param mask    Host mask (size nx*ny*nz, modified in place)
 * @param nx,ny,nz Domain extents
 * @param dx      Lattice spacing [m]
 * @param cx_LE,cy_LE Leading-edge position [m] (x, y) BEFORE rotation
 * @param chord   Chord length [m]
 * @param thick_pct Thickness in percent of chord (12 for NACA0012)
 * @param alpha_rad Angle of attack [rad] (positive = nose up)
 */
inline void stampNacaAirfoil4Digit(std::vector<unsigned char>& mask,
                                   int nx, int ny, int nz, float dx,
                                   float cx_LE, float cy_LE,
                                   float chord, float thick_pct,
                                   float alpha_rad,
                                   int k_z_lo = -1, int k_z_hi = -1) {
    // Span-limit (O, 2026-05-21): if k_z_lo / k_z_hi ≥ 0, only stamp at
    // z indices k ∈ [k_z_lo, k_z_hi). Default -1/-1 → stamp all z (legacy
    // z-extruded behavior). For finite-span wing demos.
    const int k_lo = (k_z_lo < 0) ? 0  : k_z_lo;
    const int k_hi = (k_z_hi < 0) ? nz : k_z_hi;
    const float t = thick_pct / 100.0f;
    // Standard aero convention: α>0 = "nose up" → TE below LE (chord slope -sin α).
    // Stamper applies R(-α_std) = R(+alpha_rad) on world→chord with sin negated.
    const float cos_a =  std::cos(alpha_rad);
    const float sin_a = -std::sin(alpha_rad);

    auto naca_thickness = [t](float s) -> float {
        if (s < 0.0f || s > 1.0f) return -1.0f;  // outside chord
        const float sqrt_s = std::sqrt(s);
        return 5.0f * t * (0.2969f * sqrt_s
                         - 0.1260f * s
                         - 0.3516f * s * s
                         + 0.2843f * s * s * s
                         - 0.1015f * s * s * s * s);
    };

    for (int j = 0; j < ny; ++j) {
        const float y = (j + 0.5f) * dx;
        for (int i = 0; i < nx; ++i) {
            const float x = (i + 0.5f) * dx;
            // Translate to LE-relative, then rotate by -α to get chord-aligned coords
            const float xr =  cos_a * (x - cx_LE) + sin_a * (y - cy_LE);
            const float yr = -sin_a * (x - cx_LE) + cos_a * (y - cy_LE);
            const float s = xr / chord;
            const float y_t = naca_thickness(s) * chord;
            if (s >= 0.0f && s <= 1.0f && std::abs(yr) <= y_t) {
                for (int k = k_lo; k < k_hi; ++k) {
                    const size_t id =
                        static_cast<size_t>(i) +
                        static_cast<size_t>(j) * nx +
                        static_cast<size_t>(k) * nx * ny;
                    mask[id] = core::Streaming::CELL_SOLID;
                }
            }
        }
    }
}

/**
 * @brief Stamp a thin flat plate from LE to TE, rotated by alpha_rad.
 *
 * Same world->chord-frame transform as stampNacaAirfoil4Digit (for direct
 * comparison): xr = cos α dx + sin α dy, yr = -sin α dx + cos α dy.
 *
 * A cell is solid iff 0 <= s <= 1 (within the chord span) AND
 * |yr| <= 0.5 * thickness_phys (within the plate's half-thickness).
 *
 * For a 1-cell-thick plate set thickness_phys = dx. For testing whether the
 * NACA Cl-sign-flip is purely a rotation-induced stair-step artifact (it
 * should also appear for the flat plate at α≠0 if the hypothesis is right).
 */
inline void stampFlatPlate(std::vector<unsigned char>& mask,
                            int nx, int ny, int nz, float dx,
                            float cx_LE, float cy_LE,
                            float chord, float thickness_phys,
                            float alpha_rad) {
    // Match stampNacaAirfoil4Digit convention (std aero, α>0 = nose up).
    const float cos_a =  std::cos(alpha_rad);
    const float sin_a = -std::sin(alpha_rad);
    const float half_thick = 0.5f * thickness_phys;

    for (int j = 0; j < ny; ++j) {
        const float y = (j + 0.5f) * dx;
        for (int i = 0; i < nx; ++i) {
            const float x = (i + 0.5f) * dx;
            const float xr =  cos_a * (x - cx_LE) + sin_a * (y - cy_LE);
            const float yr = -sin_a * (x - cx_LE) + cos_a * (y - cy_LE);
            const float s = xr / chord;
            if (s >= 0.0f && s <= 1.0f && std::abs(yr) <= half_thick) {
                for (int k = 0; k < nz; ++k) {
                    const size_t id =
                        static_cast<size_t>(i) +
                        static_cast<size_t>(j) * nx +
                        static_cast<size_t>(k) * nx * ny;
                    mask[id] = core::Streaming::CELL_SOLID;
                }
            }
        }
    }
}

/**
 * @brief Stamp a sphere centred at (cx, cy, cz).
 */
inline void stampSphere(std::vector<unsigned char>& mask,
                        int nx, int ny, int nz, float dx,
                        float cx_phys, float cy_phys, float cz_phys,
                        float radius_phys) {
    const float r2 = radius_phys * radius_phys;
    for (int k = 0; k < nz; ++k) {
        const float z = (k + 0.5f) * dx;
        const float dz_ = z - cz_phys;
        for (int j = 0; j < ny; ++j) {
            const float y = (j + 0.5f) * dx;
            const float dy_ = y - cy_phys;
            for (int i = 0; i < nx; ++i) {
                const float x = (i + 0.5f) * dx;
                const float dx_ = x - cx_phys;
                if (dx_ * dx_ + dy_ * dy_ + dz_ * dz_ <= r2) {
                    const size_t id =
                        static_cast<size_t>(i) +
                        static_cast<size_t>(j) * nx +
                        static_cast<size_t>(k) * nx * ny;
                    mask[id] = core::Streaming::CELL_SOLID;
                }
            }
        }
    }
}

/**
 * @brief Allocate a host-side per-link q-fraction buffer initialised to 1.0
 *        (which means "no curved BC" for every link).
 *
 * Layout matches FluidLBM's f-fields: q-major SoA, idx = id + q · n_cells.
 */
inline std::vector<float> makeUnitQFraction(int nx, int ny, int nz) {
    return std::vector<float>(static_cast<size_t>(nx) * ny * nz * core::D3Q19::Q,
                              1.0f);
}

// Use the canonical D3Q19 host tables (D3Q19::h_ex/h_ey/h_ez) to enumerate
// directions — these mirror the __constant__ device tables exactly.
// A previous version of this file shipped its own wrong table (xz/yz-edges
// swapped at q=11..18), which broke QBB silently — keep using the canonical
// source-of-truth here.

/**
 * @brief Compute per-link Bouzidi q-fractions for a z-axis cylinder.
 *
 * For every fluid cell X with a SOLID neighbour at X+e_q, this function
 * solves the line-circle intersection in the (x, y) plane and writes
 * q_frac ∈ (0,1] = (distance from X to wall) / (link length) into qfrac.
 *
 * Cells whose neighbour in direction q is fluid keep q_frac = 1.0 (unused).
 *
 * Pre-condition: mask must already be stamped (stampCylinderZ called first).
 *
 * @param qfrac    Q*N q-fraction buffer (initialised to 1.0 by makeUnitQFraction)
 * @param mask     Solid mask (CELL_FLUID / CELL_SOLID per cell)
 * @param nx,ny,nz domain extents
 * @param dx       lattice spacing [m]
 * @param cx,cy    cylinder centre [m] (same units as cell-centre = (i+0.5)·dx)
 * @param R        cylinder radius [m]
 */
inline void computeCylinderZQ(std::vector<float>& qfrac,
                              const std::vector<unsigned char>& mask,
                              int nx, int ny, int nz, float dx,
                              float cx, float cy, float R) {
    using core::D3Q19;
    const float R2 = R * R;
    const size_t n_cells = static_cast<size_t>(nx) * ny * nz;

    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                const size_t id =
                    static_cast<size_t>(i) +
                    static_cast<size_t>(j) * nx +
                    static_cast<size_t>(k) * nx * ny;
                if (mask[id] != core::Streaming::CELL_FLUID) continue;

                const float xC = (i + 0.5f) * dx;
                const float yC = (j + 0.5f) * dx;

                for (int q = 1; q < D3Q19::Q; ++q) {  // skip rest
                    const int ni = i + D3Q19::h_ex[q];
                    const int nj = j + D3Q19::h_ey[q];
                    const int nk = k + D3Q19::h_ez[q];
                    if (ni < 0 || ni >= nx || nj < 0 || nj >= ny ||
                        nk < 0 || nk >= nz) continue;
                    const size_t nid =
                        static_cast<size_t>(ni) +
                        static_cast<size_t>(nj) * nx +
                        static_cast<size_t>(nk) * nx * ny;
                    if (mask[nid] == core::Streaming::CELL_FLUID) continue;

                    // Solid neighbour. Solve intersection in (x,y) plane:
                    //   |X + s·d − C|² = R²,  s ∈ (0, 1]
                    // d = (Hex, Hey)·dx (z-component does not change distance
                    // to a z-axis cylinder; if Hez≠0 the z motion is irrelevant
                    // to the distance, just to the link length — we use the
                    // (x,y) projection length as the effective link).
                    const float dxL = D3Q19::h_ex[q] * dx;
                    const float dyL = D3Q19::h_ey[q] * dx;
                    const float A = dxL * dxL + dyL * dyL;
                    if (A < 1e-20f) {
                        // Pure-z link with z-axis cylinder: never intersects;
                        // but neighbour is solid → degenerate. Default 1.0.
                        continue;
                    }
                    const float dx0 = xC - cx;
                    const float dy0 = yC - cy;
                    const float B = 2.0f * (dxL * dx0 + dyL * dy0);
                    const float C = dx0 * dx0 + dy0 * dy0 - R2;
                    const float disc = B * B - 4.0f * A * C;
                    if (disc < 0.0f) {
                        // Numerical edge: stick with halfway BB
                        continue;
                    }
                    const float sqrt_disc = std::sqrt(disc);
                    const float s1 = (-B - sqrt_disc) / (2.0f * A);
                    const float s2 = (-B + sqrt_disc) / (2.0f * A);
                    // We want the smallest s ∈ (0, 1] (entry into solid).
                    float s = 1.0f;
                    if (s1 > 1e-6f && s1 <= 1.0f) s = s1;
                    else if (s2 > 1e-6f && s2 <= 1.0f) s = s2;
                    qfrac[id + static_cast<size_t>(q) * n_cells] =
                        std::clamp(s, 1e-6f, 1.0f);
                }
            }
        }
    }
}

/**
 * @brief Compute per-link Bouzidi q-fractions for a 3D sphere.
 *
 * Same convention as computeCylinderZQ but in 3D.
 */
inline void computeSphereQ(std::vector<float>& qfrac,
                           const std::vector<unsigned char>& mask,
                           int nx, int ny, int nz, float dx,
                           float cx, float cy, float cz, float R) {
    using core::D3Q19;
    const float R2 = R * R;
    const size_t n_cells = static_cast<size_t>(nx) * ny * nz;

    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                const size_t id =
                    static_cast<size_t>(i) +
                    static_cast<size_t>(j) * nx +
                    static_cast<size_t>(k) * nx * ny;
                if (mask[id] != core::Streaming::CELL_FLUID) continue;

                const float xC = (i + 0.5f) * dx;
                const float yC = (j + 0.5f) * dx;
                const float zC = (k + 0.5f) * dx;

                for (int q = 1; q < D3Q19::Q; ++q) {
                    const int ni = i + D3Q19::h_ex[q];
                    const int nj = j + D3Q19::h_ey[q];
                    const int nk = k + D3Q19::h_ez[q];
                    if (ni < 0 || ni >= nx || nj < 0 || nj >= ny ||
                        nk < 0 || nk >= nz) continue;
                    const size_t nid =
                        static_cast<size_t>(ni) +
                        static_cast<size_t>(nj) * nx +
                        static_cast<size_t>(nk) * nx * ny;
                    if (mask[nid] == core::Streaming::CELL_FLUID) continue;

                    const float dxL = D3Q19::h_ex[q] * dx;
                    const float dyL = D3Q19::h_ey[q] * dx;
                    const float dzL = D3Q19::h_ez[q] * dx;
                    const float A = dxL * dxL + dyL * dyL + dzL * dzL;
                    const float dx0 = xC - cx, dy0 = yC - cy, dz0 = zC - cz;
                    const float B = 2.0f * (dxL * dx0 + dyL * dy0 + dzL * dz0);
                    const float C = dx0 * dx0 + dy0 * dy0 + dz0 * dz0 - R2;
                    const float disc = B * B - 4.0f * A * C;
                    if (disc < 0.0f) continue;
                    const float sqrt_disc = std::sqrt(disc);
                    const float s1 = (-B - sqrt_disc) / (2.0f * A);
                    const float s2 = (-B + sqrt_disc) / (2.0f * A);
                    float s = 1.0f;
                    if (s1 > 1e-6f && s1 <= 1.0f) s = s1;
                    else if (s2 > 1e-6f && s2 <= 1.0f) s = s2;
                    qfrac[id + static_cast<size_t>(q) * n_cells] =
                        std::clamp(s, 1e-6f, 1.0f);
                }
            }
        }
    }
}

/**
 * @brief Allocate a host-side per-link D3Q27 q-fraction buffer initialised to
 *        1.0 (which means "no curved BC" for every link).
 *
 * Layout matches D3Q27 SoA: q-major, idx = id + q · n_cells, q ∈ [0, 27).
 */
inline std::vector<float> makeUnitQFraction27(int nx, int ny, int nz) {
    return std::vector<float>(static_cast<size_t>(nx) * ny * nz * core::D3Q27::Q,
                              1.0f);
}

/**
 * @brief CSR-like sparse storage for q-fractions. Saves ~99% memory vs dense
 *        layout (most cells have 0 solid-neighbour links and store nothing).
 *
 * Layout:
 *   offset.size()   = n_cells + 1
 *   offset[i+1] - offset[i] = number of solid-neighbour links from fluid cell i
 *   link_q[K], link_qfrac[K]  where K = offset[n_cells]
 *
 * Kernel-side lookup: for cell id, direction Q, linear-scan
 *   for e in [offset[id], offset[id+1]):
 *       if link_q[e] == Q: return link_qfrac[e]
 *   return 1.0f  (no solid in direction Q from this cell)
 */
struct SparseQFraction {
    std::vector<int>           offset;     // size n_cells + 1
    std::vector<unsigned char> link_q;     // q ∈ [1, 27)
    std::vector<float>         link_qfrac;

    // D4 (2026-05-20): optional per-link unit outward normal of the surface
    // hit by the ray cast in direction c_q. Three arrays of size K each.
    // Populated by makeSTLQFractionSparse (D3 STL builder). NACA analytical
    // path leaves these empty — its surface normal is derivable from chord-frame
    // and never needed by the current force probe. Future force-probe variants
    // (e.g., Caiazzo-Junk pressure integration, recoil/Marangoni couplings)
    // can read these directly.
    std::vector<float> link_nx;
    std::vector<float> link_ny;
    std::vector<float> link_nz;
};

/**
 * @brief Compute per-link Bouzidi q-fractions for a NACA 4-digit symmetric
 *        airfoil (z-extruded) on the D3Q27 stencil.
 *
 * For every fluid cell X with a SOLID neighbour at X+e_q (q ∈ [1, 27)), this
 * walks the link with bisection to find the smallest s ∈ (0, 1] where the
 * point P(s) = X + s·D enters the airfoil's stamped region (chord-frame
 * inside test). Cells whose neighbour in direction q is fluid keep
 * q_frac = 1.0 (unused).
 *
 * Pre-condition: mask must already be stamped (stampNacaAirfoil4Digit called
 * first). The mask convention used here matches the stamper:
 *   inside ⇔ 0 ≤ xr/c ≤ 1 AND |yr| ≤ y_t(xr/c) · c
 * where (xr, yr) = R(-α) · (P − LE) is the chord-aligned local frame.
 *
 * @param mask     Solid mask (CELL_FLUID / CELL_SOLID per cell), nx·ny·nz
 * @param nx,ny,nz Domain extents (cells)
 * @param dx       Lattice spacing [m]
 * @param cx_LE,cy_LE Leading-edge position [m] (BEFORE rotation; same units
 *                 as cell-centre = (i+0.5)·dx)
 * @param chord    Chord length [m]
 * @param thick_pct Thickness percent (12 for NACA0012)
 * @param alpha_rad Angle of attack [rad]
 */
inline std::vector<float> makeNacaQFraction(
    const std::vector<unsigned char>& mask,
    int nx, int ny, int nz, float dx,
    float cx_LE, float cy_LE,
    float chord, float thick_pct, float alpha_rad)
{
    using core::D3Q27;
    const float t = thick_pct / 100.0f;
    // Match stampNacaAirfoil4Digit: std aero convention, α>0 = nose up.
    const float cos_a =  std::cos(alpha_rad);
    const float sin_a = -std::sin(alpha_rad);
    const size_t n_cells = static_cast<size_t>(nx) * ny * nz;

    std::vector<float> qfrac(n_cells * D3Q27::Q, 1.0f);

    auto naca_y_t = [t](float s) -> float {
        if (s < 0.0f || s > 1.0f) return -1.0f;
        const float r = std::sqrt(s);
        return 5.0f * t * (0.2969f * r
                         - 0.1260f * s
                         - 0.3516f * s * s
                         + 0.2843f * s * s * s
                         - 0.1015f * s * s * s * s);
    };

    auto inside_airfoil = [&](float xw, float yw) -> bool {
        const float xr =  cos_a * (xw - cx_LE) + sin_a * (yw - cy_LE);
        const float yr = -sin_a * (xw - cx_LE) + cos_a * (yw - cy_LE);
        const float s = xr / chord;
        if (s < 0.0f || s > 1.0f) return false;
        return std::abs(yr) <= naca_y_t(s) * chord;
    };

    // Sub-sampling + bisection to localise s ∈ (0, 1] where link enters airfoil.
    // 64 scan steps (Δs = 1/64) bracket the entry; 20 bisection iters refine
    // to ~1/(64·2²⁰) ≈ 1.5e-8 (well below FP32 precision needs).
    constexpr int N_SCAN = 64;
    constexpr int N_BISECT = 20;
    constexpr float QMIN = 1e-6f;

    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                const size_t id =
                    static_cast<size_t>(i) +
                    static_cast<size_t>(j) * nx +
                    static_cast<size_t>(k) * nx * ny;
                if (mask[id] != core::Streaming::CELL_FLUID) continue;

                const float xC = (i + 0.5f) * dx;
                const float yC = (j + 0.5f) * dx;

                for (int q = 1; q < D3Q27::Q; ++q) {
                    const int ni = i + D3Q27::h_ex[q];
                    const int nj = j + D3Q27::h_ey[q];
                    const int nk = k + D3Q27::h_ez[q];
                    if (ni < 0 || ni >= nx || nj < 0 || nj >= ny ||
                        nk < 0 || nk >= nz) continue;
                    const size_t nid =
                        static_cast<size_t>(ni) +
                        static_cast<size_t>(nj) * nx +
                        static_cast<size_t>(nk) * nx * ny;
                    if (mask[nid] == core::Streaming::CELL_FLUID) continue;

                    // Solid neighbour. Airfoil is z-extruded, so the inside
                    // test depends only on (x, y); ez component contributes
                    // to the link length but not to in/out classification.
                    const float dxL = D3Q27::h_ex[q] * dx;
                    const float dyL = D3Q27::h_ey[q] * dx;

                    // Scan in N_SCAN steps to bracket the entry.
                    int idx_first = -1;
                    for (int n = 1; n <= N_SCAN; ++n) {
                        const float s = static_cast<float>(n) /
                                        static_cast<float>(N_SCAN);
                        if (inside_airfoil(xC + s * dxL, yC + s * dyL)) {
                            idx_first = n;
                            break;
                        }
                    }
                    if (idx_first < 0) {
                        // Sub-sampling missed the airfoil even though the
                        // destination cell-centre is solid. Can happen when
                        // the link grazes a corner. Default to halfway BB.
                        qfrac[id + static_cast<size_t>(q) * n_cells] = 0.5f;
                        continue;
                    }
                    float s_lo = (idx_first - 1) / static_cast<float>(N_SCAN);
                    float s_hi =  idx_first      / static_cast<float>(N_SCAN);
                    for (int b = 0; b < N_BISECT; ++b) {
                        const float sm = 0.5f * (s_lo + s_hi);
                        if (inside_airfoil(xC + sm * dxL, yC + sm * dyL))
                            s_hi = sm;
                        else
                            s_lo = sm;
                    }
                    float qf = 0.5f * (s_lo + s_hi);
                    qfrac[id + static_cast<size_t>(q) * n_cells] =
                        std::clamp(qf, QMIN, 1.0f);
                }
            }
        }
    }
    return qfrac;
}

/**
 * @brief Sparse-layout version of makeNacaQFraction. Identical algorithm
 *        (64-step scan + 20-bit bisection per fluid-to-solid link) but packs
 *        only the fluid-to-solid links into the SparseQFraction CSR struct.
 *
 * For 4GB GPUs the dense qfrac (27·N·4 bytes) becomes the dominant
 * memory consumer at D/dx≥120, so the sparse form is required for higher
 * resolution runs.
 */
inline SparseQFraction makeNacaQFractionSparse(
    const std::vector<unsigned char>& mask,
    int nx, int ny, int nz, float dx,
    float cx_LE, float cy_LE,
    float chord, float thick_pct, float alpha_rad)
{
    using core::D3Q27;
    const float t = thick_pct / 100.0f;
    const float cos_a =  std::cos(alpha_rad);
    const float sin_a = -std::sin(alpha_rad);
    const size_t n_cells = static_cast<size_t>(nx) * ny * nz;

    auto naca_y_t = [t](float s) -> float {
        if (s < 0.0f || s > 1.0f) return -1.0f;
        const float r = std::sqrt(s);
        return 5.0f * t * (0.2969f * r - 0.1260f * s
                         - 0.3516f * s * s + 0.2843f * s * s * s
                         - 0.1015f * s * s * s * s);
    };
    auto inside_airfoil = [&](float xw, float yw) -> bool {
        const float xr =  cos_a * (xw - cx_LE) + sin_a * (yw - cy_LE);
        const float yr = -sin_a * (xw - cx_LE) + cos_a * (yw - cy_LE);
        const float s = xr / chord;
        if (s < 0.0f || s > 1.0f) return false;
        return std::abs(yr) <= naca_y_t(s) * chord;
    };

    constexpr int N_SCAN = 64;
    constexpr int N_BISECT = 20;
    constexpr float QMIN = 1e-6f;

    // Pass 1: count solid-neighbour links per fluid cell.
    SparseQFraction out;
    out.offset.assign(n_cells + 1, 0);
    for (int k = 0; k < nz; ++k)
        for (int j = 0; j < ny; ++j)
            for (int i = 0; i < nx; ++i) {
                const size_t id = static_cast<size_t>(i)
                                + static_cast<size_t>(j) * nx
                                + static_cast<size_t>(k) * nx * ny;
                if (mask[id] != core::Streaming::CELL_FLUID) continue;
                int cnt = 0;
                for (int q = 1; q < D3Q27::Q; ++q) {
                    const int ni = i + D3Q27::h_ex[q];
                    const int nj = j + D3Q27::h_ey[q];
                    const int nk = k + D3Q27::h_ez[q];
                    if (ni < 0 || ni >= nx || nj < 0 || nj >= ny ||
                        nk < 0 || nk >= nz) continue;
                    const size_t nid = static_cast<size_t>(ni)
                                     + static_cast<size_t>(nj) * nx
                                     + static_cast<size_t>(nk) * nx * ny;
                    if (mask[nid] == core::Streaming::CELL_FLUID) continue;
                    ++cnt;
                }
                out.offset[id + 1] = cnt;
            }
    // Prefix sum.
    for (size_t i = 1; i <= n_cells; ++i) out.offset[i] += out.offset[i-1];
    const int K = out.offset[n_cells];
    out.link_q.assign(K, 0);
    out.link_qfrac.assign(K, 1.0f);

    // Pass 2: fill. Use a per-cell write cursor.
    std::vector<int> head = out.offset;  // head[id] tracks next slot for cell id

    for (int k = 0; k < nz; ++k)
        for (int j = 0; j < ny; ++j)
            for (int i = 0; i < nx; ++i) {
                const size_t id = static_cast<size_t>(i)
                                + static_cast<size_t>(j) * nx
                                + static_cast<size_t>(k) * nx * ny;
                if (mask[id] != core::Streaming::CELL_FLUID) continue;
                const float xC = (i + 0.5f) * dx;
                const float yC = (j + 0.5f) * dx;
                for (int q = 1; q < D3Q27::Q; ++q) {
                    const int ni = i + D3Q27::h_ex[q];
                    const int nj = j + D3Q27::h_ey[q];
                    const int nk = k + D3Q27::h_ez[q];
                    if (ni < 0 || ni >= nx || nj < 0 || nj >= ny ||
                        nk < 0 || nk >= nz) continue;
                    const size_t nid = static_cast<size_t>(ni)
                                     + static_cast<size_t>(nj) * nx
                                     + static_cast<size_t>(nk) * nx * ny;
                    if (mask[nid] == core::Streaming::CELL_FLUID) continue;

                    const float dxL = D3Q27::h_ex[q] * dx;
                    const float dyL = D3Q27::h_ey[q] * dx;
                    int idx_first = -1;
                    for (int n = 1; n <= N_SCAN; ++n) {
                        const float s = static_cast<float>(n) / static_cast<float>(N_SCAN);
                        if (inside_airfoil(xC + s * dxL, yC + s * dyL)) {
                            idx_first = n;
                            break;
                        }
                    }
                    float qf;
                    if (idx_first < 0) {
                        qf = 0.5f;  // grazing-corner fallback
                    } else {
                        float s_lo = (idx_first - 1) / static_cast<float>(N_SCAN);
                        float s_hi =  idx_first      / static_cast<float>(N_SCAN);
                        for (int b = 0; b < N_BISECT; ++b) {
                            const float sm = 0.5f * (s_lo + s_hi);
                            if (inside_airfoil(xC + sm * dxL, yC + sm * dyL))
                                s_hi = sm;
                            else
                                s_lo = sm;
                        }
                        qf = std::clamp(0.5f * (s_lo + s_hi), QMIN, 1.0f);
                    }
                    const int slot = head[id]++;
                    out.link_q[slot]     = static_cast<unsigned char>(q);
                    out.link_qfrac[slot] = qf;
                }
            }
    return out;
}

/**
 * @brief Count solid cells in a host mask (diagnostic).
 */
inline long long countSolid(const std::vector<unsigned char>& mask) {
    long long n = 0;
    for (auto v : mask) if (v != core::Streaming::CELL_FLUID) ++n;
    return n;
}

} // namespace aero
} // namespace physics
} // namespace lbm
