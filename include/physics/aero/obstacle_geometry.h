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
