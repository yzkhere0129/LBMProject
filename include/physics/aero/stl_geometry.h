/**
 * @file stl_geometry.h
 * @brief STL → Cartesian solid-mask stamp via ray-cast inside-outside test.
 *
 * For each grid cell centre, cast a ray in +x direction and count
 * intersections with STL triangles. Odd intersections = inside (solid),
 * even = outside (fluid). Standard "even-odd rule" for closed surfaces.
 *
 * D1 implementation: brute-force O(N_cells × N_tris). Fine for small STLs
 * (< 5k triangles × a few-million cells = < 1 minute on CPU).
 * D2 will replace with BVH acceleration; this file stays the entry point.
 *
 * Pre-condition: STL must be CLOSED (watertight). Open meshes give garbage
 * inside-outside results. We don't validate watertightness — caller's
 * responsibility (or use OnShape "Export STL → Closed solid" option).
 */

#pragma once

#include "io/stl_reader.h"
#include "core/streaming.h"
#include <vector>
#include <array>
#include <cmath>

namespace lbm {
namespace physics {
namespace aero {

namespace stl_detail {

/**
 * Möller-Trumbore ray-triangle intersection.
 * Ray: origin O, direction D (unit). Tests t > eps (positive direction).
 * Returns true if intersect, with t in t_out.
 */
inline bool ray_tri_intersect(const float O[3], const float D[3],
                              const std::array<float, 3>& v0,
                              const std::array<float, 3>& v1,
                              const std::array<float, 3>& v2,
                              float& t_out)
{
    constexpr float EPS = 1e-7f;
    const float e1x = v1[0] - v0[0], e1y = v1[1] - v0[1], e1z = v1[2] - v0[2];
    const float e2x = v2[0] - v0[0], e2y = v2[1] - v0[1], e2z = v2[2] - v0[2];
    const float px = D[1]*e2z - D[2]*e2y;
    const float py = D[2]*e2x - D[0]*e2z;
    const float pz = D[0]*e2y - D[1]*e2x;
    const float det = e1x*px + e1y*py + e1z*pz;
    if (det > -EPS && det < EPS) return false;  // ray parallel to triangle
    const float inv_det = 1.0f / det;
    const float tx = O[0] - v0[0], ty = O[1] - v0[1], tz = O[2] - v0[2];
    const float u = (tx*px + ty*py + tz*pz) * inv_det;
    if (u < 0.0f || u > 1.0f) return false;
    const float qx = ty*e1z - tz*e1y;
    const float qy = tz*e1x - tx*e1z;
    const float qz = tx*e1y - ty*e1x;
    const float v = (D[0]*qx + D[1]*qy + D[2]*qz) * inv_det;
    if (v < 0.0f || u + v > 1.0f) return false;
    const float t = (e2x*qx + e2y*qy + e2z*qz) * inv_det;
    if (t > EPS) { t_out = t; return true; }
    return false;
}

}  // namespace stl_detail

/**
 * @brief Stamp solid cells from STL mesh into mask.
 *
 * Ray-casts in +x direction from each cell centre. Cells with odd
 * intersection count are marked CELL_SOLID.
 *
 * O(N_cells × N_tris) cost — for D/dx=80, 1c chord, 15M cells × 5k tris
 * = 75e9 ops ≈ 25 sec single-threaded. Acceptable for D1. D2 will add BVH.
 *
 * Early-exit optimisation: skip cells outside STL bbox without ray-cast.
 * For a NACA-scale STL filling ~0.1% of the domain, this avoids 99.9%
 * of the ray casts.
 */
inline void stampSTL(std::vector<unsigned char>& mask,
                     int nx, int ny, int nz, float dx,
                     const lbm::io::STLMesh& mesh)
{
    const auto& tris = mesh.tris;
    const auto& bbox_lo = mesh.bbox_lo;
    const auto& bbox_hi = mesh.bbox_hi;

    // Cells fully outside bbox can be left FLUID.
    const int i_lo = std::max(0, (int)std::floor(bbox_lo[0] / dx) - 1);
    const int i_hi = std::min(nx, (int)std::ceil (bbox_hi[0] / dx) + 1);
    const int j_lo = std::max(0, (int)std::floor(bbox_lo[1] / dx) - 1);
    const int j_hi = std::min(ny, (int)std::ceil (bbox_hi[1] / dx) + 1);
    const int k_lo = std::max(0, (int)std::floor(bbox_lo[2] / dx) - 1);
    const int k_hi = std::min(nz, (int)std::ceil (bbox_hi[2] / dx) + 1);

    const float D[3] = { 1.0f, 0.0f, 0.0f };  // +x ray

    for (int k = k_lo; k < k_hi; ++k) {
        const float z = (k + 0.5f) * dx;
        for (int j = j_lo; j < j_hi; ++j) {
            const float y = (j + 0.5f) * dx;
            for (int i = i_lo; i < i_hi; ++i) {
                const float x = (i + 0.5f) * dx;
                const float O[3] = { x, y, z };
                int n_hit = 0;
                for (const auto& t : tris) {
                    // Early reject: ray (+x) cannot intersect triangle if all
                    // triangle vertices are at x < ray_x (-x of ray).
                    if (t.v0[0] < x && t.v1[0] < x && t.v2[0] < x) continue;
                    // Or all triangle y far from ray y / all z far from ray z.
                    const float yt_lo = std::min({t.v0[1], t.v1[1], t.v2[1]});
                    const float yt_hi = std::max({t.v0[1], t.v1[1], t.v2[1]});
                    if (y < yt_lo || y > yt_hi) continue;
                    const float zt_lo = std::min({t.v0[2], t.v1[2], t.v2[2]});
                    const float zt_hi = std::max({t.v0[2], t.v1[2], t.v2[2]});
                    if (z < zt_lo || z > zt_hi) continue;
                    float t_param;
                    if (stl_detail::ray_tri_intersect(O, D, t.v0, t.v1, t.v2, t_param)) {
                        ++n_hit;
                    }
                }
                if (n_hit & 1) {
                    const int id = i + j * nx + k * nx * ny;
                    mask[id] = lbm::core::Streaming::CELL_SOLID;
                }
            }
        }
    }
}

}  // namespace aero
}  // namespace physics
}  // namespace lbm
