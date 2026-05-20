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
#include "io/triangle_bvh.h"
#include "core/streaming.h"
#include "core/lattice_d3q27.h"
#include "physics/aero/obstacle_geometry.h"  // SparseQFraction
#include <vector>
#include <array>
#include <cmath>
#include <algorithm>

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
 * @brief Stamp solid cells from STL mesh into mask, BVH-accelerated.
 *
 * Ray-casts in +x direction from each cell centre. Cells with odd
 * intersection count are marked CELL_SOLID.
 *
 * D2 upgrade: BVH (triangle_bvh.h) reduces per-cell cost from O(N_tris)
 * to ~O(log N_tris) average. For 100k tris × 1M cells in bbox, brute
 * force = 1e11 ops (15 min); BVH ~ few seconds.
 *
 * If `bvh == nullptr`, falls back to brute-force (kept for safety; the
 * caller usually passes a pre-built BVH).
 */
inline void stampSTL(std::vector<unsigned char>& mask,
                     int nx, int ny, int nz, float dx,
                     const lbm::io::STLMesh& mesh,
                     const lbm::io::TriangleBVH* bvh = nullptr)
{
    const auto& tris = mesh.tris;
    const auto& bbox_lo = mesh.bbox_lo;
    const auto& bbox_hi = mesh.bbox_hi;

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
                auto test_tri = [&](int t_idx) {
                    const auto& t = tris[t_idx];
                    if (t.v0[0] < x && t.v1[0] < x && t.v2[0] < x) return;
                    float t_param;
                    if (stl_detail::ray_tri_intersect(O, D, t.v0, t.v1, t.v2, t_param)) {
                        ++n_hit;
                    }
                };
                if (bvh) {
                    bvh->traverse_ray(O, D, test_tri);
                } else {
                    for (int ti = 0; ti < (int)tris.size(); ++ti) test_tri(ti);
                }
                if (n_hit & 1) {
                    const int id = i + j * nx + k * nx * ny;
                    mask[id] = lbm::core::Streaming::CELL_SOLID;
                }
            }
        }
    }
}

/**
 * @brief Compute sparse qfrac for a stamped STL mask.
 *
 * For each fluid cell with at least one solid neighbour (CSR pass 1),
 * for each q ∈ [1, 27) where neighbor IS solid, cast a ray from the
 * fluid cell centre along direction c_q (unit length) and find the
 * nearest triangle intersection. qfrac = t_intersect / link_length.
 *
 * Direct ray-tri intersection (no bisection) — more accurate and faster
 * than the bisection approach used by makeNacaQFractionSparse(). BVH
 * acceleration mandatory; brute force gets pathological at 10k+ tris.
 *
 * Fallback: if no triangle intersected by the link (rare — happens when
 * the link grazes a triangle edge), qfrac = 0.5 (halfway BB).
 */
inline lbm::physics::aero::SparseQFraction makeSTLQFractionSparse(
    const std::vector<unsigned char>& mask,
    int nx, int ny, int nz, float dx,
    const lbm::io::STLMesh& mesh,
    const lbm::io::TriangleBVH& bvh)
{
    using lbm::core::D3Q27;
    constexpr float QMIN = 1e-6f;
    const size_t n_cells = static_cast<size_t>(nx) * ny * nz;
    const auto& tris = mesh.tris;

    // Pre-compute unit direction + link length (physical) per q.
    std::array<std::array<float, 3>, D3Q27::Q> u_q;
    std::array<float, D3Q27::Q> link_len;
    for (int q = 0; q < D3Q27::Q; ++q) {
        const float ex = D3Q27::h_ex[q];
        const float ey = D3Q27::h_ey[q];
        const float ez = D3Q27::h_ez[q];
        const float L = std::sqrt(ex*ex + ey*ey + ez*ez);
        link_len[q] = L * dx;
        if (L > 1e-20f) {
            u_q[q] = { ex / L, ey / L, ez / L };
        } else {
            u_q[q] = { 0.0f, 0.0f, 0.0f };  // q=0 rest particle, never queried
        }
    }

    // Pass 1: count solid-neighbour links per fluid cell.
    lbm::physics::aero::SparseQFraction out;
    out.offset.assign(n_cells + 1, 0);
    for (int k = 0; k < nz; ++k)
        for (int j = 0; j < ny; ++j)
            for (int i = 0; i < nx; ++i) {
                const size_t id = static_cast<size_t>(i)
                                + static_cast<size_t>(j) * nx
                                + static_cast<size_t>(k) * nx * ny;
                if (mask[id] != lbm::core::Streaming::CELL_FLUID) continue;
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
                    if (mask[nid] == lbm::core::Streaming::CELL_FLUID) continue;
                    ++cnt;
                }
                out.offset[id + 1] = cnt;
            }
    for (size_t i = 1; i <= n_cells; ++i) out.offset[i] += out.offset[i-1];
    const int K = out.offset[n_cells];
    out.link_q.assign(K, 0);
    out.link_qfrac.assign(K, 1.0f);

    // Pass 2: fill qfrac via BVH-accelerated ray-tri intersection.
    std::vector<int> head = out.offset;
    for (int k = 0; k < nz; ++k)
        for (int j = 0; j < ny; ++j)
            for (int i = 0; i < nx; ++i) {
                const size_t id = static_cast<size_t>(i)
                                + static_cast<size_t>(j) * nx
                                + static_cast<size_t>(k) * nx * ny;
                if (mask[id] != lbm::core::Streaming::CELL_FLUID) continue;
                const float xC = (i + 0.5f) * dx;
                const float yC = (j + 0.5f) * dx;
                const float zC = (k + 0.5f) * dx;
                for (int q = 1; q < D3Q27::Q; ++q) {
                    const int ni = i + D3Q27::h_ex[q];
                    const int nj = j + D3Q27::h_ey[q];
                    const int nk = k + D3Q27::h_ez[q];
                    if (ni < 0 || ni >= nx || nj < 0 || nj >= ny ||
                        nk < 0 || nk >= nz) continue;
                    const size_t nid = static_cast<size_t>(ni)
                                     + static_cast<size_t>(nj) * nx
                                     + static_cast<size_t>(nk) * nx * ny;
                    if (mask[nid] == lbm::core::Streaming::CELL_FLUID) continue;

                    // BVH traversal: collect min positive t over candidate tris.
                    const float O[3] = { xC, yC, zC };
                    const float D[3] = { u_q[q][0], u_q[q][1], u_q[q][2] };
                    const float t_max = link_len[q];
                    float t_best = 1e30f;
                    auto cb = [&](int t_idx) {
                        const auto& tri = tris[t_idx];
                        float t_param;
                        if (stl_detail::ray_tri_intersect(O, D, tri.v0, tri.v1, tri.v2, t_param)) {
                            if (t_param > 0.0f && t_param < t_best) t_best = t_param;
                        }
                    };
                    bvh.traverse_ray(O, D, cb);

                    float qf;
                    if (t_best >= 1e29f || t_best > t_max * 1.01f) {
                        // No hit, or hit past solid cell centre — grazing.
                        qf = 0.5f;
                    } else {
                        qf = std::clamp(t_best / t_max, QMIN, 1.0f);
                    }
                    const int slot = head[id]++;
                    out.link_q[slot]     = static_cast<unsigned char>(q);
                    out.link_qfrac[slot] = qf;
                }
            }
    return out;
}

}  // namespace aero
}  // namespace physics
}  // namespace lbm
