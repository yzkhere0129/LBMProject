/**
 * @file triangle_bvh.h
 * @brief Median-split AABB BVH for a list of triangles.
 *
 * Built once on host after STL load; queried per cell during stamp.
 * Top-down recursive split by longest-axis median of triangle centroids.
 * Leaf nodes hold up to LEAF_MAX triangle indices.
 *
 * Query operations supported:
 *   - count_ray_intersections(O, D): even-odd inside-outside test
 *   - find_closest_triangle(P): nearest-triangle for qfrac / normal (D3)
 *
 * Cost: build O(N log N), each ray query O(log N) average, O(N) worst-case.
 * Memory: 64 B/node + 4 B per leaf-triangle-index = ~76 B/tri after build.
 *
 * Header-only because it's used by stl_geometry.h and there's only one
 * client. If shared, move impl to .cpp.
 */

#pragma once

#include "io/stl_reader.h"
#include <algorithm>
#include <array>
#include <cstdint>
#include <numeric>
#include <vector>

namespace lbm {
namespace io {

class TriangleBVH {
public:
    static constexpr int LEAF_MAX = 8;

    struct Node {
        std::array<float, 3> bbox_lo;
        std::array<float, 3> bbox_hi;
        int  left  = -1;  // left child index, or -1 if leaf
        int  right = -1;  // right child index, or -1 if leaf
        int  tri_first = 0;  // for leaves: first index into tri_indices_
        int  tri_count = 0;  // for leaves: how many; 0 means internal node
    };

    /// Build BVH over `mesh.tris`. Triangles are NOT copied — indices only.
    void build(const STLMesh& mesh) {
        const int n_tri = (int)mesh.tris.size();
        tri_indices_.resize(n_tri);
        std::iota(tri_indices_.begin(), tri_indices_.end(), 0);

        // Pre-compute centroids for fast median sort.
        centroids_.resize(n_tri);
        for (int i = 0; i < n_tri; ++i) {
            const auto& t = mesh.tris[i];
            for (int d = 0; d < 3; ++d) {
                centroids_[i][d] = (t.v0[d] + t.v1[d] + t.v2[d]) / 3.0f;
            }
        }

        mesh_ = &mesh;
        nodes_.clear();
        nodes_.reserve(2 * n_tri);  // upper bound
        build_recursive(0, n_tri);
        mesh_ = nullptr;  // BVH stores no mesh pointer post-build
    }

    /// Stack-based traversal: visit all leaves whose AABB contains the ray,
    /// then call `cb(tri_idx)` for each candidate triangle.
    /// Ray is (O, D) with D unit length.
    template<class CB>
    void traverse_ray(const float O[3], const float D[3], CB cb) const {
        if (nodes_.empty()) return;
        constexpr int STACK_MAX = 64;
        int stack[STACK_MAX];
        int sp = 0;
        stack[sp++] = 0;
        while (sp > 0) {
            const int idx = stack[--sp];
            const Node& nd = nodes_[idx];
            if (!ray_aabb_test(O, D, nd.bbox_lo, nd.bbox_hi)) continue;
            if (nd.tri_count > 0) {
                // Leaf
                for (int k = 0; k < nd.tri_count; ++k) {
                    cb(tri_indices_[nd.tri_first + k]);
                }
            } else {
                if (sp + 2 <= STACK_MAX) {
                    stack[sp++] = nd.left;
                    stack[sp++] = nd.right;
                }
            }
        }
    }

    /// Find nearest-triangle index + distance from point P. Returns -1 if empty.
    /// Used by D3 qfrac computation. CB signature: float(int tri_idx, P[3]) → squared distance.
    template<class DistFn>
    int find_nearest(const float P[3], DistFn dist_fn, float& dist2_out) const {
        if (nodes_.empty()) { dist2_out = 1e30f; return -1; }
        constexpr int STACK_MAX = 64;
        int stack[STACK_MAX];
        int sp = 0;
        stack[sp++] = 0;
        float best_d2 = 1e30f;
        int   best_idx = -1;
        while (sp > 0) {
            const int idx = stack[--sp];
            const Node& nd = nodes_[idx];
            // Prune: if node's AABB-min-distance > best, skip.
            const float dmin = aabb_min_dist_sq(P, nd.bbox_lo, nd.bbox_hi);
            if (dmin >= best_d2) continue;
            if (nd.tri_count > 0) {
                for (int k = 0; k < nd.tri_count; ++k) {
                    const int t_idx = tri_indices_[nd.tri_first + k];
                    const float d2 = dist_fn(t_idx, P);
                    if (d2 < best_d2) { best_d2 = d2; best_idx = t_idx; }
                }
            } else {
                if (sp + 2 <= STACK_MAX) {
                    stack[sp++] = nd.left;
                    stack[sp++] = nd.right;
                }
            }
        }
        dist2_out = best_d2;
        return best_idx;
    }

    int n_nodes() const { return (int)nodes_.size(); }
    int n_tris()  const { return (int)tri_indices_.size(); }

private:
    std::vector<Node>  nodes_;
    std::vector<int>   tri_indices_;
    std::vector<std::array<float, 3>>  centroids_;
    const STLMesh* mesh_ = nullptr;  // only valid during build()

    void build_recursive(int first, int count) {
        const int node_idx = (int)nodes_.size();
        nodes_.emplace_back();
        Node& nd = nodes_[node_idx];

        // Compute AABB over the included triangles.
        nd.bbox_lo = { 1e30f, 1e30f, 1e30f };
        nd.bbox_hi = {-1e30f,-1e30f,-1e30f };
        for (int i = first; i < first + count; ++i) {
            const auto& tri = mesh_->tris[tri_indices_[i]];
            for (int d = 0; d < 3; ++d) {
                nd.bbox_lo[d] = std::min({nd.bbox_lo[d], tri.v0[d], tri.v1[d], tri.v2[d]});
                nd.bbox_hi[d] = std::max({nd.bbox_hi[d], tri.v0[d], tri.v1[d], tri.v2[d]});
            }
        }

        if (count <= LEAF_MAX) {
            nd.tri_first = first;
            nd.tri_count = count;
            return;
        }

        // Pick longest axis.
        int axis = 0;
        float max_ext = nd.bbox_hi[0] - nd.bbox_lo[0];
        if ((nd.bbox_hi[1] - nd.bbox_lo[1]) > max_ext) { axis = 1; max_ext = nd.bbox_hi[1] - nd.bbox_lo[1]; }
        if ((nd.bbox_hi[2] - nd.bbox_lo[2]) > max_ext) { axis = 2; max_ext = nd.bbox_hi[2] - nd.bbox_lo[2]; }

        // Median-split by centroid on chosen axis.
        const int mid = first + count / 2;
        std::nth_element(
            tri_indices_.begin() + first,
            tri_indices_.begin() + mid,
            tri_indices_.begin() + first + count,
            [this, axis](int a, int b) { return centroids_[a][axis] < centroids_[b][axis]; }
        );

        // Recursively build children. Note: emplace_back into nodes_ may
        // invalidate `nd` reference, so re-index after each call.
        const int left  = node_idx + 1;  // reserved next slot
        build_recursive(first, mid - first);
        const int right = (int)nodes_.size();
        build_recursive(mid, count - (mid - first));
        nodes_[node_idx].left  = left;
        nodes_[node_idx].right = right;
        nodes_[node_idx].tri_first = 0;
        nodes_[node_idx].tri_count = 0;
    }

    static inline bool ray_aabb_test(const float O[3], const float D[3],
                                     const std::array<float, 3>& lo,
                                     const std::array<float, 3>& hi)
    {
        // Slab method. Returns true if ray (O, D, t > 0) intersects AABB.
        float tmin = 0.0f, tmax = 1e30f;
        for (int d = 0; d < 3; ++d) {
            if (std::abs(D[d]) < 1e-20f) {
                if (O[d] < lo[d] || O[d] > hi[d]) return false;
                continue;
            }
            const float inv = 1.0f / D[d];
            float t1 = (lo[d] - O[d]) * inv;
            float t2 = (hi[d] - O[d]) * inv;
            if (t1 > t2) std::swap(t1, t2);
            if (t1 > tmin) tmin = t1;
            if (t2 < tmax) tmax = t2;
            if (tmin > tmax) return false;
        }
        return tmax >= 0.0f;
    }

    static inline float aabb_min_dist_sq(const float P[3],
                                         const std::array<float, 3>& lo,
                                         const std::array<float, 3>& hi)
    {
        float d2 = 0.0f;
        for (int d = 0; d < 3; ++d) {
            const float v = P[d];
            if      (v < lo[d]) { const float dx = lo[d] - v; d2 += dx * dx; }
            else if (v > hi[d]) { const float dx = v - hi[d]; d2 += dx * dx; }
        }
        return d2;
    }
};

}  // namespace io
}  // namespace lbm
