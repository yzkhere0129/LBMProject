/**
 * @file stl_reader.h
 * @brief ASCII + Binary STL parser → vector of triangles + bbox.
 *
 * Format references:
 *   ASCII:  "solid <name>" / "facet normal nx ny nz" / "outer loop" /
 *           "vertex x y z" × 3 / "endloop" / "endfacet" / "endsolid"
 *   Binary: 80-byte header (any content)
 *           uint32_le n_triangles
 *           n_triangles × {
 *               float[3] normal     // 12 bytes
 *               float[3] v0, v1, v2 // 36 bytes
 *               uint16  attribute   //  2 bytes
 *           } = 50 bytes per triangle
 *
 * Auto-detection: read first 5 bytes; "solid" prefix + further ASCII
 * sanity check (probe for "facet" within first 256 bytes) → ASCII.
 * Otherwise → Binary. (Some buggy binary files start with "solid"; the
 * facet-token probe disambiguates.)
 *
 * Header-only because total LOC < 200 and the parser is used in exactly
 * one place (aero driver). If reused elsewhere later, move impl to .cpp.
 */

#pragma once

#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace lbm {
namespace io {

struct Triangle {
    std::array<float, 3> v0;
    std::array<float, 3> v1;
    std::array<float, 3> v2;
    std::array<float, 3> normal;  // unit outward normal (renormalized from STL value)
};

struct STLMesh {
    std::vector<Triangle> tris;
    std::array<float, 3> bbox_lo{ {  1e30f,  1e30f,  1e30f} };
    std::array<float, 3> bbox_hi{ { -1e30f, -1e30f, -1e30f} };

    inline std::array<float, 3> centre() const {
        return { 0.5f * (bbox_lo[0] + bbox_hi[0]),
                 0.5f * (bbox_lo[1] + bbox_hi[1]),
                 0.5f * (bbox_lo[2] + bbox_hi[2]) };
    }
    inline std::array<float, 3> extent() const {
        return { bbox_hi[0] - bbox_lo[0],
                 bbox_hi[1] - bbox_lo[1],
                 bbox_hi[2] - bbox_lo[2] };
    }
};

namespace detail {

inline void update_bbox(STLMesh& m, const std::array<float, 3>& v) {
    for (int d = 0; d < 3; ++d) {
        if (v[d] < m.bbox_lo[d]) m.bbox_lo[d] = v[d];
        if (v[d] > m.bbox_hi[d]) m.bbox_hi[d] = v[d];
    }
}

inline void renormalize_normal(Triangle& t) {
    // STL "normal" is sometimes (0,0,0) or non-unit. Recompute from CCW.
    const float ax = t.v1[0] - t.v0[0], ay = t.v1[1] - t.v0[1], az = t.v1[2] - t.v0[2];
    const float bx = t.v2[0] - t.v0[0], by = t.v2[1] - t.v0[1], bz = t.v2[2] - t.v0[2];
    float nx = ay*bz - az*by, ny = az*bx - ax*bz, nz = ax*by - ay*bx;
    const float len = std::sqrt(nx*nx + ny*ny + nz*nz);
    if (len > 1e-20f) {
        t.normal[0] = nx / len;
        t.normal[1] = ny / len;
        t.normal[2] = nz / len;
    } else {
        // Degenerate triangle — keep STL value if it was unit-ish, else (0,0,1).
        const float L2 = t.normal[0]*t.normal[0] + t.normal[1]*t.normal[1] + t.normal[2]*t.normal[2];
        if (L2 < 0.5f || L2 > 2.0f) { t.normal = {0.f, 0.f, 1.f}; }
    }
}

inline bool sniff_ascii(std::ifstream& f) {
    // Read first 256 bytes (text would still work in binary mode).
    char buf[256] = {0};
    f.read(buf, sizeof(buf));
    const std::streamsize n = f.gcount();
    f.clear();
    f.seekg(0, std::ios::beg);
    // Must start with "solid" and contain "facet" within first 256 bytes.
    if (n < 5) return false;
    if (std::strncmp(buf, "solid", 5) != 0) return false;
    const std::string head(buf, buf + n);
    return head.find("facet") != std::string::npos;
}

inline STLMesh load_ascii(std::ifstream& f) {
    STLMesh m;
    std::string tok;
    Triangle tri{};
    int vert_idx = 0;
    while (f >> tok) {
        if (tok == "facet") {
            // expect: facet normal nx ny nz
            std::string subtok;
            f >> subtok;  // "normal"
            f >> tri.normal[0] >> tri.normal[1] >> tri.normal[2];
            vert_idx = 0;
        } else if (tok == "vertex") {
            std::array<float, 3> v;
            f >> v[0] >> v[1] >> v[2];
            if      (vert_idx == 0) tri.v0 = v;
            else if (vert_idx == 1) tri.v1 = v;
            else if (vert_idx == 2) tri.v2 = v;
            // (defensive) malformed STL with > 3 vertices per facet: silently ignore extras
            if (vert_idx < 3) ++vert_idx;
        } else if (tok == "endfacet") {
            renormalize_normal(tri);
            update_bbox(m, tri.v0);
            update_bbox(m, tri.v1);
            update_bbox(m, tri.v2);
            m.tris.push_back(tri);
        }
    }
    return m;
}

inline STLMesh load_binary(std::ifstream& f) {
    STLMesh m;
    char header[80];
    f.read(header, 80);
    uint32_t n_tri = 0;
    f.read(reinterpret_cast<char*>(&n_tri), sizeof(uint32_t));
    m.tris.reserve(n_tri);
    for (uint32_t k = 0; k < n_tri; ++k) {
        Triangle tri{};
        float buf[12];
        f.read(reinterpret_cast<char*>(buf), 12 * sizeof(float));
        if (!f) throw std::runtime_error("STL binary: truncated triangle at " + std::to_string(k));
        tri.normal = { buf[0], buf[1], buf[2] };
        tri.v0     = { buf[3], buf[4], buf[5] };
        tri.v1     = { buf[6], buf[7], buf[8] };
        tri.v2     = { buf[9], buf[10], buf[11] };
        uint16_t attr;
        f.read(reinterpret_cast<char*>(&attr), sizeof(uint16_t));
        renormalize_normal(tri);
        update_bbox(m, tri.v0);
        update_bbox(m, tri.v1);
        update_bbox(m, tri.v2);
        m.tris.push_back(tri);
    }
    return m;
}

}  // namespace detail

/**
 * @brief Load STL file (ASCII or binary auto-detect).
 *
 * @throws std::runtime_error on I/O error or malformed file.
 */
inline STLMesh load_stl(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("STL: cannot open " + path);
    const bool is_ascii = detail::sniff_ascii(f);
    STLMesh m = is_ascii ? detail::load_ascii(f) : detail::load_binary(f);
    if (m.tris.empty()) throw std::runtime_error("STL: empty mesh from " + path);
    return m;
}

/**
 * @brief Apply uniform scale + translation in place. Useful to fit STL
 *        into the simulation domain (model is often in mm, our domain in m,
 *        and the model needs to be placed at the obstacle location).
 *
 * Order: scale-then-translate (so the translate values are in OUTPUT units).
 */
inline void transform_mesh(STLMesh& m, float scale,
                           const std::array<float, 3>& translate)
{
    m.bbox_lo = { 1e30f, 1e30f, 1e30f };
    m.bbox_hi = {-1e30f,-1e30f,-1e30f };
    for (auto& t : m.tris) {
        for (int d = 0; d < 3; ++d) {
            t.v0[d] = t.v0[d] * scale + translate[d];
            t.v1[d] = t.v1[d] * scale + translate[d];
            t.v2[d] = t.v2[d] * scale + translate[d];
        }
        detail::renormalize_normal(t);  // normal is scale-invariant in sign;
                                        // direction unchanged for uniform scale.
        detail::update_bbox(m, t.v0);
        detail::update_bbox(m, t.v1);
        detail::update_bbox(m, t.v2);
    }
}

}  // namespace io
}  // namespace lbm
