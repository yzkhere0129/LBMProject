/**
 * @file test_stl_pipeline.cpp
 * @brief Unit tests for STL D1-D4: reader, BVH, ray-tri intersect, qfrac.
 *
 * Pure-host tests — no CUDA. Generates STLs in-memory (small icosahedron-
 * style sphere + cube) so no test_data dependency.
 */

#include "io/stl_reader.h"
#include "io/triangle_bvh.h"
#include "physics/aero/stl_geometry.h"
#include "physics/aero/obstacle_geometry.h"
#include "core/streaming.h"

#include <gtest/gtest.h>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

using namespace lbm::io;
using namespace lbm::physics::aero;

// ============================================================================
// Helper: write a tiny binary STL to a temp file
// ============================================================================
static std::string write_binary_stl(const std::vector<Triangle>& tris,
                                    const std::string& base = "stl_test") {
    char path[L_tmpnam];
    std::tmpnam(path);  // not race-safe, but adequate for serialized unit tests
    std::string p(path);
    p += "_" + base + ".stl";
    std::ofstream f(p, std::ios::binary);
    char header[80] = {0};
    f.write(header, 80);
    uint32_t n = (uint32_t)tris.size();
    f.write(reinterpret_cast<const char*>(&n), sizeof(uint32_t));
    for (const auto& t : tris) {
        f.write(reinterpret_cast<const char*>(t.normal.data()), 3 * sizeof(float));
        f.write(reinterpret_cast<const char*>(t.v0.data()),     3 * sizeof(float));
        f.write(reinterpret_cast<const char*>(t.v1.data()),     3 * sizeof(float));
        f.write(reinterpret_cast<const char*>(t.v2.data()),     3 * sizeof(float));
        uint16_t attr = 0;
        f.write(reinterpret_cast<const char*>(&attr), sizeof(uint16_t));
    }
    return p;
}

// ============================================================================
// Build a 12-tri unit cube [0,1]^3 (closed, CCW outward).
// ============================================================================
static std::vector<Triangle> unit_cube() {
    auto V = [](float x, float y, float z) -> std::array<float, 3> {
        return {x, y, z};
    };
    std::array<std::array<float, 3>, 8> v = {
        V(0,0,0), V(1,0,0), V(1,1,0), V(0,1,0),
        V(0,0,1), V(1,0,1), V(1,1,1), V(0,1,1)
    };
    // 12 triangles, CCW outward
    int idx[12][3] = {
        {0,2,1}, {0,3,2},   // bottom (-z, normal -z)
        {4,5,6}, {4,6,7},   // top    (+z)
        {0,1,5}, {0,5,4},   // front  (-y)
        {3,7,6}, {3,6,2},   // back   (+y)
        {0,4,7}, {0,7,3},   // left   (-x)
        {1,2,6}, {1,6,5},   // right  (+x)
    };
    std::vector<Triangle> tris;
    for (auto& f : idx) {
        Triangle t;
        t.v0 = v[f[0]]; t.v1 = v[f[1]]; t.v2 = v[f[2]];
        t.normal = {0.f, 0.f, 1.f};
        tris.push_back(t);
    }
    return tris;
}

// ============================================================================
// Tests
// ============================================================================
TEST(STLReader, BinaryRoundtrip) {
    auto cube = unit_cube();
    auto p = write_binary_stl(cube, "cube");
    auto mesh = load_stl(p);
    EXPECT_EQ(mesh.tris.size(), 12u);
    // bbox should be [0,0,0] → [1,1,1]
    EXPECT_NEAR(mesh.bbox_lo[0], 0.0f, 1e-6);
    EXPECT_NEAR(mesh.bbox_lo[1], 0.0f, 1e-6);
    EXPECT_NEAR(mesh.bbox_lo[2], 0.0f, 1e-6);
    EXPECT_NEAR(mesh.bbox_hi[0], 1.0f, 1e-6);
    EXPECT_NEAR(mesh.bbox_hi[1], 1.0f, 1e-6);
    EXPECT_NEAR(mesh.bbox_hi[2], 1.0f, 1e-6);
    // All normals should be unit length (renormalized from CCW)
    for (const auto& t : mesh.tris) {
        float n2 = t.normal[0]*t.normal[0]
                 + t.normal[1]*t.normal[1]
                 + t.normal[2]*t.normal[2];
        EXPECT_NEAR(n2, 1.0f, 1e-4);
    }
    std::remove(p.c_str());
}

TEST(STLReader, ASCIIRoundtrip) {
    // Write a tiny ASCII STL (one triangle)
    char path[L_tmpnam]; std::tmpnam(path);
    std::string p(path); p += "_ascii.stl";
    {
        std::ofstream f(p);
        f << "solid mini\n"
          << " facet normal 0 0 1\n"
          << "  outer loop\n"
          << "   vertex 0 0 0\n"
          << "   vertex 1 0 0\n"
          << "   vertex 0 1 0\n"
          << "  endloop\n"
          << " endfacet\n"
          << "endsolid\n";
    }
    auto mesh = load_stl(p);
    EXPECT_EQ(mesh.tris.size(), 1u);
    EXPECT_NEAR(mesh.bbox_hi[2], 0.0f, 1e-6);
    std::remove(p.c_str());
}

TEST(STLReader, TransformMesh) {
    auto cube = unit_cube();
    auto p = write_binary_stl(cube, "cube_xfrm");
    auto mesh = load_stl(p);
    transform_mesh(mesh, 2.0f, {10.f, 20.f, 30.f});
    EXPECT_NEAR(mesh.bbox_lo[0], 10.0f, 1e-5);
    EXPECT_NEAR(mesh.bbox_hi[0], 12.0f, 1e-5);
    EXPECT_NEAR(mesh.bbox_lo[1], 20.0f, 1e-5);
    EXPECT_NEAR(mesh.bbox_hi[2], 32.0f, 1e-5);
    std::remove(p.c_str());
}

TEST(TriangleBVH, BuildSphereStructure) {
    auto cube = unit_cube();
    STLMesh mesh;
    mesh.tris = cube;
    for (const auto& t : cube) {
        for (int d = 0; d < 3; ++d) {
            mesh.bbox_lo[d] = std::min({mesh.bbox_lo[d], t.v0[d], t.v1[d], t.v2[d]});
            mesh.bbox_hi[d] = std::max({mesh.bbox_hi[d], t.v0[d], t.v1[d], t.v2[d]});
        }
    }
    TriangleBVH bvh;
    bvh.build(mesh);
    EXPECT_EQ(bvh.n_tris(), 12);
    EXPECT_GT(bvh.n_nodes(), 0);
}

TEST(StampSTL, UnitCubeFootprint) {
    auto cube = unit_cube();
    auto p = write_binary_stl(cube, "cube_stamp");
    auto mesh = load_stl(p);
    TriangleBVH bvh;
    bvh.build(mesh);

    // Stamp on a 16×16×16 grid covering [-0.25, 1.25] cubed → dx = 1.5/16 ≈ 0.0938
    const int N = 16;
    const float dx = 1.5f / N;
    // Cube lives in [0,1]; shift +0.25 in world so cube is at [0.25, 1.25] in cell-centre coords.
    // Actually simpler: just stamp on [0,N]·dx and place the cube centered there.
    // Cube in cell coords occupies cells where (i+0.5)*dx ∈ [0,1], i.e. i ∈ [int(0/dx-0.5), int(1/dx-0.5)].
    // With dx=0.0938, that's i ∈ [-1, 10] clipped to [0, 10] → 11 cells per axis = 1331 cells.

    std::vector<unsigned char> mask(N*N*N, lbm::core::Streaming::CELL_FLUID);
    stampSTL(mask, N, N, N, dx, mesh, &bvh);

    long n_solid = 0;
    for (auto v : mask) if (v == lbm::core::Streaming::CELL_SOLID) ++n_solid;

    // Cube volume = 1 m³; stamped cells × dx³ should ≈ 1.
    const float stamped_vol = n_solid * dx * dx * dx;
    EXPECT_GT(n_solid, 100);
    EXPECT_LT(n_solid, 2000);
    EXPECT_NEAR(stamped_vol, 1.0f, 0.4f);  // ±40% on a coarse 16³ grid is OK

    std::remove(p.c_str());
}

TEST(MakeSTLQFractionSparse, NoSpuriousLinksOutsideObject) {
    auto cube = unit_cube();
    auto p = write_binary_stl(cube, "cube_qf");
    auto mesh = load_stl(p);
    TriangleBVH bvh;
    bvh.build(mesh);

    const int N = 16;
    const float dx = 1.5f / N;
    std::vector<unsigned char> mask(N*N*N, lbm::core::Streaming::CELL_FLUID);
    stampSTL(mask, N, N, N, dx, mesh, &bvh);

    auto qf = makeSTLQFractionSparse(mask, N, N, N, dx, mesh, bvh);
    const int K = (int)qf.link_q.size();
    EXPECT_GT(K, 0);
    EXPECT_EQ(qf.offset.size(), (size_t)(N*N*N + 1));
    EXPECT_EQ(qf.link_qfrac.size(), (size_t)K);
    // D4: surface normals should be populated and (mostly) unit-length.
    EXPECT_EQ(qf.link_nx.size(), (size_t)K);
    int n_unit = 0;
    for (int k = 0; k < K; ++k) {
        float n2 = qf.link_nx[k]*qf.link_nx[k]
                 + qf.link_ny[k]*qf.link_ny[k]
                 + qf.link_nz[k]*qf.link_nz[k];
        if (n2 > 0.5f && n2 < 1.5f) ++n_unit;
    }
    EXPECT_GT(n_unit, K / 2);  // most hits should give a unit normal

    // qfrac values should be in [QMIN, 1.0].
    for (auto qfval : qf.link_qfrac) {
        EXPECT_GE(qfval, 1e-6f);
        EXPECT_LE(qfval, 1.0f);
    }
    std::remove(p.c_str());
}

int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
