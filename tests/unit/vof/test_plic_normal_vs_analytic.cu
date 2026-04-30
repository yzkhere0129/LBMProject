/**
 * @file test_plic_normal_vs_analytic.cu
 * @brief Phase 1 PLIC upgrade — verify Youngs PLIC normals against analytic
 *        reference (planar + spherical interfaces).
 *
 * Tests:
 *   1. Planar interface (z = const): n̂ should be ≈ ±ẑ at all interface cells
 *   2. Spherical droplet: n̂ should align with radial direction
 *
 * Both tests run through the new public Phase 1 API:
 *   vof.recomputePLICReconstruction();
 *   InterfaceGeometryView v = vof.getInterfaceGeometry();
 *
 * Tolerance: dot(n_plic, n_analytic) > 0.9 for sphere (Youngs has O(1/R)
 * truncation), > 0.95 for plane (essentially exact on flat front).
 */

#include <gtest/gtest.h>
#include "physics/vof_solver.h"
#include "physics/interface_geometry.h"
#include <cuda_runtime.h>
#include <vector>
#include <cmath>

using namespace lbm::physics;

namespace {

// Helper: copy three device-pointer arrays back to host.
void copyNormalsToHost(const InterfaceGeometryView& v,
                      std::vector<float>& nx,
                      std::vector<float>& ny,
                      std::vector<float>& nz)
{
    nx.resize(v.n);
    ny.resize(v.n);
    nz.resize(v.n);
    cudaMemcpy(nx.data(), v.d_normal_x, v.n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(ny.data(), v.d_normal_y, v.n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(nz.data(), v.d_normal_z, v.n * sizeof(float), cudaMemcpyDeviceToHost);
}

}  // namespace

// ============================================================================
// Test 1A: Planar interface — horizontal plane at z = nz/2.
// Expected n̂ ≈ (0, 0, +1) on the gas side or (0, 0, -1) — VOFSolver convention
// is that n̂ points from liquid to gas. With f = 1 below (small k) and f = 0
// above (large k), n̂ should point in +z.
// ============================================================================
TEST(PLICNormalVsAnalytic, HorizontalPlane) {
    const int nx = 32, ny = 32, nz = 16;
    const float dx = 1.0f;

    VOFSolver vof(nx, ny, nz, dx,
                  VOFSolver::BoundaryType::PERIODIC,
                  VOFSolver::BoundaryType::PERIODIC,
                  VOFSolver::BoundaryType::WALL);

    // Initialize: f = 1 below k=8, f = 0 above. Two interface cells at k=7,8.
    std::vector<float> h_fill(nx * ny * nz);
    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                int idx = i + nx * (j + ny * k);
                // Linear ramp through one cell — keeps things smooth and Youngs
                // can resolve a clean gradient.
                if (k <= 7) h_fill[idx] = 1.0f;
                else if (k >= 9) h_fill[idx] = 0.0f;
                else h_fill[idx] = 0.5f;  // single-cell-thick interface
            }
        }
    }
    vof.initialize(h_fill.data());

    // Phase 1 API: explicit recompute → fresh PLIC cache.
    EXPECT_FALSE(vof.isPLICReady())
        << "Solver should be dirty immediately after initialize()";
    vof.recomputePLICReconstruction();
    EXPECT_TRUE(vof.isPLICReady())
        << "Recompute should clear the dirty flag";

    auto v = vof.getInterfaceGeometry();
    EXPECT_TRUE(v.plic_ready);
    ASSERT_NE(v.d_normal_x, nullptr);
    ASSERT_NE(v.d_alpha,    nullptr);
    EXPECT_EQ(v.nx, nx);
    EXPECT_EQ(v.ny, ny);
    EXPECT_EQ(v.nz, nz);

    std::vector<float> nx_h, ny_h, nz_h;
    copyNormalsToHost(v, nx_h, ny_h, nz_h);

    // Sample interface cells at k=8 in the middle of the domain (away from
    // edges to avoid boundary effects).
    int n_checked = 0;
    for (int j = 8; j < ny - 8; ++j) {
        for (int i = 8; i < nx - 8; ++i) {
            int idx = i + nx * (j + ny * 8);
            float n_x = nx_h[idx];
            float n_y = ny_h[idx];
            float n_z = nz_h[idx];
            float mag = std::sqrt(n_x*n_x + n_y*n_y + n_z*n_z);
            ASSERT_GT(mag, 0.5f) << "Normal must be non-trivial";
            // Unit vector check
            EXPECT_NEAR(mag, 1.0f, 0.01f);
            // n̂ ≈ +ẑ
            EXPECT_NEAR(n_z, 1.0f, 0.05f) << "(i,j)=(" << i << "," << j << ")";
            EXPECT_NEAR(n_x, 0.0f, 0.05f);
            EXPECT_NEAR(n_y, 0.0f, 0.05f);
            ++n_checked;
        }
    }
    EXPECT_GT(n_checked, 100) << "Need a representative sample";
}

// ============================================================================
// Test 1B: Spherical droplet — n̂ should align with the radial direction.
// VOFSolver convention: n̂ points from liquid (inside the sphere) to gas
// (outside), i.e. n̂ ≈ +(r̂) for a liquid sphere in gas.
// ============================================================================
TEST(PLICNormalVsAnalytic, SphericalDroplet) {
    const int nx = 32, ny = 32, nz = 32;
    const float dx = 1.0f;

    VOFSolver vof(nx, ny, nz, dx);  // periodic on all sides

    const float cx = nx / 2.0f;
    const float cy = ny / 2.0f;
    const float cz = nz / 2.0f;
    const float radius = 8.0f;

    vof.initializeDroplet(cx, cy, cz, radius);
    vof.recomputePLICReconstruction();

    auto v = vof.getInterfaceGeometry();
    ASSERT_TRUE(v.plic_ready);

    std::vector<float> nx_h, ny_h, nz_h, fill_h;
    copyNormalsToHost(v, nx_h, ny_h, nz_h);
    fill_h.resize(v.n);
    cudaMemcpy(fill_h.data(), v.d_fill, v.n * sizeof(float), cudaMemcpyDeviceToHost);

    // For each interface cell (0.05 < f < 0.95), compute analytic n̂ and
    // compare to PLIC normal via dot product.
    int interface_cells_checked = 0;
    int well_aligned = 0;
    float min_dot = 2.0f;  // sentinel
    float total_dot = 0.0f;

    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                int idx = i + nx * (j + ny * k);
                float f = fill_h[idx];
                if (f < 0.05f || f > 0.95f) continue;

                // Cell center coordinate
                float x = i + 0.5f, y = j + 0.5f, z = k + 0.5f;
                float dx_r = x - cx, dy_r = y - cy, dz_r = z - cz;
                float r = std::sqrt(dx_r*dx_r + dy_r*dy_r + dz_r*dz_r);
                if (r < 1e-3f) continue;
                // Analytic n̂ (radial, pointing outward = liquid→gas)
                float n_an_x = dx_r / r;
                float n_an_y = dy_r / r;
                float n_an_z = dz_r / r;

                float n_p_x = nx_h[idx];
                float n_p_y = ny_h[idx];
                float n_p_z = nz_h[idx];

                float dot = n_p_x * n_an_x + n_p_y * n_an_y + n_p_z * n_an_z;
                total_dot += dot;
                min_dot = std::min(min_dot, dot);
                if (dot > 0.9f) ++well_aligned;
                ++interface_cells_checked;
            }
        }
    }

    ASSERT_GT(interface_cells_checked, 50)
        << "Sphere should produce ~few hundred interface cells";

    float mean_dot = total_dot / interface_cells_checked;
    // 90% of interface cells should align well with analytic radial direction.
    float well_aligned_frac =
        static_cast<float>(well_aligned) / interface_cells_checked;

    printf("[PLIC NORMAL] sphere: interface_cells=%d, mean_dot=%.4f, "
           "min_dot=%.4f, well_aligned_frac=%.3f\n",
           interface_cells_checked, mean_dot, min_dot, well_aligned_frac);

    EXPECT_GT(mean_dot, 0.95f)
        << "Mean alignment should be >0.95 on R=8 sphere (Youngs O(1/R))";
    EXPECT_GT(well_aligned_frac, 0.85f)
        << ">85% of interface cells should pass dot>0.9 alignment";
}

// ============================================================================
// Test 1C: Empty solver — getInterfaceGeometry() must NOT crash before any
// reconstruction has been run. plic_ready should be false.
// ============================================================================
TEST(PLICNormalVsAnalytic, EmptySolverHasReadyFalse) {
    VOFSolver vof(16, 16, 16);
    auto v = vof.getInterfaceGeometry();
    EXPECT_FALSE(v.plic_ready)
        << "Brand-new solver must report plic_ready=false";
    EXPECT_EQ(v.nx, 16);
    EXPECT_NE(v.d_fill, nullptr)
        << "fill_level is always allocated, even before PLIC use";
}
