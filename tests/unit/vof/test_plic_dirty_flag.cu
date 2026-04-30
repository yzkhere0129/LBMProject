/**
 * @file test_plic_dirty_flag.cu
 * @brief Phase 1 PLIC upgrade — verify dirty-flag invalidation contract.
 *
 * Design choice (per Phase 1 review): EXPLICIT recompute, not auto-recompute.
 *   - getInterfaceGeometry() returns plic_ready=false when dirty
 *   - Caller must call recomputePLICReconstruction() to refresh
 *   - In release builds, stale data may still be read (we don't crash)
 *   - In debug builds, callers should check plic_ready before dereferencing
 *
 * Tests verify:
 *   1. Brand-new solver → plic_ready=false
 *   2. After recompute → plic_ready=true
 *   3. After initialize (re-write fill) → plic_ready=false again
 *   4. After advectFillLevel (with PLIC scheme) → plic_ready=true (auto)
 *   5. After applyEvaporationMassLoss → plic_ready=false
 *   6. recompute is idempotent (no-op when already clean)
 */

#include <gtest/gtest.h>
#include "physics/vof_solver.h"
#include "physics/interface_geometry.h"
#include <cuda_runtime.h>
#include <vector>
#include <cmath>

using namespace lbm::physics;

// Helper: read alpha at one cell (device → host). Returns NaN-safe float.
static float readAlphaAt(const InterfaceGeometryView& v, int idx) {
    if (!v.plic_ready || v.d_alpha == nullptr) return std::nanf("");
    float h = 0.0f;
    cudaMemcpy(&h, v.d_alpha + idx, sizeof(float), cudaMemcpyDeviceToHost);
    return h;
}

TEST(PLICDirtyFlag, NewSolverIsDirty) {
    VOFSolver vof(16, 16, 16);
    EXPECT_FALSE(vof.isPLICReady())
        << "Brand-new VOFSolver must be dirty (no reconstruction has run)";
    auto v = vof.getInterfaceGeometry();
    EXPECT_FALSE(v.plic_ready);
}

TEST(PLICDirtyFlag, RecomputeClearsFlag) {
    VOFSolver vof(16, 16, 16);
    std::vector<float> fill(16*16*16);
    for (int idx = 0; idx < 16*16*16; ++idx) {
        int k = idx / (16*16);
        if (k <= 7) fill[idx] = 1.0f;
        else if (k >= 9) fill[idx] = 0.0f;
        else fill[idx] = 0.5f;
    }
    vof.initialize(fill.data());
    EXPECT_FALSE(vof.isPLICReady());

    vof.recomputePLICReconstruction();
    EXPECT_TRUE(vof.isPLICReady());

    auto v = vof.getInterfaceGeometry();
    EXPECT_TRUE(v.plic_ready);
}

TEST(PLICDirtyFlag, ReinitializeMakesDirty) {
    const int N = 16*16*16;
    VOFSolver vof(16, 16, 16);

    std::vector<float> fill(N, 0.0f);
    for (int idx = 0; idx < N; ++idx) {
        int k = idx / (16*16);
        fill[idx] = (k <= 7) ? 1.0f : ((k >= 9) ? 0.0f : 0.5f);
    }
    vof.initialize(fill.data());
    vof.recomputePLICReconstruction();
    ASSERT_TRUE(vof.isPLICReady());

    // Re-initialize with a different fill → dirty flag must fire.
    std::vector<float> fill2(N, 0.0f);
    for (int idx = 0; idx < N; ++idx) {
        int k = idx / (16*16);
        fill2[idx] = (k <= 5) ? 1.0f : ((k >= 7) ? 0.0f : 0.5f);
    }
    vof.initialize(fill2.data());
    EXPECT_FALSE(vof.isPLICReady())
        << "initialize() must set plic_dirty_=true";

    auto v_stale = vof.getInterfaceGeometry();
    EXPECT_FALSE(v_stale.plic_ready);
}

TEST(PLICDirtyFlag, EvaporationMakesDirty) {
    const int nx = 8, ny = 8, nz = 8;
    const int N = nx * ny * nz;
    VOFSolver vof(nx, ny, nz);

    std::vector<float> fill(N, 0.0f);
    for (int idx = 0; idx < N; ++idx) {
        int k = idx / (nx*ny);
        fill[idx] = (k <= 3) ? 1.0f : ((k >= 5) ? 0.0f : 0.5f);
    }
    vof.initialize(fill.data());
    vof.recomputePLICReconstruction();
    ASSERT_TRUE(vof.isPLICReady());

    // Allocate a J_evap field (positive on the top layer to drive evaporation).
    float* d_J = nullptr;
    cudaMalloc(&d_J, N * sizeof(float));
    cudaMemset(d_J, 0, N * sizeof(float));

    vof.applyEvaporationMassLoss(d_J, /*rho=*/8000.0f, /*dt=*/1e-7f);
    EXPECT_FALSE(vof.isPLICReady())
        << "applyEvaporationMassLoss() must set plic_dirty_=true";

    cudaFree(d_J);
}

TEST(PLICDirtyFlag, RecomputeIdempotent) {
    VOFSolver vof(16, 16, 16);
    std::vector<float> fill(16*16*16, 0.0f);
    for (int idx = 0; idx < 16*16*16; ++idx) {
        int k = idx / (16*16);
        fill[idx] = (k <= 7) ? 1.0f : ((k >= 9) ? 0.0f : 0.5f);
    }
    vof.initialize(fill.data());
    vof.recomputePLICReconstruction();
    ASSERT_TRUE(vof.isPLICReady());

    // Snapshot alpha at a known interface cell.
    auto v1 = vof.getInterfaceGeometry();
    int probe = 8 + 16 * (8 + 16 * 8);
    float alpha1 = readAlphaAt(v1, probe);

    // Calling recompute again should not change anything (no-op when clean).
    vof.recomputePLICReconstruction();
    EXPECT_TRUE(vof.isPLICReady());
    auto v2 = vof.getInterfaceGeometry();
    float alpha2 = readAlphaAt(v2, probe);

    EXPECT_FLOAT_EQ(alpha1, alpha2)
        << "Idempotent recompute must produce bitwise identical output";
}

TEST(PLICDirtyFlag, PLICAdvectionLeavesCacheClean) {
    const int nx = 16, ny = 16, nz = 16;
    const int N = nx * ny * nz;
    VOFSolver vof(nx, ny, nz);
    vof.setAdvectionScheme(VOFAdvectionScheme::PLIC);

    std::vector<float> fill(N, 0.0f);
    for (int idx = 0; idx < N; ++idx) {
        int k = idx / (nx*ny);
        fill[idx] = (k <= 7) ? 1.0f : ((k >= 9) ? 0.0f : 0.5f);
    }
    vof.initialize(fill.data());

    // Allocate small uniform velocity field (one cell-per-step in x).
    float* d_ux = nullptr;
    float* d_uy = nullptr;
    float* d_uz = nullptr;
    cudaMalloc(&d_ux, N * sizeof(float));
    cudaMalloc(&d_uy, N * sizeof(float));
    cudaMalloc(&d_uz, N * sizeof(float));
    std::vector<float> ux(N, 0.05f), uy(N, 0.0f), uz(N, 0.0f);
    cudaMemcpy(d_ux, ux.data(), N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_uy, uy.data(), N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_uz, uz.data(), N*sizeof(float), cudaMemcpyHostToDevice);

    EXPECT_FALSE(vof.isPLICReady());
    vof.advectFillLevel(d_ux, d_uy, d_uz, /*dt=*/1.0f);
    EXPECT_TRUE(vof.isPLICReady())
        << "PLIC advection should leave the cache fresh "
        << "(advectFillLevelPLIC tail calls recomputePLICReconstruction)";

    cudaFree(d_ux); cudaFree(d_uy); cudaFree(d_uz);
}
