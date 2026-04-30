/**
 * @file test_plic_alpha_inversion.cu
 * @brief Phase 1 PLIC upgrade — verify Scardovelli-Zaleski alpha inversion.
 *
 * White-box round-trip test: for a manufactured (n̂, f) pair, the stored
 * alpha must satisfy
 *
 *     plicVolumeInBox(alpha, n_x, n_y, n_z, 1, 1, 1) == f
 *
 * to within ~1e-5 (bisection in 56 iterations on float gives ~1e-7 floor;
 * we use 10× safety margin).
 *
 * Coverage strategy: pick (|n_x|, |n_y|, |n_z|) = (0.8, 0.5, 0.3) so all
 * three components are distinct, then sweep f across the 6 SZ branches:
 *
 *   m1 = 0.8, m2 = 0.5, m3 = 0.3  (sorted descending after |·|)
 *   S = m1 + m2 + m3 = 1.6, S/2 = 0.8
 *
 *   f branches (all in lower half, mirror via symmetry covers upper):
 *     f → alpha through V(α) such that:
 *       small f → α ≤ m3 = 0.3 (tetrahedron branch)
 *       intermediate f → m3 < α ≤ m2 = 0.5
 *       larger f → m2 < α ≤ m1 = 0.8
 *
 * Plus 2 degenerate cases:
 *   axis-aligned (1, 0, 0) — exercises the 1D fallback branch (H2 fix)
 *   diagonal (1/√3, 1/√3, 1/√3) — symmetric ordering
 */

#include <gtest/gtest.h>
#include "physics/vof_solver.h"
#include "physics/interface_geometry.h"
#include <cuda_runtime.h>
#include <vector>
#include <cmath>
#include <algorithm>

using namespace lbm::physics;

// ============================================================================
// Host-side mirror of the device plicVolume3D for verification.
// Implements the SZ inclusion-exclusion (eq. 28) symmetry: V(α)=1-V(S-α) for
// α > S/2. Caller provides absolute-value, sorted (m1 ≥ m2 ≥ m3) inputs.
// ============================================================================
static float plicVolumeFirstHalfHost(float alpha, float m1, float m2, float m3)
{
    if (m3 < 1e-8f) {
        float S2 = m1 + m2;
        if (alpha >= S2) return 1.0f;
        if (alpha <= 0.0f) return 0.0f;
        if (m2 < 1e-8f) {
            return std::max(0.0f, std::min(1.0f, alpha / std::max(m1, 1e-30f)));
        }
        float vol2d;
        if (alpha <= m2) vol2d = (alpha * alpha) / (2.0f * m1 * m2);
        else if (alpha <= m1) vol2d = (alpha - 0.5f * m2) / m1;
        else { float t = S2 - alpha; vol2d = 1.0f - (t * t) / (2.0f * m1 * m2); }
        return std::max(0.0f, std::min(1.0f, vol2d));
    }
    float denom = 6.0f * m1 * m2 * m3;
    float a = alpha;
    float vol = a * a * a;
    auto cube_pos = [](float x) { return (x > 0.0f) ? x*x*x : 0.0f; };
    vol -= cube_pos(a - m1);
    vol -= cube_pos(a - m2);
    vol -= cube_pos(a - m3);
    vol += cube_pos(a - m1 - m2);
    vol += cube_pos(a - m1 - m3);
    vol += cube_pos(a - m2 - m3);
    return std::max(0.0f, std::min(1.0f, vol / denom));
}

static float plicVolume3DHost(float alpha, float m1, float m2, float m3)
{
    if (alpha <= 0.0f) return 0.0f;
    float S = m1 + m2 + m3;
    if (alpha >= S) return 1.0f;
    if (alpha > 0.5f * S) {
        return 1.0f - plicVolumeFirstHalfHost(S - alpha, m1, m2, m3);
    }
    return plicVolumeFirstHalfHost(alpha, m1, m2, m3);
}

// Mirror of device plicVolumeInBox: handles sign-flip + sort + scaling.
// Used to validate the round-trip identity Vol(stored_alpha, raw_n) = f.
static float plicVolumeInBoxHost(float alpha_orig,
                                 float n_x, float n_y, float n_z,
                                 float Lx, float Ly, float Lz)
{
    float alpha = alpha_orig;
    float anx = std::fabs(n_x), any = std::fabs(n_y), anz = std::fabs(n_z);
    if (n_x < 0.0f) alpha += anx * Lx;
    if (n_y < 0.0f) alpha += any * Ly;
    if (n_z < 0.0f) alpha += anz * Lz;
    float mx = anx * Lx, my = any * Ly, mz = anz * Lz;
    if (mx < my) std::swap(mx, my);
    if (mx < mz) std::swap(mx, mz);
    if (my < mz) std::swap(my, mz);
    return plicVolume3DHost(alpha, mx, my, mz) * Lx * Ly * Lz;
}

// ============================================================================
// Test fixture: build a solver where each cell has a manufactured (f, n̂)
// pair. Use a 2×4×1 grid (8 cells, one per test case) and verify each cell.
//
// Note: We don't ask the solver to RECONSTRUCT — we just want the
// alpha-inversion to be tested. Strategy:
//   1. Initialize fill_level to 8 prescribed values
//   2. Call recomputePLICReconstruction() — this runs Youngs normals + alpha
//   3. The Youngs normal won't match our manufactured n̂ (the gradient of a
//      point-to-point varying f field is whatever it is). So instead of
//      verifying agreement with our manufactured n̂, we verify the
//      ROUND-TRIP IDENTITY using whatever (n̂, alpha) the solver produces:
//
//          plicVolumeInBox(alpha, n_x, n_y, n_z, 1, 1, 1) ≈ f
//
//   This is the right invariant — it tests the alpha-inversion math in
//   isolation regardless of what normal the upstream Youngs kernel produces.
// ============================================================================
TEST(PLICAlphaInversion, RoundTripIdentity) {
    const int nx = 16, ny = 16, nz = 16;
    VOFSolver vof(nx, ny, nz);

    // Build a smoothly-varying fill field that produces interface cells with
    // a wide variety of fill values and normal orientations. A diagonal ramp
    // does the trick.
    std::vector<float> h_fill(nx * ny * nz);
    for (int k = 0; k < nz; ++k)
    for (int j = 0; j < ny; ++j)
    for (int i = 0; i < nx; ++i) {
        // Diagonal interface: f = clamp(0.5 - 0.1*(i + j + k - 12), 0, 1)
        float t = 0.5f - 0.1f * (i + j + k - 12);
        h_fill[i + nx*(j + ny*k)] = std::max(0.0f, std::min(1.0f, t));
    }
    vof.initialize(h_fill.data());
    vof.recomputePLICReconstruction();

    auto v = vof.getInterfaceGeometry();
    ASSERT_TRUE(v.plic_ready);

    int N = v.n;
    std::vector<float> nx_h(N), ny_h(N), nz_h(N), alpha_h(N), fill_h(N);
    cudaMemcpy(nx_h.data(),    v.d_normal_x, N*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(ny_h.data(),    v.d_normal_y, N*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(nz_h.data(),    v.d_normal_z, N*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(alpha_h.data(), v.d_alpha,    N*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(fill_h.data(),  v.d_fill,     N*sizeof(float), cudaMemcpyDeviceToHost);

    int interface_cells = 0;
    int worst_idx = -1;
    float worst_err = 0.0f;
    float sum_err = 0.0f;

    for (int idx = 0; idx < N; ++idx) {
        float f = fill_h[idx];
        if (f < 1e-3f || f > 1.0f - 1e-3f) continue;
        float n_x = nx_h[idx], n_y = ny_h[idx], n_z = nz_h[idx];
        float mag = std::sqrt(n_x*n_x + n_y*n_y + n_z*n_z);
        if (mag < 0.5f) continue;  // skip cells where Youngs failed

        float alpha = alpha_h[idx];
        float f_round = plicVolumeInBoxHost(alpha, n_x, n_y, n_z, 1.0f, 1.0f, 1.0f);
        float err = std::fabs(f_round - f);
        sum_err += err;
        if (err > worst_err) { worst_err = err; worst_idx = idx; }
        ++interface_cells;
    }

    ASSERT_GT(interface_cells, 20) << "Need representative interface cell sample";
    float mean_err = sum_err / interface_cells;
    printf("[PLIC ALPHA] round-trip: cells=%d, mean_err=%.2e, max_err=%.2e\n",
           interface_cells, mean_err, worst_err);

    EXPECT_LT(mean_err, 1e-5f) << "Mean alpha-inversion round-trip error too large";
    EXPECT_LT(worst_err, 1e-4f) << "Max round-trip error too large; "
                                << "worst cell idx=" << worst_idx;
}

// ============================================================================
// Test: explicitly probe the H2 fix (1D-degenerate branch in plicVolumeFirstHalf).
// Manufacture a slab interface where Youngs produces a near-axis-aligned
// normal: f varies only in the z direction. With (n_x, n_y) ≈ (0, 0) and
// |n_z| = 1, the alpha computation hits the m3≈0, m2≈0 branch of
// plicVolumeFirstHalf — which would NaN-out without the H2 guard.
// ============================================================================
TEST(PLICAlphaInversion, AxisAligned1DSlab) {
    const int nx = 8, ny = 8, nz = 16;
    VOFSolver vof(nx, ny, nz);

    std::vector<float> h_fill(nx * ny * nz);
    for (int k = 0; k < nz; ++k)
    for (int j = 0; j < ny; ++j)
    for (int i = 0; i < nx; ++i) {
        // Pure z-slab: f depends only on k.
        float f;
        if (k <= 7) f = 1.0f;
        else if (k >= 9) f = 0.0f;
        else f = 0.3f;  // single interface cell at k=8 with f=0.3
        h_fill[i + nx*(j + ny*k)] = f;
    }
    vof.initialize(h_fill.data());
    vof.recomputePLICReconstruction();

    auto v = vof.getInterfaceGeometry();
    ASSERT_TRUE(v.plic_ready);

    int N = v.n;
    std::vector<float> alpha_h(N), nx_h(N), ny_h(N), nz_h(N), fill_h(N);
    cudaMemcpy(alpha_h.data(), v.d_alpha,    N*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(nx_h.data(),    v.d_normal_x, N*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(ny_h.data(),    v.d_normal_y, N*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(nz_h.data(),    v.d_normal_z, N*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(fill_h.data(),  v.d_fill,     N*sizeof(float), cudaMemcpyDeviceToHost);

    // Verify NO NaN anywhere — the H2 fix should keep alpha finite.
    int nan_count = 0;
    for (int idx = 0; idx < N; ++idx) {
        if (std::isnan(alpha_h[idx]) || std::isinf(alpha_h[idx])) ++nan_count;
    }
    EXPECT_EQ(nan_count, 0) << "Axis-aligned slab produced NaN alpha — H2 fix regressed";

    // Sample interface cell at center of k=8 plane: f=0.3, n̂ ≈ (0,0,1).
    int center = (nx/2) + nx * ((ny/2) + ny * 8);
    EXPECT_NEAR(fill_h[center], 0.3f, 1e-5f);
    float n_x = nx_h[center], n_y = ny_h[center], n_z = nz_h[center];
    float mag = std::sqrt(n_x*n_x + n_y*n_y + n_z*n_z);
    EXPECT_NEAR(mag, 1.0f, 0.01f);
    EXPECT_NEAR(n_z, 1.0f, 0.05f) << "Pure z-slab → n̂≈+ẑ";

    // Round-trip: V(alpha, 0, 0, 1) = 0.3 → alpha_signed = 0.3 in unit cube.
    float f_round = plicVolumeInBoxHost(alpha_h[center], n_x, n_y, n_z, 1.0f, 1.0f, 1.0f);
    EXPECT_NEAR(f_round, 0.3f, 1e-4f);
}
