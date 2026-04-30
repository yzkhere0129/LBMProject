/**
 * @file test_plic_evap_area.cu
 * @brief Phase 4b — unit tests for plicCellSurfaceArea and the geometric PLIC
 *        area evaporation kernel.
 *
 * Analytic values (all for the unit cube [0,1]³ with a unit-normal plane):
 *
 *  Case 1 — Axis-aligned (n̂ = (0,0,1), f = 0.5):
 *    α = 0.5.  Plane z = 0.5 cuts a unit square face.
 *    A_PLIC = 1.0000  (exact, dV/dα = 1 for 1D slab).
 *
 *  Case 2 — 45-degree diagonal (n̂ = (1,1,0)/√2, f = 0.5):
 *    After sorting: m1 = m2 = 1/√2, m3 = 0.  S = √2.
 *    α = S/2 = √2/2 (by symmetry for f = 0.5).
 *    V(α) = α²/(2 m1 m2) in the lower range [0, m2]; dV/dα = α/m² = (√2/2)/(1/2) = √2.
 *    A_PLIC = √2 ≈ 1.4142.
 *
 *  Case 3 — 3D diagonal (n̂ = (1,1,1)/√3, f = 0.5):
 *    m1 = m2 = m3 = 1/√3.  S = √3.  α = √3/2.
 *    The cross-section is a regular hexagon with vertices at permutations of
 *    (0.5, 0, 1).  Side length s = 1/√2.
 *    A_PLIC = (3√3/2) × s² = (3√3/2) × (1/2) = 3√3/4 ≈ 1.2990.
 *    (The specification note "≈ √3 ≈ 1.732" is for a different geometry;
 *     the correct value for f=0.5 is 1.2990.)
 *
 *  Case 4 — Bulk gas (f = 0): A_PLIC = 0.
 *  Case 5 — Bulk liquid (f = 1): A_PLIC = 0.
 *
 *  Case 6 — Round-trip evaporation (axis-aligned):
 *    Expected df = -J × A_PLIC × dt / (ρ × dx) = -J × 1.0 × dt / (ρ × dx).
 *    Should match the legacy formula within 5%.
 */

#include <gtest/gtest.h>
#include "physics/vof_solver.h"
#include "physics/interface_geometry.h"
#include <cuda_runtime.h>
#include <vector>
#include <cmath>

using namespace lbm::physics;

// ============================================================================
// Device-side evaluation helper: call plicCellSurfaceArea from a single-thread
// kernel and copy back to host.
// ============================================================================
__global__ void evalPLICAreaKernel(float n_x, float n_y, float n_z,
                                    float alpha, float* out)
{
    *out = plicCellSurfaceArea(n_x, n_y, n_z, alpha);
}

static float evalPLICArea(float n_x, float n_y, float n_z, float alpha)
{
    float* d_out = nullptr;
    cudaMalloc(&d_out, sizeof(float));

    evalPLICAreaKernel<<<1, 1>>>(n_x, n_y, n_z, alpha, d_out);
    cudaDeviceSynchronize();

    float result = -1.0f;
    cudaMemcpy(&result, d_out, sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_out);
    return result;
}

// ============================================================================
// Test 1: Axis-aligned interface — A_PLIC ≈ 1.0
// ============================================================================
TEST(PLICCellSurfaceArea, AxisAligned) {
    // n̂ = (0, 0, 1), α = 0.5 → plane z = 0.5 in [0,1]³
    float area = evalPLICArea(0.0f, 0.0f, 1.0f, 0.5f);
    printf("[PLICArea] axis-aligned: A = %.6f  (expected 1.000000)\n", area);
    EXPECT_NEAR(area, 1.0f, 0.01f)  // within 1%
        << "Axis-aligned plane should have area 1.0 in the unit cube";
}

// ============================================================================
// Test 2: 45-degree diagonal — A_PLIC ≈ √2 ≈ 1.4142
// ============================================================================
TEST(PLICCellSurfaceArea, Diagonal45) {
    const float inv_sqrt2 = 1.0f / sqrtf(2.0f);
    // α = S/2 = 1/√2 (symmetry point f=0.5)
    float alpha = 1.0f / sqrtf(2.0f);
    float area = evalPLICArea(inv_sqrt2, inv_sqrt2, 0.0f, alpha);
    const float expected = sqrtf(2.0f);
    printf("[PLICArea] 45-deg diagonal: A = %.6f  (expected sqrt(2) = %.6f)\n",
           area, expected);
    EXPECT_NEAR(area, expected, 0.05f * expected)  // within 5%
        << "45-degree diagonal plane should have area sqrt(2)";
}

// ============================================================================
// Test 3: 3D diagonal — A_PLIC ≈ 3√3/4 ≈ 1.2990
// ============================================================================
TEST(PLICCellSurfaceArea, Diagonal3D) {
    const float inv_sqrt3 = 1.0f / sqrtf(3.0f);
    // α = S/2 = √3/2 in sorted unit-cube frame
    float alpha = sqrtf(3.0f) / 2.0f;
    float area = evalPLICArea(inv_sqrt3, inv_sqrt3, inv_sqrt3, alpha);
    const float expected = 3.0f * sqrtf(3.0f) / 4.0f;  // ≈ 1.2990
    printf("[PLICArea] 3D diagonal: A = %.6f  (expected 3*sqrt(3)/4 = %.6f)\n",
           area, expected);
    EXPECT_NEAR(area, expected, 0.05f * expected)  // within 5%
        << "3D diagonal plane should have area 3*sqrt(3)/4 at f=0.5";
}

// ============================================================================
// Test 4: Pure-gas cell (α = 0) → A_PLIC = 0
// ============================================================================
TEST(PLICCellSurfaceArea, PureGasZeroArea) {
    // Plane entirely outside the cube (liquid volume = 0)
    float area = evalPLICArea(0.0f, 0.0f, 1.0f, 0.0f);
    printf("[PLICArea] pure-gas (alpha=0): A = %.6f  (expected 0.0)\n", area);
    EXPECT_EQ(area, 0.0f) << "alpha=0 (gas cell) should give zero area";
}

// ============================================================================
// Test 5: Pure-liquid cell (α ≥ S) → A_PLIC = 0
// ============================================================================
TEST(PLICCellSurfaceArea, PureLiquidZeroArea) {
    // Plane entirely past the cube (liquid volume = 1)
    float area = evalPLICArea(0.0f, 0.0f, 1.0f, 1.1f);
    printf("[PLICArea] pure-liquid (alpha>S): A = %.6f  (expected 0.0)\n", area);
    EXPECT_EQ(area, 0.0f) << "alpha>S (liquid cell) should give zero area";
}

// ============================================================================
// Test 6: Round-trip with evap kernel — mass loss matches J·A·dt/(ρ·dx)
//         for an axis-aligned interface (A_PLIC = 1, so matches legacy formula)
// ============================================================================
TEST(PLICEvapArea, AxisAlignedMassLossMatchesLegacy) {
    const int nx = 8, ny = 8, nz = 32;
    const float dx = 1e-6f;   // 1 μm cells
    const float rho = 7900.0f;
    const float dt  = 1e-7f;
    const float J_uniform = 1.0e3f;  // kg/(m²·s)

    VOFSolver vof(nx, ny, nz, dx);
    vof.setNormalReconstructionMethod(NormalReconstructionMethod::HEIGHT_FUNCTION);

    int N = nx * ny * nz;
    std::vector<float> fill(N, 0.0f);
    // Interface at k=15/16 boundary: layers 0..14 full liquid, k=15 partial,
    // k=16 partial, rest gas — ensures a clean PLIC plane is reconstructed.
    for (int k = 0; k < nz; ++k) {
        float fv;
        if (k <= 14)      fv = 1.0f;
        else if (k == 15) fv = 0.7f;
        else if (k == 16) fv = 0.3f;
        else              fv = 0.0f;
        for (int j = 0; j < ny; ++j)
            for (int i = 0; i < nx; ++i)
                fill[i + nx*(j + ny*k)] = fv;
    }
    vof.initialize(fill.data());

    // J_evap only in the interface band
    std::vector<float> J(N, 0.0f);
    for (int k = 14; k <= 17; ++k)
        for (int j = 0; j < ny; ++j)
            for (int i = 0; i < nx; ++i)
                J[i + nx*(j + ny*k)] = J_uniform;

    float* d_J = nullptr;
    cudaMalloc(&d_J, N * sizeof(float));
    cudaMemcpy(d_J, J.data(), N * sizeof(float), cudaMemcpyHostToDevice);

    double m_before = 0.0;
    for (float f : fill) m_before += f;

    vof.applyEvaporationMassLossPLICArea(d_J, rho, dt);

    std::vector<float> fill_after(N);
    cudaMemcpy(fill_after.data(), vof.getFillLevel(), N * sizeof(float),
               cudaMemcpyDeviceToHost);
    double m_after = 0.0;
    for (float f : fill_after) m_after += f;

    cudaFree(d_J);

    double dm = m_before - m_after;

    // For an axis-aligned interface spread across two cells (k=15 at f=0.7,
    // k=16 at f=0.3), both cells are interface cells with A_PLIC ≈ 1 and both
    // receive J_uniform.  The expected mass loss per column is therefore
    // 2 × J × dt / (ρ × dx) × (nx × ny) (subject to the 2% stability limiter
    // capping the larger-f cell).
    //
    // More precisely: df_15 = min(J·dt/(ρ·dx), 0.02×f_15) and similarly for
    // k=16. We compute the expected value directly from the initial fill fractions.
    double J_dt_rhodx = (double)J_uniform * dt / (rho * dx);
    constexpr double MAX_DF = 0.02;
    double df_k15 = std::min(J_dt_rhodx, MAX_DF * 0.7);
    double df_k16 = std::min(J_dt_rhodx, MAX_DF * 0.3);
    // Total mass removed per column = (df_k15 + df_k16) per column
    double dm_expected = (df_k15 + df_k16) * (nx * ny);

    double rel_err = std::fabs(dm - dm_expected) / dm_expected;

    printf("[PLICEvapArea] mass loss: actual=%.4e  expected=%.4e  rel_err=%.3f%%\n",
           dm, dm_expected, rel_err * 100.0);
    printf("              df_k15=%.4e df_k16=%.4e  J·dt/(ρ·dx)=%.4e\n",
           df_k15, df_k16, J_dt_rhodx);

    EXPECT_LT(rel_err, 0.05)
        << "Axis-aligned PLIC area kernel mass loss must match analytic prediction "
           "within 5% (two interface cells, each A_PLIC=1, 2% limiter applied)";
    EXPECT_GT(dm, 0.0)
        << "Mass loss must be positive";
}
