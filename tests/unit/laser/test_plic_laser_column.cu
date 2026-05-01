/**
 * @file test_plic_laser_column.cu
 * @brief Phase 2 PLIC upgrade — verify laser column-march deposition.
 *
 * Tests the new computeLaserHeatSourcePLICColumnKernel that replaces the
 * legacy per-cell Beer-Lambert volumetric heat source. Acceptance criteria
 * (roadmap §3 Phase 2):
 *
 *   1. In a 1×1×N column with the laser centred on the column, ΔT in the
 *      topmost interface cell after 1 μs of heating matches the analytic
 *      value q · dt / (ρ · cp · f · dx) within 1 %.
 *
 *   2. Bulk metal cells below the interface receive NO direct heat (only
 *      conduction). Equivalently: the kernel writes Q_vol = 0 to all cells
 *      below the topmost interface cell.
 *
 *   3. Pure-gas columns (f < 1e-3 everywhere) receive zero deposition.
 *
 *   4. The Q_vol value on the interface cell satisfies the dimensional
 *      identity Q_vol = I_xy / (dx · max(f, f_min_deposit)).
 *
 * The test calls the kernel directly via a forward declaration, exercising
 * the deposition logic without bringing up the full MultiphysicsSolver.
 */

#include <gtest/gtest.h>
#include "physics/laser_source.h"
#include <cuda_runtime.h>
#include <vector>
#include <cmath>

namespace lbm {
namespace physics {
// Forward declaration of the kernel under test.
__global__ void computeLaserHeatSourcePLICColumnKernel(
    float* d_heat_source,
    const float* fill_level,
    LaserSource laser,
    int nx, int ny, int nz,
    float dx,
    float f_min_deposit);
}  // namespace physics
}  // namespace lbm

using namespace lbm::physics;

namespace {

// Build a LaserSource at the centre of the (i, j) plane with sane LPBF
// parameters. spot_radius is large enough that the central column receives
// ~full intensity.
LaserSource buildCentredLaser(int nx, int ny, float dx,
                              float power, float spot_radius,
                              float absorptivity, float penetration_depth)
{
    LaserSource laser(power, spot_radius, absorptivity, penetration_depth);
    laser.setPosition((nx * 0.5f) * dx, (ny * 0.5f) * dx, 0.0f);
    laser.vx = 0.0f;
    laser.vy = 0.0f;
    return laser;
}

// Launch the kernel and copy d_heat_source back to host.
void runKernel(const std::vector<float>& fill_h,
               int nx, int ny, int nz, float dx,
               const LaserSource& laser, float f_min_deposit,
               std::vector<float>& heat_out)
{
    const int N = nx * ny * nz;
    float *d_fill = nullptr, *d_heat = nullptr;
    cudaMalloc(&d_fill, N * sizeof(float));
    cudaMalloc(&d_heat, N * sizeof(float));
    cudaMemcpy(d_fill, fill_h.data(), N * sizeof(float), cudaMemcpyHostToDevice);

    dim3 threads(16, 16);
    dim3 blocks((nx + 15) / 16, (ny + 15) / 16);
    computePlicLaserKernelLaunch:  // tag for grep; no semantics
    computeLaserHeatSourcePLICColumnKernel<<<blocks, threads>>>(
        d_heat, d_fill, laser, nx, ny, nz, dx, f_min_deposit);
    cudaDeviceSynchronize();

    heat_out.resize(N);
    cudaMemcpy(heat_out.data(), d_heat, N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_fill);
    cudaFree(d_heat);
}

}  // namespace

// ---------------------------------------------------------------------------
// Test 1: Pure-gas column → no deposition anywhere.
// ---------------------------------------------------------------------------
TEST(PLICLaserColumn, AllGasColumnReceivesZero) {
    const int nx = 4, ny = 4, nz = 16;
    const float dx = 2e-6f;
    auto laser = buildCentredLaser(nx, ny, dx, /*P=*/200.0f, /*w0=*/40e-6f,
                                   /*α=*/0.35f, /*δ=*/10e-6f);
    std::vector<float> fill(nx * ny * nz, 0.0f);  // all gas
    std::vector<float> heat;
    runKernel(fill, nx, ny, nz, dx, laser, 0.05f, heat);

    for (int idx = 0; idx < nx * ny * nz; ++idx) {
        EXPECT_FLOAT_EQ(heat[idx], 0.0f) << "idx=" << idx;
    }
}

// ---------------------------------------------------------------------------
// Test 2: Single interface cell at top, bulk metal below — laser must
// deposit ALL energy in the interface cell, NONE in the bulk metal.
// ---------------------------------------------------------------------------
TEST(PLICLaserColumn, DepositsInTopmostInterfaceCellOnly) {
    const int nx = 8, ny = 8, nz = 24;
    const float dx = 2e-6f;
    const float P = 200.0f, w0 = 40e-6f, alpha = 0.35f, delta = 10e-6f;
    auto laser = buildCentredLaser(nx, ny, dx, P, w0, alpha, delta);

    // Initial fill: bottom (k=0..15) bulk metal f=1, single interface cell
    // at k=16 with f=0.4, top (k=17..23) gas.
    const int k_interface = 16;
    const float f_interface = 0.4f;
    std::vector<float> fill(nx * ny * nz, 0.0f);
    for (int k = 0; k < nz; ++k) {
        float fv;
        if (k < k_interface) fv = 1.0f;
        else if (k == k_interface) fv = f_interface;
        else fv = 0.0f;
        for (int j = 0; j < ny; ++j)
            for (int i = 0; i < nx; ++i)
                fill[i + nx * (j + ny * k)] = fv;
    }

    std::vector<float> heat;
    runKernel(fill, nx, ny, nz, dx, laser, 0.05f, heat);

    // Probe the central column.
    int i_c = nx / 2, j_c = ny / 2;
    float xc = i_c * dx, yc = j_c * dx;
    float I_xy = laser.computeIntensity(xc, yc) * laser.absorptivity;
    float Q_expected = I_xy / (dx * f_interface);

    // Cells above the interface: zero
    for (int k = k_interface + 1; k < nz; ++k) {
        int idx = i_c + nx * (j_c + ny * k);
        EXPECT_FLOAT_EQ(heat[idx], 0.0f) << "above-interface k=" << k;
    }
    // Interface cell: Q = I_xy / (dx · f)
    {
        int idx = i_c + nx * (j_c + ny * k_interface);
        EXPECT_NEAR(heat[idx], Q_expected, 1e-3f * Q_expected)
            << "Q_vol mismatch at interface cell";
    }
    // Bulk metal cells below: zero (no direct heating).
    for (int k = 0; k < k_interface; ++k) {
        int idx = i_c + nx * (j_c + ny * k);
        EXPECT_FLOAT_EQ(heat[idx], 0.0f) << "below-interface k=" << k;
    }
}

// ---------------------------------------------------------------------------
// Test 3: Bulk-metal-only column (no interface, just f=1 stack of cells).
// Laser deposits all energy in the topmost f=1 cell.
// ---------------------------------------------------------------------------
TEST(PLICLaserColumn, BulkMetalTopReceivesAll) {
    const int nx = 8, ny = 8, nz = 16;
    const float dx = 2e-6f;
    auto laser = buildCentredLaser(nx, ny, dx, 200.0f, 40e-6f, 0.35f, 10e-6f);

    // All cells from k=0 to k=10 are bulk metal, k=11..15 are gas.
    const int k_top = 10;
    std::vector<float> fill(nx * ny * nz, 0.0f);
    for (int k = 0; k <= k_top; ++k)
        for (int j = 0; j < ny; ++j)
            for (int i = 0; i < nx; ++i)
                fill[i + nx * (j + ny * k)] = 1.0f;

    std::vector<float> heat;
    runKernel(fill, nx, ny, nz, dx, laser, 0.05f, heat);

    int i_c = nx / 2, j_c = ny / 2;
    float xc = i_c * dx, yc = j_c * dx;
    float I_xy = laser.computeIntensity(xc, yc) * laser.absorptivity;
    float Q_expected = I_xy / (dx * 1.0f);

    // Cell at k=k_top should hold the deposit.
    int idx_top = i_c + nx * (j_c + ny * k_top);
    EXPECT_NEAR(heat[idx_top], Q_expected, 1e-3f * Q_expected);

    // Cells below k_top: zero.
    for (int k = 0; k < k_top; ++k) {
        int idx = i_c + nx * (j_c + ny * k);
        EXPECT_FLOAT_EQ(heat[idx], 0.0f) << "below-top k=" << k;
    }
    // Cells above k_top: zero.
    for (int k = k_top + 1; k < nz; ++k) {
        int idx = i_c + nx * (j_c + ny * k);
        EXPECT_FLOAT_EQ(heat[idx], 0.0f) << "above-top k=" << k;
    }
}

// ---------------------------------------------------------------------------
// Test 4: Thin meniscus cell — f far below f_min_deposit. The kernel must
// clamp the 1/f denominator at f_min_deposit so Q_vol does not blow up.
// ---------------------------------------------------------------------------
TEST(PLICLaserColumn, ThinMeniscusClampsAtFMinDeposit) {
    const int nx = 4, ny = 4, nz = 8;
    const float dx = 2e-6f;
    const float f_thin = 0.01f;        // far below floor
    const float f_min = 0.05f;
    auto laser = buildCentredLaser(nx, ny, dx, 200.0f, 40e-6f, 0.35f, 10e-6f);

    // Single interface cell at k=4 with f=0.01, gas above & below.
    std::vector<float> fill(nx * ny * nz, 0.0f);
    const int k_thin = 4;
    for (int j = 0; j < ny; ++j)
        for (int i = 0; i < nx; ++i)
            fill[i + nx * (j + ny * k_thin)] = f_thin;

    std::vector<float> heat;
    runKernel(fill, nx, ny, nz, dx, laser, f_min, heat);

    int i_c = nx / 2, j_c = ny / 2;
    int idx = i_c + nx * (j_c + ny * k_thin);
    float xc = i_c * dx, yc = j_c * dx;
    float I_xy = laser.computeIntensity(xc, yc) * laser.absorptivity;
    // Floor: Q_vol = I_xy / (dx · f_min), NOT I_xy / (dx · f_thin).
    float Q_floored = I_xy / (dx * f_min);
    EXPECT_NEAR(heat[idx], Q_floored, 1e-3f * Q_floored)
        << "Q_vol must be floored at f_min_deposit, not blown up by 1/f_thin";

    // Sanity: the un-clamped value would be 5× larger.
    float Q_unclamped = I_xy / (dx * f_thin);
    EXPECT_LT(heat[idx], 0.5f * Q_unclamped)
        << "If clamp were off, Q_vol would be 5× larger";
}
