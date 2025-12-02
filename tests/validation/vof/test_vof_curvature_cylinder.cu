/**
 * @file test_vof_curvature_cylinder.cu
 * @brief Test curvature computation for cylindrical interface (2D geometry in 3D)
 *
 * Physics: Curvature of cylinder
 * - Analytical solution: κ = 1/R (in plane perpendicular to axis)
 * - In 3D: one principal curvature is 1/R, other is 0
 * - Mean curvature: κ = (1/R + 0)/2 = 1/(2R)
 *   BUT: VOF curvature = ∇·n = 1/R in the cross-sectional plane
 *
 * Note: For a cylinder aligned with z-axis:
 * - Curvature in xy-plane: κ_xy = 1/R
 * - Curvature along z: κ_z = 0
 * - Total: κ = ∇·n = ∂n_x/∂x + ∂n_y/∂y + ∂n_z/∂z = 1/R + 0 = 1/R
 */

#include <gtest/gtest.h>
#include "physics/vof_solver.h"
#include <cuda_runtime.h>
#include <vector>
#include <cmath>
#include <iostream>
#include <algorithm>

using namespace lbm::physics;

class VOFCurvatureCylinderTest : public ::testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}

    /**
     * @brief Initialize cylindrical interface (aligned with z-axis)
     *
     * @param vof VOF solver
     * @param nx, ny, nz Domain dimensions
     * @param cx, cy Center coordinates in xy-plane
     * @param R Cylinder radius
     */
    void initializeCylinder(VOFSolver& vof, int nx, int ny, int nz,
                           float cx, float cy, float R) {
        std::vector<float> h_fill(nx * ny * nz);

        for (int k = 0; k < nz; ++k) {
            for (int j = 0; j < ny; ++j) {
                for (int i = 0; i < nx; ++i) {
                    int idx = i + nx * (j + ny * k);

                    float x = static_cast<float>(i);
                    float y = static_cast<float>(j);

                    // Distance from cylinder axis
                    float dx = x - cx;
                    float dy = y - cy;
                    float r = std::sqrt(dx * dx + dy * dy);

                    // Smooth interface using tanh
                    float dist_to_edge = R - r;
                    h_fill[idx] = 0.5f * (1.0f + tanhf(dist_to_edge / 1.5f));
                }
            }
        }

        vof.initialize(h_fill.data());
    }

    /**
     * @brief Sample curvature at interface in xy mid-plane
     */
    std::vector<float> sampleMidPlaneCurvature(const std::vector<float>& fill,
                                                const std::vector<float>& curvature,
                                                int nx, int ny, int nz) {
        std::vector<float> interface_curvatures;
        int mid_z = nz / 2;

        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                int idx = i + nx * (j + ny * mid_z);

                if (fill[idx] > 0.1f && fill[idx] < 0.9f) {
                    if (std::abs(curvature[idx]) > 1e-6f) {
                        interface_curvatures.push_back(curvature[idx]);
                    }
                }
            }
        }

        return interface_curvatures;
    }

    struct Statistics {
        float mean;
        float std_dev;
        float min;
        float max;
        int count;
    };

    Statistics computeStatistics(const std::vector<float>& values) {
        Statistics stats;
        stats.count = values.size();

        if (values.empty()) {
            stats.mean = stats.std_dev = stats.min = stats.max = 0.0f;
            return stats;
        }

        stats.mean = 0.0f;
        for (float v : values) stats.mean += v;
        stats.mean /= values.size();

        stats.std_dev = 0.0f;
        for (float v : values) {
            float diff = v - stats.mean;
            stats.std_dev += diff * diff;
        }
        stats.std_dev = std::sqrt(stats.std_dev / values.size());

        stats.min = *std::min_element(values.begin(), values.end());
        stats.max = *std::max_element(values.begin(), values.end());

        return stats;
    }
};

/**
 * @brief Test 1: Large Cylinder (R = 16 cells)
 *
 * Large radius → good resolution
 * Expected: κ ≈ 1/R, error < 15%
 */
TEST_F(VOFCurvatureCylinderTest, LargeCylinder) {
    std::cout << "\n=== VOF Curvature: Large Cylinder (R=16) ===" << std::endl;

    const int nx = 48, ny = 48, nz = 32;
    const float dx = 1.0f;
    const int num_cells = nx * ny * nz;

    const float cx = nx / 2.0f;
    const float cy = ny / 2.0f;
    const float R = 16.0f;

    // Analytical curvature: κ = 1/R (in cross-section)
    const float kappa_analytical = 1.0f / R;

    std::cout << "  Domain: " << nx << "×" << ny << "×" << nz << std::endl;
    std::cout << "  Cylinder: center=(" << cx << "," << cy << "), R=" << R << std::endl;
    std::cout << "  Analytical curvature: κ = 1/R = " << kappa_analytical << std::endl;

    // Initialize VOF
    VOFSolver vof(nx, ny, nz, dx);
    initializeCylinder(vof, nx, ny, nz, cx, cy, R);

    vof.reconstructInterface();
    vof.computeCurvature();

    // Get results
    std::vector<float> h_fill(num_cells);
    std::vector<float> h_curvature(num_cells);

    vof.copyFillLevelToHost(h_fill.data());
    vof.copyCurvatureToHost(h_curvature.data());

    // Sample curvature at mid-plane
    auto interface_curvatures = sampleMidPlaneCurvature(h_fill, h_curvature, nx, ny, nz);

    ASSERT_GT(interface_curvatures.size(), 20)
        << "Not enough interface samples";

    auto stats = computeStatistics(interface_curvatures);

    std::cout << "\n  Curvature Statistics:" << std::endl;
    std::cout << "    Samples: " << stats.count << std::endl;
    std::cout << "    Mean: κ = " << stats.mean << std::endl;
    std::cout << "    Std dev: σ = " << stats.std_dev << std::endl;
    std::cout << "    Range: [" << stats.min << ", " << stats.max << "]" << std::endl;

    float relative_error = std::abs(stats.mean - kappa_analytical) / kappa_analytical;
    std::cout << "    Relative error: " << relative_error * 100.0f << "%" << std::endl;

    // Validation: error < 15%
    EXPECT_LT(relative_error, 0.15f)
        << "Curvature error too large: " << relative_error * 100.0f << "%";

    std::cout << "  ✓ Test passed (curvature within 15%)" << std::endl;
}

/**
 * @brief Test 2: Medium Cylinder (R = 10 cells)
 *
 * Moderate resolution
 * Expected: error < 25%
 */
TEST_F(VOFCurvatureCylinderTest, MediumCylinder) {
    std::cout << "\n=== VOF Curvature: Medium Cylinder (R=10) ===" << std::endl;

    const int nx = 32, ny = 32, nz = 24;
    const float dx = 1.0f;
    const int num_cells = nx * ny * nz;

    const float cx = nx / 2.0f;
    const float cy = ny / 2.0f;
    const float R = 10.0f;

    const float kappa_analytical = 1.0f / R;

    std::cout << "  Cylinder: R = " << R << std::endl;
    std::cout << "  Analytical curvature: κ = " << kappa_analytical << std::endl;

    VOFSolver vof(nx, ny, nz, dx);
    initializeCylinder(vof, nx, ny, nz, cx, cy, R);

    vof.reconstructInterface();
    vof.computeCurvature();

    std::vector<float> h_fill(num_cells);
    std::vector<float> h_curvature(num_cells);

    vof.copyFillLevelToHost(h_fill.data());
    vof.copyCurvatureToHost(h_curvature.data());

    auto interface_curvatures = sampleMidPlaneCurvature(h_fill, h_curvature, nx, ny, nz);

    ASSERT_GT(interface_curvatures.size(), 15)
        << "Not enough interface samples";

    auto stats = computeStatistics(interface_curvatures);

    std::cout << "  Curvature: κ = " << stats.mean << " ± " << stats.std_dev << std::endl;

    float relative_error = std::abs(stats.mean - kappa_analytical) / kappa_analytical;
    std::cout << "  Relative error: " << relative_error * 100.0f << "%" << std::endl;

    EXPECT_LT(relative_error, 0.25f)
        << "Curvature error too large: " << relative_error * 100.0f << "%";

    std::cout << "  ✓ Test passed (curvature within 25%)" << std::endl;
}

/**
 * @brief Test 3: Cylinder vs Sphere Curvature Comparison
 *
 * Validates that cylinder has half the curvature of sphere with same radius
 * κ_cylinder = 1/R, κ_sphere = 2/R
 * Ratio: κ_cylinder / κ_sphere ≈ 0.5
 */
TEST_F(VOFCurvatureCylinderTest, CylinderVsSphere) {
    std::cout << "\n=== VOF Curvature: Cylinder vs Sphere Comparison ===" << std::endl;

    const int nx = 48, ny = 48, nz = 32;
    const float dx = 1.0f;
    const int num_cells = nx * ny * nz;

    const float cx = nx / 2.0f;
    const float cy = ny / 2.0f;
    const float cz = nz / 2.0f;
    const float R = 14.0f;

    // Test cylinder
    std::cout << "  Case 1: Cylinder (R=" << R << ")" << std::endl;

    VOFSolver vof_cyl(nx, ny, nz, dx);
    initializeCylinder(vof_cyl, nx, ny, nz, cx, cy, R);

    vof_cyl.reconstructInterface();
    vof_cyl.computeCurvature();

    std::vector<float> h_fill_cyl(num_cells);
    std::vector<float> h_curv_cyl(num_cells);

    vof_cyl.copyFillLevelToHost(h_fill_cyl.data());
    vof_cyl.copyCurvatureToHost(h_curv_cyl.data());

    auto curv_cyl = sampleMidPlaneCurvature(h_fill_cyl, h_curv_cyl, nx, ny, nz);
    auto stats_cyl = computeStatistics(curv_cyl);

    std::cout << "    Cylinder curvature: κ_cyl = " << stats_cyl.mean << std::endl;

    // Test sphere
    std::cout << "\n  Case 2: Sphere (R=" << R << ")" << std::endl;

    VOFSolver vof_sph(nx, ny, nz, dx);
    vof_sph.initializeDroplet(cx, cy, cz, R);

    vof_sph.reconstructInterface();
    vof_sph.computeCurvature();

    std::vector<float> h_fill_sph(num_cells);
    std::vector<float> h_curv_sph(num_cells);

    vof_sph.copyFillLevelToHost(h_fill_sph.data());
    vof_sph.copyCurvatureToHost(h_curv_sph.data());

    // Sample sphere curvature
    std::vector<float> curv_sph;
    for (int i = 0; i < num_cells; ++i) {
        if (h_fill_sph[i] > 0.1f && h_fill_sph[i] < 0.9f) {
            if (std::abs(h_curv_sph[i]) > 1e-6f) {
                curv_sph.push_back(h_curv_sph[i]);
            }
        }
    }

    auto stats_sph = computeStatistics(curv_sph);

    std::cout << "    Sphere curvature: κ_sph = " << stats_sph.mean << std::endl;

    // Compute ratio
    float ratio = stats_cyl.mean / stats_sph.mean;
    std::cout << "\n  Ratio: κ_cyl / κ_sph = " << ratio << std::endl;
    std::cout << "  Expected: ≈ 0.5 (κ_cyl = 1/R, κ_sph = 2/R)" << std::endl;

    // Validation: ratio should be close to 0.5 (within 30%)
    float ratio_error = std::abs(ratio - 0.5f) / 0.5f;
    EXPECT_LT(ratio_error, 0.30f)
        << "Cylinder/sphere curvature ratio error: " << ratio_error * 100.0f << "%";

    std::cout << "  ✓ Test passed (cylinder has ~half sphere curvature)" << std::endl;
}

/**
 * @brief Test 4: Axial Uniformity
 *
 * Validates that curvature is uniform along cylinder axis (z-direction)
 */
TEST_F(VOFCurvatureCylinderTest, AxialUniformity) {
    std::cout << "\n=== VOF Curvature: Cylinder Axial Uniformity ===" << std::endl;

    const int nx = 40, ny = 40, nz = 40;
    const float dx = 1.0f;
    const int num_cells = nx * ny * nz;

    const float cx = nx / 2.0f;
    const float cy = ny / 2.0f;
    const float R = 12.0f;

    std::cout << "  Cylinder: R = " << R << ", height = " << nz << std::endl;

    VOFSolver vof(nx, ny, nz, dx);
    initializeCylinder(vof, nx, ny, nz, cx, cy, R);

    vof.reconstructInterface();
    vof.computeCurvature();

    std::vector<float> h_fill(num_cells);
    std::vector<float> h_curvature(num_cells);

    vof.copyFillLevelToHost(h_fill.data());
    vof.copyCurvatureToHost(h_curvature.data());

    // Sample curvature at different z-levels
    std::vector<float> mean_curvatures_by_z;

    for (int k = nz / 4; k < 3 * nz / 4; k += 2) {
        std::vector<float> curv_at_z;

        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                int idx = i + nx * (j + ny * k);

                if (h_fill[idx] > 0.1f && h_fill[idx] < 0.9f) {
                    if (std::abs(h_curvature[idx]) > 1e-6f) {
                        curv_at_z.push_back(h_curvature[idx]);
                    }
                }
            }
        }

        if (!curv_at_z.empty()) {
            float mean_k = 0.0f;
            for (float k : curv_at_z) mean_k += k;
            mean_k /= curv_at_z.size();
            mean_curvatures_by_z.push_back(mean_k);
        }
    }

    ASSERT_GT(mean_curvatures_by_z.size(), 5)
        << "Not enough z-level samples";

    auto stats = computeStatistics(mean_curvatures_by_z);

    std::cout << "  Curvature along axis:" << std::endl;
    std::cout << "    Mean: κ = " << stats.mean << std::endl;
    std::cout << "    Std dev: σ = " << stats.std_dev << std::endl;
    std::cout << "    Coefficient of variation: CV = " << (stats.std_dev / stats.mean) << std::endl;

    // Validation: curvature should be uniform along axis (CV < 0.2)
    float cv = stats.std_dev / stats.mean;
    EXPECT_LT(cv, 0.2f)
        << "Curvature not uniform along axis: CV = " << cv;

    std::cout << "  ✓ Test passed (curvature uniform along axis)" << std::endl;
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
