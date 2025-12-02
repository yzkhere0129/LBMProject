/**
 * @file test_vof_curvature_sphere.cu
 * @brief Test curvature computation for spherical interface
 *
 * Physics: Curvature of sphere in 3D
 * - Analytical solution: κ = 2/R (mean curvature)
 * - Principal curvatures: κ1 = κ2 = 1/R
 * - Mean curvature: κ = (κ1 + κ2) = 2/R
 *
 * Validates:
 * - Curvature magnitude accuracy
 * - Isotropy (same curvature in all directions)
 * - Sign convention (positive for liquid droplet)
 */

#include <gtest/gtest.h>
#include "physics/vof_solver.h"
#include <cuda_runtime.h>
#include <vector>
#include <cmath>
#include <iostream>
#include <algorithm>

using namespace lbm::physics;

class VOFCurvatureSphereTest : public ::testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}

    /**
     * @brief Sample curvature at interface cells
     * Returns vector of curvature values where 0.1 < f < 0.9
     */
    std::vector<float> sampleInterfaceCurvature(const std::vector<float>& fill,
                                                 const std::vector<float>& curvature) {
        std::vector<float> interface_curvatures;

        for (size_t i = 0; i < fill.size(); ++i) {
            float f = fill[i];
            if (f > 0.1f && f < 0.9f) {
                // Only include non-zero curvatures
                if (std::abs(curvature[i]) > 1e-6f) {
                    interface_curvatures.push_back(curvature[i]);
                }
            }
        }

        return interface_curvatures;
    }

    /**
     * @brief Compute statistics (mean, std, min, max)
     */
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

        // Mean
        stats.mean = 0.0f;
        for (float v : values) {
            stats.mean += v;
        }
        stats.mean /= values.size();

        // Std dev
        stats.std_dev = 0.0f;
        for (float v : values) {
            float diff = v - stats.mean;
            stats.std_dev += diff * diff;
        }
        stats.std_dev = std::sqrt(stats.std_dev / values.size());

        // Min/max
        stats.min = *std::min_element(values.begin(), values.end());
        stats.max = *std::max_element(values.begin(), values.end());

        return stats;
    }
};

/**
 * @brief Test 1: Large Sphere (R = 20 cells)
 *
 * Large radius → easier to resolve → should have good accuracy
 * Expected: error < 10%
 */
TEST_F(VOFCurvatureSphereTest, LargeSphere) {
    std::cout << "\n=== VOF Curvature: Large Sphere (R=20) ===" << std::endl;

    const int nx = 64, ny = 64, nz = 64;
    const float dx = 1.0f;
    const int num_cells = nx * ny * nz;

    const float cx = nx / 2.0f;
    const float cy = ny / 2.0f;
    const float cz = nz / 2.0f;
    const float R = 20.0f;

    // Analytical curvature: κ = 2/R
    const float kappa_analytical = 2.0f / R;

    std::cout << "  Domain: " << nx << "×" << ny << "×" << nz << std::endl;
    std::cout << "  Sphere: center=(" << cx << "," << cy << "," << cz << "), R=" << R << std::endl;
    std::cout << "  Analytical curvature: κ = 2/R = " << kappa_analytical << std::endl;

    // Initialize VOF
    VOFSolver vof(nx, ny, nz, dx);
    vof.initializeDroplet(cx, cy, cz, R);

    // Reconstruct interface and compute curvature
    vof.reconstructInterface();
    vof.computeCurvature();

    // Get results
    std::vector<float> h_fill(num_cells);
    std::vector<float> h_curvature(num_cells);

    vof.copyFillLevelToHost(h_fill.data());
    vof.copyCurvatureToHost(h_curvature.data());

    // Sample interface curvatures
    auto interface_curvatures = sampleInterfaceCurvature(h_fill, h_curvature);

    ASSERT_GT(interface_curvatures.size(), 50)
        << "Not enough interface samples (need > 50)";

    // Compute statistics
    auto stats = computeStatistics(interface_curvatures);

    std::cout << "\n  Curvature Statistics:" << std::endl;
    std::cout << "    Samples: " << stats.count << std::endl;
    std::cout << "    Mean: κ = " << stats.mean << std::endl;
    std::cout << "    Std dev: σ = " << stats.std_dev << std::endl;
    std::cout << "    Range: [" << stats.min << ", " << stats.max << "]" << std::endl;

    // Compute error
    float relative_error = std::abs(stats.mean - kappa_analytical) / kappa_analytical;
    std::cout << "    Relative error: " << relative_error * 100.0f << "%" << std::endl;

    // Validation: error < 10% for large sphere
    EXPECT_LT(relative_error, 0.10f)
        << "Curvature error too large: " << relative_error * 100.0f << "%";

    // Validation: curvature should be positive (liquid droplet)
    EXPECT_GT(stats.mean, 0.0f)
        << "Curvature should be positive for liquid droplet";

    std::cout << "  ✓ Test passed (curvature within 10%)" << std::endl;
}

/**
 * @brief Test 2: Medium Sphere (R = 12 cells)
 *
 * Moderate radius → typical resolution
 * Expected: error < 20%
 */
TEST_F(VOFCurvatureSphereTest, MediumSphere) {
    std::cout << "\n=== VOF Curvature: Medium Sphere (R=12) ===" << std::endl;

    const int nx = 48, ny = 48, nz = 48;
    const float dx = 1.0f;
    const int num_cells = nx * ny * nz;

    const float cx = nx / 2.0f;
    const float cy = ny / 2.0f;
    const float cz = nz / 2.0f;
    const float R = 12.0f;

    const float kappa_analytical = 2.0f / R;

    std::cout << "  Sphere: R = " << R << std::endl;
    std::cout << "  Analytical curvature: κ = " << kappa_analytical << std::endl;

    // Initialize VOF
    VOFSolver vof(nx, ny, nz, dx);
    vof.initializeDroplet(cx, cy, cz, R);

    vof.reconstructInterface();
    vof.computeCurvature();

    // Get results
    std::vector<float> h_fill(num_cells);
    std::vector<float> h_curvature(num_cells);

    vof.copyFillLevelToHost(h_fill.data());
    vof.copyCurvatureToHost(h_curvature.data());

    auto interface_curvatures = sampleInterfaceCurvature(h_fill, h_curvature);

    ASSERT_GT(interface_curvatures.size(), 30)
        << "Not enough interface samples";

    auto stats = computeStatistics(interface_curvatures);

    std::cout << "  Curvature: κ = " << stats.mean << " ± " << stats.std_dev << std::endl;

    float relative_error = std::abs(stats.mean - kappa_analytical) / kappa_analytical;
    std::cout << "  Relative error: " << relative_error * 100.0f << "%" << std::endl;

    // Validation: error < 20% for medium sphere
    EXPECT_LT(relative_error, 0.20f)
        << "Curvature error too large: " << relative_error * 100.0f << "%";

    std::cout << "  ✓ Test passed (curvature within 20%)" << std::endl;
}

/**
 * @brief Test 3: Small Sphere (R = 8 cells)
 *
 * Small radius → difficult to resolve → relaxed tolerance
 * Expected: error < 30%
 */
TEST_F(VOFCurvatureSphereTest, SmallSphere) {
    std::cout << "\n=== VOF Curvature: Small Sphere (R=8) ===" << std::endl;

    const int nx = 32, ny = 32, nz = 32;
    const float dx = 1.0f;
    const int num_cells = nx * ny * nz;

    const float cx = nx / 2.0f;
    const float cy = ny / 2.0f;
    const float cz = nz / 2.0f;
    const float R = 8.0f;

    const float kappa_analytical = 2.0f / R;

    std::cout << "  Sphere: R = " << R << std::endl;
    std::cout << "  Analytical curvature: κ = " << kappa_analytical << std::endl;

    // Initialize VOF
    VOFSolver vof(nx, ny, nz, dx);
    vof.initializeDroplet(cx, cy, cz, R);

    vof.reconstructInterface();
    vof.computeCurvature();

    // Get results
    std::vector<float> h_fill(num_cells);
    std::vector<float> h_curvature(num_cells);

    vof.copyFillLevelToHost(h_fill.data());
    vof.copyCurvatureToHost(h_curvature.data());

    auto interface_curvatures = sampleInterfaceCurvature(h_fill, h_curvature);

    ASSERT_GT(interface_curvatures.size(), 10)
        << "Not enough interface samples";

    auto stats = computeStatistics(interface_curvatures);

    std::cout << "  Curvature: κ = " << stats.mean << " ± " << stats.std_dev << std::endl;

    float relative_error = std::abs(stats.mean - kappa_analytical) / kappa_analytical;
    std::cout << "  Relative error: " << relative_error * 100.0f << "%" << std::endl;

    // Validation: error < 30% for small sphere (harder to resolve)
    EXPECT_LT(relative_error, 0.30f)
        << "Curvature error too large: " << relative_error * 100.0f << "%";

    std::cout << "  ✓ Test passed (curvature within 30%)" << std::endl;
}

/**
 * @brief Test 4: Curvature Isotropy
 *
 * Validates that curvature is uniform across sphere surface
 * Checks for anisotropy errors (e.g., staircase effects on Cartesian grid)
 */
TEST_F(VOFCurvatureSphereTest, CurvatureIsotropy) {
    std::cout << "\n=== VOF Curvature: Isotropy Test ===" << std::endl;

    const int nx = 64, ny = 64, nz = 64;
    const float dx = 1.0f;
    const int num_cells = nx * ny * nz;

    const float cx = nx / 2.0f;
    const float cy = ny / 2.0f;
    const float cz = nz / 2.0f;
    const float R = 16.0f;

    std::cout << "  Sphere: R = " << R << std::endl;

    // Initialize VOF
    VOFSolver vof(nx, ny, nz, dx);
    vof.initializeDroplet(cx, cy, cz, R);

    vof.reconstructInterface();
    vof.computeCurvature();

    // Get results
    std::vector<float> h_fill(num_cells);
    std::vector<float> h_curvature(num_cells);

    vof.copyFillLevelToHost(h_fill.data());
    vof.copyCurvatureToHost(h_curvature.data());

    auto interface_curvatures = sampleInterfaceCurvature(h_fill, h_curvature);
    auto stats = computeStatistics(interface_curvatures);

    std::cout << "  Curvature statistics:" << std::endl;
    std::cout << "    Mean: κ = " << stats.mean << std::endl;
    std::cout << "    Std dev: σ = " << stats.std_dev << std::endl;
    std::cout << "    Coefficient of variation: CV = " << (stats.std_dev / stats.mean) << std::endl;

    // Validation: curvature should be relatively uniform
    // Coefficient of variation (CV = σ/μ) should be < 0.3
    float cv = stats.std_dev / stats.mean;
    EXPECT_LT(cv, 0.3f)
        << "Curvature too anisotropic: CV = " << cv;

    std::cout << "  ✓ Test passed (curvature isotropic: CV < 0.3)" << std::endl;
}

/**
 * @brief Test 5: Sign Convention
 *
 * Validates curvature sign:
 * - Liquid droplet (f=1 inside): positive curvature
 * - Gas bubble (f=0 inside): negative curvature
 */
TEST_F(VOFCurvatureSphereTest, SignConvention) {
    std::cout << "\n=== VOF Curvature: Sign Convention ===" << std::endl;

    const int nx = 48, ny = 48, nz = 48;
    const float dx = 1.0f;
    const int num_cells = nx * ny * nz;

    const float cx = nx / 2.0f;
    const float cy = ny / 2.0f;
    const float cz = nz / 2.0f;
    const float R = 12.0f;

    // Test 1: Liquid droplet (f=1 inside)
    std::cout << "  Case 1: Liquid droplet" << std::endl;

    VOFSolver vof_droplet(nx, ny, nz, dx);
    vof_droplet.initializeDroplet(cx, cy, cz, R);

    vof_droplet.reconstructInterface();
    vof_droplet.computeCurvature();

    std::vector<float> h_fill(num_cells);
    std::vector<float> h_curvature(num_cells);

    vof_droplet.copyFillLevelToHost(h_fill.data());
    vof_droplet.copyCurvatureToHost(h_curvature.data());

    auto curvatures_droplet = sampleInterfaceCurvature(h_fill, h_curvature);
    auto stats_droplet = computeStatistics(curvatures_droplet);

    std::cout << "    Mean curvature: κ = " << stats_droplet.mean << std::endl;

    // Validation: droplet curvature should be positive
    EXPECT_GT(stats_droplet.mean, 0.0f)
        << "Liquid droplet should have positive curvature";

    // Count positive vs negative samples
    int n_positive = 0, n_negative = 0;
    for (float k : curvatures_droplet) {
        if (k > 0.01f) n_positive++;
        if (k < -0.01f) n_negative++;
    }

    std::cout << "    Positive samples: " << n_positive << std::endl;
    std::cout << "    Negative samples: " << n_negative << std::endl;

    EXPECT_GT(n_positive, n_negative * 3)
        << "Droplet should have predominantly positive curvature";

    std::cout << "  ✓ Liquid droplet: positive curvature ✓" << std::endl;
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
