/**
 * @file test_vtk_liquid_fraction.cu
 * @brief Unit test for VTK writer with liquid fraction field
 *
 * This test verifies that:
 * 1. VTK writer correctly writes liquid fraction values
 * 2. Written VTK files can be read back and verified
 * 3. Non-zero liquid fraction values are preserved
 * 4. Multiple fields (Temperature, LiquidFraction, PhaseState) are written correctly
 */

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cmath>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>

#include "io/vtk_writer.h"

using namespace lbm;

class VTKLiquidFractionTest : public ::testing::Test {
protected:
    void SetUp() override {
        nx = 10;
        ny = 10;
        nz = 10;
        num_cells = nx * ny * nz;

        dx = 1e-6f;
        dy = 1e-6f;
        dz = 1e-6f;

        system("mkdir -p test_output");
    }

    void TearDown() override {
        // Cleanup test files
        system("rm -f test_output/vtk_test_*.vtk");
    }

    // Helper function to read a scalar field from VTK file
    bool readVTKScalarField(const std::string& filename,
                           const std::string& field_name,
                           std::vector<float>& data) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Cannot open file: " << filename << "\n";
            return false;
        }

        std::string line;
        bool found_field = false;

        // Search for the field
        while (std::getline(file, line)) {
            if (line.find("SCALARS " + field_name) != std::string::npos) {
                found_field = true;
                // Skip LOOKUP_TABLE line
                std::getline(file, line);
                break;
            }
        }

        if (!found_field) {
            std::cerr << "Field " << field_name << " not found in " << filename << "\n";
            return false;
        }

        // Read data
        data.clear();
        float value;
        while (file >> value) {
            data.push_back(value);
            if (data.size() >= num_cells) {
                break;
            }
        }

        file.close();
        return data.size() == num_cells;
    }

    int nx, ny, nz, num_cells;
    float dx, dy, dz;
};

/**
 * Test 1: Write and verify zero liquid fraction field
 */
TEST_F(VTKLiquidFractionTest, WriteAndVerifyZeroField) {
    std::cout << "\n=== TEST: Write and Verify Zero Liquid Fraction ===\n";

    // Create fields
    float* h_temp = new float[num_cells];
    float* h_fl = new float[num_cells];
    float* h_phase = new float[num_cells];

    // Fill with values (all solid)
    for (int i = 0; i < num_cells; ++i) {
        h_temp[i] = 300.0f;      // Room temperature
        h_fl[i] = 0.0f;          // No melting
        h_phase[i] = 0.0f;       // Solid
    }

    // Write VTK file
    std::string filename = "test_output/vtk_test_zero_fl.vtk";
    io::VTKWriter::writeStructuredGrid(
        filename, h_temp, h_fl, h_phase,
        nx, ny, nz, dx, dy, dz,
        "Temperature", "LiquidFraction", "PhaseState"
    );

    std::cout << "VTK file written: " << filename << "\n";

    // Read back liquid fraction
    std::vector<float> fl_read;
    bool success = readVTKScalarField(filename, "LiquidFraction", fl_read);

    ASSERT_TRUE(success) << "Should successfully read LiquidFraction field";
    ASSERT_EQ(fl_read.size(), num_cells) << "Should read correct number of values";

    // Verify all zeros
    bool all_zero = true;
    for (size_t i = 0; i < fl_read.size(); ++i) {
        if (fl_read[i] != 0.0f) {
            all_zero = false;
            break;
        }
    }

    std::cout << "All liquid fraction values are zero: "
              << (all_zero ? "YES" : "NO") << "\n";

    delete[] h_temp;
    delete[] h_fl;
    delete[] h_phase;

    EXPECT_TRUE(all_zero) << "All liquid fraction values should be zero";
}

/**
 * Test 2: Write and verify non-zero liquid fraction field
 */
TEST_F(VTKLiquidFractionTest, WriteAndVerifyNonZeroField) {
    std::cout << "\n=== TEST: Write and Verify Non-Zero Liquid Fraction ===\n";

    float* h_temp = new float[num_cells];
    float* h_fl = new float[num_cells];
    float* h_phase = new float[num_cells];

    // Fill with values (all liquid)
    for (int i = 0; i < num_cells; ++i) {
        h_temp[i] = 2500.0f;     // Above melting point
        h_fl[i] = 1.0f;          // Fully melted
        h_phase[i] = 2.0f;       // Liquid
    }

    // Write VTK file
    std::string filename = "test_output/vtk_test_nonzero_fl.vtk";
    io::VTKWriter::writeStructuredGrid(
        filename, h_temp, h_fl, h_phase,
        nx, ny, nz, dx, dy, dz,
        "Temperature", "LiquidFraction", "PhaseState"
    );

    std::cout << "VTK file written: " << filename << "\n";

    // Read back liquid fraction
    std::vector<float> fl_read;
    bool success = readVTKScalarField(filename, "LiquidFraction", fl_read);

    ASSERT_TRUE(success) << "Should successfully read LiquidFraction field";
    ASSERT_EQ(fl_read.size(), num_cells) << "Should read correct number of values";

    // Verify all ones
    float min_fl = 1.0f, max_fl = 0.0f;
    for (size_t i = 0; i < fl_read.size(); ++i) {
        min_fl = fminf(min_fl, fl_read[i]);
        max_fl = fmaxf(max_fl, fl_read[i]);
    }

    std::cout << "Liquid fraction range: [" << min_fl << ", " << max_fl << "]\n";

    delete[] h_temp;
    delete[] h_fl;
    delete[] h_phase;

    EXPECT_NEAR(min_fl, 1.0f, 1e-5f) << "Minimum should be 1.0";
    EXPECT_NEAR(max_fl, 1.0f, 1e-5f) << "Maximum should be 1.0";
}

/**
 * Test 3: Write and verify spatially varying liquid fraction
 */
TEST_F(VTKLiquidFractionTest, WriteAndVerifySpatiallyVaryingField) {
    std::cout << "\n=== TEST: Write and Verify Spatially Varying Field ===\n";

    float* h_temp = new float[num_cells];
    float* h_fl = new float[num_cells];
    float* h_phase = new float[num_cells];

    int center_x = nx / 2;
    int center_y = ny / 2;
    int center_z = nz / 2;

    // Create Gaussian liquid fraction profile
    // Center: fully melted (fl=1.0)
    // Edges: solid (fl=0.0)
    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                int idx = i + j * nx + k * nx * ny;

                float dx = (i - center_x);
                float dy = (j - center_y);
                float dz = (k - center_z);
                float r2 = dx*dx + dy*dy + dz*dz;

                // Gaussian profile
                float fl = expf(-r2 / 10.0f);
                h_fl[idx] = fl;

                // Temperature and phase consistent with fl
                h_temp[idx] = 300.0f + fl * 2200.0f;
                if (fl < 0.01f) {
                    h_phase[idx] = 0.0f;  // Solid
                } else if (fl > 0.99f) {
                    h_phase[idx] = 2.0f;  // Liquid
                } else {
                    h_phase[idx] = 1.0f;  // Mushy
                }
            }
        }
    }

    // Write VTK file
    std::string filename = "test_output/vtk_test_varying_fl.vtk";
    io::VTKWriter::writeStructuredGrid(
        filename, h_temp, h_fl, h_phase,
        nx, ny, nz, dx, dy, dz,
        "Temperature", "LiquidFraction", "PhaseState"
    );

    std::cout << "VTK file written: " << filename << "\n";

    // Read back liquid fraction
    std::vector<float> fl_read;
    bool success = readVTKScalarField(filename, "LiquidFraction", fl_read);

    ASSERT_TRUE(success) << "Should successfully read LiquidFraction field";

    // Verify values match what we wrote
    int n_matches = 0;
    float max_error = 0.0f;
    for (int i = 0; i < num_cells; ++i) {
        float error = fabsf(fl_read[i] - h_fl[i]);
        max_error = fmaxf(max_error, error);
        if (error < 1e-5f) {
            n_matches++;
        }
    }

    float match_percent = 100.0f * n_matches / num_cells;

    std::cout << "Values matching original: " << n_matches << " / " << num_cells
              << " (" << match_percent << "%)\n";
    std::cout << "Maximum error: " << max_error << "\n";

    delete[] h_temp;
    delete[] h_fl;
    delete[] h_phase;

    EXPECT_GT(match_percent, 99.0f) << "At least 99% of values should match";
    EXPECT_LT(max_error, 1e-4f) << "Maximum error should be negligible";
}

/**
 * Test 4: Verify all three fields are written correctly
 */
TEST_F(VTKLiquidFractionTest, VerifyAllThreeFields) {
    std::cout << "\n=== TEST: Verify All Three Fields ===\n";

    float* h_temp = new float[num_cells];
    float* h_fl = new float[num_cells];
    float* h_phase = new float[num_cells];

    // Create test pattern
    for (int i = 0; i < num_cells; ++i) {
        // Alternating pattern
        if (i % 3 == 0) {
            h_temp[i] = 300.0f;
            h_fl[i] = 0.0f;
            h_phase[i] = 0.0f;
        } else if (i % 3 == 1) {
            h_temp[i] = 1900.0f;
            h_fl[i] = 0.5f;
            h_phase[i] = 1.0f;
        } else {
            h_temp[i] = 2500.0f;
            h_fl[i] = 1.0f;
            h_phase[i] = 2.0f;
        }
    }

    // Write VTK file
    std::string filename = "test_output/vtk_test_all_fields.vtk";
    io::VTKWriter::writeStructuredGrid(
        filename, h_temp, h_fl, h_phase,
        nx, ny, nz, dx, dy, dz,
        "Temperature", "LiquidFraction", "PhaseState"
    );

    // Read back all three fields
    std::vector<float> temp_read, fl_read, phase_read;
    bool temp_ok = readVTKScalarField(filename, "Temperature", temp_read);
    bool fl_ok = readVTKScalarField(filename, "LiquidFraction", fl_read);
    bool phase_ok = readVTKScalarField(filename, "PhaseState", phase_read);

    ASSERT_TRUE(temp_ok) << "Should read Temperature field";
    ASSERT_TRUE(fl_ok) << "Should read LiquidFraction field";
    ASSERT_TRUE(phase_ok) << "Should read PhaseState field";

    // Verify all fields
    int temp_matches = 0, fl_matches = 0, phase_matches = 0;
    for (int i = 0; i < num_cells; ++i) {
        if (fabsf(temp_read[i] - h_temp[i]) < 1e-3f) temp_matches++;
        if (fabsf(fl_read[i] - h_fl[i]) < 1e-5f) fl_matches++;
        if (fabsf(phase_read[i] - h_phase[i]) < 1e-5f) phase_matches++;
    }

    std::cout << "Temperature matches: " << temp_matches << " / " << num_cells << "\n";
    std::cout << "LiquidFraction matches: " << fl_matches << " / " << num_cells << "\n";
    std::cout << "PhaseState matches: " << phase_matches << " / " << num_cells << "\n";

    delete[] h_temp;
    delete[] h_fl;
    delete[] h_phase;

    EXPECT_EQ(temp_matches, num_cells) << "All temperature values should match";
    EXPECT_EQ(fl_matches, num_cells) << "All liquid fraction values should match";
    EXPECT_EQ(phase_matches, num_cells) << "All phase state values should match";
}

/**
 * Test 5: Test with realistic melting scenario values
 *
 * This test uses values similar to what would appear in actual simulation:
 * - Center: T=2000K, fl=0.8, phase=mushy
 * - Edge: T=300K, fl=0.0, phase=solid
 */
TEST_F(VTKLiquidFractionTest, RealisticMeltingScenario) {
    std::cout << "\n=== TEST: Realistic Melting Scenario ===\n";

    float* h_temp = new float[num_cells];
    float* h_fl = new float[num_cells];
    float* h_phase = new float[num_cells];

    int center_idx = (nx/2) + (ny/2) * nx + (nz/2) * nx * ny;

    // Create realistic melting pattern
    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                int idx = i + j * nx + k * nx * ny;

                float dx = (i - nx/2.0f);
                float dy = (j - ny/2.0f);
                float dz = (k - nz/2.0f);
                float r = sqrtf(dx*dx + dy*dy + dz*dz);

                // Realistic temperature profile
                if (r < 2.0f) {
                    // Hot center
                    h_temp[idx] = 2000.0f;
                    h_fl[idx] = 0.8f;
                    h_phase[idx] = 1.0f;  // Mushy
                } else if (r < 4.0f) {
                    // Transition zone
                    h_temp[idx] = 1500.0f;
                    h_fl[idx] = 0.3f;
                    h_phase[idx] = 1.0f;  // Mushy
                } else {
                    // Cold edges
                    h_temp[idx] = 300.0f;
                    h_fl[idx] = 0.0f;
                    h_phase[idx] = 0.0f;  // Solid
                }
            }
        }
    }

    // Write VTK file
    std::string filename = "test_output/vtk_test_realistic.vtk";
    io::VTKWriter::writeStructuredGrid(
        filename, h_temp, h_fl, h_phase,
        nx, ny, nz, dx, dy, dz,
        "Temperature", "LiquidFraction", "PhaseState"
    );

    std::cout << "VTK file written: " << filename << "\n";

    // Read back and verify
    std::vector<float> fl_read;
    bool success = readVTKScalarField(filename, "LiquidFraction", fl_read);

    ASSERT_TRUE(success) << "Should read LiquidFraction field";

    // Find center value
    float center_fl_written = h_fl[center_idx];
    float center_fl_read = fl_read[center_idx];

    std::cout << "Center liquid fraction (written): " << center_fl_written << "\n";
    std::cout << "Center liquid fraction (read): " << center_fl_read << "\n";

    // Count non-zero values
    int n_nonzero_written = 0, n_nonzero_read = 0;
    for (int i = 0; i < num_cells; ++i) {
        if (h_fl[i] > 0.01f) n_nonzero_written++;
        if (fl_read[i] > 0.01f) n_nonzero_read++;
    }

    std::cout << "Non-zero fl cells (written): " << n_nonzero_written << "\n";
    std::cout << "Non-zero fl cells (read): " << n_nonzero_read << "\n";

    delete[] h_temp;
    delete[] h_fl;
    delete[] h_phase;

    EXPECT_NEAR(center_fl_read, 0.8f, 1e-5f)
        << "Center liquid fraction should be preserved";
    EXPECT_EQ(n_nonzero_read, n_nonzero_written)
        << "Number of non-zero cells should match";
}

/**
 * Test 6: File existence and format validation
 */
TEST_F(VTKLiquidFractionTest, FileExistenceAndFormat) {
    std::cout << "\n=== TEST: File Existence and Format ===\n";

    float* h_temp = new float[num_cells];
    float* h_fl = new float[num_cells];
    float* h_phase = new float[num_cells];

    for (int i = 0; i < num_cells; ++i) {
        h_temp[i] = 1000.0f + i;
        h_fl[i] = 0.5f;
        h_phase[i] = 1.0f;
    }

    std::string filename = "test_output/vtk_test_format.vtk";
    io::VTKWriter::writeStructuredGrid(
        filename, h_temp, h_fl, h_phase,
        nx, ny, nz, dx, dy, dz,
        "Temperature", "LiquidFraction", "PhaseState"
    );

    // Check file exists
    std::ifstream file(filename);
    ASSERT_TRUE(file.is_open()) << "VTK file should exist";

    // Check VTK header
    std::string line;
    std::getline(file, line);
    bool has_vtk_header = (line.find("# vtk DataFile") != std::string::npos);

    // Check for required keywords
    bool has_dimensions = false;
    bool has_point_data = false;
    bool has_liquid_fraction = false;

    while (std::getline(file, line)) {
        if (line.find("DIMENSIONS") != std::string::npos) has_dimensions = true;
        if (line.find("POINT_DATA") != std::string::npos) has_point_data = true;
        if (line.find("SCALARS LiquidFraction") != std::string::npos) has_liquid_fraction = true;
    }

    file.close();

    std::cout << "VTK header present: " << (has_vtk_header ? "YES" : "NO") << "\n";
    std::cout << "DIMENSIONS keyword: " << (has_dimensions ? "YES" : "NO") << "\n";
    std::cout << "POINT_DATA keyword: " << (has_point_data ? "YES" : "NO") << "\n";
    std::cout << "LiquidFraction field: " << (has_liquid_fraction ? "YES" : "NO") << "\n";

    delete[] h_temp;
    delete[] h_fl;
    delete[] h_phase;

    EXPECT_TRUE(has_vtk_header) << "File should have VTK header";
    EXPECT_TRUE(has_dimensions) << "File should have DIMENSIONS";
    EXPECT_TRUE(has_point_data) << "File should have POINT_DATA";
    EXPECT_TRUE(has_liquid_fraction) << "File should have LiquidFraction field";
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
