/**
 * @file test_vtk_vector_io.cu
 * @brief Unit test for VTK vector field I/O functionality
 *
 * This test validates the new vector field output capabilities added in Phase 5.
 * It verifies:
 * 1. VTK VECTORS format correctness
 * 2. writeVectorField() function
 * 3. writeStructuredGridWithVectors() function
 * 4. File format compliance with VTK legacy format specification
 */

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cmath>
#include <vector>
#include <fstream>
#include <string>
#include <sstream>
#include <algorithm>

#include "io/vtk_writer.h"

using namespace lbm;

/**
 * @brief Test fixture for VTK vector I/O tests
 */
class VTKVectorIOTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create small test domain
        nx_ = 8;
        ny_ = 8;
        nz_ = 4;
        num_cells_ = nx_ * ny_ * nz_;

        dx_ = 1e-3f;  // 1 mm
        dy_ = 1e-3f;
        dz_ = 1e-3f;

        // Allocate host arrays
        velocity_x_.resize(num_cells_);
        velocity_y_.resize(num_cells_);
        velocity_z_.resize(num_cells_);
        temperature_.resize(num_cells_);
        liquid_fraction_.resize(num_cells_);
        phase_state_.resize(num_cells_);

        // Initialize with analytical test pattern
        for (int k = 0; k < nz_; ++k) {
            for (int j = 0; j < ny_; ++j) {
                for (int i = 0; i < nx_; ++i) {
                    int idx = i + j * nx_ + k * nx_ * ny_;

                    // Position in [0, 1]
                    float x = static_cast<float>(i) / (nx_ - 1);
                    float y = static_cast<float>(j) / (ny_ - 1);
                    float z = static_cast<float>(k) / (nz_ - 1);

                    // Circular velocity pattern (vortex in xy-plane)
                    float cx = 0.5f, cy = 0.5f;  // Center
                    float dx = x - cx;
                    float dy_val = y - cy;
                    velocity_x_[idx] = -dy_val * 0.1f;  // 0.1 m/s scale
                    velocity_y_[idx] = dx * 0.1f;
                    velocity_z_[idx] = 0.01f * z;  // Weak vertical flow

                    // Temperature gradient
                    temperature_[idx] = 300.0f + 100.0f * z;

                    // Liquid fraction gradient
                    liquid_fraction_[idx] = z;

                    // Phase state
                    if (z < 0.3f) {
                        phase_state_[idx] = 0.0f;  // Solid
                    } else if (z < 0.7f) {
                        phase_state_[idx] = 1.0f;  // Mushy
                    } else {
                        phase_state_[idx] = 2.0f;  // Liquid
                    }
                }
            }
        }
    }

    void TearDown() override {
        // Clean up test files
        system("rm -f test_vtk_*.vtk");
    }

    int nx_, ny_, nz_, num_cells_;
    float dx_, dy_, dz_;
    std::vector<float> velocity_x_, velocity_y_, velocity_z_;
    std::vector<float> temperature_, liquid_fraction_, phase_state_;
};

/**
 * @brief Test 1: Write vector field only
 */
TEST_F(VTKVectorIOTest, WriteVectorFieldOnly) {
    std::cout << "=== Test: Write Vector Field Only ===\n";

    std::string filename = "test_vtk_vector_only";

    // Write vector field
    ASSERT_NO_THROW({
        io::VTKWriter::writeVectorField(
            filename,
            velocity_x_.data(),
            velocity_y_.data(),
            velocity_z_.data(),
            nx_, ny_, nz_,
            dx_, dy_, dz_,
            "Velocity"
        );
    }) << "Writing vector field should not throw exception";

    // Verify file was created
    std::ifstream file(filename + ".vtk");
    ASSERT_TRUE(file.is_open()) << "VTK file should be created";

    // Read and verify file contents
    std::string line;
    int line_num = 0;
    bool found_vectors = false;
    int vector_count = 0;

    while (std::getline(file, line)) {
        line_num++;

        // Check header
        if (line_num == 1) {
            EXPECT_EQ(line, "# vtk DataFile Version 3.0");
        } else if (line_num == 3) {
            EXPECT_EQ(line, "ASCII");
        } else if (line_num == 4) {
            EXPECT_EQ(line, "DATASET STRUCTURED_POINTS");
        }

        // Check VECTORS declaration
        if (line.find("VECTORS Velocity float") != std::string::npos) {
            found_vectors = true;
        }

        // Count vector entries (lines with 3 space-separated floats)
        if (found_vectors && vector_count < num_cells_) {
            std::istringstream iss(line);
            float vx, vy, vz;
            if (iss >> vx >> vy >> vz) {
                vector_count++;

                // Verify values are reasonable (not NaN/Inf)
                EXPECT_TRUE(std::isfinite(vx)) << "vx should be finite";
                EXPECT_TRUE(std::isfinite(vy)) << "vy should be finite";
                EXPECT_TRUE(std::isfinite(vz)) << "vz should be finite";
            }
        }
    }

    file.close();

    EXPECT_TRUE(found_vectors) << "VECTORS keyword should be present";
    EXPECT_EQ(vector_count, num_cells_) << "Should write all velocity vectors";

    std::cout << "  File: " << filename << ".vtk\n";
    std::cout << "  Grid: " << nx_ << "x" << ny_ << "x" << nz_ << "\n";
    std::cout << "  Vectors written: " << vector_count << " / " << num_cells_ << "\n";
    std::cout << "  PASS\n\n";
}

/**
 * @brief Test 2: Write structured grid with scalars and vectors
 */
TEST_F(VTKVectorIOTest, WriteStructuredGridWithVectors) {
    std::cout << "=== Test: Write Structured Grid with Scalars and Vectors ===\n";

    std::string filename = "test_vtk_multiphysics";

    // Write full multiphysics data
    ASSERT_NO_THROW({
        io::VTKWriter::writeStructuredGridWithVectors(
            filename,
            temperature_.data(),
            liquid_fraction_.data(),
            phase_state_.data(),
            nullptr,  // fill_level not used in this test
            velocity_x_.data(),
            velocity_y_.data(),
            velocity_z_.data(),
            nx_, ny_, nz_,
            dx_, dy_, dz_
        );
    }) << "Writing multiphysics data should not throw exception";

    // Verify file was created
    std::ifstream file(filename + ".vtk");
    ASSERT_TRUE(file.is_open()) << "VTK file should be created";

    // Read and verify file contents
    std::string line;
    bool found_vectors = false;
    bool found_temperature = false;
    bool found_liquid_fraction = false;
    bool found_phase_state = false;

    while (std::getline(file, line)) {
        if (line.find("VECTORS Velocity float") != std::string::npos) {
            found_vectors = true;
        }
        if (line.find("SCALARS Temperature float") != std::string::npos) {
            found_temperature = true;
        }
        if (line.find("SCALARS LiquidFraction float") != std::string::npos) {
            found_liquid_fraction = true;
        }
        if (line.find("SCALARS PhaseState float") != std::string::npos) {
            found_phase_state = true;
        }
    }

    file.close();

    EXPECT_TRUE(found_vectors) << "VECTORS Velocity should be present";
    EXPECT_TRUE(found_temperature) << "SCALARS Temperature should be present";
    EXPECT_TRUE(found_liquid_fraction) << "SCALARS LiquidFraction should be present";
    EXPECT_TRUE(found_phase_state) << "SCALARS PhaseState should be present";

    std::cout << "  File: " << filename << ".vtk\n";
    std::cout << "  Fields found:\n";
    std::cout << "    - Velocity (VECTORS): " << (found_vectors ? "YES" : "NO") << "\n";
    std::cout << "    - Temperature (SCALARS): " << (found_temperature ? "YES" : "NO") << "\n";
    std::cout << "    - LiquidFraction (SCALARS): " << (found_liquid_fraction ? "YES" : "NO") << "\n";
    std::cout << "    - PhaseState (SCALARS): " << (found_phase_state ? "YES" : "NO") << "\n";
    std::cout << "  PASS\n\n";
}

/**
 * @brief Test 3: Verify vector values are correctly written
 */
TEST_F(VTKVectorIOTest, VerifyVectorValues) {
    std::cout << "=== Test: Verify Vector Values ===\n";

    std::string filename = "test_vtk_verify_values";

    // Write vector field
    io::VTKWriter::writeVectorField(
        filename,
        velocity_x_.data(),
        velocity_y_.data(),
        velocity_z_.data(),
        nx_, ny_, nz_,
        dx_, dy_, dz_,
        "Velocity"
    );

    // Read back and verify a few sample points
    std::ifstream file(filename + ".vtk");
    ASSERT_TRUE(file.is_open());

    std::string line;
    bool in_data = false;
    int data_idx = 0;
    int verified_count = 0;
    const float tolerance = 1e-5f;

    while (std::getline(file, line)) {
        if (line.find("VECTORS Velocity float") != std::string::npos) {
            in_data = true;
            continue;
        }

        if (in_data && data_idx < num_cells_) {
            std::istringstream iss(line);
            float vx, vy, vz;
            if (iss >> vx >> vy >> vz) {
                // Verify against expected values
                float expected_vx = velocity_x_[data_idx];
                float expected_vy = velocity_y_[data_idx];
                float expected_vz = velocity_z_[data_idx];

                float error_x = std::abs(vx - expected_vx);
                float error_y = std::abs(vy - expected_vy);
                float error_z = std::abs(vz - expected_vz);

                EXPECT_LT(error_x, tolerance) << "vx mismatch at index " << data_idx;
                EXPECT_LT(error_y, tolerance) << "vy mismatch at index " << data_idx;
                EXPECT_LT(error_z, tolerance) << "vz mismatch at index " << data_idx;

                verified_count++;
                data_idx++;
            }
        }
    }

    file.close();

    EXPECT_EQ(verified_count, num_cells_) << "Should verify all vector values";

    std::cout << "  Verified " << verified_count << " vector values\n";
    std::cout << "  Maximum error tolerance: " << tolerance << "\n";
    std::cout << "  PASS\n\n";
}

/**
 * @brief Test 4: Check VTK format compliance
 */
TEST_F(VTKVectorIOTest, VTKFormatCompliance) {
    std::cout << "=== Test: VTK Format Compliance ===\n";

    std::string filename = "test_vtk_format_check";

    io::VTKWriter::writeStructuredGridWithVectors(
        filename,
        temperature_.data(),
        liquid_fraction_.data(),
        phase_state_.data(),
        nullptr,  // fill_level not used in this test
        velocity_x_.data(),
        velocity_y_.data(),
        velocity_z_.data(),
        nx_, ny_, nz_,
        dx_, dy_, dz_
    );

    std::ifstream file(filename + ".vtk");
    ASSERT_TRUE(file.is_open());

    std::vector<std::string> required_keywords = {
        "# vtk DataFile Version",
        "ASCII",
        "DATASET STRUCTURED_POINTS",
        "DIMENSIONS",
        "ORIGIN",
        "SPACING",
        "POINT_DATA",
        "VECTORS",
        "SCALARS"
    };

    std::vector<bool> found(required_keywords.size(), false);
    std::string line;

    while (std::getline(file, line)) {
        for (size_t i = 0; i < required_keywords.size(); ++i) {
            if (line.find(required_keywords[i]) != std::string::npos) {
                found[i] = true;
            }
        }
    }

    file.close();

    std::cout << "  VTK format compliance check:\n";
    for (size_t i = 0; i < required_keywords.size(); ++i) {
        std::cout << "    " << required_keywords[i] << ": "
                  << (found[i] ? "FOUND" : "MISSING") << "\n";
        EXPECT_TRUE(found[i]) << "Required keyword missing: " << required_keywords[i];
    }

    bool all_found = std::all_of(found.begin(), found.end(), [](bool v) { return v; });
    EXPECT_TRUE(all_found) << "All required VTK keywords should be present";

    std::cout << "  PASS\n\n";
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
