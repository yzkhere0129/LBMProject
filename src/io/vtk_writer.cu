/**
 * @file vtk_writer.cu
 * @brief Implementation of VTK file writer
 */

#include "io/vtk_writer.h"
#include <iostream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include "utils/cuda_check.h"

namespace lbm {
namespace io {

void VTKWriter::writeStructuredPoints(
    const std::string& filename,
    const float* data,
    int nx, int ny, int nz,
    float dx, float dy, float dz,
    const std::string& field_name) {

    std::ofstream file(filename + ".vtk");
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + filename + ".vtk");
    }

    // VTK header
    file << "# vtk DataFile Version 3.0\n";
    file << "LBM Simulation Data\n";
    file << "ASCII\n";
    file << "DATASET STRUCTURED_POINTS\n";

    // Grid definition
    file << "DIMENSIONS " << nx << " " << ny << " " << nz << "\n";
    file << "ORIGIN 0.0 0.0 0.0\n";
    file << "SPACING " << dx << " " << dy << " " << dz << "\n";

    // Scalar field data
    int n_points = nx * ny * nz;
    file << "\nPOINT_DATA " << n_points << "\n";
    file << "SCALARS " << field_name << " float 1\n";
    file << "LOOKUP_TABLE default\n";

    // Write data in VTK order (x varies fastest)
    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                int idx = i + j * nx + k * nx * ny;
                file << data[idx] << "\n";
            }
        }
    }

    file.close();
    std::cout << "VTK file written: " << filename << ".vtk\n";
}

void VTKWriter::writeLaserHeatingSnapshot(
    const std::string& filename,
    const float* temperature,
    float laser_x, float laser_y, float laser_z,
    float laser_radius,
    int nx, int ny, int nz,
    float dx, float dy, float dz,
    float time) {

    std::ofstream file(filename + ".vtk");
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + filename + ".vtk");
    }

    // VTK header with time information
    file << "# vtk DataFile Version 3.0\n";
    file << "Laser Heating Simulation at t=" << std::scientific << time << " s\n";
    file << "ASCII\n";
    file << "DATASET STRUCTURED_POINTS\n";

    // Grid definition
    file << "DIMENSIONS " << nx << " " << ny << " " << nz << "\n";
    file << "ORIGIN 0.0 0.0 0.0\n";
    file << "SPACING " << dx << " " << dy << " " << dz << "\n";

    // Temperature field
    int n_points = nx * ny * nz;
    file << "\nPOINT_DATA " << n_points << "\n";
    file << "SCALARS Temperature float 1\n";
    file << "LOOKUP_TABLE default\n";

    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                int idx = i + j * nx + k * nx * ny;
                file << temperature[idx] << "\n";
            }
        }
    }

    // Laser position indicator field
    file << "\nSCALARS LaserIntensity float 1\n";
    file << "LOOKUP_TABLE default\n";

    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                // Calculate position in physical coordinates
                float x = i * dx;
                float y = j * dy;
                float z = k * dz;

                // Distance from laser center
                float r2 = (x - laser_x) * (x - laser_x) +
                          (y - laser_y) * (y - laser_y);

                // Gaussian-like intensity profile at surface
                float intensity = 0.0f;
                if (z <= laser_z + 3.0f * dz) {  // Near surface
                    if (r2 < laser_radius * laser_radius) {
                        intensity = expf(-2.0f * r2 / (laser_radius * laser_radius));
                    }
                }

                file << intensity << "\n";
            }
        }
    }

    file.close();
}

std::string VTKWriter::getTimeSeriesFilename(
    const std::string& base_filename,
    int step) {

    std::ostringstream oss;
    oss << base_filename << "_" << std::setw(6) << std::setfill('0') << step;
    return oss.str();
}

void VTKWriter::write2DSlice(
    const std::string& filename,
    const float* data_3d,
    int nx, int ny, int nz,
    int slice_axis, int slice_index,
    float dx, float dy, float dz,
    const std::string& field_name) {

    std::ofstream file(filename + ".vtk");
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + filename + ".vtk");
    }

    // Determine 2D dimensions based on slice axis
    int n2d_x, n2d_y;
    float d2d_x, d2d_y;

    if (slice_axis == 0) {  // YZ plane
        n2d_x = ny;
        n2d_y = nz;
        d2d_x = dy;
        d2d_y = dz;
    } else if (slice_axis == 1) {  // XZ plane
        n2d_x = nx;
        n2d_y = nz;
        d2d_x = dx;
        d2d_y = dz;
    } else {  // XY plane (z-slice)
        n2d_x = nx;
        n2d_y = ny;
        d2d_x = dx;
        d2d_y = dy;
    }

    // VTK header for 2D data
    file << "# vtk DataFile Version 3.0\n";
    file << "2D Slice of " << field_name << "\n";
    file << "ASCII\n";
    file << "DATASET STRUCTURED_POINTS\n";
    file << "DIMENSIONS " << n2d_x << " " << n2d_y << " 1\n";
    file << "ORIGIN 0.0 0.0 0.0\n";
    file << "SPACING " << d2d_x << " " << d2d_y << " 1.0\n";

    // Extract and write 2D slice
    int n_points_2d = n2d_x * n2d_y;
    file << "\nPOINT_DATA " << n_points_2d << "\n";
    file << "SCALARS " << field_name << " float 1\n";
    file << "LOOKUP_TABLE default\n";

    if (slice_axis == 2) {  // XY plane (most common)
        int k = slice_index;
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                int idx = i + j * nx + k * nx * ny;
                file << data_3d[idx] << "\n";
            }
        }
    } else if (slice_axis == 1) {  // XZ plane
        int j = slice_index;
        for (int k = 0; k < nz; ++k) {
            for (int i = 0; i < nx; ++i) {
                int idx = i + j * nx + k * nx * ny;
                file << data_3d[idx] << "\n";
            }
        }
    } else {  // YZ plane
        int i = slice_index;
        for (int k = 0; k < nz; ++k) {
            for (int j = 0; j < ny; ++j) {
                int idx = i + j * nx + k * nx * ny;
                file << data_3d[idx] << "\n";
            }
        }
    }

    file.close();
}

void VTKWriter::writeStructuredGrid(
    const std::string& filename,
    const float* field1,
    const float* field2,
    const float* field3,
    int nx, int ny, int nz,
    float dx, float dy, float dz,
    const std::string& field1_name,
    const std::string& field2_name,
    const std::string& field3_name) {

    // Ensure .vtk extension
    std::string output_filename = filename;
    if (output_filename.substr(output_filename.length() - 4) != ".vtk") {
        output_filename += ".vtk";
    }

    std::ofstream file(output_filename);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + output_filename);
    }

    // VTK header
    file << "# vtk DataFile Version 3.0\n";
    file << "LBM Phase Change Simulation Data\n";
    file << "ASCII\n";
    file << "DATASET STRUCTURED_POINTS\n";

    // Grid definition
    file << "DIMENSIONS " << nx << " " << ny << " " << nz << "\n";
    file << "ORIGIN 0.0 0.0 0.0\n";
    file << "SPACING " << dx << " " << dy << " " << dz << "\n";

    // Point data section
    int n_points = nx * ny * nz;
    file << "\nPOINT_DATA " << n_points << "\n";

    // Write first field
    file << "SCALARS " << field1_name << " float 1\n";
    file << "LOOKUP_TABLE default\n";
    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                int idx = i + j * nx + k * nx * ny;
                file << field1[idx] << "\n";
            }
        }
    }

    // Write second field
    file << "\nSCALARS " << field2_name << " float 1\n";
    file << "LOOKUP_TABLE default\n";
    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                int idx = i + j * nx + k * nx * ny;
                file << field2[idx] << "\n";
            }
        }
    }

    // Write third field
    file << "\nSCALARS " << field3_name << " float 1\n";
    file << "LOOKUP_TABLE default\n";
    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                int idx = i + j * nx + k * nx * ny;
                file << field3[idx] << "\n";
            }
        }
    }

    file.close();
}

void VTKWriter::writeStructuredGridWithVectors(
    const std::string& filename,
    const float* temperature,
    const float* liquid_fraction,
    const float* phase_state,
    const float* fill_level,
    const float* velocity_x,
    const float* velocity_y,
    const float* velocity_z,
    int nx, int ny, int nz,
    float dx, float dy, float dz) {

    // Ensure .vtk extension
    std::string output_filename = filename;
    if (output_filename.substr(output_filename.length() - 4) != ".vtk") {
        output_filename += ".vtk";
    }

    std::ofstream file(output_filename);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + output_filename);
    }

    // VTK header
    file << "# vtk DataFile Version 3.0\n";
    file << "LBM Multiphysics Simulation with Flow\n";
    file << "ASCII\n";
    file << "DATASET STRUCTURED_POINTS\n";

    // Grid definition
    file << "DIMENSIONS " << nx << " " << ny << " " << nz << "\n";
    file << "ORIGIN 0.0 0.0 0.0\n";
    file << "SPACING " << dx << " " << dy << " " << dz << "\n";

    // Point data section
    int n_points = nx * ny * nz;
    file << "\nPOINT_DATA " << n_points << "\n";

    // Write velocity vector field FIRST (ParaView convention)
    // VTK VECTORS format: each line has 3 components (vx vy vz)
    file << "VECTORS Velocity float\n";
    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                int idx = i + j * nx + k * nx * ny;
                file << velocity_x[idx] << " "
                     << velocity_y[idx] << " "
                     << velocity_z[idx] << "\n";
            }
        }
    }

    // Write temperature scalar field
    file << "\nSCALARS Temperature float 1\n";
    file << "LOOKUP_TABLE default\n";
    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                int idx = i + j * nx + k * nx * ny;
                file << temperature[idx] << "\n";
            }
        }
    }

    // Write liquid fraction scalar field
    file << "\nSCALARS LiquidFraction float 1\n";
    file << "LOOKUP_TABLE default\n";
    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                int idx = i + j * nx + k * nx * ny;
                file << liquid_fraction[idx] << "\n";
            }
        }
    }

    // Write phase state scalar field
    file << "\nSCALARS PhaseState float 1\n";
    file << "LOOKUP_TABLE default\n";
    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                int idx = i + j * nx + k * nx * ny;
                file << phase_state[idx] << "\n";
            }
        }
    }

    // Write VOF fill level scalar field (for free surface visualization)
    // Only write if fill_level data is provided
    if (fill_level != nullptr) {
        file << "\nSCALARS FillLevel float 1\n";
        file << "LOOKUP_TABLE default\n";
        for (int k = 0; k < nz; ++k) {
            for (int j = 0; j < ny; ++j) {
                for (int i = 0; i < nx; ++i) {
                    int idx = i + j * nx + k * nx * ny;
                    file << fill_level[idx] << "\n";
                }
            }
        }
    }

    file.close();
}

void VTKWriter::writeVectorField(
    const std::string& filename,
    const float* velocity_x,
    const float* velocity_y,
    const float* velocity_z,
    int nx, int ny, int nz,
    float dx, float dy, float dz,
    const std::string& field_name) {

    // Ensure .vtk extension
    std::string output_filename = filename;
    if (output_filename.substr(output_filename.length() - 4) != ".vtk") {
        output_filename += ".vtk";
    }

    std::ofstream file(output_filename);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + output_filename);
    }

    // VTK header
    file << "# vtk DataFile Version 3.0\n";
    file << "Vector Field: " << field_name << "\n";
    file << "ASCII\n";
    file << "DATASET STRUCTURED_POINTS\n";

    // Grid definition
    file << "DIMENSIONS " << nx << " " << ny << " " << nz << "\n";
    file << "ORIGIN 0.0 0.0 0.0\n";
    file << "SPACING " << dx << " " << dy << " " << dz << "\n";

    // Vector field data
    int n_points = nx * ny * nz;
    file << "\nPOINT_DATA " << n_points << "\n";
    file << "VECTORS " << field_name << " float\n";

    // Write vector components (vx, vy, vz) on each line
    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                int idx = i + j * nx + k * nx * ny;
                file << velocity_x[idx] << " "
                     << velocity_y[idx] << " "
                     << velocity_z[idx] << "\n";
            }
        }
    }

    file.close();
}

} // namespace io
} // namespace lbm