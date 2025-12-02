/**
 * @file vtk_writer.h
 * @brief VTK file writer for scientific visualization
 *
 * Provides utilities to export LBM simulation data to VTK format
 * for visualization in ParaView and other scientific visualization tools.
 */

#pragma once
#include <string>
#include <fstream>

namespace lbm {
namespace io {

/**
 * @brief VTK file writer for structured grid data
 *
 * Writes scalar and vector fields to VTK legacy format files.
 * Supports both single snapshots and time series output.
 */
class VTKWriter {
public:
    /**
     * @brief Write 3D scalar field as VTK structured points
     *
     * @param filename Output filename (without extension)
     * @param data Scalar field data (row-major order)
     * @param nx,ny,nz Grid dimensions
     * @param dx,dy,dz Grid spacing in meters
     * @param field_name Name of the scalar field
     */
    static void writeStructuredPoints(
        const std::string& filename,
        const float* data,
        int nx, int ny, int nz,
        float dx, float dy, float dz,
        const std::string& field_name = "Temperature"
    );

    /**
     * @brief Write temperature field with laser position indicator
     *
     * Exports both temperature field and laser spot location for
     * combined visualization in ParaView.
     *
     * @param filename Output filename
     * @param temperature Temperature field data
     * @param laser_x,laser_y,laser_z Laser center position in meters
     * @param laser_radius Laser spot radius in meters
     * @param nx,ny,nz Grid dimensions
     * @param dx,dy,dz Grid spacing
     * @param time Current simulation time in seconds
     */
    static void writeLaserHeatingSnapshot(
        const std::string& filename,
        const float* temperature,
        float laser_x, float laser_y, float laser_z,
        float laser_radius,
        int nx, int ny, int nz,
        float dx, float dy, float dz,
        float time
    );

    /**
     * @brief Generate timestamped filename for animation series
     *
     * @param base_filename Base name for the file series
     * @param step Current time step number
     * @return Formatted filename with step number
     */
    static std::string getTimeSeriesFilename(
        const std::string& base_filename,
        int step
    );

    /**
     * @brief Write 2D slice from 3D data
     *
     * Extracts and writes a 2D slice from 3D field data.
     * Useful for quick 2D visualization and debugging.
     *
     * @param filename Output filename
     * @param data_3d Full 3D field data
     * @param nx,ny,nz 3D grid dimensions
     * @param slice_axis Axis to slice along (0=x, 1=y, 2=z)
     * @param slice_index Index of the slice
     * @param dx,dy,dz Grid spacing
     * @param field_name Name of the field
     */
    static void write2DSlice(
        const std::string& filename,
        const float* data_3d,
        int nx, int ny, int nz,
        int slice_axis, int slice_index,
        float dx, float dy, float dz,
        const std::string& field_name = "Temperature"
    );

    /**
     * @brief Write structured grid with multiple scalar fields
     *
     * Exports 3D structured grid data with three scalar fields for
     * comprehensive visualization of phase change simulations.
     *
     * @param filename Output filename
     * @param field1 First scalar field data
     * @param field2 Second scalar field data
     * @param field3 Third scalar field data
     * @param nx,ny,nz Grid dimensions
     * @param dx,dy,dz Grid spacing
     * @param field1_name Name of first field
     * @param field2_name Name of second field
     * @param field3_name Name of third field
     */
    static void writeStructuredGrid(
        const std::string& filename,
        const float* field1,
        const float* field2,
        const float* field3,
        int nx, int ny, int nz,
        float dx, float dy, float dz,
        const std::string& field1_name = "Temperature",
        const std::string& field2_name = "LiquidFraction",
        const std::string& field3_name = "PhaseState"
    );

    /**
     * @brief Write structured grid with multiple scalars and a vector field
     *
     * Exports 3D structured grid with scalar fields (temperature, liquid fraction, phase, fill level)
     * and a vector field (velocity) for comprehensive multiphysics visualization.
     * The vector field is written in VTK VECTORS format for proper visualization
     * of flow patterns using glyphs, streamlines, and stream tracers in ParaView.
     *
     * @param filename Output filename (without extension)
     * @param temperature Temperature field data [K]
     * @param liquid_fraction Liquid fraction field data [0-1]
     * @param phase_state Phase state field data (0=solid, 1=mushy, 2=liquid)
     * @param fill_level VOF fill level field data [0-1] for free surface tracking
     * @param velocity_x Velocity field x-component [m/s]
     * @param velocity_y Velocity field y-component [m/s]
     * @param velocity_z Velocity field z-component [m/s]
     * @param nx,ny,nz Grid dimensions
     * @param dx,dy,dz Grid spacing [m]
     */
    static void writeStructuredGridWithVectors(
        const std::string& filename,
        const float* temperature,
        const float* liquid_fraction,
        const float* phase_state,
        const float* fill_level,
        const float* velocity_x,
        const float* velocity_y,
        const float* velocity_z,
        int nx, int ny, int nz,
        float dx, float dy, float dz
    );

    /**
     * @brief Write velocity vector field only (for debugging)
     *
     * Exports only the velocity vector field for focused flow visualization.
     *
     * @param filename Output filename (without extension)
     * @param velocity_x Velocity field x-component [m/s]
     * @param velocity_y Velocity field y-component [m/s]
     * @param velocity_z Velocity field z-component [m/s]
     * @param nx,ny,nz Grid dimensions
     * @param dx,dy,dz Grid spacing [m]
     * @param field_name Name of the vector field (default: "Velocity")
     */
    static void writeVectorField(
        const std::string& filename,
        const float* velocity_x,
        const float* velocity_y,
        const float* velocity_z,
        int nx, int ny, int nz,
        float dx, float dy, float dz,
        const std::string& field_name = "Velocity"
    );
};

} // namespace io
} // namespace lbm