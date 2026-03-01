/**
 * @file field_registry.h
 * @brief Configurable field output registry for VTK writer
 *
 * Decouples field selection from VTK writer implementation.
 * Fields are registered once at setup, then written based on user selection.
 * Adding new output fields no longer requires adding new VTKWriter methods.
 */

#pragma once

#include <string>
#include <vector>

namespace lbm {
namespace io {

/**
 * @brief Describes a single component of a field available for VTK output.
 *
 * For scalar fields, one descriptor holds the device pointer.
 * For vector fields, three consecutive descriptors hold _x, _y, _z components.
 */
struct FieldDescriptor {
    std::string name;          ///< Field name (e.g., "temperature", "velocity_x")
    const float* device_ptr;   ///< Device pointer to field data
    int components;            ///< 1 = scalar, 3 = vector

    FieldDescriptor(const std::string& n, const float* ptr, int comp = 1)
        : name(n), device_ptr(ptr), components(comp) {}
};

/**
 * @brief Registry of fields available for VTK output.
 *
 * Fields are registered at simulation setup. At output time, the caller
 * can write all fields or a named subset. Linear search by name is used
 * internally -- adequate for the expected count of <20 fields.
 */
class FieldRegistry {
public:
    /// Register a scalar field (temperature, pressure, fill_level, etc.)
    void registerScalar(const std::string& name, const float* device_ptr);

    /// Register a vector field (stores 3 consecutive descriptors: name_x, name_y, name_z)
    void registerVector(const std::string& name,
                        const float* vx, const float* vy, const float* vz);

    /// Get all registered scalar field descriptors
    const std::vector<FieldDescriptor>& getScalars() const { return scalars_; }

    /// Get all registered vector field descriptors (3 consecutive entries per vector)
    const std::vector<FieldDescriptor>& getVectors() const { return vectors_; }

    /// Get device pointer for a scalar field by name (returns nullptr if not found)
    const float* getScalar(const std::string& name) const;

    /// Check if a field (scalar or vector) is registered
    bool hasField(const std::string& name) const;

    /// Get list of all registered field names (scalars first, then vectors)
    std::vector<std::string> getFieldNames() const;

    /// Clear all registrations
    void clear();

private:
    std::vector<FieldDescriptor> scalars_;
    std::vector<FieldDescriptor> vectors_;  ///< Stored as 3 consecutive descriptors per vector
};

} // namespace io
} // namespace lbm
