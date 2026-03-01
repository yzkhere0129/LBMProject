/**
 * @file field_registry.cpp
 * @brief Implementation of FieldRegistry for configurable VTK field output
 */

#include "io/field_registry.h"
#include <algorithm>

namespace lbm {
namespace io {

void FieldRegistry::registerScalar(const std::string& name, const float* device_ptr) {
    if (!device_ptr) return;
    // Overwrite if already registered
    for (auto& fd : scalars_) {
        if (fd.name == name) {
            fd.device_ptr = device_ptr;
            return;
        }
    }
    scalars_.emplace_back(name, device_ptr, 1);
}

void FieldRegistry::registerVector(const std::string& name,
                                   const float* vx, const float* vy, const float* vz) {
    if (!vx || !vy || !vz) return;
    // Overwrite if already registered
    for (size_t i = 0; i + 2 < vectors_.size(); i += 3) {
        if (vectors_[i].name == name + "_x") {
            vectors_[i].device_ptr     = vx;
            vectors_[i + 1].device_ptr = vy;
            vectors_[i + 2].device_ptr = vz;
            return;
        }
    }
    vectors_.emplace_back(name + "_x", vx, 3);
    vectors_.emplace_back(name + "_y", vy, 3);
    vectors_.emplace_back(name + "_z", vz, 3);
}

const float* FieldRegistry::getScalar(const std::string& name) const {
    for (const auto& fd : scalars_) {
        if (fd.name == name) return fd.device_ptr;
    }
    return nullptr;
}

bool FieldRegistry::hasField(const std::string& name) const {
    for (const auto& fd : scalars_) {
        if (fd.name == name) return true;
    }
    // For vectors, check the base name (without _x/_y/_z suffix)
    for (size_t i = 0; i < vectors_.size(); i += 3) {
        // vectors_[i].name is "basename_x", strip the "_x" to get base name
        const std::string& vname = vectors_[i].name;
        if (vname.size() > 2) {
            std::string base = vname.substr(0, vname.size() - 2);
            if (base == name) return true;
        }
    }
    return false;
}

std::vector<std::string> FieldRegistry::getFieldNames() const {
    std::vector<std::string> names;
    names.reserve(scalars_.size() + vectors_.size() / 3);
    for (const auto& fd : scalars_) {
        names.push_back(fd.name);
    }
    for (size_t i = 0; i < vectors_.size(); i += 3) {
        // Return the base name without "_x" suffix
        const std::string& vname = vectors_[i].name;
        if (vname.size() > 2) {
            names.push_back(vname.substr(0, vname.size() - 2));
        }
    }
    return names;
}

void FieldRegistry::clear() {
    scalars_.clear();
    vectors_.clear();
}

} // namespace io
} // namespace lbm
