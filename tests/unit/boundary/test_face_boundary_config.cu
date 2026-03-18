/**
 * @file test_face_boundary_config.cu
 * @brief Unit tests for per-face boundary condition configuration
 */

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include "physics/multiphysics_solver.h"

using namespace lbm::physics;

class FaceBoundaryConfigTest : public ::testing::Test {
protected:
    void SetUp() override {
        cudaSetDevice(0);
    }
};

// --- FaceBoundaryConfig struct tests ---

TEST_F(FaceBoundaryConfigTest, DefaultValues) {
    FaceBoundaryConfig cfg;

    // X and Y: periodic by default
    EXPECT_EQ(cfg.x_min, BoundaryType::PERIODIC);
    EXPECT_EQ(cfg.x_max, BoundaryType::PERIODIC);
    EXPECT_EQ(cfg.y_min, BoundaryType::PERIODIC);
    EXPECT_EQ(cfg.y_max, BoundaryType::PERIODIC);

    // Z: wall at bottom, periodic at top (LPBF default)
    EXPECT_EQ(cfg.z_min, BoundaryType::WALL);
    EXPECT_EQ(cfg.z_max, BoundaryType::PERIODIC);

    // Thermal: periodic sides, convective bottom, adiabatic top
    EXPECT_EQ(cfg.thermal_x_min, ThermalBCType::PERIODIC);
    EXPECT_EQ(cfg.thermal_x_max, ThermalBCType::PERIODIC);
    EXPECT_EQ(cfg.thermal_y_min, ThermalBCType::PERIODIC);
    EXPECT_EQ(cfg.thermal_y_max, ThermalBCType::PERIODIC);
    EXPECT_EQ(cfg.thermal_z_min, ThermalBCType::CONVECTIVE);
    EXPECT_EQ(cfg.thermal_z_max, ThermalBCType::ADIABATIC);
}

TEST_F(FaceBoundaryConfigTest, PeriodicAxisDetection) {
    FaceBoundaryConfig cfg;

    // Default: X and Y periodic, Z not periodic (z_min is WALL)
    EXPECT_TRUE(cfg.isPeriodicX());
    EXPECT_TRUE(cfg.isPeriodicY());
    EXPECT_FALSE(cfg.isPeriodicZ());

    // Make Z fully periodic
    cfg.z_min = BoundaryType::PERIODIC;
    EXPECT_TRUE(cfg.isPeriodicZ());

    // One WALL face breaks periodicity
    cfg.x_max = BoundaryType::WALL;
    EXPECT_FALSE(cfg.isPeriodicX());
}

TEST_F(FaceBoundaryConfigTest, FluidBCDerivation) {
    FaceBoundaryConfig cfg;

    // Default: X/Y periodic, Z has one wall face -> WALL
    EXPECT_EQ(cfg.fluidBCX(), BoundaryType::PERIODIC);
    EXPECT_EQ(cfg.fluidBCY(), BoundaryType::PERIODIC);
    EXPECT_EQ(cfg.fluidBCZ(), BoundaryType::WALL);

    // Make both Z faces periodic
    cfg.z_min = BoundaryType::PERIODIC;
    EXPECT_EQ(cfg.fluidBCZ(), BoundaryType::PERIODIC);
}

TEST_F(FaceBoundaryConfigTest, VofBCDerivation) {
    FaceBoundaryConfig cfg;

    EXPECT_EQ(cfg.vofBCX(), VOFSolver::BoundaryType::PERIODIC);
    EXPECT_EQ(cfg.vofBCY(), VOFSolver::BoundaryType::PERIODIC);
    EXPECT_EQ(cfg.vofBCZ(), VOFSolver::BoundaryType::WALL);
}

TEST_F(FaceBoundaryConfigTest, ThermalBCForFace) {
    FaceBoundaryConfig cfg;

    EXPECT_EQ(cfg.thermalBCForFace(0), ThermalBCType::PERIODIC);   // x_min
    EXPECT_EQ(cfg.thermalBCForFace(1), ThermalBCType::PERIODIC);   // x_max
    EXPECT_EQ(cfg.thermalBCForFace(2), ThermalBCType::PERIODIC);   // y_min
    EXPECT_EQ(cfg.thermalBCForFace(3), ThermalBCType::PERIODIC);   // y_max
    EXPECT_EQ(cfg.thermalBCForFace(4), ThermalBCType::CONVECTIVE); // z_min
    EXPECT_EQ(cfg.thermalBCForFace(5), ThermalBCType::ADIABATIC);  // z_max
    EXPECT_EQ(cfg.thermalBCForFace(99), ThermalBCType::PERIODIC);  // invalid -> PERIODIC
}

TEST_F(FaceBoundaryConfigTest, HasAnyThermalBC) {
    FaceBoundaryConfig cfg;

    EXPECT_TRUE(cfg.hasAnyThermalBC(ThermalBCType::PERIODIC));    // sides
    EXPECT_TRUE(cfg.hasAnyThermalBC(ThermalBCType::CONVECTIVE));  // z_min
    EXPECT_TRUE(cfg.hasAnyThermalBC(ThermalBCType::ADIABATIC));   // z_max
    EXPECT_FALSE(cfg.hasAnyThermalBC(ThermalBCType::DIRICHLET));
    EXPECT_FALSE(cfg.hasAnyThermalBC(ThermalBCType::RADIATION));
}

TEST_F(FaceBoundaryConfigTest, SetUniform) {
    FaceBoundaryConfig cfg;
    cfg.setUniform(BoundaryType::WALL, ThermalBCType::DIRICHLET);

    EXPECT_EQ(cfg.x_min, BoundaryType::WALL);
    EXPECT_EQ(cfg.x_max, BoundaryType::WALL);
    EXPECT_EQ(cfg.y_min, BoundaryType::WALL);
    EXPECT_EQ(cfg.y_max, BoundaryType::WALL);
    EXPECT_EQ(cfg.z_min, BoundaryType::WALL);
    EXPECT_EQ(cfg.z_max, BoundaryType::WALL);

    EXPECT_EQ(cfg.thermal_x_min, ThermalBCType::DIRICHLET);
    EXPECT_EQ(cfg.thermal_z_max, ThermalBCType::DIRICHLET);

    EXPECT_FALSE(cfg.isPeriodicX());
    EXPECT_FALSE(cfg.isPeriodicY());
    EXPECT_FALSE(cfg.isPeriodicZ());
}

TEST_F(FaceBoundaryConfigTest, FromLegacyPeriodic) {
    auto cfg = FaceBoundaryConfig::fromLegacy(0);

    EXPECT_EQ(cfg.x_min, BoundaryType::PERIODIC);
    EXPECT_EQ(cfg.z_max, BoundaryType::PERIODIC);
    EXPECT_EQ(cfg.thermal_z_min, ThermalBCType::PERIODIC);
    EXPECT_TRUE(cfg.isPeriodicX());
    EXPECT_TRUE(cfg.isPeriodicZ());
}

TEST_F(FaceBoundaryConfigTest, FromLegacyWallDirichlet) {
    auto cfg = FaceBoundaryConfig::fromLegacy(1);

    EXPECT_EQ(cfg.x_min, BoundaryType::WALL);
    EXPECT_EQ(cfg.z_max, BoundaryType::WALL);
    EXPECT_EQ(cfg.thermal_z_min, ThermalBCType::DIRICHLET);
    EXPECT_EQ(cfg.thermal_x_max, ThermalBCType::DIRICHLET);
    EXPECT_FALSE(cfg.isPeriodicX());
}

TEST_F(FaceBoundaryConfigTest, FromLegacyWallAdiabatic) {
    auto cfg = FaceBoundaryConfig::fromLegacy(2);

    EXPECT_EQ(cfg.x_min, BoundaryType::WALL);
    EXPECT_EQ(cfg.thermal_z_min, ThermalBCType::ADIABATIC);
    EXPECT_EQ(cfg.thermal_x_max, ThermalBCType::ADIABATIC);
}

// --- MultiphysicsConfig integration tests ---

TEST_F(FaceBoundaryConfigTest, ConfigDefaultBoundaries) {
    MultiphysicsConfig config;

    // Default FaceBoundaryConfig should be present
    EXPECT_TRUE(config.boundaries.isPeriodicX());
    EXPECT_TRUE(config.boundaries.isPeriodicY());
    EXPECT_FALSE(config.boundaries.isPeriodicZ());  // z_min is WALL

    // Legacy boundary_type should still default to 0
    EXPECT_EQ(config.boundary_type, 0);
}

TEST_F(FaceBoundaryConfigTest, ConfigCopy) {
    MultiphysicsConfig config;
    config.boundaries.z_min = BoundaryType::PERIODIC;
    config.boundaries.thermal_z_min = ThermalBCType::RADIATION;
    config.boundaries.radiation_emissivity = 0.5f;

    MultiphysicsConfig copy(config);
    EXPECT_TRUE(copy.boundaries.isPeriodicZ());
    EXPECT_EQ(copy.boundaries.thermal_z_min, ThermalBCType::RADIATION);
    EXPECT_FLOAT_EQ(copy.boundaries.radiation_emissivity, 0.5f);
}

TEST_F(FaceBoundaryConfigTest, ConfigAssignment) {
    MultiphysicsConfig config;
    config.boundaries.setUniform(BoundaryType::WALL, ThermalBCType::ADIABATIC);

    MultiphysicsConfig other;
    other = config;
    EXPECT_FALSE(other.boundaries.isPeriodicX());
    EXPECT_EQ(other.boundaries.thermal_x_min, ThermalBCType::ADIABATIC);
}

// --- LPBF-typical configuration test ---

TEST_F(FaceBoundaryConfigTest, LPBFTypicalConfig) {
    // Typical LPBF setup:
    // - Periodic sides (x, y)
    // - No-slip substrate at z=0 with convective cooling
    // - Open top with radiation
    FaceBoundaryConfig cfg;
    cfg.x_min = BoundaryType::PERIODIC;
    cfg.x_max = BoundaryType::PERIODIC;
    cfg.y_min = BoundaryType::PERIODIC;
    cfg.y_max = BoundaryType::PERIODIC;
    cfg.z_min = BoundaryType::WALL;
    cfg.z_max = BoundaryType::PERIODIC;

    cfg.thermal_x_min = ThermalBCType::PERIODIC;
    cfg.thermal_x_max = ThermalBCType::PERIODIC;
    cfg.thermal_y_min = ThermalBCType::PERIODIC;
    cfg.thermal_y_max = ThermalBCType::PERIODIC;
    cfg.thermal_z_min = ThermalBCType::CONVECTIVE;
    cfg.thermal_z_max = ThermalBCType::RADIATION;

    cfg.convective_h = 1000.0f;
    cfg.convective_T_inf = 300.0f;
    cfg.radiation_emissivity = 0.3f;
    cfg.radiation_T_ambient = 300.0f;

    // Verify derived BCs
    EXPECT_EQ(cfg.fluidBCX(), BoundaryType::PERIODIC);
    EXPECT_EQ(cfg.fluidBCY(), BoundaryType::PERIODIC);
    EXPECT_EQ(cfg.fluidBCZ(), BoundaryType::WALL);

    // Verify thermal BC dispatch
    EXPECT_EQ(cfg.thermalBCForFace(4), ThermalBCType::CONVECTIVE);  // z_min
    EXPECT_EQ(cfg.thermalBCForFace(5), ThermalBCType::RADIATION);   // z_max

    EXPECT_TRUE(cfg.hasAnyThermalBC(ThermalBCType::CONVECTIVE));
    EXPECT_TRUE(cfg.hasAnyThermalBC(ThermalBCType::RADIATION));
    EXPECT_FALSE(cfg.hasAnyThermalBC(ThermalBCType::DIRICHLET));
}
