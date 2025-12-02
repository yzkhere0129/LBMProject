/**
 * @file test_multiphysics_temp_anomaly.cu
 * @brief Diagnostic test using MultiphysicsSolver to reproduce Tier2 anomaly
 *
 * This test attempts to reproduce the temperature oscillation using the
 * full MultiphysicsSolver stack, since individual ThermalLBM tests pass.
 *
 * Key findings from test_temperature_anomaly.cu:
 *   - Test 1-6 mostly PASSED
 *   - ThermalLBM alone shows monotonic T_max increase
 *   - Bug is likely in MultiphysicsSolver coupling
 *
 * This test focuses on:
 *   - MultiphysicsSolver step() function
 *   - Phase change solver interaction
 *   - VOF/fluid coupling effects on thermal field
 */

#include "physics/multiphysics_solver.h"
#include <cuda_runtime.h>
#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <algorithm>
#include <cstring>

using namespace lbm::physics;

//==============================================================================
// Test Configuration
//==============================================================================

constexpr int NX = 200;
constexpr int NY = 100;
constexpr int NZ = 50;
constexpr float DX = 2e-6f;     // 2 um
constexpr float DT = 0.1e-6f;   // 0.1 us
constexpr float T_INIT = 300.0f;
constexpr float LASER_POWER = 10.0f;  // Absorbed power

//==============================================================================
// Helper Functions
//==============================================================================

struct TempStats {
    float T_max, T_min, T_avg;
    int max_x, max_y, max_z;
};

TempStats getStats(const float* temp, int nx, int ny, int nz) {
    TempStats s;
    s.T_max = -1e30f;
    s.T_min = 1e30f;
    s.T_avg = 0.0f;

    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                int idx = i + j*nx + k*nx*ny;
                float T = temp[idx];
                if (T > s.T_max) {
                    s.T_max = T;
                    s.max_x = i;
                    s.max_y = j;
                    s.max_z = k;
                }
                if (T < s.T_min) s.T_min = T;
                s.T_avg += T;
            }
        }
    }
    s.T_avg /= (nx * ny * nz);
    return s;
}

void printBanner(const char* title) {
    printf("\n");
    printf("================================================================\n");
    printf("  %s\n", title);
    printf("================================================================\n");
}

//==============================================================================
// TEST A: Thermal-Only MultiphysicsSolver
//==============================================================================

bool testThermalOnlyMultiphysics() {
    printBanner("TEST A: Thermal-Only MultiphysicsSolver");

    printf("Configuration:\n");
    printf("  Grid: %d x %d x %d\n", NX, NY, NZ);
    printf("  Physics: THERMAL + LASER only\n");
    printf("  Duration: 3000 us (30000 steps)\n");
    printf("\n");

    MultiphysicsConfig config;
    config.nx = NX;
    config.ny = NY;
    config.nz = NZ;
    config.dx = DX;
    config.dt = DT;

    // Enable ONLY thermal and laser
    config.enable_thermal = true;
    config.enable_thermal_advection = false;
    config.enable_phase_change = false;
    config.enable_fluid = false;
    config.enable_vof = false;
    config.enable_vof_advection = false;
    config.enable_surface_tension = false;
    config.enable_marangoni = false;
    config.enable_darcy = false;
    config.enable_buoyancy = false;
    config.enable_laser = true;
    config.enable_radiation_bc = true;
    config.enable_substrate_cooling = false;

    // Laser config (10W absorbed)
    config.laser_power = LASER_POWER / 0.35f;  // ~28.57W nominal
    config.laser_absorptivity = 0.35f;
    config.laser_spot_radius = 50e-6f;
    config.laser_penetration_depth = 10e-6f;
    config.laser_start_x = NX * DX / 2.0f;
    config.laser_start_y = NY * DX / 2.0f;
    config.laser_scan_vx = 0.0f;
    config.laser_scan_vy = 0.0f;
    config.laser_shutoff_time = -1.0f;  // Always on

    // Radiation BC
    config.emissivity = 0.35f;
    config.ambient_temperature = 300.0f;

    // Material (Ti6Al4V)
    strncpy(config.material.name, "Ti6Al4V", 63);
    config.material.rho_solid = 4420.0f;
    config.material.rho_liquid = 4110.0f;
    config.material.cp_solid = 670.0f;
    config.material.cp_liquid = 831.0f;
    config.material.k_solid = 7.0f;
    config.material.k_liquid = 30.0f;
    config.material.T_solidus = 1878.0f;
    config.material.T_liquidus = 1928.0f;
    config.material.L_fusion = 286000.0f;
    config.material.T_vaporization = 3560.0f;
    config.material.L_vaporization = 9830000.0f;
    config.material.emissivity = 0.35f;

    config.thermal_diffusivity = 7.0f / (config.material.rho_solid * config.material.cp_solid);

    MultiphysicsSolver solver(config);
    solver.initialize(T_INIT, 0.5f);  // Interface at 50% height

    int num_cells = NX * NY * NZ;
    std::vector<float> h_temp(num_cells);

    printf("Step    Time(us)    T_max(K)    T_min(K)    Max_loc(i,j,k)    Status\n");
    printf("----    --------    --------    --------    --------------    ------\n");

    float prev_T_max = T_INIT;
    int anomaly_count = 0;

    int total_steps = 30000;
    int print_interval = 1000;

    for (int step = 0; step < total_steps; ++step) {
        solver.step(DT);

        if ((step + 1) % print_interval == 0) {
            solver.copyTemperatureToHost(h_temp.data());
            TempStats stats = getStats(h_temp.data(), NX, NY, NZ);

            float time_us = (step + 1) * DT * 1e6f;
            float dT_max = stats.T_max - prev_T_max;

            const char* status = "RISING";
            if (dT_max < -50.0f) {
                status = "DROP <<< ANOMALY";
                anomaly_count++;
            } else if (dT_max < 0.0f) {
                status = "SLIGHT DROP";
            }

            printf("%5d   %8.1f    %8.1f    %8.1f    (%3d,%3d,%3d)      %s\n",
                   step + 1, time_us, stats.T_max, stats.T_min,
                   stats.max_x, stats.max_y, stats.max_z, status);

            prev_T_max = stats.T_max;
        }
    }

    printf("\n");
    printf("========================================\n");
    if (anomaly_count > 0) {
        printf("RESULT: FAIL - %d intervals with T_max drop > 50K\n", anomaly_count);
    } else {
        printf("RESULT: PASS - No significant T_max drops\n");
    }
    printf("========================================\n");

    return (anomaly_count == 0);
}

//==============================================================================
// TEST B: Thermal + Phase Change
//==============================================================================

bool testThermalWithPhaseChange() {
    printBanner("TEST B: Thermal + Phase Change");

    printf("Configuration:\n");
    printf("  Grid: %d x %d x %d\n", NX, NY, NZ);
    printf("  Physics: THERMAL + LASER + PHASE_CHANGE\n");
    printf("  Duration: 3000 us (30000 steps)\n");
    printf("\n");

    MultiphysicsConfig config;
    config.nx = NX;
    config.ny = NY;
    config.nz = NZ;
    config.dx = DX;
    config.dt = DT;

    // Enable thermal, laser, and phase change
    config.enable_thermal = true;
    config.enable_thermal_advection = false;
    config.enable_phase_change = true;   // <<< KEY DIFFERENCE
    config.enable_fluid = false;
    config.enable_vof = false;
    config.enable_vof_advection = false;
    config.enable_surface_tension = false;
    config.enable_marangoni = false;
    config.enable_darcy = false;
    config.enable_buoyancy = false;
    config.enable_laser = true;
    config.enable_radiation_bc = true;
    config.enable_substrate_cooling = false;

    // Laser config
    config.laser_power = LASER_POWER / 0.35f;
    config.laser_absorptivity = 0.35f;
    config.laser_spot_radius = 50e-6f;
    config.laser_penetration_depth = 10e-6f;
    config.laser_start_x = NX * DX / 2.0f;
    config.laser_start_y = NY * DX / 2.0f;
    config.laser_scan_vx = 0.0f;
    config.laser_scan_vy = 0.0f;
    config.laser_shutoff_time = -1.0f;

    config.emissivity = 0.35f;
    config.ambient_temperature = 300.0f;

    // Material
    strncpy(config.material.name, "Ti6Al4V", 63);
    config.material.rho_solid = 4420.0f;
    config.material.rho_liquid = 4110.0f;
    config.material.cp_solid = 670.0f;
    config.material.cp_liquid = 831.0f;
    config.material.k_solid = 7.0f;
    config.material.k_liquid = 30.0f;
    config.material.T_solidus = 1878.0f;
    config.material.T_liquidus = 1928.0f;
    config.material.L_fusion = 286000.0f;
    config.material.T_vaporization = 3560.0f;
    config.material.L_vaporization = 9830000.0f;
    config.material.emissivity = 0.35f;

    config.thermal_diffusivity = 7.0f / (config.material.rho_solid * config.material.cp_solid);

    MultiphysicsSolver solver(config);
    solver.initialize(T_INIT, 0.5f);

    int num_cells = NX * NY * NZ;
    std::vector<float> h_temp(num_cells);

    printf("Step    Time(us)    T_max(K)    T_min(K)    Status\n");
    printf("----    --------    --------    --------    ------\n");

    float prev_T_max = T_INIT;
    int anomaly_count = 0;

    int total_steps = 30000;
    int print_interval = 1000;

    for (int step = 0; step < total_steps; ++step) {
        solver.step(DT);

        if ((step + 1) % print_interval == 0) {
            solver.copyTemperatureToHost(h_temp.data());
            TempStats stats = getStats(h_temp.data(), NX, NY, NZ);

            float time_us = (step + 1) * DT * 1e6f;
            float dT_max = stats.T_max - prev_T_max;

            const char* status = "RISING";
            if (dT_max < -50.0f) {
                status = "DROP <<< ANOMALY";
                anomaly_count++;
            } else if (dT_max < 0.0f) {
                status = "SLIGHT DROP";
            }

            printf("%5d   %8.1f    %8.1f    %8.1f    %s\n",
                   step + 1, time_us, stats.T_max, stats.T_min, status);

            prev_T_max = stats.T_max;
        }
    }

    printf("\n");
    printf("========================================\n");
    if (anomaly_count > 0) {
        printf("RESULT: FAIL - %d intervals with T_max drop > 50K\n", anomaly_count);
        printf("  Phase change solver may be causing temperature oscillation.\n");
    } else {
        printf("RESULT: PASS - No significant T_max drops\n");
    }
    printf("========================================\n");

    return (anomaly_count == 0);
}

//==============================================================================
// TEST C: Full Multiphysics (like Tier2)
//==============================================================================

bool testFullMultiphysics() {
    printBanner("TEST C: Full Multiphysics (Tier2-like)");

    printf("Configuration:\n");
    printf("  Grid: %d x %d x %d\n", NX, NY, NZ);
    printf("  Physics: ALL (thermal, fluid, VOF, Marangoni, phase change)\n");
    printf("  Duration: 3000 us (30000 steps)\n");
    printf("  This mimics the Tier2 test configuration.\n");
    printf("\n");

    MultiphysicsConfig config;
    config.nx = NX;
    config.ny = NY;
    config.nz = NZ;
    config.dx = DX;
    config.dt = DT;

    // Enable EVERYTHING like Tier2
    config.enable_thermal = true;
    config.enable_thermal_advection = true;
    config.enable_phase_change = true;
    config.enable_fluid = true;
    config.enable_vof = true;
    config.enable_vof_advection = true;
    config.enable_surface_tension = true;
    config.enable_marangoni = true;
    config.enable_darcy = true;
    config.enable_buoyancy = true;
    config.enable_laser = true;
    config.enable_radiation_bc = true;
    config.enable_substrate_cooling = true;

    // Laser config
    config.laser_power = LASER_POWER / 0.35f;
    config.laser_absorptivity = 0.35f;
    config.laser_spot_radius = 50e-6f;
    config.laser_penetration_depth = 10e-6f;
    config.laser_start_x = NX * DX / 2.0f;
    config.laser_start_y = NY * DX / 2.0f;
    config.laser_scan_vx = 0.0f;
    config.laser_scan_vy = 0.0f;
    config.laser_shutoff_time = -1.0f;

    config.emissivity = 0.35f;
    config.ambient_temperature = 300.0f;
    config.substrate_h_conv = 1000.0f;
    config.substrate_temperature = 300.0f;

    // Material
    strncpy(config.material.name, "Ti6Al4V", 63);
    config.material.rho_solid = 4420.0f;
    config.material.rho_liquid = 4110.0f;
    config.material.cp_solid = 670.0f;
    config.material.cp_liquid = 831.0f;
    config.material.k_solid = 7.0f;
    config.material.k_liquid = 30.0f;
    config.material.T_solidus = 1878.0f;
    config.material.T_liquidus = 1928.0f;
    config.material.L_fusion = 286000.0f;
    config.material.T_vaporization = 3560.0f;
    config.material.L_vaporization = 9830000.0f;
    config.material.emissivity = 0.35f;
    config.material.surface_tension = 1.65f;
    config.material.dsigma_dT = -0.26e-3f;
    config.material.mu_liquid = 4.0e-3f;

    config.thermal_diffusivity = 7.0f / (config.material.rho_solid * config.material.cp_solid);
    config.kinematic_viscosity = 0.0333f;  // Lattice units
    config.density = 4110.0f;
    config.surface_tension_coeff = 1.65f;
    config.dsigma_dT = -0.26e-3f;

    MultiphysicsSolver solver(config);
    solver.initialize(T_INIT, 0.5f);

    int num_cells = NX * NY * NZ;
    std::vector<float> h_temp(num_cells);

    printf("Step    Time(us)    T_max(K)    T_min(K)    v_max(m/s)    Status\n");
    printf("----    --------    --------    --------    ----------    ------\n");

    float prev_T_max = T_INIT;
    int anomaly_count = 0;

    int total_steps = 30000;
    int print_interval = 1000;

    for (int step = 0; step < total_steps; ++step) {
        solver.step(DT);

        if ((step + 1) % print_interval == 0) {
            solver.copyTemperatureToHost(h_temp.data());
            TempStats stats = getStats(h_temp.data(), NX, NY, NZ);

            float v_max = solver.getMaxVelocity();
            float time_us = (step + 1) * DT * 1e6f;
            float dT_max = stats.T_max - prev_T_max;

            const char* status = "RISING";
            if (dT_max < -50.0f) {
                status = "DROP <<< ANOMALY";
                anomaly_count++;
            } else if (dT_max < 0.0f) {
                status = "SLIGHT DROP";
            }

            printf("%5d   %8.1f    %8.1f    %8.1f    %10.3e    %s\n",
                   step + 1, time_us, stats.T_max, stats.T_min, v_max, status);

            prev_T_max = stats.T_max;
        }
    }

    printf("\n");
    printf("========================================\n");
    if (anomaly_count > 0) {
        printf("RESULT: FAIL - %d intervals with T_max drop > 50K\n", anomaly_count);
        printf("  Temperature oscillation reproduced!\n");
        printf("  Bug is in MultiphysicsSolver full coupling.\n");
    } else {
        printf("RESULT: PASS - No significant T_max drops\n");
    }
    printf("========================================\n");

    return (anomaly_count == 0);
}

//==============================================================================
// MAIN
//==============================================================================

int main() {
    printf("\n");
    printf("================================================================\n");
    printf("  MULTIPHYSICS TEMPERATURE ANOMALY DIAGNOSTIC\n");
    printf("================================================================\n");
    printf("\n");
    printf("Purpose: Isolate which MultiphysicsSolver components cause\n");
    printf("         the temperature oscillation seen in Tier2 testing.\n");
    printf("\n");
    printf("Test Strategy:\n");
    printf("  A. Thermal + Laser only          → Should PASS\n");
    printf("  B. Thermal + Laser + PhaseChange → May FAIL\n");
    printf("  C. Full Multiphysics             → Expected FAIL (Tier2 bug)\n");
    printf("\n");

    bool testA = testThermalOnlyMultiphysics();
    bool testB = testThermalWithPhaseChange();
    bool testC = testFullMultiphysics();

    printf("\n");
    printf("================================================================\n");
    printf("  DIAGNOSTIC SUMMARY\n");
    printf("================================================================\n");
    printf("\n");
    printf("Test A (Thermal+Laser):           %s\n", testA ? "PASS" : "FAIL <<<");
    printf("Test B (Thermal+Laser+Phase):     %s\n", testB ? "PASS" : "FAIL <<<");
    printf("Test C (Full Multiphysics):       %s\n", testC ? "PASS" : "FAIL <<<");
    printf("\n");

    printf("INTERPRETATION:\n");
    if (!testA) {
        printf("  - Test A FAILED: Bug in basic thermal+laser integration\n");
    }
    if (testA && !testB) {
        printf("  - Test B FAILED but A passed: Bug in phase change solver\n");
    }
    if (testA && testB && !testC) {
        printf("  - Test C FAILED but A,B passed: Bug in fluid/VOF/Marangoni coupling\n");
    }
    if (testA && testB && testC) {
        printf("  - All tests PASSED: Cannot reproduce Tier2 anomaly\n");
        printf("    The bug may be in specific Tier2 configuration or initial conditions.\n");
    }
    printf("\n");

    return (testA && testB && testC) ? 0 : 1;
}
