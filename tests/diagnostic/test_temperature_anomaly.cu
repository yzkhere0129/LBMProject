/**
 * @file test_temperature_anomaly.cu
 * @brief Diagnostic tests to isolate temperature oscillation root cause
 *
 * CRITICAL ISSUE:
 * Tier2 test revealed temperature anomaly:
 *   - T_max dropped from 1978K to 606K during t=400-1200us
 *   - Then recovered to 2230K by t=3000us
 *   - This oscillation should NOT happen with continuous laser heating
 *
 * This diagnostic suite isolates the root cause through systematic testing:
 *   1. Minimal laser heating (no BCs) - monotonic increase expected
 *   2. Heat source verification - confirm 10W deposited
 *   3. Energy conservation - track total thermal energy
 *   4. Buffer swap verification - check ping-pong correctness
 *   5. Radiation BC isolation - check for over-cooling
 */

#include "physics/thermal_lbm.h"
#include "physics/laser_source.h"
#include "physics/material_properties.h"
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

// Grid size: 200x100x50 as specified
constexpr int NX = 200;
constexpr int NY = 100;
constexpr int NZ = 50;
constexpr float DX = 2e-6f;    // 2 um grid spacing
constexpr float DT = 0.1e-6f;  // 0.1 us timestep
constexpr float T_INIT = 300.0f;  // Initial temperature [K]
constexpr float LASER_POWER = 10.0f;  // Absorbed laser power [W]

// Material properties (Ti6Al4V)
static MaterialProperties createMaterial() {
    MaterialProperties mat;
    strncpy(mat.name, "Ti6Al4V", sizeof(mat.name) - 1);
    mat.name[sizeof(mat.name) - 1] = '\0';
    mat.rho_solid = 4420.0f;
    mat.rho_liquid = 4110.0f;
    mat.cp_solid = 670.0f;
    mat.cp_liquid = 831.0f;
    mat.k_solid = 7.0f;    // W/(m*K)
    mat.k_liquid = 30.0f;  // W/(m*K)
    mat.T_solidus = 1878.0f;
    mat.T_liquidus = 1928.0f;
    mat.L_fusion = 286000.0f;
    mat.L_vaporization = 9830000.0f;
    mat.T_vaporization = 3560.0f;
    mat.emissivity = 0.35f;
    return mat;
}

//==============================================================================
// Utility Functions
//==============================================================================

struct TemperatureStats {
    float T_max;
    float T_min;
    float T_avg;
    float T_center;
    float total_energy;
    int max_idx;
    int min_idx;
};

TemperatureStats computeStats(const std::vector<float>& temp, const MaterialProperties& mat) {
    TemperatureStats stats;
    stats.T_max = -1e30f;
    stats.T_min = 1e30f;
    stats.T_avg = 0.0f;
    stats.total_energy = 0.0f;
    stats.max_idx = 0;
    stats.min_idx = 0;

    float dV = DX * DX * DX;
    int num_cells = NX * NY * NZ;

    for (int i = 0; i < num_cells; ++i) {
        float T = temp[i];
        if (T > stats.T_max) { stats.T_max = T; stats.max_idx = i; }
        if (T < stats.T_min) { stats.T_min = T; stats.min_idx = i; }
        stats.T_avg += T;

        // Thermal energy relative to T_INIT
        float dT = T - T_INIT;
        stats.total_energy += mat.rho_solid * mat.cp_solid * dT * dV;
    }
    stats.T_avg /= num_cells;

    // Center temperature
    int center_idx = (NZ/2) * NX * NY + (NY/2) * NX + (NX/2);
    stats.T_center = temp[center_idx];

    return stats;
}

void printBanner(const char* title) {
    printf("\n");
    printf("================================================================\n");
    printf("  %s\n", title);
    printf("================================================================\n");
}

//==============================================================================
// TEST 1: Minimal Laser Heating Test
//==============================================================================

/**
 * Purpose: Track T_max evolution with laser heating, no cooling BCs
 * Expected: T_max should monotonically increase for all 10,000 steps
 *
 * If this fails: Problem is in laser heating or thermal solver core
 */
bool testLaserHeatingTracking() {
    printBanner("TEST 1: Minimal Laser Heating Test");

    printf("Configuration:\n");
    printf("  Grid: %d x %d x %d\n", NX, NY, NZ);
    printf("  dx: %.1e m, dt: %.1e s\n", DX, DT);
    printf("  Laser power: %.1f W (absorbed)\n", LASER_POWER);
    printf("  Initial temp: %.0f K\n", T_INIT);
    printf("  Total time: 1000 us (10000 steps)\n");
    printf("  Boundary conditions: NONE (adiabatic)\n");
    printf("\n");

    MaterialProperties mat = createMaterial();
    float k_thermal = 7.0f;  // W/(m*K)
    float alpha = k_thermal / (mat.rho_solid * mat.cp_solid);

    // Create thermal solver WITHOUT phase change
    ThermalLBM thermal(NX, NY, NZ, mat, alpha, false, DT, DX);
    thermal.initialize(T_INIT);

    // Create laser source at domain center
    float x_center = NX * DX / 2.0f;
    float y_center = NY * DX / 2.0f;

    // LaserSource(power, spot_radius, absorptivity, penetration_depth)
    // To get 10W absorbed: P * absorptivity = 10W => P = 10/0.35 ~ 28.57W
    LaserSource laser(LASER_POWER / 0.35f, 50e-6f, 0.35f, 10e-6f);
    laser.setPosition(x_center, y_center, 0.0f);
    laser.setScanVelocity(0.0f, 0.0f);  // Stationary

    // Allocate device memory for heat source
    int num_cells = NX * NY * NZ;
    float* d_heat_source;
    cudaMalloc(&d_heat_source, num_cells * sizeof(float));

    // Compute volumetric heat source
    dim3 threads(8, 8, 8);
    dim3 blocks((NX + 7) / 8, (NY + 7) / 8, (NZ + 7) / 8);

    computeLaserHeatSourceKernel<<<blocks, threads>>>(
        d_heat_source, laser, DX, DX, DX, NX, NY, NZ);
    cudaDeviceSynchronize();

    // Verify total power
    std::vector<float> h_heat_source(num_cells);
    cudaMemcpy(h_heat_source.data(), d_heat_source,
               num_cells * sizeof(float), cudaMemcpyDeviceToHost);

    float total_power = 0.0f;
    for (int i = 0; i < num_cells; ++i) {
        total_power += h_heat_source[i];
    }
    total_power *= (DX * DX * DX);

    printf("Laser Heat Source Verification:\n");
    printf("  Expected absorbed power: %.2f W\n", LASER_POWER);
    printf("  Computed absorbed power: %.2f W\n", total_power);
    printf("  Power error: %.1f%%\n", 100.0f * fabs(total_power - LASER_POWER) / LASER_POWER);
    printf("\n");

    // Tracking variables
    float prev_T_max = T_INIT;
    int monotonic_violations = 0;
    std::vector<float> T_max_history;

    printf("Step    Time(us)    T_max(K)    T_min(K)    dT_max      Status\n");
    printf("----    --------    --------    --------    -----       ------\n");

    std::vector<float> h_temp(num_cells);

    int total_steps = 10000;
    int print_interval = 100;

    for (int step = 0; step < total_steps; ++step) {
        // Add heat source
        thermal.addHeatSource(d_heat_source, DT);

        // BGK collision (pure diffusion)
        thermal.collisionBGK(nullptr, nullptr, nullptr);

        // Streaming
        thermal.streaming();

        // Compute temperature
        thermal.computeTemperature();

        // Check every print_interval steps
        if ((step + 1) % print_interval == 0) {
            thermal.copyTemperatureToHost(h_temp.data());
            TemperatureStats stats = computeStats(h_temp, mat);

            float time_us = (step + 1) * DT * 1e6f;
            float dT_max = stats.T_max - prev_T_max;

            const char* status = "RISING";
            if (dT_max < -1.0f) {  // Allow 1K tolerance for numerical noise
                status = "FALLING <<< ANOMALY";
                monotonic_violations++;
            }

            printf("%5d   %8.1f    %8.1f    %8.1f    %+7.1f     %s\n",
                   step + 1, time_us, stats.T_max, stats.T_min, dT_max, status);

            T_max_history.push_back(stats.T_max);
            prev_T_max = stats.T_max;
        }
    }

    cudaFree(d_heat_source);

    printf("\n");
    printf("========================================\n");
    printf("RESULT: ");
    if (monotonic_violations == 0) {
        printf("PASS - T_max monotonically increasing\n");
    } else {
        printf("FAIL - %d intervals with T_max decreasing\n", monotonic_violations);
    }
    printf("========================================\n");

    return (monotonic_violations == 0);
}

//==============================================================================
// TEST 2: Heat Source Verification Test
//==============================================================================

/**
 * Purpose: Confirm laser heat source kernel produces correct values
 * Every 1000 steps, print sum of heat source, location of max, temperature at center
 */
bool testHeatSourceVerification() {
    printBanner("TEST 2: Heat Source Verification Test");

    MaterialProperties mat = createMaterial();
    float k_thermal = 7.0f;
    float alpha = k_thermal / (mat.rho_solid * mat.cp_solid);

    ThermalLBM thermal(NX, NY, NZ, mat, alpha, false, DT, DX);
    thermal.initialize(T_INIT);

    float x_center = NX * DX / 2.0f;
    float y_center = NY * DX / 2.0f;

    LaserSource laser(LASER_POWER / 0.35f, 50e-6f, 0.35f, 10e-6f);
    laser.setPosition(x_center, y_center, 0.0f);

    int num_cells = NX * NY * NZ;
    float* d_heat_source;
    cudaMalloc(&d_heat_source, num_cells * sizeof(float));

    dim3 threads(8, 8, 8);
    dim3 blocks((NX + 7) / 8, (NY + 7) / 8, (NZ + 7) / 8);

    std::vector<float> h_heat_source(num_cells);
    std::vector<float> h_temp(num_cells);

    printf("Step    Time(us)    Sum_Q(W)    Q_max(W/m3)    Q_max_pos(i,j,k)    T_center(K)\n");
    printf("----    --------    --------    -----------    ----------------    -----------\n");

    int total_steps = 10000;
    int print_interval = 1000;
    bool all_checks_pass = true;

    for (int step = 0; step < total_steps; ++step) {
        // Recompute heat source (simulates what happens in real simulation)
        computeLaserHeatSourceKernel<<<blocks, threads>>>(
            d_heat_source, laser, DX, DX, DX, NX, NY, NZ);
        cudaDeviceSynchronize();

        thermal.addHeatSource(d_heat_source, DT);
        thermal.collisionBGK(nullptr, nullptr, nullptr);
        thermal.streaming();
        thermal.computeTemperature();

        if ((step + 1) % print_interval == 0) {
            // Get heat source data
            cudaMemcpy(h_heat_source.data(), d_heat_source,
                       num_cells * sizeof(float), cudaMemcpyDeviceToHost);

            float total_power = 0.0f;
            float Q_max = 0.0f;
            int max_i = 0, max_j = 0, max_k = 0;

            for (int k = 0; k < NZ; ++k) {
                for (int j = 0; j < NY; ++j) {
                    for (int i = 0; i < NX; ++i) {
                        int idx = k * NX * NY + j * NX + i;
                        float Q = h_heat_source[idx];
                        total_power += Q;
                        if (Q > Q_max) {
                            Q_max = Q;
                            max_i = i; max_j = j; max_k = k;
                        }
                    }
                }
            }
            total_power *= (DX * DX * DX);

            thermal.copyTemperatureToHost(h_temp.data());
            int center_idx = (NZ/2) * NX * NY + (NY/2) * NX + (NX/2);
            float T_center = h_temp[center_idx];

            float time_us = (step + 1) * DT * 1e6f;
            printf("%5d   %8.1f    %8.2f    %11.2e    (%3d,%3d,%3d)        %8.1f\n",
                   step + 1, time_us, total_power, Q_max, max_i, max_j, max_k, T_center);

            // Verify power is correct (within 10%)
            float power_error = fabs(total_power - LASER_POWER) / LASER_POWER;
            if (power_error > 0.1f) {
                printf("  WARNING: Power error %.1f%% exceeds 10%% threshold\n", power_error * 100);
                all_checks_pass = false;
            }
        }
    }

    cudaFree(d_heat_source);

    printf("\n");
    printf("========================================\n");
    printf("RESULT: %s\n", all_checks_pass ? "PASS - Heat source ~10W every check" : "FAIL - Heat source anomaly");
    printf("========================================\n");

    return all_checks_pass;
}

//==============================================================================
// TEST 3: Energy Conservation Micro-Test
//==============================================================================

/**
 * Purpose: Verify energy is not being lost
 * Calculate: E_total = sum(rho * cp * T * dV) over all cells
 * Expected: dE/dt ~ P_laser (10W) when no cooling BCs
 */
bool testEnergyConservation() {
    printBanner("TEST 3: Energy Conservation Micro-Test");

    MaterialProperties mat = createMaterial();
    float k_thermal = 7.0f;
    float alpha = k_thermal / (mat.rho_solid * mat.cp_solid);

    ThermalLBM thermal(NX, NY, NZ, mat, alpha, false, DT, DX);
    thermal.initialize(T_INIT);

    float x_center = NX * DX / 2.0f;
    float y_center = NY * DX / 2.0f;

    LaserSource laser(LASER_POWER / 0.35f, 50e-6f, 0.35f, 10e-6f);
    laser.setPosition(x_center, y_center, 0.0f);

    int num_cells = NX * NY * NZ;
    float* d_heat_source;
    cudaMalloc(&d_heat_source, num_cells * sizeof(float));

    dim3 threads(8, 8, 8);
    dim3 blocks((NX + 7) / 8, (NY + 7) / 8, (NZ + 7) / 8);

    computeLaserHeatSourceKernel<<<blocks, threads>>>(
        d_heat_source, laser, DX, DX, DX, NX, NY, NZ);
    cudaDeviceSynchronize();

    std::vector<float> h_temp(num_cells);

    printf("Tracking total thermal energy over 1000 steps (100 us):\n");
    printf("\n");
    printf("Step    Time(us)    E_total(J)    dE/dt(W)    Expected(W)    Error(%%)\n");
    printf("----    --------    ----------    --------    -----------    --------\n");

    float prev_energy = 0.0f;
    float prev_time = 0.0f;
    float dV = DX * DX * DX;

    int total_steps = 1000;
    int print_interval = 100;
    bool energy_ok = true;

    for (int step = 0; step < total_steps; ++step) {
        thermal.addHeatSource(d_heat_source, DT);
        thermal.collisionBGK(nullptr, nullptr, nullptr);
        thermal.streaming();
        thermal.computeTemperature();

        if ((step + 1) % print_interval == 0) {
            thermal.copyTemperatureToHost(h_temp.data());

            // Compute total thermal energy
            float total_energy = 0.0f;
            for (int i = 0; i < num_cells; ++i) {
                float dT = h_temp[i] - T_INIT;
                total_energy += mat.rho_solid * mat.cp_solid * dT * dV;
            }

            float time_s = (step + 1) * DT;
            float time_us = time_s * 1e6f;

            float dE_dt = 0.0f;
            if (step > 0) {
                float dt_interval = time_s - prev_time;
                dE_dt = (total_energy - prev_energy) / dt_interval;
            }

            float error = 0.0f;
            if (step > 0) {
                error = 100.0f * fabs(dE_dt - LASER_POWER) / LASER_POWER;
            }

            printf("%5d   %8.1f    %10.3e    %8.2f    %8.2f       %6.1f\n",
                   step + 1, time_us, total_energy, dE_dt, LASER_POWER, error);

            // Check if dE/dt is within 20% of expected
            if (step > 0 && error > 20.0f) {
                printf("  WARNING: dE/dt error %.1f%% exceeds 20%% threshold\n", error);
                energy_ok = false;
            }

            prev_energy = total_energy;
            prev_time = time_s;
        }
    }

    cudaFree(d_heat_source);

    printf("\n");
    printf("========================================\n");
    printf("RESULT: %s\n", energy_ok ? "PASS - dE/dt ~ 10W (within 20%%)" : "FAIL - Energy conservation violated");
    printf("========================================\n");

    return energy_ok;
}

//==============================================================================
// TEST 4: Buffer Swap Verification
//==============================================================================

/**
 * Purpose: Verify ping-pong buffers are correctly swapped
 * After streaming, verify temperature computed from distributions matches expected
 */
bool testBufferSwapVerification() {
    printBanner("TEST 4: Buffer Swap Verification");

    MaterialProperties mat = createMaterial();
    float k_thermal = 7.0f;
    float alpha = k_thermal / (mat.rho_solid * mat.cp_solid);

    // Use smaller domain for faster testing
    int nx = 20, ny = 20, nz = 20;
    ThermalLBM thermal(nx, ny, nz, mat, alpha, false, DT, DX);
    thermal.initialize(T_INIT);

    int num_cells = nx * ny * nz;
    std::vector<float> h_temp(num_cells);

    printf("Testing buffer swap consistency over 100 steps:\n");
    printf("\n");
    printf("Step    T_center(K)    T_max(K)    T_min(K)    dT_center    Status\n");
    printf("----    -----------    --------    --------    ---------    ------\n");

    // Create small heat source at center
    float* d_heat_source;
    cudaMalloc(&d_heat_source, num_cells * sizeof(float));
    cudaMemset(d_heat_source, 0, num_cells * sizeof(float));

    // Set heat source only at center
    std::vector<float> h_heat(num_cells, 0.0f);
    int center_idx = (nz/2) * nx * ny + (ny/2) * nx + (nx/2);
    h_heat[center_idx] = 1e15f;  // Very high heat source at one point
    cudaMemcpy(d_heat_source, h_heat.data(), num_cells * sizeof(float), cudaMemcpyHostToDevice);

    float prev_T_center = T_INIT;
    bool swap_ok = true;

    for (int step = 0; step < 100; ++step) {
        thermal.addHeatSource(d_heat_source, DT);
        thermal.collisionBGK(nullptr, nullptr, nullptr);
        thermal.streaming();
        thermal.computeTemperature();

        if ((step + 1) % 10 == 0) {
            thermal.copyTemperatureToHost(h_temp.data());

            float T_center = h_temp[center_idx];
            float T_max = *std::max_element(h_temp.begin(), h_temp.end());
            float T_min = *std::min_element(h_temp.begin(), h_temp.end());
            float dT_center = T_center - prev_T_center;

            const char* status = "OK";

            // Check for suspicious jumps (could indicate buffer swap issue)
            if (T_center < prev_T_center - 100.0f) {
                status = "JUMP DOWN <<< ANOMALY";
                swap_ok = false;
            } else if (!std::isfinite(T_center)) {
                status = "NaN/Inf <<< ERROR";
                swap_ok = false;
            }

            printf("%4d    %11.1f    %8.1f    %8.1f    %+9.1f    %s\n",
                   step + 1, T_center, T_max, T_min, dT_center, status);

            prev_T_center = T_center;
        }
    }

    cudaFree(d_heat_source);

    printf("\n");
    printf("========================================\n");
    printf("RESULT: %s\n", swap_ok ? "PASS - Buffer swap appears correct" : "FAIL - Buffer swap anomaly detected");
    printf("========================================\n");

    return swap_ok;
}

//==============================================================================
// TEST 5: Radiation BC Isolation Test
//==============================================================================

/**
 * Purpose: Check if radiation BC is over-cooling
 * Initialize at 2000K uniformly, run with radiation only (no laser)
 * Expected: Gradual cooling (~0.5-1 K/step at high T)
 * Red flag: Rapid cooling (>10 K/step) indicates BC bug
 */
bool testRadiationBCIsolation() {
    printBanner("TEST 5: Radiation BC Isolation Test");

    printf("Configuration:\n");
    printf("  Initial temperature: 2000 K (uniformly)\n");
    printf("  Radiation BC: epsilon=0.35, T_ambient=300K\n");
    printf("  Expected: Gradual cooling (0.5-1 K/step)\n");
    printf("  Red flag: >10 K/step indicates BC bug\n");
    printf("\n");

    MaterialProperties mat = createMaterial();
    float k_thermal = 7.0f;
    float alpha = k_thermal / (mat.rho_solid * mat.cp_solid);

    ThermalLBM thermal(NX, NY, NZ, mat, alpha, false, DT, DX);
    thermal.initialize(2000.0f);  // Start at 2000K
    thermal.setEmissivity(0.35f);

    int num_cells = NX * NY * NZ;
    std::vector<float> h_temp(num_cells);

    printf("Step    Time(us)    T_max(K)    T_avg(K)    dT_max/step    Status\n");
    printf("----    --------    --------    --------    -----------    ------\n");

    float prev_T_max = 2000.0f;
    int rapid_cooling_count = 0;

    int total_steps = 1000;
    int print_interval = 100;

    for (int step = 0; step < total_steps; ++step) {
        // Apply radiation BC only (no heat source)
        thermal.applyRadiationBC(DT, DX, 0.35f, 300.0f);

        thermal.collisionBGK(nullptr, nullptr, nullptr);
        thermal.streaming();
        thermal.computeTemperature();

        if ((step + 1) % print_interval == 0) {
            thermal.copyTemperatureToHost(h_temp.data());
            TemperatureStats stats = computeStats(h_temp, mat);

            float time_us = (step + 1) * DT * 1e6f;
            float dT_per_step = (stats.T_max - prev_T_max) / print_interval;

            const char* status = "NORMAL";
            if (dT_per_step < -10.0f) {
                status = "RAPID COOLING <<< RED FLAG";
                rapid_cooling_count++;
            } else if (dT_per_step > 0.0f) {
                status = "HEATING? <<< UNEXPECTED";
            }

            printf("%5d   %8.1f    %8.1f    %8.1f    %+11.2f    %s\n",
                   step + 1, time_us, stats.T_max, stats.T_avg, dT_per_step, status);

            prev_T_max = stats.T_max;
        }
    }

    printf("\n");
    printf("========================================\n");
    if (rapid_cooling_count > 0) {
        printf("RESULT: FAIL - %d intervals with rapid cooling (>10 K/step)\n", rapid_cooling_count);
        printf("  This suggests radiation BC is over-cooling the domain.\n");
    } else {
        printf("RESULT: PASS - Radiation cooling rate is reasonable\n");
    }
    printf("========================================\n");

    return (rapid_cooling_count == 0);
}

//==============================================================================
// TEST 6: Combined Laser + Radiation Test
//==============================================================================

/**
 * Purpose: Test laser heating with radiation BC enabled
 * This mimics the Tier2 test conditions
 */
bool testLaserWithRadiation() {
    printBanner("TEST 6: Combined Laser + Radiation Test");

    printf("This test mimics Tier2 conditions:\n");
    printf("  Laser: 10W absorbed at center\n");
    printf("  Radiation BC: epsilon=0.35, T_ambient=300K\n");
    printf("  Looking for temperature oscillation pattern\n");
    printf("\n");

    MaterialProperties mat = createMaterial();
    float k_thermal = 7.0f;
    float alpha = k_thermal / (mat.rho_solid * mat.cp_solid);

    ThermalLBM thermal(NX, NY, NZ, mat, alpha, false, DT, DX);
    thermal.initialize(T_INIT);
    thermal.setEmissivity(0.35f);

    float x_center = NX * DX / 2.0f;
    float y_center = NY * DX / 2.0f;

    LaserSource laser(LASER_POWER / 0.35f, 50e-6f, 0.35f, 10e-6f);
    laser.setPosition(x_center, y_center, 0.0f);

    int num_cells = NX * NY * NZ;
    float* d_heat_source;
    cudaMalloc(&d_heat_source, num_cells * sizeof(float));

    dim3 threads(8, 8, 8);
    dim3 blocks((NX + 7) / 8, (NY + 7) / 8, (NZ + 7) / 8);

    computeLaserHeatSourceKernel<<<blocks, threads>>>(
        d_heat_source, laser, DX, DX, DX, NX, NY, NZ);
    cudaDeviceSynchronize();

    std::vector<float> h_temp(num_cells);

    printf("Step    Time(us)    T_max(K)    T_min(K)    E_total(J)    Status\n");
    printf("----    --------    --------    --------    ----------    ------\n");

    float prev_T_max = T_INIT;
    int drop_count = 0;

    // Run for 3000us (30000 steps) to match Tier2 timeframe
    int total_steps = 30000;
    int print_interval = 1000;  // Print every 100us

    for (int step = 0; step < total_steps; ++step) {
        // Add laser heat source
        thermal.addHeatSource(d_heat_source, DT);

        // Apply radiation BC
        thermal.applyRadiationBC(DT, DX, 0.35f, 300.0f);

        thermal.collisionBGK(nullptr, nullptr, nullptr);
        thermal.streaming();
        thermal.computeTemperature();

        if ((step + 1) % print_interval == 0) {
            thermal.copyTemperatureToHost(h_temp.data());
            TemperatureStats stats = computeStats(h_temp, mat);

            float time_us = (step + 1) * DT * 1e6f;
            float dT_max = stats.T_max - prev_T_max;

            const char* status = "RISING";
            if (dT_max < -50.0f) {
                status = "SIGNIFICANT DROP <<< ANOMALY";
                drop_count++;
            } else if (dT_max < 0.0f) {
                status = "SLIGHT DROP";
            }

            printf("%5d   %8.1f    %8.1f    %8.1f    %10.3e    %s\n",
                   step + 1, time_us, stats.T_max, stats.T_min, stats.total_energy, status);

            prev_T_max = stats.T_max;
        }
    }

    cudaFree(d_heat_source);

    printf("\n");
    printf("========================================\n");
    if (drop_count > 0) {
        printf("RESULT: FAIL - %d intervals with significant T_max drop\n", drop_count);
        printf("  Temperature oscillation pattern reproduced!\n");
        printf("  Root cause likely in radiation BC or boundary handling.\n");
    } else {
        printf("RESULT: PASS - No significant temperature drops\n");
    }
    printf("========================================\n");

    return (drop_count == 0);
}

//==============================================================================
// MAIN
//==============================================================================

int main() {
    printf("\n");
    printf("================================================================\n");
    printf("  TEMPERATURE ANOMALY DIAGNOSTIC SUITE\n");
    printf("================================================================\n");
    printf("\n");
    printf("Purpose: Isolate root cause of Tier2 temperature oscillation\n");
    printf("  - T_max dropped from 1978K to 606K during t=400-1200us\n");
    printf("  - Then recovered to 2230K by t=3000us\n");
    printf("\n");
    printf("Test Suite:\n");
    printf("  1. Minimal laser heating (no BCs)\n");
    printf("  2. Heat source verification\n");
    printf("  3. Energy conservation\n");
    printf("  4. Buffer swap verification\n");
    printf("  5. Radiation BC isolation\n");
    printf("  6. Combined laser + radiation\n");
    printf("\n");

    // Run all tests
    bool test1 = testLaserHeatingTracking();
    bool test2 = testHeatSourceVerification();
    bool test3 = testEnergyConservation();
    bool test4 = testBufferSwapVerification();
    bool test5 = testRadiationBCIsolation();
    bool test6 = testLaserWithRadiation();

    // Summary
    printf("\n");
    printf("================================================================\n");
    printf("  DIAGNOSTIC SUMMARY\n");
    printf("================================================================\n");
    printf("\n");
    printf("Test 1 (Laser Heating):      %s\n", test1 ? "PASS" : "FAIL <<<");
    printf("Test 2 (Heat Source):        %s\n", test2 ? "PASS" : "FAIL <<<");
    printf("Test 3 (Energy Conservation):%s\n", test3 ? "PASS" : "FAIL <<<");
    printf("Test 4 (Buffer Swap):        %s\n", test4 ? "PASS" : "FAIL <<<");
    printf("Test 5 (Radiation BC):       %s\n", test5 ? "PASS" : "FAIL <<<");
    printf("Test 6 (Laser + Radiation):  %s\n", test6 ? "PASS" : "FAIL <<<");
    printf("\n");

    // Interpretation
    printf("INTERPRETATION:\n");
    if (!test1) {
        printf("  - Test 1 FAILED: Core thermal solver has bug\n");
        printf("    Check: collision kernel, streaming, heat source addition\n");
    }
    if (!test2) {
        printf("  - Test 2 FAILED: Heat source computation is wrong\n");
        printf("    Check: computeLaserHeatSourceKernel, laser parameters\n");
    }
    if (!test3) {
        printf("  - Test 3 FAILED: Energy not conserved\n");
        printf("    Check: addHeatSource dt scaling, boundary leakage\n");
    }
    if (!test4) {
        printf("  - Test 4 FAILED: Buffer swap issue\n");
        printf("    Check: swapDistributions(), ping-pong pointers\n");
    }
    if (!test5) {
        printf("  - Test 5 FAILED: Radiation BC over-cooling\n");
        printf("    Check: applyRadiationBC(), Stefan-Boltzmann formula\n");
    }
    if (!test6 && test1 && test5) {
        printf("  - Test 6 FAILED but Test 1 and Test 5 passed:\n");
        printf("    Bug is in the INTERACTION between laser and radiation BC\n");
    }

    if (test1 && test2 && test3 && test4 && test5 && test6) {
        printf("  - All tests PASSED: Temperature oscillation NOT reproduced\n");
        printf("    The bug may be in MultiphysicsSolver coupling, not ThermalLBM\n");
    }
    printf("\n");

    return (test1 && test2 && test3 && test4 && test5 && test6) ? 0 : 1;
}
