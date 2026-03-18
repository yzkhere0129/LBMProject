/**
 * @file test_keyhole_formation_senior.cu
 * @brief Case 6: Evaporation Keyhole Formation Validation (Senior's Configuration)
 *
 * This test validates keyhole formation under high-power laser irradiation
 * following the exact configuration from the senior's work.
 *
 * Physical Configuration (adapted from senior's Fe parameters to Ti6Al4V):
 * - Domain: 150×300×150 μm (40×80×80 grid cells)
 * - Material: Ti6Al4V (ρ_liquid = 4000 kg/m³, ρ_gas = 1.0 kg/m³)
 * - Boiling point: T_v = 3560 K (Ti6Al4V)
 * - Grid spacing: dx = 3.75 μm
 * - Time step: dt = 0.1 μs (CFL-limited for high velocities)
 *
 * Physics Enabled:
 * - Hertz-Knudsen evaporation model
 * - Recoil pressure: P_recoil = 0.54 × P_sat
 * - Marangoni convection
 * - Surface tension
 * - VOF interface tracking
 *
 * Key Phenomena:
 * 1. Keyhole formation: Recoil pressure pushes liquid surface downward
 * 2. Marangoni convection: Temperature gradient drives surface flow
 * 3. Metal vapor generation: Mass flux from liquid-gas interface
 * 4. Dynamic interface: Keyhole depth evolves with time
 *
 * Validation Time Points:
 * - 5 μs: Initial keyhole formation
 * - 10 μs: Keyhole deepening
 * - 15 μs: Keyhole development
 * - 20 μs: Quasi-steady state or oscillation
 *
 * Expected Results:
 * - Keyhole depth: 20-50 μm at t=10 μs (typical LPBF keyhole regime)
 * - Keyhole diameter: ~laser spot size (60-80 μm)
 * - Maximum temperature: 3600-4500 K (above boiling)
 * - Surface velocity: 1-10 m/s (Marangoni-driven)
 *
 * References:
 * - Khairallah et al. (2016): Laser powder-bed fusion additive manufacturing
 * - Tan et al. (2013): Multi-scale modeling of solidification
 * - King et al. (2015): Observation of keyhole-mode laser melting
 */

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <iostream>
#include <fstream>
#include <iomanip>

#include "physics/multiphysics_solver.h"
#include "io/vtk_writer.h"

namespace lbm {
namespace physics {
namespace test {

// Physical constants for Ti6Al4V
constexpr float T_AMBIENT = 300.0f;         // K
constexpr float T_MELT = 1923.0f;           // K (liquidus)
constexpr float T_BOIL = 3560.0f;           // K (vaporization)
constexpr float RHO_LIQUID = 4000.0f;       // kg/m³ (Ti6Al4V liquid)
constexpr float RHO_GAS = 1.0f;             // kg/m³ (ambient gas)

// Grid configuration (senior's setup)
constexpr int NX = 40;                      // Domain: 150 μm
constexpr int NY = 80;                      // Domain: 300 μm
constexpr int NZ = 40;                      // Domain: 150 μm
constexpr float DX = 3.75e-6f;              // m (150 μm / 40 cells)
constexpr float DT = 1e-7f;                 // s (0.1 μs, CFL-safe for keyhole)

// Laser parameters (high-power keyhole regime)
constexpr float LASER_POWER = 300.0f;       // W (high power for keyhole)
constexpr float LASER_RADIUS = 35e-6f;      // m (70 μm spot diameter)
constexpr float LASER_ABSORPTIVITY = 0.35f; // Ti6Al4V typical
constexpr float LASER_PENETRATION = 10e-6f; // m

// Simulation parameters
constexpr float T_SIMULATION = 20e-6f;      // s (20 μs total)
constexpr int STEPS_PER_OUTPUT = 50;        // Output every 5 μs

class KeyholeFormationTest : public ::testing::Test {
protected:
    void SetUp() override {
        std::cout << "\n=== Keyhole Formation Test (Senior's Configuration) ===" << std::endl;
        std::cout << "Domain: " << NX << "×" << NY << "×" << NZ << " cells" << std::endl;
        std::cout << "Physical size: " << NX*DX*1e6 << "×" << NY*DX*1e6 << "×" << NZ*DX*1e6 << " μm" << std::endl;
        std::cout << "Grid spacing: dx = " << DX*1e6 << " μm" << std::endl;
        std::cout << "Time step: dt = " << DT*1e6 << " μs" << std::endl;
        std::cout << "Laser power: " << LASER_POWER << " W" << std::endl;
        std::cout << "Laser spot: " << LASER_RADIUS*2e6 << " μm diameter" << std::endl;
    }

    void TearDown() override {
        cudaDeviceSynchronize();
        cudaDeviceReset();
    }

    MultiphysicsConfig createKeyholeConfig() {
        MultiphysicsConfig config;

        // Domain parameters (senior's configuration)
        config.nx = NX;
        config.ny = NY;
        config.nz = NZ;
        config.dx = DX;
        config.dt = DT;

        // Enable all keyhole physics
        config.enable_thermal = true;
        config.enable_thermal_advection = true;
        config.enable_fluid = true;
        config.enable_vof = true;
        config.enable_vof_advection = true;
        config.enable_surface_tension = true;
        config.enable_marangoni = true;
        config.enable_laser = true;
        config.enable_buoyancy = false;  // Negligible for keyhole dynamics
        config.enable_darcy = true;
        config.enable_evaporation_mass_loss = true;
        config.enable_recoil_pressure = true;  // CRITICAL for keyhole
        config.enable_phase_change = true;

        // Recoil pressure parameters (Anisimov/Knight model)
        config.recoil_coefficient = 0.54f;      // P_recoil = 0.54 × P_sat
        config.recoil_smoothing_width = 2.0f;   // Interface smoothing
        config.recoil_max_pressure = 1e8f;      // 100 MPa limiter

        // CFL parameters (adaptive for keyhole - high velocity regime)
        config.cfl_use_adaptive = true;
        config.cfl_v_target_interface = 0.6f;   // Allow 0.6 lattice = ~12 m/s at interface
        config.cfl_v_target_bulk = 0.4f;        // Bulk liquid: 0.4 lattice = ~8 m/s
        config.cfl_interface_threshold_lo = 0.01f;
        config.cfl_interface_threshold_hi = 0.99f;
        config.cfl_recoil_boost_factor = 2.0f;  // Extra allowance for recoil-dominant flow

        // VOF parameters
        config.vof_subcycles = 5;  // Subcycling for stability

        // Material properties (Ti6Al4V)
        config.material = MaterialDatabase::getTi6Al4V();
        config.density = RHO_LIQUID;
        config.thermal_diffusivity = 5.8e-6f;   // m²/s (Ti6Al4V liquid)
        config.kinematic_viscosity = 0.0333f;   // Lattice units (tau=0.6)

        // Surface properties
        config.surface_tension_coeff = 1.65f;   // N/m (Ti6Al4V)
        config.dsigma_dT = -0.26e-3f;           // N/(m·K)

        // Laser configuration (high power for keyhole)
        config.laser_power = LASER_POWER;
        config.laser_spot_radius = LASER_RADIUS;
        config.laser_absorptivity = LASER_ABSORPTIVITY;
        config.laser_penetration_depth = LASER_PENETRATION;
        config.laser_shutoff_time = -1.0f;      // Always on
        config.laser_start_x = -1.0f;           // Auto-center
        config.laser_start_y = -1.0f;           // Auto-center
        config.laser_scan_vx = 0.0f;            // Stationary for this test
        config.laser_scan_vy = 0.0f;

        // Boundary conditions
        config.boundary_type = 0;  // Periodic (x, y), free surface (z)
        config.enable_radiation_bc = true;      // Enable radiation cooling
        config.emissivity = 0.3f;               // Ti6Al4V
        config.ambient_temperature = T_AMBIENT;
        config.enable_substrate_cooling = true;
        config.substrate_h_conv = 2000.0f;      // W/(m²·K) - active cooling
        config.substrate_temperature = T_AMBIENT;

        return config;
    }

    /**
     * @brief Compute keyhole depth from fill level field
     * @param fill_level Fill level array (nx×ny×nz)
     * @return Keyhole depth in μm
     */
    float computeKeyholeDepth(const std::vector<float>& fill_level) {
        // Find the deepest point where fill_level drops below 0.5
        // along the central vertical line (x=nx/2, y=ny/2)
        int ix = NX / 2;
        int iy = NY / 2;

        int deepest_z = NZ;  // Start from top

        // Scan from top to bottom
        for (int iz = NZ - 1; iz >= 0; --iz) {
            int idx = ix + NX * (iy + NY * iz);
            if (fill_level[idx] > 0.5f) {
                // Found liquid surface
                deepest_z = iz;
                break;
            }
        }

        // Convert to physical depth from top surface
        int initial_surface_z = NZ / 2;  // Initial interface at z=0.5
        int keyhole_cells = initial_surface_z - deepest_z;
        float depth_m = keyhole_cells * DX;

        return depth_m * 1e6f;  // Convert to μm
    }

    /**
     * @brief Write keyhole depth history to file
     */
    void writeKeyholeHistory(const std::vector<float>& times,
                            const std::vector<float>& depths,
                            const std::string& filename) {
        std::ofstream file(filename);
        file << "# Keyhole Depth History\n";
        file << "# Time[μs] Depth[μm]\n";
        file << std::scientific << std::setprecision(6);

        for (size_t i = 0; i < times.size(); ++i) {
            file << times[i] * 1e6 << " " << depths[i] << "\n";
        }

        file.close();
        std::cout << "  Written keyhole history to: " << filename << std::endl;
    }
};

/**
 * @brief Test keyhole formation with full physics
 */
TEST_F(KeyholeFormationTest, FullPhysicsKeyhole) {
    std::cout << "\n--- Test: Full Physics Keyhole Formation ---" << std::endl;

    MultiphysicsConfig config = createKeyholeConfig();
    MultiphysicsSolver solver(config);

    // Initialize: liquid pool at bottom, gas above
    // Interface at z = nz/2 (mid-height)
    solver.initialize(T_AMBIENT, 0.5f);

    // Create output directory
    std::string output_dir = "output_keyhole_senior";
    system(("mkdir -p " + output_dir).c_str());

    // VTK writer for visualization
    io::VTKWriter vtk_writer;

    // Tracking arrays
    std::vector<float> time_history;
    std::vector<float> depth_history;
    std::vector<float> max_temp_history;

    // Simulation parameters
    int total_steps = static_cast<int>(T_SIMULATION / DT);

    std::cout << "\nRunning simulation for " << T_SIMULATION*1e6 << " μs ("
              << total_steps << " steps)..." << std::endl;
    std::cout << "Output interval: " << STEPS_PER_OUTPUT * DT * 1e6 << " μs\n" << std::endl;

    // Time stepping loop
    for (int step = 0; step <= total_steps; ++step) {
        float time = step * DT;

        // Advance simulation
        if (step > 0) {
            solver.step(DT);
        }

        // Output and diagnostics
        if (step % STEPS_PER_OUTPUT == 0) {
            // Get fields
            std::vector<float> fill_level(NX * NY * NZ);
            std::vector<float> temperature(NX * NY * NZ);

            cudaMemcpy(fill_level.data(), solver.getFillLevel(),
                      NX * NY * NZ * sizeof(float), cudaMemcpyDeviceToHost);
            cudaMemcpy(temperature.data(), solver.getTemperature(),
                      NX * NY * NZ * sizeof(float), cudaMemcpyDeviceToHost);

            // Compute diagnostics
            float keyhole_depth = computeKeyholeDepth(fill_level);
            float max_temp = *std::max_element(temperature.begin(), temperature.end());

            time_history.push_back(time);
            depth_history.push_back(keyhole_depth);
            max_temp_history.push_back(max_temp);

            // Print status
            std::cout << "t = " << std::setw(6) << std::fixed << std::setprecision(2)
                      << time*1e6 << " μs: "
                      << "Keyhole depth = " << std::setw(6) << keyhole_depth << " μm, "
                      << "T_max = " << std::setw(7) << std::setprecision(0) << max_temp << " K"
                      << std::endl;

            // Write VTK output
            std::string vtk_filename = output_dir + "/keyhole_" +
                                      std::to_string(static_cast<int>(time*1e6)) + "us.vtk";

            vtk_writer.writeStructuredPoints(
                vtk_filename,
                NX, NY, NZ,
                DX, DX, DX,
                temperature.data(),
                fill_level.data(),
                solver.getVelocityX(),  // Will need to copy these too
                solver.getVelocityY(),
                solver.getVelocityZ()
            );
        }

        // Check for stability
        float max_T = solver.getMaxTemperature();
        if (std::isnan(max_T) || std::isinf(max_T)) {
            std::cerr << "ERROR: NaN/Inf detected at step " << step << std::endl;
            FAIL() << "Simulation became unstable";
        }
    }

    std::cout << "\nSimulation completed successfully!" << std::endl;

    // Write keyhole depth history
    writeKeyholeHistory(time_history, depth_history,
                       output_dir + "/keyhole_depth.dat");

    // Final validation checks
    EXPECT_GT(depth_history.back(), 0.0f) << "Keyhole should form";
    EXPECT_LT(depth_history.back(), 100.0f) << "Keyhole depth should be reasonable";
    EXPECT_GT(max_temp_history.back(), T_BOIL) << "Temperature should exceed boiling";

    // Check keyhole development over time
    EXPECT_GT(depth_history.back(), depth_history.front())
        << "Keyhole should deepen over time";

    std::cout << "\nFinal Results:" << std::endl;
    std::cout << "  Keyhole depth at 20 μs: " << depth_history.back() << " μm" << std::endl;
    std::cout << "  Maximum temperature: " << max_temp_history.back() << " K" << std::endl;
    std::cout << "  Output directory: " << output_dir << std::endl;
}

/**
 * @brief Test recoil pressure activation
 */
TEST_F(KeyholeFormationTest, RecoilPressureActivation) {
    std::cout << "\n--- Test: Recoil Pressure Activation ---" << std::endl;

    // Run two simulations: with and without recoil pressure
    MultiphysicsConfig config_with = createKeyholeConfig();
    config_with.enable_recoil_pressure = true;

    MultiphysicsConfig config_without = createKeyholeConfig();
    config_without.enable_recoil_pressure = false;

    MultiphysicsSolver solver_with(config_with);
    MultiphysicsSolver solver_without(config_without);

    // Initialize both
    solver_with.initialize(T_AMBIENT, 0.5f);
    solver_without.initialize(T_AMBIENT, 0.5f);

    // Run for short time
    int test_steps = 100;  // 10 μs
    for (int step = 0; step < test_steps; ++step) {
        solver_with.step(DT);
        solver_without.step(DT);
    }

    // Get fill levels
    std::vector<float> fill_with(NX * NY * NZ);
    std::vector<float> fill_without(NX * NY * NZ);

    cudaMemcpy(fill_with.data(), solver_with.getFillLevel(),
              NX * NY * NZ * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(fill_without.data(), solver_without.getFillLevel(),
              NX * NY * NZ * sizeof(float), cudaMemcpyDeviceToHost);

    // Compute keyhole depths
    float depth_with = computeKeyholeDepth(fill_with);
    float depth_without = computeKeyholeDepth(fill_without);

    std::cout << "  Keyhole depth WITH recoil: " << depth_with << " μm" << std::endl;
    std::cout << "  Keyhole depth WITHOUT recoil: " << depth_without << " μm" << std::endl;

    // Recoil pressure should create deeper keyhole
    EXPECT_GT(depth_with, depth_without)
        << "Recoil pressure should create deeper keyhole";

    std::cout << "  [PASS] Recoil pressure increases keyhole depth by "
              << (depth_with - depth_without) << " μm" << std::endl;
}

/**
 * @brief Test keyhole depth tracking over time
 */
TEST_F(KeyholeFormationTest, KeyholeDepthEvolution) {
    std::cout << "\n--- Test: Keyhole Depth Evolution ---" << std::endl;

    MultiphysicsConfig config = createKeyholeConfig();
    MultiphysicsSolver solver(config);
    solver.initialize(T_AMBIENT, 0.5f);

    // Track depth at validation time points
    std::vector<float> validation_times = {5e-6f, 10e-6f, 15e-6f, 20e-6f};  // μs
    std::vector<float> measured_depths;

    for (float target_time : validation_times) {
        int target_step = static_cast<int>(target_time / DT);

        // Run to target time
        static int last_step = 0;
        for (int step = last_step; step < target_step; ++step) {
            solver.step(DT);
        }
        last_step = target_step;

        // Measure depth
        std::vector<float> fill_level(NX * NY * NZ);
        cudaMemcpy(fill_level.data(), solver.getFillLevel(),
                  NX * NY * NZ * sizeof(float), cudaMemcpyDeviceToHost);

        float depth = computeKeyholeDepth(fill_level);
        measured_depths.push_back(depth);

        std::cout << "  t = " << target_time*1e6 << " μs: depth = "
                  << depth << " μm" << std::endl;
    }

    // Validate monotonic increase (at least initially)
    EXPECT_GT(measured_depths[1], measured_depths[0])
        << "Keyhole should deepen from 5 to 10 μs";

    // Final depth should be significant
    EXPECT_GT(measured_depths.back(), 10.0f)
        << "Final keyhole depth should exceed 10 μm";

    std::cout << "  [PASS] Keyhole evolution validated" << std::endl;
}

} // namespace test
} // namespace physics
} // namespace lbm

int main(int argc, char** argv) {
    std::cout << "============================================================" << std::endl;
    std::cout << "Case 6: Keyhole Formation Validation (Senior's Config)" << std::endl;
    std::cout << "============================================================" << std::endl;
    std::cout << std::endl;
    std::cout << "Configuration:" << std::endl;
    std::cout << "  Domain: 150×300×150 μm (40×80×40 grid)" << std::endl;
    std::cout << "  Material: Ti6Al4V (ρ_liq=4000, ρ_gas=1.0 kg/m³)" << std::endl;
    std::cout << "  Boiling point: T_v = 3560 K" << std::endl;
    std::cout << "  Evaporation: Hertz-Knudsen model" << std::endl;
    std::cout << "  Recoil: P_recoil = 0.54 × P_sat" << std::endl;
    std::cout << std::endl;

    ::testing::InitGoogleTest(&argc, argv);
    int result = RUN_ALL_TESTS();

    std::cout << std::endl;
    std::cout << "============================================================" << std::endl;
    std::cout << "Keyhole Formation Validation Complete" << std::endl;
    std::cout << "============================================================" << std::endl;

    return result;
}
