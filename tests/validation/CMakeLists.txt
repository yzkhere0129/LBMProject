# Validation Tests CMakeLists.txt
#
# These tests validate the physics implementation against literature values
# and experimental data for metal additive manufacturing processes.

# Enable CUDA
enable_language(CUDA)

# Find required packages
find_package(CUDAToolkit REQUIRED)

# Add Thrust include directory
include_directories(${CUDAToolkit_INCLUDE_DIRS})

# ========================================
# Unit Conversion Bug Diagnostic Test
# ========================================
#
# This test demonstrates the unit conversion bug found in some validation tests
# where physical dt/dx are used with lattice-scale viscosity values.
#
# Purpose: Show the bug, explain it, and validate the fix.

add_executable(test_unit_conversion_bug_diagnostic
    test_unit_conversion_bug_diagnostic.cu
)

set_target_properties(test_unit_conversion_bug_diagnostic PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "70;75;80;86"
)

target_include_directories(test_unit_conversion_bug_diagnostic PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(test_unit_conversion_bug_diagnostic
    gtest
    gtest_main
)

add_test(NAME UnitConversionBugDiagnostic
    COMMAND test_unit_conversion_bug_diagnostic
)

set_tests_properties(UnitConversionBugDiagnostic PROPERTIES
    TIMEOUT 10
    LABELS "validation;diagnostic;unit_conversion"
)

# ========================================
# Omega Clamping Bug Fix Test
# ========================================
#
# This test validates the fix for the omega clamping bug (2026-01-04).
#
# BUG: Previous code clamped omega to 1.85 when omega > 1.9, which destroyed
# physical accuracy because alpha = cs² × (tau - 0.5) × dx²/dt
#
# FIX: Remove clamping, reject unstable configurations with clear error.
#
# Success criteria:
# - Omega matches expected value (no silent clamping)
# - Constructor throws exception for omega > 1.95

add_executable(test_omega_clamping_fix
    test_omega_clamping_fix.cu
)

set_target_properties(test_omega_clamping_fix PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "70;75;80;86"
)

target_include_directories(test_omega_clamping_fix PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(test_omega_clamping_fix
    lbm_physics
    lbm_core
    CUDA::cudart
    CUDA::cuda_driver
)

# ========================================
# TEST 2C: Marangoni Velocity Validation
# ========================================
#
# This test validates that Marangoni-driven surface flow produces realistic
# velocities matching literature values for LPBF (0.5-2 m/s for Ti6Al4V).
#
# Success criteria:
# - CRITICAL: Velocity in range 0.5-2.0 m/s
# - ACCEPTABLE: Velocity in range 0.1-10.0 m/s (order of magnitude)
#
# References:
# - Khairallah et al. (2016): Marangoni velocity 0.5-2 m/s for Ti6Al4V

add_executable(test_marangoni_velocity
    test_marangoni_velocity.cu
)

set_target_properties(test_marangoni_velocity PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "70;75;80;86"
)

target_include_directories(test_marangoni_velocity PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(test_marangoni_velocity
    lbm_physics
    lbm_core
    lbm_io
    gtest
    gtest_main
    CUDA::cudart
    CUDA::cuda_driver
)

add_test(NAME MarangoniVelocityValidation
    COMMAND test_marangoni_velocity
)

set_tests_properties(MarangoniVelocityValidation PROPERTIES
    TIMEOUT 300  # 5 minutes (2000 timesteps)
    LABELS "validation;marangoni;velocity;critical"
)

# ========================================
# TEST: Realistic LPBF Initial Conditions
# ========================================
#
# This test validates that the realistic LPBF simulation starts from cold
# solid metal (300K), not pre-melted liquid (2000K) like validation tests.
#
# Success criteria:
# - CRITICAL: Initial temperature ~ 300K (not >1900K)
# - CRITICAL: Temperature increases over time (laser heating works)
# - CRITICAL: Laser module is enabled

add_executable(test_realistic_lpbf_initial_conditions
    test_realistic_lpbf_initial_conditions.cu
)

set_target_properties(test_realistic_lpbf_initial_conditions PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "70;75;80;86"
)

target_include_directories(test_realistic_lpbf_initial_conditions PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(test_realistic_lpbf_initial_conditions
    lbm_physics
    lbm_core
    lbm_io
    gtest
    gtest_main
    CUDA::cudart
    CUDA::cuda_driver
)

add_test(NAME RealisticLPBFInitialConditions
    COMMAND test_realistic_lpbf_initial_conditions
)

set_tests_properties(RealisticLPBFInitialConditions PROPERTIES
    TIMEOUT 120  # 2 minutes
    LABELS "validation;lpbf;initial_conditions;critical"
)

# ========================================
# TEST: Divergence-Free Velocity Field
# ========================================
#
# This test validates that LBM enforces incompressibility (div(u) = 0).
# Critical for VOF mass conservation!
#
# Success criteria:
# - CRITICAL: |div(u)| < 1e-3 with zero force
# - CRITICAL: |div(u)| < 1e-3 with uniform force
# - CRITICAL: |div(u)| < 1e-3 with spatially-varying force

add_executable(test_divergence_free
    test_divergence_free.cu
)

set_target_properties(test_divergence_free PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "70;75;80;86"
)

target_include_directories(test_divergence_free PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(test_divergence_free
    lbm_physics
    lbm_core
    lbm_io
    CUDA::cudart
    CUDA::cuda_driver
)

add_test(NAME DivergenceFreeVelocity
    COMMAND test_divergence_free
)

set_tests_properties(DivergenceFreeVelocity PROPERTIES
    TIMEOUT 120  # 2 minutes
    LABELS "validation;divergence;incompressibility;critical"
)

# ========================================
# TEST: Phase Change Robustness
# ========================================
#
# This test validates that the phase change solver with bisection fallback
# works correctly in realistic simulation scenarios.
#
# Success criteria:
# - CRITICAL: No NaN during melting
# - CRITICAL: Temperature stays in physical range
# - CRITICAL: Stable through laser shutoff
# - CRITICAL: Stable in mushy zone

add_executable(test_phase_change_robustness
    test_phase_change_robustness.cu
)

set_target_properties(test_phase_change_robustness PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "70;75;80;86"
)

target_include_directories(test_phase_change_robustness PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(test_phase_change_robustness
    lbm_physics
    lbm_core
    lbm_io
    gtest
    gtest_main
    CUDA::cudart
    CUDA::cuda_driver
)

add_test(NAME PhaseChangeRobustness
    COMMAND test_phase_change_robustness
)

set_tests_properties(PhaseChangeRobustness PROPERTIES
    TIMEOUT 120
    LABELS "validation;phase_change;robustness;critical"
)

# ========================================
# TEST: Laser Shutoff Configurable
# ========================================
#
# This test validates that the laser shutoff time parameter works correctly.
#
# Success criteria:
# - CRITICAL: Laser turns off at specified time
# - CRITICAL: Temperature stops increasing after shutoff
# - CRITICAL: Negative shutoff time means laser always on
# - CRITICAL: System remains stable after shutoff

add_executable(test_laser_shutoff_configurable
    test_laser_shutoff_configurable.cu
)

set_target_properties(test_laser_shutoff_configurable PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "70;75;80;86"
)

target_include_directories(test_laser_shutoff_configurable PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(test_laser_shutoff_configurable
    lbm_physics
    lbm_core
    lbm_io
    gtest
    gtest_main
    CUDA::cudart
    CUDA::cuda_driver
)

add_test(NAME LaserShutoffConfigurable
    COMMAND test_laser_shutoff_configurable
)

set_tests_properties(LaserShutoffConfigurable PROPERTIES
    TIMEOUT 120
    LABELS "validation;laser;shutoff;stability;critical"
)

# ========================================
# WEEK 1 FIX VALIDATION TESTS
# ========================================
#
# Test Suite 1: Evaporation Power Fix Validation (660× bug correction)
# Test Suite 2: Substrate Cooling BC Validation

# Test 1.1: Hertz-Knudsen Formula Unit Test
add_executable(test_evaporation_hertz_knudsen
    test_evaporation_hertz_knudsen.cu
)

set_target_properties(test_evaporation_hertz_knudsen PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "70;75;80;86"
)

target_include_directories(test_evaporation_hertz_knudsen PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(test_evaporation_hertz_knudsen
    lbm_physics
    lbm_core
    gtest
    gtest_main
    CUDA::cudart
    CUDA::cuda_driver
)

add_test(NAME EvaporationHertzKnudsenFormula
    COMMAND test_evaporation_hertz_knudsen
)

set_tests_properties(EvaporationHertzKnudsenFormula PROPERTIES
    TIMEOUT 60
    LABELS "validation;evaporation;hertz_knudsen;week1;critical"
)

# Test 2.1: Substrate Cooling BC Unit Test
add_executable(test_substrate_cooling_bc
    test_substrate_cooling_bc.cu
)

set_target_properties(test_substrate_cooling_bc PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "70;75;80;86"
)

target_include_directories(test_substrate_cooling_bc PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(test_substrate_cooling_bc
    lbm_physics
    lbm_core
    gtest
    gtest_main
    CUDA::cudart
    CUDA::cuda_driver
)

add_test(NAME SubstrateCoolingBoundaryCondition
    COMMAND test_substrate_cooling_bc
)

set_tests_properties(SubstrateCoolingBoundaryCondition PROPERTIES
    TIMEOUT 180
    LABELS "validation;substrate;cooling;bc;week1;critical"
)

# ========================================
# WEEK 2 ANALYTICAL BENCHMARK TESTS
# ========================================
#
# Test Suite: Physics Validation Against Analytical Solutions
# These tests verify the mathematical correctness of the physics implementation
# by comparing numerical solutions to known analytical solutions.

# Benchmark 1: Pure Heat Conduction (1D)
# Validates thermal diffusion against Gaussian analytical solution
add_executable(test_pure_conduction
    test_pure_conduction.cu
)

set_target_properties(test_pure_conduction PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "70;75;80;86"
)

target_include_directories(test_pure_conduction PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(test_pure_conduction
    lbm_physics
    lbm_core
    gtest
    gtest_main
    CUDA::cudart
    CUDA::cuda_driver
)

add_test(NAME PureConductionBenchmark
    COMMAND test_pure_conduction
)

set_tests_properties(PureConductionBenchmark PROPERTIES
    TIMEOUT 300
    LABELS "validation;benchmark;conduction;analytical;week2;critical"
)

# Benchmark 2: Stefan Problem (1D Melting)
# Validates phase change against moving boundary analytical solution
add_executable(test_stefan_problem
    test_stefan_problem.cu
)

set_target_properties(test_stefan_problem PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "70;75;80;86"
)

target_include_directories(test_stefan_problem PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(test_stefan_problem
    lbm_physics
    lbm_core
    gtest
    gtest_main
    CUDA::cudart
    CUDA::cuda_driver
)

add_test(NAME StefanProblemBenchmark
    COMMAND test_stefan_problem
)

set_tests_properties(StefanProblemBenchmark PROPERTIES
    TIMEOUT 600
    LABELS "validation;benchmark;stefan;phase_change;analytical;week2;critical"
)

# Benchmark 3: Evaporation Rate Validation
# Validates evaporative mass flux against Hertz-Knudsen formula
add_executable(test_evaporation_rate
    test_evaporation_rate.cu
)

set_target_properties(test_evaporation_rate PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "70;75;80;86"
)

target_include_directories(test_evaporation_rate PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(test_evaporation_rate
    gtest
    gtest_main
    CUDA::cudart
    CUDA::cuda_driver
)

add_test(NAME EvaporationRateBenchmark
    COMMAND test_evaporation_rate
)

set_tests_properties(EvaporationRateBenchmark PROPERTIES
    TIMEOUT 120
    LABELS "validation;benchmark;evaporation;hertz_knudsen;analytical;week2;critical"
)

# ========================================
# WEEK 2 CONVERGENCE VALIDATION TESTS
# ========================================
#
# Automated test suite to verify grid/timestep convergence and
# prevent regression of Week 2 discovered bugs.
#
# Test Suite:
# 1. Grid Independence Validator
# 2. Timestep Convergence Validator
# 3. Energy Conservation Validator
# 4. Regression Test (50W baseline)
# 5. CFL Stability Check
# 6. Configuration Parser Test

# Test 1: Grid Independence Validator
# Analytical grid convergence study using 1D Gaussian diffusion
# Tests 4 grid resolutions (25, 50, 100, 200 cells) and verifies second-order convergence
add_executable(test_grid_convergence
    test_grid_convergence.cu
)

set_target_properties(test_grid_convergence PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "70;75;80;86"
)

target_include_directories(test_grid_convergence PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(test_grid_convergence
    lbm_physics
    lbm_core
    CUDA::cudart
    CUDA::cuda_driver
)

add_test(NAME GridIndependenceValidation
    COMMAND test_grid_convergence
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)

set_tests_properties(GridIndependenceValidation PROPERTIES
    TIMEOUT 600  # 10 minutes (4 grid resolutions, analytical test)
    LABELS "validation;convergence;grid;analytical;critical"
)

# Test 2: Timestep Convergence Validator
add_executable(test_timestep_convergence
    test_timestep_convergence.cu
)

set_target_properties(test_timestep_convergence PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "70;75;80;86"
)

target_include_directories(test_timestep_convergence PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(test_timestep_convergence
    lbm_physics
    lbm_core
    gtest
    gtest_main
    CUDA::cudart
    CUDA::cuda_driver
)

add_test(NAME TimestepConvergenceValidation
    COMMAND test_timestep_convergence
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)

set_tests_properties(TimestepConvergenceValidation PROPERTIES
    TIMEOUT 600  # 10 minutes (4 timesteps with analytical validation)
    LABELS "validation;convergence;timestep;analytical;critical"
)

# Test 3: Energy Conservation Validator
add_executable(test_energy_conservation_timestep
    test_energy_conservation_timestep.cu
)

set_target_properties(test_energy_conservation_timestep PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "70;75;80;86"
)

target_include_directories(test_energy_conservation_timestep PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(test_energy_conservation_timestep
    CUDA::cudart
)

add_test(NAME EnergyConservationTimestepValidation
    COMMAND test_energy_conservation_timestep
)

set_tests_properties(EnergyConservationTimestepValidation PROPERTIES
    TIMEOUT 1800  # 30 minutes (3 simulations)
    LABELS "validation;energy;conservation;week2;critical"
)

# Test 4: Regression Test (50W Baseline)
add_executable(test_regression_50W
    test_regression_50W.cu
)

set_target_properties(test_regression_50W PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "70;75;80;86"
)

target_include_directories(test_regression_50W PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(test_regression_50W
    CUDA::cudart
)

add_test(NAME RegressionTest50W
    COMMAND test_regression_50W
)

set_tests_properties(RegressionTest50W PROPERTIES
    TIMEOUT 600  # 10 minutes
    LABELS "validation;regression;baseline;week2;critical"
)

# Test 5: CFL Stability Check
add_executable(test_cfl_stability
    test_cfl_stability.cu
)

set_target_properties(test_cfl_stability PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "70;75;80;86"
)

target_include_directories(test_cfl_stability PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(test_cfl_stability
    CUDA::cudart
)

add_test(NAME CFLStabilityCheck
    COMMAND test_cfl_stability
)

set_tests_properties(CFLStabilityCheck PROPERTIES
    TIMEOUT 60  # 1 minute (analytical test)
    LABELS "validation;stability;cfl;week2;critical"
)

# Test 6: Configuration Parser Test
add_executable(test_config_parser
    test_config_parser.cu
)

set_target_properties(test_config_parser PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "70;75;80;86"
)

target_include_directories(test_config_parser PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(test_config_parser
    CUDA::cudart
)

add_test(NAME ConfigurationParserValidation
    COMMAND test_config_parser
)

set_tests_properties(ConfigurationParserValidation PROPERTIES
    TIMEOUT 30  # Quick unit test - should complete in ~1 second
    LABELS "validation;config;parser;week2;critical"
)

# ========================================
# BUG 3 REGRESSION TEST
# ========================================
#
# Automated test to prevent regression of Bug 3 (energy diagnostic dt-scaling)
# This test verifies that energy conservation diagnostics work correctly
# across different timestep sizes and follow normal convergence behavior.
#
# Bug 3 Summary:
# - Energy errors showed paradoxical behavior (fine timestep → WORST error)
# - Root cause: Energy diagnostic calculation had incorrect dt scaling
# - Expected after fix: fine timestep → BEST error (normal convergence)
#
# Test Runtime: ~3-5 minutes (3 short simulations of 20μs each)

add_executable(test_bug3_energy_diagnostic
    test_bug3_energy_diagnostic.cu
)

set_target_properties(test_bug3_energy_diagnostic PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "70;75;80;86"
)

target_include_directories(test_bug3_energy_diagnostic PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(test_bug3_energy_diagnostic
    CUDA::cudart
)

add_test(NAME Bug3EnergyDiagnosticRegression
    COMMAND test_bug3_energy_diagnostic
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)

set_tests_properties(Bug3EnergyDiagnosticRegression PROPERTIES
    TIMEOUT 600  # 10 minutes (3 simulations)
    LABELS "validation;regression;energy;bug3;week2;critical"
)

# ========================================
# WEEK 3 READINESS VALIDATION SUITE
# ========================================
#
# Comprehensive test suite to validate all critical requirements before
# proceeding with Week 3 vapor phase implementation.
#
# This suite tests:
# - P0 (Critical): Thermal/Fluid tau scaling, steady state, energy conservation, stability
# - P1 (High Priority): Timestep convergence, temperature validation, CFL stability
#
# Decision Criteria:
# - FULL GO: Score >= 85/100
# - CONDITIONAL GO: Score 70-84/100
# - NO GO: Score < 70/100

add_executable(test_week3_readiness
    test_week3_readiness.cu
)

set_target_properties(test_week3_readiness PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "70;75;80;86"
)

target_include_directories(test_week3_readiness PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(test_week3_readiness
    lbm_physics
    lbm_core
    gtest
    gtest_main
    CUDA::cudart
    CUDA::cuda_driver
)

add_test(NAME Week3ReadinessValidation
    COMMAND test_week3_readiness
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)

set_tests_properties(Week3ReadinessValidation PROPERTIES
    TIMEOUT 300  # 5 minutes
    LABELS "validation;week3;readiness;critical;go_nogo"
)

# ========================================
# SENIOR'S THESIS VALIDATION CASES
# ========================================
#
# These tests replicate the exact validation cases from the senior's thesis
# to verify correctness against established benchmarks.

# Case 1: 3D Heat Diffusion (Gaussian pulse)
# Validates thermal solver against analytical 3D Gaussian diffusion solution
add_executable(test_3d_heat_diffusion_senior
    test_3d_heat_diffusion_senior.cu
)

set_target_properties(test_3d_heat_diffusion_senior PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "70;75;80;86"
)

target_include_directories(test_3d_heat_diffusion_senior PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(test_3d_heat_diffusion_senior
    lbm_physics
    lbm_core
    lbm_io
    gtest
    gtest_main
    CUDA::cudart
    CUDA::cuda_driver
)

add_test(NAME Case1_3D_HeatDiffusion_Senior
    COMMAND test_3d_heat_diffusion_senior
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)

set_tests_properties(Case1_3D_HeatDiffusion_Senior PROPERTIES
    TIMEOUT 600  # 10 minutes
    LABELS "validation;senior;thermal;3d;gaussian;critical"
)

# Case 1 FIXED: 3D Heat Diffusion with aligned timesteps (staircase fix)
# This version fixes the staircase pattern by aligning output times with LBM timesteps
add_executable(test_3d_heat_diffusion_fixed
    test_3d_heat_diffusion_fixed.cu
)

set_target_properties(test_3d_heat_diffusion_fixed PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "70;75;80;86"
)

target_include_directories(test_3d_heat_diffusion_fixed PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(test_3d_heat_diffusion_fixed
    lbm_physics
    lbm_core
    lbm_io
    gtest
    gtest_main
    CUDA::cudart
    CUDA::cuda_driver
)

add_test(NAME Case1_3D_HeatDiffusion_Fixed
    COMMAND test_3d_heat_diffusion_fixed
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)

set_tests_properties(Case1_3D_HeatDiffusion_Fixed PROPERTIES
    TIMEOUT 600  # 10 minutes
    LABELS "validation;senior;thermal;3d;gaussian;fixed;staircase"
)

# ========================================
# CASE 5: LASER MELTING VALIDATION (SENIOR'S CONFIGURATION)
# ========================================
#
# Comprehensive multiphysics validation test following senior's Fe-based
# simulation, adapted for Ti6Al4V. This test validates the complete coupling
# of thermal, fluid, VOF, Marangoni, and phase change solvers.
#
# Configuration:
# - Domain: 150×300×150 μm (40×80×80 cells)
# - Grid spacing: dx = 3.75 μm
# - Material: Ti6Al4V (adapted from Fe)
# - Laser: 200W, center heating, shutoff at 50 μs
# - Total time: 100 μs (heating + solidification)
#
# Validation metrics:
# - Melt pool formation and growth (0-50 μs)
# - Solidification after laser shutoff (50-100 μs)
# - Marangoni velocity in realistic range (0.5-2 m/s)
# - Melt pool depth evolution
#
# Output:
# - VTK files for ParaView visualization
# - CSV time series (depth, temperature, velocity)
# - Comparison plots (Python post-processing)

add_executable(test_laser_melting_senior
    test_laser_melting_senior.cu
)

set_target_properties(test_laser_melting_senior PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "70;75;80;86"
)

target_include_directories(test_laser_melting_senior PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(test_laser_melting_senior
    lbm_physics
    lbm_core
    lbm_io
    gtest
    gtest_main
    CUDA::cudart
    CUDA::cuda_driver
)

add_test(NAME Case5_LaserMelting_Senior
    COMMAND test_laser_melting_senior
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)

set_tests_properties(Case5_LaserMelting_Senior PROPERTIES
    TIMEOUT 1800  # 30 minutes (100μs simulation with 1ns timestep)
    LABELS "validation;senior;laser;melting;multiphysics;critical"
)

# ========================================
# CASE 6: KEYHOLE FORMATION VALIDATION (SENIOR'S CONFIGURATION)
# ========================================
#
# This test validates keyhole formation under high-power laser irradiation
# following the senior's configuration (adapted from Fe to Ti6Al4V).
#
# Configuration:
# - Domain: 150×300×150 μm (40×80×40 grid)
# - Grid spacing: dx = 3.75 μm
# - Material: Ti6Al4V (ρ_liq=4000, ρ_gas=1.0 kg/m³)
# - Boiling point: T_v = 3560 K
# - Evaporation: Hertz-Knudsen model
# - Recoil pressure: P_recoil = 0.54 × P_sat
# - Laser: 300W (high power for keyhole regime)
# - Total time: 20 μs (keyhole development)
#
# Validation metrics:
# - Keyhole depth evolution (5, 10, 15, 20 μs)
# - Recoil pressure activation and effect
# - Temperature exceeds boiling point
# - No NaN/instability
#
# Success criteria:
# - CRITICAL: Keyhole forms (depth > 10 μm at t=20 μs)
# - CRITICAL: Depth increases over time (monotonic initially)
# - CRITICAL: Temperature exceeds boiling point (T > 3560 K)
# - CRITICAL: Recoil pressure creates deeper keyhole than without
#
# Output:
# - VTK files for ParaView visualization
# - Keyhole depth time series (keyhole_depth.dat)
# - Comparison plots (Python post-processing)
#
# References:
# - Khairallah et al. (2016): LPBF keyhole formation
# - King et al. (2015): Keyhole-mode laser melting observations
# - Tan et al. (2013): Multi-scale modeling of solidification
#
# Temporarily disabled due to VTK writer API mismatch - needs fixing
# add_executable(test_keyhole_formation_senior
#     test_keyhole_formation_senior.cu
# )
#
# set_target_properties(test_keyhole_formation_senior PROPERTIES
#     CUDA_SEPARABLE_COMPILATION ON
#     CUDA_ARCHITECTURES "70;75;80;86"
# )
#
# target_include_directories(test_keyhole_formation_senior PRIVATE
#     ${CMAKE_SOURCE_DIR}/include
#     ${CMAKE_SOURCE_DIR}/src
# )
#
# target_link_libraries(test_keyhole_formation_senior
#     lbm_physics
#     lbm_core
#     lbm_io
#     gtest
#     gtest_main
#     CUDA::cudart
#     CUDA::cuda_driver
# )
#
# add_test(NAME Case6_KeyholeFormation_Senior
#     COMMAND test_keyhole_formation_senior
#     WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
# )
#
# set_tests_properties(Case6_KeyholeFormation_Senior PROPERTIES
#     TIMEOUT 1200  # 20 minutes (high-power simulation with evaporation/recoil)
#     LABELS "validation;senior;keyhole;evaporation;recoil;critical"
# )

# ========================================
# Future Validation Tests
# ========================================
#
# Planned additions:
# - test_laplace_pressure.cu (surface tension validation)
# - test_melt_pool_dimensions.cu (LPBF melt pool geometry)

# ========================================
# LASER ENERGY DIAGNOSTIC TESTS
# ========================================
#
# Diagnostic test suite to debug temperature magnitude differences
# between LBMProject and walberla laser melting simulations.
#
# Investigation points:
# - Total absorbed power integration (should equal P × η)
# - Beer-Lambert depth penetration verification
# - Energy accumulation rate over time
# - Comparison of LBM vs walberla heat source formulations
#
# Expected findings:
# - walberla uses spot_radius as absorption depth (50 μm)
# - LBMProject uses penetration_depth (10 μm)
# - This 5× difference in depth causes ~5× difference in peak temperature

add_executable(test_laser_energy_diagnostic
    test_laser_energy_diagnostic.cu
)

set_target_properties(test_laser_energy_diagnostic PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "70;75;80;86"
)

target_include_directories(test_laser_energy_diagnostic PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(test_laser_energy_diagnostic
    lbm_physics
    lbm_core
    gtest
    gtest_main
    CUDA::cudart
    CUDA::cuda_driver
)

add_test(NAME LaserEnergyDiagnostic
    COMMAND test_laser_energy_diagnostic
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)

set_tests_properties(LaserEnergyDiagnostic PROPERTIES
    TIMEOUT 300  # 5 minutes
    LABELS "validation;laser;energy;diagnostic;debug"
)

# ========================================
# MULTIPHYSICS VALIDATION TESTS
# ========================================

# Add multiphysics validation subdirectory
add_subdirectory(multiphysics)

# ========================================
# THERMAL PARAMETER DIAGNOSTIC TESTS
# ========================================
#
# Diagnostic test suite to systematically identify thermal parameter
# discrepancies between LBMProject and walberla.
#
# This test performs detailed analysis of:
# 1. Thermal diffusivity calculation
# 2. Lattice Boltzmann parameters (tau, omega)
# 3. Laser energy deposition
# 4. Temperature field evolution
# 5. Energy balance checks
#
# Purpose: Identify why peak temperature (3,544.8 K) is much lower
# than walberla reference (17,500 K).

add_executable(diagnostic_thermal_parameters
    diagnostic_thermal_parameters.cu
)

set_target_properties(diagnostic_thermal_parameters PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "70;75;80;86"
)

target_include_directories(diagnostic_thermal_parameters PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(diagnostic_thermal_parameters
    lbm_physics
    lbm_core
    gtest
    gtest_main
    CUDA::cudart
    CUDA::cuda_driver
)

add_test(NAME ThermalParameterDiagnostic
    COMMAND diagnostic_thermal_parameters
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)

set_tests_properties(ThermalParameterDiagnostic PROPERTIES
    TIMEOUT 60  # 1 minute (analytical tests)
    LABELS "validation;thermal;diagnostic;parameters;walberla"
)

# ========================================
# FINITE DIFFERENCE REFERENCE TEST
# ========================================
#
# Standalone FD thermal solver to verify understanding of walberla's behavior.
# This test implements explicit finite difference heat equation solver with
# EXACTLY the same parameters as walberla to confirm expected ~19,500 K peak.
#
# Purpose: Determine if LBM vs walberla temperature difference is due to:
# 1. FD vs LBM method difference (physical), or
# 2. Parameter mismatch (bug)
#
# Expected result: Should match walberla's ~19,500 K if correctly implemented.

add_executable(test_fd_thermal_reference
    test_fd_thermal_reference.cu
)

set_target_properties(test_fd_thermal_reference PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "70;75;80;86"
)

target_include_directories(test_fd_thermal_reference PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(test_fd_thermal_reference
    gtest
    gtest_main
    CUDA::cudart
    CUDA::cuda_driver
)

add_test(NAME FDThermalReference
    COMMAND test_fd_thermal_reference
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)

set_tests_properties(FDThermalReference PROPERTIES
    TIMEOUT 600  # 10 minutes (50 μs simulation with dt=100 ns = 500 steps)
    LABELS "validation;thermal;fd;reference;walberla;diagnostic"
)

# ========================================
# ENERGY BUDGET CONSERVATION TEST
# ========================================
#
# Rigorous energy conservation test to identify energy loss mechanisms
# in the LBM thermal solver.
#
# This test implements strict energy budget analysis with adiabatic boundaries:
#
# EXPECTED BEHAVIOR:
# - Laser input: P × η = 200 × 0.35 = 70 W
# - Over 50 μs: E_total = 70 × 50e-6 = 3.5 mJ
# - With ADIABATIC boundaries: E_stored should equal E_input
#
# ENERGY BALANCE:
#   E_input = ∫∫∫ Q(x,y,z) × dt dV  [J] - Total energy deposited by laser
#   E_stored = ∫∫∫ ρ × cp × (T - T0) dV  [J] - Total sensible energy
#   E_balance = E_input - E_stored  [J] - Energy discrepancy
#
# If E_balance > 5% of E_input, there is a BUG.
#
# Diagnostic outputs:
# - Detailed energy balance report (ENERGY_BUDGET_ANALYSIS.md)
# - Time series of energy input/stored
# - Spatial energy distribution
# - Identification of loss mechanisms
#
# Purpose: Determine if energy conservation is the root cause of
# temperature discrepancies with walberla simulations.

add_executable(test_energy_budget
    test_energy_budget.cu
)

set_target_properties(test_energy_budget PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "70;75;80;86"
)

target_include_directories(test_energy_budget PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(test_energy_budget
    lbm_physics
    lbm_core
    gtest
    gtest_main
    CUDA::cudart
    CUDA::cuda_driver
)

add_test(NAME EnergyBudgetConservation
    COMMAND test_energy_budget
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)

set_tests_properties(EnergyBudgetConservation PROPERTIES
    TIMEOUT 300  # 5 minutes (50 μs simulation with multiple diagnostics)
    LABELS "validation;energy;conservation;budget;diagnostic;critical"
)

# ========================================
# THERMAL SOURCE ENERGY CONSERVATION TEST
# ========================================
#
# Unit test to verify that addHeatSourceKernel correctly deposits energy:
#   dT_actual = Q * dt / (rho * cp)
#
# This is a minimal, focused test to check if the heat source implementation
# has a fundamental bug (e.g., missing correction factor, wrong formula).
#
# Test procedure:
# 1. Initialize uniform temperature field T0
# 2. Apply uniform heat source Q for one timestep
# 3. Verify: T_final = T0 + Q*dt/(rho*cp)
# 4. No collision, no streaming (pure source term test)
#
# Success criteria:
# - CRITICAL: Relative error < 0.0001% (should be exact)
# - CRITICAL: Temperature field is uniform (max deviation < 1e-3 K)
#
# This test is designed to isolate the source term from all other physics
# to determine if the 53% temperature discrepancy vs walberla is due to:
# 1. Bug in source term formula (would fail this test)
# 2. System-level difference (boundary conditions, etc.) (would pass this test)

add_executable(thermal_source_energy_test
    thermal_source_energy_test.cu
)

set_target_properties(thermal_source_energy_test PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "70;75;80;86"
)

target_include_directories(thermal_source_energy_test PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(thermal_source_energy_test
    lbm_physics
    lbm_core
    CUDA::cudart
    CUDA::cuda_driver
)

add_test(NAME ThermalSourceEnergyConservation
    COMMAND thermal_source_energy_test
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)

set_tests_properties(ThermalSourceEnergyConservation PROPERTIES
    TIMEOUT 60  # 1 minute (should complete in < 1 second)
    LABELS "validation;thermal;source;energy;unit;critical"
)

# ========================================
# WALBERLA THERMAL MATCH TEST
# ========================================
#
# Pure thermal solver test to exactly match walberla FD solver behavior.
# This test isolates thermal conduction from multiphysics effects to achieve
# numerical consistency with walberla.
#
# Configuration (EXACT walberla match):
# - Explicit FD heat equation: dT/dt = α∇²T + Q/(ρ·cp)
# - 3D Laplacian (6-point stencil)
# - Gaussian laser with Beer-Lambert absorption
# - Domain: 400×400×200 μm (200×200×100 cells)
# - dx = 2 μm, dt = 100 ns
# - Ti6Al4V: ρ=4430, cp=526, k=6.7, α=2.874e-6 m²/s
# - Laser: P=200W, r0=50μm, η=0.35, δ=50μm (δ = r0 in walberla)
# - Dirichlet BC: T=300K on all boundaries
#
# Goal: Match walberla peak temperature within 5%

add_executable(test_thermal_walberla_match
    test_thermal_walberla_match.cu
)

set_target_properties(test_thermal_walberla_match PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "70;75;80;86"
)

target_include_directories(test_thermal_walberla_match PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(test_thermal_walberla_match
    lbm_physics
    lbm_core
    gtest
    gtest_main
    CUDA::cudart
    CUDA::cuda_driver
)

add_test(NAME ThermalWalberlaMatch
    COMMAND test_thermal_walberla_match
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)

set_tests_properties(ThermalWalberlaMatch PROPERTIES
    TIMEOUT 600  # 10 minutes
    LABELS "validation;thermal;walberla;match;lbm;fd;critical"
)

# ========================================
# CRITICAL BUG FIX VALIDATION SUITE
# ========================================
#
# Comprehensive test suite for critical bug fixes:
# 1. CUDA error checking infrastructure
# 2. Evaporation division by zero prevention
# 3. Material property validation
# 4. Evaporation buffer zero initialization
# 5. Boundary cell skipping in phase change
#
# These tests verify that the fixes work correctly and prevent
# regression of critical bugs that could cause silent failures,
# NaN propagation, or incorrect physics.

add_executable(test_bug_fixes
    test_bug_fixes.cu
)

set_target_properties(test_bug_fixes PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "70;75;80;86"
)

target_include_directories(test_bug_fixes PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(test_bug_fixes
    lbm_physics
    lbm_core
    gtest
    gtest_main
    CUDA::cudart
    CUDA::cuda_driver
)

add_test(NAME CriticalBugFixValidation
    COMMAND test_bug_fixes
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)

set_tests_properties(CriticalBugFixValidation PROPERTIES
    TIMEOUT 300  # 5 minutes
    LABELS "validation;bug_fixes;critical;regression"
)

# Gaussian Diffusion Analytical Validation Test
add_executable(test_gaussian_diffusion
    test_gaussian_diffusion.cu
)

set_target_properties(test_gaussian_diffusion PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "70;75;80;86"
)

target_include_directories(test_gaussian_diffusion PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(test_gaussian_diffusion
    lbm_physics
    lbm_core
    gtest
    gtest_main
    CUDA::cudart
    CUDA::cuda_driver
)

add_test(NAME GaussianDiffusion
    COMMAND test_gaussian_diffusion
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)

set_tests_properties(GaussianDiffusion PROPERTIES
    TIMEOUT 300  # 5 minutes
    LABELS "validation;thermal;analytical;gaussian;diffusion;critical"
)

# ========================================
# ENERGY CONSERVATION VALIDATION TEST
# ========================================
#
# Comprehensive energy conservation test for thermal LBM solver with laser heat source.
# This test validates that energy is correctly conserved during laser heating by:
# 1. Computing expected absorbed energy: E_in = P × η × t
# 2. Measuring actual energy stored: E_stored = Σ(ρ·cp·ΔT·dV)
# 3. Verifying energy balance with and without boundary losses
#
# TEST SCENARIOS:
# 1. Adiabatic boundaries (no losses): |E_in - E_stored| / E_in < 0.5%
# 2. With radiation BC: E_in = E_stored + E_radiated (balance residual < 1%)
#
# PARAMETERS (match walberla configuration):
# - Laser: P=200W, η=0.35, r₀=50μm, δ=50μm
# - Material: Ti6Al4V (ρ=4430, cp=526)
# - Domain: 100×100×50 cells, dx=4μm
# - Duration: 10μs (short test for quick validation)
#
# Success criteria:
# - CRITICAL: Energy residual < 0.5% for adiabatic case
# - CRITICAL: Energy balance closes with radiation losses (< 1%)
# - CRITICAL: No NaN or instability
# - CRITICAL: Radiation BC produces measurable energy loss
#
# Context:
# - Thermal solver validated against walberla (2% peak error)
# - Gaussian diffusion shows 0.01% energy conservation
# - This test provides formal laser heating energy conservation validation

add_executable(test_energy_conservation
    test_energy_conservation.cu
)

set_target_properties(test_energy_conservation PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "70;75;80;86"
)

target_include_directories(test_energy_conservation PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(test_energy_conservation
    lbm_physics
    lbm_core
    gtest
    gtest_main
    CUDA::cudart
    CUDA::cuda_driver
)

add_test(NAME EnergyConservationValidation
    COMMAND test_energy_conservation
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)

set_tests_properties(EnergyConservationValidation PROPERTIES
    TIMEOUT 300  # 5 minutes (10 μs simulation × 2 tests)
    LABELS "validation;energy;conservation;laser;thermal;critical"
)

# Minimal heat source test (commented out - file doesn't exist)
# add_executable(test_minimal_heat_source
#     ${CMAKE_SOURCE_DIR}/test_minimal_heat_source.cu
# )
# target_link_libraries(test_minimal_heat_source
#     lbm_physics
#     lbm_core
# )
# target_include_directories(test_minimal_heat_source PRIVATE
#     ${CMAKE_SOURCE_DIR}/include
# )
# set_target_properties(test_minimal_heat_source PROPERTIES
#     CUDA_SEPARABLE_COMPILATION ON
#     CUDA_ARCHITECTURES ${CUDA_ARCHITECTURES}
# )

# ========================================
# PHASE 1 FLUID VALIDATION SUITE
# ========================================
#
# Validation tests for fluid dynamics solver using canonical benchmarks:
# 1. Lid-driven cavity (Re=100, Re=400) - Ghia et al. 1982
# 2. Grid convergence with Poiseuille flow

# Test 1: Lid-Driven Cavity Re=100
add_executable(test_lid_driven_cavity_re100
    test_lid_driven_cavity_re100.cu
)

set_target_properties(test_lid_driven_cavity_re100 PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "70;75;80;86"
)

target_include_directories(test_lid_driven_cavity_re100 PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/tests
)

target_link_libraries(test_lid_driven_cavity_re100
    lbm_physics
    lbm_core
    gtest
    gtest_main
    CUDA::cudart
    CUDA::cuda_driver
)

add_test(NAME LidDrivenCavityRe100
    COMMAND test_lid_driven_cavity_re100
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)

set_tests_properties(LidDrivenCavityRe100 PROPERTIES
    TIMEOUT 600  # 10 minutes
    LABELS "validation;fluid;benchmark;critical;phase1"
)

# Test 2: Lid-Driven Cavity Re=400
add_executable(test_lid_driven_cavity_re400
    test_lid_driven_cavity_re400.cu
)

set_target_properties(test_lid_driven_cavity_re400 PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "70;75;80;86"
)

target_include_directories(test_lid_driven_cavity_re400 PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/tests
)

target_link_libraries(test_lid_driven_cavity_re400
    lbm_physics
    lbm_core
    gtest
    gtest_main
    CUDA::cudart
    CUDA::cuda_driver
)

add_test(NAME LidDrivenCavityRe400
    COMMAND test_lid_driven_cavity_re400
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)

set_tests_properties(LidDrivenCavityRe400 PROPERTIES
    TIMEOUT 900  # 15 minutes (higher Re needs more iterations)
    LABELS "validation;fluid;benchmark;phase1"
)

# Test 3: Fluid Grid Convergence (Poiseuille Flow)
add_executable(test_fluid_grid_convergence
    test_fluid_grid_convergence.cu
)

set_target_properties(test_fluid_grid_convergence PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "70;75;80;86"
)

target_include_directories(test_fluid_grid_convergence PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/tests
)

target_link_libraries(test_fluid_grid_convergence
    lbm_physics
    lbm_core
    gtest
    gtest_main
    CUDA::cudart
    CUDA::cuda_driver
)

add_test(NAME FluidGridConvergence
    COMMAND test_fluid_grid_convergence
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)

set_tests_properties(FluidGridConvergence PROPERTIES
    TIMEOUT 600  # 10 minutes (4 grids: 25, 50, 100, 200)
    LABELS "validation;fluid;convergence;analytical;phase1"
)

# ========================================
# Test 4: Natural Convection (de Vahl Davis 1983)
# ========================================
#
# Validation test for coupled thermal-fluid solver using the canonical
# natural convection benchmark in a differentially heated square cavity.
#
# Reference:
# de Vahl Davis, G. (1983). "Natural convection of air in a square cavity:
# A bench mark numerical solution." IJNMF, 3(3), 249-264.
#
# Tests:
# - Ra = 10³ (conduction-dominated)
# - Ra = 10⁴ (standard benchmark case)
# - Ra = 10⁵ (convection-dominated, DISABLED by default)
#
# Validation metrics:
# - Average Nusselt number (Nu_avg within 5%)
# - Maximum velocities (u_max, v_max within 10%)
# - Steady-state convergence
#
add_executable(test_natural_convection
    test_natural_convection.cu
)

set_target_properties(test_natural_convection PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "70;75;80;86"
)

target_include_directories(test_natural_convection PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/tests
)

target_link_libraries(test_natural_convection
    lbm_physics
    lbm_core
    gtest
    gtest_main
    CUDA::cudart
    CUDA::cuda_driver
)

add_test(NAME NaturalConvectionRa1e4
    COMMAND test_natural_convection --gtest_filter=NaturalConvectionTest.RayleighNumber1e4
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)

set_tests_properties(NaturalConvectionRa1e4 PROPERTIES
    TIMEOUT 900  # 15 minutes (main benchmark case)
    LABELS "validation;thermal;fluid;buoyancy;benchmark;critical;phase2"
)

add_test(NAME NaturalConvectionRa1e3
    COMMAND test_natural_convection --gtest_filter=NaturalConvectionTest.RayleighNumber1e3
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)

set_tests_properties(NaturalConvectionRa1e3 PROPERTIES
    TIMEOUT 600  # 10 minutes (faster convergence for low Ra)
    LABELS "validation;thermal;fluid;buoyancy;benchmark;phase2"
)

# ========================================
# Test 5: Couette-Poiseuille Flow
# ========================================
#
# Validation test for fluid dynamics combining shear-driven (Couette)
# and pressure-driven (Poiseuille) flow in a 2D channel.
#
# Reference:
# Classical fluid mechanics. Analytical solution:
# u(y) = 3*y^2 - 2*y (normalized coordinates y in [0,1])
#
# Tests:
# - Combined boundary-driven and body-force-driven flow
# - Moving wall boundary condition (top wall at U=1 m/s)
# - No-slip wall boundary condition (bottom wall)
# - Body force (pressure gradient)
# - Steady-state convergence
#
# Validation metrics:
# - L2 error < 5% against analytical solution
# - Velocity profile shape matches parabolic+linear combination
# - Boundary conditions satisfied (u(0)=0, u(1) satisfies BC)
#
# Expected runtime: 5-10 minutes to reach steady state
#
add_executable(test_couette_poiseuille
    test_couette_poiseuille.cu
)

set_target_properties(test_couette_poiseuille PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "70;75;80;86"
)

target_include_directories(test_couette_poiseuille PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/tests
)

target_link_libraries(test_couette_poiseuille
    lbm_physics
    lbm_core
    gtest
    gtest_main
    CUDA::cudart
    CUDA::cuda_driver
)

add_test(NAME CouettePoiseuille
    COMMAND test_couette_poiseuille
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)

set_tests_properties(CouettePoiseuille PROPERTIES
    TIMEOUT 600  # 10 minutes
    LABELS "validation;fluid;benchmark;analytical;phase1;critical"
)

# ========================================
# PURE COUETTE FLOW (walberla standard)
# ========================================
#
# Validates pure shear-driven flow with NO body force.
# Analytical solution: u(y) = U_top * (y/H) [LINEAR]
# Target: L2 error < 1% (walberla achieves <1e-10%)
#
add_executable(test_pure_couette
    test_pure_couette.cu
)

set_target_properties(test_pure_couette PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "70;75;80;86"
)

target_include_directories(test_pure_couette PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/tests
)

target_link_libraries(test_pure_couette
    lbm_physics
    lbm_core
    gtest
    gtest_main
    CUDA::cudart
    CUDA::cuda_driver
)

add_test(NAME PureCouetteFlow
    COMMAND test_pure_couette
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)

set_tests_properties(PureCouetteFlow PROPERTIES
    TIMEOUT 300  # 5 minutes
    LABELS "validation;fluid;couette;analytical;walberla;critical"
)

# ========================================
# PURE POISEUILLE FLOW (walberla standard)
# ========================================
#
# Validates pure pressure-driven flow with NO moving walls.
# Analytical solution: u(y) = u_max * 4*eta*(1-eta) [PARABOLIC]
# Target: L2 error < 2%, flow rate error < 1% (walberla achieves 0.65%)
#
add_executable(test_pure_poiseuille
    test_pure_poiseuille.cu
)

set_target_properties(test_pure_poiseuille PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "70;75;80;86"
)

target_include_directories(test_pure_poiseuille PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/tests
)

target_link_libraries(test_pure_poiseuille
    lbm_physics
    lbm_core
    gtest
    gtest_main
    CUDA::cudart
    CUDA::cuda_driver
)

add_test(NAME PurePoiseuilleFlow
    COMMAND test_pure_poiseuille
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)

set_tests_properties(PurePoiseuilleFlow PROPERTIES
    TIMEOUT 300  # 5 minutes
    LABELS "validation;fluid;poiseuille;analytical;walberla;critical"
)

# ========================================
# TRT COLLISION MODEL VALIDATION TESTS
# ========================================
#
# Validation tests for Two-Relaxation-Time (TRT) collision operator.
# TRT improves accuracy over BGK/SRT by using separate relaxation rates
# for symmetric and anti-symmetric populations.
#
# Expected improvements:
# - Pure Couette: 0.043% → <0.01% (approaching machine precision)
# - Pure Poiseuille: 1.13% → ~0.3-0.5% (2-3× better accuracy)
#
# Tests:
# 1. test_trt_couette.cu - Pure Couette flow with TRT
# 2. test_trt_poiseuille.cu - Pure Poiseuille flow with TRT
# 3. test_trt_magic_number.cu - Verify magic number effect on wall accuracy

# Test 1: TRT Pure Couette Flow
add_executable(test_trt_couette
    test_trt_couette.cu
)

set_target_properties(test_trt_couette PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "70;75;80;86"
)

target_include_directories(test_trt_couette PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/tests
)

target_link_libraries(test_trt_couette
    lbm_physics
    lbm_core
    gtest
    gtest_main
    CUDA::cudart
    CUDA::cuda_driver
)

add_test(NAME TRTCouetteFlow
    COMMAND test_trt_couette
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)

set_tests_properties(TRTCouetteFlow PROPERTIES
    TIMEOUT 300  # 5 minutes
    LABELS "validation;fluid;couette;trt;analytical;critical"
)

# Test 2: TRT Pure Poiseuille Flow
add_executable(test_trt_poiseuille
    test_trt_poiseuille.cu
)

set_target_properties(test_trt_poiseuille PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "70;75;80;86"
)

target_include_directories(test_trt_poiseuille PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/tests
)

target_link_libraries(test_trt_poiseuille
    lbm_physics
    lbm_core
    gtest
    gtest_main
    CUDA::cudart
    CUDA::cuda_driver
)

add_test(NAME TRTPoiseuilleFlow
    COMMAND test_trt_poiseuille
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)

set_tests_properties(TRTPoiseuilleFlow PROPERTIES
    TIMEOUT 300  # 5 minutes
    LABELS "validation;fluid;poiseuille;trt;analytical;critical"
)

# Test 3: TRT vs BGK Comparison (validates magic number Λ=3/16)
add_executable(test_trt_magic_number
    test_trt_magic_number.cu
)

set_target_properties(test_trt_magic_number PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "70;75;80;86"
)

target_include_directories(test_trt_magic_number PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/tests
)

target_link_libraries(test_trt_magic_number
    lbm_physics
    lbm_core
    gtest
    gtest_main
    CUDA::cudart
    CUDA::cuda_driver
)

add_test(NAME TRTvsBGKComparison
    COMMAND test_trt_magic_number
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)

set_tests_properties(TRTvsBGKComparison PROPERTIES
    TIMEOUT 600  # 10 minutes (runs 2 simulations: BGK + TRT)
    LABELS "validation;fluid;trt;bgk;comparison;critical"
)

# Test 4: TRT Combined Couette-Poiseuille Flow
add_executable(test_trt_couette_poiseuille
    test_trt_couette_poiseuille.cu
)

set_target_properties(test_trt_couette_poiseuille PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "70;75;80;86"
)

target_include_directories(test_trt_couette_poiseuille PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/tests
)

target_link_libraries(test_trt_couette_poiseuille
    lbm_physics
    lbm_core
    lbm_io
    gtest
    gtest_main
    CUDA::cudart
    CUDA::cuda_driver
)

add_test(NAME TRTCouettePoiseuilleFlow
    COMMAND test_trt_couette_poiseuille
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)

set_tests_properties(TRTCouettePoiseuilleFlow PROPERTIES
    TIMEOUT 600  # 10 minutes
    LABELS "validation;fluid;couette_poiseuille;trt;analytical;critical"
)

# ========================================
# WALBERLA-MATCHING COUETTE-POISEUILLE TEST
# ========================================
#
# This test EXACTLY reproduces walberla's CouettePoiseuilleTest.cpp configuration
# to validate that our LBM implementation matches walberla's behavior and accuracy.
#
# Configuration (EXACT walberla match):
# - Grid: 4 × 64 × 4 (quasi-2D channel)
# - Channel height: H = 63 (NY - 1, node-based coordinates)
# - Reynolds number: Re = 100 (changed from 10 for SRT stability)
# - Top wall velocity: U_top = 0.05
# - TRT collision with magic number Λ = 3/16
# - Body force: f_x = -6*U_top*ν/H²
# - Analytical solution: u(η) = U_top*(3η² - 2η)
# - Include ALL cells in error computation (j=0 to j=NY-1)
#
# Expected results:
# - L2 error < 2% (target: match or beat walberla)
# - Clear improvement over BGK (factor 1.5-5×)
# - Steady state convergence
# - No NaN or instability
#
# This test is CRITICAL for validating that our TRT implementation
# achieves walberla-level accuracy for combined flow problems.
#
add_executable(test_couette_poiseuille_walberla_match
    test_couette_poiseuille_walberla_match.cu
)

set_target_properties(test_couette_poiseuille_walberla_match PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "70;75;80;86"
)

target_include_directories(test_couette_poiseuille_walberla_match PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/tests
)

target_link_libraries(test_couette_poiseuille_walberla_match
    lbm_physics
    lbm_core
    lbm_io
    gtest
    gtest_main
    CUDA::cudart
    CUDA::cuda_driver
)

add_test(NAME CouettePoiseuilleWalberlaMatch
    COMMAND test_couette_poiseuille_walberla_match
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)

set_tests_properties(CouettePoiseuilleWalberlaMatch PROPERTIES
    TIMEOUT 600  # 10 minutes
    LABELS "validation;fluid;couette_poiseuille;walberla;trt;exact_match;critical"
)

# ========================================
# COUETTE-POISEUILLE RE=10 PROOF TEST
# ========================================
#
# This proof test demonstrates that Reynolds number is the dominant factor
# in LBM accuracy. It uses the EXACT SAME configuration as walberla_match
# but changes only Re from 100 to 10. Expected: L2 error drops from ~2% to < 0.5%.
#
# Purpose: Prove the ~2% error at Re=100 is due to physics (low viscosity), not bugs.
#

add_executable(test_couette_poiseuille_re10_proof
    test_couette_poiseuille_re10_proof.cu
)

set_target_properties(test_couette_poiseuille_re10_proof PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "70;75;80;86"
)

target_include_directories(test_couette_poiseuille_re10_proof PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/tests
)

target_link_libraries(test_couette_poiseuille_re10_proof
    lbm_physics
    lbm_core
    gtest
    gtest_main
    CUDA::cudart
    CUDA::cuda_driver
)

add_test(NAME CouettePoiseuilleRe10Proof
    COMMAND test_couette_poiseuille_re10_proof
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)

set_tests_properties(CouettePoiseuilleRe10Proof PROPERTIES
    TIMEOUT 300  # 5 minutes (faster convergence at Re=10)
    LABELS "validation;fluid;couette_poiseuille;reynolds;proof;critical"
)

# ========================================
# COUETTE FLOW DIAGNOSTIC TEST
# ========================================
#
# This diagnostic test identifies the root cause of the 1000× accuracy
# difference between our implementation and walberla's machine precision.
#
# Three validation methods:
# 1. Interior only (j=1 to j=62) - OUR CURRENT APPROACH
# 2. All cells (j=0 to j=63) - WALBERLA APPROACH
# 3. Midpoint shift (j+0.5)
#
# Expected result: Method 2 should achieve 10-100× better accuracy,
# proving that boundary cell exclusion is the root cause of our 0.051% error.
#
add_executable(test_couette_diagnostic
    test_couette_diagnostic.cu
)

set_target_properties(test_couette_diagnostic PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "70;75;80;86"
)

target_include_directories(test_couette_diagnostic PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/tests
)

target_link_libraries(test_couette_diagnostic
    lbm_physics
    lbm_core
    gtest
    gtest_main
    CUDA::cudart
    CUDA::cuda_driver
)

add_test(NAME CouetteDiagnostic
    COMMAND test_couette_diagnostic
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)

set_tests_properties(CouetteDiagnostic PROPERTIES
    TIMEOUT 300  # 5 minutes
    LABELS "validation;fluid;couette;diagnostic;walberla;critical"
)

# ========================================
# WALBERLA EXACT MATCH TEST - COUETTE FLOW
# ========================================
#
# This test EXACTLY reproduces walberla's BoundaryForceCouette.cpp test
# to achieve machine precision (<1e-6 error) matching walberla's results.
#
# Configuration (matching walberla):
# - Grid: 16 × 16 × 16 (L=16)
# - TRT collision with magic number Λ = 3/16
# - omega = 1.0 (tau = 1.0)
# - Top wall velocity: (0.01, 0, 0)
# - Bottom wall: no-slip (stationary)
# - 2000 timesteps
# - Pure Couette flow (NO body force)
#
# Analytical solution: u(y) = U_wall * (y/H) [LINEAR]
#
# Expected result: L2 error < 1e-6 (machine precision)
#
# Reference: /home/yzk/walberla/tests/lbm/boundary/BoundaryForceCouette.cpp
#
add_executable(test_walberla_couette_match
    test_walberla_couette_match.cu
)

set_target_properties(test_walberla_couette_match PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "70;75;80;86"
)

target_include_directories(test_walberla_couette_match PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/tests
)

target_link_libraries(test_walberla_couette_match
    lbm_physics
    lbm_core
    gtest
    gtest_main
    CUDA::cudart
    CUDA::cuda_driver
)

add_test(NAME WalberlaCouetteMatch
    COMMAND test_walberla_couette_match
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)

set_tests_properties(WalberlaCouetteMatch PROPERTIES
    TIMEOUT 300  # 5 minutes
    LABELS "validation;fluid;couette;walberla;exact_match;critical;trt"
)

# ========================================
# PURE POISEUILLE FLOW TEST - BODY FORCE ONLY
# ========================================
#
# This test isolates the body force implementation by testing pure
# Poiseuille flow with NO moving walls. Both walls are stationary (u=0).
#
# Purpose: Diagnose the 1.977% error in Couette-Poiseuille by isolating
# the body force component from the Couette (moving wall) component.
#
# Configuration:
# - Grid: 4 × 64 × 4 (same as Couette-Poiseuille)
# - Channel height: H = 63
# - nu = 0.0315, fx = -2.38095e-06 (same as Couette-Poiseuille)
# - TRT collision with Λ = 3/16
# - Both walls stationary: u_bottom = u_top = 0
# - Flow driven purely by body force
#
# Analytical solution: u(y) = fx/(2*nu) * y * (H - y) [PARABOLIC]
#
# Expected result:
# - If body force is correct: L2 error < 0.1%
# - If body force has bug: L2 error ~ 1-2% (similar to Couette-Poiseuille)
#
# This will definitively determine if the 1.977% error originates from:
# a) Body force implementation (if this test also shows ~2% error)
# b) Interaction between body force and moving walls (if this test passes)
#
add_executable(test_pure_poiseuille_walberla
    test_pure_poiseuille_walberla.cu
)

set_target_properties(test_pure_poiseuille_walberla PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "70;75;80;86"
)

target_include_directories(test_pure_poiseuille_walberla PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/tests
)

target_link_libraries(test_pure_poiseuille_walberla
    lbm_physics
    lbm_core
    lbm_io
    gtest
    gtest_main
    CUDA::cudart
    CUDA::cuda_driver
)

add_test(NAME PurePoiseuille
    COMMAND test_pure_poiseuille_walberla
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)

set_tests_properties(PurePoiseuille PROPERTIES
    TIMEOUT 600  # 10 minutes
    LABELS "validation;fluid;poiseuille;body_force;diagnostic;critical"
)

# ========================================
# COUETTE-POISEUILLE DIAGNOSTIC SUITE
# ========================================
#
# Systematic diagnostic tests to identify error sources and reduce
# L2 error from 1.977% to < 1%.
#
# Tests:
# 1. Grid refinement effect (30, 64, 128 cells)
# 2. Mach number effect (U=0.05 vs U=0.01)
# 3. BGK vs TRT comparison
# 4. Error distribution analysis
# 5. Combined optimization (best settings)
#
# Purpose: Determine CONCRETE actions (not theory) to achieve < 1% error.
#
add_executable(test_couette_poiseuille_diagnostic
    test_couette_poiseuille_diagnostic.cu
)

set_target_properties(test_couette_poiseuille_diagnostic PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "70;75;80;86"
)

target_include_directories(test_couette_poiseuille_diagnostic PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/tests
)

target_link_libraries(test_couette_poiseuille_diagnostic
    lbm_physics
    lbm_core
    lbm_io
    gtest
    gtest_main
    CUDA::cudart
    CUDA::cuda_driver
)

add_test(NAME CouettePoiseuileDiagnostic
    COMMAND test_couette_poiseuille_diagnostic
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)

set_tests_properties(CouettePoiseuileDiagnostic PROPERTIES
    TIMEOUT 1800  # 30 minutes (runs 5 comprehensive diagnostics)
    LABELS "validation;fluid;diagnostic;optimization;critical"
)

# ============================================================================
# Couette-Poiseuille Precision Test
# Target: <0.01% L2 error with proper convergence
# ============================================================================
add_executable(test_couette_poiseuille_precision
    test_couette_poiseuille_precision.cu
)

set_target_properties(test_couette_poiseuille_precision PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "70;75;80;86"
)

target_include_directories(test_couette_poiseuille_precision PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/tests
)

target_link_libraries(test_couette_poiseuille_precision
    lbm_physics
    lbm_core
    lbm_io
    gtest
    gtest_main
    CUDA::cudart
    CUDA::cuda_driver
)

add_test(NAME CouettePoiseuilePrecision
    COMMAND test_couette_poiseuille_precision
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)

set_tests_properties(CouettePoiseuilePrecision PROPERTIES
    TIMEOUT 3600  # 60 minutes (128-cell needs long convergence)
    LABELS "validation;fluid;precision;convergence;critical"
)

# ============================================================================
# Couette-Poiseuille 64x128 Re=100 Test
# Matches reference image specifications
# ============================================================================
add_executable(test_couette_poiseuille_64x128_re100
    test_couette_poiseuille_64x128_re100.cu
)

set_target_properties(test_couette_poiseuille_64x128_re100 PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "70;75;80;86"
)

target_include_directories(test_couette_poiseuille_64x128_re100 PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/tests
)

target_link_libraries(test_couette_poiseuille_64x128_re100
    lbm_physics
    lbm_core
    lbm_io
    gtest
    gtest_main
    CUDA::cudart
    CUDA::cuda_driver
)

add_test(NAME CouettePoiseuille64x128Re100
    COMMAND test_couette_poiseuille_64x128_re100
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)

set_tests_properties(CouettePoiseuille64x128Re100 PROPERTIES
    TIMEOUT 3600  # 60 minutes (128-cell grid, Re=100 may need long convergence)
    LABELS "validation;fluid;couette_poiseuille;re100;64x128;critical"
)

# ========================================
# VOF VALIDATION TESTS
# ========================================

# Test: Oscillating Droplet
# Validates surface tension implementation through droplet oscillation.
# Initial ellipsoidal droplet relaxes to sphere under surface tension,
# oscillating at frequency predicted by Lamb's formula: f₂ = (1/2π)√(8σ/(ρR³))
#
# Configuration:
# - Grid: 100³ (can be reduced to 64³ for faster testing)
# - Initial: Ellipsoid with aspect ratio 1.2
# - Surface tension: σ = 0.072 N/m (water-air)
# - Density: ρ = 1000 kg/m³
# - Radius: R = 20 cells (2 mm with dx=0.1mm)
#
# Validation metrics:
# - Frequency error < 10%
# - Mass conservation < 1% loss
# - Clean sinusoidal oscillation
#
# Expected runtime: 10-20 minutes (5 oscillation periods)
#
add_executable(test_oscillating_droplet
    test_oscillating_droplet.cu
)

set_target_properties(test_oscillating_droplet PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "70;75;80;86"
)

target_include_directories(test_oscillating_droplet PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/tests
)

target_link_libraries(test_oscillating_droplet
    lbm_physics
    lbm_core
    gtest
    gtest_main
    CUDA::cudart
    CUDA::cuda_driver
)

add_test(NAME OscillatingDropletValidation
    COMMAND test_oscillating_droplet
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)

set_tests_properties(OscillatingDropletValidation PROPERTIES
    TIMEOUT 1800  # 30 minutes
    LABELS "validation;vof;surface_tension;oscillating_droplet;critical"
)

# ========================================
# VOF Buoyancy Force Test
# ========================================
#
# This test validates the VOF-based buoyancy force calculation against
# analytical expectations for a stratified two-phase system.
#
# Configuration:
# - Grid: 10x10x10 cells
# - Stratified layers: gas (top), interface (middle), liquid (bottom)
# - ρ_liquid = 1000 kg/m³ (water)
# - ρ_gas = 1.2 kg/m³ (air)
# - g = -9.81 m/s² (downward)
#
# Validation metrics:
# - Gas region (f=0): F_z = (ρ_gas - ρ_liquid) × g ≈ 9796 N/m³ (upward)
# - Interface (f=0.5): F_z ≈ 4898 N/m³ (interpolated)
# - Liquid region (f=1): F_z = 0 N/m³ (reference phase)
#
# Expected runtime: < 5 seconds
#
add_executable(test_vof_buoyancy
    test_vof_buoyancy.cu
)

set_target_properties(test_vof_buoyancy PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "70;75;80;86"
)

target_include_directories(test_vof_buoyancy PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(test_vof_buoyancy
    lbm_physics
    lbm_core
    CUDA::cudart
)

add_test(NAME VOFBuoyancyForce
    COMMAND test_vof_buoyancy
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)

set_tests_properties(VOFBuoyancyForce PROPERTIES
    TIMEOUT 10
    LABELS "validation;force;buoyancy;vof;unit_test"
)

# ========================================
# M2 Rising Bubble 2D Validation Test
# ========================================
#
# This test validates VOF interface tracking with buoyancy-driven motion.
# A 2D (quasi-2D) gas bubble rises through liquid due to density difference.
#
# Configuration:
# - Grid: 100×200×1 (tall quasi-2D domain)
# - Bubble: Circular, R=10 cells, centered at (0.5W, 0.3H)
# - Density ratio: 10:1 (liquid:gas, easier than 1000:1)
# - Surface tension: σ = 0.072 N/m
# - Gravity: g = -9.81 m/s² (in -y direction)
#
# Validation metrics:
# - Bubble rises (centroid y increases)
# - Buoyancy force magnitude matches analytical
# - Force direction is upward (positive y)
# - Mass conservation < 1%
# - Interface remains smooth
#
# Expected runtime: 5-10 minutes
#
add_executable(test_rising_bubble_2d
    vof/test_rising_bubble_2d.cu
)

set_target_properties(test_rising_bubble_2d PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "70;75;80;86"
)

target_include_directories(test_rising_bubble_2d PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(test_rising_bubble_2d
    lbm_physics
    lbm_core
    gtest
    gtest_main
    CUDA::cudart
)

add_test(NAME M2_RisingBubble2D
    COMMAND test_rising_bubble_2d
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)

set_tests_properties(M2_RisingBubble2D PROPERTIES
    TIMEOUT 600  # 10 minutes
    LABELS "validation;vof;buoyancy;rising_bubble;m2;critical"
)

# ========================================
# Rayleigh-Taylor Instability Validation Test
# ========================================
#
# This test validates VOF interface tracking for gravitational instability.
# Classic benchmark: heavy fluid over light fluid develops bubble-spike structure.
#
# Configuration:
# - Grid: 128×512×4 (8mm × 32mm, quasi-2D)
# - Heavy fluid: ρ_h = 1.225 kg/m³ (air, on top)
# - Light fluid: ρ_l = 0.169 kg/m³ (helium, below)
# - Initial: Sinusoidal perturbation y = -0.05*cos(2πx)
# - Surface tension: σ = 0.01 N/m
# - Gravity: g = 9.81 m/s²
# - Total time: 1.0 s
#
# Validation metrics:
# - Mass conservation < 1%
# - Bubble-spike structure develops
# - No NaN/Inf
# - Monotonic instability growth
#
# References:
# - Tryggvason (1988): Numerical simulations of Rayleigh-Taylor instability
# - Youngs (1984): Numerical simulation of turbulent mixing
#
# Expected runtime: 15-20 minutes (100,000 timesteps with simplified dynamics)
#
add_executable(test_rayleigh_taylor
    vof/test_rayleigh_taylor.cu
)

set_target_properties(test_rayleigh_taylor PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "70;75;80;86"
)

target_include_directories(test_rayleigh_taylor PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(test_rayleigh_taylor
    lbm_physics
    lbm_core
    gtest
    gtest_main
    CUDA::cudart
)

add_test(NAME RayleighTaylorInstability
    COMMAND test_rayleigh_taylor
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)

set_tests_properties(RayleighTaylorInstability PROPERTIES
    TIMEOUT 1800  # 30 minutes (100k timesteps with simplified dynamics)
    LABELS "validation;vof;instability;rayleigh_taylor;critical"
)

# ========================================
# Rayleigh-Taylor Debug Test
# ========================================
#
# Debug version to isolate mass conservation issues.
# Tests VOF advection components separately to identify root cause of 11.7% error.
#
add_executable(test_rayleigh_taylor_debug
    vof/test_rayleigh_taylor_debug.cu
)

set_target_properties(test_rayleigh_taylor_debug PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "70;75;80;86"
)

target_include_directories(test_rayleigh_taylor_debug PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(test_rayleigh_taylor_debug
    lbm_physics
    lbm_core
    gtest
    gtest_main
    CUDA::cudart
)

add_test(NAME RayleighTaylorDebug
    COMMAND test_rayleigh_taylor_debug
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)

set_tests_properties(RayleighTaylorDebug PROPERTIES
    TIMEOUT 1800  # 30 minutes
    LABELS "validation;vof;debug;mass_conservation"
)

# ========================================
# Rayleigh-Taylor Fixed Test
# ========================================
#
# Fixed version with proper mass conservation (<0.1% error).
# Fixes: Removed velocity limiter, reduced dt to ensure natural CFL < 0.5
#
add_executable(test_rayleigh_taylor_fixed
    vof/test_rayleigh_taylor_fixed.cu
)

set_target_properties(test_rayleigh_taylor_fixed PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "70;75;80;86"
)

target_include_directories(test_rayleigh_taylor_fixed PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(test_rayleigh_taylor_fixed
    lbm_physics
    lbm_core
    gtest
    gtest_main
    CUDA::cudart
)

add_test(NAME RayleighTaylorFixed
    COMMAND test_rayleigh_taylor_fixed
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)

set_tests_properties(RayleighTaylorFixed PROPERTIES
    TIMEOUT 3600  # 60 minutes (10× more timesteps)
    LABELS "validation;vof;instability;rayleigh_taylor;fixed;critical"
)

# ========================================
# Rayleigh-Taylor Conservative Test
# ========================================
# Uses flux-conservative VOF advection for proper mass conservation
#
add_executable(test_rayleigh_taylor_conservative
    vof/test_rayleigh_taylor_conservative.cu
)

set_target_properties(test_rayleigh_taylor_conservative PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "70;75;80;86"
)

target_include_directories(test_rayleigh_taylor_conservative PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(test_rayleigh_taylor_conservative
    gtest
    gtest_main
    CUDA::cudart
)

add_test(NAME RayleighTaylorConservative
    COMMAND test_rayleigh_taylor_conservative
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)

set_tests_properties(RayleighTaylorConservative PROPERTIES
    TIMEOUT 600
    LABELS "validation;vof;conservative;mass_conservation;critical"
)

# ========================================
# Rayleigh-Taylor LBM Test
# ========================================
# Uses full FluidLBM solver for divergence-free velocity fields
#
add_executable(test_rayleigh_taylor_lbm
    vof/test_rayleigh_taylor_lbm.cu
)

set_target_properties(test_rayleigh_taylor_lbm PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "70;75;80;86"
)

target_include_directories(test_rayleigh_taylor_lbm PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(test_rayleigh_taylor_lbm
    lbm_physics
    lbm_core
    gtest
    gtest_main
    CUDA::cudart
)

add_test(NAME RayleighTaylorLBM
    COMMAND test_rayleigh_taylor_lbm
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)

set_tests_properties(RayleighTaylorLBM PROPERTIES
    TIMEOUT 600
    LABELS "validation;vof;lbm;mass_conservation;critical"
)

# ========================================
# Rayleigh-Taylor Full Validation Test
# ========================================
# Complete validation with analytical solution comparison and VTK output
#
add_executable(test_rayleigh_taylor_validation
    vof/test_rayleigh_taylor_validation.cu
)

set_target_properties(test_rayleigh_taylor_validation PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "70;75;80;86"
)

target_include_directories(test_rayleigh_taylor_validation PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(test_rayleigh_taylor_validation
    lbm_physics
    lbm_core
    gtest
    gtest_main
    CUDA::cudart
)

add_test(NAME RayleighTaylorValidation
    COMMAND test_rayleigh_taylor_validation
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)

# ========================================
# Rayleigh-Taylor Force Diagnostic Test
# ========================================
# Focused diagnostic for force-velocity coupling in RT instability
#
add_executable(test_rayleigh_taylor_force_diagnostic
    vof/test_rayleigh_taylor_force_diagnostic.cu
)

set_target_properties(test_rayleigh_taylor_force_diagnostic PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "70;75;80;86"
)

target_include_directories(test_rayleigh_taylor_force_diagnostic PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(test_rayleigh_taylor_force_diagnostic
    lbm_physics
    lbm_core
    gtest
    gtest_main
    CUDA::cudart
)

add_test(NAME RayleighTaylorForceDiagnostic
    COMMAND test_rayleigh_taylor_force_diagnostic
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)

# ========================================
# Rayleigh-Taylor Mushroom Structures Test
# ========================================
# Classic RT instability benchmark producing mushroom-shaped structures
# Domain: 1m × 4m × 0.1m, rho_heavy=3.0, rho_light=1.0
# Time range: 0 to 1.0s with keyframes at 0.2s, 0.7s, 0.8s, 0.9s, 1.0s
# Tracks h₁ (bubble) and h₂ (spike) positions
#
add_executable(test_rayleigh_taylor_mushroom
    vof/test_rayleigh_taylor_mushroom.cu
)

set_target_properties(test_rayleigh_taylor_mushroom PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "70;75;80;86"
)

target_include_directories(test_rayleigh_taylor_mushroom PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(test_rayleigh_taylor_mushroom
    lbm_physics
    lbm_core
    gtest
    gtest_main
    CUDA::cudart
)

add_test(NAME RayleighTaylorMushroom
    COMMAND test_rayleigh_taylor_mushroom
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)

# ========================================
# Rayleigh-Taylor Mass Conservation Diagnostic
# ========================================
# Diagnostic test to identify sources of mass loss in RT simulation
# Tracks: zonal masses, VOF statistics, boundary fluxes, interface diffusion
#
add_executable(test_rt_mass_conservation_diagnostic
    vof/test_rt_mass_conservation_diagnostic.cu
)

set_target_properties(test_rt_mass_conservation_diagnostic PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "70;75;80;86"
)

target_include_directories(test_rt_mass_conservation_diagnostic PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(test_rt_mass_conservation_diagnostic
    lbm_physics
    lbm_core
    gtest
    gtest_main
    CUDA::cudart
)

add_test(NAME RTMassConservationDiagnostic
    COMMAND test_rt_mass_conservation_diagnostic
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)

# ========================================
# Rayleigh-Taylor Comprehensive Validation
# ========================================
# Complete test suite for RT instability validation against reference image
# Tests: boundary conditions, interface shape, height evolution, mass conservation, symmetry
# Reference: /home/yzk/LBMProject/1118/屏幕截图 2025-12-18 201835.png
#
add_executable(test_rayleigh_taylor_comprehensive
    vof/test_rayleigh_taylor_comprehensive.cu
)

set_target_properties(test_rayleigh_taylor_comprehensive PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "70;75;80;86"
)

target_include_directories(test_rayleigh_taylor_comprehensive PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(test_rayleigh_taylor_comprehensive
    lbm_physics
    lbm_core
    gtest
    gtest_main
    CUDA::cudart
)

add_test(NAME RayleighTaylorComprehensive
    COMMAND test_rayleigh_taylor_comprehensive
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)

set_tests_properties(RayleighTaylorComprehensive PROPERTIES
    TIMEOUT 3600
    LABELS "validation;vof;rayleigh_taylor;comprehensive;critical"
)

# ========================================
# Force Conversion Units Test
# ========================================
# Dimensional analysis to verify correct force conversion formula
#
add_executable(test_force_conversion_units
    vof/test_force_conversion_units.cu
)

set_target_properties(test_force_conversion_units PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "70;75;80;86"
)

target_include_directories(test_force_conversion_units PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(test_force_conversion_units
    gtest
    gtest_main
    CUDA::cudart
)

add_test(NAME ForceConversionUnits
    COMMAND test_force_conversion_units
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)

set_tests_properties(RayleighTaylorValidation PROPERTIES
    TIMEOUT 1800
    LABELS "validation;vof;analytical;rayleigh_taylor;critical"
)

#
# ============================================================================
# Phase Inversion Test
# ============================================================================
# VOF benchmark: Oil block rising in water due to buoyancy
# Tests density-ratio ~1.1 interface tracking
#
add_executable(test_phase_inversion
    vof/test_phase_inversion.cu
)

set_target_properties(test_phase_inversion PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "70;75;80;86"
)

target_include_directories(test_phase_inversion PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(test_phase_inversion
    lbm_physics
    lbm_core
    gtest
    gtest_main
    CUDA::cudart
)

add_test(NAME PhaseInversion
    COMMAND test_phase_inversion
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)

set_tests_properties(PhaseInversion PROPERTIES
    TIMEOUT 600  # 10 minutes
    LABELS "validation;vof;buoyancy;phase_inversion"
)

# ========================================
# VOF Mass Conservation TVD Test
# ========================================
# Comprehensive mass conservation test suite for VOF advection
# Designed to validate improvements from TVD scheme implementation
#
# Test cases:
# 1. Translation: Uniform velocity, circular interface
# 2. Rotation: Solid body rotation (360°)
# 3. Shear: Linear velocity gradient
# 4. High CFL: Stress test at CFL=0.4
#
# Success criteria:
# - First-order upwind: Mass error 5-20% (baseline)
# - TVD schemes: Mass error <1% (target)
#
add_executable(test_vof_mass_conservation_tvd
    vof/test_vof_mass_conservation_tvd.cu
)

set_target_properties(test_vof_mass_conservation_tvd PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "70;75;80;86"
)

target_include_directories(test_vof_mass_conservation_tvd PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(test_vof_mass_conservation_tvd
    lbm_physics
    lbm_core
    gtest
    gtest_main
    CUDA::cudart
)

add_test(NAME VOFMassConservationTVD
    COMMAND test_vof_mass_conservation_tvd
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)

set_tests_properties(VOFMassConservationTVD PROPERTIES
    TIMEOUT 600  # 10 minutes (4 test cases)
    LABELS "validation;vof;mass_conservation;tvd;advection"
)
