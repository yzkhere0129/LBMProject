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
    CUDA::cudart
)

add_test(NAME GridIndependenceValidation
    COMMAND test_grid_convergence
)

set_tests_properties(GridIndependenceValidation PROPERTIES
    TIMEOUT 1800  # 30 minutes (3 simulations)
    LABELS "validation;convergence;grid;week2;critical"
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
    CUDA::cudart
)

add_test(NAME TimestepConvergenceValidation
    COMMAND test_timestep_convergence
)

set_tests_properties(TimestepConvergenceValidation PROPERTIES
    TIMEOUT 1800  # 30 minutes (3 simulations)
    LABELS "validation;convergence;timestep;week2;critical"
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
# Future Validation Tests
# ========================================
#
# Planned additions:
# - test_laplace_pressure.cu (surface tension validation)
# - test_melt_pool_dimensions.cu (LPBF melt pool geometry)
# - test_recoil_pressure.cu (vapor pressure validation)
# - test_keyhole_formation.cu (deep penetration welding)

# ========================================
# MULTIPHYSICS VALIDATION TESTS
# ========================================

# Add multiphysics validation subdirectory
add_subdirectory(multiphysics)
