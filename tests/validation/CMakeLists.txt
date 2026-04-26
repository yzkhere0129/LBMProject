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

add_executable(test_marangoni_velocity
    test_marangoni_velocity.cu
)

set_target_properties(test_marangoni_velocity PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "86"
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

add_executable(test_realistic_lpbf_initial_conditions
    test_realistic_lpbf_initial_conditions.cu
)

set_target_properties(test_realistic_lpbf_initial_conditions PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "86"
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

add_executable(test_divergence_free
    test_divergence_free.cu
)

set_target_properties(test_divergence_free PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "86"
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

add_executable(test_phase_change_robustness
    test_phase_change_robustness.cu
)

set_target_properties(test_phase_change_robustness PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "86"
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

add_executable(test_laser_shutoff_configurable
    test_laser_shutoff_configurable.cu
)

set_target_properties(test_laser_shutoff_configurable PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "86"
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

# Test 1.1: Hertz-Knudsen Formula Unit Test
add_executable(test_evaporation_hertz_knudsen
    test_evaporation_hertz_knudsen.cu
)

set_target_properties(test_evaporation_hertz_knudsen PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "86"
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
    CUDA_ARCHITECTURES "86"
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

# Benchmark 1: Pure Heat Conduction (1D)
add_executable(test_pure_conduction
    test_pure_conduction.cu
)

set_target_properties(test_pure_conduction PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "86"
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
add_executable(test_stefan_problem
    test_stefan_problem.cu
)

set_target_properties(test_stefan_problem PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "86"
)

target_include_directories(test_stefan_problem PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(test_stefan_problem
    lbm_thermal_only
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
add_executable(test_evaporation_rate
    test_evaporation_rate.cu
)

set_target_properties(test_evaporation_rate PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "86"
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

# Test 1: Grid Independence Validator
add_executable(test_grid_convergence
    test_grid_convergence.cu
)

set_target_properties(test_grid_convergence PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "86"
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
    TIMEOUT 600
    LABELS "validation;convergence;grid;analytical;critical"
)

# Test 2: Timestep Convergence Validator
add_executable(test_timestep_convergence
    test_timestep_convergence.cu
)

set_target_properties(test_timestep_convergence PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "86"
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
    TIMEOUT 600
    LABELS "validation;convergence;timestep;analytical;critical"
)

# Test 3: Energy Conservation Validator
add_executable(test_energy_conservation_timestep
    test_energy_conservation_timestep.cu
)

set_target_properties(test_energy_conservation_timestep PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "86"
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
    TIMEOUT 1800
    LABELS "validation;energy;conservation;week2;critical"
)

# Test 4: Regression Test (50W Baseline)
add_executable(test_regression_50W
    test_regression_50W.cu
)

set_target_properties(test_regression_50W PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "86"
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
    TIMEOUT 600
    LABELS "validation;regression;baseline;week2;critical"
)

# Test 5: CFL Stability Check
add_executable(test_cfl_stability
    test_cfl_stability.cu
)

set_target_properties(test_cfl_stability PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "86"
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
    TIMEOUT 60
    LABELS "validation;stability;cfl;week2;critical"
)

# Test 6: Configuration Parser Test
add_executable(test_config_parser
    test_config_parser.cu
)

set_target_properties(test_config_parser PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "86"
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
    TIMEOUT 30
    LABELS "validation;config;parser;week2;critical"
)

# ========================================
# SENIOR'S THESIS VALIDATION CASES
# ========================================

# Case 1: 3D Heat Diffusion (Gaussian pulse)
add_executable(test_3d_heat_diffusion_senior
    test_3d_heat_diffusion_senior.cu
)

set_target_properties(test_3d_heat_diffusion_senior PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "86"
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
    TIMEOUT 600
    LABELS "validation;senior;thermal;3d;gaussian;critical"
)

# Case 1 FIXED: 3D Heat Diffusion with aligned timesteps
add_executable(test_3d_heat_diffusion_fixed
    test_3d_heat_diffusion_fixed.cu
)

set_target_properties(test_3d_heat_diffusion_fixed PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "86"
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
    TIMEOUT 600
    LABELS "validation;senior;thermal;3d;gaussian;fixed;staircase"
)

# ========================================
# CASE 5: LASER MELTING VALIDATION (SENIOR'S CONFIGURATION)
# ========================================

add_executable(test_laser_melting_senior
    test_laser_melting_senior.cu
)

set_target_properties(test_laser_melting_senior PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "86"
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
    TIMEOUT 1800
    LABELS "validation;senior;laser;melting;multiphysics;critical"
)

# ========================================
# LASER MELTING VALIDATION: IRON/STEEL
# ========================================

add_executable(test_laser_melting_iron
    test_laser_melting_iron.cu
)

set_target_properties(test_laser_melting_iron PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "86"
)

target_include_directories(test_laser_melting_iron PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(test_laser_melting_iron
    lbm_physics
    lbm_core
    lbm_io
    gtest
    gtest_main
    CUDA::cudart
    CUDA::cuda_driver
)

add_test(NAME LaserMeltingIron
    COMMAND test_laser_melting_iron
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)

set_tests_properties(LaserMeltingIron PROPERTIES
    TIMEOUT 600
    LABELS "validation;laser_melting;phase_2;steel;openfoam_reference"
)

# ========================================
# ENERGY BUDGET CONSERVATION TEST
# ========================================

add_executable(test_energy_budget
    test_energy_budget.cu
)

set_target_properties(test_energy_budget PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "86"
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
    TIMEOUT 300
    LABELS "validation;energy;conservation;budget;critical"
)

# ========================================
# THERMAL SOURCE ENERGY CONSERVATION TEST
# ========================================

add_executable(thermal_source_energy_test
    thermal_source_energy_test.cu
)

set_target_properties(thermal_source_energy_test PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "86"
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
    TIMEOUT 60
    LABELS "validation;thermal;source;energy;unit;critical"
)

# ========================================
# WALBERLA THERMAL MATCH TEST
# ========================================

add_executable(test_thermal_walberla_match
    test_thermal_walberla_match.cu
)

set_target_properties(test_thermal_walberla_match PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "86"
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
    TIMEOUT 600
    LABELS "validation;thermal;walberla;match;lbm;fd;critical"
)

# ========================================
# FINITE DIFFERENCE REFERENCE TEST
# ========================================

add_executable(test_fd_thermal_reference
    test_fd_thermal_reference.cu
)

set_target_properties(test_fd_thermal_reference PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "86"
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
    TIMEOUT 600
    LABELS "validation;thermal;fd;reference;walberla"
)

# ========================================
# Gaussian Diffusion Analytical Validation
# ========================================

add_executable(test_gaussian_diffusion
    test_gaussian_diffusion.cu
)

set_target_properties(test_gaussian_diffusion PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "86"
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
    TIMEOUT 300
    LABELS "validation;thermal;analytical;gaussian;diffusion;critical"
)

# ========================================
# ENERGY CONSERVATION VALIDATION TEST
# ========================================

add_executable(test_energy_conservation
    test_energy_conservation.cu
)

set_target_properties(test_energy_conservation PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "86"
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
    TIMEOUT 300
    LABELS "validation;energy;conservation;laser;thermal;critical"
)

# ========================================
# PHASE 1 FLUID VALIDATION SUITE
# ========================================

# Test 1: Lid-Driven Cavity Re=100
add_executable(test_lid_driven_cavity_re100
    test_lid_driven_cavity_re100.cu
)

set_target_properties(test_lid_driven_cavity_re100 PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "86"
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
    TIMEOUT 600
    LABELS "validation;fluid;benchmark;critical;phase1"
)

# Test 2: Lid-Driven Cavity Re=400
add_executable(test_lid_driven_cavity_re400
    test_lid_driven_cavity_re400.cu
)

set_target_properties(test_lid_driven_cavity_re400 PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "86"
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
    TIMEOUT 900
    LABELS "validation;fluid;benchmark;phase1"
)

# Test 3: Fluid Grid Convergence (Poiseuille Flow)
add_executable(test_fluid_grid_convergence
    test_fluid_grid_convergence.cu
)

set_target_properties(test_fluid_grid_convergence PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "86"
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
    TIMEOUT 600
    LABELS "validation;fluid;convergence;analytical;phase1"
)

# Test 4: Natural Convection (de Vahl Davis 1983)
add_executable(test_natural_convection
    test_natural_convection.cu
)

set_target_properties(test_natural_convection PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "86"
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
    TIMEOUT 900
    LABELS "validation;thermal;fluid;buoyancy;benchmark;critical;phase2"
)

add_test(NAME NaturalConvectionRa1e3
    COMMAND test_natural_convection --gtest_filter=NaturalConvectionTest.RayleighNumber1e3
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)

set_tests_properties(NaturalConvectionRa1e3 PROPERTIES
    TIMEOUT 600
    LABELS "validation;thermal;fluid;buoyancy;benchmark;phase2"
)

# ========================================
# Test 5: Couette-Poiseuille Flow (canonical BGK)
# ========================================

add_executable(test_couette_poiseuille
    test_couette_poiseuille.cu
)

set_target_properties(test_couette_poiseuille PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "86"
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
    TIMEOUT 600
    LABELS "validation;fluid;benchmark;analytical;phase1;critical"
)

# Test 6: TRT Combined Couette-Poiseuille Flow (canonical TRT)
add_executable(test_trt_couette_poiseuille
    test_trt_couette_poiseuille.cu
)

set_target_properties(test_trt_couette_poiseuille PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "86"
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
    TIMEOUT 600
    LABELS "validation;fluid;couette_poiseuille;trt;analytical;critical"
)

# Test 6b: TRT-degenerate (ω⁻ = ω⁺) ≡ BGK anchor test
add_executable(test_trt_degenerate_to_bgk
    test_trt_degenerate_to_bgk.cu
)

set_target_properties(test_trt_degenerate_to_bgk PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

target_include_directories(test_trt_degenerate_to_bgk PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/tests
)

target_link_libraries(test_trt_degenerate_to_bgk
    lbm_physics
    lbm_core
    lbm_io
    gtest
    gtest_main
    CUDA::cudart
    CUDA::cuda_driver
)

add_test(NAME TRTDegenerateToBGK
    COMMAND test_trt_degenerate_to_bgk
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)

set_tests_properties(TRTDegenerateToBGK PROPERTIES
    TIMEOUT 60
    LABELS "validation;fluid;trt;bgk;anchor;critical"
)

# ========================================
# VOF VALIDATION TESTS
# ========================================

# Oscillating Droplet
add_executable(test_oscillating_droplet
    test_oscillating_droplet.cu
)

set_target_properties(test_oscillating_droplet PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "86"
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
    TIMEOUT 1800
    LABELS "validation;vof;surface_tension;oscillating_droplet;critical"
)

# VOF Buoyancy Force Test
add_executable(test_vof_buoyancy
    test_vof_buoyancy.cu
)

set_target_properties(test_vof_buoyancy PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "86"
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

# M2 Rising Bubble 2D Validation Test
add_executable(test_rising_bubble_2d
    vof/test_rising_bubble_2d.cu
)

set_target_properties(test_rising_bubble_2d PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "86"
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
    TIMEOUT 600
    LABELS "validation;vof;buoyancy;rising_bubble;m2;critical"
)

# Rayleigh-Taylor Gerris EXACT Parameters (canonical benchmark)
add_executable(test_rayleigh_taylor_gerris_exact
    vof/test_rayleigh_taylor_gerris_exact.cu
)

set_target_properties(test_rayleigh_taylor_gerris_exact PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "86"
)

target_include_directories(test_rayleigh_taylor_gerris_exact PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(test_rayleigh_taylor_gerris_exact
    lbm_physics
    lbm_core
    gtest
    gtest_main
    CUDA::cudart
)

add_test(NAME RayleighTaylorGerrisExact
    COMMAND test_rayleigh_taylor_gerris_exact
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)

# Rayleigh-Taylor Full Validation
add_executable(test_rayleigh_taylor_validation
    vof/test_rayleigh_taylor_validation.cu
)

set_target_properties(test_rayleigh_taylor_validation PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "86"
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

set_tests_properties(RayleighTaylorValidation PROPERTIES
    TIMEOUT 1800
    LABELS "validation;vof;analytical;rayleigh_taylor;critical"
)

# TRT Validation for Re=3000 RT Instability
add_executable(test_rt_trt_validation
    vof/test_rt_trt_validation.cu
)

set_target_properties(test_rt_trt_validation PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "86"
)

target_include_directories(test_rt_trt_validation PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(test_rt_trt_validation
    lbm_physics
    lbm_core
    gtest
    gtest_main
    CUDA::cudart
)

add_test(NAME RTTRTValidation
    COMMAND test_rt_trt_validation
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)

set_tests_properties(RTTRTValidation PROPERTIES
    TIMEOUT 600
    LABELS "validation;trt;rayleigh_taylor;re3000"
)

# Phase Inversion Test
add_executable(test_phase_inversion
    vof/test_phase_inversion.cu
)

set_target_properties(test_phase_inversion PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "86"
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
    TIMEOUT 600
    LABELS "validation;vof;buoyancy;phase_inversion"
)

# VOF Mass Conservation TVD Test
add_executable(test_vof_mass_conservation_tvd
    vof/test_vof_mass_conservation_tvd.cu
)

set_target_properties(test_vof_mass_conservation_tvd PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "86"
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
    TIMEOUT 600
    LABELS "validation;vof;mass_conservation;tvd;advection"
)

# VOF Mass Correction Test
add_executable(test_vof_mass_correction
    vof/test_vof_mass_correction.cu
)

set_target_properties(test_vof_mass_correction PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "86"
)

target_include_directories(test_vof_mass_correction PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(test_vof_mass_correction
    lbm_physics
    lbm_core
    CUDA::cudart
)

# ========================================
# MULTIPHYSICS VALIDATION TESTS
# ========================================

# Add multiphysics validation subdirectory
add_subdirectory(multiphysics)

# ========================================
# MARANGONI VELOCITY SCALE TEST
# ========================================

add_executable(test_marangoni_velocity_scale
    test_marangoni_velocity_scale.cu
)

set_target_properties(test_marangoni_velocity_scale PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "86"
)

target_include_directories(test_marangoni_velocity_scale PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(test_marangoni_velocity_scale
    lbm_physics
    lbm_core
    gtest
    gtest_main
    CUDA::cudart
    CUDA::cuda_driver
)

add_test(NAME MarangoniVelocityScale
    COMMAND test_marangoni_velocity_scale
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)

set_tests_properties(MarangoniVelocityScale PROPERTIES
    TIMEOUT 600
    LABELS "validation;marangoni;velocity;scale"
)

# ========================================
# TAYLOR-GREEN 2D TEST
# ========================================

add_executable(test_taylor_green_2d
    test_taylor_green_2d.cu
)

set_target_properties(test_taylor_green_2d PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "86"
)

target_include_directories(test_taylor_green_2d PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(test_taylor_green_2d
    lbm_physics
    lbm_core
    gtest
    gtest_main
    CUDA::cudart
    CUDA::cuda_driver
)

add_test(NAME TaylorGreen2D
    COMMAND test_taylor_green_2d
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)

set_tests_properties(TaylorGreen2D PROPERTIES
    TIMEOUT 600
    LABELS "validation;fluid;taylor_green"
)

# ========================================
# THERMAL TIMESTEP CONVERGENCE TEST
# ========================================

add_executable(test_thermal_timestep_convergence
    test_thermal_timestep_convergence.cu
)

set_target_properties(test_thermal_timestep_convergence PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "86"
)

target_include_directories(test_thermal_timestep_convergence PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(test_thermal_timestep_convergence
    lbm_physics
    lbm_core
    gtest
    gtest_main
    CUDA::cudart
    CUDA::cuda_driver
)

add_test(NAME ThermalTimestepConvergence
    COMMAND test_thermal_timestep_convergence
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)

set_tests_properties(ThermalTimestepConvergence PROPERTIES
    TIMEOUT 600
    LABELS "validation;thermal;timestep;convergence"
)

# ========================================
# REMAINING VALIDATION TESTS
# ========================================

# Newton-Bisection Fallback
add_executable(test_newton_bisection_fallback
    test_newton_bisection_fallback.cu
)

set_target_properties(test_newton_bisection_fallback PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "86"
)

target_include_directories(test_newton_bisection_fallback PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(test_newton_bisection_fallback
    lbm_physics
    lbm_core
    gtest
    gtest_main
    CUDA::cudart
    CUDA::cuda_driver
)

add_test(NAME NewtonBisectionFallback
    COMMAND test_newton_bisection_fallback
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)

set_tests_properties(NewtonBisectionFallback PROPERTIES
    TIMEOUT 120
    LABELS "validation;phase_change;numerical"
)

# Powder Bed Generation
add_executable(test_powder_bed_generation
    test_powder_bed_generation.cu
)

set_target_properties(test_powder_bed_generation PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "86"
)

target_include_directories(test_powder_bed_generation PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(test_powder_bed_generation
    lbm_physics
    lbm_core
    gtest
    gtest_main
    CUDA::cudart
    CUDA::cuda_driver
)

add_test(NAME PowderBedGeneration
    COMMAND test_powder_bed_generation
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)

set_tests_properties(PowderBedGeneration PROPERTIES
    TIMEOUT 120
    LABELS "validation;powder_bed;lpbf"
)

# Khairallah Melt Pool
add_executable(test_khairallah_melt_pool
    test_khairallah_melt_pool.cu
)

set_target_properties(test_khairallah_melt_pool PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "86"
)

target_include_directories(test_khairallah_melt_pool PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(test_khairallah_melt_pool
    lbm_physics
    lbm_core
    lbm_io
    gtest
    gtest_main
    CUDA::cudart
    CUDA::cuda_driver
)

add_test(NAME KhairallahMeltPool
    COMMAND test_khairallah_melt_pool
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)

set_tests_properties(KhairallahMeltPool PROPERTIES
    TIMEOUT 2400
    LABELS "validation;laser_melting;melt_pool;khairallah"
)

# TVD VOF Advection
add_executable(test_tvd_vof_advection
    test_tvd_vof_advection.cu
)

set_target_properties(test_tvd_vof_advection PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "86"
)

target_include_directories(test_tvd_vof_advection PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(test_tvd_vof_advection
    lbm_physics
    lbm_core
    gtest
    gtest_main
    CUDA::cudart
)

add_test(NAME TVDVOFAdvection
    COMMAND test_tvd_vof_advection
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)

set_tests_properties(TVDVOFAdvection PROPERTIES
    TIMEOUT 300
    LABELS "validation;vof;tvd;advection"
)

# ========================================
# COMPREHENSIVE PHASE CHANGE VALIDATION
# ========================================

add_executable(test_phase_change_comprehensive
    test_phase_change_comprehensive.cu
)

set_target_properties(test_phase_change_comprehensive PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "86"
)

target_include_directories(test_phase_change_comprehensive PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(test_phase_change_comprehensive
    lbm_physics
    lbm_core
    gtest
    gtest_main
    CUDA::cudart
    CUDA::cuda_driver
)

add_test(NAME PhaseChangeComprehensive
    COMMAND test_phase_change_comprehensive
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)

set_tests_properties(PhaseChangeComprehensive PROPERTIES
    TIMEOUT 300
    LABELS "validation;phase_change;enthalpy;newton;energy;comprehensive;critical"
)

# ========================================
# ENTHALPY DIAGNOSTIC ACCEPTANCE TEST
# ========================================

add_executable(test_enthalpy_diag_validation
    test_enthalpy_diag.cu
)

set_target_properties(test_enthalpy_diag_validation PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "86"
)

target_include_directories(test_enthalpy_diag_validation PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(test_enthalpy_diag_validation
    lbm_physics
    lbm_core
    gtest
    gtest_main
    CUDA::cudart
    CUDA::cuda_driver
)

add_test(NAME EnthalpyDiagAcceptance
    COMMAND test_enthalpy_diag_validation
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)

set_tests_properties(EnthalpyDiagAcceptance PROPERTIES
    TIMEOUT 600
    LABELS "validation;enthalpy;diagnostic;phase_change;energy_balance;acceptance;critical"
)
