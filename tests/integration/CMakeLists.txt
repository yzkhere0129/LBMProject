# Integration Tests CMakeLists.txt

# Enable CUDA
enable_language(CUDA)

# Find required packages
find_package(CUDAToolkit REQUIRED)

# Add Thrust include directory
include_directories(${CUDAToolkit_INCLUDE_DIRS})

# ========================================
# WEEK 1 FIX INTEGRATION TESTS
# ========================================

# Test 1.2: Evaporation Energy Balance Integration Test
add_executable(test_evaporation_energy_balance
    test_evaporation_energy_balance.cu
)

set_target_properties(test_evaporation_energy_balance PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "86"
)

target_include_directories(test_evaporation_energy_balance PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(test_evaporation_energy_balance
    lbm_physics
    lbm_core
    gtest
    gtest_main
    CUDA::cudart
)

add_test(NAME EvaporationEnergyBalance
    COMMAND test_evaporation_energy_balance
)

set_tests_properties(EvaporationEnergyBalance PROPERTIES
    TIMEOUT 120
    LABELS "integration;evaporation;energy;week1;critical"
)

# Test 2.2: Substrate Temperature Reduction Integration Test
add_executable(test_substrate_temperature_reduction
    test_substrate_temperature_reduction.cu
)

set_target_properties(test_substrate_temperature_reduction PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "86"
)

target_include_directories(test_substrate_temperature_reduction PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(test_substrate_temperature_reduction
    lbm_physics
    lbm_core
    gtest
    gtest_main
    CUDA::cudart
)

add_test(NAME SubstrateTemperatureReduction
    COMMAND test_substrate_temperature_reduction
)

set_tests_properties(SubstrateTemperatureReduction PROPERTIES
    TIMEOUT 300
    LABELS "integration;substrate;temperature;week1;critical"
)

# ========================================
# EXISTING INTEGRATION TESTS
# ========================================

# Integration test executable for laser heating (simplified version)
add_executable(test_laser_heating_simplified
    test_laser_heating_simplified.cu
)

# Full integration test (when all features are ready)
# add_executable(test_laser_heating
#     test_laser_heating.cu
# )

# Set CUDA properties
set_target_properties(test_laser_heating_simplified PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "86"
)

# Include directories
target_include_directories(test_laser_heating_simplified PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

# Link libraries
target_link_libraries(test_laser_heating_simplified
    lbm_physics
    lbm_core
    gtest
    gtest_main
    CUDA::cudart
    CUDA::cuda_driver
)

# Add test to CTest
add_test(NAME LaserHeatingIntegrationSimplified
    COMMAND test_laser_heating_simplified
)

# Set test properties
set_tests_properties(LaserHeatingIntegrationSimplified PROPERTIES
    TIMEOUT 120  # 2 minutes timeout
    LABELS "integration;physics"
)

# Laser melting integration test with phase change
add_executable(test_laser_melting
    test_laser_melting.cu
)

set_target_properties(test_laser_melting PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "86"
)

target_include_directories(test_laser_melting PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(test_laser_melting
    lbm_physics
    lbm_core
    lbm_io
    gtest
    gtest_main
    CUDA::cudart
    CUDA::cuda_driver
)

add_test(NAME LaserMeltingIntegration
    COMMAND test_laser_melting
)

set_tests_properties(LaserMeltingIntegration PROPERTIES
    TIMEOUT 180  # 3 minutes timeout (more time for phase change)
    LABELS "integration;physics;phase_change"
)

# Stefan problem validation test (1D phase change)
add_executable(test_stefan_1d
    test_stefan_1d.cu
)

set_target_properties(test_stefan_1d PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "86"
)

target_include_directories(test_stefan_1d PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(test_stefan_1d
    lbm_physics
    lbm_core
    gtest
    gtest_main
    CUDA::cudart
    CUDA::cuda_driver
)

add_test(NAME StefanProblemValidation
    COMMAND test_stefan_1d
)

set_tests_properties(StefanProblemValidation PROPERTIES
    TIMEOUT 300  # 5 minutes timeout (10000 steps)
    LABELS "integration;physics;phase_change;validation"
)

# Poiseuille flow validation test using FluidLBM class
add_executable(test_poiseuille_flow_fluidlbm
    test_poiseuille_flow_fluidlbm.cu
)

set_target_properties(test_poiseuille_flow_fluidlbm PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "86"
)

target_include_directories(test_poiseuille_flow_fluidlbm PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(test_poiseuille_flow_fluidlbm
    lbm_physics
    lbm_core
    gtest
    gtest_main
    CUDA::cudart
    CUDA::cuda_driver
)

add_test(NAME PoiseuilleFlowFluidLBM
    COMMAND test_poiseuille_flow_fluidlbm
)

set_tests_properties(PoiseuilleFlowFluidLBM PROPERTIES
    TIMEOUT 180  # 3 minutes timeout (runs longer for convergence)
    LABELS "integration;fluid;validation;fluidlbm"
)

# Thermal-fluid coupling integration test (natural convection)
add_executable(test_thermal_fluid_coupling
    test_thermal_fluid_coupling.cu
)

set_target_properties(test_thermal_fluid_coupling PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "86"
)

target_include_directories(test_thermal_fluid_coupling PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(test_thermal_fluid_coupling
    lbm_physics
    lbm_core
    gtest
    gtest_main
    CUDA::cudart
    CUDA::cuda_driver
)

add_test(NAME ThermalFluidCouplingIntegration
    COMMAND test_thermal_fluid_coupling
)

set_tests_properties(ThermalFluidCouplingIntegration PROPERTIES
    TIMEOUT 300  # 5 minutes timeout (10000 steps simulation)
    LABELS "integration;coupling;thermal;fluid;convection"
)

# Phase 5: Laser melting with melt pool convection
add_executable(test_laser_melting_convection
    test_laser_melting_convection.cu
)

set_target_properties(test_laser_melting_convection PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "86"
)

target_include_directories(test_laser_melting_convection PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(test_laser_melting_convection
    lbm_physics
    lbm_core
    gtest
    gtest_main
    CUDA::cudart
    CUDA::cuda_driver
)

add_test(NAME LaserMeltingConvectionIntegration
    COMMAND test_laser_melting_convection
)

set_tests_properties(LaserMeltingConvectionIntegration PROPERTIES
    TIMEOUT 300  # 5 minutes timeout
    LABELS "integration;coupling;laser;melting;convection;phase5"
)

# ========================================
# RECOIL PRESSURE INTEGRATION TESTS
# ========================================

# Recoil pressure surface depression test
add_executable(test_recoil_surface_depression
    test_recoil_surface_depression.cu
)

set_target_properties(test_recoil_surface_depression PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "86"
)

target_include_directories(test_recoil_surface_depression PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(test_recoil_surface_depression
    lbm_physics
    lbm_core
    gtest
    gtest_main
    CUDA::cudart
)

add_test(NAME RecoilSurfaceDepressionIntegration
    COMMAND test_recoil_surface_depression
)

set_tests_properties(RecoilSurfaceDepressionIntegration PROPERTIES
    TIMEOUT 180
    LABELS "integration;recoil;keyhole;phase6"
)

# ========================================
# MULTIPHYSICS INTEGRATION TESTS
# ========================================

# Add multiphysics test subdirectory
add_subdirectory(multiphysics)

# Optional: Add more integration tests here as they are created
# add_executable(test_multiphase_flow ...)
# add_executable(test_free_surface ...)
# add_executable(test_full_am_process ...)