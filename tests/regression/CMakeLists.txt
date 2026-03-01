# Regression Tests CMakeLists.txt
#
# These tests ensure that known bugs remain fixed

# Enable CUDA
enable_language(CUDA)

# Find required packages
find_package(CUDAToolkit REQUIRED)

# ========================================
# WEEK 1 FIX REGRESSION TESTS
# ========================================

# Test 2.3: Substrate BC Long-Run Stability Stress Test
add_executable(test_substrate_bc_stability
    test_substrate_bc_stability.cu
)

set_target_properties(test_substrate_bc_stability PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "86"
)

target_include_directories(test_substrate_bc_stability PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(test_substrate_bc_stability
    lbm_physics
    lbm_core
    gtest
    gtest_main
    CUDA::cudart
)

add_test(NAME SubstrateBCStability
    COMMAND test_substrate_bc_stability
)

set_tests_properties(SubstrateBCStability PROPERTIES
    TIMEOUT 600  # 10 minutes for 100k step test
    LABELS "regression;substrate;stability;week1;critical"
)

# ========================================
# CRITICAL BUG FIX REGRESSION TESTS (2026-01-10)
# ========================================
#
# These tests validate three critical bug fixes from this session:
# 1. VOF advection bulk cell bug (vof_solver.cu:419-421)
# 2. Timestep convergence omega clamping (test_timestep_convergence.cu:279-284)
# 3. Material database cp_liquid value (test_materials.cu:48)

# Test 1: VOF Advection Bulk Cell Bug Fix
add_executable(test_vof_advection_bulk_cells
    test_vof_advection_bulk_cells.cu
)

set_target_properties(test_vof_advection_bulk_cells PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "86"
)

target_include_directories(test_vof_advection_bulk_cells PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(test_vof_advection_bulk_cells
    lbm_physics
    lbm_core
    gtest
    gtest_main
    CUDA::cudart
)

add_test(NAME VOFAdvectionBulkCellsRegression
    COMMAND test_vof_advection_bulk_cells
)

set_tests_properties(VOFAdvectionBulkCellsRegression PROPERTIES
    TIMEOUT 300  # 5 minutes
    LABELS "regression;vof;advection;bulk_cells;critical;2026-01-10"
)

# Test 2: Timestep Omega Clamping Bug Fix
add_executable(test_timestep_omega_clamping
    test_timestep_omega_clamping.cu
)

set_target_properties(test_timestep_omega_clamping PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "86"
)

target_include_directories(test_timestep_omega_clamping PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(test_timestep_omega_clamping
    gtest
    gtest_main
    CUDA::cudart
)

add_test(NAME TimestepOmegaClampingRegression
    COMMAND test_timestep_omega_clamping
)

set_tests_properties(TimestepOmegaClampingRegression PROPERTIES
    TIMEOUT 60  # 1 minute (analytical tests)
    LABELS "regression;timestep;omega;convergence;critical;2026-01-10"
)

# Test 3: Material Database Values Bug Fix
add_executable(test_material_database_values
    materials/test_material_database_values.cu
)

set_target_properties(test_material_database_values PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "86"
)

target_include_directories(test_material_database_values PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(test_material_database_values
    lbm_physics
    lbm_core
    gtest
    gtest_main
    CUDA::cudart
)

add_test(NAME MaterialDatabaseValuesRegression
    COMMAND test_material_database_values
)

set_tests_properties(MaterialDatabaseValuesRegression PROPERTIES
    TIMEOUT 120  # 2 minutes
    LABELS "regression;materials;database;walberla;critical;2026-01-10"
)

# ========================================
# EXISTING REGRESSION TESTS
# ========================================

# Add subdirectory for stability regression tests
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/stability)
    add_subdirectory(stability)
endif()

# Realistic temperature regression test
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/test_realistic_temperature.cu)
    add_executable(test_realistic_temperature
        test_realistic_temperature.cu
    )

    set_target_properties(test_realistic_temperature PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES "86"
    )

    target_include_directories(test_realistic_temperature PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/src
    )

    target_link_libraries(test_realistic_temperature
        lbm_physics
        lbm_core
        gtest
        gtest_main
        CUDA::cudart
    )

    add_test(NAME RealisticTemperatureRegression
        COMMAND test_realistic_temperature
    )

    set_tests_properties(RealisticTemperatureRegression PROPERTIES
        TIMEOUT 120
        LABELS "regression;temperature"
    )
endif()
