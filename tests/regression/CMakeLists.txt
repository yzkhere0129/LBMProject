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
    CUDA_ARCHITECTURES "70;75;80;86"
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
        CUDA_ARCHITECTURES "70;75;80;86"
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
