# Integration Tests for MultiphysicsSolver
# Comprehensive test suite for multiphysics coupling validation

enable_language(CUDA)
find_package(CUDAToolkit REQUIRED)

# Helper function to add multiphysics integration test
function(add_multiphysics_test test_name test_source)
    add_executable(${test_name} ${test_source})

    set_target_properties(${test_name} PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES "70;75;80;86"
    )

    target_include_directories(${test_name} PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/src
    )

    target_link_libraries(${test_name}
        lbm_physics
        lbm_core
        lbm_diagnostics
        gtest
        gtest_main
        CUDA::cudart
    )

    add_test(NAME ${test_name} COMMAND ${test_name})
endfunction()

# ============================================================================
# ENERGY CONSERVATION TESTS
# ============================================================================

add_multiphysics_test(test_energy_conservation_no_source
    test_energy_conservation_no_source.cu)
set_tests_properties(test_energy_conservation_no_source PROPERTIES
    TIMEOUT 60
    LABELS "integration;multiphysics;energy;conservation;critical")

add_multiphysics_test(test_energy_conservation_laser_only
    test_energy_conservation_laser_only.cu)
set_tests_properties(test_energy_conservation_laser_only PROPERTIES
    TIMEOUT 120
    LABELS "integration;multiphysics;energy;laser;critical")

add_multiphysics_test(test_energy_conservation_full
    test_energy_conservation_full.cu)
set_tests_properties(test_energy_conservation_full PROPERTIES
    TIMEOUT 180
    LABELS "integration;multiphysics;energy;full;critical")

# ============================================================================
# COUPLING CORRECTNESS TESTS
# ============================================================================

add_multiphysics_test(test_mp_thermal_fluid_coupling
    test_thermal_fluid_coupling.cu)
set_tests_properties(test_mp_thermal_fluid_coupling PROPERTIES
    TIMEOUT 120
    LABELS "integration;multiphysics;coupling;buoyancy;critical")

add_multiphysics_test(test_vof_fluid_coupling
    test_vof_fluid_coupling.cu)
set_tests_properties(test_vof_fluid_coupling PROPERTIES
    TIMEOUT 120
    LABELS "integration;multiphysics;coupling;vof;critical")

add_multiphysics_test(test_thermal_vof_coupling
    test_thermal_vof_coupling.cu)
set_tests_properties(test_thermal_vof_coupling PROPERTIES
    TIMEOUT 120
    LABELS "integration;multiphysics;coupling;evaporation;critical")

add_multiphysics_test(test_phase_fluid_coupling
    test_phase_fluid_coupling.cu)
set_tests_properties(test_phase_fluid_coupling PROPERTIES
    TIMEOUT 120
    LABELS "integration;multiphysics;coupling;darcy;critical")

# ============================================================================
# FORCE BALANCE TESTS
# ============================================================================

add_multiphysics_test(test_force_balance_static
    test_force_balance_static.cu)
set_tests_properties(test_force_balance_static PROPERTIES
    TIMEOUT 60
    LABELS "integration;multiphysics;forces;equilibrium")

add_multiphysics_test(test_force_magnitude_ordering
    test_force_magnitude_ordering.cu)
set_tests_properties(test_force_magnitude_ordering PROPERTIES
    TIMEOUT 60
    LABELS "integration;multiphysics;forces;validation")

add_multiphysics_test(test_force_direction
    test_force_direction.cu)
set_tests_properties(test_force_direction PROPERTIES
    TIMEOUT 60
    LABELS "integration;multiphysics;forces;direction")

# ============================================================================
# CFL STABILITY TESTS
# ============================================================================

add_multiphysics_test(test_cfl_limiting_effectiveness
    test_cfl_limiting_effectiveness.cu)
set_tests_properties(test_cfl_limiting_effectiveness PROPERTIES
    TIMEOUT 120
    LABELS "integration;multiphysics;cfl;stability;critical")

add_multiphysics_test(test_cfl_limiting_conservation
    test_cfl_limiting_conservation.cu)
set_tests_properties(test_cfl_limiting_conservation PROPERTIES
    TIMEOUT 120
    LABELS "integration;multiphysics;cfl;conservation")

add_multiphysics_test(test_extreme_gradients
    test_extreme_gradients.cu)
set_tests_properties(test_extreme_gradients PROPERTIES
    TIMEOUT 180
    LABELS "integration;multiphysics;stability;extreme")

# ============================================================================
# SUBCYCLING TESTS
# ============================================================================

add_multiphysics_test(test_vof_subcycling_convergence
    test_vof_subcycling_convergence.cu)
set_tests_properties(test_vof_subcycling_convergence PROPERTIES
    TIMEOUT 300
    LABELS "integration;multiphysics;vof;subcycling")

add_multiphysics_test(test_subcycling_1_vs_10
    test_subcycling_1_vs_10.cu)
set_tests_properties(test_subcycling_1_vs_10 PROPERTIES
    TIMEOUT 180
    LABELS "integration;multiphysics;vof;subcycling")

# ============================================================================
# UNIT CONVERSION TESTS
# ============================================================================

add_multiphysics_test(test_unit_conversion_roundtrip
    test_unit_conversion_roundtrip.cu)
set_tests_properties(test_unit_conversion_roundtrip PROPERTIES
    TIMEOUT 30
    LABELS "integration;multiphysics;units;critical")

add_multiphysics_test(test_unit_conversion_consistency
    test_unit_conversion_consistency.cu)
set_tests_properties(test_unit_conversion_consistency PROPERTIES
    TIMEOUT 60
    LABELS "integration;multiphysics;units;critical")

# ============================================================================
# STEADY STATE TESTS
# ============================================================================

add_multiphysics_test(test_steady_state_temperature
    test_steady_state_temperature.cu)
set_tests_properties(test_steady_state_temperature PROPERTIES
    TIMEOUT 300
    LABELS "integration;multiphysics;steady_state;thermal")

add_multiphysics_test(test_steady_state_flow
    test_steady_state_flow.cu)
set_tests_properties(test_steady_state_flow PROPERTIES
    TIMEOUT 300
    LABELS "integration;multiphysics;steady_state;fluid")

add_multiphysics_test(test_melt_pool_dimensions
    test_melt_pool_dimensions.cu)
set_tests_properties(test_melt_pool_dimensions PROPERTIES
    TIMEOUT 300
    LABELS "integration;multiphysics;melt_pool;validation")

# ============================================================================
# ROBUSTNESS TESTS
# ============================================================================

add_multiphysics_test(test_high_power_laser
    test_high_power_laser.cu)
set_tests_properties(test_high_power_laser PROPERTIES
    TIMEOUT 180
    LABELS "integration;multiphysics;robustness;laser")

add_multiphysics_test(test_rapid_solidification
    test_rapid_solidification.cu)
set_tests_properties(test_rapid_solidification PROPERTIES
    TIMEOUT 180
    LABELS "integration;multiphysics;robustness;phase_change")

add_multiphysics_test(test_nan_detection
    test_nan_detection.cu)
set_tests_properties(test_nan_detection PROPERTIES
    TIMEOUT 60
    LABELS "integration;multiphysics;robustness;nan;critical")

# ============================================================================
# MODULE ENABLE/DISABLE TESTS
# ============================================================================

add_multiphysics_test(test_disable_marangoni
    test_disable_marangoni.cu)
set_tests_properties(test_disable_marangoni PROPERTIES
    TIMEOUT 60
    LABELS "integration;multiphysics;config;modules")

add_multiphysics_test(test_disable_vof
    test_disable_vof.cu)
set_tests_properties(test_disable_vof PROPERTIES
    TIMEOUT 60
    LABELS "integration;multiphysics;config;modules")

add_multiphysics_test(test_minimal_config
    test_minimal_config.cu)
set_tests_properties(test_minimal_config PROPERTIES
    TIMEOUT 60
    LABELS "integration;multiphysics;config;minimal")

# ============================================================================
# REGRESSION TESTS
# ============================================================================

add_multiphysics_test(test_known_good_output
    test_known_good_output.cu)
set_tests_properties(test_known_good_output PROPERTIES
    TIMEOUT 120
    LABELS "integration;multiphysics;regression;golden")

add_multiphysics_test(test_deterministic
    test_deterministic.cu)
set_tests_properties(test_deterministic PROPERTIES
    TIMEOUT 180
    LABELS "integration;multiphysics;regression;deterministic;critical")

# ============================================================================
# TEST SUITES
# ============================================================================

# Critical tests (must pass before commit)
add_custom_target(multiphysics_critical_tests
    COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure -L "multiphysics.*critical"
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Running critical multiphysics tests"
)

# Full test suite
add_custom_target(multiphysics_all_tests
    COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure -L "multiphysics"
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Running all multiphysics tests"
)

# Energy conservation suite
add_custom_target(multiphysics_energy_tests
    COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure -L "multiphysics.*energy"
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Running multiphysics energy conservation tests"
)

# Coupling suite
add_custom_target(multiphysics_coupling_tests
    COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure -L "multiphysics.*coupling"
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Running multiphysics coupling tests"
)
