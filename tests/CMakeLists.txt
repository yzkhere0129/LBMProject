# Test configuration

# Function to add a CUDA test
function(add_cuda_test test_name test_source)
    add_executable(${test_name} ${test_source})

    set_target_properties(${test_name} PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
    )

    target_link_libraries(${test_name}
        lbm_core
        lbm_physics
        gtest
        gtest_main
        CUDA::cudart
    )

    gtest_discover_tests(${test_name})
endfunction()

# Unit tests - Core
add_cuda_test(test_d3q19 unit/lattice/test_d3q19.cu)
add_cuda_test(test_bgk unit/collision/test_bgk.cu)
add_cuda_test(test_streaming unit/streaming/test_streaming.cu)
add_cuda_test(test_boundary unit/boundary/test_boundary_conditions.cu)

# Unit tests - Physics/Thermal
add_cuda_test(test_lattice_d3q7 unit/thermal/test_lattice_d3q7.cu)
add_cuda_test(test_thermal_lbm unit/thermal/test_thermal_lbm.cu)
add_cuda_test(test_dirichlet_bc unit/thermal/test_dirichlet_bc.cu)

# ============================================================================
# STABILITY REGRESSION TESTS
# Critical tests for high-Pe stability fixes (thermal solver)
# These tests MUST pass before any merge to main branch
# ============================================================================

# Unit Test 1: TVD Flux Limiter (prevents negative populations)
add_cuda_test(test_flux_limiter unit/stability/test_flux_limiter.cu)

# Unit Test 2: Temperature Bounds (prevents T runaway and NaN)
add_cuda_test(test_temperature_bounds unit/stability/test_temperature_bounds.cu)

# Regression Test: Omega Reduction (ensures omega_T <= 1.50)
add_cuda_test(test_omega_reduction regression/stability/test_omega_reduction.cu)

# Performance Test: Flux Limiter Overhead (ensures < 20% overhead)
add_cuda_test(test_flux_limiter_overhead performance/test_flux_limiter_overhead.cu)

# Unit tests - Physics/Laser
add_cuda_test(test_laser_source unit/laser/test_laser_source.cu)

# Unit tests - Physics/Materials
add_cuda_test(test_materials unit/materials/test_materials.cu)

# Unit tests - Physics/Phase Change
add_cuda_test(test_liquid_fraction unit/phase_change/test_liquid_fraction.cu)
add_cuda_test(test_enthalpy unit/phase_change/test_enthalpy.cu)
add_cuda_test(test_phase_properties unit/phase_change/test_phase_properties.cu)

# Unit tests - Physics/Fluid
add_cuda_test(test_fluid_lbm unit/fluid/test_fluid_lbm.cu)
add_cuda_test(test_fluid_debug unit/fluid/test_fluid_debug.cu)
add_cuda_test(test_fluid_boundaries unit/fluid/test_fluid_boundaries.cu)
add_cuda_test(test_guo_force unit/fluid/test_guo_force.cu)
add_cuda_test(test_moving_wall_bc unit/fluid/test_moving_wall_bc.cu)

# Unit tests - Liquid Fraction Copy (verifies device-to-host transfer)
add_cuda_test(test_liquid_fraction_copy unit/test_liquid_fraction_copy.cu)

# Unit tests - VTK Writer with Liquid Fraction (needs lbm_io library)
add_executable(test_vtk_liquid_fraction unit/test_vtk_liquid_fraction.cu)
set_target_properties(test_vtk_liquid_fraction PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
target_link_libraries(test_vtk_liquid_fraction
    lbm_core
    lbm_physics
    lbm_io
    gtest
    gtest_main
    CUDA::cudart
)
gtest_discover_tests(test_vtk_liquid_fraction)

# Unit tests - VTK Vector I/O (Phase 5: Vector field output)
# DISABLED: API mismatch with VTKWriter::writeStructuredGridWithVectors
# TODO: Update test to match new VTKWriter API
# add_executable(test_vtk_vector_io unit/test_vtk_vector_io.cu)
# set_target_properties(test_vtk_vector_io PROPERTIES
#     CUDA_SEPARABLE_COMPILATION ON
# )
# target_link_libraries(test_vtk_vector_io
#     lbm_io
#     gtest
#     gtest_main
#     CUDA::cudart
# )
# gtest_discover_tests(test_vtk_vector_io)

# Unit tests - Phase 6: VOF (Free Surface) - EXISTING TESTS
add_cuda_test(test_vof_advection unit/vof/test_vof_advection.cu)
add_cuda_test(test_vof_reconstruction unit/vof/test_vof_reconstruction.cu)
add_cuda_test(test_vof_curvature unit/vof/test_vof_curvature.cu)
add_cuda_test(test_vof_cell_conversion unit/vof/test_vof_cell_conversion.cu)
add_cuda_test(test_vof_surface_tension unit/vof/test_vof_surface_tension.cu)
add_cuda_test(test_vof_marangoni unit/vof/test_vof_marangoni.cu)
add_cuda_test(test_vof_contact_angle unit/vof/test_vof_contact_angle.cu)
add_cuda_test(test_vof_mass_conservation unit/vof/test_vof_mass_conservation.cu)
add_cuda_test(test_vof_interface_compression unit/vof/test_vof_interface_compression.cu)

# ============================================================================
# COMPREHENSIVE VOF TEST SUITE (80% → 95% coverage target)
# ============================================================================

# 1. Interface Advection Tests (Unit)
add_cuda_test(test_vof_advection_uniform unit/vof/test_vof_advection_uniform.cu)
add_cuda_test(test_vof_advection_shear unit/vof/test_vof_advection_shear.cu)

# 2. Evaporation Coupling Tests (Unit)
add_cuda_test(test_vof_evaporation_mass_loss unit/vof/test_vof_evaporation_mass_loss.cu)

# 3. Validation Tests (Analytical Solutions)
# These go in validation/ subdirectory for physics validation against theory
add_cuda_test(test_vof_advection_rotation validation/vof/test_vof_advection_rotation.cu)
add_cuda_test(test_vof_curvature_sphere validation/vof/test_vof_curvature_sphere.cu)
add_cuda_test(test_vof_curvature_cylinder validation/vof/test_vof_curvature_cylinder.cu)
add_cuda_test(test_vof_oscillating_droplet validation/vof/test_vof_oscillating_droplet.cu)
add_cuda_test(test_buoyancy_unit validation/vof/test_buoyancy_unit.cu)

# Unit tests - Phase 6: Marangoni Validation (New small module tests)
add_cuda_test(test_interface_geometry unit/vof/test_interface_geometry.cu)
add_cuda_test(test_marangoni_force unit/vof/test_marangoni_force.cu)
add_cuda_test(test_no_slip_boundary unit/fluid/test_no_slip_boundary.cu)

# Unit tests - Recoil Pressure Module (Evaporation-driven keyhole formation)
add_cuda_test(test_recoil_pressure unit/vof/test_recoil_pressure.cu)

# Unit tests - Evaporation Temperature Check (Bug: evaporation continues after cooling)
add_cuda_test(test_evaporation_temperature_check unit/vof/test_evaporation_temperature_check.cu)

# Unit tests - Phase 6.5: Multiphysics Integration
add_cuda_test(test_multiphysics_step1 unit/multiphysics/test_multiphysics_step1.cu)
add_cuda_test(test_multiphysics_step2 unit/multiphysics/test_multiphysics_step2.cu)
add_cuda_test(test_multiphysics_step3 unit/multiphysics/test_multiphysics_step3.cu)
add_cuda_test(test_darcy_damping_solid unit/multiphysics/test_darcy_damping_solid.cu)

# Unit test - Force Initialization (HIGH PRIORITY: prevents garbage value accumulation)
add_cuda_test(test_force_initialization unit/test_force_initialization.cu)

# ============================================================================
# REGRESSION TESTS - Bug Fixes Validation
# ============================================================================

# Regression test - VOF advection bulk cell bug fix
# CRITICAL: Tests that bulk cells (f~0 and f~1) properly propagate advected values
# Bug: Interface compression kernel returned early without writing to output buffer
# Fix: Added fill_level[idx] = f; before early return (vof_solver.cu:419-421)
add_cuda_test(test_vof_advection_bulk_cells regression/test_vof_advection_bulk_cells.cu)

# Regression test - Timestep convergence omega clamping
# Tests that all timesteps in convergence study produce stable omega values
# Bug: dt=0.5μs caused omega=1.92 (above stability limit), breaking convergence
# Fix: Changed timesteps from {4,2,1,0.5}μs to {8,4,2,1}μs
add_cuda_test(test_timestep_omega_clamping regression/test_timestep_omega_clamping.cu)

# Debug tests - Minimal reproduction tests
add_executable(test_fluidlbm_nan_debug debug/test_fluidlbm_nan_debug.cu)
set_target_properties(test_fluidlbm_nan_debug PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
target_link_libraries(test_fluidlbm_nan_debug
    lbm_core
    lbm_physics
    CUDA::cudart
)

# Debug test - Force application detailed diagnostic
add_executable(test_force_application_detailed debug/test_force_application_detailed.cu)
set_target_properties(test_force_application_detailed PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
target_link_libraries(test_force_application_detailed
    lbm_core
    lbm_physics
    CUDA::cudart
)

# Debug test - Buoyancy force minimal test
add_executable(test_buoyancy_force_minimal debug/test_buoyancy_force_minimal.cu)
set_target_properties(test_buoyancy_force_minimal PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
target_link_libraries(test_buoyancy_force_minimal
    lbm_core
    lbm_physics
    CUDA::cudart
)

# Debug test - Test without boundary conditions
add_executable(test_no_boundaries debug/test_no_boundaries.cu)
set_target_properties(test_no_boundaries PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
target_link_libraries(test_no_boundaries
    lbm_core
    lbm_physics
    CUDA::cudart
)

# Debug test - Unit conversion test
add_executable(test_unit_conversion debug/test_unit_conversion.cu)
set_target_properties(test_unit_conversion PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
target_link_libraries(test_unit_conversion
    lbm_core
    lbm_physics
    CUDA::cudart
)

# Debug test - Force kernel direct test
add_executable(test_debug_force_kernel debug/test_debug_force_kernel.cu)
set_target_properties(test_debug_force_kernel PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
target_link_libraries(test_debug_force_kernel
    lbm_core
    lbm_physics
    CUDA::cudart
)

# Debug tests - Test Ladder for Zero Velocity Bug
# Level 1: Kernel tests
add_executable(test_collision_kernel_direct debug/test_collision_kernel_direct.cu)
set_target_properties(test_collision_kernel_direct PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
target_link_libraries(test_collision_kernel_direct
    lbm_core
    lbm_physics
    CUDA::cudart
)

add_executable(test_macroscopic_kernel debug/test_macroscopic_kernel.cu)
set_target_properties(test_macroscopic_kernel PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
target_link_libraries(test_macroscopic_kernel
    lbm_core
    lbm_physics
    CUDA::cudart
)

# Level 2: Function tests
add_executable(test_force_velocity_pipeline debug/test_force_velocity_pipeline.cu)
set_target_properties(test_force_velocity_pipeline PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
target_link_libraries(test_force_velocity_pipeline
    lbm_core
    lbm_physics
    CUDA::cudart
)

# Level 3: Integration tests
add_executable(test_buoyancy_force_check debug/test_buoyancy_force_check.cu)
set_target_properties(test_buoyancy_force_check PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
target_link_libraries(test_buoyancy_force_check
    lbm_core
    lbm_physics
    CUDA::cudart
)

add_executable(test_darcy_survival debug/test_darcy_survival.cu)
set_target_properties(test_darcy_survival PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
target_link_libraries(test_darcy_survival
    lbm_core
    lbm_physics
    CUDA::cudart
)

# Level 4: Instrumented Phase 5
add_executable(test_phase5_instrumented debug/test_phase5_instrumented.cu)
set_target_properties(test_phase5_instrumented PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
target_link_libraries(test_phase5_instrumented
    lbm_core
    lbm_physics
    lbm_io
    CUDA::cudart
)

# Debug test - Multiphysics Force Diagnostic
add_cuda_test(test_multiphysics_force_diagnostic debug/test_multiphysics_force_diagnostic.cu)

# Diagnostic tests - Emergency Zero-Velocity Bug Isolation Suite
# These tests isolate the exact failure point in FluidLBM -> velocity computation
add_executable(test0_unit_conversion diagnostic/test0_unit_conversion.cu)
set_target_properties(test0_unit_conversion PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
target_link_libraries(test0_unit_conversion
    lbm_core
    lbm_physics
    CUDA::cudart
)

add_executable(test1_fluid_velocity_only diagnostic/test1_fluid_velocity_only.cu)
set_target_properties(test1_fluid_velocity_only PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
target_link_libraries(test1_fluid_velocity_only
    lbm_core
    lbm_physics
    CUDA::cudart
)

add_executable(test2_buoyancy_force diagnostic/test2_buoyancy_force.cu)
set_target_properties(test2_buoyancy_force PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
target_link_libraries(test2_buoyancy_force
    lbm_core
    lbm_physics
    CUDA::cudart
)

add_executable(test3_darcy_damping diagnostic/test3_darcy_damping.cu)
set_target_properties(test3_darcy_damping PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
target_link_libraries(test3_darcy_damping
    lbm_core
    lbm_physics
    CUDA::cudart
)

add_executable(test4_thermal_advection_coupling diagnostic/test4_thermal_advection_coupling.cu)
set_target_properties(test4_thermal_advection_coupling PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
target_link_libraries(test4_thermal_advection_coupling
    lbm_core
    lbm_physics
    CUDA::cudart
)

add_executable(test5_config_flags diagnostic/test5_config_flags.cu)
set_target_properties(test5_config_flags PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
target_link_libraries(test5_config_flags
    lbm_core
    lbm_physics
    CUDA::cudart
)

# Diagnostic test - Temperature Anomaly (Tier2 bug isolation)
add_executable(test_temperature_anomaly diagnostic/test_temperature_anomaly.cu)
set_target_properties(test_temperature_anomaly PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
target_link_libraries(test_temperature_anomaly
    lbm_core
    lbm_physics
    CUDA::cudart
)

# Diagnostic test - Multiphysics Temperature Anomaly
add_executable(test_multiphysics_temp_anomaly diagnostic/test_multiphysics_temp_anomaly.cu)
set_target_properties(test_multiphysics_temp_anomaly PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
target_link_libraries(test_multiphysics_temp_anomaly
    lbm_core
    lbm_physics
    lbm_diagnostics
    CUDA::cudart
)

# Diagnostic test - Marangoni Force Flow Tracing
add_executable(test_marangoni_force_diagnostic diagnostic/test_marangoni_force_diagnostic.cu)
set_target_properties(test_marangoni_force_diagnostic PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
target_link_libraries(test_marangoni_force_diagnostic
    lbm_core
    lbm_physics
    gtest
    gtest_main
    CUDA::cudart
)

# Diagnostic test - Surface Tension Unit Conversion
add_executable(test_force_unit_conversion diagnostic/test_force_unit_conversion.cu)
set_target_properties(test_force_unit_conversion PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
target_link_libraries(test_force_unit_conversion
    lbm_core
    lbm_physics
    CUDA::cudart
)

# Diagnostic test - CollisionBGK Force Application (Oscillating Droplet Velocity Bug)
add_executable(test_collisionBGK_force diagnostic/test_collisionBGK_force.cu)
set_target_properties(test_collisionBGK_force PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
target_link_libraries(test_collisionBGK_force
    lbm_core
    lbm_physics
    gtest
    gtest_main
    CUDA::cudart
)

# Integration tests
add_cuda_test(test_poiseuille integration/test_poiseuille_flow.cu)
add_cuda_test(test_gravity_driven_flow integration/test_gravity_driven_flow.cu)
add_cuda_test(test_marangoni_force_validation integration/test_marangoni_force_validation.cu)
add_cuda_test(test_marangoni_force_validation_fixed integration/test_marangoni_force_validation_fixed.cu)

# STABILITY INTEGRATION TEST: High-Pe stability (500-step end-to-end test)
add_cuda_test(test_high_pe_stability integration/stability/test_high_pe_stability.cu)

# Integration tests - Phase 6: VOF
add_cuda_test(test_static_droplet integration/test_static_droplet.cu)
add_cuda_test(test_marangoni_flow integration/test_marangoni_flow.cu)
add_cuda_test(test_laser_surface_deformation integration/test_laser_surface_deformation.cu)

# Integration test - Phase 6: Marangoni System Validation
add_cuda_test(test_marangoni_system integration/test_marangoni_system.cu)

# Add subdirectory for integration tests with custom configurations
add_subdirectory(integration)

# Add subdirectory for validation tests (physics validation against literature)
add_subdirectory(validation)

# Create a target to run all tests
add_custom_target(run_tests
    COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure
    DEPENDS test_d3q19 test_bgk test_streaming test_boundary test_poiseuille test_lattice_d3q7
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)

# Create a target for unit tests only
add_custom_target(run_unit_tests
    COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure -R "test_(d3q19|bgk|streaming|boundary)"
    DEPENDS test_d3q19 test_bgk test_streaming test_boundary
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)

# Create a target for integration tests only
add_custom_target(run_integration_tests
    COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure -R "test_poiseuille"
    DEPENDS test_poiseuille
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)

# ============================================================================
# STABILITY TEST TARGETS
# ============================================================================

# Fast pre-commit stability checks (< 30 seconds)
add_custom_target(run_stability_quick
    COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure -R "test_(flux_limiter|temperature_bounds|omega_reduction)"
    DEPENDS test_flux_limiter test_temperature_bounds test_omega_reduction
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Running quick stability regression tests (pre-commit)"
)

# Medium pre-push stability checks (< 2 minutes)
add_custom_target(run_stability_medium
    COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure -R "test_(flux_limiter|temperature_bounds|omega_reduction|high_pe_stability)"
    DEPENDS test_flux_limiter test_temperature_bounds test_omega_reduction test_high_pe_stability
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Running medium stability tests (pre-push)"
)

# Full stability suite including performance (< 5 minutes)
add_custom_target(run_stability_full
    COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure -R "stability"
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Running full stability regression suite"
)